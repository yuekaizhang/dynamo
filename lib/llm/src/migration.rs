// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::sync::Arc;

use anyhow::{Error, Result};
use futures::{stream, stream::StreamExt};

use async_nats::client::{
    RequestError as NatsRequestError, RequestErrorKind::NoResponders as NatsNoResponders,
};

use crate::{
    model_card::ModelDeploymentCard,
    protocols::common::llm_backend::{LLMEngineOutput, PreprocessedRequest},
};

use dynamo_runtime::{
    pipeline::{
        async_trait, AsyncEngineContextProvider, ManyOut, Operator, ResponseStream,
        ServerStreamingEngine, SingleIn,
    },
    protocols::{annotated::Annotated, maybe_error::MaybeError},
};

pub struct Migration {
    migration_limit: u32,
}

impl Migration {
    pub async fn from_mdc(mdc: ModelDeploymentCard) -> Result<Arc<Self>> {
        Ok(Arc::new(Self {
            migration_limit: mdc.migration_limit,
        }))
    }
}

#[async_trait]
impl
    Operator<
        SingleIn<PreprocessedRequest>,
        ManyOut<Annotated<LLMEngineOutput>>,
        SingleIn<PreprocessedRequest>,
        ManyOut<Annotated<LLMEngineOutput>>,
    > for Migration
{
    async fn generate(
        &self,
        request: SingleIn<PreprocessedRequest>,
        next: ServerStreamingEngine<PreprocessedRequest, Annotated<LLMEngineOutput>>,
    ) -> Result<ManyOut<Annotated<LLMEngineOutput>>> {
        let (preprocessed_request, context) = request.transfer(());
        let engine_ctx = context.context();
        let retry_manager =
            RetryManager::build(preprocessed_request, next, self.migration_limit).await?;
        let response_stream = stream::unfold(retry_manager, |mut retry_manager| async move {
            retry_manager
                .next()
                .await
                .map(|response| (response, retry_manager))
        });
        Ok(ResponseStream::new(Box::pin(response_stream), engine_ctx))
    }
}

struct RetryManager {
    request: PreprocessedRequest,
    next_generate: ServerStreamingEngine<PreprocessedRequest, Annotated<LLMEngineOutput>>,
    next_stream: Option<ManyOut<Annotated<LLMEngineOutput>>>,
    retries_left: u32,
}

impl RetryManager {
    pub async fn build(
        preprocessed_request: PreprocessedRequest,
        next: ServerStreamingEngine<PreprocessedRequest, Annotated<LLMEngineOutput>>,
        retries_left: u32,
    ) -> Result<Self> {
        let mut slf = Self {
            request: preprocessed_request,
            next_generate: next,
            next_stream: None,
            retries_left: retries_left + 1, // +1 to account for the initial attempt
        };
        slf.new_stream().await?;
        Ok(slf)
    }

    pub async fn next(&mut self) -> Option<Annotated<LLMEngineOutput>> {
        loop {
            let response_stream = match self.next_stream.as_mut() {
                Some(stream) => stream,
                None => {
                    tracing::error!("next() called with next_stream is None - should not happen");
                    return Some(Annotated::from_err(
                        Error::msg("next_stream is None").into(),
                    ));
                }
            };
            if let Some(response) = response_stream.next().await {
                if let Some(err) = response.err() {
                    const STREAM_ERR_MSG: &str = "Stream ended before generation completed";
                    if format!("{:?}", err) == STREAM_ERR_MSG {
                        tracing::warn!("Stream disconnected... recreating stream...");
                        if let Err(err) = self.new_stream().await {
                            tracing::warn!("Cannot recreate stream: {:?}", err);
                        } else {
                            continue;
                        }
                    }
                }
                self.track_response(&response);
                return Some(response);
            }
            return None;
        }
    }

    async fn new_stream(&mut self) -> Result<()> {
        let mut response_stream: Option<Result<ManyOut<Annotated<LLMEngineOutput>>>> = None;
        while self.retries_left > 0 {
            self.retries_left -= 1;
            // TODO: Is there anything needed to pass between context?
            let request = SingleIn::new(self.request.clone());
            response_stream = Some(self.next_generate.generate(request).await);
            if let Some(err) = response_stream.as_ref().unwrap().as_ref().err() {
                if let Some(req_err) = err.downcast_ref::<NatsRequestError>() {
                    if matches!(req_err.kind(), NatsNoResponders) {
                        tracing::warn!("Creating new stream... retrying...");
                        continue;
                    }
                }
            }
            break;
        }
        match response_stream {
            Some(Ok(next_stream)) => {
                self.next_stream = Some(next_stream);
                Ok(())
            }
            Some(Err(err)) => Err(err), // should propagate original error if any
            None => Err(Error::msg(
                "Migration limit exhausted", // should propagate original error if any
            )),
        }
    }

    fn track_response(&mut self, response: &Annotated<LLMEngineOutput>) {
        if self.retries_left == 0 {
            return;
        }
        let llm_engine_output = match response.data.as_ref() {
            Some(output) => output,
            None => return,
        };
        if let Some(max_tokens) = self.request.stop_conditions.max_tokens {
            self.request.stop_conditions.max_tokens =
                Some(max_tokens.saturating_sub(llm_engine_output.token_ids.len() as u32));
        }
        for token_id in llm_engine_output.token_ids.iter() {
            self.request.token_ids.push(*token_id);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::protocols::common::{OutputOptions, SamplingOptions, StopConditions};
    use dynamo_runtime::pipeline::context::Controller;
    use dynamo_runtime::pipeline::AsyncEngine;
    use std::sync::atomic::{AtomicU32, Ordering};
    use tokio::sync::mpsc;

    // Helper to create a mock preprocessed request
    fn create_mock_request(max_tokens: u32) -> PreprocessedRequest {
        PreprocessedRequest {
            model: "mock".to_string(),
            token_ids: vec![1, 2, 3],
            batch_token_ids: None,
            stop_conditions: StopConditions {
                max_tokens: Some(max_tokens),
                ..Default::default()
            },
            sampling_options: SamplingOptions::default(),
            output_options: OutputOptions::default(),
            eos_token_ids: vec![],
            mdc_sum: None,
            annotations: vec![],
            estimated_prefix_hit_num_blocks: None,
        }
    }

    // Helper to create mock LLM engine output
    fn create_mock_output(token_id: u32) -> Annotated<LLMEngineOutput> {
        Annotated::from_data(LLMEngineOutput {
            token_ids: vec![token_id],
            tokens: None,
            text: Some(format!("token_{}", token_id)),
            cum_log_probs: None,
            log_probs: None,
            top_logprobs: None,
            finish_reason: None,
            index: None,
        })
    }

    #[derive(Debug, Clone)]
    enum MockBehavior {
        /// Always succeeds with all responses
        Success,
        /// Fails on first call with NoResponders error, then succeeds on subsequent calls
        FailThenSuccess,
        /// Succeeds initially, fails mid-stream with specific error, then succeeds on retry
        MidStreamFail { fail_after: usize },
        /// Succeeds initially, fails mid-stream with specific error, then always fails on retry attempts
        MidStreamFailAlways { fail_after: usize },
        /// Succeeds initially, fails mid-stream, then always fails with stream error on retry attempts
        MidStreamFailAlwaysStreamError { fail_after: usize },
        /// Always fails with NoResponders error (same as FailThenSuccess first call)
        AlwaysFail,
    }

    // Unified mock server streaming engine that can simulate different scenarios
    struct MockEngine {
        behavior: MockBehavior,
        num_responses: usize,
        token_offset: u32,
        call_count: Arc<AtomicU32>,
    }

    impl MockEngine {
        fn new(behavior: MockBehavior, num_responses: usize, token_offset: u32) -> Self {
            Self {
                behavior,
                num_responses,
                token_offset,
                call_count: Arc::new(AtomicU32::new(0)),
            }
        }
    }

    #[async_trait]
    impl
        AsyncEngine<
            SingleIn<PreprocessedRequest>,
            ManyOut<Annotated<LLMEngineOutput>>,
            anyhow::Error,
        > for MockEngine
    {
        async fn generate(
            &self,
            request: SingleIn<PreprocessedRequest>,
        ) -> Result<ManyOut<Annotated<LLMEngineOutput>>> {
            let call_num = self.call_count.fetch_add(1, Ordering::SeqCst);
            let (preprocessed_request, _) = request.transfer(());

            // Calculate how many responses we've already generated based on request token_ids
            // Initial request has [1, 2, 3], so anything beyond that are generated responses
            let initial_tokens = 3; // [1, 2, 3]
            let responses_already_generated = preprocessed_request
                .token_ids
                .len()
                .saturating_sub(initial_tokens);

            // Assert that max_tokens reflects the expected remaining tokens
            let expected_max_tokens =
                self.num_responses
                    .saturating_sub(responses_already_generated) as u32;
            assert_eq!(
                preprocessed_request.stop_conditions.max_tokens,
                Some(expected_max_tokens),
                "max_tokens should be {} but got {:?}",
                expected_max_tokens,
                preprocessed_request.stop_conditions.max_tokens
            );

            match &self.behavior {
                MockBehavior::Success => {
                    // Always succeed with remaining responses
                    self.send_responses(responses_already_generated, self.num_responses)
                        .await
                }
                MockBehavior::FailThenSuccess => {
                    if call_num == 0 {
                        // First call - return "No responders available" error to trigger retry
                        let nats_error: NatsRequestError = NatsNoResponders.into();
                        return Err(nats_error.into());
                    } else {
                        // Subsequent calls - succeed with remaining responses
                        self.send_responses(responses_already_generated, self.num_responses)
                            .await
                    }
                }
                MockBehavior::MidStreamFail { fail_after } => {
                    let (tx, rx) = mpsc::channel(1);
                    let token_offset = self.token_offset;
                    let fail_after = *fail_after;
                    let num_responses = self.num_responses;

                    if call_num == 0 {
                        // First call - send some responses then an error to simulate disconnection
                        tokio::spawn(async move {
                            // Send responses from current position to fail_after
                            for i in responses_already_generated..fail_after.min(num_responses) {
                                let response = create_mock_output(token_offset + 1 + i as u32);
                                if tx.send(response).await.is_err() {
                                    break;
                                }
                            }
                            // Send the specific error that triggers retry logic
                            let error_response = Annotated::from_err(
                                anyhow::Error::msg("Stream ended before generation completed")
                                    .into(),
                            );
                            let _ = tx.send(error_response).await;
                        });
                    } else {
                        // Second call - send remaining responses from where we left off
                        tokio::spawn(async move {
                            for i in responses_already_generated..num_responses {
                                let response = create_mock_output(token_offset + 1 + i as u32);
                                if tx.send(response).await.is_err() {
                                    break;
                                }
                            }
                        });
                    }

                    let stream = tokio_stream::wrappers::ReceiverStream::new(rx);
                    let ctx = Arc::new(Controller::default());
                    Ok(dynamo_runtime::pipeline::ResponseStream::new(
                        Box::pin(stream),
                        ctx,
                    ))
                }
                MockBehavior::MidStreamFailAlways { fail_after } => {
                    if call_num == 0 {
                        // First call - send some responses then an error to simulate disconnection
                        let (tx, rx) = mpsc::channel(1);
                        let token_offset = self.token_offset;
                        let fail_after = *fail_after;
                        let num_responses = self.num_responses;

                        tokio::spawn(async move {
                            // Send responses from current position to fail_after
                            for i in responses_already_generated..fail_after.min(num_responses) {
                                let response = create_mock_output(token_offset + 1 + i as u32);
                                if tx.send(response).await.is_err() {
                                    break;
                                }
                            }
                            // Send the specific error that triggers retry logic
                            let error_response = Annotated::from_err(
                                anyhow::Error::msg("Stream ended before generation completed")
                                    .into(),
                            );
                            let _ = tx.send(error_response).await;
                        });

                        let stream = tokio_stream::wrappers::ReceiverStream::new(rx);
                        let ctx = Arc::new(Controller::default());
                        Ok(dynamo_runtime::pipeline::ResponseStream::new(
                            Box::pin(stream),
                            ctx,
                        ))
                    } else {
                        // Subsequent calls - always fail with NoResponders error (same as AlwaysFail)
                        let nats_error: NatsRequestError = NatsNoResponders.into();
                        Err(nats_error.into())
                    }
                }
                MockBehavior::MidStreamFailAlwaysStreamError { fail_after } => {
                    let (tx, rx) = mpsc::channel(1);
                    let token_offset = self.token_offset;
                    let fail_after = *fail_after;
                    let num_responses = self.num_responses;

                    if call_num == 0 {
                        // First call - send some responses then an error to simulate disconnection
                        tokio::spawn(async move {
                            // Send responses from current position to fail_after
                            for i in responses_already_generated..fail_after.min(num_responses) {
                                let response = create_mock_output(token_offset + 1 + i as u32);
                                if tx.send(response).await.is_err() {
                                    break;
                                }
                            }
                            // Send the specific error that triggers retry logic
                            let error_response = Annotated::from_err(
                                anyhow::Error::msg("Stream ended before generation completed")
                                    .into(),
                            );
                            let _ = tx.send(error_response).await;
                        });

                        let stream = tokio_stream::wrappers::ReceiverStream::new(rx);
                        let ctx = Arc::new(Controller::default());
                        Ok(dynamo_runtime::pipeline::ResponseStream::new(
                            Box::pin(stream),
                            ctx,
                        ))
                    } else {
                        // Subsequent calls - immediately send stream error (no successful responses)
                        tokio::spawn(async move {
                            // Send the stream error immediately
                            let error_response = Annotated::from_err(
                                anyhow::Error::msg("Stream ended before generation completed")
                                    .into(),
                            );
                            let _ = tx.send(error_response).await;
                        });

                        let stream = tokio_stream::wrappers::ReceiverStream::new(rx);
                        let ctx = Arc::new(Controller::default());
                        Ok(dynamo_runtime::pipeline::ResponseStream::new(
                            Box::pin(stream),
                            ctx,
                        ))
                    }
                }
                MockBehavior::AlwaysFail => {
                    // Always fail with NoResponders error (same as FailThenSuccess first call)
                    let nats_error: NatsRequestError = NatsNoResponders.into();
                    Err(nats_error.into())
                }
            }
        }
    }

    impl MockEngine {
        async fn send_responses(
            &self,
            start: usize,
            end: usize,
        ) -> Result<ManyOut<Annotated<LLMEngineOutput>>> {
            let (tx, rx) = mpsc::channel(1);
            let token_offset = self.token_offset;

            tokio::spawn(async move {
                for i in start..end {
                    let response = create_mock_output(token_offset + 1 + i as u32);
                    if tx.send(response).await.is_err() {
                        break;
                    }
                }
            });

            let stream = tokio_stream::wrappers::ReceiverStream::new(rx);
            let ctx = Arc::new(Controller::default());
            Ok(dynamo_runtime::pipeline::ResponseStream::new(
                Box::pin(stream),
                ctx,
            ))
        }
    }

    /// Test case 1: No migration needed
    /// Tests the normal case where the RetryManager successfully processes all responses
    /// from a single stream without any failures or need for retries/migration.
    /// Expected behavior: All 10 responses should be received successfully.
    #[tokio::test]
    async fn test_retry_manager_no_migration() {
        let request = create_mock_request(10);
        let mock_engine = Arc::new(MockEngine::new(MockBehavior::Success, 10, 100));
        let next_generate: ServerStreamingEngine<PreprocessedRequest, Annotated<LLMEngineOutput>> =
            mock_engine;

        let mut retry_manager = RetryManager::build(request, next_generate, 0)
            .await
            .expect("Failed to build RetryManager");

        let mut responses = Vec::new();
        while let Some(response) = retry_manager.next().await {
            responses.push(response);
        }

        assert_eq!(responses.len(), 10);
        for (i, response) in responses.iter().enumerate() {
            assert!(response.err().is_none());
            if let Some(output) = &response.data {
                assert_eq!(output.token_ids, vec![101 + i as u32]); // 101, 102, 103, ..., 110
            }
        }
    }

    /// Test case 2: New request migration
    /// Tests the scenario where a worker becomes unreachable for new requests initially,
    /// triggering the RetryManager to retry the request. The MockEngine with FailThenSuccess
    /// fails on the first call with a "No responders available" error, then succeeds
    /// on subsequent calls, simulating a worker becoming available after initial failure.
    /// Expected behavior: All 10 responses should be received successfully after retry.
    #[tokio::test]
    async fn test_retry_manager_new_request_migration() {
        let request = create_mock_request(10);
        let mock_engine = Arc::new(MockEngine::new(MockBehavior::FailThenSuccess, 10, 100));
        let next_generate: ServerStreamingEngine<PreprocessedRequest, Annotated<LLMEngineOutput>> =
            mock_engine;

        let mut retry_manager = RetryManager::build(request, next_generate, 3)
            .await
            .expect("Failed to build RetryManager");

        let mut responses = Vec::new();
        while let Some(response) = retry_manager.next().await {
            responses.push(response);
        }

        assert_eq!(responses.len(), 10);
        for (i, response) in responses.iter().enumerate() {
            assert!(response.err().is_none());
            if let Some(output) = &response.data {
                assert_eq!(output.token_ids, vec![101 + i as u32]); // 101, 102, 103, ..., 110
            }
        }
    }

    /// Test case 3: Ongoing request migration
    /// Tests the scenario where a worker fails mid-stream during an ongoing request.
    /// This simulates a connection being lost after partial response delivery, requiring
    /// the RetryManager to detect the failure (via "Stream ended before generation completed" error),
    /// create a new stream, and continue from where it left off.
    /// Expected behavior: 5 responses from first stream + 5 responses from retry stream = 10 total.
    #[tokio::test]
    async fn test_retry_manager_ongoing_request_migration() {
        let request = create_mock_request(10);
        let mock_engine = Arc::new(MockEngine::new(
            MockBehavior::MidStreamFail { fail_after: 5 },
            10,
            100,
        ));
        let next_generate: ServerStreamingEngine<PreprocessedRequest, Annotated<LLMEngineOutput>> =
            mock_engine;

        let mut retry_manager = RetryManager::build(request, next_generate, 3)
            .await
            .expect("Failed to build RetryManager");

        let mut responses = Vec::new();
        while let Some(response) = retry_manager.next().await {
            responses.push(response);
        }

        // Should have received all 10 responses (5 from first stream + 5 from second stream)
        assert_eq!(responses.len(), 10);

        // Check that we received responses from both streams
        for (i, response) in responses.iter().enumerate() {
            assert!(response.err().is_none());
            if let Some(output) = &response.data {
                assert_eq!(output.token_ids, vec![101 + i as u32]); // 101, 102, 103, ..., 110
            }
        }
    }

    /// Test case 4: New request migration - indefinite failure
    /// Tests the scenario where a worker becomes unreachable for new requests indefinitely.
    /// The RetryManager should exhaust all retries and return the original error from the first attempt.
    /// Expected behavior: Should receive an error after all retries are exhausted, with the original error.
    #[tokio::test]
    async fn test_retry_manager_new_request_migration_indefinite_failure() {
        let request = create_mock_request(0);
        let mock_engine = Arc::new(MockEngine::new(MockBehavior::AlwaysFail, 0, 100));
        let next_generate: ServerStreamingEngine<PreprocessedRequest, Annotated<LLMEngineOutput>> =
            mock_engine;

        // Should fail to build due to initial stream creation failure after exhausting all 3 retries
        let retry_manager_result = RetryManager::build(request, next_generate, 3).await;

        assert!(retry_manager_result.is_err());
        if let Err(error) = retry_manager_result {
            assert!(error.to_string().contains("no responders"));
        }
    }

    /// Test case 5: Ongoing request migration - indefinite failure
    /// Tests the scenario where a worker fails mid-stream indefinitely during ongoing requests.
    /// The RetryManager should exhaust all retries and return the original stream disconnection error.
    /// Expected behavior: Should receive some responses from first stream, then error after retries exhausted.
    #[tokio::test]
    async fn test_retry_manager_ongoing_request_migration_indefinite_failure() {
        let request = create_mock_request(10);
        let mock_engine = Arc::new(MockEngine::new(
            MockBehavior::MidStreamFailAlways { fail_after: 3 },
            10,
            100,
        ));
        let next_generate: ServerStreamingEngine<PreprocessedRequest, Annotated<LLMEngineOutput>> =
            mock_engine;

        let mut retry_manager = RetryManager::build(request, next_generate, 3) // 3 retries
            .await
            .expect("Failed to build RetryManager");

        let mut responses = Vec::new();

        // Collect all responses (both successful and error responses)
        while let Some(response) = retry_manager.next().await {
            responses.push(response);
        }

        // Should have received 4 total responses: 3 successful + 1 error
        assert_eq!(responses.len(), 4);

        // First 3 responses should be successful with tokens 101, 102, 103
        for (i, response) in responses[0..3].iter().enumerate() {
            assert!(response.err().is_none());
            if let Some(output) = &response.data {
                assert_eq!(output.token_ids, vec![101 + i as u32]); // 101, 102, 103
            }
        }

        // 4th response should be an error after retries are exhausted
        let error_response = &responses[3];
        assert!(error_response.err().is_some());
        if let Some(error) = error_response.err() {
            assert!(error
                .to_string()
                .contains("Stream ended before generation completed"));
        }
    }

    /// Test case 6: Ongoing request migration - indefinite failure with stream errors
    /// Tests the scenario where a worker fails mid-stream indefinitely during ongoing requests,
    /// and all retry attempts also fail with stream errors instead of NATS errors.
    /// Expected behavior: Should receive some responses from first stream, then error after retries exhausted.
    #[tokio::test]
    async fn test_retry_manager_ongoing_request_migration_indefinite_failure_stream_error() {
        let request = create_mock_request(10);
        let mock_engine = Arc::new(MockEngine::new(
            MockBehavior::MidStreamFailAlwaysStreamError { fail_after: 3 },
            10,
            100,
        ));
        let next_generate: ServerStreamingEngine<PreprocessedRequest, Annotated<LLMEngineOutput>> =
            mock_engine;

        let mut retry_manager = RetryManager::build(request, next_generate, 3) // 3 retries
            .await
            .expect("Failed to build RetryManager");

        let mut responses = Vec::new();

        // Collect all responses (both successful and error responses)
        while let Some(response) = retry_manager.next().await {
            responses.push(response);
        }

        // Should have received 4 total responses: 3 successful + 1 error
        assert_eq!(responses.len(), 4);

        // First 3 responses should be successful with tokens 101, 102, 103
        for (i, response) in responses[0..3].iter().enumerate() {
            assert!(response.err().is_none());
            if let Some(output) = &response.data {
                assert_eq!(output.token_ids, vec![101 + i as u32]); // 101, 102, 103
            }
        }

        // 4th response should be an error after retries are exhausted
        let error_response = &responses[3];
        assert!(error_response.err().is_some());
        if let Some(error) = error_response.err() {
            assert!(error
                .to_string()
                .contains("Stream ended before generation completed"));
        }
    }
}
