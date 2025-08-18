// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

//! MockSchedulerEngine - AsyncEngine wrapper around the Scheduler
//!
//! This module provides an AsyncEngine implementation that wraps the Scheduler
//! to provide streaming token generation with realistic timing simulation.

use crate::kv_router::publisher::WorkerMetricsPublisher;
use crate::mocker::protocols::DirectRequest;
use crate::mocker::protocols::{MockEngineArgs, OutputSignal};
use crate::mocker::scheduler::Scheduler;
use crate::protocols::common::llm_backend::{LLMEngineOutput, PreprocessedRequest};
use crate::protocols::TokenIdType;
use dynamo_runtime::protocols::annotated::Annotated;
use dynamo_runtime::DistributedRuntime;
use tokio_util::sync::CancellationToken;

use dynamo_runtime::{
    component::Component,
    engine::AsyncEngineContextProvider,
    pipeline::{async_trait, AsyncEngine, Error, ManyOut, ResponseStream, SingleIn},
    traits::DistributedRuntimeProvider,
    Result,
};

use crate::kv_router::protocols::{KvCacheEvent, KvCacheEventData};
use crate::kv_router::publisher::KvEventPublisher;
use futures::StreamExt;
use rand::Rng;
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::{mpsc, Mutex, OnceCell};
use tokio::time::{interval, Duration};
use tokio_stream::wrappers::ReceiverStream;
use uuid::Uuid;

pub const MOCKER_COMPONENT: &str = "mocker";

/// Generate a random token ID from 1k to 5k
fn generate_random_token() -> TokenIdType {
    let mut rng = rand::rng();
    rng.random_range(1000..5000)
}

/// AsyncEngine wrapper around the Scheduler that generates random character tokens
#[derive(Clone)]
pub struct MockVllmEngine {
    active_requests: Arc<Mutex<HashMap<Uuid, mpsc::UnboundedSender<OutputSignal>>>>,
    request_senders: Arc<OnceCell<Vec<mpsc::UnboundedSender<DirectRequest>>>>,
    engine_args: MockEngineArgs,
}

impl MockVllmEngine {
    /// Create a new MockVllmEngine with the given parameters
    pub fn new(args: MockEngineArgs) -> Self {
        Self {
            active_requests: Arc::new(Mutex::new(HashMap::new())),
            request_senders: Arc::new(OnceCell::new()),
            engine_args: args,
        }
    }

    pub async fn start(&self, component: Component) -> Result<()> {
        let cancel_token = component.drt().runtime().child_token();

        let (schedulers, kv_event_receiver) = self.start_schedulers(
            self.engine_args.clone(),
            self.active_requests.clone(),
            cancel_token.clone(),
        );

        Self::start_metrics_publishing(&schedulers, Some(component.clone()), cancel_token.clone())
            .await?;

        // Start KV events publishing with the actual receivers from schedulers
        if self.engine_args.enable_prefix_caching {
            Self::start_kv_events_publishing(
                kv_event_receiver,
                Some(component.clone()),
                self.engine_args.block_size,
                cancel_token.clone(),
            )
            .await?;
        }

        Ok(())
    }

    pub fn direct(&self, request: DirectRequest, dp_rank: usize) {
        let senders = self.request_senders.get().expect("Not initialized");
        let _ = senders[dp_rank].send(request);
    }

    /// Create schedulers and spawn their background tasks for distributing token notifications
    /// Returns schedulers and their corresponding KV event receivers
    fn start_schedulers(
        &self,
        args: MockEngineArgs,
        active_requests: Arc<Mutex<HashMap<Uuid, mpsc::UnboundedSender<OutputSignal>>>>,
        cancel_token: CancellationToken,
    ) -> (
        Vec<Scheduler>,
        Vec<mpsc::UnboundedReceiver<KvCacheEventData>>,
    ) {
        let mut schedulers = Vec::<Scheduler>::new();
        let mut kv_event_receivers = Vec::new();
        let mut senders = Vec::with_capacity(args.dp_size as usize);

        // Create multiple schedulers and their background tasks
        for dp_rank in 0..args.dp_size {
            // Create a shared output channel that this scheduler will use
            let (output_tx, mut output_rx) = mpsc::unbounded_channel::<OutputSignal>();

            // Create a channel for KV events from this scheduler
            let (kv_events_tx, kv_events_rx) = mpsc::unbounded_channel::<KvCacheEventData>();

            let scheduler = Scheduler::new(
                args.clone(),
                Some(dp_rank),
                Some(output_tx),
                Some(kv_events_tx), // Pass the KV events sender to scheduler
                Some(cancel_token.clone()),
            );

            senders.push(scheduler.request_sender());
            schedulers.push(scheduler);
            kv_event_receivers.push(kv_events_rx);

            // Spawn a background task for this scheduler to distribute token notifications to active requests
            // let output_rx = Arc::new(Mutex::new(output_rx));
            let active_requests_clone = active_requests.clone();
            let cancel_token_cloned = cancel_token.clone();

            tokio::spawn(async move {
                loop {
                    tokio::select! {
                        signal_result = output_rx.recv() => {
                            let Some(signal) = signal_result else {
                                break; // Channel closed
                            };

                            // Notify the specific request that a token was generated
                            let active = active_requests_clone.lock().await;
                            if let Some(request_tx) = active.get(&signal.uuid) {
                                let _ = request_tx.send(signal);
                            }
                        }
                        _ = cancel_token_cloned.cancelled() => {
                            break;
                        }
                    }
                }
            });
        }

        // Set the senders once
        self.request_senders
            .set(senders)
            .expect("Already initialized");

        (schedulers, kv_event_receivers)
    }

    /// Start background tasks to poll and publish metrics every second
    async fn start_metrics_publishing(
        schedulers: &[Scheduler],
        component: Option<Component>,
        cancel_token: CancellationToken,
    ) -> Result<()> {
        tracing::info!("Creating metrics publisher");
        let metrics_publisher = Arc::new(WorkerMetricsPublisher::new()?);
        tracing::info!("Metrics publisher created");

        if let Some(comp) = component {
            tracing::info!("Creating metrics endpoint");
            tokio::spawn({
                let publisher = metrics_publisher.clone();
                async move {
                    if let Err(e) = publisher.create_endpoint(comp.clone()).await {
                        tracing::error!("Metrics endpoint failed: {e}");
                    }
                }
            });

            // Give it a moment to start
            tokio::time::sleep(Duration::from_millis(100)).await;
            tracing::info!("Metrics endpoint started (background)");
        }

        tracing::info!("Starting metrics background tasks");
        for (dp_rank, scheduler) in schedulers.iter().enumerate() {
            let scheduler = scheduler.clone();
            let publisher = metrics_publisher.clone();
            let dp_rank = dp_rank as u32;
            let cancel_token = cancel_token.clone();

            tokio::spawn(async move {
                let mut interval = interval(Duration::from_millis(100));

                loop {
                    tokio::select! {
                        _ = interval.tick() => {
                            // Get metrics from scheduler
                            let metrics = scheduler.get_forward_pass_metrics().await;

                            // Publish metrics
                            if let Err(e) = publisher.publish(Arc::new(metrics)) {
                                tracing::warn!("Failed to publish metrics for DP rank {dp_rank}: {e}");
                            } else {
                                tracing::trace!("Published metrics for DP rank {}", dp_rank);
                            }
                        }
                        _ = cancel_token.cancelled() => {
                            tracing::info!("Metrics publishing cancelled for DP rank {dp_rank}");
                            break;
                        }
                    }
                }
            });
        }
        tracing::info!("Metrics background tasks started");
        Ok(())
    }

    /// Start background tasks to collect and publish KV events from schedulers
    async fn start_kv_events_publishing(
        kv_event_receivers: Vec<mpsc::UnboundedReceiver<KvCacheEventData>>,
        component: Option<Component>,
        block_size: usize,
        cancel_token: CancellationToken,
    ) -> Result<()> {
        tracing::info!("Starting KV events publishing");

        // Only start KV events publishing if we have a component
        let Some(comp) = component else {
            tracing::warn!("No component provided, skipping KV events publishing");
            return Ok(());
        };
        tracing::info!("Component found for KV events publishing");

        tracing::debug!("Getting worker_id");
        let worker_id = comp
            .drt()
            .primary_lease()
            .expect("Cannot publish KV events without lease") // ← This will PANIC on static!
            .id();
        // let worker_id = 0;
        tracing::debug!("Worker_id set to: {worker_id}");

        tracing::info!("Creating KV event publisher");
        let kv_event_publisher = Arc::new(KvEventPublisher::new(
            comp.clone(),
            worker_id,
            block_size as u32,
            None,
        )?);
        tracing::info!("KV event publisher created");

        tracing::info!(
            "Starting KV event background tasks for {} receivers",
            kv_event_receivers.len()
        );
        for (dp_rank, mut kv_events_rx) in kv_event_receivers.into_iter().enumerate() {
            tracing::debug!("Starting background task for DP rank {dp_rank}");
            let publisher = kv_event_publisher.clone();
            let dp_rank = dp_rank as u32;
            let cancel_token = cancel_token.clone();

            tokio::spawn(async move {
                tracing::debug!("Background task started for DP rank {dp_rank}");
                loop {
                    tokio::select! {
                        // Receive actual KV events from the scheduler
                        Some(event_data) = kv_events_rx.recv() => {
                            // Convert KvCacheEventData to KvCacheEvent with random UUID as event_id
                            let event = KvCacheEvent {
                                event_id: Uuid::new_v4().as_u128() as u64,
                                data: event_data,
                            };

                            // Publish the event
                            if let Err(e) = publisher.publish(event) {
                                tracing::warn!("Failed to publish KV event for DP rank {dp_rank}: {e}");
                            } else {
                                tracing::trace!("Published KV event for DP rank {dp_rank}");
                            }
                        }
                        _ = cancel_token.cancelled() => {
                            tracing::info!("KV events publishing cancelled for DP rank {dp_rank}");
                            break;
                        }
                    }
                }
            });
        }
        tracing::info!("All KV event background tasks started");

        Ok(())
    }
}

#[async_trait]
impl AsyncEngine<SingleIn<PreprocessedRequest>, ManyOut<LLMEngineOutput>, Error>
    for MockVllmEngine
{
    async fn generate(
        &self,
        input: SingleIn<PreprocessedRequest>,
    ) -> Result<ManyOut<LLMEngineOutput>, Error> {
        let (request, ctx) = input.into_parts();

        // Extract dp_rank from annotations if present
        let dp_rank = request
            .annotations
            .iter()
            .find_map(|ann| {
                if ann.starts_with("dp_rank:") {
                    ann.strip_prefix("dp_rank:").and_then(|s| s.parse().ok())
                } else {
                    None
                }
            })
            .unwrap_or(0);

        // Validate dp_rank
        if dp_rank >= self.engine_args.dp_size {
            return Err(Error::msg(format!(
                "dp_rank {} is out of bounds for dp_size {}",
                dp_rank, self.engine_args.dp_size
            )));
        }

        let request_uuid = ctx.id().parse().unwrap_or(Uuid::new_v4());

        // Convert PreprocessedRequest to DirectRequest for scheduler
        let direct_request = DirectRequest {
            tokens: request.token_ids.clone(),
            max_output_tokens: request
                .stop_conditions
                .max_tokens
                .expect("max_output_tokens must be specified for mocker")
                as usize,
            uuid: Some(request_uuid),
            dp_rank: Some(dp_rank),
        };

        let (request_tx, mut request_rx) = mpsc::unbounded_channel::<OutputSignal>();
        {
            let mut active = self.active_requests.lock().await;
            active.insert(request_uuid, request_tx);
        }

        // Send the request to the appropriate scheduler based on dp_rank
        self.direct(direct_request, dp_rank as usize);

        // Create a simple channel for the stream
        let (stream_tx, stream_rx) = mpsc::channel::<LLMEngineOutput>(64);

        let active_requests = self.active_requests.clone();
        let async_context = ctx.context();
        let max_tokens = request.stop_conditions.max_tokens.unwrap_or(100) as usize;

        // Spawn a task to handle the complex async logic
        tokio::spawn(async move {
            let mut token_count = 0;

            loop {
                tokio::select! {
                    maybe_signal = request_rx.recv() => {
                        let Some(signal) = maybe_signal else {
                            let _ = stream_tx.send(LLMEngineOutput::error("All output transmitters closed".to_string())).await;
                            break;
                        };

                        if signal.completed && token_count < max_tokens - 1 {
                            let _ = stream_tx.send(LLMEngineOutput::error("Completion signal received before max tokens reached".to_string())).await;
                            break;
                        }

                        if signal.completed {
                            let _ = stream_tx.send(LLMEngineOutput::length()).await;
                            break;
                        }

                        // Generate a new token
                        let token_id = generate_random_token();
                        token_count += 1;

                        let output = LLMEngineOutput {
                            token_ids: vec![token_id],
                            tokens: None,  // Let backend handle detokenization
                            text: None,
                            cum_log_probs: None,
                            log_probs: None,
                            top_logprobs: None,
                            finish_reason: None,
                            index: None,
                        };

                        if stream_tx.send(output).await.is_err() {
                            break;
                        }
                    }

                    _ = async_context.stopped() => {
                        let _ = stream_tx.send(LLMEngineOutput::cancelled()).await;
                        break;
                    }
                }
            }

            // Clean up: remove this request from active requests
            let mut active = active_requests.lock().await;
            active.remove(&request_uuid);
        });

        // Create a simple ReceiverStream which is naturally Send + Sync
        let stream = ReceiverStream::new(stream_rx);
        Ok(ResponseStream::new(Box::pin(stream), ctx.context()))
    }
}

pub struct AnnotatedMockEngine {
    inner: Arc<MockVllmEngine>,
}

impl AnnotatedMockEngine {
    pub fn new(
        inner: MockVllmEngine,
        distributed_runtime: DistributedRuntime,
        endpoint: dynamo_runtime::protocols::Endpoint,
    ) -> Self {
        let inner = Arc::new(inner);
        let inner_clone = inner.clone();

        // Start background task to wait for component service and start the engine
        tokio::spawn(async move {
            loop {
                // Try to create component
                let Ok(namespace) = distributed_runtime.namespace(&endpoint.namespace) else {
                    tracing::debug!("Namespace not available yet, retrying...");
                    tokio::time::sleep(Duration::from_millis(100)).await;
                    continue;
                };

                let Ok(component) = namespace.component(&endpoint.component) else {
                    tracing::debug!("Component not available yet, retrying...");
                    tokio::time::sleep(Duration::from_millis(100)).await;
                    continue;
                };

                // Check if service is available by trying to list instances
                let Ok(instances) = component.list_instances().await else {
                    tracing::debug!("Cannot list instances yet, retrying...");
                    tokio::time::sleep(Duration::from_millis(100)).await;
                    continue;
                };

                if instances.is_empty() {
                    tracing::debug!("No instances available yet, retrying...");
                    tokio::time::sleep(Duration::from_millis(100)).await;
                    continue;
                }

                tracing::info!("Component service is now available, starting mocker engine");

                // Start the engine with the component
                if let Err(e) = inner_clone.start(component).await {
                    tracing::error!("Failed to start mocker engine: {e}");
                }
                break;
            }
        });

        Self { inner }
    }
}

#[async_trait]
impl AsyncEngine<SingleIn<PreprocessedRequest>, ManyOut<Annotated<LLMEngineOutput>>, Error>
    for AnnotatedMockEngine
{
    async fn generate(
        &self,
        input: SingleIn<PreprocessedRequest>,
    ) -> Result<ManyOut<Annotated<LLMEngineOutput>>, Error> {
        let stream = self.inner.generate(input).await?;
        let context = stream.context();

        // Convert stream of LLMEngineOutput to Annotated<LLMEngineOutput>
        let annotated_stream = stream.map(Annotated::from_data);

        Ok(ResponseStream::new(Box::pin(annotated_stream), context))
    }
}

/// Create a mocker engine as ExecutionContext
pub async fn make_mocker_engine(
    distributed_runtime: DistributedRuntime,
    endpoint: dynamo_runtime::protocols::Endpoint,
    args: MockEngineArgs,
) -> Result<crate::backend::ExecutionContext, Error> {
    // Create the mocker engine
    tracing::info!("Creating mocker engine with config: {args:?}");
    let annotated_engine =
        AnnotatedMockEngine::new(MockVllmEngine::new(args), distributed_runtime, endpoint);

    Ok(Arc::new(annotated_engine))
}

#[cfg(test)]
mod integration_tests {
    use super::*;
    use crate::kv_router::indexer::RouterEvent;
    use crate::kv_router::KV_EVENT_SUBJECT;
    use crate::protocols::common::{OutputOptions, SamplingOptions, StopConditions};
    use dynamo_runtime::{
        pipeline::Context,
        pipeline::{network::Ingress, PushRouter},
        traits::events::EventSubscriber,
        DistributedRuntime, Worker,
    };
    use futures::StreamExt;
    use tokio::time::timeout;

    #[tokio::test]
    #[ignore] // Run with: cargo test -- --ignored
    async fn test_mock_vllm_engine_full_integration() -> Result<()> {
        const DP_SIZE: u32 = 2;
        const TOKENS_PER_REQUEST: usize = 20;
        const BLOCK_SIZE: usize = 2;

        // Create runtime and distributed runtime
        let worker = Worker::from_settings()?;
        let runtime = worker.runtime();
        let distributed = DistributedRuntime::from_settings(runtime.clone()).await?;
        tracing::info!("✓ Runtime and distributed runtime created");

        // Create component for MockVllmEngine (needed for publishers)
        let test_component = distributed
            .namespace("test")?
            .component(MOCKER_COMPONENT)?
            .service_builder()
            .create()
            .await?;
        tracing::info!("✓ Test component created");

        // Create MockVllmEngine WITH component (enables publishers)
        let args = MockEngineArgs::builder()
            .speedup_ratio(10.0)
            .dp_size(DP_SIZE)
            .block_size(BLOCK_SIZE)
            .build()
            .unwrap();

        let engine = MockVllmEngine::new(args);
        engine.start(test_component.clone()).await?;
        tokio::time::sleep(tokio::time::Duration::from_millis(500)).await;
        let engine = Arc::new(engine);
        tracing::info!("✓ MockVllmEngine created with DP_SIZE: {DP_SIZE}");

        // Set up KV events subscriber
        let mut kv_events_subscriber = test_component.subscribe(KV_EVENT_SUBJECT).await?;
        tracing::info!("✓ KV events subscriber created");

        // Wrap with Ingress and register with component/endpoint
        let ingress = Ingress::for_engine(engine)?;
        tracing::info!("✓ Ingress wrapper created");

        // Start the server in background
        let server_handle = tokio::spawn({
            let test_component = test_component.clone();
            async move {
                if let Err(e) = test_component
                    .endpoint("generate")
                    .endpoint_builder()
                    .handler(ingress)
                    .start()
                    .await
                {
                    eprintln!("❌ Generate endpoint failed: {e}");
                }
            }
        });
        tracing::info!("✓ Server started in background");

        // Give server time to start
        tokio::time::sleep(tokio::time::Duration::from_millis(500)).await;
        tracing::info!("✓ Server startup delay completed");

        // Print all registered instances from etcd
        match test_component.list_instances().await {
            Ok(instances) => {
                tracing::info!("📋 Found {} registered instances:", instances.len());
                for instance in instances {
                    tracing::info!(
                        "  • {}/{}/{} (ID: {})",
                        instance.namespace,
                        instance.component,
                        instance.endpoint,
                        instance.instance_id
                    );
                }
            }
            Err(e) => {
                tracing::error!("❌ Failed to list instances: {e}");
            }
        }

        // Create client
        let client = distributed
            .namespace("test")?
            .component(MOCKER_COMPONENT)?
            .endpoint("generate")
            .client()
            .await?;
        tracing::info!("✓ Client created");

        let router = PushRouter::from_client(client, Default::default()).await?;
        tracing::info!("✓ Router created");

        // Create test requests for both DP workers
        let create_request = |tokens: Vec<TokenIdType>, dp_rank: u32| PreprocessedRequest {
            model: "mock".to_string(),
            token_ids: tokens,
            batch_token_ids: None,
            stop_conditions: StopConditions {
                max_tokens: Some(TOKENS_PER_REQUEST as u32),
                ..Default::default()
            },
            sampling_options: SamplingOptions::default(),
            output_options: OutputOptions::default(),
            eos_token_ids: vec![],
            mdc_sum: None,
            annotations: vec![format!("dp_rank:{dp_rank}")],
            estimated_prefix_hit_num_blocks: None,
        };

        let requests = vec![
            create_request(vec![1, 2, 3, 4, 5], 0),
            create_request(vec![1, 2, 3, 4, 5], 0),
            create_request(vec![1, 2, 3, 4, 5], 1),
            create_request(vec![1, 2, 3, 4, 5], 1),
        ];
        tracing::info!(
            "✓ Test requests created ({} requests total)",
            requests.len()
        );

        // Test each request
        for (i, request) in requests.into_iter().enumerate() {
            tracing::info!("Testing request {}", i + 1);

            let response_stream = router.generate(Context::new(request)).await?;
            let responses: Vec<LLMEngineOutput> = response_stream.collect().await;

            // Should have at least one response
            assert!(
                !responses.is_empty(),
                "Request {} should produce at least one response",
                i + 1
            );

            // Count total tokens generated (excluding final message)
            let mut total_tokens = 0;
            let mut has_finish_reason = false;

            for response in &responses {
                total_tokens += response.token_ids.len();
                if response.finish_reason.is_some() {
                    has_finish_reason = true;
                }
            }

            // Should have a finish reason in the last response
            assert!(
                has_finish_reason,
                "Request {} should have a finish reason",
                i + 1
            );

            // Verify we got approximately the expected number of tokens
            assert!(
                total_tokens <= TOKENS_PER_REQUEST + 1, // +1 for potential final empty response
                "Request {} generated {} tokens, expected at most {}",
                i + 1,
                total_tokens,
                TOKENS_PER_REQUEST + 1
            );

            tracing::info!(
                "✓ Request {} completed successfully with {} tokens",
                i + 1,
                total_tokens
            );
        }

        tracing::info!("🎉 All requests completed successfully!");

        // Try to receive at least one KV event with 100ms timeout
        tracing::info!("Waiting for KV event with 100ms timeout...");
        let msg = timeout(Duration::from_millis(100), kv_events_subscriber.next())
            .await
            .map_err(|_| Error::msg("Timeout waiting for KV event"))?
            .ok_or_else(|| Error::msg("KV events stream ended unexpectedly"))?;

        match serde_json::from_slice::<RouterEvent>(&msg.payload) {
            Ok(event) => {
                tracing::info!("✓ Received KV event: {event:?}");
            }
            Err(e) => {
                return Err(Error::msg(format!("Failed to deserialize KV event: {e}")));
            }
        }

        // Use KvMetricsAggregator to get metrics more easily
        let cancel_token = test_component.drt().runtime().child_token();
        let metrics_aggregator = crate::kv_router::metrics_aggregator::KvMetricsAggregator::new(
            test_component.clone(),
            cancel_token,
        )
        .await;
        tokio::time::sleep(tokio::time::Duration::from_millis(500)).await;

        let processed_endpoints = metrics_aggregator.get_endpoints();
        tracing::info!(
            "Found {} metrics endpoints",
            processed_endpoints.endpoints.len()
        );

        // Verify we found at least one metrics endpoint
        assert!(
            !processed_endpoints.endpoints.is_empty(),
            "Should find at least one metrics endpoint"
        );
        tracing::info!(
            "✓ Successfully found {} metrics endpoints",
            processed_endpoints.endpoints.len()
        );

        // Verify the metrics endpoints contain valid data
        for (worker_id, endpoint) in &processed_endpoints.endpoints {
            tracing::info!("✓ Worker {} metrics: {:?}", worker_id, endpoint.data);
        }

        tracing::info!("🎉 Event verification completed!");

        // Cleanup
        distributed.shutdown();
        server_handle.await?;

        Ok(())
    }
}
