// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The Preprocessor consists of the following modules
//!
//! - `translation`: This module converts the allowed Ingress message types to the corresponding
//!   internal representation.
//! - `apply`: This module applies ModelConfig defaults to any empty optional fields specified
//! - `prompt`: This module applies any prompt template logic to the internal Request object.
//! - `tokenize`: This module tokenizes the formatted prompt string and returns the token ids.
//!
//! The Preprocessor will accept any IngressRequest and transform it to a BackendRequest.

pub mod prompt;
pub mod tools;

use anyhow::Result;
use async_openai::types::EncodingFormat;
use futures::stream::{self, StreamExt};
use prompt::OAIPromptFormatter;
use rayon::iter::{IntoParallelRefIterator, ParallelIterator};
use std::{collections::HashMap, sync::Arc};
use tracing;

use crate::model_card::{ModelDeploymentCard, ModelInfo, TokenizerKind};
use crate::preprocessor::prompt::OAIChatLikeRequest;
use crate::tokenizers::Encoding;

use dynamo_runtime::engine::{AsyncEngine, AsyncEngineContextProvider, ResponseStream};
use dynamo_runtime::pipeline::{
    async_trait, AsyncEngineContext, Error, ManyOut, Operator, SingleIn,
};
use dynamo_runtime::protocols::annotated::{Annotated, AnnotationsProvider};

use crate::protocols::{
    common::{OutputOptionsProvider, SamplingOptionsProvider, StopConditionsProvider},
    openai::{
        chat_completions::{NvCreateChatCompletionRequest, NvCreateChatCompletionStreamResponse},
        completions::{NvCreateCompletionRequest, NvCreateCompletionResponse},
        embeddings::{NvCreateEmbeddingRequest, NvCreateEmbeddingResponse},
        nvext::NvExtProvider,
        DeltaGeneratorExt,
    },
};
use crate::tokenizers::{traits::Tokenizer, HuggingFaceTokenizer};

use crate::preprocessor::prompt::{PromptFormatter, PromptInput, TextInput, TokenInput};

pub use crate::protocols::common::llm_backend::{BackendOutput, PreprocessedRequest};
pub use crate::protocols::common::preprocessor::PreprocessedEmbeddingRequest;

use crate::protocols::common::llm_backend::EmbeddingsEngineOutput;

pub const ANNOTATION_FORMATTED_PROMPT: &str = "formatted_prompt";
pub const ANNOTATION_TOKEN_IDS: &str = "token_ids";
pub const ANNOTATION_LLM_METRICS: &str = "llm_metrics";
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct LLMMetricAnnotation {
    pub input_tokens: usize,
    pub output_tokens: usize,
    pub chunk_tokens: usize,
}

impl LLMMetricAnnotation {
    /// Convert this metrics struct to an Annotated event
    pub fn to_annotation<T>(&self) -> Result<Annotated<T>, serde_json::Error> {
        Annotated::from_annotation(ANNOTATION_LLM_METRICS, self)
    }

    /// Extract LLM metrics from an Annotated event, if present
    pub fn from_annotation<T>(
        annotation: &Annotated<T>,
    ) -> Result<Option<LLMMetricAnnotation>, Box<dyn std::error::Error>> {
        if annotation.event.is_none() {
            return Ok(None);
        }
        if annotation.event.as_ref().unwrap() != ANNOTATION_LLM_METRICS {
            return Ok(None);
        }
        let comments = annotation
            .comment
            .as_ref()
            .ok_or("missing comments block")?;
        if comments.len() != 1 {
            return Err("malformed comments block - expected exactly 1 comment".into());
        }
        let metrics: LLMMetricAnnotation = serde_json::from_str(&comments[0])?;
        Ok(Some(metrics))
    }
}

pub struct OpenAIPreprocessor {
    mdcsum: String,
    formatter: Arc<dyn OAIPromptFormatter>,
    tokenizer: Arc<dyn Tokenizer>,
    model_info: Arc<dyn ModelInfo>,
}

impl OpenAIPreprocessor {
    pub async fn new(mdc: ModelDeploymentCard) -> Result<Arc<Self>> {
        let mdcsum = mdc.mdcsum();
        let formatter = PromptFormatter::from_mdc(mdc.clone()).await?;
        let PromptFormatter::OAI(formatter) = formatter;

        let tokenizer = match &mdc.tokenizer {
            Some(TokenizerKind::HfTokenizerJson(file)) => HuggingFaceTokenizer::from_file(file)?,
            Some(TokenizerKind::GGUF(tokenizer)) => {
                HuggingFaceTokenizer::from_tokenizer(*tokenizer.clone())
            }
            None => {
                anyhow::bail!(
                    "Blank ModelDeploymentCard cannot be used for pre-processing, no tokenizer"
                );
            }
        };
        let tokenizer = Arc::new(tokenizer);

        let Some(model_info) = mdc.model_info else {
            anyhow::bail!(
                "Blank ModelDeploymentCard cannot be used for pre-processing, no model_info"
            );
        };
        let model_info = model_info.get_model_info().await?;

        Ok(Arc::new(Self {
            formatter,
            tokenizer,
            model_info,
            mdcsum,
        }))
    }

    /// Encode a string to it's tokens
    pub fn tokenize(&self, s: &str) -> anyhow::Result<Encoding> {
        self.tokenizer.encode(s)
    }

    /// Translate a [`NvCreateChatCompletionRequest`] request to a common completion request.
    /// Returns both the common completion request and a hashmap of annotations.
    ///
    /// Annotations evaluated by this method include:
    /// - `formatted_prompt`
    /// - `token_ids`
    pub fn preprocess_request<
        R: OAIChatLikeRequest
            + AnnotationsProvider
            + SamplingOptionsProvider
            + StopConditionsProvider
            + OutputOptionsProvider
            + NvExtProvider,
    >(
        &self,
        request: &R,
    ) -> Result<(PreprocessedRequest, HashMap<String, String>)> {
        let mut annotations = HashMap::new();
        let mut builder = PreprocessedRequest::builder();
        builder.model(request.model());

        // match request type before any conversion/processing
        match request.prompt_input_type() {
            PromptInput::Tokens(_) => {
                if let Some(token_input) = request.extract_tokens() {
                    match token_input {
                        TokenInput::Single(tokens) => {
                            builder.token_ids(tokens);
                        }
                        TokenInput::Batch(token_batches) => {
                            if token_batches.len() == 1 {
                                builder.token_ids(token_batches[0].clone());
                            } else {
                                builder.batch_token_ids(Some(token_batches));
                                builder.token_ids(vec![]);
                            }
                        }
                    }
                }
            }
            PromptInput::Text(_) => {
                if let Some(text_input) = request.extract_text() {
                    match text_input {
                        TextInput::Single(_) => {
                            let use_raw_prompt = request
                                .nvext()
                                .is_some_and(|ext| ext.use_raw_prompt.unwrap_or(false));

                            let formatted_prompt = if use_raw_prompt {
                                match request.raw_prompt() {
                                    Some(prompt) => prompt,
                                    None => {
                                        tracing::warn!("Raw prompt requested but not available");
                                        self.formatter.render(request)?
                                    }
                                }
                            } else {
                                self.formatter.render(request)?
                            };

                            let encoding = self.tokenizer.encode(&formatted_prompt)?;

                            if request.has_annotation(ANNOTATION_FORMATTED_PROMPT) {
                                annotations.insert(
                                    ANNOTATION_FORMATTED_PROMPT.to_string(),
                                    formatted_prompt,
                                );
                            }

                            if request.has_annotation(ANNOTATION_TOKEN_IDS) {
                                annotations.insert(
                                    ANNOTATION_TOKEN_IDS.to_string(),
                                    serde_json::to_string(encoding.token_ids())?,
                                );
                            }

                            builder.token_ids(encoding.token_ids().to_vec());
                        }
                        TextInput::Batch(texts) => {
                            let token_batches: Vec<Vec<u32>> = texts
                                .par_iter()
                                .map(|text| {
                                    self.tokenizer
                                        .encode(text)
                                        .map(|encoded| encoded.token_ids().to_vec())
                                })
                                .collect::<Result<Vec<_>>>()?;
                            builder.batch_token_ids(Some(token_batches));
                            builder.token_ids(vec![]);
                        }
                    }
                }
            }
        }

        let mut stop_conditions = request.extract_stop_conditions()?;
        if let Some(stop_tokens) = &mut stop_conditions.stop_token_ids_hidden {
            for eos_token in self.model_info.eos_token_ids() {
                if !stop_tokens.contains(&eos_token) {
                    stop_tokens.push(eos_token);
                }
            }
        } else {
            stop_conditions.stop_token_ids_hidden = Some(self.model_info.eos_token_ids());
        }

        // apply ignore eos if not already set
        stop_conditions.apply_ignore_eos();

        if !stop_conditions.ignore_eos.unwrap_or(false) {
            builder.eos_token_ids(self.model_info.eos_token_ids());
        }

        builder.stop_conditions(stop_conditions);
        builder.sampling_options(request.extract_sampling_options()?);
        builder.output_options(request.extract_output_options()?);
        builder.annotations(request.annotations().unwrap_or_default());
        builder.mdc_sum(Some(self.mdcsum.clone()));
        builder.estimated_prefix_hit_num_blocks(None);

        Ok((builder.build()?, annotations))
    }

    /// Preprocess an embedding request, handling both text and token ID inputs.
    ///
    /// For text inputs, tokenizes the text using the configured tokenizer.
    /// For token ID inputs, uses the provided token IDs directly and skips tokenization.
    ///
    /// Returns both the preprocessed request and a hashmap of annotations.
    pub async fn preprocess_embedding_request(
        &self,
        request: &NvCreateEmbeddingRequest,
    ) -> Result<(PreprocessedEmbeddingRequest, HashMap<String, String>)> {
        let mut annotations = HashMap::new();
        let mut builder = PreprocessedEmbeddingRequest::builder();

        let all_token_ids = match &request.inner.input {
            async_openai::types::EmbeddingInput::String(s) => {
                let encoding = self.tokenizer.encode(s)?;
                vec![encoding.token_ids().to_vec()]
            }
            async_openai::types::EmbeddingInput::StringArray(arr) => {
                let input_strs: Vec<String> = arr.to_vec();
                let encodings = tokio::task::spawn_blocking({
                    let tokenizer = self.tokenizer.clone();
                    let strs = input_strs.clone();
                    move || {
                        tokenizer.encode_batch(&strs.iter().map(|s| s.as_str()).collect::<Vec<_>>())
                    }
                })
                .await??;
                let token_arrays: Vec<Vec<u32>> = encodings
                    .into_iter()
                    .map(|encoding| encoding.token_ids().to_vec())
                    .collect();
                token_arrays
            }
            async_openai::types::EmbeddingInput::IntegerArray(token_ids) => vec![token_ids.clone()],
            async_openai::types::EmbeddingInput::ArrayOfIntegerArray(token_arrays) => {
                token_arrays.clone()
            }
        };

        // Handle annotations
        if request.has_annotation(ANNOTATION_TOKEN_IDS) {
            annotations.insert(
                ANNOTATION_TOKEN_IDS.to_string(),
                serde_json::to_string(&all_token_ids)?,
            );
        }

        builder.token_ids(all_token_ids);
        builder.model(request.inner.model.clone());
        builder.encoding_format(request.inner.encoding_format.as_ref().map(|f| match f {
            EncodingFormat::Float => "float".to_string(),
            EncodingFormat::Base64 => "base64".to_string(),
        }));
        builder.dimensions(request.inner.dimensions);

        builder.annotations(request.annotations().unwrap_or_default());
        builder.mdc_sum(Some(self.mdcsum.clone()));

        Ok((builder.build()?, annotations))
    }

    pub fn transform_postprocessor_stream<Resp: Send + Sync + 'static + std::fmt::Debug>(
        stream: ManyOut<Annotated<BackendOutput>>,
        generator: Box<dyn DeltaGeneratorExt<Resp>>,
    ) -> ManyOut<Annotated<Resp>> {
        let context = stream.context();

        struct State<Resp: Send + Sync + 'static + std::fmt::Debug> {
            response_stream: ManyOut<Annotated<BackendOutput>>,
            response_generator: Box<dyn DeltaGeneratorExt<Resp>>,
            context: Arc<dyn AsyncEngineContext>,
            cancelled: bool,
            cumulative_output_tokens: usize,
        }

        let state = State {
            response_stream: stream,
            response_generator: generator,
            context: context.clone(),
            cancelled: false,
            cumulative_output_tokens: 0,
        };

        // transform the common response stream into a chat response stream
        let stream = stream::unfold(state, |mut inner| {
            async move {
                if let Some(response) = inner.response_stream.next().await {
                    if inner.cancelled {
                        tracing::debug!(
                            request_id = inner.context.id(),
                            "Cancellation issued last message; closing stream"
                        );
                        return None;
                    }

                    tracing::trace!(
                        request_id = inner.context.id(),
                        "Processing common response: {:?}",
                        response
                    );

                    let (chunk_tokens, isl) = if let Some(ref backend_output) = response.data {
                        let chunk_tokens = backend_output.token_ids.len();
                        inner.cumulative_output_tokens += chunk_tokens;

                        let isl = inner.response_generator.get_isl().unwrap_or(0) as usize;

                        (chunk_tokens, isl)
                    } else {
                        (0, 0)
                    };

                    let current_osl = inner.cumulative_output_tokens;

                    let mut response = response.map_data(|data| {
                        inner
                            .response_generator
                            .choice_from_postprocessor(data)
                            .inspect_err(|e| {
                                tracing::error!(
                                    request_id = inner.context.id(),
                                    "Error processing common response: {:?}",
                                    e
                                );
                                inner.cancelled = true;
                                inner.context.stop_generating();
                            })
                            .map_err(|e| e.to_string())
                    });

                    // Create LLM metrics annotation
                    let llm_metrics = LLMMetricAnnotation {
                        input_tokens: isl,
                        output_tokens: current_osl,
                        chunk_tokens,
                    };

                    if let Ok(metrics_annotated) = llm_metrics.to_annotation::<()>() {
                        // Only set event if not already set to avoid overriding existing events (like errors)
                        if response.event.is_none() {
                            response.event = metrics_annotated.event;
                            response.comment = metrics_annotated.comment;
                        }
                    }

                    tracing::trace!(
                        request_id = inner.context.id(),
                        "OpenAI NvCreateChatCompletionStreamResponse: {:?}",
                        response
                    );

                    Some((response, inner))
                } else {
                    // stream closed with out graceful closure
                    // we did not detect an is_finished/completed message
                    // Ok(None)
                    None
                }
            }
        });

        ResponseStream::new(Box::pin(stream), context)
    }

    /// Transform engine embedding output stream to OpenAI embedding response stream
    pub fn transform_embedding_postprocessor_stream(
        stream: ManyOut<Annotated<EmbeddingsEngineOutput>>,
        original_request: NvCreateEmbeddingRequest,
    ) -> ManyOut<Annotated<NvCreateEmbeddingResponse>> {
        let context = stream.context();

        let transformed_stream = stream.map(move |output| {
            output.map_data(|engine_output| {
                // Convert engine output to OpenAI response format
                let embeddings: Vec<async_openai::types::Embedding> = engine_output
                    .embeddings
                    .into_iter()
                    .enumerate()
                    .map(|(index, embedding)| async_openai::types::Embedding {
                        index: index as u32,
                        object: "embedding".to_string(),
                        embedding: embedding.into_iter().map(|f| f as f32).collect(),
                    })
                    .collect();

                let response = NvCreateEmbeddingResponse {
                    inner: async_openai::types::CreateEmbeddingResponse {
                        object: "list".to_string(),
                        model: original_request.inner.model.clone(),
                        data: embeddings,
                        usage: async_openai::types::EmbeddingUsage {
                            prompt_tokens: engine_output.prompt_tokens,
                            total_tokens: engine_output.total_tokens,
                        },
                    },
                };

                Ok(response)
            })
        });

        ResponseStream::new(Box::pin(transformed_stream), context)
    }
}

// for pals, we do not want to add the generation prompt to the formatted prompt
// we also need to know if the template support this add_generation_prompt bool
// any prompt template that does not support this should return an error
// oob - we should update any prompt template that does not support this to support it

#[async_trait]
impl
    Operator<
        SingleIn<NvCreateChatCompletionRequest>,
        ManyOut<Annotated<NvCreateChatCompletionStreamResponse>>,
        SingleIn<PreprocessedRequest>,
        ManyOut<Annotated<BackendOutput>>,
    > for OpenAIPreprocessor
{
    async fn generate(
        &self,
        request: SingleIn<NvCreateChatCompletionRequest>,
        next: Arc<
            dyn AsyncEngine<
                SingleIn<PreprocessedRequest>,
                ManyOut<Annotated<BackendOutput>>,
                Error,
            >,
        >,
    ) -> Result<ManyOut<Annotated<NvCreateChatCompletionStreamResponse>>, Error> {
        // unpack the request
        let (request, context) = request.into_parts();

        // create a response generator
        let response_generator = request.response_generator();
        let mut response_generator = Box::new(response_generator);

        // convert the chat completion request to a common completion request
        let (common_request, annotations) = self.preprocess_request(&request)?;

        // update isl
        response_generator.update_isl(common_request.token_ids.len() as u32);

        // repack the common completion request
        let common_request = context.map(|_| common_request);

        // create a stream of annotations this will be prepend to the response stream
        let annotations: Vec<Annotated<NvCreateChatCompletionStreamResponse>> = annotations
            .into_iter()
            .flat_map(|(k, v)| Annotated::from_annotation(k, &v))
            .collect();
        let annotations_stream = stream::iter(annotations);

        // forward the common completion request to the next operator
        let response_stream = next.generate(common_request).await?;

        // transform the postprocessor stream
        let stream = Self::transform_postprocessor_stream(response_stream, response_generator);
        let context = stream.context();

        // prepend the annotations to the response stream
        let stream = annotations_stream.chain(stream);

        // return the response stream
        Ok(ResponseStream::new(Box::pin(stream), context))
    }
}

#[async_trait]
impl
    Operator<
        SingleIn<NvCreateCompletionRequest>,
        ManyOut<Annotated<NvCreateCompletionResponse>>,
        SingleIn<PreprocessedRequest>,
        ManyOut<Annotated<BackendOutput>>,
    > for OpenAIPreprocessor
{
    async fn generate(
        &self,
        request: SingleIn<NvCreateCompletionRequest>,
        next: Arc<
            dyn AsyncEngine<
                SingleIn<PreprocessedRequest>,
                ManyOut<Annotated<BackendOutput>>,
                Error,
            >,
        >,
    ) -> Result<ManyOut<Annotated<NvCreateCompletionResponse>>, Error> {
        // unpack the request
        let (request, context) = request.into_parts();

        // create a response generator
        let response_generator = request.response_generator();
        let mut response_generator = Box::new(response_generator);
        // convert the chat completion request to a common completion request
        let (common_request, annotations) = self.preprocess_request(&request)?;

        // update isl
        response_generator.update_isl(common_request.token_ids.len() as u32);

        // repack the common completion request
        let common_request = context.map(|_| common_request);

        // create a stream of annotations this will be prepend to the response stream
        let annotations: Vec<Annotated<NvCreateCompletionResponse>> = annotations
            .into_iter()
            .flat_map(|(k, v)| Annotated::from_annotation(k, &v))
            .collect();
        let annotations_stream = stream::iter(annotations);

        // forward the common completion request to the next operator
        let response_stream = next.generate(common_request).await?;

        // transform the postprocessor stream
        let stream = Self::transform_postprocessor_stream(response_stream, response_generator);
        let context = stream.context();

        // prepend the annotations to the response stream
        let stream = annotations_stream.chain(stream);

        // return the response stream
        Ok(ResponseStream::new(Box::pin(stream), context))
    }
}

#[async_trait]
impl
    Operator<
        SingleIn<NvCreateEmbeddingRequest>,
        ManyOut<Annotated<NvCreateEmbeddingResponse>>,
        SingleIn<PreprocessedEmbeddingRequest>,
        ManyOut<Annotated<EmbeddingsEngineOutput>>,
    > for OpenAIPreprocessor
{
    async fn generate(
        &self,
        request: SingleIn<NvCreateEmbeddingRequest>,
        next: Arc<
            dyn AsyncEngine<
                SingleIn<PreprocessedEmbeddingRequest>,
                ManyOut<Annotated<EmbeddingsEngineOutput>>,
                Error,
            >,
        >,
    ) -> Result<ManyOut<Annotated<NvCreateEmbeddingResponse>>, Error> {
        // Unpack request
        let (request, context) = request.into_parts();

        // Preprocess the embedding request
        let (preprocessed_request, annotations) =
            self.preprocess_embedding_request(&request).await?;

        // Forward to next stage
        let preprocessed_request = context.map(|_| preprocessed_request);
        let response_stream = next.generate(preprocessed_request).await?;

        // Transform response stream back to OpenAI format
        let stream = Self::transform_embedding_postprocessor_stream(response_stream, request);
        let context = stream.context();

        // Prepend annotations
        let annotations_stream = stream::iter(
            annotations
                .into_iter()
                .flat_map(|(k, v)| Annotated::from_annotation(k, &v))
                .collect::<Vec<_>>(),
        );

        let combined_stream = annotations_stream.chain(stream);
        Ok(ResponseStream::new(Box::pin(combined_stream), context))
    }
}
