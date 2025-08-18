// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::HashMap;
use std::{num::NonZero, sync::Arc};

use async_openai::types::FinishReason;
use async_stream::stream;
use async_trait::async_trait;
use either::Either;
use indexmap::IndexMap;
use mistralrs::{
    AutoDeviceMapParams, Constraint, DefaultSchedulerMethod, Device, DeviceMapSetting,
    GGUFLoaderBuilder, GGUFSpecificConfig, IsqType, MemoryGpuConfig, MistralRs, MistralRsBuilder,
    ModelDType, NormalLoaderBuilder, NormalRequest, NormalSpecificConfig, PagedAttentionConfig,
    PagedCacheType, Request, RequestMessage, ResponseOk, SamplingParams, SchedulerConfig,
    StopTokens, TokenSource, VisionLoaderBuilder, VisionLoaderType, VisionSpecificConfig,
};
use tokio::sync::mpsc::channel;

use dynamo_runtime::engine::{AsyncEngine, AsyncEngineContextProvider, ResponseStream};
use dynamo_runtime::pipeline::error as pipeline_error;
use dynamo_runtime::pipeline::{Error, ManyOut, SingleIn};
use dynamo_runtime::protocols::annotated::Annotated;

use dynamo_llm::protocols::openai::{
    chat_completions::{NvCreateChatCompletionRequest, NvCreateChatCompletionStreamResponse},
    completions::{prompt_to_string, NvCreateCompletionRequest, NvCreateCompletionResponse},
    embeddings::{NvCreateEmbeddingRequest, NvCreateEmbeddingResponse},
};

use dynamo_llm::engines::{EngineDispatcher, StreamingEngine};
use dynamo_llm::local_model::LocalModel;

/// How many requests mistral will run at once in the paged attention scheduler.
/// It actually runs 1 fewer than this.
/// I would call this the batch size but apparently that's something else.
const PAGED_ATTENTION_MAX_NUM_SEQS: usize = 10;

/// Experimental: Switch this to true to enable paged attention on CUDA devices.
/// Under load (dynamo-run batch mode) paged attention sometimes returns an immediate
/// finish_reason=stop and no tokens for one of the requests.
const EXP_ENABLE_PAGED_ATTENTION: bool = false;

/// Initial message we send to mistral.rs to warm it up. We may not need this.
const WARMUP_MESSAGE: &str = "This is a test message. Respond only with 'OK'.";

pub async fn make_engine(model: &LocalModel) -> pipeline_error::Result<Arc<dyn StreamingEngine>> {
    let engine = MistralRsEngine::new(model).await?;
    let engine: Arc<dyn StreamingEngine> = Arc::new(EngineDispatcher::new(engine));
    Ok(engine)
}

/// Gets the best device, cpu, cuda if compiled with CUDA
fn best_device() -> pipeline_error::Result<Device> {
    #[cfg(not(feature = "metal"))]
    {
        Ok(Device::cuda_if_available(0)?)
    }
    #[cfg(feature = "metal")]
    {
        Ok(Device::new_metal(0)?)
    }
}

struct MistralRsEngine {
    mistralrs: Arc<MistralRs>,
    context_length: usize,
    display_name: String,
}

impl MistralRsEngine {
    async fn new(model: &LocalModel) -> pipeline_error::Result<Self> {
        let model_path = model.path();
        // Name some None's for clarity
        let chat_template = None;
        let tokenizer_json = None;
        let no_kv_cache = false;
        let jinja_explicit = None;
        let display_name = model.display_name();
        let loader = if model_path.is_file() {
            // Load from a GGUF
            let Some(model_filename) = model_path.file_name() else {
                pipeline_error::bail!("Missing filename in model path");
            };
            let Some(model_dir) = model_path.parent() else {
                pipeline_error::bail!("Invalid model path");
            };

            GGUFLoaderBuilder::new(
                chat_template,
                None,
                model_dir.display().to_string(),
                vec![model_filename.to_string_lossy().into_owned()],
                GGUFSpecificConfig::default(),
                no_kv_cache,
                jinja_explicit,
            )
            .build()
        } else if is_vision_model(display_name) {
            let vlt = if is_gemma3(display_name) {
                VisionLoaderType::Gemma3
            } else if is_llama4(display_name) {
                VisionLoaderType::Llama4
            } else {
                panic!("Unsupported vision model {display_name}");
            };
            VisionLoaderBuilder::new(
                VisionSpecificConfig::default(),
                chat_template,
                tokenizer_json,
                Some(model_path.display().to_string()),
                jinja_explicit,
            )
            .build(Some(vlt))
        } else {
            // Load from a HF repo dir
            NormalLoaderBuilder::new(
                NormalSpecificConfig::default(),
                chat_template,
                tokenizer_json,
                Some(model_path.display().to_string()),
                no_kv_cache,
                jinja_explicit,
            )
            .build(None)?
        };

        let mut max_seq_len = model.card().context_length as usize;
        if max_seq_len == 0 {
            tracing::info!("context_length is 0. Probably error reading from model.");
            max_seq_len = AutoDeviceMapParams::DEFAULT_MAX_SEQ_LEN;
        }

        // Paged attention requires cuda
        let paged_attention_config = if cfg!(feature = "cuda") && EXP_ENABLE_PAGED_ATTENTION {
            Some(PagedAttentionConfig::new(
                None, // Block size, default 32
                4096, // CPU memory in MiB
                MemoryGpuConfig::ContextSize(max_seq_len),
                PagedCacheType::Auto,
            )?)
        } else {
            None
        };

        let device_map_params = if is_vision_model(model.display_name()) {
            AutoDeviceMapParams::Vision {
                max_seq_len,
                max_batch_size: AutoDeviceMapParams::DEFAULT_MAX_BATCH_SIZE,
                max_image_shape: (0, 0),
                max_num_images: 0,
            }
        } else {
            AutoDeviceMapParams::Text {
                max_seq_len,
                max_batch_size: AutoDeviceMapParams::DEFAULT_MAX_BATCH_SIZE,
            }
        };

        // Load, into a Pipeline
        let pipeline = loader.load_model_from_hf(
            None,
            TokenSource::None, // The model was already downloaded
            &ModelDType::Auto,
            &best_device()?,
            false,
            DeviceMapSetting::Auto(device_map_params),
            if is_llama4(display_name) {
                Some(IsqType::Q4K)
            } else {
                None
            },
            paged_attention_config,
        )?;
        let scheduler = if cfg!(feature = "cuda") && EXP_ENABLE_PAGED_ATTENTION {
            tracing::debug!("Using mistralrs PagedAttentionMeta scheduler");
            let config = match pipeline.lock().await.get_metadata().cache_config.as_ref() {
                Some(conf) => conf.clone(),
                None => {
                    anyhow::bail!("Failed loading model config");
                }
            };
            SchedulerConfig::PagedAttentionMeta {
                max_num_seqs: PAGED_ATTENTION_MAX_NUM_SEQS,
                config,
            }
        } else {
            SchedulerConfig::DefaultScheduler {
                // Safety: unwrap trivially safe here
                method: DefaultSchedulerMethod::Fixed(NonZero::new(max_seq_len).unwrap()),
            }
        };
        // Create the MistralRs, which is a runner
        let throughput_logging = false;
        let search_embedding_model = None;
        let builder = MistralRsBuilder::new(
            pipeline.clone(),
            scheduler,
            throughput_logging,
            search_embedding_model,
        )
        .with_prefix_cache_n(16);
        let engine = MistralRsEngine {
            mistralrs: builder.build().await,
            context_length: max_seq_len,
            display_name: display_name.to_string(),
        };

        // skip the id used for dummy run https://github.com/EricLBuehler/mistral.rs/issues/1218
        let _ = engine.mistralrs.next_request_id();

        // Perform warmup request
        let (tx, mut rx) = channel(1);
        let request_id = engine.mistralrs.next_request_id();
        let warmup_request = Request::Normal(Box::new(NormalRequest {
            id: request_id,
            model_id: Some(display_name.to_string()),
            messages: RequestMessage::Chat {
                messages: vec![IndexMap::from([
                    ("role".to_string(), Either::Left("user".to_string())),
                    (
                        "content".to_string(),
                        Either::Left(WARMUP_MESSAGE.to_string()),
                    ),
                ])],
                enable_thinking: Some(false),
            },
            sampling_params: SamplingParams::deterministic(),
            response: tx,
            return_logprobs: false,
            is_streaming: false,
            constraint: Constraint::None,
            suffix: None,
            tools: None,
            tool_choice: None,
            logits_processors: None,
            return_raw_logits: false,
            web_search_options: None,
        }));

        // Send warmup request and consume response
        if let Ok(sender) = engine.mistralrs.get_sender(None) {
            if let Ok(()) = sender.send(warmup_request).await {
                if let Some(response) = rx.recv().await {
                    match response.as_result() {
                        Ok(r) => {
                            tracing::debug!(request_id, "Warmup response: {r:?}");
                        }
                        Err(err) => {
                            tracing::error!(request_id, %err, "Failed converting response to result.");
                        }
                    }
                }
            }
        }

        Ok(engine)
    }
}

#[async_trait]
impl
    AsyncEngine<
        SingleIn<NvCreateChatCompletionRequest>,
        ManyOut<Annotated<NvCreateChatCompletionStreamResponse>>,
        Error,
    > for MistralRsEngine
{
    async fn generate(
        &self,
        request: SingleIn<NvCreateChatCompletionRequest>,
    ) -> Result<ManyOut<Annotated<NvCreateChatCompletionStreamResponse>>, Error> {
        let (request, context) = request.transfer(());
        let ctx = context.context();
        let (tx, mut rx) = channel(10_000);

        let mut messages = vec![];
        for m in request.inner.messages {
            let async_openai::types::ChatCompletionRequestMessage::User(inner_m) = m else {
                continue;
            };
            let async_openai::types::ChatCompletionRequestUserMessageContent::Text(content) =
                inner_m.content
            else {
                anyhow::bail!("Only Text type chat completion supported");
            };
            let r = IndexMap::from([
                ("role".to_string(), Either::Left("user".to_string())),
                ("content".to_string(), Either::Left(content)),
            ]);
            messages.push(r);
        }
        if messages.is_empty() {
            anyhow::bail!("Empty request");
        }

        let det = SamplingParams::deterministic();
        // allow deprecated because max_tokens
        #[allow(deprecated)]
        let sampling_params = SamplingParams {
            temperature: request
                .inner
                .temperature
                .map(|t| t as f64)
                .or(det.temperature),
            top_p: request.inner.top_p.map(|t| t as f64).or(det.top_p),
            top_n_logprobs: request
                .inner
                .top_logprobs
                .map(|t| t as usize)
                .unwrap_or(det.top_n_logprobs),
            frequency_penalty: request.inner.frequency_penalty.or(det.frequency_penalty),
            presence_penalty: request.inner.presence_penalty.or(det.presence_penalty),
            stop_toks: request.inner.stop.map(to_stop_tokens).or(det.stop_toks),
            max_len: {
                let requested_max_tokens = request
                    .inner
                    .max_completion_tokens
                    .or(request.inner.max_tokens)
                    .map(|m| m as usize);

                // Ensure max_len doesn't exceed context length
                match requested_max_tokens {
                    Some(max_tokens) => Some(std::cmp::min(max_tokens, self.context_length)),
                    None => det
                        .max_len
                        .map(|len| std::cmp::min(len, self.context_length)),
                }
            },
            logits_bias: request
                .inner
                .logit_bias
                .map(to_logit_bias)
                .or(det.logits_bias),
            // These are not in async-openai yet
            top_k: det.top_k,
            min_p: det.min_p,
            n_choices: 1,
            dry_params: det.dry_params,
        };
        let request_id = self.mistralrs.next_request_id();
        let mistralrs_request = Request::Normal(Box::new(NormalRequest {
            id: request_id,
            model_id: Some(self.display_name.clone()),
            messages: RequestMessage::Chat {
                messages,
                enable_thinking: None,
            },
            sampling_params,
            response: tx,
            return_logprobs: request.inner.logprobs.unwrap_or_default(),
            is_streaming: true,
            constraint: Constraint::None,
            suffix: None,
            tools: None,
            tool_choice: None,
            logits_processors: None,
            return_raw_logits: false,
            web_search_options: None,
        }));

        self.mistralrs
            .get_sender(None)?
            .send(mistralrs_request)
            .await?;

        let output = stream! {
            while let Some(response) = rx.recv().await {
                let response = match response.as_result() {
                    Ok(r) => r,
                    Err(err) => {
                        tracing::error!(request_id, %err, "Failed converting mistralrs channel response to result.");
                        break;
                    }
                };
                match response {
                    ResponseOk::Chunk(c) => {
                        let Some(from_assistant) = c.choices[0].delta.content.clone() else {
                            tracing::warn!(request_id, "No content from mistralrs. Abandoning request.");
                            break;
                        };
                        let finish_reason = match &c.choices[0].finish_reason.as_deref() {
                            Some("stop") | Some("canceled") => {
                                Some(FinishReason::Stop)
                            }
                            Some("length") => {
                                Some(FinishReason::Length)
                            }
                            Some(s) => {
                                tracing::warn!(request_id, stop_reason = s, "Unknow stop reason");
                                Some(FinishReason::Stop)
                            }
                            None => None,
                        };
                        //tracing::trace!("from_assistant: {from_assistant}");

                        #[allow(deprecated)]
                        let inner = async_openai::types::CreateChatCompletionStreamResponse{
                            id: c.id,
                            choices: vec![async_openai::types::ChatChoiceStream{
                                index: 0,
                                delta: async_openai::types::ChatCompletionStreamResponseDelta{
                                    //role: c.choices[0].delta.role,
                                    role: Some(async_openai::types::Role::Assistant),
                                    content: Some(from_assistant),
                                    tool_calls: None,
                                    refusal: None,
                                    function_call: None,
                                },
                                logprobs: None,
                                finish_reason,
                            }],
                            model: c.model,
                            created: c.created as u32,
                            object: c.object.clone(),
                            usage: None,
                            system_fingerprint: Some(c.system_fingerprint),
                            service_tier: None,
                        };
                        let delta = NvCreateChatCompletionStreamResponse{inner};
                        let ann = Annotated{
                            id: None,
                            data: Some(delta),
                            event: None,
                            comment: None,
                        };
                        yield ann;

                        if finish_reason.is_some() {
                            //tracing::trace!(request_id, "Finish reason: {finish_reason:?}");
                            break;
                        }
                    },
                    x => tracing::error!(request_id, "Unhandled. {x:?}"),
                }
            }
        };
        Ok(ResponseStream::new(Box::pin(output), ctx))
    }
}

/// openai stop tokens to mistralrs stop tokens
fn to_stop_tokens(t: async_openai::types::Stop) -> StopTokens {
    match t {
        async_openai::types::Stop::String(s) => StopTokens::Seqs(vec![s]),
        async_openai::types::Stop::StringArray(v) => StopTokens::Seqs(v),
    }
}

/// openai logit bias (strings/json) to mistralrs (u32/f32)
/// I think the input looks like this: {"3721": -100, "17765": 100}
fn to_logit_bias(lb: HashMap<String, serde_json::Value>) -> HashMap<u32, f32> {
    let mut out = HashMap::new();
    for (key, value) in &lb {
        let token_id: u32 = match key.parse() {
            Ok(t) => t,
            Err(err) => {
                tracing::warn!(
                    "Unexpected logit_bias map. Key '{key}' is not an int: {lb:?}. {err}."
                );
                return HashMap::new();
            }
        };
        let Some(bias) = value.as_f64() else {
            tracing::warn!("Unexpected logit_bias map. Value '{value}' is not a float: {lb:?}");
            return HashMap::new();
        };
        out.insert(token_id, bias as f32);
    }
    out
}

#[async_trait]
impl
    AsyncEngine<
        SingleIn<NvCreateCompletionRequest>,
        ManyOut<Annotated<NvCreateCompletionResponse>>,
        Error,
    > for MistralRsEngine
{
    async fn generate(
        &self,
        request: SingleIn<NvCreateCompletionRequest>,
    ) -> Result<ManyOut<Annotated<NvCreateCompletionResponse>>, Error> {
        let (request, context) = request.transfer(());
        let ctx = context.context();
        let (tx, mut rx) = channel(10_000);
        let response_generator = request.response_generator();

        let messages = RequestMessage::Completion {
            text: prompt_to_string(&request.inner.prompt),
            echo_prompt: false,
            best_of: Some(1),
        };
        let det = SamplingParams::deterministic();
        // allow deprecated because max_tokens
        #[allow(deprecated)]
        let sampling_params = SamplingParams {
            temperature: request
                .inner
                .temperature
                .map(|t| t as f64)
                .or(det.temperature),
            top_p: request.inner.top_p.map(|t| t as f64).or(det.top_p),
            top_n_logprobs: request
                .inner
                .logprobs
                .map(|t| t as usize)
                .unwrap_or(det.top_n_logprobs),
            frequency_penalty: request.inner.frequency_penalty.or(det.frequency_penalty),
            presence_penalty: request.inner.presence_penalty.or(det.presence_penalty),
            stop_toks: request
                .inner
                .stop
                .clone()
                .map(to_stop_tokens)
                .or(det.stop_toks),
            max_len: {
                let requested_max_tokens = request.inner.max_tokens.map(|m| m as usize);

                // Ensure max_len doesn't exceed context length
                match requested_max_tokens {
                    Some(max_tokens) => Some(std::cmp::min(max_tokens, self.context_length)),
                    None => det
                        .max_len
                        .map(|len| std::cmp::min(len, self.context_length)),
                }
            },
            logits_bias: request
                .inner
                .logit_bias
                .clone()
                .map(to_logit_bias)
                .or(det.logits_bias),
            // These are not in async-openai yet
            top_k: det.top_k,
            min_p: det.min_p,
            n_choices: 1,
            dry_params: det.dry_params,
        };

        let request_id = self.mistralrs.next_request_id();
        let mistralrs_request = Request::Normal(Box::new(NormalRequest {
            id: request_id,
            model_id: Some(self.display_name.clone()),
            messages,
            sampling_params,
            response: tx,
            return_logprobs: false,
            is_streaming: true,
            constraint: Constraint::None,
            suffix: None,
            tools: None,
            tool_choice: None,
            logits_processors: None,
            return_raw_logits: false,
            web_search_options: None,
        }));

        self.mistralrs
            .get_sender(None)?
            .send(mistralrs_request)
            .await?;

        let output = stream! {
            while let Some(response) = rx.recv().await {
                let response = match response.as_result() {
                    Ok(r) => r,
                    Err(err) => {
                        tracing::error!(request_id, %err, "Failed converting mistralrs channel response to result.");
                        break;
                    }
                };
                match response {
                    ResponseOk::CompletionChunk(c) => {
                        let from_assistant = c.choices[0].text.clone();

                        let finish_reason = match &c.choices[0].finish_reason.as_deref() {
                            Some("stop") | Some("canceled") => {
                                Some(FinishReason::Stop)
                            }
                            Some("length") => {
                                Some(FinishReason::Length)
                            }
                            Some(s) => {
                                tracing::warn!(request_id, stop_reason = s, "Unknow stop reason");
                                Some(FinishReason::Stop)
                            }
                            None => None,
                        };
                        #[allow(deprecated)]
                        let inner = response_generator.create_choice(0, Some(from_assistant), None, None);
                        let ann = Annotated{
                            id: None,
                            data: Some(inner),
                            event: None,
                            comment: None,
                        };
                        yield ann;

                        if finish_reason.is_some() {
                            break;
                        }
                    },
                    x => tracing::error!(request_id, "Unhandled. {x:?}"),
                }
            }
        };
        Ok(ResponseStream::new(Box::pin(output), ctx))
    }
}

fn is_vision_model(s: &str) -> bool {
    is_gemma3(s) || is_llama4(s)
}

fn is_gemma3(s: &str) -> bool {
    s.to_lowercase().contains("gemma-3")
}

fn is_llama4(s: &str) -> bool {
    s.to_lowercase().contains("llama-4")
}

#[async_trait]
impl
    AsyncEngine<
        SingleIn<NvCreateEmbeddingRequest>,
        ManyOut<Annotated<NvCreateEmbeddingResponse>>,
        Error,
    > for MistralRsEngine
{
    async fn generate(
        &self,
        _request: SingleIn<NvCreateEmbeddingRequest>,
    ) -> Result<ManyOut<Annotated<NvCreateEmbeddingResponse>>, Error> {
        unimplemented!()
    }
}
