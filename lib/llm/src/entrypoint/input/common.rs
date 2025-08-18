// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::pin::Pin;

use crate::{
    backend::{Backend, ExecutionContext},
    discovery::{ModelManager, ModelWatcher, MODEL_ROOT_PATH},
    engines::StreamingEngineAdapter,
    entrypoint::{self, EngineConfig},
    kv_router::{KvPushRouter, KvRouter},
    migration::Migration,
    model_card::ModelDeploymentCard,
    preprocessor::OpenAIPreprocessor,
    protocols::common::llm_backend::{BackendOutput, LLMEngineOutput, PreprocessedRequest},
    request_template::RequestTemplate,
    types::{
        openai::chat_completions::{
            NvCreateChatCompletionRequest, NvCreateChatCompletionStreamResponse,
            OpenAIChatCompletionsStreamingEngine,
        },
        Annotated,
    },
};

use dynamo_runtime::{
    component::Client,
    distributed::DistributedConfig,
    engine::{AsyncEngineStream, Data},
    pipeline::{
        Context, ManyOut, Operator, PushRouter, RouterMode, SegmentSource, ServiceBackend,
        ServiceEngine, ServiceFrontend, SingleIn, Source,
    },
    DistributedRuntime, Runtime,
};
use std::sync::Arc;

pub struct PreparedEngine {
    pub service_name: String,
    pub engine: OpenAIChatCompletionsStreamingEngine,
    pub inspect_template: bool,
    pub card: Option<ModelDeploymentCard>,
    pub request_template: Option<RequestTemplate>,
}

impl PreparedEngine {
    pub fn has_tokenizer(&self) -> bool {
        if let Some(card) = self.card.as_ref() {
            card.has_tokenizer()
        } else {
            false
        }
    }
}

/// Turns an EngineConfig into an OpenAI chat-completions and completions supported StreamingEngine.
pub async fn prepare_engine(
    runtime: Runtime,
    engine_config: EngineConfig,
) -> anyhow::Result<PreparedEngine> {
    match engine_config {
        EngineConfig::Dynamic(local_model) => {
            let distributed_runtime = DistributedRuntime::from_settings(runtime.clone()).await?;

            let Some(etcd_client) = distributed_runtime.etcd_client() else {
                anyhow::bail!("Cannot be both static mode and run with dynamic discovery.");
            };
            let model_manager = Arc::new(ModelManager::new());
            let watch_obj = Arc::new(ModelWatcher::new(
                distributed_runtime,
                model_manager.clone(),
                dynamo_runtime::pipeline::RouterMode::RoundRobin,
                None,
            ));
            let models_watcher = etcd_client.kv_get_and_watch_prefix(MODEL_ROOT_PATH).await?;
            let (_prefix, _watcher, receiver) = models_watcher.dissolve();

            let inner_watch_obj = watch_obj.clone();
            let _watcher_task = tokio::spawn(async move {
                inner_watch_obj.watch(receiver).await;
            });
            tracing::info!("Waiting for remote model..");

            // TODO: We use the first model to appear, usually we have only one
            // We should add slash commands to text input `/model <name>` to choose,
            // '/models` to list, and notifications when models are added / removed.

            let model_service_name = watch_obj.wait_for_chat_model().await;
            tracing::info!("Connected to {model_service_name}");
            let engine = model_manager.get_chat_completions_engine(&model_service_name)?;
            Ok(PreparedEngine {
                service_name: model_service_name,
                engine,
                inspect_template: false,
                card: None,
                request_template: local_model.request_template(),
            })
        }
        EngineConfig::StaticRemote(local_model) => {
            // For now we only do ModelType.Backend
            // For batch/text we only do Chat Completions

            // The card should have been loaded at 'build' phase earlier
            let card = local_model.card();
            let router_mode = local_model.router_config().router_mode;

            let dst_config = DistributedConfig::from_settings(true);
            let distributed_runtime = DistributedRuntime::new(runtime, dst_config).await?;

            let endpoint_id = local_model.endpoint_id();
            let component = distributed_runtime
                .namespace(&endpoint_id.namespace)?
                .component(&endpoint_id.component)?;

            let client = component.endpoint(&endpoint_id.name).client().await?;

            let kv_chooser = if router_mode == RouterMode::KV {
                let model_manager = Arc::new(ModelManager::new());
                Some(
                    model_manager
                        .kv_chooser_for(
                            local_model.display_name(),
                            &component,
                            card.kv_cache_block_size,
                            Some(local_model.router_config().kv_router_config),
                        )
                        .await?,
                )
            } else {
                None
            };

            let chat_engine = entrypoint::build_routed_pipeline::<
                NvCreateChatCompletionRequest,
                NvCreateChatCompletionStreamResponse,
            >(card, &client, router_mode, kv_chooser.clone())
            .await?;

            let service_name = local_model.service_name().to_string();
            tracing::info!("Static connecting to {service_name}");
            Ok(PreparedEngine {
                service_name,
                engine: chat_engine,
                inspect_template: false,
                request_template: local_model.request_template(),
                card: Some(local_model.into_card()),
            })
        }
        EngineConfig::StaticFull { engine, model, .. } => {
            let service_name = model.service_name().to_string();
            tracing::debug!("Model: {service_name} with engine pre-processing");
            let engine = Arc::new(StreamingEngineAdapter::new(engine));
            Ok(PreparedEngine {
                service_name,
                engine,
                inspect_template: false,
                request_template: model.request_template(),
                card: Some(model.into_card()),
            })
        }
        EngineConfig::StaticCore {
            engine: inner_engine,
            model,
            ..
        } => {
            let pipeline = build_pipeline::<
                NvCreateChatCompletionRequest,
                NvCreateChatCompletionStreamResponse,
            >(model.card(), inner_engine)
            .await?;

            let service_name = model.service_name().to_string();
            tracing::debug!("Model: {service_name} with Dynamo pre-processing");
            Ok(PreparedEngine {
                service_name,
                engine: pipeline,
                inspect_template: true,
                request_template: model.request_template(),
                card: Some(model.into_card()),
            })
        }
    }
}

pub async fn build_pipeline<Req, Resp>(
    card: &ModelDeploymentCard,
    engine: ExecutionContext,
) -> anyhow::Result<Arc<ServiceFrontend<SingleIn<Req>, ManyOut<Annotated<Resp>>>>>
where
    Req: Data,
    Resp: Data,
    OpenAIPreprocessor: Operator<
        Context<Req>,
        Pin<Box<dyn AsyncEngineStream<Annotated<Resp>>>>,
        Context<PreprocessedRequest>,
        Pin<Box<dyn AsyncEngineStream<Annotated<BackendOutput>>>>,
    >,
{
    let frontend = ServiceFrontend::<SingleIn<Req>, ManyOut<Annotated<Resp>>>::new();
    let preprocessor = OpenAIPreprocessor::new((*card).clone())
        .await?
        .into_operator();
    let backend = Backend::from_mdc((*card).clone()).await?.into_operator();
    let engine = ServiceBackend::from_engine(engine);

    Ok(frontend
        .link(preprocessor.forward_edge())?
        .link(backend.forward_edge())?
        .link(engine)?
        .link(backend.backward_edge())?
        .link(preprocessor.backward_edge())?
        .link(frontend)?)
}

pub async fn build_routed_pipeline<Req, Resp>(
    card: &ModelDeploymentCard,
    client: &Client,
    router_mode: RouterMode,
    chooser: Option<Arc<KvRouter>>,
) -> anyhow::Result<ServiceEngine<SingleIn<Req>, ManyOut<Annotated<Resp>>>>
where
    Req: Data,
    Resp: Data,
    OpenAIPreprocessor: Operator<
        Context<Req>,
        Pin<Box<dyn AsyncEngineStream<Annotated<Resp>>>>,
        Context<PreprocessedRequest>,
        Pin<Box<dyn AsyncEngineStream<Annotated<BackendOutput>>>>,
    >,
{
    let frontend = SegmentSource::<SingleIn<Req>, ManyOut<Annotated<Resp>>>::new();
    let preprocessor = OpenAIPreprocessor::new(card.clone()).await?.into_operator();
    let backend = Backend::from_mdc(card.clone()).await?.into_operator();
    let migration = Migration::from_mdc(card.clone()).await?.into_operator();
    let router = PushRouter::<PreprocessedRequest, Annotated<LLMEngineOutput>>::from_client(
        client.clone(),
        router_mode,
    )
    .await?;
    let service_backend = match router_mode {
        RouterMode::Random | RouterMode::RoundRobin | RouterMode::Direct(_) => {
            ServiceBackend::from_engine(Arc::new(router))
        }
        RouterMode::KV => {
            let Some(chooser) = chooser else {
                anyhow::bail!("RouterMode::KV requires KVRouter to not be null");
            };
            let kv_push_router = KvPushRouter::new(router, chooser);
            ServiceBackend::from_engine(Arc::new(kv_push_router))
        }
    };

    let engine = frontend
        .link(preprocessor.forward_edge())?
        .link(backend.forward_edge())?
        .link(migration.forward_edge())?
        .link(service_backend)?
        .link(migration.backward_edge())?
        .link(backend.backward_edge())?
        .link(preprocessor.backward_edge())?
        .link(frontend)?;
    Ok(engine)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::openai::{
        chat_completions::{NvCreateChatCompletionRequest, NvCreateChatCompletionStreamResponse},
        completions::{NvCreateCompletionRequest, NvCreateCompletionResponse},
    };

    const HF_PATH: &str = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/data/sample-models/mock-llama-3.1-8b-instruct"
    );

    #[tokio::test]
    async fn test_build_chat_completions_pipeline_core_engine_succeeds() -> anyhow::Result<()> {
        // Create test model card
        let card = ModelDeploymentCard::load(HF_PATH).await?;
        let engine = crate::engines::make_engine_core();

        // Build pipeline for chat completions
        let pipeline = build_pipeline::<
            NvCreateChatCompletionRequest,
            NvCreateChatCompletionStreamResponse,
        >(&card, engine)
        .await?;

        // Verify pipeline was created
        assert!(Arc::strong_count(&pipeline) >= 1);

        Ok(())
    }

    #[tokio::test]
    async fn test_build_completions_pipeline_core_engine_succeeds() -> anyhow::Result<()> {
        // Create test model card
        let card = ModelDeploymentCard::load(HF_PATH).await?;
        let engine = crate::engines::make_engine_core();

        // Build pipeline for completions
        let pipeline =
            build_pipeline::<NvCreateCompletionRequest, NvCreateCompletionResponse>(&card, engine)
                .await?;

        // Verify pipeline was created
        assert!(Arc::strong_count(&pipeline) >= 1);

        Ok(())
    }
}
