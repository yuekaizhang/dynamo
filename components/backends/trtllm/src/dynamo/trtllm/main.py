# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import logging
import signal
import sys

import uvloop
from tensorrt_llm import SamplingParams
from tensorrt_llm.llmapi import (
    BuildConfig,
    CapacitySchedulerPolicy,
    DynamicBatchConfig,
    KvCacheConfig,
    SchedulerConfig,
)
from tensorrt_llm.llmapi.llm_utils import update_llm_args_with_extra_options
from tensorrt_llm.llmapi.tokenizer import tokenizer_factory
from torch.cuda import device_count
from transformers import AutoConfig

from dynamo.llm import ModelRuntimeConfig, ModelType, register_llm
from dynamo.runtime import DistributedRuntime, dynamo_worker
from dynamo.runtime.logging import configure_dynamo_logging
from dynamo.trtllm.engine import TensorRTLLMEngine, get_llm_engine
from dynamo.trtllm.multimodal_processor import MultimodalRequestProcessor
from dynamo.trtllm.publisher import get_publisher
from dynamo.trtllm.request_handlers.handlers import (
    RequestHandlerConfig,
    RequestHandlerFactory,
)
from dynamo.trtllm.utils.trtllm_utils import (
    Config,
    cmd_line_args,
    is_first_worker,
    parse_endpoint,
)

# Default buffer size for kv cache events.
DEFAULT_KV_EVENT_BUFFER_MAX_SIZE = 1024

configure_dynamo_logging()


async def graceful_shutdown(runtime):
    logging.info("Received shutdown signal, shutting down DistributedRuntime")
    runtime.shutdown()
    logging.info("DistributedRuntime shutdown complete")


async def get_engine_runtime_config(
    engine: TensorRTLLMEngine, config: Config
) -> ModelRuntimeConfig:
    """Retrieve runtime configuration from TensorRT-LLM engine."""
    runtime_config = ModelRuntimeConfig()

    try:
        # Extract total_kv_blocks from engine stats
        stats = engine.llm.get_stats_async(timeout=5)
        stat = await anext(stats)
        runtime_config.total_kv_blocks = stat["kvCacheStats"]["maxNumBlocks"]
        logging.info(
            f"Set runtime config total_kv_blocks: {runtime_config.total_kv_blocks}"
        )

        # Extract max number of sequences
        runtime_config.max_num_seqs = config.max_batch_size
        logging.info(f"Set runtime config max_num_seqs: {runtime_config.max_num_seqs}")

        # Get max_num_batched_tokens from config
        runtime_config.max_num_batched_tokens = config.max_num_tokens
        logging.info(
            f"Set runtime config max_num_batched_tokens: {runtime_config.max_num_batched_tokens}"
        )

        return runtime_config

    except Exception as e:
        logging.error(f"Failed to get runtime config from TensorRT-LLM engine: {e}")
        # Return config with default/None values if retrieval fails
        return runtime_config


@dynamo_worker(static=False)
async def worker(runtime: DistributedRuntime):
    # Set up signal handler for graceful shutdown
    loop = asyncio.get_running_loop()

    def signal_handler():
        # Schedule the shutdown coroutine instead of calling it directly
        asyncio.create_task(graceful_shutdown(runtime))

    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, signal_handler)

    logging.info("Signal handlers set up for graceful shutdown")

    config = cmd_line_args()
    await init(runtime, config)


async def init(runtime: DistributedRuntime, config: Config):
    """
    Instantiate and serve
    """
    logging.info(f"Initializing the worker with config: {config}")

    next_client = None
    if config.next_endpoint:
        logging.info(
            f"Initializing next worker client for endpoint: {config.next_endpoint}"
        )
        parsed_namespace, parsed_component_name, parsed_endpoint_name = parse_endpoint(
            config.next_endpoint
        )
        next_client = (
            await runtime.namespace(parsed_namespace)
            .component(parsed_component_name)
            .endpoint(parsed_endpoint_name)
            .client()
        )

    component = runtime.namespace(config.namespace).component(config.component)
    await component.create_service()

    # Convert model path to Path object if it's a local path, otherwise keep as string
    model_path = str(config.model_path)

    if config.gpus_per_node is None:
        gpus_per_node = device_count()
        if gpus_per_node == 0:
            raise ValueError("No GPU devices found on the node")
    else:
        gpus_per_node = config.gpus_per_node

    build_config = BuildConfig(
        max_batch_size=config.max_batch_size,
        max_num_tokens=config.max_num_tokens,
        max_beam_width=config.max_beam_width,
        max_seq_len=config.max_seq_len,
    )

    kv_cache_config = KvCacheConfig(
        free_gpu_memory_fraction=config.free_gpu_memory_fraction
    )

    dynamic_batch_config = DynamicBatchConfig(
        enable_batch_size_tuning=True,
        enable_max_num_tokens_tuning=False,
        dynamic_batch_moving_average_window=128,
    )
    scheduler_config = SchedulerConfig(
        capacity_scheduler_policy=CapacitySchedulerPolicy.GUARANTEED_NO_EVICT,
        dynamic_batch_config=dynamic_batch_config,
    )
    modality = getattr(config, "modality", None) or "text"
    arg_map = {
        "model": model_path,
        "scheduler_config": scheduler_config,
        "tensor_parallel_size": config.tensor_parallel_size,
        "pipeline_parallel_size": config.pipeline_parallel_size,
        "moe_expert_parallel_size": config.expert_parallel_size,
        "backend": "pytorch",
        "skip_tokenizer_init": True,
        "build_config": build_config,
        "kv_cache_config": kv_cache_config,
        "gpus_per_node": gpus_per_node,
        "max_num_tokens": config.max_num_tokens,
        "max_seq_len": config.max_seq_len,
        "max_beam_width": config.max_beam_width,
        "max_batch_size": config.max_batch_size,
    }

    if config.extra_engine_args != "":
        # TODO: Support extra engine args from json file as well.
        arg_map = update_llm_args_with_extra_options(arg_map, config.extra_engine_args)
    if config.publish_events_and_metrics:
        # 'event_buffer_max_size' is required to enable TRTLLM to publish kv cache events.
        kv_cache_config = None
        if "kv_cache_config" not in arg_map:
            kv_cache_config = {}
            kv_cache_config["event_buffer_max_size"] = DEFAULT_KV_EVENT_BUFFER_MAX_SIZE
        else:
            kv_cache_config = arg_map["kv_cache_config"]
            if "event_buffer_max_size" not in kv_cache_config:
                kv_cache_config[
                    "event_buffer_max_size"
                ] = DEFAULT_KV_EVENT_BUFFER_MAX_SIZE
        arg_map["kv_cache_config"] = kv_cache_config

        # Only pytorch backend is supported for now to publish events and metrics.
        if "backend" not in arg_map:
            arg_map["backend"] = "pytorch"
        elif arg_map["backend"] != "pytorch":
            logging.error(
                "Only pytorch backend is supported for now to publish events and metrics."
            )
            sys.exit(1)

    logging.info(f"TensorRT-LLM engine args: {arg_map}")
    engine_args = arg_map

    # Populate default sampling params from the model
    tokenizer = tokenizer_factory(arg_map["model"])
    default_sampling_params = SamplingParams()
    default_sampling_params._setup(tokenizer)
    default_sampling_params.stop = None
    modelType = ModelType.Backend
    multimodal_processor = None

    if modality == "multimodal":
        engine_args["skip_tokenizer_init"] = False
        modelType = ModelType.Chat
        model_config = AutoConfig.from_pretrained(
            config.model_path, trust_remote_code=True
        )
        multimodal_processor = MultimodalRequestProcessor(
            model_type=model_config.model_type,
            model_dir=config.model_path,
            tokenizer=tokenizer,
        )

    else:
        # We already detokenize inside HandlerBase. No need to also do it in TRTLLM.
        default_sampling_params.detokenize = False

    async with get_llm_engine(engine_args) as engine:
        endpoint = component.endpoint(config.endpoint)

        if is_first_worker(config):
            # Register the model with runtime config
            await register_llm(
                modelType,
                endpoint,
                config.model_path,
                config.served_model_name,
                kv_cache_block_size=config.kv_block_size,
                migration_limit=config.migration_limit,
            )
        # publisher will be set later if publishing is enabled.
        handler_config = RequestHandlerConfig(
            component=component,
            engine=engine,
            default_sampling_params=default_sampling_params,
            publisher=None,
            disaggregation_mode=config.disaggregation_mode,
            disaggregation_strategy=config.disaggregation_strategy,
            next_client=next_client,
            multimodal_processor=multimodal_processor,
        )

        if config.publish_events_and_metrics and is_first_worker(config):
            # Initialize and pass in the publisher to the request handler to
            # publish events and metrics.
            kv_listener = runtime.namespace(config.namespace).component(
                config.component
            )
            async with get_publisher(
                component,
                engine,
                kv_listener,
                int(endpoint.lease_id()),
                config.kv_block_size,
            ) as publisher:
                handler_config.publisher = publisher
                handler = RequestHandlerFactory().get_request_handler(handler_config)
                await endpoint.serve_endpoint(handler.generate)
        else:
            handler = RequestHandlerFactory().get_request_handler(handler_config)
            await endpoint.serve_endpoint(handler.generate)


def main():
    uvloop.run(worker())


if __name__ == "__main__":
    main()
