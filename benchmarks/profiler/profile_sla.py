# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import asyncio
import logging
import math
import os

import numpy as np
import yaml
from utils.config import CONFIG_MODIFIERS, WORKER_COMPONENT_NAMES
from utils.defaults import DECODE_NUM_REQUESTS_RANGE
from utils.dynamo_deployment import (
    DynamoDeploymentClient,
    cleanup_remaining_deployments,
)
from utils.genai_perf import benchmark_decode, benchmark_prefill
from utils.plot import plot_decode_performance, plot_prefill_performance
from utils.profile_cache import (
    check_decode_results_exist,
    check_prefill_results_exist,
    load_existing_decode_results,
    load_existing_prefill_results,
)
from utils.profile_decode import profile_decode
from utils.profile_prefill import profile_prefill

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
formatter = logging.Formatter(
    "%(asctime)s - %(name)s - %(levelname)s - %(message)s", "%Y-%m-%d %H:%M:%S"
)
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)


async def run_profile(args):
    # List to track all created deployment clients for cleanup in case of failure
    deployment_clients = []

    try:
        config_modifier = CONFIG_MODIFIERS[args.backend]

        if args.example_dir is None:
            logger.info(
                "Example directory not provided, inferring from config file location..."
            )
            try:
                args.example_dir = os.path.dirname(os.path.dirname(args.config))
            except Exception:
                logger.error(
                    "Failed to infer example directory, please provide explicitly using --example-dir <path-to-example-dir>"
                )
                exit(1)

        with open(args.config, "r") as f:
            config = yaml.safe_load(f)

        profile_tp_size = [
            2**i
            for i in range(int(math.log2(args.max_num_gpus_per_engine)) + 1)
            if args.min_num_gpus_per_engine <= 2**i <= args.max_num_gpus_per_engine
        ]
        logger.info(f"Profiling TP sizes: {profile_tp_size}")

        os.makedirs(args.output_dir, exist_ok=True)

        model_name = config_modifier.get_model_name(config)

        # Log skip behavior
        if args.force_rerun:
            logger.info(
                "Force rerun enabled - will re-run all tests even if results exist"
            )
        elif args.skip_existing_results:
            logger.info(
                "Skip existing results enabled - will skip TP sizes with existing results"
            )
        else:
            logger.info("Skip existing results disabled - will re-run all tests")

        # first profile prefill
        prefill_tp_size = []
        prefill_ttft = []
        prefill_thpt_per_gpu = []
        logger.info("Profiling prefill...")
        prefill_config = config_modifier.convert_config(config, "prefill")
        frontend_port = config_modifier.get_port(config)
        for tp_size in profile_tp_size:
            logger.info(f"Profiling prefill with TP size {tp_size}...")

            # Check if results already exist for this TP size
            if (
                args.skip_existing_results
                and not args.force_rerun
                and check_prefill_results_exist(args.output_dir, tp_size, args.isl)
            ):
                logger.info(f"Skipping prefill TP{tp_size} - results already exist")
                ttft, thpt_per_gpu = load_existing_prefill_results(
                    args.output_dir, tp_size, args.isl
                )
                if ttft is not None and thpt_per_gpu is not None:
                    prefill_tp_size.append(tp_size)
                    prefill_ttft.append(ttft)
                    prefill_thpt_per_gpu.append(thpt_per_gpu)
                    logger.info(
                        f"Loaded existing prefill results: TP{tp_size} TTFT={ttft:.2f}ms, throughput={thpt_per_gpu:.2f} tokens/s/GPU"
                    )
                continue

            prefill_config = config_modifier.set_config_tp_size(prefill_config, tp_size)
            logger.info(f"Dynamo config: {prefill_config}")

            work_dir = f"{args.output_dir}/prefill_tp{tp_size}"
            os.makedirs(work_dir, exist_ok=True)

            prefill_config_fn = f"{work_dir}/config.yaml"
            with open(prefill_config_fn, "w") as f:
                yaml.dump(prefill_config, f)

            client = DynamoDeploymentClient(
                namespace=args.namespace,
                base_log_dir=work_dir,
                model_name=model_name,
                service_name=args.service_name,
                frontend_port=frontend_port,
                deployment_name=prefill_config["metadata"]["name"],
            )
            logger.info(f"Created client with service_name: {client.service_name}")
            deployment_clients.append(client)  # Track for cleanup
            await client.create_deployment(prefill_config_fn)
            logger.info("Waiting for deployment to be ready...")
            await client.wait_for_deployment_ready()
            logger.info("Deployment is ready")

            logger.info("Getting deployment logs...")
            await client.get_deployment_logs()
            logger.info(
                f"Logs have been saved to {client.base_log_dir / client.deployment_name}"
            )

            # run genai-perf
            base_url = client.get_service_url()
            genai_perf_artifact_dir = f"{work_dir}/gap_isl{args.isl}"
            gap_result = benchmark_prefill(
                args.isl, genai_perf_artifact_dir, model_name, base_url=base_url
            )
            if gap_result is not None:
                ttft = gap_result["time_to_first_token"]["avg"]
                prefill_tp_size.append(tp_size)
                prefill_ttft.append(ttft)
                prefill_thpt_per_gpu.append(args.isl / ttft / tp_size * 1000)

            print("Cleaning up deployment...")
            await client.delete_deployment()
            deployment_clients.remove(client)
            print("Deployment deleted")

        # Plot the results as a 2D scatter plot
        if prefill_tp_size and prefill_ttft and prefill_thpt_per_gpu:
            plot_prefill_performance(
                prefill_tp_size,
                prefill_ttft,
                prefill_thpt_per_gpu,
                args.ttft,
                args.output_dir,
            )

        # then profile decode
        decode_tp_size = []
        decode_itl = []
        decode_thpt_per_gpu = []
        decode_concurrency = []
        decode_kv_cache_size = []
        decode_results = []  # Store partial results for plotting later
        logger.info("Profiling decode...")
        decode_config = config_modifier.convert_config(config, "decode")
        for tp_size in profile_tp_size:
            logger.info(f"Profiling decode with TP size {tp_size}...")

            # Check if results already exist for this TP size
            if (
                args.skip_existing_results
                and not args.force_rerun
                and check_decode_results_exist(
                    args.output_dir, tp_size, args.isl, args.osl
                )
            ):
                logger.info(f"Skipping decode TP{tp_size} - results already exist")
                existing_results = load_existing_decode_results(
                    args.output_dir, tp_size, args.isl, args.osl
                )
                if existing_results:
                    # Add existing results to our arrays
                    engine_decode_itl = []
                    engine_decode_thpt_per_gpu = []
                    for itl, thpt_per_gpu, concurrency in existing_results:
                        decode_tp_size.append(tp_size)
                        decode_itl.append(itl)
                        decode_thpt_per_gpu.append(thpt_per_gpu)
                        decode_concurrency.append(concurrency)
                        # We need to get kv_cache_size from existing logs or estimate it
                        estimated_kv_cache = max(
                            100000, concurrency * (args.isl + args.osl) * 2
                        )  # Conservative estimate
                        decode_kv_cache_size.append(estimated_kv_cache)
                        engine_decode_itl.append(itl)
                        engine_decode_thpt_per_gpu.append(thpt_per_gpu)

                    # Store results for plotting
                    decode_results.append(
                        (tp_size, engine_decode_itl, engine_decode_thpt_per_gpu)
                    )
                    logger.info(
                        f"Loaded {len(existing_results)} existing decode results for TP{tp_size}"
                    )
                continue

            decode_config = config_modifier.set_config_tp_size(decode_config, tp_size)
            logger.info(f"Dynamo config: {decode_config}")

            work_dir = f"{args.output_dir}/decode_tp{tp_size}"
            os.makedirs(work_dir, exist_ok=True)

            decode_config_fn = f"{work_dir}/config.yaml"
            with open(decode_config_fn, "w") as f:
                yaml.dump(decode_config, f)

            client = DynamoDeploymentClient(
                namespace=args.namespace,
                base_log_dir=work_dir,
                model_name=model_name,
                service_name=args.service_name,
                frontend_port=frontend_port,
                deployment_name=decode_config["metadata"]["name"],
            )
            deployment_clients.append(client)  # Track for cleanup
            await client.create_deployment(decode_config_fn)
            logger.info("Waiting for deployment to be ready...")
            await client.wait_for_deployment_ready()
            logger.info("Deployment is ready")

            logger.info("Getting deployment logs...")
            await client.get_deployment_logs()
            logger.info(
                f"Logs have been saved to {client.base_log_dir / client.deployment_name}"
            )

            max_kv_tokens = config_modifier.get_kv_cache_size_from_dynamo_log(
                f"{work_dir}/{client.deployment_name}/{WORKER_COMPONENT_NAMES[args.backend].decode_worker_k8s_name.lower()}/0.log"
            )
            max_concurrency = max_kv_tokens // (args.isl + args.osl)
            sweep_num_request = [
                num for num in DECODE_NUM_REQUESTS_RANGE if num < max_concurrency
            ]
            logger.info(
                f"Sweeping num_request range based on maximum number of kv tokens: {sweep_num_request}"
            )

            engine_decode_itl = []
            engine_decode_thpt_per_gpu = []
            base_url = client.get_service_url()
            for num_request in sweep_num_request:
                genai_perf_artifact_dir = f"{work_dir}/gap_request{num_request}_isl{args.isl}_osl{args.osl}_n{num_request}"
                gap_result = benchmark_decode(
                    args.isl,
                    args.osl,
                    num_request,
                    genai_perf_artifact_dir,
                    model_name,
                    base_url=base_url,
                )
                if gap_result is not None:
                    itl = gap_result["inter_token_latency"]["avg"]
                    thpt_per_gpu = (
                        gap_result["output_token_throughput"]["avg"] / tp_size
                    )
                    engine_decode_itl.append(itl)
                    engine_decode_thpt_per_gpu.append(thpt_per_gpu)
                    decode_tp_size.append(tp_size)
                    decode_itl.append(itl)
                    decode_thpt_per_gpu.append(thpt_per_gpu)
                    decode_concurrency.append(num_request)
                    decode_kv_cache_size.append(max_kv_tokens)

            print("Cleaning up deployment...")
            await client.delete_deployment()
            deployment_clients.remove(client)
            print("Deployment deleted")

            # Store partial results for plotting later
            decode_results.append(
                (tp_size, engine_decode_itl, engine_decode_thpt_per_gpu)
            )

        # Plot all decode results after profiling is complete
        if decode_results:
            plot_decode_performance(decode_results, args.itl, args.output_dir)

        logger.info("Analyzing results and generate recommendations...")
        # select best tp size for prefill
        if min(prefill_ttft) > args.ttft:
            logger.info(
                "No TP size satisfies the TTFT requirement, please try a smaller model or a more powerful GPU SKU"
            )
            selected_prefill_idx = int(np.argmin(np.array(prefill_ttft)))
        else:
            valid_indices = [
                i for i, ttft in enumerate(prefill_ttft) if ttft <= args.ttft
            ]
            # Among valid TP sizes, select the one with highest throughput per GPU
            valid_thpts = [prefill_thpt_per_gpu[i] for i in valid_indices]
            max_thpt_idx = valid_indices[int(np.argmax(valid_thpts))]
            selected_prefill_idx = max_thpt_idx
        logger.info(
            f"Suggested prefill TP:{prefill_tp_size[selected_prefill_idx]} (TTFT {prefill_ttft[selected_prefill_idx]:.2f} ms, throughput {prefill_thpt_per_gpu[selected_prefill_idx]:.2f} tokens/s/GPU)"
        )

        # scale up if estimated TTFT is 120% of target TTFT
        prefill_queue_size_upper_bound = max(
            0.1, args.ttft * 1.2 / prefill_ttft[selected_prefill_idx] - 1
        )
        # scale down if estimated TTFT is 80% of target TTFT
        prefill_queue_size_lower_bound = max(
            0.1, args.ttft * 0.8 / prefill_ttft[selected_prefill_idx] - 1
        )
        logger.info(
            f"Suggested planner upper/lower bound for prefill queue size: {prefill_queue_size_upper_bound:.2f}/{prefill_queue_size_lower_bound:.2f}"
        )

        # select best tp size for decode
        if min(decode_itl) > args.itl:
            logger.info(
                "No TP size satisfies the ITL requirement, please try a smaller model or a more powerful GPU SKU"
            )
            selected_decode_idx = int(np.argmin(np.array(decode_itl)))
        else:
            valid_indices = [i for i, itl in enumerate(decode_itl) if itl <= args.itl]
            # Among valid TP sizes, select the one with highest throughput per GPU
            valid_thpts = [decode_thpt_per_gpu[i] for i in valid_indices]
            max_thpt_idx = valid_indices[int(np.argmax(valid_thpts))]
            selected_decode_idx = max_thpt_idx
        logger.info(
            f"Suggested decode TP:{decode_tp_size[selected_decode_idx]} (ITL {decode_itl[selected_decode_idx]:.2f} ms, throughput {decode_thpt_per_gpu[selected_decode_idx]:.2f} tokens/s/GPU)"
        )

        # calculate kv cache utlization for the selected TP and concurrency
        selected_decode_kv_cache_utilization = (
            decode_concurrency[selected_decode_idx]
            * (args.isl + args.osl / 2)
            / decode_kv_cache_size[selected_decode_idx]
        )
        # set a +- 20% range for the kv cache utilization
        logger.info(
            f"Suggested planner upper/lower bound for decode kv cache utilization: {min(1, selected_decode_kv_cache_utilization + 0.2):.2f}/{max(0.1, selected_decode_kv_cache_utilization - 0.2):.2f}"
        )

        # interpolate ISL - TTFT with best prefill TP
        best_prefill_tp = prefill_tp_size[selected_prefill_idx]
        logger.info(
            f"Profiling prefill under best TP {best_prefill_tp} with different ISL..."
        )
        prefill_config = config_modifier.convert_config(config, "prefill")
        prefill_config = config_modifier.set_config_tp_size(
            prefill_config, best_prefill_tp
        )
        logger.info(f"Dynamo config: {prefill_config}")

        work_dir = f"{args.output_dir}/selected_prefill_interpolation"
        os.makedirs(work_dir, exist_ok=True)

        prefill_config_fn = f"{work_dir}/config.yaml"
        with open(prefill_config_fn, "w") as f:
            yaml.dump(prefill_config, f)

        client = DynamoDeploymentClient(
            namespace=args.namespace,
            base_log_dir=work_dir,
            model_name=model_name,
            service_name=args.service_name,
            frontend_port=frontend_port,
            deployment_name=prefill_config["metadata"]["name"],
        )
        deployment_clients.append(client)  # Track for cleanup
        await client.create_deployment(prefill_config_fn)
        logger.info("Waiting for deployment to be ready...")
        try:
            await client.wait_for_deployment_ready()
            logger.info("Deployment is ready")
            skip_profile = False
        except TimeoutError:
            logger.error(
                "Deployment failed to become ready within timeout, skipping profiling"
            )
            skip_profile = True

        if not skip_profile:
            logger.info("Getting deployment logs...")
            await client.get_deployment_logs()
            logger.info(
                f"Logs have been saved to {client.base_log_dir / client.deployment_name}"
            )

        base_url = client.get_service_url()

        profile_prefill(
            work_dir,
            model_name,
            base_url,
            best_prefill_tp,
            args.max_context_length,
            args.prefill_interpolation_granularity,
        )

        print("Cleaning up deployment...")
        await client.delete_deployment()
        deployment_clients.remove(client)
        print("Deployment deleted")

        # interpolate ITL - Active_KV_Cache - Decode_Context_Length with best decode TP
        best_decode_tp = decode_tp_size[selected_decode_idx]
        logger.info(f"Profiling decode with TP size {best_decode_tp}...")
        decode_config = config_modifier.set_config_tp_size(
            decode_config, best_decode_tp
        )
        logger.info(f"Dynamo config: {decode_config}")

        work_dir = f"{args.output_dir}/selected_decode_interpolation"
        os.makedirs(work_dir, exist_ok=True)

        decode_config_fn = f"{work_dir}/config.yaml"
        with open(decode_config_fn, "w") as f:
            yaml.dump(decode_config, f)

        client = DynamoDeploymentClient(
            namespace=args.namespace,
            base_log_dir=work_dir,
            model_name=model_name,
            service_name=args.service_name,
            frontend_port=frontend_port,
            deployment_name=decode_config["metadata"]["name"],
        )
        deployment_clients.append(client)  # Track for cleanup
        await client.create_deployment(decode_config_fn)
        logger.info("Waiting for deployment to be ready...")
        await client.wait_for_deployment_ready()
        logger.info("Deployment is ready")

        logger.info("Getting deployment logs...")
        await client.get_deployment_logs()
        logger.info(
            f"Logs have been saved to {client.base_log_dir / client.deployment_name}"
        )

        max_kv_tokens = config_modifier.get_kv_cache_size_from_dynamo_log(
            f"{work_dir}/{client.deployment_name}/{WORKER_COMPONENT_NAMES[args.backend].decode_worker_k8s_name.lower()}/0.log"
        )

        base_url = client.get_service_url()

        profile_decode(
            work_dir,
            model_name,
            base_url,
            best_decode_tp,
            max_kv_tokens,
            args.max_context_length,
            args.decode_interpolation_granularity,
        )

        print("Cleaning up deployment...")
        await client.delete_deployment()
        deployment_clients.remove(client)
        print("Deployment deleted")

    except Exception as e:
        logger.error(f"Profile job failed with error: {e}")
        raise
    finally:
        # Always clean up any remaining deployments, even if the job failed
        logger.info("Performing final cleanup of any remaining deployments...")
        await cleanup_remaining_deployments(deployment_clients, args.namespace)
        logger.info("Final cleanup completed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Profile the TTFT and ITL of the Prefill and Decode engine with different parallelization mapping. When profiling prefill we mock/fix decode,when profiling decode we mock/fix prefill."
    )
    parser.add_argument(
        "--namespace",
        type=str,
        default="dynamo-sla-profiler",
        help="Kubernetes namespace to deploy the DynamoGraphDeployment",
    )
    parser.add_argument(
        "--backend",
        type=str,
        default="vllm",
        choices=["vllm", "sglang"],
        help="backend type, currently support [vllm, sglang]",
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the DynamoGraphDeployment config file",
    )
    parser.add_argument(
        "--example-dir",
        type=str,
        default=None,
        help="path to the example directory, if not provided, will try to infer from config file location",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="profiling_results",
        help="Path to the output results directory",
    )
    parser.add_argument(
        "--min-num-gpus-per-engine",
        type=int,
        default=1,
        help="minimum number of GPUs per engine",
    )
    parser.add_argument(
        "--max-num-gpus-per-engine",
        type=int,
        default=8,
        help="maximum number of GPUs per engine",
    )
    parser.add_argument(
        "--skip-existing-results",
        action="store_true",
        help="Skip TP sizes that already have results in the output directory",
    )
    parser.add_argument(
        "--force-rerun",
        action="store_true",
        help="Force re-running all tests even if results already exist (overrides --skip-existing-results)",
    )
    parser.add_argument(
        "--isl", type=int, default=3000, help="target input sequence length"
    )
    parser.add_argument(
        "--osl", type=int, default=500, help="target output sequence length"
    )
    parser.add_argument(
        "--ttft", type=int, default=50, help="target Time To First Token in ms"
    )
    parser.add_argument(
        "--itl", type=int, default=10, help="target Inter Token Latency in ms"
    )
    # below are arguments used for interpolating TTFT and ITL under different ISL/OSL
    parser.add_argument(
        "--max-context-length",
        type=int,
        default=16384,
        help="maximum context length supported by the served model",
    )
    parser.add_argument(
        "--prefill-interpolation-granularity",
        type=int,
        default=16,
        help="how many samples to benchmark to interpolate TTFT under different ISL",
    )
    parser.add_argument(
        "--decode-interpolation-granularity",
        type=int,
        default=6,
        help="how many samples to benchmark to interpolate ITL under different active kv cache size and decode context length",
    )
    parser.add_argument(
        "--service-name",
        type=str,
        default="",
        help="Service name for port forwarding (default: {deployment_name}-frontend)",
    )
    args = parser.parse_args()

    asyncio.run(run_profile(args))
