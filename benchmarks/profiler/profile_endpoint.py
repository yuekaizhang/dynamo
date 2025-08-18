# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import argparse
import logging
import os

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

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="profile a given endpoint's performance for prefill or decode"
    )
    parser.add_argument(
        "--mode",
        type=str,
        required=True,
        choices=["prefill", "decode"],
        help="mode to profile",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        required=True,
        help="model name",
    )
    parser.add_argument(
        "--url",
        type=str,
        required=True,
        help="base url of the endpoint",
    )
    parser.add_argument(
        "--num_gpus",
        type=int,
        required=True,
        help="number of gpus",
    )
    parser.add_argument(
        "--max_kv_tokens",
        type=int,
        required=False,
        default=0,
        help="max kv tokens of the endpoint (only used for decode)",
    )
    parser.add_argument(
        "--work_dir",
        type=str,
        default="endpoint_profiling_results/",
        help="work directory to save the results",
    )
    parser.add_argument(
        "--max_context_length",
        type=int,
        default=16384,
        help="max context length of the endpoint",
    )
    parser.add_argument(
        "--interpolation_granularity",
        type=int,
        default=8,
        help="interpolation granularity for the results",
    )
    args = parser.parse_args()

    os.makedirs(args.work_dir, exist_ok=True)
    if args.mode == "prefill":
        profile_prefill(
            args.work_dir,
            args.model_name,
            args.url,
            args.num_gpus,
            args.max_context_length,
            args.interpolation_granularity,
        )
    elif args.mode == "decode":
        assert args.max_kv_tokens > 0, "max_kv_tokens must be provided for decode"
        profile_decode(
            args.work_dir,
            args.model_name,
            args.url,
            args.num_gpus,
            args.max_kv_tokens,
            args.max_context_length,
            args.interpolation_granularity,
        )
    else:
        raise ValueError(f"Invalid mode: {args.mode}")
