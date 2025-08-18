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

from typing import Optional

from vllm.config import VllmConfig
from vllm.v1.metrics.loggers import StatLoggerBase
from vllm.v1.metrics.stats import IterationStats, SchedulerStats

from dynamo.llm import (
    ForwardPassMetrics,
    KvStats,
    SpecDecodeStats,
    WorkerMetricsPublisher,
    WorkerStats,
)
from dynamo.runtime import Component


class NullStatLogger(StatLoggerBase):
    def __init__(self):
        pass

    def record(
        self,
        scheduler_stats: Optional[SchedulerStats],
        iteration_stats: Optional[IterationStats],
        engine_idx: int = 0,
    ):
        pass

    def log_engine_initialized(self):
        pass


class DynamoStatLoggerPublisher(StatLoggerBase):
    """Stat logger publisher. Wrapper for the WorkerMetricsPublisher to match the StatLoggerBase interface."""

    def __init__(self, component: Component, dp_rank: int) -> None:
        self.inner = WorkerMetricsPublisher()
        self.inner.create_endpoint(component)
        self.dp_rank = dp_rank
        self.num_gpu_block = 1
        self.request_total_slots = 1

    # TODO: Remove this and pass as metadata through etcd
    def set_num_gpu_block(self, num_blocks):
        self.num_gpu_block = num_blocks

    # TODO: Remove this and pass as metadata through etcd
    def set_num_request_total_slots(self, request_total_slots):
        self.request_total_slots = request_total_slots

    def record(
        self,
        scheduler_stats: SchedulerStats,
        iteration_stats: Optional[IterationStats],
        engine_idx: int = 0,
    ):
        # request_total_slots and kv_total_blocks are properties of model + gpu
        # we should only publish them once, not every metric update
        # they should be part of some runtime metadata tied to MDC or put in etcd ?
        hit_rate = 0
        if scheduler_stats.prefix_cache_stats.queries > 0:
            hit_rate = (
                scheduler_stats.prefix_cache_stats.hits
                / scheduler_stats.prefix_cache_stats.queries
            )

        worker_stats = WorkerStats(
            request_active_slots=scheduler_stats.num_running_reqs,
            request_total_slots=self.request_total_slots,
            num_requests_waiting=scheduler_stats.num_waiting_reqs,
            data_parallel_rank=self.dp_rank,
        )

        kv_stats = KvStats(
            kv_active_blocks=int(self.num_gpu_block * scheduler_stats.kv_cache_usage),
            kv_total_blocks=self.num_gpu_block,
            gpu_cache_usage_perc=scheduler_stats.kv_cache_usage,
            gpu_prefix_cache_hit_rate=hit_rate,  # TODO: This is a point in time update, not cumulative. Will be problematic on router side if we try to use it.
        )

        spec_dec_stats = scheduler_stats.spec_decoding_stats
        if spec_dec_stats:
            spec_dec_stats = SpecDecodeStats(
                num_spec_tokens=spec_dec_stats.num_spec_tokens,
                num_drafts=spec_dec_stats.num_drafts,
                num_draft_tokens=spec_dec_stats.num_draft_tokens,
                num_accepted_tokens=spec_dec_stats.num_accepted_tokens,
                num_accepted_tokens_per_pos=spec_dec_stats.num_accepted_tokens_per_pos,
            )

        metrics = ForwardPassMetrics(
            worker_stats=worker_stats,
            kv_stats=kv_stats,
            spec_decode_stats=spec_dec_stats,
        )

        self.inner.publish(metrics)

    def init_publish(self):
        worker_stats = WorkerStats(
            request_active_slots=0,
            request_total_slots=self.request_total_slots,
            num_requests_waiting=0,
            data_parallel_rank=self.dp_rank,
        )

        kv_stats = KvStats(
            kv_active_blocks=0,
            kv_total_blocks=self.num_gpu_block,
            gpu_cache_usage_perc=0,
            gpu_prefix_cache_hit_rate=0,
        )

        metrics = ForwardPassMetrics(
            worker_stats=worker_stats,
            kv_stats=kv_stats,
            spec_decode_stats=None,
        )

        self.inner.publish(metrics)

    def log_engine_initialized(self) -> None:
        pass


class StatLoggerFactory:
    """Factory for creating stat logger publishers. Required by vLLM."""

    def __init__(self, component: Component, dp_rank: int = 0) -> None:
        self.component = component
        self.created_logger: Optional[DynamoStatLoggerPublisher] = None
        self.dp_rank = dp_rank

    def create_stat_logger(self, dp_rank: int) -> StatLoggerBase:
        if self.dp_rank != dp_rank:
            return NullStatLogger()
        logger = DynamoStatLoggerPublisher(self.component, dp_rank)
        self.created_logger = logger

        return logger

    def __call__(self, vllm_config: VllmConfig, dp_rank: int) -> StatLoggerBase:
        return self.create_stat_logger(dp_rank=dp_rank)

    # TODO Remove once we publish metadata to etcd
    def set_num_gpu_blocks_all(self, num_blocks):
        if self.created_logger:
            self.created_logger.set_num_gpu_block(num_blocks)

    def set_request_total_slots_all(self, request_total_slots):
        if self.created_logger:
            self.created_logger.set_num_request_total_slots(request_total_slots)

    def init_publish(self):
        if self.created_logger:
            self.created_logger.init_publish()
