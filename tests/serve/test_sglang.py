# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
import os
import re
from dataclasses import dataclass
from typing import Any, List

import pytest
import requests

from tests.utils.managed_process import ManagedProcess

logger = logging.getLogger(__name__)


def validate_log_patterns(log_file, patterns):
    """Validate log patterns after test completion."""
    if not os.path.exists(log_file):
        raise AssertionError(f"Log file not found: {log_file}")

    with open(log_file, "r", encoding="utf-8", errors="ignore") as f:
        content = f.read()

    compiled = [re.compile(p) for p in patterns]
    missing = []

    for pattern, rx in zip(patterns, compiled):
        if not rx.search(content):
            missing.append(pattern)

    if missing:
        # Include sample of log content for debugging
        sample = content[-1000:] if len(content) > 1000 else content
        raise AssertionError(
            f"Missing expected log patterns: {missing}\n\nLog sample:\n{sample}"
        )

    return True


@dataclass
class SGLangConfig:
    """Configuration for SGLang test scenarios"""

    script_name: str
    marks: List[Any]
    name: str


class SGLangProcess(ManagedProcess):
    """Simple process manager for sglang shell scripts"""

    def __init__(self, script_name, request):
        self.port = 8000
        sglang_dir = os.environ.get(
            "SGLANG_DIR", "/workspace/components/backends/sglang"
        )
        script_path = os.path.join(sglang_dir, "launch", script_name)

        # Verify script exists
        if not os.path.exists(script_path):
            raise FileNotFoundError(f"SGLang script not found: {script_path}")

        # Make script executable and run it
        command = ["bash", script_path]

        # Focus kv-router logs for kv_events run
        env = os.environ.copy()
        if script_name == "agg_router.sh":
            env.setdefault(
                "DYN_LOG",
                "dynamo_llm::kv_router::publisher=trace,dynamo_llm::kv_router::scheduler=info",
            )

        super().__init__(
            command=command,
            env=env,
            timeout=900,
            display_output=True,
            working_dir=sglang_dir,
            health_check_ports=[],  # Disable port health check
            health_check_urls=[
                (f"http://localhost:{self.port}/v1/models", self._check_models_api)
            ],
            delayed_start=60,  # Give SGLang more time to fully start
            terminate_existing=False,
            stragglers=[],  # Don't kill any stragglers automatically
        )

    def _check_models_api(self, response):
        """Check if models API is working and returns models"""
        try:
            if response.status_code != 200:
                return False
            data = response.json()
            return data.get("data") and len(data["data"]) > 0
        except Exception:
            return False


# SGLang test configurations
sglang_configs = {
    "aggregated": SGLangConfig(
        script_name="agg.sh", marks=[pytest.mark.gpu_1], name="aggregated"
    ),
    "disaggregated": SGLangConfig(
        script_name="disagg.sh", marks=[pytest.mark.gpu_2], name="disaggregated"
    ),
    "kv_events": SGLangConfig(
        script_name="agg_router.sh", marks=[pytest.mark.gpu_2], name="kv_events"
    ),
}


@pytest.fixture(
    params=[
        pytest.param("aggregated", marks=[pytest.mark.gpu_1]),
        pytest.param("disaggregated", marks=[pytest.mark.gpu_2]),
        pytest.param("kv_events", marks=[pytest.mark.gpu_2]),
    ]
)
def sglang_config_test(request):
    """Fixture that provides different SGLang test configurations"""
    return sglang_configs[request.param]


@pytest.mark.e2e
@pytest.mark.slow
@pytest.mark.sglang
def test_sglang_deployment(request, runtime_services, sglang_config_test):
    """Test SGLang deployment scenarios"""

    # First check if sglang is available
    try:
        import sglang

        logger.info(f"SGLang version: {sglang.__version__}")
    except ImportError:
        pytest.skip("SGLang not available")

    config = sglang_config_test

    with SGLangProcess(config.script_name, request) as server:
        # Test chat completions
        prompts = [
            "why is roger federer the best tennis player of all time?",
            "why is novak djokovic not the best tennis player of all time?",
            "why is rafa nadal a sneaky good grass court player?",
            "explain the difference between federer and nadal's backhand.",
            "who is the most clutch tennis player in history?",
        ]
        responses = []
        for prompt in prompts:
            response = requests.post(
                f"http://localhost:{server.port}/v1/chat/completions",
                json={
                    "model": "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
                    "messages": [
                        {
                            "role": "user",
                            "content": prompt,
                        }
                    ],
                    "max_tokens": 50,
                },
                timeout=120,
            )
            assert response.status_code == 200
            result = response.json()
            assert "choices" in result
            assert len(result["choices"]) > 0
            content = result["choices"][0]["message"]["content"]
            assert len(content) > 0
            responses.append(content)
            logger.info(f"SGLang {config.name} response: {content}")

        # For kv_events (KV routing path), assert KV publisher/scheduler log lines appear
        if config.name == "kv_events":
            log_file = os.path.join(server.log_dir, "bash.log.txt")
            assert os.path.exists(log_file), f"Log file not found: {log_file}"

            patterns = [
                r"ZMQ listener .* received batch with \d+ events \(seq=\d+\)",
                r"Event processor for worker_id \d+ processing event: Stored\(",
                r"Selected worker: \d+, logit: ",
            ]

            validate_log_patterns(log_file, patterns)

        # Test completions endpoint for disaggregated only
        if config.name == "disaggregated":
            response = requests.post(
                f"http://localhost:{server.port}/v1/completions",
                json={
                    "model": "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
                    "prompt": "Roger Federer is the greatest tennis player of all time",
                    "max_tokens": 30,
                },
                timeout=120,
            )

            assert response.status_code == 200
            result = response.json()
            assert "choices" in result
            assert len(result["choices"]) > 0
            text = result["choices"][0]["text"]
            assert len(text) > 0
            logger.info(f"SGLang completions response: {text}")


@pytest.mark.skip(
    reason="Requires 4 GPUs - enable when hardware is consistently available"
)
def test_sglang_disagg_dp_attention(request, runtime_services):
    """Test sglang disaggregated with DP attention (requires 4 GPUs)"""

    with SGLangProcess("disagg_dp_attn.sh", request) as server:
        # Test chat completions with the DP attention model
        response = requests.post(
            f"http://localhost:{server.port}/v1/chat/completions",
            json={
                "model": "silence09/DeepSeek-R1-Small-2layers",  # DP attention model
                "messages": [{"role": "user", "content": "Tell me about MoE models"}],
                "max_tokens": 50,
            },
            timeout=120,
        )

        # TODO: Once this is enabled, we can test out the rest of the HTTP endpoints around
        # flush_cache and expert distribution recording

        assert response.status_code == 200
        result = response.json()
        assert "choices" in result
        assert len(result["choices"]) > 0
        content = result["choices"][0]["message"]["content"]
        assert len(content) > 0
        logger.info(f"SGLang DP attention response: {content}")
