<!--
SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-->

# Running DeepSeek-R1 Disaggregated with WideEP on GB200s

Dynamo supports SGLang's GB200 implementation of wide expert parallelism and large scale P/D for DeepSeek-R1! You can read their blog post [here](https://lmsys.org/blog/2025-06-16-gb200-part-1/) for more details. Full end to end optimization is still a work in progress but you can get this up and running with the following steps. In ths example, we will run 1 prefill worker on 2 GB200 nodes (4 GPUs each) and 1 decode worker on 12 GB200 nodes (total 56 GPUs).

## Instructions

1. Build the Dynamo container

```bash
cd $DYNAMO_ROOT
docker build \
  -f container/Dockerfile.sglang-wideep \
  -t dynamo-wideep-gb200 \
  --build-arg MODE=blackwell \
  --build-arg SGLANG_IMAGE_TAG=v0.5.0rc0-cu129-gb200 \
  --build-arg ARCH=arm64 \
  --build-arg ARCH_ALT=aarch64 \
  .
```

2. You can run this container on each 4xGB200 node using the following command.

> [!IMPORTANT]
> We recommend downloading DeepSeek-R1 and then mounting it to the container. You can find the model [here](https://huggingface.co/deepseek-ai/DeepSeek-R1)

```bash
docker run \
    --gpus all \
    -it \
    --rm \
    --network host \
    --volume /PATH_TO_DSR1_MODEL/:/model/ \
    --shm-size=10G \
    --ulimit memlock=-1 \
    --ulimit stack=67108864 \
    --ulimit nofile=65536:65536 \
    --cap-add CAP_SYS_PTRACE \
    --ipc host \
    dynamo-wideep-gb200:latest
```

3. On the head prefill node, run the helper script provided to generate commands to start the `nats-server`, `etcd`. This script will also tell you which environment variables to export on each node to make deployment easier.

```bash
./utils/gen_env_vars.sh
```

4. Run the ingress and prefill worker

```bash
# run ingress
python3 -m dynamo.frontend --http-port=8000 &
# run prefill worker
SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=2048 \
MC_TE_METRIC=true \
SGLANG_DISAGGREGATION_HEARTBEAT_MAX_FAILURE=100000 \
SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT=100000 \
SGLANG_DISAGGREGATION_WAITING_TIMEOUT=100000 \
SGLANG_MOONCAKE_CUSTOM_MEM_POOL=True \
MC_FORCE_MNNVL=1 \
NCCL_MNNVL_ENABLE=1 \
NCCL_CUMEM_ENABLE=1 \
SGLANG_USE_MESSAGE_QUEUE_BROADCASTER=0 \
SGL_DISABLE_TP_MEMORY_INBALANCE_CHECK=1 \
PYTHONUNBUFFERED=1 \
python3 -m dynamo.sglang.worker \
  --served-model-name deepseek-ai/DeepSeek-R1 \
  --model-path /model/ \
  --skip-tokenizer-init \
  --trust-remote-code \
  --disaggregation-mode prefill \
  --dist-init-addr ${HEAD_PREFILL_NODE_IP}:29500 \
  --disaggregation-bootstrap-port 30001 \
  --nnodes 2 \
  --node-rank 0 \
  --tp-size 8 \
  --dp-size 8 \
  --enable-dp-attention \
  --host 0.0.0.0 \
  --decode-log-interval 1 \
  --max-running-requests 6144 \
  --context-length 2716 \
  --disable-radix-cache \
  --enable-deepep-moe \
  --deepep-mode low_latency \
  --moe-dense-tp-size 1 \
  --enable-dp-lm-head \
  --disable-shared-experts-fusion \
  --ep-num-redundant-experts 32 \
  --ep-dispatch-algorithm static \
  --eplb-algorithm deepseek \
  --attention-backend cutlass_mla \
  --watchdog-timeout 1000000 \
  --disable-cuda-graph \
  --chunked-prefill-size 16384 \
  --max-total-tokens 32768 \
  --mem-fraction-static 0.8 \
  --log-level debug
```

5. Run the decode worker on the head decode node

```bash
SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=768 \
MC_TE_METRIC=true \
SGLANG_DISAGGREGATION_HEARTBEAT_MAX_FAILURE=100000 \
SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT=100000 \
SGLANG_DISAGGREGATION_WAITING_TIMEOUT=100000 \
SGLANG_HACK_SEQ_BOOTSTRAP_ROOM=1 \
SGLANG_MOONCAKE_CUSTOM_MEM_POOL=True \
NCCL_MNNVL_ENABLE=1 \
MC_FORCE_MNNVL=1 \
NCCL_CUMEM_ENABLE=1 \
SGLANG_USE_MESSAGE_QUEUE_BROADCASTER=0 \
SGL_DISABLE_TP_MEMORY_INBALANCE_CHECK=1 \
PYTHONUNBUFFERED=1 \
python3 -m dynamo.sglang.decode_worker \
  --served-model-name deepseek-ai/DeepSeek-R1 \
  --model-path /model/ \
  --skip-tokenizer-init \
  --trust-remote-code \
  --disaggregation-mode decode \
  --dist-init-addr ${HEAD_DECODE_NODE_IP}:29500 \
  --disaggregation-bootstrap-port 30001 \
  --nnodes 12 \
  --node-rank 0 \
  --tp-size 48 \
  --dp-size 48 \
  --enable-dp-attention \
  --host 0.0.0.0 \
  --decode-log-interval 1 \
  --max-running-requests 36864 \
  --context-length 2716 \
  --disable-radix-cache \
  --enable-deepep-moe \
  --deepep-mode low_latency \
  --moe-dense-tp-size 1 \
  --enable-dp-lm-head \
  --cuda-graph-bs 768 \
  --disable-shared-experts-fusion \
  --ep-num-redundant-experts 32 \
  --ep-dispatch-algorithm static \
  --eplb-algorithm deepseek \
  --attention-backend cutlass_mla \
  --watchdog-timeout 1000000 \
  --chunked-prefill-size 36864 \
  --mem-fraction-static 0.82 \
  --log-level debug
```

On the other decode nodes (this example has 12 total decode nodes), run the same command but change `--node-rank` to 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11
