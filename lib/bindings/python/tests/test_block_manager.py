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


import asyncio

import pytest
import torch

# Attempt to import the optional module
try:
    from dynamo.llm import BlockManager
except ImportError:
    pytest.importorskip(
        "optional_module", reason="block-manager feature is not enabled"
    )

pytestmark = pytest.mark.pre_merge


WORKER_ID = 0
NUM_LAYER = 5
OUTER_DIM = 2
PAGE_SIZE = 4
INNER_DIM = 13
DTYPE, TORCH_DTYPE = "FP32", torch.float32
HOST_NUM_BLOCKS = 16
DEVICE_NUM_BLOCKS = 16
DEVICE_ID = 0


def new_block_manager():
    return BlockManager(
        WORKER_ID,
        NUM_LAYER,
        OUTER_DIM,
        PAGE_SIZE,
        INNER_DIM,
        DTYPE,
        HOST_NUM_BLOCKS,
        DEVICE_NUM_BLOCKS,
        DEVICE_ID,
    )


@pytest.fixture
def block_manager():
    return new_block_manager()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
async def test_block_manager_initialization():
    # Python should drop the BlockManager instance as soon as it goes out of scope, but
    # it may not be garbage collected immediately, depending on the garbage collector.
    BlockManager(WORKER_ID, NUM_LAYER, OUTER_DIM, PAGE_SIZE, INNER_DIM)
    BlockManager(WORKER_ID, NUM_LAYER, OUTER_DIM, PAGE_SIZE, INNER_DIM, DTYPE)
    BlockManager(
        WORKER_ID, NUM_LAYER, OUTER_DIM, PAGE_SIZE, INNER_DIM, DTYPE, HOST_NUM_BLOCKS
    )
    BlockManager(
        WORKER_ID,
        NUM_LAYER,
        OUTER_DIM,
        PAGE_SIZE,
        INNER_DIM,
        DTYPE,
        device_num_blocks=DEVICE_NUM_BLOCKS,
    )
    BlockManager(
        WORKER_ID,
        NUM_LAYER,
        OUTER_DIM,
        PAGE_SIZE,
        INNER_DIM,
        DTYPE,
        HOST_NUM_BLOCKS,
        DEVICE_NUM_BLOCKS,
    )
    BlockManager(
        WORKER_ID,
        NUM_LAYER,
        OUTER_DIM,
        PAGE_SIZE,
        INNER_DIM,
        DTYPE,
        device_num_blocks=DEVICE_NUM_BLOCKS,
        device_id=DEVICE_ID,
    )
    BlockManager(
        WORKER_ID,
        NUM_LAYER,
        OUTER_DIM,
        PAGE_SIZE,
        INNER_DIM,
        DTYPE,
        HOST_NUM_BLOCKS,
        DEVICE_NUM_BLOCKS,
        DEVICE_ID,
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
async def test_cpu_block_access(block_manager: BlockManager):
    block_count = 2
    block_list = block_manager.allocate_host_blocks_blocking(block_count)
    blocks = block_list.to_list()
    assert len(blocks) == block_count
    tensors = [torch.from_dlpack(b) for b in blocks]
    for tensor in tensors:
        assert tensor.get_device() == -1  # CPU
        assert tensor.shape == (1, NUM_LAYER, OUTER_DIM, PAGE_SIZE, INNER_DIM)
        assert tensor.dtype == TORCH_DTYPE
    # print(tensors)
    for tensor in tensors:
        tensor[0][0][0][0][0] = 1.0
        tensor[0][NUM_LAYER - 1][OUTER_DIM - 1][PAGE_SIZE - 1][INNER_DIM - 1] = 1.0
    # print(tensors)
    blocks_ = block_list.to_list()
    assert blocks is not blocks_
    assert len(blocks) == len(blocks_)
    tensors_ = [torch.from_dlpack(b) for b in blocks_]
    for tensor, tensor_ in zip(tensors, tensors_):
        assert tensor is not tensor_
        assert tensor.shape == tensor_.shape
        assert tensor.dtype == tensor_.dtype
        assert torch.allclose(tensor, tensor_)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
async def test_gpu_block_access(block_manager: BlockManager):
    block_count = 6
    block_list = block_manager.allocate_device_blocks_blocking(block_count)
    blocks = block_list.to_list()
    assert len(blocks) == block_count
    tensors = [torch.from_dlpack(b) for b in blocks]
    for tensor in tensors:
        assert tensor.get_device() == DEVICE_ID  # GPU
        assert tensor.shape == (1, NUM_LAYER, OUTER_DIM, PAGE_SIZE, INNER_DIM)
        assert tensor.dtype == TORCH_DTYPE
    # print(tensors)
    for tensor in tensors:
        tensor[0][0][0][0][0] = 1.0
        tensor[0][NUM_LAYER - 1][OUTER_DIM - 1][PAGE_SIZE - 1][INNER_DIM - 1] = 1.0
    # print(tensors)
    blocks_ = block_list.to_list()
    assert blocks is not blocks_
    assert len(blocks) == len(blocks_)
    tensors_ = [torch.from_dlpack(b) for b in blocks_]
    for tensor, tensor_ in zip(tensors, tensors_):
        assert tensor is not tensor_
        assert tensor.shape == tensor_.shape
        assert tensor.dtype == tensor_.dtype
        assert torch.allclose(tensor, tensor_)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
async def test_block_list_iteration(block_manager: BlockManager):
    block_count = 4
    block_list = await block_manager.allocate_host_blocks(block_count)
    # Test __len__()
    assert len(block_list) == block_count
    # Test __getitem__()
    for i in range(block_count):
        block = block_list[i]
        tensor = torch.from_dlpack(block)
        tensor[0][0][0][0][0] = 1.0 + i
    # Test __iter__() and __next__()
    idx = 1.0
    for block in block_list:
        tensor = torch.from_dlpack(block)
        assert tensor[0][0][0][0][0] == idx
        tensor[0][0][0][0][0] += 0.5
        idx += 1.0
    assert idx == 1.0 + block_count
    # Test __iter__() should reset current index
    idx = 1.0
    for block in block_list:
        tensor = torch.from_dlpack(block)
        assert tensor[0][0][0][0][0] == idx + 0.5
        idx += 1.0
    assert idx == 1.0 + block_count


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
async def test_block_copy_g1_g2(block_manager: BlockManager):
    # Allocate device (G1) and host (G2) block
    host_block_list = await block_manager.allocate_host_blocks(1)
    device_block_list = await block_manager.allocate_device_blocks(1)
    # Populate host block with unique values
    host_tensor = torch.from_dlpack(host_block_list[0])
    for i in range(NUM_LAYER):
        for j in range(OUTER_DIM):
            for k in range(PAGE_SIZE):
                for w in range(INNER_DIM):
                    host_tensor[0][i][j][k][w] = (
                        i * OUTER_DIM * PAGE_SIZE * INNER_DIM
                        + j * PAGE_SIZE * INNER_DIM
                        + k * INNER_DIM
                        + w
                    )
    # Copy host block to device block after permuting
    permute_dims = (0, 2, 4, 3, 1)
    device_tensor_ = torch.from_dlpack(device_block_list[0]).permute(*permute_dims)
    device_tensor_.copy_(host_tensor.permute(*permute_dims))
    # Assert device block is contiguous and updated in block manager
    device_tensor = torch.from_dlpack(device_block_list[0])
    for i in range(NUM_LAYER):
        for j in range(OUTER_DIM):
            for k in range(PAGE_SIZE):
                for w in range(INNER_DIM):
                    assert (
                        device_tensor[0][i][j][k][w]
                        == i * OUTER_DIM * PAGE_SIZE * INNER_DIM
                        + j * PAGE_SIZE * INNER_DIM
                        + k * INNER_DIM
                        + w
                    )
    # Set host block to zero and assert updated in block manager
    host_tensor_ = torch.from_dlpack(host_block_list[0]).permute(*permute_dims)
    host_tensor_.zero_()
    assert torch.all(host_tensor == 0)
    # Copy device block back to host block
    host_tensor_.copy_(device_tensor_)
    # Assert host block is updated in block manager
    for i in range(NUM_LAYER):
        for j in range(OUTER_DIM):
            for k in range(PAGE_SIZE):
                for w in range(INNER_DIM):
                    assert (
                        host_tensor[0][i][j][k][w]
                        == i * OUTER_DIM * PAGE_SIZE * INNER_DIM
                        + j * PAGE_SIZE * INNER_DIM
                        + k * INNER_DIM
                        + w
                    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
async def test_cpu_layer_access(block_manager: BlockManager):
    block_list = block_manager.allocate_host_blocks_blocking(1)
    block = block_list[0]
    layers = block.to_list()
    assert len(layers) == NUM_LAYER
    tensors = [torch.from_dlpack(bl) for bl in layers]
    for tensor in tensors:
        assert tensor.get_device() == -1  # CPU
        assert tensor.shape == (1, 1, OUTER_DIM, PAGE_SIZE, INNER_DIM)
        assert tensor.dtype == TORCH_DTYPE
    # print(tensors)
    for tensor in tensors:
        tensor[0][0][0][0][0] = 1.0
        tensor[0][0][OUTER_DIM - 1][PAGE_SIZE - 1][INNER_DIM - 1] = 1.0
    # print(tensors)
    layers_ = block.to_list()
    assert layers is not layers_
    assert len(layers) == len(layers_)
    tensors_ = [torch.from_dlpack(bl) for bl in layers_]
    for tensor, tensor_ in zip(tensors, tensors_):
        assert tensor is not tensor_
        assert tensor.shape == tensor_.shape
        assert tensor.dtype == tensor_.dtype
        assert torch.allclose(tensor, tensor_)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
async def test_gpu_layer_access(block_manager: BlockManager):
    block_list = block_manager.allocate_device_blocks_blocking(1)
    block = block_list[0]
    layers = block.to_list()
    assert len(layers) == NUM_LAYER
    tensors = [torch.from_dlpack(bl) for bl in layers]
    for tensor in tensors:
        assert tensor.get_device() == DEVICE_ID  # GPU
        assert tensor.shape == (1, 1, OUTER_DIM, PAGE_SIZE, INNER_DIM)
        assert tensor.dtype == TORCH_DTYPE
    # print(tensors)
    for tensor in tensors:
        tensor[0][0][0][0][0] = 1.0
        tensor[0][0][OUTER_DIM - 1][PAGE_SIZE - 1][INNER_DIM - 1] = 1.0
    # print(tensors)
    layers_ = block.to_list()
    assert layers is not layers_
    assert len(layers) == len(layers_)
    tensors_ = [torch.from_dlpack(bl) for bl in layers_]
    for tensor, tensor_ in zip(tensors, tensors_):
        assert tensor is not tensor_
        assert tensor.shape == tensor_.shape
        assert tensor.dtype == tensor_.dtype
        assert torch.allclose(tensor, tensor_)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
async def test_block_iteration(block_manager: BlockManager):
    block = (await block_manager.allocate_host_blocks(1))[0]
    # Test __len__()
    assert len(block) == NUM_LAYER
    # Test __getitem__()
    for i in range(NUM_LAYER):
        layer = block[i]
        tensor = torch.from_dlpack(layer)
        tensor[0][0][0][0][0] = 1.0 + i
    # Test __iter__() and __next__()
    idx = 1.0
    for layer in block:
        tensor = torch.from_dlpack(layer)
        assert tensor[0][0][0][0][0] == idx
        tensor[0][0][0][0][0] += 0.5
        idx += 1.0
    assert idx == 1.0 + NUM_LAYER
    # Test __iter__() should reset current index
    idx = 1.0
    for layer in block:
        tensor = torch.from_dlpack(layer)
        assert tensor[0][0][0][0][0] == idx + 0.5
        idx += 1.0
    assert idx == 1.0 + NUM_LAYER


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
async def test_block_layer_copy_g1_g2(block_manager: BlockManager):
    # Allocate device (G1) and host (G2) block
    host_block = (await block_manager.allocate_host_blocks(1))[0]
    device_block = (await block_manager.allocate_device_blocks(1))[0]
    # Populate host block at layer level with unique values
    host_layer_tensors = [torch.from_dlpack(bl) for bl in host_block]
    for i in range(NUM_LAYER):
        host_layer_tensor = host_layer_tensors[i]
        for j in range(OUTER_DIM):
            for k in range(PAGE_SIZE):
                for w in range(INNER_DIM):
                    host_layer_tensor[0][0][j][k][w] = (
                        i * OUTER_DIM * PAGE_SIZE * INNER_DIM
                        + j * PAGE_SIZE * INNER_DIM
                        + k * INNER_DIM
                        + w
                    )
    # Copy host block to device block after permuting
    permute_dims = (0, 2, 4, 3, 1)
    host_block_tensor_ = torch.from_dlpack(host_block).permute(*permute_dims)
    device_block_tensor_ = torch.from_dlpack(device_block).permute(*permute_dims)
    device_block_tensor_.copy_(host_block_tensor_)
    # Assert device block is contiguous and updated in block manager at layer level
    device_layer_tensors = [torch.from_dlpack(bl) for bl in device_block]
    for i in range(NUM_LAYER):
        device_layer_tensor = device_layer_tensors[i]
        for j in range(OUTER_DIM):
            for k in range(PAGE_SIZE):
                for w in range(INNER_DIM):
                    assert (
                        device_layer_tensor[0][0][j][k][w]
                        == i * OUTER_DIM * PAGE_SIZE * INNER_DIM
                        + j * PAGE_SIZE * INNER_DIM
                        + k * INNER_DIM
                        + w
                    )
    # Set host block to zero and assert updated in block manager
    host_block_tensor = torch.from_dlpack(host_block)
    host_block_tensor.zero_()
    assert torch.all(host_block_tensor_ == 0)
    # Copy device block back to host block
    host_block_tensor_.copy_(device_block_tensor_)
    # Assert host block is updated in block manager
    for i in range(NUM_LAYER):
        for j in range(OUTER_DIM):
            for k in range(PAGE_SIZE):
                for w in range(INNER_DIM):
                    assert (
                        host_block_tensor[0][i][j][k][w]
                        == i * OUTER_DIM * PAGE_SIZE * INNER_DIM
                        + j * PAGE_SIZE * INNER_DIM
                        + k * INNER_DIM
                        + w
                    )


async def main():
    await test_block_manager_initialization()
    await test_cpu_block_access(new_block_manager())
    await test_gpu_block_access(new_block_manager())
    await test_block_list_iteration(new_block_manager())
    await test_block_copy_g1_g2(new_block_manager())
    await test_cpu_layer_access(new_block_manager())
    await test_gpu_layer_access(new_block_manager())
    await test_block_iteration(new_block_manager())
    await test_block_layer_copy_g1_g2(new_block_manager())


if __name__ == "__main__":
    asyncio.run(main())
