# SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch


@pytest.fixture(autouse=True)
def set_random_seed():
    """Set the random seed for torch CPU and CUDA for deterministic tests."""
    seed = 0
    torch.manual_seed(seed)
    # Seed all CUDA devices if available
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# Global skip condition if CUDA is not available
cuda_available = torch.cuda.is_available()
skip_if_no_cuda = pytest.mark.skipif(not cuda_available, reason='CUDA not available')


def test_cuda_available():
    """Tests if CUDA is available to PyTorch."""
    assert cuda_available, 'CUDA is not available to PyTorch'


@skip_if_no_cuda
def test_cuda_device_count():
    """Tests if at least one CUDA device is detected."""
    assert torch.cuda.device_count() > 0, 'No CUDA devices found'


@skip_if_no_cuda
def test_tensor_to_cuda():
    """Tests creating a tensor and moving it to the CUDA device."""
    tensor_cpu = torch.randn(3, 3)
    try:
        tensor_gpu = tensor_cpu.cuda()
        assert tensor_gpu.is_cuda, 'Tensor failed to move to CUDA device'
        assert tensor_gpu.device.type == 'cuda'
    except Exception as e:
        pytest.fail(f'Failed to move tensor to CUDA: {e}')


@skip_if_no_cuda
def test_cuda_matmul():
    """Tests matrix multiplication on the CUDA device (checks cuBLAS implicitly)."""
    device = torch.device('cuda')
    tensor_a = torch.randn(10, 5, device=device)
    tensor_b = torch.randn(5, 8, device=device)
    try:
        result = torch.matmul(tensor_a, tensor_b)
        assert result.shape == (10, 8), 'Matrix multiplication result has incorrect shape'
        assert result.is_cuda, 'Matrix multiplication result is not on CUDA device'
    except Exception as e:
        pytest.fail(f'CUDA matrix multiplication failed: {e}')


@skip_if_no_cuda
def test_cuda_backward():
    """Tests the backward pass (gradient computation) on CUDA tensors."""
    device = torch.device('cuda')
    x = torch.randn(5, 5, device=device, requires_grad=True)
    y = torch.randn(5, 5, device=device)
    z = (x * y).sum()
    try:
        z.backward()
        assert x.grad is not None, 'Gradient was not computed'
        assert x.grad.is_cuda, 'Gradient tensor is not on CUDA device'
        assert x.grad.shape == x.shape, 'Gradient has incorrect shape'
    except Exception as e:
        pytest.fail(f'CUDA backward pass failed: {e}')


def test_cudnn_available():
    """Checks if cuDNN is available."""
    cudnn_available = torch.backends.cudnn.is_available()
    print(f'cuDNN available: {cudnn_available}')
    assert cudnn_available


# Optional: Print some version info when tests run
def test_print_versions():
    """Prints relevant library versions."""
    print(f'PyTorch version: {torch.__version__}')
    if cuda_available:
        print(f'PyTorch CUDA version: {torch.version.cuda}')
        print(f'Detected CUDA devices: {torch.cuda.device_count()}')
        for i in range(torch.cuda.device_count()):
            print(f'  Device {i}: {torch.cuda.get_device_name(i)}')
    else:
        print('CUDA not available.')
