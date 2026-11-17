"""Milestone 0 smoke tests: the toolchain we depend on is present and behaves as expected."""

import pytest
import torch


def test_torch_autograd_works() -> None:
    x = torch.tensor([2.0, 3.0], requires_grad=True)
    (x**2).sum().backward()
    assert torch.allclose(x.grad, torch.tensor([4.0, 6.0]))


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA device")
def test_cuda_bf16_matmul() -> None:
    # Ampere (sm_86) supports bf16 tensor-core matmuls, which pretraining relies on.
    assert torch.cuda.is_bf16_supported()
    a = torch.randn(256, 256, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(256, 256, device="cuda", dtype=torch.bfloat16)
    ref = a.float() @ b.float()
    assert torch.allclose((a @ b).float(), ref, atol=1.0, rtol=0.05)
