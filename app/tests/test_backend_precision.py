from __future__ import annotations

import torch

from app.models.backend import _set_model_precision


def test_fp32_cast_repairs_half_checkpoint_for_cpu():
    layer = torch.nn.Conv2d(3, 4, kernel_size=3).half()
    assert layer.weight.dtype == torch.float16
    assert layer.bias is not None and layer.bias.dtype == torch.float16

    _set_model_precision(layer, use_fp16=False)

    assert layer.weight.dtype == torch.float32
    assert layer.bias is not None and layer.bias.dtype == torch.float32
    x = torch.randn(1, 3, 16, 16, dtype=torch.float32)
    y = layer(x)
    assert y.dtype == torch.float32


def test_fp16_cast_keeps_input_and_model_compatible():
    layer = torch.nn.Conv2d(3, 4, kernel_size=3).float()
    _set_model_precision(layer, use_fp16=True)
    assert layer.weight.dtype == torch.float16
    x = torch.randn(1, 3, 16, 16, dtype=torch.float16)
    y = layer(x)
    assert y.dtype == torch.float16
