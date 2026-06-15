# Copyright 2026 onnx2coreml contributors.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""Parity tests for normalization ops (BatchNorm, LayerNorm, InstanceNorm, GroupNorm)."""

from __future__ import annotations

import numpy as np
import pytest
from helpers import assert_parity, single_op_model
from onnx import TensorProto

pytestmark = pytest.mark.ops

FORMATS = ["mlpackage", "mlmodel"]


def _force_output_type(model, shape):
    """Stamp a concrete (FLOAT, ``shape``) type onto the single graph output.

    onnx 1.21 shape inference does not infer GroupNormalization's output type, so
    the harness leaves it UNDEFINED and ONNX Runtime refuses to load the reference
    model. GroupNorm is shape-preserving, so we set the output to the input shape.
    """
    out = model.graph.output[0]
    tt = out.type.tensor_type
    tt.elem_type = TensorProto.FLOAT
    tt.ClearField("shape")
    for d in shape:
        tt.shape.dim.add().dim_value = d
    return model


@pytest.mark.parametrize("fmt", FORMATS)
def test_batch_normalization(fmt):
    rng = np.random.default_rng(0)
    x = rng.standard_normal((2, 3, 4, 4)).astype(np.float32)
    scale = rng.standard_normal(3).astype(np.float32)
    b = rng.standard_normal(3).astype(np.float32)
    mean = rng.standard_normal(3).astype(np.float32)
    # variance must be positive.
    var = np.abs(rng.standard_normal(3)).astype(np.float32) + 0.5
    model = single_op_model(
        "BatchNormalization",
        {"X": x},
        initializers={"scale": scale, "B": b, "mean": mean, "var": var},
    )
    assert_parity(model, {"X": x}, fmt=fmt)


@pytest.mark.parametrize("fmt", FORMATS)
def test_layer_normalization_last_axis(fmt):
    rng = np.random.default_rng(1)
    x = rng.standard_normal((2, 3, 8)).astype(np.float32)
    scale = rng.standard_normal(8).astype(np.float32)
    b = rng.standard_normal(8).astype(np.float32)
    model = single_op_model(
        "LayerNormalization",
        {"X": x},
        attrs={"axis": -1},
        initializers={"Scale": scale, "B": b},
    )
    assert_parity(model, {"X": x}, fmt=fmt)


@pytest.mark.parametrize("fmt", FORMATS)
def test_layer_normalization_axis2_4d(fmt):
    # axis=2 on a 4-D input normalizes over the trailing axes [2, 3]; Scale/B are
    # shaped like those normalized dims, i.e. (4, 4).
    rng = np.random.default_rng(2)
    x = rng.standard_normal((2, 3, 4, 4)).astype(np.float32)
    scale = rng.standard_normal((4, 4)).astype(np.float32)
    b = rng.standard_normal((4, 4)).astype(np.float32)
    model = single_op_model(
        "LayerNormalization",
        {"X": x},
        attrs={"axis": 2},
        initializers={"Scale": scale, "B": b},
    )
    assert_parity(model, {"X": x}, fmt=fmt)


@pytest.mark.parametrize("fmt", FORMATS)
def test_instance_normalization(fmt):
    rng = np.random.default_rng(3)
    x = rng.standard_normal((2, 3, 4, 4)).astype(np.float32)
    scale = rng.standard_normal(3).astype(np.float32)
    b = rng.standard_normal(3).astype(np.float32)
    model = single_op_model(
        "InstanceNormalization",
        {"X": x},
        initializers={"scale": scale, "B": b},
    )
    assert_parity(model, {"X": x}, fmt=fmt)


@pytest.mark.parametrize("fmt", FORMATS)
def test_group_normalization_per_channel(fmt):
    # opset 21: scale/bias are per-channel, shape (C,). (opset 18 requires
    # (num_groups,) -- see test_group_normalization_per_group for that layout.)
    rng = np.random.default_rng(4)
    x = rng.standard_normal((2, 6, 4, 4)).astype(np.float32)
    scale = rng.standard_normal(6).astype(np.float32)
    bias = rng.standard_normal(6).astype(np.float32)
    model = single_op_model(
        "GroupNormalization",
        {"X": x},
        attrs={"num_groups": 3, "epsilon": 1e-5},
        initializers={"scale": scale, "bias": bias},
        opset=21,
    )
    assert_parity(_force_output_type(model, x.shape), {"X": x}, fmt=fmt)


@pytest.mark.xfail(
    reason="onnx 1.21 checker rejects GroupNormalization at opset 18 as deprecated "
    "(redefined at opset 21); the converter's load-time check_model refuses it. The "
    "per-group (num_groups,) scale layout only exists at opset 18, so it is "
    "unreachable here. The lowering still handles it (verified numerically vs "
    "onnxruntime, max abs diff 2.4e-7).",
    raises=Exception,
    strict=True,
)
@pytest.mark.parametrize("fmt", FORMATS)
def test_group_normalization_per_group(fmt):
    # opset 18: scale/bias are per-group, shape (num_groups,).
    rng = np.random.default_rng(5)
    x = rng.standard_normal((2, 6, 4, 4)).astype(np.float32)
    scale = rng.standard_normal(2).astype(np.float32)
    bias = rng.standard_normal(2).astype(np.float32)
    model = single_op_model(
        "GroupNormalization",
        {"X": x},
        attrs={"num_groups": 2, "epsilon": 1e-5},
        initializers={"scale": scale, "bias": bias},
        opset=18,
    )
    assert_parity(_force_output_type(model, x.shape), {"X": x}, fmt=fmt)
