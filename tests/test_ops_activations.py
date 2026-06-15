# Copyright 2026 onnx2coreml contributors.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""Numerical-parity tests for activation-function lowerings."""

from __future__ import annotations

import numpy as np
import pytest
from helpers import assert_parity, single_op_model

pytestmark = pytest.mark.ops

_FMTS = ["mlpackage", "mlmodel"]


def _x(shape=(2, 3, 4, 5)) -> np.ndarray:
    # Spans negative + positive so every piecewise branch is exercised.
    rng = np.random.default_rng(0)
    return rng.standard_normal(shape).astype(np.float32)


@pytest.mark.parametrize("fmt", _FMTS)
@pytest.mark.parametrize("op", ["Relu", "Sigmoid", "Tanh", "Softplus"])
def test_unary(op, fmt):
    x = _x()
    model = single_op_model(op, {"x": x})
    assert_parity(model, {"x": x}, fmt=fmt)


@pytest.mark.parametrize("fmt", _FMTS)
def test_leaky_relu(fmt):
    x = _x()
    model = single_op_model("LeakyRelu", {"x": x}, attrs={"alpha": 0.15})
    assert_parity(model, {"x": x}, fmt=fmt)


@pytest.mark.parametrize("fmt", _FMTS)
def test_elu(fmt):
    x = _x()
    model = single_op_model("Elu", {"x": x}, attrs={"alpha": 1.3})
    assert_parity(model, {"x": x}, fmt=fmt)


@pytest.mark.parametrize("fmt", _FMTS)
def test_prelu(fmt):
    # Per-channel slope (channel dim = 3); ONNX broadcasts a [C,1,1] slope.
    x = _x((2, 3, 4, 5))
    slope = np.array([0.1, 0.25, -0.05], dtype=np.float32).reshape(3, 1, 1)
    model = single_op_model("PRelu", {"x": x}, initializers={"slope": slope})
    assert_parity(model, {"x": x}, fmt=fmt)


@pytest.mark.parametrize("fmt", _FMTS)
@pytest.mark.parametrize("slope_shape", [(1,), (1, 1, 1)])
def test_prelu_scalar_slope(fmt, slope_shape):
    # A single shared slope must broadcast to every channel (C = 3).
    x = _x((2, 3, 4, 5))
    slope = np.array(0.2, dtype=np.float32).reshape(slope_shape)
    model = single_op_model("PRelu", {"x": x}, initializers={"slope": slope})
    assert_parity(model, {"x": x}, fmt=fmt)


@pytest.mark.parametrize("fmt", _FMTS)
@pytest.mark.parametrize("approximate", ["none", "tanh"])
def test_gelu(approximate, fmt):
    x = _x()
    # ONNX Gelu is opset 20. The "tanh" path differs from Core ML's internal
    # TANH_APPROXIMATION only in evaluation order, so it needs a looser atol.
    atol = 1e-3 if approximate == "tanh" else 1e-4
    model = single_op_model(
        "Gelu", {"x": x}, attrs={"approximate": approximate}, opset=20
    )
    assert_parity(model, {"x": x}, fmt=fmt, rtol=1e-3, atol=atol)


@pytest.mark.parametrize("fmt", _FMTS)
@pytest.mark.parametrize("axis", [-1, 1])
def test_softmax(axis, fmt):
    x = _x()
    model = single_op_model("Softmax", {"x": x}, attrs={"axis": axis})
    assert_parity(model, {"x": x}, fmt=fmt)


@pytest.mark.parametrize("fmt", _FMTS)
@pytest.mark.parametrize("axis", [-1, 1])
def test_log_softmax(axis, fmt):
    x = _x()
    model = single_op_model("LogSoftmax", {"x": x}, attrs={"axis": axis})
    assert_parity(model, {"x": x}, fmt=fmt)


@pytest.mark.parametrize("fmt", _FMTS)
def test_hard_sigmoid(fmt):
    x = _x()
    model = single_op_model("HardSigmoid", {"x": x}, attrs={"alpha": 0.2, "beta": 0.5})
    assert_parity(model, {"x": x}, fmt=fmt)


@pytest.mark.parametrize("fmt", _FMTS)
def test_hard_swish(fmt):
    x = _x()
    model = single_op_model("HardSwish", {"x": x})
    assert_parity(model, {"x": x}, fmt=fmt, rtol=1e-3, atol=1e-4)


@pytest.mark.parametrize("fmt", _FMTS)
def test_prelu_multidim_slope(fmt):
    # A slope that varies across more than the channel dim: (C, H, W) broadcasts
    # unidirectionally to (N, C, H, W). MIL's prelu cannot express this, so the
    # lowering uses the elementwise identity instead.
    x = _x((2, 3, 4, 5))
    slope = _x((3, 4, 5)) * 0.1
    model = single_op_model("PRelu", {"x": x}, initializers={"slope": slope})
    assert_parity(model, {"x": x}, fmt=fmt)


@pytest.mark.parametrize("fmt", _FMTS)
def test_prelu_rank2(fmt):
    # PRelu on a rank-2 (N, C) tensor: MIL prelu needs rank >= 3, so the lowering
    # routes it through a unit spatial dim.
    x = _x((4, 6))
    slope = np.linspace(0.1, 0.6, 6, dtype=np.float32)
    model = single_op_model("PRelu", {"x": x}, initializers={"slope": slope})
    assert_parity(model, {"x": x}, fmt=fmt)
