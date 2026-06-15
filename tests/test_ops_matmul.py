# Copyright 2026 onnx2coreml contributors.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""Numerical-parity tests for MatMul and Gemm lowerings."""

from __future__ import annotations

import numpy as np
import pytest
from helpers import assert_parity, single_op_model

pytestmark = pytest.mark.ops

FORMATS = ["mlpackage", "mlmodel"]


def _rand(*shape: int) -> np.ndarray:
    return np.random.default_rng(0).standard_normal(shape).astype(np.float32)


@pytest.mark.parametrize("fmt", FORMATS)
def test_matmul_2d(fmt: str) -> None:
    inputs = {"A": _rand(3, 4), "B": _rand(4, 5)}
    model = single_op_model("MatMul", inputs)
    assert_parity(model, inputs, fmt=fmt)


@pytest.mark.parametrize("fmt", FORMATS)
def test_matmul_batched(fmt: str) -> None:
    inputs = {"A": _rand(2, 3, 4), "B": _rand(2, 4, 5)}
    model = single_op_model("MatMul", inputs)
    assert_parity(model, inputs, fmt=fmt)


@pytest.mark.parametrize("fmt", FORMATS)
def test_matmul_broadcast(fmt: str) -> None:
    # (B, M, K) x (K, N): the rank-2 operand broadcasts across the batch dim.
    inputs = {"A": _rand(2, 3, 4), "B": _rand(4, 5)}
    model = single_op_model("MatMul", inputs)
    assert_parity(model, inputs, fmt=fmt)


@pytest.mark.parametrize("fmt", FORMATS)
@pytest.mark.parametrize("trans_a", [0, 1])
@pytest.mark.parametrize("trans_b", [0, 1])
def test_gemm_transpose_no_bias(fmt: str, trans_a: int, trans_b: int) -> None:
    m, k, n = 3, 4, 5
    a = _rand(k, m) if trans_a else _rand(m, k)
    b = _rand(n, k) if trans_b else _rand(k, n)
    inputs = {"A": a, "B": b}
    model = single_op_model(
        "Gemm", inputs, attrs={"transA": trans_a, "transB": trans_b}
    )
    assert_parity(model, inputs, fmt=fmt)


@pytest.mark.parametrize("fmt", FORMATS)
@pytest.mark.parametrize("trans_a", [0, 1])
@pytest.mark.parametrize("trans_b", [0, 1])
def test_gemm_bias_initializer_scaled(
    fmt: str, trans_a: int, trans_b: int
) -> None:
    # C as an initializer, with non-unit alpha/beta and broadcastable bias.
    m, k, n = 3, 4, 5
    a = _rand(k, m) if trans_a else _rand(m, k)
    b = _rand(n, k) if trans_b else _rand(k, n)
    c = _rand(n).astype(np.float32)
    inputs = {"A": a, "B": b}
    model = single_op_model(
        "Gemm",
        inputs,
        attrs={"transA": trans_a, "transB": trans_b, "alpha": 0.75, "beta": 0.5},
        initializers={"C": c},
    )
    assert_parity(model, inputs, fmt=fmt)


@pytest.mark.parametrize("fmt", FORMATS)
def test_gemm_bias_input_scaled(fmt: str) -> None:
    # C as a full third input (M, N), exercising the input (non-const) path.
    m, k, n = 3, 4, 5
    inputs = {"A": _rand(m, k), "B": _rand(k, n), "C": _rand(m, n)}
    model = single_op_model("Gemm", inputs, attrs={"alpha": 1.5, "beta": 2.0})
    assert_parity(model, inputs, fmt=fmt)


@pytest.mark.parametrize("fmt", FORMATS)
@pytest.mark.parametrize("const_input", [True, False])
def test_inverse_2x2(fmt, const_input):
    # com.microsoft::Inverse over batched 2x2 matrices: folded when the matrix is a
    # constant, closed-form 2x2 inverse when it is a runtime input.
    from onnx import TensorProto, helper, numpy_helper

    rng = np.random.default_rng(0)
    mat = (rng.standard_normal((1, 10, 2, 2)).astype(np.float32)
           + 2.0 * np.eye(2, dtype=np.float32))
    x = rng.standard_normal((1, 10, 2, 2)).astype(np.float32)
    nodes = [
        helper.make_node("Inverse", ["M"], ["inv"], domain="com.microsoft"),
        helper.make_node("Add", ["inv", "x"], ["out0"]),
    ]
    input_vis = [helper.make_tensor_value_info("x", TensorProto.FLOAT, x.shape)]
    inits = []
    feed = {"x": x}
    if const_input:
        inits = [numpy_helper.from_array(mat, "M")]
    else:
        input_vis.append(helper.make_tensor_value_info("M", TensorProto.FLOAT, mat.shape))
        feed["M"] = mat
    graph = helper.make_graph(
        nodes, "inverse", input_vis,
        [helper.make_tensor_value_info("out0", TensorProto.FLOAT, mat.shape)],
        initializer=inits,
    )
    model = helper.make_model(
        graph, opset_imports=[helper.make_opsetid("", 17), helper.make_opsetid("com.microsoft", 1)],
        ir_version=10,
    )
    assert_parity(model, feed, fmt=fmt)
