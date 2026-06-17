# Copyright 2026 onnx2coreml contributors.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""Parity tests for elementwise ops."""

from __future__ import annotations

import numpy as np
import onnx
import pytest
from helpers import assert_parity, single_op_model
from onnx import TensorProto, helper

pytestmark = pytest.mark.ops

_SEED = np.random.default_rng(0)


_BINARY_SHAPES = [((2, 3), (2, 3)), ((2, 3), (3,)), ((4, 1, 3), (1, 2, 3))]


@pytest.mark.parametrize("fmt", ["mlpackage", "mlmodel"])
@pytest.mark.parametrize("shapes", _BINARY_SHAPES)
def test_add(fmt, shapes):
    a = _SEED.random(shapes[0]).astype(np.float32)
    b = _SEED.random(shapes[1]).astype(np.float32)
    assert_parity(single_op_model("Add", {"a": a, "b": b}), {"a": a, "b": b}, fmt=fmt)


@pytest.mark.parametrize("fmt", ["mlpackage", "mlmodel"])
@pytest.mark.parametrize("op", ["Sub", "Mul", "Div", "Pow"])
@pytest.mark.parametrize("shapes", _BINARY_SHAPES)
def test_arithmetic(fmt, op, shapes):
    # Keep operands positive so Pow stays real-valued.
    a = _SEED.random(shapes[0]).astype(np.float32) + 0.5
    b = _SEED.random(shapes[1]).astype(np.float32) + 0.5
    assert_parity(single_op_model(op, {"a": a, "b": b}), {"a": a, "b": b}, fmt=fmt)


@pytest.mark.parametrize("fmt", ["mlpackage", "mlmodel"])
@pytest.mark.parametrize("exp_shape", [(1,), ()])
def test_pow_scalar_constant_exponent(fmt, exp_shape):
    # The NeuralNetwork backend requires a scalar exponent for pow-by-constant.
    x = _SEED.random((2, 3, 4)).astype(np.float32) + 0.5
    exponent = np.array(2.0, dtype=np.float32).reshape(exp_shape)
    model = single_op_model("Pow", {"x": x}, initializers={"exp": exponent})
    assert_parity(model, {"x": x}, fmt=fmt)


@pytest.mark.parametrize("fmt", ["mlpackage", "mlmodel"])
@pytest.mark.parametrize("op", ["Sqrt", "Exp", "Log", "Reciprocal"])
@pytest.mark.parametrize("shape", [(2, 3), (4, 1, 5)])
def test_unary_positive(fmt, op, shape):
    # These are only defined / well-conditioned on positive inputs.
    x = _SEED.random(shape).astype(np.float32) + 0.5
    assert_parity(single_op_model(op, {"x": x}), {"x": x}, fmt=fmt)


@pytest.mark.parametrize("fmt", ["mlpackage", "mlmodel"])
@pytest.mark.parametrize("op", ["Abs", "Neg", "Erf"])
@pytest.mark.parametrize("shape", [(2, 3), (4, 1, 5)])
def test_unary_signed(fmt, op, shape):
    x = (_SEED.random(shape).astype(np.float32) - 0.5) * 4.0
    assert_parity(single_op_model(op, {"x": x}), {"x": x}, fmt=fmt)


@pytest.mark.parametrize("fmt", ["mlpackage", "mlmodel"])
@pytest.mark.parametrize("op", ["Min", "Max"])
@pytest.mark.parametrize("shapes", _BINARY_SHAPES)
def test_min_max_binary(fmt, op, shapes):
    a = _SEED.random(shapes[0]).astype(np.float32)
    b = _SEED.random(shapes[1]).astype(np.float32)
    assert_parity(single_op_model(op, {"a": a, "b": b}), {"a": a, "b": b}, fmt=fmt)


@pytest.mark.parametrize("fmt", ["mlpackage", "mlmodel"])
@pytest.mark.parametrize("op", ["Min", "Max"])
def test_min_max_variadic(fmt, op):
    # ONNX Min/Max are variadic: exercise the pairwise fold with 3 inputs.
    a = _SEED.random((2, 3)).astype(np.float32)
    b = _SEED.random((2, 3)).astype(np.float32)
    c = _SEED.random((2, 3)).astype(np.float32)
    ins = {"a": a, "b": b, "c": c}
    assert_parity(single_op_model(op, ins), ins, fmt=fmt)


@pytest.mark.parametrize("fmt", ["mlpackage", "mlmodel"])
def test_clip_both_bounds(fmt):
    x = (_SEED.random((3, 4)).astype(np.float32) - 0.5) * 4.0
    inits = {"lo": np.float32(-0.5), "hi": np.float32(0.5)}
    assert_parity(
        single_op_model("Clip", {"x": x}, initializers=inits),
        {"x": x},
        fmt=fmt,
    )


@pytest.mark.parametrize("fmt", ["mlpackage", "mlmodel"])
def test_clip_min_only(fmt):
    # Only the min bound is supplied (input 1); the absent max (input 2) must
    # fall back to the dtype's representable max in the lowering.
    x = (_SEED.random((3, 4)).astype(np.float32) - 0.5) * 4.0
    model = single_op_model("Clip", {"x": x}, initializers={"lo": np.float32(0.0)})
    assert_parity(model, {"x": x}, fmt=fmt)


@pytest.mark.parametrize("fmt", ["mlpackage", "mlmodel"])
@pytest.mark.parametrize("op", ["Equal", "Greater", "Less"])
@pytest.mark.parametrize("shapes", _BINARY_SHAPES)
def test_comparison(fmt, op, shapes):
    # Draw from a small integer-valued range so true and false cases both occur.
    a = _SEED.integers(0, 3, size=shapes[0]).astype(np.float32)
    b = _SEED.integers(0, 3, size=shapes[1]).astype(np.float32)
    assert_parity(single_op_model(op, {"a": a, "b": b}), {"a": a, "b": b}, fmt=fmt)


def _greater_where_model(shape) -> onnx.ModelProto:
    """Two-node graph ``Where(Greater(g0, g1), x, y)``.

    The bool condition is produced *inside* the graph (Core ML I/O does not
    support bool features at this target), so all four graph inputs are float.
    """
    fp = TensorProto.FLOAT
    names = ["g0", "g1", "x", "y"]
    vis = [helper.make_tensor_value_info(n, fp, shape) for n in names]
    nodes = [
        helper.make_node("Greater", ["g0", "g1"], ["cond"]),
        helper.make_node("Where", ["cond", "x", "y"], ["out0"]),
    ]
    graph = helper.make_graph(
        nodes,
        "test_Where",
        vis,
        [helper.make_tensor_value_info("out0", TensorProto.UNDEFINED, None)],
    )
    model = helper.make_model(
        graph, opset_imports=[helper.make_opsetid("", 17)], ir_version=10
    )
    return onnx.shape_inference.infer_shapes(model, strict_mode=True)


@pytest.mark.parametrize("fmt", ["mlpackage", "mlmodel"])
@pytest.mark.parametrize("shape", [(2, 3), (4, 1, 3)])
def test_where(fmt, shape):
    # Integer-valued operands so Greater yields a mix of true/false.
    g0 = _SEED.integers(0, 3, size=shape).astype(np.float32)
    g1 = _SEED.integers(0, 3, size=shape).astype(np.float32)
    x = _SEED.random(shape).astype(np.float32)
    y = _SEED.random(shape).astype(np.float32)
    ins = {"g0": g0, "g1": g1, "x": x, "y": y}
    assert_parity(_greater_where_model(shape), ins, fmt=fmt)


# --- trigonometric / hyperbolic --------------------------------------------

_TRIG_DOMAIN = {
    "Sin": (-3.0, 3.0), "Cos": (-3.0, 3.0), "Tan": (-1.0, 1.0),
    "Asin": (-0.9, 0.9), "Acos": (-0.9, 0.9), "Atan": (-3.0, 3.0),
    "Sinh": (-2.0, 2.0), "Cosh": (-2.0, 2.0), "Atanh": (-0.9, 0.9),
}


@pytest.mark.parametrize("fmt", ["mlpackage", "mlmodel"])
@pytest.mark.parametrize("op", list(_TRIG_DOMAIN))
@pytest.mark.parametrize("shape", [(2, 3), (4, 1, 5)])
def test_trig(fmt, op, shape):
    lo, hi = _TRIG_DOMAIN[op]
    x = (_SEED.random(shape).astype(np.float32) * (hi - lo) + lo).astype(np.float32)
    assert_parity(single_op_model(op, {"x": x}), {"x": x}, fmt=fmt)


@pytest.mark.parametrize("fmt", ["mlpackage", "mlmodel"])
@pytest.mark.parametrize("op", ["Floor", "Ceil", "Round", "Sign"])
@pytest.mark.parametrize("shape", [(2, 3), (4, 5)])
def test_rounding(fmt, op, shape):
    x = ((_SEED.random(shape).astype(np.float32) - 0.5) * 8.0).astype(np.float32)
    assert_parity(single_op_model(op, {"x": x}), {"x": x}, fmt=fmt)


@pytest.mark.parametrize("fmt", ["mlpackage", "mlmodel"])
def test_mod_int(fmt):
    a = _SEED.integers(0, 50, size=(2, 4)).astype(np.int32)
    b = _SEED.integers(1, 9, size=(2, 4)).astype(np.int32)
    assert_parity(single_op_model("Mod", {"a": a, "b": b}), {"a": a, "b": b}, fmt=fmt)


# --- bool-producing ops, consumed by Where (Core ML I/O has no bool) --------

def _compare_where_model(op, shape) -> onnx.ModelProto:
    fp = TensorProto.FLOAT
    vis = [helper.make_tensor_value_info(n, fp, shape) for n in ["g0", "g1", "x", "y"]]
    nodes = [
        helper.make_node(op, ["g0", "g1"], ["cond"]),
        helper.make_node("Where", ["cond", "x", "y"], ["out0"]),
    ]
    graph = helper.make_graph(
        nodes, f"test_{op}", vis,
        [helper.make_tensor_value_info("out0", TensorProto.UNDEFINED, None)],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)], ir_version=10)
    return onnx.shape_inference.infer_shapes(model, strict_mode=True)


@pytest.mark.parametrize("fmt", ["mlpackage", "mlmodel"])
@pytest.mark.parametrize("op", ["GreaterOrEqual", "LessOrEqual"])
def test_compare(fmt, op):
    shape = (2, 4)
    ins = {n: _SEED.integers(0, 3, size=shape).astype(np.float32) for n in ("g0", "g1")}
    ins["x"] = _SEED.random(shape).astype(np.float32)
    ins["y"] = _SEED.random(shape).astype(np.float32)
    assert_parity(_compare_where_model(op, shape), ins, fmt=fmt)


def _logical_where_model(op, n_inputs, shape) -> onnx.ModelProto:
    fp = TensorProto.FLOAT
    inputs, nodes, conds = [], [], []
    for i in range(n_inputs):
        inputs += [f"a{i}", f"b{i}"]
        nodes.append(helper.make_node("Greater", [f"a{i}", f"b{i}"], [f"c{i}"]))
        conds.append(f"c{i}")
    nodes.append(helper.make_node(op, conds, ["cond"]))
    inputs += ["x", "y"]
    nodes.append(helper.make_node("Where", ["cond", "x", "y"], ["out0"]))
    vis = [helper.make_tensor_value_info(n, fp, shape) for n in inputs]
    graph = helper.make_graph(
        nodes, f"test_{op}", vis,
        [helper.make_tensor_value_info("out0", TensorProto.UNDEFINED, None)],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)], ir_version=10)
    return onnx.shape_inference.infer_shapes(model, strict_mode=True)


@pytest.mark.parametrize("fmt", ["mlpackage", "mlmodel"])
@pytest.mark.parametrize(("op", "n"), [("And", 2), ("Or", 2), ("Xor", 2), ("Not", 1)])
def test_logical(fmt, op, n):
    shape = (2, 4)
    ins = {}
    for i in range(n):
        ins[f"a{i}"] = _SEED.integers(0, 3, size=shape).astype(np.float32)
        ins[f"b{i}"] = _SEED.integers(0, 3, size=shape).astype(np.float32)
    ins["x"] = _SEED.random(shape).astype(np.float32)
    ins["y"] = _SEED.random(shape).astype(np.float32)
    assert_parity(_logical_where_model(op, n, shape), ins, fmt=fmt)


@pytest.mark.parametrize("fmt", ["mlpackage", "mlmodel"])
def test_isnan(fmt):
    shape = (2, 4)
    x = _SEED.random(shape).astype(np.float32)
    x.flat[0] = np.nan
    x.flat[5] = np.nan
    a = _SEED.random(shape).astype(np.float32)
    b = _SEED.random(shape).astype(np.float32)
    fp = TensorProto.FLOAT
    nodes = [
        helper.make_node("IsNaN", ["x"], ["cond"]),
        helper.make_node("Where", ["cond", "a", "b"], ["out0"]),
    ]
    vis = [helper.make_tensor_value_info(n, fp, shape) for n in ["x", "a", "b"]]
    graph = helper.make_graph(
        nodes, "test_IsNaN", vis,
        [helper.make_tensor_value_info("out0", TensorProto.UNDEFINED, None)],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)], ir_version=10)
    assert_parity(onnx.shape_inference.infer_shapes(model, strict_mode=True),
                  {"x": x, "a": a, "b": b}, fmt=fmt)


@pytest.mark.parametrize("fmt", ["mlpackage", "mlmodel"])
def test_clip_int(fmt):
    x = _SEED.integers(-10, 10, size=(2, 4)).astype(np.int32)
    model = single_op_model(
        "Clip", {"x": x},
        initializers={"lo": np.array(-3, np.int32), "hi": np.array(4, np.int32)},
    )
    assert_parity(model, {"x": x}, fmt=fmt)
