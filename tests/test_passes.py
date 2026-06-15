# Copyright 2026 onnx2coreml contributors.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""Tests for the ONNX preprocessing passes (``onnx2coreml._passes``).

Each pass is checked two ways: a structural assertion on the transformed proto
(the node/initializer it should remove is gone, names preserved) and an
end-to-end parity check that ``convert`` still matches ONNX Runtime on both
container formats. The passes run on every conversion, so the parity checks are
the real safety net.
"""

from __future__ import annotations

import numpy as np
import onnx
import pytest
from helpers import assert_parity
from onnx import TensorProto, helper, numpy_helper

from onnx2coreml._passes import _safe, run
from onnx2coreml._passes._cleanup import (
    eliminate_dead_nodes,
    prune_initializers,
    remove_dropout,
    remove_identity,
)
from onnx2coreml._passes._fold import fold_constants
from onnx2coreml._passes._model import normalize_opset

FMTS = ["mlpackage", "mlmodel"]
_RNG = np.random.default_rng(0)
_FP = TensorProto.FLOAT


def _model(nodes, inputs, outputs, *, initializers=None, opset=17) -> onnx.ModelProto:
    """Build and shape-infer a multi-node float model with named graph I/O."""
    graph = helper.make_graph(
        nodes,
        "test_pass",
        [helper.make_tensor_value_info(n, _FP, shp) for n, shp in inputs],
        [helper.make_tensor_value_info(n, _FP, shp) for n, shp in outputs],
        initializer=[numpy_helper.from_array(a, name=n) for n, a in (initializers or {}).items()],
    )
    model = helper.make_model(
        graph, opset_imports=[helper.make_opsetid("", opset)], ir_version=10
    )
    return onnx.shape_inference.infer_shapes(model, strict_mode=True)


def _op_types(model: onnx.ModelProto) -> list[str]:
    return [n.op_type for n in model.graph.node]


def _init_names(model: onnx.ModelProto) -> set[str]:
    return {i.name for i in model.graph.initializer}


# --------------------------------------------------------------------------- #
# remove_identity
# --------------------------------------------------------------------------- #


def _identity_model() -> tuple[onnx.ModelProto, dict[str, np.ndarray]]:
    # x -> Identity -> t -> Relu -> out0  (the Identity is interior)
    nodes = [
        helper.make_node("Identity", ["x"], ["t"]),
        helper.make_node("Relu", ["t"], ["out0"]),
    ]
    m = _model(nodes, [("x", [2, 3])], [("out0", [2, 3])])
    return m, {"x": (_RNG.random((2, 3)).astype(np.float32) - 0.5) * 4.0}


def test_remove_identity_drops_node():
    m, _ = _identity_model()
    out = remove_identity(m)
    assert "Identity" not in _op_types(out)
    assert [o.name for o in out.graph.output] == ["out0"]  # output name preserved


def test_run_eliminates_identity():
    m, _ = _identity_model()
    assert "Identity" not in _op_types(run(m))


@pytest.mark.parametrize("fmt", FMTS)
def test_remove_identity_parity(fmt):
    m, ins = _identity_model()
    assert_parity(m, ins, fmt=fmt)


def test_remove_identity_at_graph_output():
    # x -> Relu -> r -> Identity -> out0 : Identity output IS the graph output,
    # so the Relu producer is renamed onto "out0" and the Identity dropped.
    nodes = [
        helper.make_node("Relu", ["x"], ["r"]),
        helper.make_node("Identity", ["r"], ["out0"]),
    ]
    m = _model(nodes, [("x", [2, 3])], [("out0", [2, 3])])
    out = remove_identity(m)
    assert "Identity" not in _op_types(out)
    assert [o.name for o in out.graph.output] == ["out0"]


# --------------------------------------------------------------------------- #
# remove_dropout
# --------------------------------------------------------------------------- #


def _dropout_model() -> tuple[onnx.ModelProto, dict[str, np.ndarray]]:
    # x -> Relu -> r -> Dropout -> out0. Dropout is unsupported by lowering, so
    # the pass must remove it for the conversion to even reach coverage.
    nodes = [
        helper.make_node("Relu", ["x"], ["r"]),
        helper.make_node("Dropout", ["r", "ratio"], ["out0"]),
    ]
    m = _model(
        nodes,
        [("x", [2, 3])],
        [("out0", [2, 3])],
        initializers={"ratio": np.array(0.0, dtype=np.float32)},
    )
    return m, {"x": (_RNG.random((2, 3)).astype(np.float32) - 0.5) * 4.0}


def test_remove_dropout_drops_node():
    m, _ = _dropout_model()
    out = remove_dropout(m)
    assert "Dropout" not in _op_types(out)
    assert [o.name for o in out.graph.output] == ["out0"]


@pytest.mark.parametrize("fmt", FMTS)
def test_remove_dropout_parity(fmt):
    m, ins = _dropout_model()
    # Without the pass this would fail the coverage gate; parity proves it both
    # converts and matches ONNX Runtime (Dropout is identity at inference).
    assert_parity(m, ins, fmt=fmt)


def test_remove_dropout_interior():
    # x -> Dropout -> d -> Relu -> out0 (Dropout interior, single data output).
    nodes = [
        helper.make_node("Dropout", ["x"], ["d"]),
        helper.make_node("Relu", ["d"], ["out0"]),
    ]
    m = _model(nodes, [("x", [2, 3])], [("out0", [2, 3])])
    out = remove_dropout(m)
    assert _op_types(out) == ["Relu"]


def test_remove_dropout_keeps_when_mask_used():
    # If the mask output is consumed, the node is not safe to drop.
    nodes = [
        helper.make_node("Dropout", ["x"], ["d", "mask"]),
        helper.make_node("Relu", ["d"], ["out0"]),
        helper.make_node("Cast", ["mask"], ["out1"], to=TensorProto.FLOAT),
    ]
    m = _model(nodes, [("x", [2, 3])], [("out0", [2, 3]), ("out1", [2, 3])])
    out = remove_dropout(m)
    assert "Dropout" in _op_types(out)


# --------------------------------------------------------------------------- #
# fold_constants
# --------------------------------------------------------------------------- #


def _fold_model() -> tuple[onnx.ModelProto, dict[str, np.ndarray]]:
    # Add(c0, c1) -> s, then Sub(s, c2) -> k feed a real Mul(x, k). The two-deep
    # constant chain should collapse entirely, leaving only the Mul.
    c0 = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
    c1 = np.array([[4.0, 5.0, 6.0]], dtype=np.float32)
    c2 = np.array([[0.5, 0.5, 0.5]], dtype=np.float32)
    nodes = [
        helper.make_node("Add", ["c0", "c1"], ["s"]),
        helper.make_node("Sub", ["s", "c2"], ["k"]),
        helper.make_node("Mul", ["x", "k"], ["out0"]),
    ]
    m = _model(
        nodes,
        [("x", [2, 3])],
        [("out0", [2, 3])],
        initializers={"c0": c0, "c1": c1, "c2": c2},
    )
    return m, {"x": _RNG.random((2, 3)).astype(np.float32)}


def test_fold_constants_collapses_chain():
    m, _ = _fold_model()
    out = fold_constants(m)
    assert _op_types(out) == ["Mul"]  # Add and Sub folded away
    assert "k" in _init_names(out)  # the folded result is now an initializer


@pytest.mark.parametrize("fmt", FMTS)
def test_fold_constants_parity(fmt):
    m, ins = _fold_model()
    assert_parity(m, ins, fmt=fmt)


def test_fold_then_prune_removes_orphans():
    # After folding, the intermediate constants c0/c1/c2/s are unreferenced and
    # should be pruned; only the final folded initializer "k" remains.
    m, _ = _fold_model()
    out = prune_initializers(fold_constants(m))
    assert _init_names(out) == {"k"}


# --------------------------------------------------------------------------- #
# eliminate_dead_nodes / prune_initializers
# --------------------------------------------------------------------------- #


def test_eliminate_dead_nodes():
    # "dead" (Neg) feeds nothing and is not a graph output -> removed.
    nodes = [
        helper.make_node("Relu", ["x"], ["out0"]),
        helper.make_node("Neg", ["x"], ["dead"]),
    ]
    m = _model(nodes, [("x", [2, 3])], [("out0", [2, 3])])
    out = eliminate_dead_nodes(m)
    assert _op_types(out) == ["Relu"]


def test_eliminate_dead_nodes_chain():
    # dead0 -> dead1, neither observed -> both collected at the fixpoint.
    nodes = [
        helper.make_node("Relu", ["x"], ["out0"]),
        helper.make_node("Neg", ["x"], ["dead0"]),
        helper.make_node("Abs", ["dead0"], ["dead1"]),
    ]
    m = _model(nodes, [("x", [2, 3])], [("out0", [2, 3])])
    assert _op_types(eliminate_dead_nodes(m)) == ["Relu"]


def test_prune_initializers_drops_unused():
    used = np.array([[1.0, 1.0, 1.0]], dtype=np.float32)
    unused = np.array([[9.0, 9.0, 9.0]], dtype=np.float32)
    nodes = [helper.make_node("Add", ["x", "used"], ["out0"])]
    m = _model(
        nodes,
        [("x", [2, 3])],
        [("out0", [2, 3])],
        initializers={"used": used, "unused": unused},
    )
    out = prune_initializers(m)
    assert _init_names(out) == {"used"}


# --------------------------------------------------------------------------- #
# defensiveness
# --------------------------------------------------------------------------- #


def test_safe_swallows_pass_failure():
    sentinel = object()

    def boom(_model):
        raise RuntimeError("pass blew up")

    assert _safe(boom, sentinel) is sentinel  # original returned unchanged


def test_normalize_opset_no_downgrade():
    # Opset 17 is already >= baseline 13; the pass must leave it untouched
    # (version_converter has no down-adapters and would otherwise raise).
    m, _ = _identity_model()
    out = normalize_opset(m, baseline=13)
    assert out is m
    versions = {op.domain: op.version for op in out.opset_import}
    assert versions[""] == 17


def test_run_is_identity_on_clean_model():
    # A graph with nothing to fold/clean must come back semantically intact:
    # same op set, same I/O names.
    nodes = [helper.make_node("Relu", ["x"], ["out0"])]
    m = _model(nodes, [("x", [2, 3])], [("out0", [2, 3])])
    out = run(m)
    assert _op_types(out) == ["Relu"]
    assert [i.name for i in out.graph.input] == ["x"]
    assert [o.name for o in out.graph.output] == ["out0"]
