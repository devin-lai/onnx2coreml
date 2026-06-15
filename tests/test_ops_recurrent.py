# Copyright 2026 onnx2coreml contributors.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""Numerical-parity tests for recurrent lowerings (LSTM)."""

from __future__ import annotations

import numpy as np
import onnx
import pytest
from helpers import assert_parity
from onnx import TensorProto, helper, numpy_helper

pytestmark = pytest.mark.ops

_FMTS = ["mlpackage", "mlmodel"]


def _lstm_model(direction: str, *, n_outputs: int = 1, seq=4, batch=2, inp=3, hidden=5):
    num_dir = 2 if direction == "bidirectional" else 1
    rng = np.random.default_rng(abs(hash(direction)) % 1000 + n_outputs)
    x = rng.standard_normal((seq, batch, inp)).astype(np.float32)
    w = (rng.standard_normal((num_dir, 4 * hidden, inp)) * 0.3).astype(np.float32)
    r = (rng.standard_normal((num_dir, 4 * hidden, hidden)) * 0.3).astype(np.float32)
    b = (rng.standard_normal((num_dir, 8 * hidden)) * 0.1).astype(np.float32)
    out_names = ["Y", "Y_h", "Y_c"][:n_outputs]
    node = helper.make_node(
        "LSTM", ["X", "W", "R", "B"], out_names, direction=direction, hidden_size=hidden
    )
    graph = helper.make_graph(
        [node], "lstm",
        [helper.make_tensor_value_info("X", TensorProto.FLOAT, x.shape)],
        [helper.make_tensor_value_info(n, TensorProto.UNDEFINED, None) for n in out_names],
        initializer=[numpy_helper.from_array(a, n) for n, a in (("W", w), ("R", r), ("B", b))],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)], ir_version=10)
    return onnx.shape_inference.infer_shapes(model), {"X": x}


@pytest.mark.parametrize("fmt", _FMTS)
@pytest.mark.parametrize("direction", ["forward", "reverse", "bidirectional"])
def test_lstm_direction(fmt, direction):
    model, ins = _lstm_model(direction)
    assert_parity(model, ins, fmt=fmt)


@pytest.mark.parametrize("fmt", _FMTS)
def test_lstm_all_three_outputs(fmt):
    # Y, Y_h, Y_c together — exercises the state output reshapes.
    model, ins = _lstm_model("bidirectional", n_outputs=3)
    assert_parity(model, ins, fmt=fmt)
