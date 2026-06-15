# Copyright 2026 onnx2coreml contributors.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""Recurrent-layer lowerings (LSTM).

The fiddly part is reconciling ONNX's and MIL's weight conventions, all of which
we can do once on the (constant) weights at convert time:

* **Gate order.** ONNX packs gates as ``[input, output, forget, cell]``; MIL wants
  ``[input, forget, output, cell]``. :func:`_reorder_gates` permutes the 4 stacked
  ``H``-row blocks accordingly.
* **Bias.** ONNX carries input-hidden and hidden-hidden biases separately (a single
  ``8H`` vector); MIL takes their sum (``4H``).
* **State / output layout.** ONNX uses a ``num_directions`` axis (states
  ``[num_dir, b, H]``, output ``[s, num_dir, b, H]``); MIL folds direction into the
  hidden axis (states ``[b, num_dir*H]``, output ``[s, b, num_dir*H]``), so states
  are transposed/reshaped in and the output back out.

Genuinely unsupported ONNX features raise rather than silently mismatch:
``sequence_lens`` (variable lengths), peephole connections, ``input_forget``,
``layout=1`` (batch-major), and non-default activations.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import onnx

from .._mil import mb
from ._common import const_array, get_attr
from ._context import Lowering, LoweringContext

# ONNX default LSTM activations are (f, g, h) = (Sigmoid, Tanh, Tanh), which map
# onto MIL's recurrent / cell / output activations. Only this default is emitted.
_DEFAULT_ACTIVATIONS = (b"Sigmoid", b"Tanh", b"Tanh")


def _reorder_gates(arr: np.ndarray, hidden: int) -> np.ndarray:
    """Permute stacked gate blocks from ONNX ``[i, o, f, c]`` to MIL ``[i, f, o, c]``.

    ``arr`` has its 4 gate blocks stacked on axis 0 (shape ``(4H, ...)``).
    """
    i, o, f, c = (arr[g * hidden : (g + 1) * hidden] for g in range(4))
    return np.concatenate([i, f, o, c], axis=0)


def _direction_bias(bias_8h: np.ndarray, hidden: int) -> np.ndarray:
    """ONNX per-direction bias ``[Wb(4H) ; Rb(4H)]`` -> MIL summed, reordered ``4H``."""
    return _reorder_gates(bias_8h[: 4 * hidden] + bias_8h[4 * hidden :], hidden)


def _state(ctx: LoweringContext, name: str, batch: int, num_dir: int, hidden: int) -> Any:
    """Initial hidden/cell state as MIL ``[b, num_dir*H]``.

    Absent (or empty) ONNX state -> zeros; otherwise transpose/reshape the ONNX
    ``[num_dir, b, H]`` state into MIL's direction-folded layout.
    """
    if not name:
        return mb.const(val=np.zeros((batch, num_dir * hidden), dtype=np.float32))
    var = ctx.values_map[name]
    transposed = mb.transpose(x=var, perm=[1, 0, 2])  # [b, num_dir, H]
    return mb.reshape(x=transposed, shape=[batch, num_dir * hidden])


def _lstm(ctx: LoweringContext, node: onnx.NodeProto) -> list[Any]:
    x = ctx.values_map[node.input[0]]  # [seq, batch, input]
    weight = const_array(ctx, node, 1)  # [num_dir, 4H, I], gate order i,o,f,c
    recurrence = const_array(ctx, node, 2)  # [num_dir, 4H, H]
    if weight is None or recurrence is None:
        raise ValueError("LSTM requires constant W and R")
    bias = const_array(ctx, node, 3)  # [num_dir, 8H] or None

    def _present(idx: int) -> str:
        return node.input[idx] if len(node.input) > idx and node.input[idx] else ""

    if _present(4):
        raise ValueError("LSTM with sequence_lens is not supported")
    if _present(7):
        raise ValueError("LSTM with peephole connections is not supported")
    if get_attr(node, "input_forget", 0):
        raise ValueError("LSTM with input_forget=1 is not supported")
    if int(get_attr(node, "layout", 0)) != 0:
        raise ValueError("LSTM with layout=1 (batch-major) is not supported")
    activations = get_attr(node, "activations")
    if activations is not None and tuple(activations) != _DEFAULT_ACTIVATIONS:
        raise ValueError("LSTM with non-default activations is not supported")

    direction = get_attr(node, "direction", b"forward")
    direction = direction.decode() if isinstance(direction, bytes) else direction
    hidden = int(get_attr(node, "hidden_size"))
    num_dir = weight.shape[0]
    seq, batch = int(x.shape[0]), int(x.shape[1])

    kwargs: dict[str, Any] = {
        "x": x,
        "initial_h": _state(ctx, _present(5), batch, num_dir, hidden),
        "initial_c": _state(ctx, _present(6), batch, num_dir, hidden),
        "weight_ih": _reorder_gates(weight[0], hidden),
        "weight_hh": _reorder_gates(recurrence[0], hidden),
        "direction": direction,
        "output_sequence": True,
        "recurrent_activation": "sigmoid",
        "cell_activation": "tanh",
        "activation": "tanh",
    }
    if bias is not None:
        kwargs["bias"] = _direction_bias(bias[0], hidden)
    if direction == "bidirectional":
        kwargs["weight_ih_back"] = _reorder_gates(weight[1], hidden)
        kwargs["weight_hh_back"] = _reorder_gates(recurrence[1], hidden)
        if bias is not None:
            kwargs["bias_back"] = _direction_bias(bias[1], hidden)
    clip = get_attr(node, "clip")
    if clip is not None:
        kwargs["clip"] = float(clip)

    y, y_h, y_c = mb.lstm(**kwargs)

    # Re-expand MIL's direction-folded outputs back to ONNX's num_directions axis.
    outputs: list[Any] = []
    for i, out_name in enumerate(node.output):
        name = {"name": out_name} if out_name else {}
        if i == 0:  # Y: [s, b, num_dir*H] -> [s, num_dir, b, H]
            y4 = mb.reshape(x=y, shape=[seq, batch, num_dir, hidden])
            if direction == "reverse":
                # MIL's standalone reverse emits the sequence in computed (reversed)
                # time order; ONNX keeps original time order. Flip the time axis.
                transposed = mb.transpose(x=y4, perm=[0, 2, 1, 3])
                outputs.append(mb.reverse(x=transposed, axes=[0], **name))
            else:
                outputs.append(mb.transpose(x=y4, perm=[0, 2, 1, 3], **name))
        else:  # Y_h / Y_c: [b, num_dir*H] -> [num_dir, b, H]
            state = y_h if i == 1 else y_c
            s3 = mb.reshape(x=state, shape=[batch, num_dir, hidden])
            outputs.append(mb.transpose(x=s3, perm=[1, 0, 2], **name))
    return outputs


REGISTRY: dict[str, Lowering] = {"LSTM": _lstm}

__all__ = ["REGISTRY"]
