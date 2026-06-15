# Copyright 2026 onnx2coreml contributors.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""Fuse a decomposed scaled-dot-product-attention block into one node.

The canonical decomposition this recognizes is::

    K_t    = Transpose(K)                 # swaps the last two axes
    scores = MatMul(Q, K_t)
    scaled = Mul(scores, c)  |  Div(scores, c)   # c is a convert-time constant
    probs  = Softmax(scaled, axis=-1)
    out    = MatMul(probs, V)

When (and only when) the full pattern is present, the five nodes collapse into a
single ``ScaledDotProductAttention`` node with inputs ``[Q, K, V]`` and a float
``scale`` attribute equal to the effective multiplier (``c`` for ``Mul``,
``1/c`` for ``Div``). The fused node's output name is the final ``MatMul``'s, so
downstream edges and graph outputs are preserved.

The rewrite is intentionally strict: a mismatch on any leg (transpose perm, a
non-constant or non-scalar scale, a softmax axis that is not the last) leaves the
graph untouched. The matching lowering in ``_lowering/_attention.py`` re-expands
the node into the same MIL primitives, so the fusion is semantics-preserving.
"""

from __future__ import annotations

import onnx
from onnx import helper, numpy_helper

from .._utils import get_attr, op_key
from ._rewrite import ChainMatch, find_chains, replace_chain

# The interior op_type sequences we collapse. The K-transpose sits *beside* the
# first MatMul (a side input), so it is validated separately, not as a chain link.
_CHAINS = (
    ["MatMul", "Mul", "Softmax", "MatMul"],
    ["MatMul", "Div", "Softmax", "MatMul"],
)


def _scalar_initializer(graph: onnx.GraphProto, name: str) -> float | None:
    """Return ``name``'s value as a Python float if it is a scalar initializer."""
    for init in graph.initializer:
        if init.name == name:
            arr = numpy_helper.to_array(init)
            if arr.size != 1:
                return None
            return float(arr.reshape(()))
    return None


def _transpose_swaps_last_two(node: onnx.NodeProto, rank: int | None) -> bool:
    """True if ``node`` is a Transpose that swaps exactly the last two axes.

    With no ``perm`` attribute Transpose reverses all axes, which swaps the last
    two only for a rank-2 input; otherwise an explicit identity-but-last-two perm
    is required. ``rank`` is the (static) rank of the transposed tensor when known.
    """
    if op_key(node) != "Transpose":
        return False
    perm = get_attr(node, "perm")
    if perm is None:
        # Default perm reverses all axes -> last-two swap only at rank 2.
        return rank == 2
    perm = list(perm)
    n = len(perm)
    if n < 2:
        return False
    expected = list(range(n))
    expected[-1], expected[-2] = expected[-2], expected[-1]
    return perm == expected


def _value_rank(graph: onnx.GraphProto, name: str) -> int | None:
    """Static rank of value ``name`` from graph value-info/inputs, if recorded."""
    for collection in (graph.value_info, graph.input, graph.output):
        for vi in collection:
            if vi.name == name and vi.type.HasField("tensor_type"):
                shape = vi.type.tensor_type.shape
                if shape.dim:
                    return len(shape.dim)
    return None


def _try_fuse_chain(graph: onnx.GraphProto, match: ChainMatch) -> bool:
    """Validate ``match`` as an attention block and, if valid, rewrite it.

    Returns True iff the graph was modified.
    """
    qk, scale_node, softmax, pv = match.nodes

    # Softmax must reduce the last axis (the scores' key dimension). Only the
    # canonical -1 is accepted; a positive last-axis index is left unfused
    # rather than risk a wrong match (conservative by design).
    if get_attr(softmax, "axis", -1) != -1:
        return False

    # First MatMul: input[0] is Q, input[1] must be Transpose(K) swapping last two.
    if len(qk.input) < 2:
        return False
    q_name, kt_name = qk.input[0], qk.input[1]
    transpose = next(
        (n for n in graph.node if kt_name in n.output and n.output[0] == kt_name),
        None,
    )
    if transpose is None or len(transpose.input) < 1:
        return False
    k_name = transpose.input[0]
    if not _transpose_swaps_last_two(transpose, _value_rank(graph, k_name)):
        return False
    # The transposed-K value must be private to this MatMul: feeding anything
    # else (or being a graph output) means we cannot drop the Transpose.
    if any(
        kt_name in other.input for other in graph.node if other is not qk
    ) or kt_name in {o.name for o in graph.output}:
        return False

    # Scale node: scores op constant. Recover the effective multiplier.
    scores_name = qk.output[0]
    scale = _scale_multiplier(graph, scale_node, scores_name)
    if scale is None:
        return False

    # Second MatMul: input[0] is the softmax probs, input[1] is V.
    if len(pv.input) < 2 or pv.input[0] != softmax.output[0]:
        return False
    v_name = pv.input[1]

    fused = helper.make_node(
        "ScaledDotProductAttention",
        inputs=[q_name, k_name, v_name],
        outputs=[match.output],
        name=(pv.name or "") + "_sdpa",
        scale=float(scale),
    )
    # Collapse the 4-node chain and drop the now-orphaned Transpose in one pass.
    replace_chain(graph, match, fused, also_remove=[transpose])
    return True


def _scale_multiplier(
    graph: onnx.GraphProto, node: onnx.NodeProto, scores_name: str
) -> float | None:
    """Effective multiplier applied to ``scores`` by a Mul/Div ``node``.

    For ``Mul`` the other operand must be a scalar constant ``c`` -> ``c``.
    For ``Div`` the divisor (input[1]) must be a scalar constant ``c`` and the
    dividend must be ``scores`` -> ``1/c``. Anything else is rejected.
    """
    key = op_key(node)
    if key == "Mul":
        if len(node.input) != 2 or scores_name not in node.input:
            return None
        other = node.input[1] if node.input[0] == scores_name else node.input[0]
        return _scalar_initializer(graph, other)
    if key == "Div":
        # Division is not commutative: scores must be the numerator.
        if len(node.input) != 2 or node.input[0] != scores_name:
            return None
        c = _scalar_initializer(graph, node.input[1])
        if c is None or c == 0.0:
            return None
        return 1.0 / c
    return None


def fuse(model: onnx.ModelProto) -> onnx.ModelProto:
    """Collapse every decomposed-attention block found in ``model.graph``.

    Mutates ``model`` in place (and returns it). Chains are re-discovered after
    each successful rewrite so the node-index bookkeeping never goes stale.
    """
    graph = model.graph
    changed = True
    while changed:
        changed = False
        for op_types in _CHAINS:
            for match in find_chains(graph, op_types):
                if _try_fuse_chain(graph, match):
                    changed = True
                    break  # node set moved; rescan from scratch
            if changed:
                break
    return model
