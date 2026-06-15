# Copyright 2026 onnx2coreml contributors.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""Graph cleanup passes: drop no-op nodes and unreferenced tensors.

Each pass is semantics-preserving in inference and operates on a copy of the
proto. Graph input/output *names* are never renamed: where a removable node
feeds a graph output directly, the node's producer is renamed onto the output
name instead, so the converter (which looks up outputs by name) still resolves.
"""

from __future__ import annotations

import onnx

_SUBGRAPH_ATTR_TYPES = (onnx.AttributeProto.GRAPH, onnx.AttributeProto.GRAPHS)


def _has_subgraph(node: onnx.NodeProto) -> bool:
    return any(attr.type in _SUBGRAPH_ATTR_TYPES for attr in node.attribute)


def _copy(model: onnx.ModelProto) -> onnx.ModelProto:
    clone = onnx.ModelProto()
    clone.CopyFrom(model)
    return clone


def _rewire_consumer_inputs(graph: onnx.GraphProto, old: str, new: str) -> None:
    """Point every node input referencing ``old`` at ``new`` instead."""
    for node in graph.node:
        for i, name in enumerate(node.input):
            if name == old:
                node.input[i] = new


def _consumers(graph: onnx.GraphProto, name: str) -> int:
    """Count node inputs (across the graph) that reference ``name``."""
    return sum(inp == name for node in graph.node for inp in node.input)


def _try_drop_passthrough(graph: onnx.GraphProto, node: onnx.NodeProto) -> bool:
    """Remove a single-input/single-relevant-output passthrough node (Identity,
    inference Dropout), rewiring its consumers. Returns whether it was removed.

    Two cases:
      * the output is interior -> rewire consumers from output to input;
      * the output is a graph output -> rename the input's *producer* onto the
        output name, but only when that is unambiguous and side-effect-free.
    Anything ambiguous is left untouched, keeping the pass conservative.
    """
    src = node.input[0]
    dst = node.output[0]
    if not src or not dst:
        return False

    graph_outputs = {o.name for o in graph.output}
    if dst not in graph_outputs:
        _rewire_consumer_inputs(graph, dst, src)
        graph.node.remove(node)
        return True

    # dst is a graph output: we must keep the name `dst`. Try to make the
    # producer of `src` emit `dst` directly. Only safe when `src` is an interior
    # tensor produced by exactly one other node and consumed only by this node.
    graph_inputs = {i.name for i in graph.input}
    initializers = {i.name for i in graph.initializer}
    if src in graph_inputs or src in initializers or src in graph_outputs:
        return False
    if _consumers(graph, src) != 1:  # only this passthrough consumes src
        return False
    producers = [n for n in graph.node if src in n.output and n is not node]
    if len(producers) != 1:
        return False

    producer = producers[0]
    for i, name in enumerate(producer.output):
        if name == src:
            producer.output[i] = dst
    graph.node.remove(node)
    return True


def remove_identity(model: onnx.ModelProto) -> onnx.ModelProto:
    """Drop ``Identity`` nodes, rewiring their consumers to the input tensor."""
    out = _copy(model)
    graph = out.graph
    changed = False
    for node in list(graph.node):
        if node.op_type != "Identity" or node.domain not in ("", "ai.onnx"):
            continue
        if _has_subgraph(node) or len(node.input) < 1 or len(node.output) < 1:
            continue
        if _try_drop_passthrough(graph, node):
            changed = True
    return out if changed else model


def remove_dropout(model: onnx.ModelProto) -> onnx.ModelProto:
    """Drop inference ``Dropout`` nodes (input passes through unchanged).

    Only removed when the optional mask output is unused, since the mask is not
    reproduced. The ``training_mode`` / ``ratio`` inputs do not affect the data
    output at inference, so they are ignored.
    """
    out = _copy(model)
    graph = out.graph
    graph_outputs = {o.name for o in graph.output}
    changed = False
    for node in list(graph.node):
        if node.op_type != "Dropout" or node.domain not in ("", "ai.onnx"):
            continue
        if _has_subgraph(node) or len(node.input) < 1 or len(node.output) < 1:
            continue
        # Bail if the mask (second output) is observed anywhere.
        mask = node.output[1] if len(node.output) > 1 else ""
        if mask and (mask in graph_outputs or _consumers(graph, mask)):
            continue
        if _try_drop_passthrough(graph, node):
            changed = True
    return out if changed else model


def eliminate_dead_nodes(model: onnx.ModelProto) -> onnx.ModelProto:
    """Remove nodes whose outputs feed nothing and are not graph outputs.

    Iterates to a fixpoint so chains of now-dead producers are also collected.
    Nodes with subgraph bodies are conservatively kept (their effects are not
    captured by the simple use-count below).
    """
    out = _copy(model)
    graph = out.graph
    graph_outputs = {o.name for o in graph.output}
    changed = False
    progressed = True
    while progressed:
        progressed = False
        consumed: set[str] = {inp for node in graph.node for inp in node.input}
        for node in list(graph.node):
            if _has_subgraph(node):
                continue
            if any(o in graph_outputs or o in consumed for o in node.output):
                continue
            graph.node.remove(node)
            progressed = True
            changed = True
    return out if changed else model


def prune_initializers(model: onnx.ModelProto) -> onnx.ModelProto:
    """Drop initializers that no node references.

    Initializers that double as declared graph inputs are kept so the graph's
    input list stays consistent with its initializer list.
    """
    out = _copy(model)
    graph = out.graph
    referenced: set[str] = {inp for node in graph.node for inp in node.input}
    referenced.update(i.name for i in graph.input)
    referenced.update(o.name for o in graph.output)

    keep = [init for init in graph.initializer if init.name in referenced]
    if len(keep) == len(graph.initializer):
        return model
    del graph.initializer[:]
    graph.initializer.extend(keep)
    return out
