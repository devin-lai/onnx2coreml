# Copyright 2026 onnx2coreml contributors.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""Graph-walking and node-introspection helpers shared across lowerings."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import onnx
from onnx import helper

_DEFAULT_DOMAINS = {"", "ai.onnx", "ai.onnx.ml"}


def op_key(node: onnx.NodeProto) -> str:
    """Registry key for a node: ``OpType`` for the default domain, else
    ``domain::OpType``."""
    if node.domain in _DEFAULT_DOMAINS:
        return node.op_type
    return f"{node.domain}::{node.op_type}"


def get_attr(node: onnx.NodeProto, name: str, default: Any = None) -> Any:
    """Return a node attribute's Python value, or ``default`` if absent."""
    for attr in node.attribute:
        if attr.name == name:
            return helper.get_attribute_value(attr)
    return default


def operands(
    values_map: dict[str, Any], node: onnx.NodeProto, idxs: list[int]
) -> list[Any]:
    """Fetch the MIL vars for the given input indices.

    An omitted optional input (empty name or past the end) yields ``None``.
    """
    out: list[Any] = []
    for i in idxs:
        if i < len(node.input) and node.input[i]:
            out.append(values_map[node.input[i]])
        else:
            out.append(None)
    return out


def topo_iter(graph: onnx.GraphProto) -> Iterator[onnx.NodeProto]:
    """Yield nodes in dependency order.

    ONNX graphs are usually already topologically sorted, but the spec does not
    guarantee it, so we sort defensively (Kahn's algorithm by value availability).
    """
    available: set[str] = {""}  # empty name = omitted optional input
    available.update(i.name for i in graph.input)
    available.update(i.name for i in graph.initializer)

    pending = list(graph.node)
    progressed = True
    while pending and progressed:
        progressed = False
        still: list[onnx.NodeProto] = []
        for node in pending:
            if all(name in available for name in node.input):
                yield node
                available.update(node.output)
                progressed = True
            else:
                still.append(node)
        pending = still
    # Any remaining nodes have unresolved inputs (e.g. produced later / cycles);
    # emit them in original order so conversion still surfaces a useful error.
    yield from pending


def iter_graph_nodes(graph: onnx.GraphProto) -> Iterator[onnx.NodeProto]:
    """Yield every node in the graph and, recursively, in subgraph attributes
    (If/Loop/Scan bodies) — used by the coverage gate."""
    for node in graph.node:
        yield node
        for attr in node.attribute:
            if attr.type == onnx.AttributeProto.GRAPH:
                yield from iter_graph_nodes(attr.g)
            elif attr.type == onnx.AttributeProto.GRAPHS:
                for g in attr.graphs:
                    yield from iter_graph_nodes(g)
