# Copyright 2026 onnx2coreml contributors.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""A tiny, conservative subgraph matcher/rewriter over ``onnx.GraphProto``.

The only shape it recognizes is a *linear chain*: a run of nodes
``n[0] -> n[1] -> ... -> n[k-1]`` where each node's first output is the first
producer feeding the next node, the chain's interior values are private (not a
graph output and consumed by no node outside the chain), and the op_types match
a requested sequence. That is exactly enough to fold a decomposed-attention
block into one node without risking an unrelated rewrite.

``replace_chain`` splices a single replacement node in for the whole run and
rewires edges: the replacement consumes ``inputs`` and produces the chain's
final output name, every matched node is dropped, and graph outputs are left
untouched (the final name is preserved, so downstream consumers are unaffected).
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import onnx

from .._utils import op_key


@dataclass(frozen=True)
class ChainMatch:
    """A matched linear chain of nodes (in chain order)."""

    nodes: tuple[onnx.NodeProto, ...]

    @property
    def output(self) -> str:
        """The value name produced by the last node of the chain."""
        return self.nodes[-1].output[0]


def _consumers(graph: onnx.GraphProto) -> dict[str, list[onnx.NodeProto]]:
    """Map each value name to the list of nodes that take it as an input."""
    out: dict[str, list[onnx.NodeProto]] = {}
    for node in graph.node:
        for name in node.input:
            if name:
                out.setdefault(name, []).append(node)
    return out


def find_chains(graph: onnx.GraphProto, op_types: list[str]) -> list[ChainMatch]:
    """Find every linear chain whose op_types equal ``op_types`` in order.

    A chain qualifies only when it is safe to collapse: each interior value
    (the output of every node except the last) is consumed exactly once -- by
    the next node in the chain -- and is not a graph output. This guarantees the
    interior is private to the chain, so removing it changes nothing else.
    """
    if not op_types:
        return []

    consumers = _consumers(graph)
    graph_outputs = {o.name for o in graph.output}

    matches: list[ChainMatch] = []
    for start in graph.node:
        if op_key(start) != op_types[0]:
            continue
        chain: list[onnx.NodeProto] = [start]
        ok = True
        for want in op_types[1:]:
            prev = chain[-1]
            feed = prev.output[0]
            # Interior value must be private: exactly one consumer (the next
            # chain node) and not exposed as a graph output.
            users = consumers.get(feed, [])
            if not feed or feed in graph_outputs or len(users) != 1:
                ok = False
                break
            nxt = users[0]
            if op_key(nxt) != want:
                ok = False
                break
            chain.append(nxt)
        if ok and len(chain) == len(op_types):
            matches.append(ChainMatch(tuple(chain)))
    return matches


def replace_chain(
    graph: onnx.GraphProto,
    match: ChainMatch,
    replacement: onnx.NodeProto,
    *,
    also_remove: Iterable[onnx.NodeProto] = (),
) -> None:
    """Replace ``match``'s nodes (plus any ``also_remove`` side nodes) with
    ``replacement`` in a single rebuild.

    ``replacement`` must already produce ``match.output`` so downstream consumers
    and any graph output binding keep working unchanged. The replacement is
    inserted where the chain began, preserving relative node order.

    All node identification happens before the list is rebuilt: protobuf
    ``extend`` copies messages, so any node reference taken afterwards would point
    at a stale object. Callers must therefore remove every node in one call.
    """
    if replacement.output[0] != match.output:
        raise ValueError(
            "replacement must produce the chain's final output to preserve edges"
        )
    drop = set(map(id, match.nodes)) | set(map(id, also_remove))
    insert_at = next(i for i, n in enumerate(graph.node) if id(n) in drop)
    kept = [n for n in graph.node if id(n) not in drop]
    kept.insert(insert_at, replacement)
    del graph.node[:]
    graph.node.extend(kept)
