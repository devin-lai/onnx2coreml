# Copyright 2026 onnx2coreml contributors.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""Operator-coverage analysis: which ops in a model can be converted."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import onnx

from . import _lowering
from ._utils import iter_graph_nodes, op_key


def supported_ops() -> set[str]:
    """The set of ONNX op keys with a built-in lowering."""
    return _lowering.supported_keys()


@dataclass
class CoverageReport:
    """Per-op convertibility for a model."""

    supported: dict[str, int] = field(default_factory=dict)
    unsupported: dict[str, list[str]] = field(default_factory=dict)

    @property
    def convertible(self) -> bool:
        """True when every op in the model has a lowering."""
        return not self.unsupported


def analyze(model: onnx.ModelProto | str | Path | bytes) -> CoverageReport:
    """Report supported/unsupported ops for a model (incl. subgraphs).

    The same cleanup/fusion passes as conversion are run first, so folded nodes
    do not appear as false unsupported-op hits.
    """
    if not isinstance(model, onnx.ModelProto):
        from ._io import load

        model = load(model)

    from . import _fusion, _passes

    model = _fusion.run(_passes.run(model))
    report = CoverageReport()
    for i, node in enumerate(iter_graph_nodes(model.graph)):
        key = op_key(node)
        if _lowering.resolve(key) is not None:
            report.supported[key] = report.supported.get(key, 0) + 1
        else:
            report.unsupported.setdefault(key, []).append(node.name or f"node_{i}")
    return report
