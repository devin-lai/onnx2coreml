# Copyright 2026 onnx2coreml contributors.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""The context object handed to every op lowering.

Kept dependency-light (no coremltools imports) so it can sit at the bottom of
the import graph without cycles.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import onnx


@dataclass
class LoweringContext:
    """State threaded through node lowering.

    Attributes
    ----------
    values_map:
        Maps an ONNX value name to the MIL ``Var`` that produces it.
    target:
        The resolved coremltools deployment target / opset for this conversion.
    converter:
        The owning :class:`~onnx2coreml.converter.Converter`, used by control-flow
        ops to lower subgraphs recursively.
    """

    values_map: dict[str, Any]
    target: Any
    converter: Any


# A lowering takes the context and a node, emits MIL ops, and returns the MIL
# Var (or list of Vars) for the node's output(s).
Lowering = Callable[["LoweringContext", "onnx.NodeProto"], Any]
