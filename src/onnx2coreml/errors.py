# Copyright 2026 onnx2coreml contributors.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""Structured exception hierarchy for onnx2coreml.

All converter errors derive from :class:`Onnx2CoreMLError`, so callers can catch
the whole family with one ``except``. Each subclass carries enough structured
context to act on the failure without parsing the message string.
"""

from __future__ import annotations

__all__ = [
    "ConversionError",
    "ModelValidationError",
    "Onnx2CoreMLError",
    "TargetError",
    "UnsupportedOpError",
]


class Onnx2CoreMLError(Exception):
    """Base class for every error raised by onnx2coreml."""


class UnsupportedOpError(Onnx2CoreMLError):
    """Raised by the coverage gate before any MIL is emitted.

    Aggregates *all* ops with no lowering into a single report rather than
    failing on the first one, so the user sees the full gap in one pass.

    Parameters
    ----------
    missing:
        Maps each unsupported op key (``"OpType"`` or ``"domain::OpType"``) to a
        list of example node names that use it.
    """

    def __init__(self, missing: dict[str, list[str]]) -> None:
        self.missing = missing
        lines = ["The following ONNX ops have no Core ML lowering:"]
        for key in sorted(missing):
            nodes = missing[key]
            sample = ", ".join(nodes[:3])
            more = f", ... (+{len(nodes) - 3} more)" if len(nodes) > 3 else ""
            lines.append(f"  {key} ({len(nodes)} node(s), e.g. {sample}{more})")
        lines.append(
            "\nRegister a custom lowering to proceed:\n"
            '    @converter.register("OpType")\n'
            "    def lower(ctx, node): ...\n"
            "Run `onnx2coreml inspect <model>` for the full coverage report."
        )
        super().__init__("\n".join(lines))


class ConversionError(Onnx2CoreMLError):
    """A lowering failed for a specific node.

    Wraps the underlying exception with the node name and op key so the failure
    points at a concrete place in the graph.
    """

    def __init__(self, node_name: str, op_key: str, cause: BaseException) -> None:
        self.node_name = node_name
        self.op_key = op_key
        self.cause = cause
        super().__init__(
            f"Failed to lower node '{node_name}' (op {op_key}): {cause}"
        )


class ModelValidationError(Onnx2CoreMLError):
    """The input ONNX model failed to load, check, or run.

    Raised before conversion begins, so a malformed or non-runnable model is
    attributed to the input rather than surfacing later as an opaque conversion
    failure.
    """


class TargetError(Onnx2CoreMLError):
    """The requested deployment target or format cannot express the model.

    Raised when an op requires a higher minimum deployment target than
    requested, or when the requested format (e.g. NeuralNetwork / .mlmodel)
    cannot represent an op that is only available in ML Program / .mlpackage.
    """
