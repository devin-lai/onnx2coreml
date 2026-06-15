# Copyright 2026 onnx2coreml contributors.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""Output-format, deployment-target, precision, and compute-unit resolution.

Centralizes the mapping from user-facing strings to coremltools enums so adding
a future iOS/macOS target or compute unit is a single edit here.
"""

from __future__ import annotations

from enum import Enum

from ._mil import ComputeUnit, ct, precision
from .errors import TargetError


class Format(Enum):
    """Core ML output container / model type."""

    MLPACKAGE = "mlpackage"  # ML Program (MIL) — primary
    MLMODEL = "mlmodel"  # NeuralNetwork — secondary

    @classmethod
    def from_str(cls, value: str | Format) -> Format:
        if isinstance(value, cls):
            return value
        try:
            return cls(str(value).lower())
        except ValueError:
            raise TargetError(
                f"unknown format {value!r}; expected 'mlpackage' or 'mlmodel'"
            ) from None


# String -> coremltools deployment target. Extend as Apple ships new targets.
_TARGETS = {
    "ios15": ct.target.iOS15,
    "ios16": ct.target.iOS16,
    "ios17": ct.target.iOS17,
    "ios18": ct.target.iOS18,
    "macos12": ct.target.macOS12,
    "macos13": ct.target.macOS13,
    "macos14": ct.target.macOS14,
    "macos15": ct.target.macOS15,
}

# The MIL function is authored at this opset per format. ML Program targets a
# modern opset (iOS17) for the best op coverage; NeuralNetwork must author at an
# older opset (iOS15) because coremltools' neuralnetwork backend rejects programs
# stamped with a newer opset than it supports.
_DEFAULT_TARGET = {Format.MLPACKAGE: "ios17", Format.MLMODEL: "ios15"}


def resolve_target(name: str | None, fmt: Format):
    """Resolve a deployment-target string to a coremltools target.

    ``None`` selects a sensible default per format. NeuralNetwork (.mlmodel) has
    no explicit target (coremltools picks the lowest compatible spec).
    """
    if name is None:
        name = _DEFAULT_TARGET[fmt]
    if name is None:
        return None
    key = str(name).lower()
    if key not in _TARGETS:
        raise TargetError(
            f"unknown deployment target {name!r}; "
            f"expected one of {sorted(_TARGETS)}"
        )
    return _TARGETS[key]


_PRECISIONS = {"fp16": precision.FLOAT16, "fp32": precision.FLOAT32}


def resolve_precision(name: str | None, fmt: Format):
    """Resolve a compute-precision string. Defaults to fp16 for ML Program.

    Returns ``None`` for NeuralNetwork (its precision is governed by the runtime,
    not the ``compute_precision`` argument).
    """
    if fmt is Format.MLMODEL:
        return None
    if name is None:
        return precision.FLOAT16
    key = str(name).lower()
    if key not in _PRECISIONS:
        raise TargetError(
            f"unknown compute precision {name!r}; expected 'fp16' or 'fp32'"
        )
    return _PRECISIONS[key]


_COMPUTE_UNITS = {
    "all": ComputeUnit.ALL,
    "cpu_only": ComputeUnit.CPU_ONLY,
    "cpu_and_gpu": ComputeUnit.CPU_AND_GPU,
    "cpu_and_ne": ComputeUnit.CPU_AND_NE,
}


def resolve_compute_units(name: str | None):
    """Resolve a compute-unit string. ``None`` leaves the coremltools default."""
    if name is None:
        return None
    key = str(name).lower()
    if key not in _COMPUTE_UNITS:
        raise TargetError(
            f"unknown compute units {name!r}; expected one of {sorted(_COMPUTE_UNITS)}"
        )
    return _COMPUTE_UNITS[key]
