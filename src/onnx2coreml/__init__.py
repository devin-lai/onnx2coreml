# Copyright 2026 onnx2coreml contributors.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""onnx2coreml — convert ONNX models to Apple Core ML (.mlpackage / .mlmodel)."""

from __future__ import annotations

from .__version__ import __version__
from ._convert import convert
from ._coverage import CoverageReport, analyze, supported_ops
from ._verify import VerifyReport, verify
from .converter import Converter
from .errors import (
    ConversionError,
    ModelValidationError,
    Onnx2CoreMLError,
    TargetError,
    UnsupportedOpError,
)

__all__ = [
    "ConversionError",
    "Converter",
    "CoverageReport",
    "ModelValidationError",
    "Onnx2CoreMLError",
    "TargetError",
    "UnsupportedOpError",
    "VerifyReport",
    "__version__",
    "analyze",
    "convert",
    "supported_ops",
    "verify",
]
