# Changelog

All notable changes to onnx2coreml are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/), and the project adheres to
[Semantic Versioning](https://semver.org/).

## [Unreleased]

## [1.0.0] — 2026-06-15

First stable release.

### Added
- Operator coverage expanded to 105 ops: LSTM (recurrent), GridSample, DepthToSpace,
  TopK, the ReduceL1/L2/LogSum/LogSumExp/SumSquare family, GatherND/GatherElements,
  ScatterND/ScatterElements, NonZero, matrix Inverse (`com.microsoft`), and the
  trigonometric, logical, and comparison operators.
- Selective fp16 precision: `convert(..., fp32_op_types={...})` keeps numerically
  sensitive ops in fp32 while the rest runs fp16.
- Model-zoo evaluation and export tooling under `tools/`.

### Fixed
- PRelu now handles slopes that broadcast across more than the channel dimension
  (e.g. `(C, H, W)`); the lowering uses the exact elementwise identity rather than
  Core ML's per-channel `prelu` op.
- ConvTranspose now honors `output_padding`.
- PRelu handles non-rank-4 inputs and scalar slopes; Clip handles integer tensors;
  Expand handles boolean tensors.
- Constants are saturated into the int32 and fp16 ranges to avoid overflow (int64
  sentinels and FLT_MAX attention-mask fills).

## [0.1.0] — 2026-06-15

Initial beta.

### Added
- ONNX → Core ML conversion to both `.mlpackage` (ML Program / MIL) and `.mlmodel`
  (NeuralNetwork), built on coremltools' MIL builder.
- 69 operator lowerings across elementwise, activations, convolution/pooling,
  normalization, linear, shape/movement, reduction, and attention families — each
  numerically verified against ONNX Runtime on the live Core ML runtime.
- Graph preprocessing passes: opset normalization, shape inference, constant folding,
  and cleanup (identity/dropout removal, dead-node elimination, initializer pruning).
- Attention fusion: a decomposed scaled-dot-product-attention chain is rewritten to a
  single `ScaledDotProductAttention` node and lowered to iOS15-safe MIL primitives.
- Coverage analysis (`supported_ops`, `analyze`) and a coverage gate that aggregates all
  unsupported ops into one `UnsupportedOpError`.
- Numerical-parity verification (`verify`) and a `Converter` custom-lowering escape hatch.
- Command-line interface: `convert`, `inspect`, `verify`, `schema` (all with `--json`).
- Structured error hierarchy: `Onnx2CoreMLError`, `ModelValidationError`,
  `UnsupportedOpError`, `ConversionError`, `TargetError`.

### Notes
- All coremltools coupling is isolated behind the `_mil` and `_backend` facades so format
  and spec evolution stays contained.
- Fixed input shapes only in this release.
