# onnx2coreml

[![PyPI](https://img.shields.io/pypi/v/onnx2coreml.svg)](https://pypi.org/project/onnx2coreml/)
[![CI](https://github.com/devin-lai/onnx2coreml/actions/workflows/ci.yml/badge.svg)](https://github.com/devin-lai/onnx2coreml/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.11%20%7C%203.12%20%7C%203.13-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-BSD--3--Clause-blue.svg)](LICENSE)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

Convert ONNX models to Apple Core ML — **`.mlpackage`** (ML Program / MIL, primary) and
**`.mlmodel`** (NeuralNetwork, secondary) — built on coremltools' MIL builder, with
numerical-parity verification against ONNX Runtime.

- **ML Program first** for modern on-device performance (fp16 and Apple Neural Engine),
  with full NeuralNetwork support for older-OS reach.
- **Maintainable as Core ML evolves**: all coremltools coupling is isolated behind two
  facades, so spec/format changes touch two files, not the operator lowerings.
- **Verified, not assumed**: every operator has numerical-parity tests run against the live
  Core ML runtime on macOS.

## Install

```bash
uv pip install onnx2coreml                # from PyPI

# from a clone, for development:
uv pip install -e ".[dev,verify,test]"
```

Requires Python 3.11–3.13 and macOS for prediction/verification (model *building* works
anywhere coremltools installs).

## Quickstart

```python
import onnx2coreml as o2c

# ML Program (.mlpackage) — the default, fp16, iOS17+
mlmodel = o2c.convert("model.onnx", format="mlpackage", minimum_deployment_target="iOS17")
mlmodel.save("model.mlpackage")

# NeuralNetwork (.mlmodel)
o2c.convert("model.onnx", format="mlmodel").save("model.mlmodel")

# Check coverage before converting
report = o2c.analyze("model.onnx")
print(report.convertible, report.unsupported)

# Numerical parity vs ONNX Runtime
print(o2c.verify("model.onnx", "model.mlpackage"))
```

`convert()` returns a coremltools `MLModel`. Options: `format`,
`minimum_deployment_target`, `compute_precision` (`fp16`/`fp32`), `compute_units`,
`fuse`, `verify`.

## CLI

```bash
onnx2coreml convert model.onnx -o model.mlpackage --format mlpackage --verify
onnx2coreml inspect model.onnx          # op histogram + per-format convertibility
onnx2coreml verify  model.onnx model.mlpackage
onnx2coreml schema                      # version, supported op count, error codes
```

All subcommands accept `--json`.

## Supported operators (106 ops)

| Family | Operators |
|--------|-----------|
| Elementwise / math | Add, Sub, Mul, Div, Pow, Sqrt, Exp, Log, Abs, Erf, Reciprocal, Neg, Floor, Ceil, Round, Sign, Sin, Cos, Tan, Asin, Acos, Atan, Sinh, Cosh, Atanh, Min, Max, Clip, Mod, Equal, Greater, Less, GreaterOrEqual, LessOrEqual, And, Or, Xor, Not, IsNaN, Where |
| Activations | Relu, LeakyRelu, PRelu, Sigmoid, Tanh, Gelu, Softmax, LogSoftmax, Softplus, Elu, HardSigmoid, HardSwish |
| Convolution / pooling | Conv, ConvTranspose, MaxPool, AveragePool, GlobalAveragePool, GlobalMaxPool |
| Normalization | BatchNormalization, LayerNormalization, InstanceNormalization, GroupNormalization |
| Linear | MatMul, Gemm, Inverse (`com.microsoft`) |
| Shape / movement | Reshape, Transpose, Flatten, Squeeze, Unsqueeze, Concat, Split, Pad, Cast, Identity, DepthToSpace, SpaceToDepth, GridSample, Resize, Upsample |
| Indexing / gather-scatter | Gather, GatherND, GatherElements, ScatterND, ScatterElements, NonZero, Slice, Expand, Tile, Shape, ConstantOfShape |
| Reduction | ReduceMean, ReduceSum, ReduceMax, ReduceMin, ReduceProd, ReduceL1, ReduceL2, ReduceLogSum, ReduceLogSumExp, ReduceSumSquare, ArgMax, ArgMin, TopK |
| Recurrent | LSTM |
| Attention | ScaledDotProductAttention (produced by the attention fusion pass) |

`onnx2coreml inspect <model>` reports exactly which ops a given model needs and whether
each is covered. Unsupported ops raise a single aggregated `UnsupportedOpError`; register
a custom lowering with `Converter().register("OpType")` as an escape hatch.

## How it works

```
ONNX → load → passes (opset/shape/fold/cleanup) → fusion (attention→SDPA) →
coverage-gate → lower to MIL (mb.*) → coremltools backend → .mlpackage / .mlmodel → verify
```

See `AGENTS.md` for the developer guide and architecture overview.

## Known limitations (v1.1.0)

- Fixed input shapes.
- A full all-axes reduction with `keepdims=0` yields shape `(1,)` (Core ML has no rank-0
  scalar).
- Resize `linear`+`half_pixel` via explicit sizes is `.mlpackage`-only (the NeuralNetwork
  backend lacks the sampling mode); the scales path covers it on both formats.

## License

BSD-3-Clause.
