# onnx2coreml — Design Specification

- **Date:** 2026-06-15
- **Status:** Implemented for v1.0.0
- **Author:** devin-lai
- **Scope:** v1 of a standalone ONNX → Core ML converter

---

## 1. Summary

`onnx2coreml` is a standalone Python package that converts ONNX models into Core ML
formats: **`.mlpackage` (MLProgram / MIL) as the primary target** and **`.mlmodel`
(NeuralNetwork) as a fully-supported secondary target**. It is engineered for
commercial-grade quality, broad and growing operator coverage, and long-term
maintainability as the Core ML model formats evolve.

The converter uses a proven, modular architecture (op-registry dispatch, a preprocess →
fuse → coverage-gate → lower pipeline, structured errors, three-layer numerical testing)
and lowers ONNX onto **coremltools' MIL builder**, reusing coremltools' serialization
backends. Format/spec evolution is therefore
absorbed by coremltools behind two thin facades in this codebase, not by the operator
lowerings.

## 2. Goals & non-goals

### Goals
- Convert ONNX → `.mlpackage` (MLProgram) — the primary, performance-oriented path
  (fp16, Apple Neural Engine friendly).
- Convert ONNX → `.mlmodel` (NeuralNetwork) — secondary path for older-OS reach.
- Commercial-grade quality: no dead code, structured errors, numerical parity tests,
  typed, linted, CI-ready.
- Maintainable as Core ML formats evolve: all coremltools coupling isolated behind
  facades; deployment-target/opset mapping centralized.
- A v1 operator set of 105 ops that runs real CNN **and** transformer blocks end-to-end,
  with a clean path to expand coverage iteratively.

### Non-goals (v1) — YAGNI
- No advanced compression beyond fp16 (`ct.optimize` palettization/quantization/pruning
  is out of scope; `compute_precision` fp16/fp32 only).
- No custom-layer authoring beyond the public custom-lowering registration hook.
- No training, no on-device fine-tuning, no model-surgery utilities.

## 3. Locked decisions

| # | Decision | Choice |
|---|----------|--------|
| 1 | Conversion engine | **Build on coremltools' MIL builder** (`mb.*`) and reuse coremltools serialization; coremltools is a runtime dependency. |
| 2 | Primary output format | **`.mlpackage` (MLProgram/MIL) first**, `.mlmodel` (NeuralNetwork) fully supported as secondary. |
| 3 | Repo structure | **Standalone** repo with clean internal seams, so the ONNX frontend could be extracted into a reusable package later — without committing to that coupling now. |
| 4 | v1 coverage scope | **105 ops** end-to-end with parity + both formats, then expand. |

### Assumptions (confirmed)
- Language/runtime: Python 3.11–3.13.
- Env & lockfile: `uv`; build backend: `setuptools`.
- `convert()` returns a coremltools `MLModel`; the caller calls `.save(path)`.
- Default `minimum_deployment_target` for `.mlpackage` is **iOS17**, configurable.
- ONNX graph passes are implemented in-package (using `onnx` + `numpy`); no hard
  dependency on `onnxsim`.
- Distribution/import name is **`onnx2coreml`**, deliberately distinct from the
  deprecated `onnx-coreml` / `onnx_coreml` package on PyPI.

## 4. Architecture

### 4.1 Pipeline (end-to-end data flow)

```
ONNX (path | ModelProto | bytes)
  → _io.load        load + onnx.checker.check_model
  → _passes         opset-normalize (>=13) · shape/type inference · constant-fold ·
                    cleanup (identity/dropout removal · dead-node elimination · prune initializers)
  → _fusion         (optional, semantics-preserving) attention chain -> sdpa
  → _coverage       GATE: aggregate ALL ops with no lowering -> single UnsupportedOpError
  → _lowering       topological node walk -> emit mb.* MIL ops; initializers -> mb.const;
                    track ONNX value name -> MIL Var in values_map; set function outputs
  → _backend        MIL Program -> MLModel   (mlprogram -> .mlpackage | neuralnetwork -> .mlmodel)
  → _verify         (optional) numerical parity: onnxruntime (reference) vs MLModel.predict
  → MLModel.save()  container chosen by format/extension
```

The coverage-gate runs **before** any MIL op is emitted, so a model with unsupported ops
fails fast with one aggregated report rather than partway through emission.

### 4.2 Module layout

Package import name: `onnx2coreml`; `src/` layout.

```
src/onnx2coreml/
  __init__.py        public API: convert, verify, supported_ops, analyze, Converter, errors
  _convert.py        convert(...) user-facing entry
  converter.py       Converter class: orchestrates passes -> fusion -> coverage -> lowering ->
                     backend; holds per-instance custom-lowering registry
  _io.py             load ONNX (path/proto/bytes); save Core ML model
  _mil.py            ★ FACADE: the ONLY module importing coremltools' Builder / types / target /
                     conversion entry points
  _backend.py        ★ FACADE: MIL Program -> MLModel -> save, per format (isolates serialization)
  _target.py         deployment-target -> opset/spec-version mapping; format + precision selection
  _types.py          ONNX dtype <-> MIL dtype mapping; 64->32-bit narrowing policy
  _coverage.py       supported_ops(); analyze(model) -> CoverageReport
  _verify.py         numerical parity (onnxruntime reference vs Core ML predict) -> VerifyReport
  _utils.py          op_key, attribute extraction, operand helpers, topological iteration
  errors.py          exception hierarchy
  _passes/
    __init__.py      pass pipeline runner
    _model.py        opset normalization, shape/type inference
    _fold.py         constant folding
    _cleanup.py      identity/dropout removal, dead-node elimination, initializer pruning
  _fusion/
    __init__.py      fusion pipeline runner
    _rewrite.py      pattern-match + greedy rewrite engine
    _attention.py    attention chain -> ScaledDotProductAttention
  _lowering/
    __init__.py      master registry: merge per-module REGISTRY dicts (dup-check) + dispatch
    _common.py       shared lowering primitives (_unary, _binary, broadcast helpers)
    _elementwise.py  arithmetic, comparison, unary math, Where
    _matmul.py       MatMul, Gemm
    _conv.py         Conv, ConvTranspose, pooling
    _norm.py         BatchNorm, LayerNorm, InstanceNorm, GroupNorm
    _activations.py  Relu, Sigmoid, Tanh, Gelu, Softmax, ...
    _shape.py        Reshape, Transpose, Squeeze/Unsqueeze, Concat, Split, Flatten, Pad, Cast, ...
    _reduce.py       ReduceMean/Sum/Max/Min/Prod, ArgMax, ArgMin
    _indexing.py     Gather, Slice, Expand, Tile, Where, ConstantOfShape, Shape
    _attention.py    ScaledDotProductAttention
  _cli.py            CLI: convert / inspect / verify / schema
  __version__.py
```

### 4.3 Maintainability seams (a hard requirement, called out explicitly)

- **`_mil.py`** and **`_backend.py`** are the only modules that import coremltools.
  A coremltools reorganization or a new Core ML spec version touches these two files,
  never the operator lowerings. This is a facade pattern applied to the coremltools
  boundary.
- **`_target.py`** centralizes the deployment-target → opset/spec-version table, so adding
  a future iOS/macOS target is a single edit.
- `_lowering/`, `_passes/`, `_fusion/`, `_coverage.py` are written with clean seams so an
  ONNX-frontend could later be extracted into a reusable package, per decision #3 —
  without committing to that coupling now.

## 5. Operator registry & lowering

A registry pattern that scales past 100 ops without decorators or import-order
fragility:

- Each `_lowering/_*.py` exposes `REGISTRY: dict[str, Callable]`.
- `_lowering/__init__.py` merges all module REGISTRY dicts into one resolver, raising at
  import time on any duplicate key (prevents silent shadowing).
- Op key is `OpType` or `domain::OpType` for non-default domains.

Lowering function signature:

```python
def lower(ctx: LoweringContext, node: onnx.NodeProto) -> Var | list[Var]: ...
```

`LoweringContext` carries:
- `values_map: dict[str, Var]` — ONNX value name → MIL `Var`,
- `target` — the resolved deployment target / opset,
- `converter` — handle for recursive subgraph lowering (reserved for If/Loop in a later
  iteration).

`mb.*` calls emit into the currently active MIL `Function` block. Initializers are
materialized as `mb.const`. Graph outputs are set from `values_map` at the end of the walk.

## 6. v1 operator set (105 ops)

Chosen to run real CNN vision models and transformer blocks end-to-end.

- **Elementwise / math:** Add, Sub, Mul, Div, Pow, Sqrt, Exp, Log, Abs, Erf,
  Reciprocal, Neg, Floor, Ceil, Round, Sign, Sin, Cos, Tan, Asin, Acos, Atan, Sinh,
  Cosh, Atanh, Min, Max, Clip, Mod, Equal, Greater, Less, GreaterOrEqual,
  LessOrEqual, And, Or, Xor, Not, IsNaN, Where
- **Activations:** Relu, LeakyRelu, PRelu, Sigmoid, Tanh, Gelu, Softmax,
  LogSoftmax, Softplus, Elu, HardSigmoid, HardSwish
- **Conv / pool:** Conv, ConvTranspose, MaxPool, AveragePool, GlobalAveragePool,
  GlobalMaxPool
- **Normalization:** BatchNormalization, LayerNormalization, InstanceNormalization,
  GroupNormalization
- **Linear:** MatMul, Gemm, Inverse (`com.microsoft`)
- **Shape / movement:** Reshape, Transpose, Flatten, Squeeze, Unsqueeze, Concat, Split,
  Pad, Cast, Identity, DepthToSpace, GridSample, Resize, Upsample
- **Indexing / gather-scatter:** Gather, GatherND, GatherElements, ScatterND,
  ScatterElements, NonZero, Slice, Expand, Tile, Shape, ConstantOfShape
- **Reduce / arg:** ReduceMean, ReduceSum, ReduceMax, ReduceMin, ReduceProd, ReduceL1,
  ReduceL2, ReduceLogSum, ReduceLogSumExp, ReduceSumSquare, ArgMax, ArgMin, TopK
- **Recurrent:** LSTM
- **Fused:** ScaledDotProductAttention (produced by the attention fusion pass)

## 7. Public API

```python
import onnx2coreml as o2c

mlmodel = o2c.convert(
    "model.onnx",                       # str | Path | onnx.ModelProto | bytes
    format="mlpackage",                 # "mlpackage" (default) | "mlmodel"
    minimum_deployment_target="iOS17",  # str | coremltools target; default iOS17 for mlpackage
    compute_precision="fp16",           # "fp16" (default for mlprogram) | "fp32"
    fuse=True,                          # enable fusion passes
    verify=False,                       # run numerical parity check after conversion
)
mlmodel.save("model.mlpackage")         # coremltools MLModel; container by format/extension
```

Additional public surface:
- `o2c.supported_ops() -> set[str]`
- `o2c.analyze(model) -> CoverageReport` — unsupported ops per requested format
- `o2c.verify(onnx_model, coreml_model, *, rtol, atol, min_psnr) -> VerifyReport`
- `o2c.Converter` — low-level orchestrator exposing
  `@converter.register("OpType")` as the unsupported-op escape hatch, plus `to_mil()`.

## 8. CLI

Console script `onnx2coreml` (entry point `onnx2coreml._cli:main`):

```
onnx2coreml convert model.onnx -o model.mlpackage
    [--format mlpackage|mlmodel] [--target iOS17] [--precision fp16|fp32]
    [--compute-units all|cpu_only|cpu_and_gpu|cpu_and_ne] [--no-fuse] [--verify] [--json]
onnx2coreml inspect model.onnx          # op histogram + convertibility verdict per format
onnx2coreml verify  model.onnx model.mlpackage [--rtol R --atol A --min-psnr P]
onnx2coreml schema                      # version, supported op count, error codes
```

All subcommands accept `--json` for machine-readable output.

## 9. Error model

Coverage-gate runs before any emission. Exception hierarchy:

- `Onnx2CoreMLError` — base.
- `UnsupportedOpError` — aggregates every op with no lowering: op key, node count,
  example node names, and how to register a custom lowering.
- `ConversionError` — a lowering failed for a specific node; wraps the underlying cause
  plus node name and op key.
- `ModelValidationError` — ONNX failed to load or `check_model`; the failure is
  attributed to the input rather than surfacing later as an opaque conversion error.
- `TargetError` — Core ML-specific: an op needs a higher deployment target than requested,
  or cannot be expressed in the requested format (e.g., NeuralNetwork cannot represent an
  op available only in MLProgram).

## 10. Verification & testing

### Verification (`_verify.py`)
Numerical parity: run the source ONNX on **onnxruntime** (reference) and the converted
model via **coremltools `MLModel.predict`** (the real Core ML runtime on macOS); compare
max absolute error, max relative error, and PSNR against tolerances. Returns a structured
`VerifyReport` (per-output metrics + pass/fail).

### Testing (three layers + Core ML reality)
1. **Per-op parity:** a `single_op_model(op, inputs)` factory builds a minimal ONNX model,
   converts it, and asserts onnxruntime-vs-Core-ML parity. Parametrized over dtypes,
   shapes, and both formats. Seeded RNG for reproducibility.
2. **Fusion tests:** build canonical patterns (e.g., the SDPA chain), assert the fused op
   is present and output parity holds.
3. **API / CLI tests:** `convert()` contract, CLI `--json` output stability, coverage
   analysis correctness.
4. **Model integration tests:** a few small real models (a tiny ResNet block, a tiny
   transformer block) end-to-end.

`@pytest.mark.coreml` gates tests that require the macOS Core ML runtime for `predict`.
Where the runtime is unavailable (Linux/CI), tests fall back to **build + serialize +
spec-validate** and skip only the numeric `predict` step. Note that predicting models
that target newer iOS opsets requires a correspondingly recent macOS; the marker layer
accounts for this.

Tolerances (defaults, configurable per test): fp32 `rtol=1e-3 / atol=1e-4`;
fp16 `rtol=1e-2 / atol=1e-3`; PSNR floor configurable, off by default.

## 11. Packaging & tooling

- Python 3.11–3.13, `src/` layout, package `onnx2coreml`.
- Build backend: `setuptools`; env & lockfile via `uv`.
- Runtime deps: `coremltools>=8`, `onnx>=1.16`, `numpy`.
- Optional extras:
  - `verify = ["onnxruntime"]`
  - `test = ["onnxruntime", "pytest", "pytest-xdist"]`
  - `dev = ["ruff", "mypy", "pre-commit", "build", "twine"]`
  - `docs = ["sphinx", ...]`
- Lint/type/test: ruff (`E,F,I,W,B,UP,SIM,C4,PERF,RET,RUF,PT`), mypy, pytest
  (`pythonpath=["src"]`), pre-commit, branch coverage.
- Console script: `onnx2coreml = onnx2coreml._cli:main`.

## 12. Risks & mitigations

- **Layout / axis-order mismatches (ONNX NCHW vs Core ML conventions).** This sank the old
  onnx-coreml via scattered heuristics. Mitigation: rely on coremltools MIL ops' documented
  semantics, insert explicit transposes during lowering rather than inferring intent, and
  cover with per-op parity tests over multiple shapes.
- **Dynamic / flexible shapes.** Mitigation: reject unknown dimensions explicitly until
  range/enumerated-shape support is added and covered by parity tests.
- **NeuralNetwork format gaps.** Some ops only express cleanly in MLProgram. Mitigation:
  `TargetError` with a clear message and a per-format coverage report from `analyze()`.
- **coremltools runtime availability in CI.** Mitigation: the `@pytest.mark.coreml`
  fallback to spec-validation keeps non-macOS CI green while macOS jobs run full parity.
- **64→32-bit narrowing.** Core ML prefers 32-bit; INT64/DOUBLE inputs need a policy.
  Mitigation: explicit narrowing in `_types.py` with documented behavior and tests.

## 13. Success criteria for v1

- `convert()` produces a loadable `.mlpackage` and `.mlmodel` for models built from the v1
  op set.
- Per-op parity tests pass for every v1 op on macOS (onnxruntime vs Core ML predict) within
  tolerances; non-macOS CI passes via spec-validation fallback.
- At least two small real models (one CNN, one transformer block) convert and verify
  end-to-end.
- `inspect` reports per-format convertibility; unsupported ops produce a single aggregated
  `UnsupportedOpError`.
- Clean `ruff` + `mypy`, packaged installable wheel, working console script.

## 14. Post-v1 expansion (tracked, not built now)

Control flow (If/Loop/Scan), GRU, quantized ops (QLinear*), compression integration, and
possible extraction of a reusable ONNX frontend.
