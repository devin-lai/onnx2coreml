# Developer guide for onnx2coreml

Guidance for contributors working in this repo. Keep changes surgical, tested, and
consistent with the patterns below.

## What this is

An ONNX → Core ML converter producing `.mlpackage` (ML Program / MIL) and `.mlmodel`
(NeuralNetwork), built on top of coremltools' MIL builder.

## Architecture (one pass through the pipeline)

```
ONNX → _io.load → _passes.run → _fusion.run → coverage-gate → Converter.to_mil
     → _backend.program_to_mlmodel → MLModel (.mlpackage / .mlmodel) → _verify
```

- `_mil.py` and `_backend.py` are the **only** modules that import coremltools. Keep it
  that way — this is how the project stays maintainable as Core ML evolves.
- `converter.py` orchestrates: coverage gate, then a topological walk emitting MIL ops.
- `_target.py` maps deployment-target/format/precision strings to coremltools enums. The
  MIL function is authored at iOS17 for `.mlpackage` and **iOS15 for `.mlmodel`** (the
  NeuralNetwork backend rejects newer opsets).

## Adding an operator lowering

1. Pick the right family module under `src/onnx2coreml/_lowering/` (e.g. `_conv.py`).
2. Write `def lower(ctx, node) -> Var | list[Var]:` — read inputs via
   `operands(ctx.values_map, node, [...])`, emit `mb.*` ops, and **name the final op for
   each output with `name=node.output[i]`** (so Core ML's predicted output keys match ONNX
   names). Never pass `name=None` to an intermediate op — omit the kwarg.
3. Add the op key to that module's `REGISTRY` dict. Keys are unique across modules (a
   duplicate raises at import).
4. Find the exact MIL op + parameter names in
   `../coremltools/coremltools/converters/mil/mil/ops/defs/` — do not guess.
5. Add a parity test in `tests/test_ops_<family>.py` parametrized over
   `fmt in ["mlpackage", "mlmodel"]`.

Helpers in `_lowering/_common.py`: `binary(mb_op)`, `unary(mb_op)`,
`const_array(ctx, node, idx)`, `get_attr`, `operands`.

## Testing

```bash
.venv/bin/python -m pytest tests/ -q          # full suite
.venv/bin/python -m pytest tests/test_ops_conv.py -q
```

- The harness (`tests/helpers.py`) builds a single-op ONNX model, runs ONNX Runtime as the
  reference, converts, and compares against `MLModel.predict`.
- Parity is pinned to **fp32 + CPU_ONLY** so a mismatch means a real bug, not fp16 rounding.
- `assert_parity` serializes predictions with a file lock (the Core ML runtime races on
  concurrent loads).
- Mark genuine, verified platform limits as `xfail` with the reason inline — never to hide
  a lowering bug.

## Conventions

- Python 3.11–3.13, `uv` for env/lock, `setuptools` build backend.
- Every file: BSD header, `from __future__ import annotations`, a module docstring, type
  hints. Comments only where the *why* is non-obvious.
- Gates (must pass before done): `.venv/bin/ruff check src tests`, `.venv/bin/mypy src`,
  full `pytest`.
- Errors derive from `Onnx2CoreMLError`; the coverage gate runs before any MIL is emitted.
- Core ML compute is 32-bit: int64→int32 and float64→float32 narrowing lives in `_types.py`.
