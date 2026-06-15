# Contributing to onnx2coreml

Thanks for your interest in improving onnx2coreml. This file covers the practical
workflow; `AGENTS.md` holds the deeper architectural guide and the operator-lowering
conventions.

## Development setup

```bash
uv pip install -e ".[dev,verify,test]"
pre-commit install
```

Python 3.11–3.13 is supported. Building a model works anywhere coremltools installs;
prediction and numerical-parity verification require macOS.

## Quality gates

Every change must pass these before review:

```bash
ruff check src tests
mypy src
python -m pytest tests/ -q
```

`pre-commit` runs ruff and mypy on commit. CI runs the same gates plus the full test
suite across the supported Python versions on macOS.

## Adding an operator

The end-to-end recipe lives in `AGENTS.md` under "Adding an operator lowering". In short:
pick the right family module in `src/onnx2coreml/_lowering/`, write `lower(ctx, node)`,
register the op key, confirm the MIL op and parameter names against the coremltools op
definitions rather than guessing, and add a parity test in `tests/test_ops_<family>.py`
parametrized over both `mlpackage` and `mlmodel`.

Parity tests compare against ONNX Runtime at fp32 on `CPU_ONLY`, so a mismatch signals a
real lowering bug rather than fp16 rounding. Mark genuine, verified platform limits as
`xfail` with the reason inline — never to paper over a bug.

## Pull requests

- Keep changes surgical and consistent with the surrounding code.
- Describe what changed and why; link any relevant issue.
- Update `CHANGELOG.md` under `[Unreleased]` when behavior or coverage changes.
- New code carries the BSD header, `from __future__ import annotations`, a module
  docstring, and type hints.

## Reporting bugs

Open an issue with the ONNX model (or a minimal reproducer), the command or API call,
and the full error. Conversion failures raise a structured `Onnx2CoreMLError` subclass —
include its message and op type.

## License

By contributing, you agree that your contributions are licensed under the project's
BSD-3-Clause license.
