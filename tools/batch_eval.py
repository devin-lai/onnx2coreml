# Copyright 2026 onnx2coreml contributors.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""Batch conversion and parity checks for a flat directory of ``*.onnx`` files.

This is the small-corpus sibling of ``zoo_eval.py``: it consumes raw ONNX graphs
instead of packaged model-zoo bundles. Each graph is converted to every requested
Core ML format, optionally saved, and optionally checked against ONNX Runtime
with the same seeded inputs. A subprocess boundary around each graph keeps
timeouts and native crashes local to that graph.

Usage::

    python tools/batch_eval.py --onnx-dir tmp_test_onnx/manis_onnx \\
        --out-dir tmp_test_output
    python tools/batch_eval.py --formats mlpackage,mlmodel --limit 10
    python tools/batch_eval.py --no-save          # convert+verify only
    python tools/batch_eval.py --no-verify        # convert (+save) only, fast

The script is a development tool and is not installed with the package.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path

_VALID_FORMATS = ("mlpackage", "mlmodel")


@dataclass
class Options:
    onnx_dir: Path
    out_dir: Path
    formats: tuple[str, ...] = _VALID_FORMATS
    precision: str = "fp16"
    timeout: float = 1200.0
    pass_psnr: float = 40.0
    weak_psnr: float = 20.0
    save: bool = True
    verify: bool = True


@dataclass
class FormatResult:
    """Outcome of converting one model to one Core ML format."""

    fmt: str
    status: str = "error"  # ok | convert_failed | verify_failed | error
    verdict: str = "n/a"  # pass | weak | fail | n/a
    convert_ok: bool = False
    saved_path: str | None = None
    verify_ran: bool = False
    failure_class: str | None = None
    missing_ops: list[str] | None = None
    error_type: str | None = None
    error_message: str | None = None
    error_node: str | None = None
    error_op: str | None = None
    strict_parity: bool | None = None
    min_psnr: float | None = None
    outputs: list[dict] | None = None
    convert_s: float | None = None
    verify_s: float | None = None
    notes: str | None = None


@dataclass
class Result:
    """All per-format outcomes for one ONNX graph."""

    model: str
    onnx: str
    input_shapes: dict[str, list[int]] = field(default_factory=dict)
    formats: list[FormatResult] = field(default_factory=list)
    # Set only if the worker died before producing per-format records.
    status: str | None = None
    failure_class: str | None = None
    error_message: str | None = None


def classify_error(exc: BaseException) -> dict:
    """Map an exception to a stable failure class + structured fields."""
    from onnx2coreml.errors import (
        ConversionError,
        ModelValidationError,
        TargetError,
        UnsupportedOpError,
    )

    info: dict = {
        "failure_class": "other",
        "error_type": type(exc).__name__,
        "error_message": str(exc)[:2000],
    }
    if isinstance(exc, UnsupportedOpError):
        info["failure_class"] = "unsupported_op"
        info["missing_ops"] = sorted(exc.missing)
    elif isinstance(exc, ConversionError):
        info["failure_class"] = "lowering_error"
        info["error_node"] = exc.node_name
        info["error_op"] = exc.op_key
        info["error_message"] = str(exc.cause)[:2000]
    elif isinstance(exc, ModelValidationError):
        info["failure_class"] = "model_validation"
    elif isinstance(exc, TargetError):
        info["failure_class"] = "target_error"
    else:
        msg = str(exc).lower()
        if "dynamic or unknown dimension" in msg:
            info["failure_class"] = "dynamic_shape"
    return info


def _is_inf(x: float) -> bool:
    return x == float("inf")


def _is_ane_build_error(exc: BaseException) -> bool:
    """Whether a predict failure looks like an ANE compiler failure."""
    msg = str(exc).lower()
    return any(s in msg for s in ("building plan", "execution plan", "ane", "anecompile"))


def _verdict(report, opts: Options) -> str:
    if report.passed:
        return "pass"
    floats = [o.psnr for o in report.outputs if not _is_inf(o.psnr)]
    min_psnr = min(floats) if floats else float("inf")
    if min_psnr >= opts.pass_psnr:
        return "pass"
    if min_psnr >= opts.weak_psnr:
        return "weak"
    return "fail"


def _save_path(opts: Options, model_stem: str, fmt: str) -> Path:
    suffix = ".mlpackage" if fmt == "mlpackage" else ".mlmodel"
    return opts.out_dir / fmt / f"{model_stem}{suffix}"


def _convert_one_format(
    proto, inputs, model_stem: str, fmt: str, opts: Options
) -> FormatResult:
    import onnx2coreml as o2c
    from onnx2coreml._verify import verify_model

    res = FormatResult(fmt=fmt)

    # ML Program honors compute_precision; NeuralNetwork uses its runtime default.
    precision = opts.precision if fmt == "mlpackage" else None
    t0 = time.perf_counter()
    try:
        mlmodel = o2c.convert(proto, format=fmt, compute_precision=precision)
    except BaseException as exc:
        res.convert_s = time.perf_counter() - t0
        res.status = "convert_failed"
        res.__dict__.update(classify_error(exc))
        return res
    res.convert_ok = True
    res.convert_s = time.perf_counter() - t0

    if opts.save:
        path = _save_path(opts, model_stem, fmt)
        path.parent.mkdir(parents=True, exist_ok=True)
        mlmodel.save(str(path))
        res.saved_path = str(path)

    if not opts.verify:
        res.status = "ok"
        return res

    t1 = time.perf_counter()
    try:
        report = verify_model(proto, mlmodel, inputs=inputs)
    except BaseException as exc:
        report = None
        if _is_ane_build_error(exc):
            try:
                mlmodel = o2c.convert(
                    proto, format=fmt, compute_precision=precision,
                    compute_units="cpu_and_gpu",
                )
                report = verify_model(proto, mlmodel, inputs=inputs)
                res.notes = "ANE compilation failed; verified on CPU+GPU"
            except BaseException as exc2:
                exc = exc2
        if report is None:
            res.verify_s = time.perf_counter() - t1
            res.status = "verify_failed"
            res.failure_class = "predict_error"
            res.error_type = type(exc).__name__
            res.error_message = str(exc)[:2000]
            return res
    res.verify_s = time.perf_counter() - t1
    res.verify_ran = True

    res.outputs = [o.as_dict() for o in report.outputs]
    res.strict_parity = report.passed
    psnrs = [o.psnr for o in report.outputs if not _is_inf(o.psnr)]
    res.min_psnr = min(psnrs) if psnrs else float("inf")
    res.status = "ok"
    res.verdict = _verdict(report, opts)
    return res


def run_one(onnx_path: Path, result: Result, opts: Options) -> Result:
    import numpy as np
    import onnx

    from onnx2coreml._verify import generate_inputs

    proto = onnx.load_model(str(onnx_path))

    # Share inputs across formats so their metrics are directly comparable.
    inputs = None
    if opts.verify:
        try:
            inputs = generate_inputs(proto)
            result.input_shapes = {k: list(np.asarray(v).shape) for k, v in inputs.items()}
        except BaseException as exc:
            result.error_message = str(exc)[:500]

    for fmt in opts.formats:
        result.formats.append(
            _convert_one_format(proto, inputs, onnx_path.stem, fmt, opts)
        )
    return result


def _worker_main(args: argparse.Namespace) -> int:
    opts = Options(
        onnx_dir=Path(args.onnx_dir),
        out_dir=Path(args.out_dir),
        formats=tuple(args.formats.split(",")),
        precision=args.precision,
        pass_psnr=args.pass_psnr,
        weak_psnr=args.weak_psnr,
        save=not args.no_save,
        verify=not args.no_verify,
    )
    result = Result(model=Path(args.onnx).stem, onnx=Path(args.onnx).name)
    try:
        result = run_one(Path(args.onnx), result, opts)
    except BaseException as exc:
        result.status = "error"
        result.failure_class = "worker_exception"
        result.error_message = str(exc)[:2000]
    Path(args.result).write_text(json.dumps(asdict(result)), encoding="utf-8")
    return 0


def discover(opts: Options) -> list[Path]:
    return sorted(opts.onnx_dir.glob("*.onnx"))


def _spawn_one(onnx_path: Path, opts: Options) -> Result:
    """Run one model in an isolated subprocess with a timeout."""
    result_file = opts.out_dir / "_result.json"
    if result_file.exists():
        result_file.unlink()
    cmd = [
        sys.executable, str(Path(__file__).resolve()),
        "--run-one",
        "--onnx", str(onnx_path),
        "--result", str(result_file),
        "--onnx-dir", str(opts.onnx_dir),
        "--out-dir", str(opts.out_dir),
        "--formats", ",".join(opts.formats),
        "--precision", opts.precision,
        "--pass-psnr", str(opts.pass_psnr),
        "--weak-psnr", str(opts.weak_psnr),
    ]
    if not opts.save:
        cmd.append("--no-save")
    if not opts.verify:
        cmd.append("--no-verify")

    base = Result(model=onnx_path.stem, onnx=onnx_path.name)
    try:
        proc = subprocess.run(
            cmd, timeout=opts.timeout, capture_output=True, text=True,
            env={**os.environ, "PYTHONWARNINGS": "ignore"},
        )
    except subprocess.TimeoutExpired:
        base.status = "error"
        base.failure_class = "timeout"
        base.error_message = f"exceeded {opts.timeout:g}s"
        return base

    if result_file.exists():
        data = json.loads(result_file.read_text(encoding="utf-8"))
        result_file.unlink()
        data["formats"] = [FormatResult(**f) for f in data.get("formats", [])]
        return Result(**data)

    # No result file => the worker died (segfault / OOM / native abort).
    base.status = "error"
    base.failure_class = "crash"
    tail = (proc.stderr or proc.stdout or "").strip().splitlines()[-5:]
    base.error_message = f"exit={proc.returncode}; " + " | ".join(tail)
    return base


def orchestrate(opts: Options, only: set[str] | None, limit: int | None) -> None:
    opts.out_dir.mkdir(parents=True, exist_ok=True)
    models = discover(opts)
    if only:
        models = [m for m in models if m.stem in only]
    if limit:
        models = models[:limit]

    report_path = opts.out_dir / "report.json"
    results: list[Result] = []
    started = time.perf_counter()
    print(f"[batch] {len(models)} models x {opts.formats} -> {opts.out_dir}", flush=True)

    for i, onnx_path in enumerate(models, 1):
        r = _spawn_one(onnx_path, opts)
        results.append(r)
        _write_report(report_path, results, opts, started)
        print(f"[{i}/{len(models)}] {onnx_path.stem}: {_one_line(r)}", flush=True)

    _print_summary(results, opts)
    print(f"\n[batch] report: {report_path}", flush=True)


def _fmt_line(fr: FormatResult) -> str:
    if fr.status == "ok" and fr.verify_ran:
        psnr = "inf" if fr.min_psnr == float("inf") else f"{fr.min_psnr:.1f}dB"
        return f"{fr.fmt}={fr.verdict.upper()}({psnr})"
    if fr.status == "ok":
        return f"{fr.fmt}=BUILT"
    if fr.failure_class == "unsupported_op":
        return f"{fr.fmt}=UNSUPPORTED[{','.join(fr.missing_ops or [])}]"
    detail = fr.error_op or (fr.error_message or "")[:48]
    return f"{fr.fmt}={(fr.failure_class or fr.status or '?').upper()}[{detail}]"


def _one_line(r: Result) -> str:
    if r.status and r.status != "ok":
        return f"{r.status.upper()} [{r.failure_class}] {r.error_message or ''}"[:160]
    return "  ".join(_fmt_line(f) for f in r.formats)


def _write_report(path: Path, results: list[Result], opts: Options, started: float) -> None:
    payload = {
        "options": {
            "formats": list(opts.formats), "precision": opts.precision,
            "save": opts.save, "verify": opts.verify,
            "pass_psnr": opts.pass_psnr, "weak_psnr": opts.weak_psnr,
            "reference": "onnxruntime CPUExecutionProvider (fp32)",
        },
        "elapsed_s": time.perf_counter() - started,
        "count": len(results),
        "results": [asdict(r) for r in results],
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _print_summary(results: list[Result], opts: Options) -> None:
    n = len(results)
    print("\n" + "=" * 64, flush=True)
    print(f"SUMMARY  ({n} models, formats={','.join(opts.formats)})", flush=True)

    op_models: dict[str, set[str]] = {}
    for fmt in opts.formats:
        frs = [(r, f) for r in results for f in r.formats if f.fmt == fmt]
        convert_ok = [f for _, f in frs if f.convert_ok]
        verify_ran = [f for _, f in frs if f.verify_ran]
        verdicts = Counter(f.verdict for f in verify_ran)
        classes = Counter(f.failure_class for _, f in frs if f.failure_class)
        print(f"\n  [{fmt}]", flush=True)
        print(f"    converted        : {len(convert_ok)}/{len(frs)}", flush=True)
        if opts.verify:
            print(f"    accuracy-checked : {len(verify_ran)}/{len(frs)}", flush=True)
            print(f"    pass / weak / fail: {verdicts.get('pass', 0)} / "
                  f"{verdicts.get('weak', 0)} / {verdicts.get('fail', 0)}", flush=True)
        if classes:
            print("    failure classes:", flush=True)
            for cls, c in classes.most_common():
                print(f"      {cls:18s} {c}", flush=True)
        for r, f in frs:
            for op in f.missing_ops or []:
                op_models.setdefault(op, set()).add(f"{fmt}:{r.model}")

    worker_fail = Counter(r.failure_class for r in results if r.status and r.status != "ok")
    if worker_fail:
        print("\n  worker-level failures:", flush=True)
        for cls, c in worker_fail.most_common():
            print(f"    {cls:18s} {c}", flush=True)

    if op_models:
        print("\n  missing ops (by #graph-formats blocked):", flush=True)
        for op, models in sorted(op_models.items(), key=lambda kv: (-len(kv[1]), kv[0])):
            print(f"    {op:28s} {len(models)}", flush=True)


def _build_parser() -> argparse.ArgumentParser:
    here = Path(__file__).resolve().parent.parent
    p = argparse.ArgumentParser(description="onnx2coreml flat-directory batch harness")
    p.add_argument("--onnx-dir", default=str(here / "tmp_test_onnx" / "manis_onnx"))
    p.add_argument("--out-dir", default=str(here / "tmp_test_output"))
    p.add_argument("--formats", default="mlpackage,mlmodel",
                   help="comma-separated subset of: mlpackage,mlmodel")
    p.add_argument("--precision", default="fp16", choices=["fp16", "fp32"])
    p.add_argument("--timeout", type=float, default=1200.0)
    p.add_argument("--pass-psnr", type=float, default=40.0)
    p.add_argument("--weak-psnr", type=float, default=20.0)
    p.add_argument("--only", default=None, help="comma-separated model stems")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--no-save", action="store_true", help="do not write artifacts")
    p.add_argument("--no-verify", action="store_true", help="convert (+save) only")
    # worker-mode flags
    p.add_argument("--run-one", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--onnx", help=argparse.SUPPRESS)
    p.add_argument("--result", help=argparse.SUPPRESS)
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    bad = [f for f in args.formats.split(",") if f not in _VALID_FORMATS]
    if bad:
        raise SystemExit(f"unknown format(s) {bad}; expected subset of {_VALID_FORMATS}")
    if args.run_one:
        return _worker_main(args)
    opts = Options(
        onnx_dir=Path(args.onnx_dir),
        out_dir=Path(args.out_dir),
        formats=tuple(args.formats.split(",")),
        precision=args.precision,
        timeout=args.timeout,
        pass_psnr=args.pass_psnr,
        weak_psnr=args.weak_psnr,
        save=not args.no_save,
        verify=not args.no_verify,
    )
    only = set(args.only.split(",")) if args.only else None
    orchestrate(opts, only, args.limit)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
