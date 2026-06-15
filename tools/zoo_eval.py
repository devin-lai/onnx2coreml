# Copyright 2026 onnx2coreml contributors.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""Batch model-zoo evaluation harness for onnx2coreml.

Drives the *public* converter API over a directory of ``*.zip`` model bundles
(each holding one or more ``.onnx`` graphs + external ``.data`` weights +
``metadata.json``). For every ONNX graph it:

1. extracts the bundle and locates each ``.onnx`` (a bundle may hold several,
   e.g. encoder/decoder or CLIP image/text);
2. converts it to a Core ML ``.mlpackage`` at **fp16** compute precision;
3. validates accuracy by running the *same* seeded inputs through ONNX Runtime
   on CPU in **fp32** (the reference) and the Core ML model, reporting per-output
   PSNR / max-abs / max-rel error;
4. classifies any failure (unsupported op, dynamic shape, lowering error, predict
   error, timeout, crash) so the aggregate report directly drives general fixes.

Each model runs in its own subprocess with a timeout, so a single hang or native
crash never sinks the sweep. Results are written incrementally to ``report.json``
and summarized at the end, including a missing-op histogram ranked by how many
models each unsupported op blocks.

Usage::

    python tools/zoo_eval.py                      # full sweep
    python tools/zoo_eval.py --only mobilenet_v2,resnet18
    python tools/zoo_eval.py --limit 5            # first 5 bundles
    python tools/zoo_eval.py --timeout 1200 --pass-psnr 40 --weak-psnr 20

The harness is a development/CI tool; it is intentionally not part of the shipped
package and lives under ``tools/``.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import zipfile
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path

# Seed shared with onnx2coreml._verify so generated inputs are reproducible and
# identical to the library's own verification path.
SEED = 0x0117C0DE


# --------------------------------------------------------------------------- #
# Options
# --------------------------------------------------------------------------- #
@dataclass
class Options:
    models_dir: Path
    out_dir: Path
    work_dir: Path
    fmt: str = "mlpackage"
    precision: str = "fp16"
    target: str | None = "iOS17"
    timeout: float = 1800.0
    pass_psnr: float = 40.0
    weak_psnr: float = 20.0
    keep_mlpackage: bool = False
    keep_extracted: bool = False

    @property
    def artifacts_dir(self) -> Path:
        return self.out_dir / "mlpackages"


# --------------------------------------------------------------------------- #
# Per-output / per-model result records (JSON-serializable)
# --------------------------------------------------------------------------- #
@dataclass
class Result:
    model: str
    onnx: str
    zip: str
    status: str = "error"  # ok | convert_failed | verify_failed | error
    verdict: str = "n/a"  # pass | weak | fail | n/a
    convert_ok: bool = False
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
    input_shapes: dict[str, list[int]] = field(default_factory=dict)
    convert_s: float | None = None
    verify_s: float | None = None
    notes: str | None = None


# --------------------------------------------------------------------------- #
# Bundle discovery / extraction (orchestrator side — stdlib only)
# --------------------------------------------------------------------------- #
def discover_bundles(opts: Options) -> list[Path]:
    return sorted(opts.models_dir.glob("*.zip"))


def extract_bundle(zip_path: Path, opts: Options) -> Path:
    """Extract ``zip_path`` into ``work_dir/<stem>`` idempotently; return that dir."""
    dest = opts.work_dir / zip_path.stem
    marker = dest / ".extracted"
    if marker.exists():
        return dest
    dest.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(dest)
    marker.write_text("ok\n", encoding="utf-8")
    return dest


def find_onnx_files(extracted: Path) -> list[Path]:
    return sorted(extracted.rglob("*.onnx"))


def find_metadata(extracted: Path) -> Path | None:
    hits = sorted(extracted.rglob("metadata.json"))
    return hits[0] if hits else None


# --------------------------------------------------------------------------- #
# Input synthesis (worker side) — metadata-aware, seeded, shared by both runtimes
# --------------------------------------------------------------------------- #
def _metadata_shapes(metadata: dict | None, onnx_filename: str) -> dict[str, list[int]]:
    """Map input-name -> static shape from a bundle's metadata.json, if present."""
    if not metadata:
        return {}
    files = metadata.get("model_files", {})
    entry = files.get(onnx_filename)
    if entry is None and len(files) == 1:
        entry = next(iter(files.values()))
    if not entry:
        return {}
    out: dict[str, list[int]] = {}
    for name, spec in entry.get("inputs", {}).items():
        shape = spec.get("shape")
        if shape and all(isinstance(d, int) and d > 0 for d in shape):
            out[name] = list(shape)
    return out


def _metadata_ranges(metadata: dict | None, onnx_filename: str) -> dict[str, tuple]:
    if not metadata:
        return {}
    files = metadata.get("model_files", {})
    entry = files.get(onnx_filename)
    if entry is None and len(files) == 1:
        entry = next(iter(files.values()))
    if not entry:
        return {}
    out: dict[str, tuple] = {}
    for name, spec in entry.get("inputs", {}).items():
        rng = spec.get("value_range")
        if rng and len(rng) == 2:
            out[name] = (float(rng[0]), float(rng[1]))
    return out


def build_inputs(proto, metadata, onnx_filename, rng):
    """Seeded inputs for every non-initializer graph input.

    Shapes come from the graph; any dynamic dim is filled from metadata.json.
    Floats are drawn from the input's metadata ``value_range`` (default ``[0,1)``);
    integers get small non-negative values valid as indices/counts.
    """
    import numpy as np
    import onnx

    md_shapes = _metadata_shapes(metadata, onnx_filename)
    md_ranges = _metadata_ranges(metadata, onnx_filename)
    init = {i.name for i in proto.graph.initializer}
    feed: dict = {}
    shapes: dict[str, list[int]] = {}
    for vi in proto.graph.input:
        if vi.name in init:
            continue
        tt = vi.type.tensor_type
        shape: list[int] = []
        dynamic = False
        for d in tt.shape.dim:
            if d.HasField("dim_value") and d.dim_value > 0:
                shape.append(d.dim_value)
            else:
                dynamic = True
                break
        if dynamic:
            if vi.name not in md_shapes:
                raise ValueError(
                    f"input '{vi.name}' has a dynamic dim and no static shape in "
                    "metadata.json"
                )
            shape = md_shapes[vi.name]
        np_dtype = onnx.helper.tensor_dtype_to_np_dtype(tt.elem_type)
        kind = np.dtype(np_dtype).kind
        if kind == "f":
            lo, hi = md_ranges.get(vi.name, (0.0, 1.0))
            arr = rng.random(shape, dtype=np.float32)
            arr = lo + (hi - lo) * arr
            feed[vi.name] = arr.astype(np_dtype)
        elif kind == "b":
            feed[vi.name] = rng.integers(0, 2, size=shape).astype(np.bool_)
        elif kind in "iu":
            # Binary 0/1: valid as an attention mask (values > 1 break the common
            # (1 - mask) * -BIG masking and produce NaN even in ONNX Runtime), and
            # safe as token ids / indices.
            feed[vi.name] = rng.integers(0, 2, size=shape).astype(np_dtype)
        else:
            raise ValueError(f"input '{vi.name}' has unsupported dtype {np_dtype}")
        shapes[vi.name] = shape
    return feed, shapes


# --------------------------------------------------------------------------- #
# Failure classification (worker side)
# --------------------------------------------------------------------------- #
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


# --------------------------------------------------------------------------- #
# The actual per-model work (worker side)
# --------------------------------------------------------------------------- #
def run_one(onnx_path: Path, metadata_path: Path | None, result: Result, opts: Options) -> Result:
    import numpy as np
    import onnx

    import onnx2coreml as o2c
    from onnx2coreml._verify import verify_model

    metadata = None
    if metadata_path and metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

    proto = onnx.load_model(str(onnx_path))

    # 1) Convert ---------------------------------------------------------------
    t0 = time.perf_counter()
    try:
        mlmodel = o2c.convert(
            proto,
            format=opts.fmt,
            compute_precision=opts.precision,
            minimum_deployment_target=opts.target,
        )
    except BaseException as exc:
        result.convert_s = time.perf_counter() - t0
        result.status = "convert_failed"
        result.__dict__.update(classify_error(exc))
        return result
    result.convert_ok = True
    result.convert_s = time.perf_counter() - t0

    if opts.keep_mlpackage:
        opts.artifacts_dir.mkdir(parents=True, exist_ok=True)
        suffix = ".mlpackage" if opts.fmt == "mlpackage" else ".mlmodel"
        mlmodel.save(str(opts.artifacts_dir / f"{result.model}__{onnx_path.stem}{suffix}"))

    # 2) Build inputs ----------------------------------------------------------
    try:
        rng = np.random.default_rng(SEED)
        inputs, shapes = build_inputs(proto, metadata, onnx_path.name, rng)
        result.input_shapes = shapes
    except BaseException as exc:
        result.status = "verify_failed"
        result.failure_class = "input_synthesis"
        result.error_type = type(exc).__name__
        result.error_message = str(exc)[:2000]
        return result

    # 3) Verify (fp16 Core ML vs fp32 ORT-CPU) --------------------------------
    t1 = time.perf_counter()
    try:
        report = verify_model(proto, mlmodel, inputs=inputs)
    except BaseException as exc:
        # Some large graphs fail to compile on the Apple Neural Engine ("Error in
        # building plan" / "ANECCompile FAILED"). The conversion is fine — the ANE
        # compiler is the limit — so retry on CPU+GPU, which excludes the ANE.
        if _is_ane_build_error(exc):
            try:
                mlmodel = o2c.convert(
                    proto, format=opts.fmt, compute_precision=opts.precision,
                    minimum_deployment_target=opts.target, compute_units="cpu_and_gpu",
                )
                report = verify_model(proto, mlmodel, inputs=inputs)
                result.notes = "ANE compilation failed; verified on CPU+GPU"
            except BaseException as exc2:
                exc = exc2
                report = None
        else:
            report = None
        if report is None:
            result.verify_s = time.perf_counter() - t1
            result.status = "verify_failed"
            result.failure_class = "predict_error"
            result.error_type = type(exc).__name__
            result.error_message = str(exc)[:2000]
            return result
    result.verify_s = time.perf_counter() - t1
    result.verify_ran = True

    out_dicts = [o.as_dict() for o in report.outputs]
    result.outputs = out_dicts
    result.strict_parity = report.passed
    psnrs = [o.psnr for o in report.outputs if not _is_inf(o.psnr)]
    result.min_psnr = min(psnrs) if psnrs else float("inf")
    result.status = "ok"
    result.verdict = _verdict(report, opts)
    return result


def _is_inf(x: float) -> bool:
    return x == float("inf")


def _is_ane_build_error(exc: BaseException) -> bool:
    """Whether a predict failure is the Apple Neural Engine compiler giving up.

    Covers both phrasings Core ML uses ("Error in building plan", "Failed to build
    the model execution plan ... error code: -5") plus the explicit ANEF errors.
    """
    msg = str(exc).lower()
    return any(s in msg for s in ("building plan", "execution plan", "ane", "anecompile"))


def _verdict(report, opts: Options) -> str:
    if report.passed:  # within strict fp32 tolerance — unambiguously faithful
        return "pass"
    floats = [o.psnr for o in report.outputs if not _is_inf(o.psnr)]
    min_psnr = min(floats) if floats else float("inf")
    if min_psnr >= opts.pass_psnr:
        return "pass"
    if min_psnr >= opts.weak_psnr:
        return "weak"
    return "fail"


# --------------------------------------------------------------------------- #
# Worker entry point (one model, own process)
# --------------------------------------------------------------------------- #
def _worker_main(args: argparse.Namespace) -> int:
    opts = Options(
        models_dir=Path(args.models_dir),
        out_dir=Path(args.out_dir),
        work_dir=Path(args.work_dir),
        fmt=args.format,
        precision=args.precision,
        target=args.target or None,
        pass_psnr=args.pass_psnr,
        weak_psnr=args.weak_psnr,
        keep_mlpackage=args.keep_mlpackage,
    )
    result = Result(model=args.name, onnx=Path(args.onnx).name, zip=args.zip)
    meta = Path(args.metadata) if args.metadata else None
    try:
        result = run_one(Path(args.onnx), meta, result, opts)
    except BaseException as exc:
        result.status = "error"
        result.failure_class = "worker_exception"
        result.error_type = type(exc).__name__
        result.error_message = str(exc)[:2000]
    Path(args.result).write_text(json.dumps(asdict(result)), encoding="utf-8")
    return 0


# --------------------------------------------------------------------------- #
# Orchestrator
# --------------------------------------------------------------------------- #
def _spawn_one(
    onnx_path: Path,
    metadata_path: Path | None,
    model_name: str,
    zip_name: str,
    opts: Options,
) -> Result:
    """Run one model in an isolated subprocess with a timeout."""
    result_file = opts.out_dir / "_result.json"
    if result_file.exists():
        result_file.unlink()
    cmd = [
        sys.executable, str(Path(__file__).resolve()),
        "--run-one",
        "--name", model_name,
        "--zip", zip_name,
        "--onnx", str(onnx_path),
        "--result", str(result_file),
        "--models-dir", str(opts.models_dir),
        "--out-dir", str(opts.out_dir),
        "--work-dir", str(opts.work_dir),
        "--format", opts.fmt,
        "--precision", opts.precision,
        "--target", opts.target or "",
        "--pass-psnr", str(opts.pass_psnr),
        "--weak-psnr", str(opts.weak_psnr),
    ]
    if metadata_path:
        cmd += ["--metadata", str(metadata_path)]
    if opts.keep_mlpackage:
        cmd += ["--keep-mlpackage"]

    base = Result(model=model_name, onnx=onnx_path.name, zip=zip_name)
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
        return Result(**data)

    # No result file => the worker died (segfault / OOM / native abort).
    base.status = "error"
    base.failure_class = "crash"
    tail = (proc.stderr or proc.stdout or "").strip().splitlines()[-5:]
    base.error_message = f"exit={proc.returncode}; " + " | ".join(tail)
    return base


def orchestrate(opts: Options, only: set[str] | None, limit: int | None) -> None:
    opts.out_dir.mkdir(parents=True, exist_ok=True)
    opts.work_dir.mkdir(parents=True, exist_ok=True)
    bundles = discover_bundles(opts)
    if only:
        bundles = [b for b in bundles if b.stem.replace("-onnx-float", "") in only or b.stem in only]
    if limit:
        bundles = bundles[:limit]

    report_path = opts.out_dir / "report.json"
    results: list[Result] = []
    started = time.perf_counter()
    print(f"[zoo] {len(bundles)} bundles -> {opts.out_dir}", flush=True)

    for bi, zip_path in enumerate(bundles, 1):
        model_name = zip_path.stem.replace("-onnx-float", "")
        try:
            extracted = extract_bundle(zip_path, opts)
        except Exception as exc:
            r = Result(model=model_name, onnx="", zip=zip_path.name, status="error",
                       failure_class="extract_error", error_message=str(exc)[:500])
            results.append(r)
            _write_report(report_path, results, opts, started)
            print(f"[{bi}/{len(bundles)}] {model_name}: EXTRACT FAIL {exc}", flush=True)
            continue

        metadata_path = find_metadata(extracted)
        onnx_files = find_onnx_files(extracted)
        if not onnx_files:
            r = Result(model=model_name, onnx="", zip=zip_path.name, status="error",
                       failure_class="no_onnx", error_message="no .onnx in bundle")
            results.append(r)
            _write_report(report_path, results, opts, started)
            print(f"[{bi}/{len(bundles)}] {model_name}: NO ONNX", flush=True)
            continue

        for onnx_path in onnx_files:
            r = _spawn_one(onnx_path, metadata_path, model_name, zip_path.name, opts)
            results.append(r)
            _write_report(report_path, results, opts, started)
            print(f"[{bi}/{len(bundles)}] {model_name}/{onnx_path.name}: "
                  f"{_one_line(r)}", flush=True)

        if not opts.keep_extracted:
            _rmtree(extracted)

    _print_summary(results, opts)
    print(f"\n[zoo] report: {report_path}", flush=True)


def _one_line(r: Result) -> str:
    if r.status == "ok":
        psnr = "inf" if r.min_psnr == float("inf") else f"{r.min_psnr:.1f}dB"
        return f"{r.verdict.upper()} (min_psnr={psnr}, {r.convert_s:.0f}+{r.verify_s:.0f}s)"
    if r.failure_class == "unsupported_op":
        return f"CONVERT_FAIL unsupported: {','.join(r.missing_ops or [])}"
    detail = r.error_op or r.error_message or ""
    return f"{r.status.upper()} [{r.failure_class}] {detail}"[:160]


def _write_report(path: Path, results: list[Result], opts: Options, started: float) -> None:
    payload = {
        "options": {
            "format": opts.fmt, "precision": opts.precision, "target": opts.target,
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
    convert_ok = [r for r in results if r.convert_ok]
    verify_ran = [r for r in results if r.verify_ran]
    verdicts = Counter(r.verdict for r in verify_ran)
    classes = Counter(r.failure_class for r in results if r.failure_class)

    print("\n" + "=" * 64, flush=True)
    print(f"SUMMARY  ({n} graphs)", flush=True)
    print(f"  converted        : {len(convert_ok)}/{n}", flush=True)
    print(f"  accuracy-checked : {len(verify_ran)}/{n}", flush=True)
    print(f"  verdict pass     : {verdicts.get('pass', 0)}", flush=True)
    print(f"  verdict weak     : {verdicts.get('weak', 0)}", flush=True)
    print(f"  verdict fail     : {verdicts.get('fail', 0)}", flush=True)
    print("\n  failure classes:", flush=True)
    for cls, c in classes.most_common():
        print(f"    {cls:18s} {c}", flush=True)

    # Missing-op histogram, ranked by number of models each op blocks.
    op_models: dict[str, set[str]] = {}
    for r in results:
        for op in r.missing_ops or []:
            op_models.setdefault(op, set()).add(f"{r.model}/{r.onnx}")
    if op_models:
        print("\n  missing ops (by #graphs blocked):", flush=True)
        for op, models in sorted(op_models.items(), key=lambda kv: (-len(kv[1]), kv[0])):
            print(f"    {op:28s} {len(models)}", flush=True)


def _rmtree(path: Path) -> None:
    import shutil
    shutil.rmtree(path, ignore_errors=True)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def _build_parser() -> argparse.ArgumentParser:
    here = Path(__file__).resolve().parent.parent
    p = argparse.ArgumentParser(description="onnx2coreml model-zoo evaluation harness")
    p.add_argument("--models-dir", default=str(here / "tmp_test_onnx"))
    p.add_argument("--out-dir", default=str(here / "tmp_test_onnx" / "_artifacts"))
    p.add_argument("--work-dir", default=str(here / "tmp_test_onnx" / "_work"))
    p.add_argument("--format", default="mlpackage", choices=["mlpackage", "mlmodel"])
    p.add_argument("--precision", default="fp16", choices=["fp16", "fp32"])
    p.add_argument("--target", default="iOS17")
    p.add_argument("--timeout", type=float, default=1800.0)
    p.add_argument("--pass-psnr", type=float, default=40.0)
    p.add_argument("--weak-psnr", type=float, default=20.0)
    p.add_argument("--only", default=None, help="comma-separated model names")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--keep-mlpackage", action="store_true")
    p.add_argument("--keep-extracted", action="store_true")
    # worker-mode flags
    p.add_argument("--run-one", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--name", help=argparse.SUPPRESS)
    p.add_argument("--zip", help=argparse.SUPPRESS)
    p.add_argument("--onnx", help=argparse.SUPPRESS)
    p.add_argument("--metadata", help=argparse.SUPPRESS)
    p.add_argument("--result", help=argparse.SUPPRESS)
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.run_one:
        return _worker_main(args)
    opts = Options(
        models_dir=Path(args.models_dir),
        out_dir=Path(args.out_dir),
        work_dir=Path(args.work_dir),
        fmt=args.format,
        precision=args.precision,
        target=args.target or None,
        timeout=args.timeout,
        pass_psnr=args.pass_psnr,
        weak_psnr=args.weak_psnr,
        keep_mlpackage=args.keep_mlpackage,
        keep_extracted=args.keep_extracted,
    )
    only = set(args.only.split(",")) if args.only else None
    orchestrate(opts, only, args.limit)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
