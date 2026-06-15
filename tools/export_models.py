# Copyright 2026 onnx2coreml contributors.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""Export converted Core ML models, sorted by accuracy.

For every ONNX graph in the zoo that converts and **meets the accuracy bar**
(fp16 Core ML vs fp32 ONNX Runtime, PSNR >= ``--min-psnr``), this writes both
container formats:

    tmp_test_output/mlpackage/<model>__<graph>.mlpackage   (ML Program, fp16)
    tmp_test_output/mlmodel/<model>__<graph>.mlmodel        (NeuralNetwork)

A graph that fails to convert or misses the bar has its ONNX (graph + external
``.data`` + metadata) copied to ``tmp_test_output/failure_onnx/<model>/``. The
source zoo zips are never modified. A per-graph PSNR table is written to
``tmp_test_output/accuracy_report.md``.

The ``.mlmodel`` is best-effort: the iOS15 NeuralNetwork backend cannot carry a
few ops (e.g. ``resample`` for GridSample), so such graphs get the ``.mlpackage``
only — they are not treated as failures.

Each graph runs in its own subprocess (timeout + crash isolation), mirroring
``zoo_eval``.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

# zoo_eval is a sibling module; Python puts the script's directory on sys.path.
from zoo_eval import (
    SEED,
    Options,
    _is_ane_build_error,
    build_inputs,
    extract_bundle,
    find_metadata,
    find_onnx_files,
)

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
_MIN_PSNR = 20.0


# --------------------------------------------------------------------------- #
# Worker: convert + verify + save one graph in both formats
# --------------------------------------------------------------------------- #
def _verify_psnr(proto, mlmodel, inputs):
    from onnx2coreml._verify import verify_model

    report = verify_model(proto, mlmodel, inputs=inputs)
    psnrs = [o.psnr for o in report.outputs if o.psnr == o.psnr and o.psnr != float("inf")]
    return min(psnrs) if psnrs else float("inf")


def _export_one(onnx_path: Path, metadata_path: Path | None, out: dict, args) -> dict:
    import numpy as np
    import onnx

    import onnx2coreml as o2c

    metadata = json.loads(metadata_path.read_text()) if metadata_path and metadata_path.exists() else None
    proto = onnx.load_model(str(onnx_path))
    inputs, _ = build_inputs(proto, metadata, onnx_path.name, np.random.default_rng(SEED))
    target, name = args.target or None, out["name"]

    # --- ML Program (.mlpackage, fp16) ---------------------------------------
    mlpkg = o2c.convert(proto, format="mlpackage", compute_precision="fp16", minimum_deployment_target=target)
    try:
        out["mlpackage_psnr"] = _verify_psnr(proto, mlpkg, inputs)
    except BaseException as exc:  # ANE compile failure -> retry CPU+GPU
        if not _is_ane_build_error(exc):
            raise
        mlpkg = o2c.convert(
            proto, format="mlpackage", compute_precision="fp16",
            minimum_deployment_target=target, compute_units="cpu_and_gpu",
        )
        out["mlpackage_psnr"] = _verify_psnr(proto, mlpkg, inputs)
        out["note"] = "verified on CPU+GPU (ANE compile failed)"
    mlpkg.save(str(Path(args.out_dir) / "mlpackage" / f"{name}.mlpackage"))
    out["mlpackage"] = True

    # --- NeuralNetwork (.mlmodel), best-effort -------------------------------
    try:
        mlm = o2c.convert(proto, format="mlmodel")
        out["mlmodel_psnr"] = _verify_psnr(proto, mlm, inputs)
        if out["mlmodel_psnr"] >= _MIN_PSNR:
            mlm.save(str(Path(args.out_dir) / "mlmodel" / f"{name}.mlmodel"))
            out["mlmodel"] = True
        else:
            out["mlmodel_skip"] = "below accuracy bar"
    except BaseException as exc:  # NN backend may not support an op
        out["mlmodel_skip"] = str(exc)[:160]
    return out


def _worker_main(args) -> int:
    out: dict = {"name": args.name, "model": args.model, "onnx": Path(args.onnx).name}
    try:
        out = _export_one(Path(args.onnx), Path(args.metadata) if args.metadata else None, out, args)
        out["status"] = "ok"
    except BaseException as exc:  # last-resort guard for the subprocess
        out["status"] = "error"
        out["error"] = str(exc)[:300]
    Path(args.result).write_text(json.dumps(out), encoding="utf-8")
    return 0


# --------------------------------------------------------------------------- #
# Orchestrator
# --------------------------------------------------------------------------- #
def _copy_failure(onnx_path: Path, model: str, fail_dir: Path) -> None:
    """Copy a graph's ONNX, its external .data, and metadata into failure_onnx."""
    dest = fail_dir / model
    dest.mkdir(parents=True, exist_ok=True)
    for f in [onnx_path, onnx_path.with_suffix(".data"), onnx_path.parent / "metadata.json"]:
        if f.exists():
            shutil.copy2(f, dest / f.name)


def _spawn(onnx_path, metadata_path, model, name, args):
    result_file = Path(args.out_dir) / "_result.json"
    result_file.unlink(missing_ok=True)
    cmd = [
        sys.executable, str(Path(__file__).resolve()), "--worker",
        "--model", model, "--name", name, "--onnx", str(onnx_path),
        "--result", str(result_file), "--out-dir", args.out_dir, "--target", args.target or "",
    ]
    if metadata_path:
        cmd += ["--metadata", str(metadata_path)]
    try:
        proc = subprocess.run(cmd, timeout=args.timeout, capture_output=True, text=True)
    except subprocess.TimeoutExpired:
        return {"name": name, "model": model, "onnx": onnx_path.name, "status": "error",
                "error": f"timeout {args.timeout:g}s"}
    if result_file.exists():
        data = json.loads(result_file.read_text())
        result_file.unlink(missing_ok=True)
        return data
    tail = (proc.stderr or proc.stdout or "").strip().splitlines()[-3:]
    return {"name": name, "model": model, "onnx": onnx_path.name, "status": "error",
            "error": "crash: " + " | ".join(tail)}


def _meets_bar(rep: dict) -> bool:
    return rep.get("status") == "ok" and rep.get("verdict") in ("pass", "weak")


def orchestrate(args) -> None:
    out_root = Path(args.out_dir)
    for sub in ("mlpackage", "mlmodel", "failure_onnx"):
        (out_root / sub).mkdir(parents=True, exist_ok=True)
    report = {f"{r['model']}/{r['onnx']}": r for r in json.loads(Path(args.report).read_text())["results"]}
    opts = Options(models_dir=Path(args.models_dir), out_dir=out_root, work_dir=Path(args.work_dir))

    bundles = sorted(opts.models_dir.glob("*.zip"))
    if args.only:
        wanted = set(args.only.split(","))
        bundles = [b for b in bundles if b.stem.replace("-onnx-float", "") in wanted]
    rows: list[dict] = []
    started = time.perf_counter()
    print(f"[export] {len(bundles)} bundles -> {out_root}", flush=True)
    for bi, zip_path in enumerate(bundles, 1):
        model = zip_path.stem.replace("-onnx-float", "")
        extracted = extract_bundle(zip_path, opts)
        metadata_path = find_metadata(extracted)
        for onnx_path in find_onnx_files(extracted):
            key = f"{model}/{onnx_path.name}"
            name = f"{model}__{onnx_path.stem}"
            rep = report.get(key, {"status": "missing", "verdict": "n/a"})
            if _meets_bar(rep):
                row = _spawn(onnx_path, metadata_path, model, name, args)
                row["gate_psnr"] = rep.get("min_psnr")
            else:
                _copy_failure(onnx_path, model, out_root / "failure_onnx")
                row = {"name": name, "model": model, "onnx": onnx_path.name, "status": "failure_onnx",
                       "gate_psnr": rep.get("min_psnr"),
                       "reason": rep.get("failure_class") or rep.get("verdict") or rep.get("status")}
            rows.append(row)
            print(f"[{bi}/{len(bundles)}] {name}: {_line(row)}", flush=True)
        if not opts.keep_extracted:
            shutil.rmtree(extracted, ignore_errors=True)

    _write_md(out_root / "accuracy_report.md", rows, started)
    print(f"\n[export] report: {out_root / 'accuracy_report.md'}", flush=True)


def _fmt(p) -> str:
    if p is None:
        return "—"
    return "inf" if p == float("inf") else f"{p:.1f}"


def _line(row: dict) -> str:
    if row["status"] == "failure_onnx":
        return f"FAILURE_ONNX ({row.get('reason')}, {_fmt(row.get('gate_psnr'))} dB)"
    if row["status"] == "ok":
        mm = "+mlmodel" if row.get("mlmodel") else f"(mlpackage only: {row.get('mlmodel_skip','')[:40]})"
        return f"mlpackage {_fmt(row.get('mlpackage_psnr'))}dB {mm}"
    return f"ERROR {row.get('error','')[:80]}"


def _write_md(path: Path, rows: list[dict], started: float) -> None:
    gen = [r for r in rows if r["status"] == "ok"]
    both = [r for r in gen if r.get("mlmodel")]
    failed = [r for r in rows if r["status"] == "failure_onnx"]
    errs = [r for r in rows if r["status"] == "error"]
    lines = [
        "# onnx2coreml — conversion & accuracy report",
        "",
        f"Accuracy: fp16 Core ML (.mlpackage) vs fp32 ONNX Runtime (CPU). Bar: min PSNR ≥ {_MIN_PSNR:g} dB.",
        "",
        f"- **{len(gen)}** graphs met the bar → `.mlpackage` written ({len(both)} also `.mlmodel`).",
        f"- **{len(failed)}** graphs below the bar or non-converting → ONNX copied to `failure_onnx/`.",
        f"- **{len(errs)}** export errors.",
        f"- elapsed: {time.perf_counter() - started:.0f}s",
        "",
        "## Generated models",
        "",
        "| model / graph | mlpackage PSNR (dB) | mlmodel PSNR (dB) | mlmodel | note |",
        "|---|---|---|---|---|",
    ]
    for r in sorted(gen, key=lambda r: r["name"]):
        mm = "✓" if r.get("mlmodel") else f"— ({r.get('mlmodel_skip','')[:48]})"
        lines.append(
            f"| {r['name']} | {_fmt(r.get('mlpackage_psnr'))} | {_fmt(r.get('mlmodel_psnr'))} | {mm} | {r.get('note','')} |"
        )
    lines += ["", "## failure_onnx (below bar or non-converting)", "",
              "| model / graph | gate PSNR (dB) | reason |", "|---|---|---|"]
    for r in sorted(failed, key=lambda r: r["name"]):
        lines.append(f"| {r['name']} | {_fmt(r.get('gate_psnr'))} | {r.get('reason')} |")
    if errs:
        lines += ["", "## export errors", "", "| model / graph | error |", "|---|---|"]
        for r in sorted(errs, key=lambda r: r["name"]):
            lines.append(f"| {r['name']} | {r.get('error','')[:120]} |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Export Core ML models sorted by accuracy")
    p.add_argument("--models-dir", default=str(_ROOT / "tmp_test_onnx"))
    p.add_argument("--work-dir", default=str(_ROOT / "tmp_test_onnx" / "_work"))
    p.add_argument("--out-dir", default=str(_ROOT / "tmp_test_output"))
    p.add_argument("--report", default=str(_ROOT / "tmp_test_onnx" / "_artifacts_v3" / "report.json"))
    p.add_argument("--target", default="iOS17")
    p.add_argument("--timeout", type=float, default=1500.0)
    p.add_argument("--only", default=None)
    p.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--model", help=argparse.SUPPRESS)
    p.add_argument("--name", help=argparse.SUPPRESS)
    p.add_argument("--onnx", help=argparse.SUPPRESS)
    p.add_argument("--metadata", help=argparse.SUPPRESS)
    p.add_argument("--result", help=argparse.SUPPRESS)
    args = p.parse_args(argv)
    if args.worker:
        return _worker_main(args)
    orchestrate(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
