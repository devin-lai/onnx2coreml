# Copyright 2026 onnx2coreml contributors.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""Command-line interface: ``onnx2coreml <convert|inspect|verify|schema>``.

Each subcommand supports ``--json`` for machine-readable output and exits
non-zero on failure. Human output is terse and actionable; the JSON output is
the stable contract for scripts.
"""

from __future__ import annotations

import argparse
import json
import sys

from . import analyze, convert
from .__version__ import __url__, __version__
from ._coverage import supported_ops
from ._io import save
from ._target import Format
from .errors import Onnx2CoreMLError

# Stable error-code list, surfaced by ``schema`` and used by callers to branch
# on failure class without parsing messages.
_ERROR_CODES = [
    "Onnx2CoreMLError",
    "ModelValidationError",
    "UnsupportedOpError",
    "ConversionError",
    "TargetError",
]


def main(argv: list[str] | None = None) -> int:
    """Entry point for the ``onnx2coreml`` console script.

    Returns a process exit code (0 success, non-zero failure).
    """
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.command is None:
        parser.print_help(sys.stderr)
        return 2
    try:
        return args.func(args)
    except Onnx2CoreMLError as exc:
        return _fail(getattr(args, "json", False), exc)
    except FileNotFoundError as exc:
        return _fail(getattr(args, "json", False), exc)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="onnx2coreml",
        description="Convert ONNX models to Apple Core ML (.mlpackage / .mlmodel).",
        epilog=f"Project home and documentation: {__url__}",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"onnx2coreml {__version__}\n{__url__}",
    )
    sub = parser.add_subparsers(dest="command")

    p_conv = sub.add_parser("convert", help="convert an ONNX model to Core ML")
    p_conv.add_argument("model", help="path to the input .onnx model")
    p_conv.add_argument("-o", "--output", required=True, help="output .mlpackage / .mlmodel path")
    p_conv.add_argument(
        "--format", choices=["mlpackage", "mlmodel"], default="mlpackage",
        help="output container (default: mlpackage / ML Program)",
    )
    p_conv.add_argument("--target", default=None, help="minimum deployment target, e.g. iOS17")
    p_conv.add_argument(
        "--precision", choices=["fp16", "fp32"], default=None,
        help="compute precision (ML Program only; default fp16)",
    )
    p_conv.add_argument(
        "--compute-units", choices=["all", "cpu_only", "cpu_and_gpu", "cpu_and_ne"],
        default=None, help="compute units (default: coremltools default)",
    )
    p_conv.add_argument("--no-fuse", action="store_true", help="disable graph fusion passes")
    p_conv.add_argument(
        "--verify", action="store_true",
        help="run numerical parity against ONNX Runtime after conversion",
    )
    p_conv.add_argument("--json", action="store_true", help="machine-readable output")
    p_conv.set_defaults(func=_cmd_convert)

    p_insp = sub.add_parser("inspect", help="report op coverage / convertibility")
    p_insp.add_argument("model", help="path to the input .onnx model")
    p_insp.add_argument("--json", action="store_true", help="machine-readable output")
    p_insp.set_defaults(func=_cmd_inspect)

    p_ver = sub.add_parser("verify", help="check Core ML parity against the ONNX source")
    p_ver.add_argument("model", help="path to the input .onnx model")
    p_ver.add_argument("coreml", help="path to the converted .mlpackage / .mlmodel")
    p_ver.add_argument("--rtol", type=float, default=1e-3, help="relative tolerance (default 1e-3)")
    p_ver.add_argument("--atol", type=float, default=1e-4, help="absolute tolerance (default 1e-4)")
    p_ver.add_argument("--min-psnr", type=float, default=None, help="minimum required PSNR (dB)")
    p_ver.add_argument("--json", action="store_true", help="machine-readable output")
    p_ver.set_defaults(func=_cmd_verify)

    p_schema = sub.add_parser("schema", help="print version, op count, and error codes")
    p_schema.add_argument("--json", action="store_true", help="machine-readable output")
    p_schema.set_defaults(func=_cmd_schema)

    return parser


def _cmd_convert(args: argparse.Namespace) -> int:
    mlmodel = convert(
        args.model,
        format=args.format,
        minimum_deployment_target=args.target,
        compute_precision=args.precision,
        compute_units=args.compute_units,
        fuse=not args.no_fuse,
    )
    out_path = save(mlmodel, args.output)

    report = None
    if args.verify:
        from ._io import load
        from ._verify import verify_model

        report = verify_model(load(args.model), mlmodel)

    if args.json:
        payload: dict = {"output": out_path, "format": args.format}
        if report is not None:
            payload["verify"] = report.as_dict()
        _emit(payload)
    else:
        print(f"wrote {out_path}")
        if report is not None:
            print(report)
    if report is not None and not report.passed:
        return 1
    return 0


def _cmd_inspect(args: argparse.Namespace) -> int:
    report = analyze(args.model)
    # Per-format convertibility: .mlmodel (NeuralNetwork) shares the same lowering
    # coverage in this version, so convertibility is format-independent here.
    convertible = report.convertible
    formats = {fmt.value: convertible for fmt in Format}

    if args.json:
        _emit(
            {
                "supported": report.supported,
                "unsupported": report.unsupported,
                "convertible": convertible,
                "formats": formats,
            }
        )
    else:
        total = sum(report.supported.values())
        print(f"ops: {total} node(s), {len(report.supported)} distinct supported type(s)")
        for key in sorted(report.supported):
            print(f"  {key}: {report.supported[key]}")
        if report.unsupported:
            print(f"unsupported: {len(report.unsupported)} type(s)")
            for key in sorted(report.unsupported):
                nodes = report.unsupported[key]
                print(f"  {key}: {len(nodes)} node(s)")
        for name, ok in formats.items():
            print(f"{name}: {'convertible' if ok else 'NOT convertible'}")
    return 0 if convertible else 1


def _cmd_verify(args: argparse.Namespace) -> int:
    from ._verify import verify

    report = verify(
        args.model,
        args.coreml,
        rtol=args.rtol,
        atol=args.atol,
        min_psnr=args.min_psnr,
    )
    if args.json:
        _emit(report.as_dict())
    else:
        print(report)
    return 0 if report.passed else 1


def _cmd_schema(args: argparse.Namespace) -> int:
    payload = {
        "version": __version__,
        "supported_op_count": len(supported_ops()),
        "error_codes": _ERROR_CODES,
    }
    if args.json:
        _emit(payload)
    else:
        print(f"onnx2coreml {payload['version']}")
        print(f"supported ops: {payload['supported_op_count']}")
        print("error codes: " + ", ".join(_ERROR_CODES))
    return 0


def _emit(payload: dict) -> None:
    # allow_nan=False: any stray non-finite float is a bug, not silently emitted
    # as non-standard JSON. PSNR ``inf`` is already stringified upstream.
    json.dump(payload, sys.stdout, indent=2, allow_nan=False)
    sys.stdout.write("\n")


def _fail(as_json: bool, exc: BaseException) -> int:
    code = type(exc).__name__
    if as_json:
        json.dump({"error": code, "message": str(exc)}, sys.stdout, indent=2)
        sys.stdout.write("\n")
    else:
        print(f"error [{code}]: {exc}", file=sys.stderr)
    return 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
