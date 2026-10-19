"""Command-line interface, designed to be easy for both people and AI agents.

    cutout remove-bg photo.jpg                     # -> photo-no-background.png
    cutout remove-fg photo.jpg --fill inpaint
    cutout erase photo.jpg --region 120,40,380,200
    cutout batch ./photos -o ./out --json          # machine-readable output
    cutout mcp                                     # MCP server on stdio
    cutout serve                                   # web UI on :8000

Add --json to any command for a JSON result on stdout. Exit code is 0 on
success, 1 on error (with the error on stderr, or as {"error": ...} with --json).
"""

import argparse
import json
import sys

from . import agent_ops
from .backends import BackendError, get_backend


def _region(text: str) -> list[float]:
    try:
        parts = [float(v) for v in text.split(",")]
    except ValueError:
        parts = []
    if len(parts) != 4:
        raise argparse.ArgumentTypeError("region must be x0,y0,x1,y1")
    return parts


def _common(p: argparse.ArgumentParser, output: bool = True, model: bool = True, edges: bool = True) -> None:
    if output:
        p.add_argument("-o", "--output", help="output file or directory (default: next to the input)")
        p.add_argument("--overwrite", action="store_true", help="replace existing output files")
    if model:
        p.add_argument("--model", help="detection model id (see `cutout models`)")
    if edges:
        p.add_argument("--feather", type=float, default=0, help="soften mask edges by N px")
        p.add_argument("--shift", type=int, default=0, help="grow (+) / shrink (-) the mask by N px")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="cutout", description="Remove image backgrounds and objects.")
    parser.add_argument("--json", action="store_true", help="print results as JSON")
    parser.add_argument("--api-url", help="use a running Cutout Studio server (default: $CUTOUT_API_URL, else local)")
    # The same flags are accepted after the subcommand too (`cutout remove-bg x.jpg --json`).
    shared = argparse.ArgumentParser(add_help=False)
    shared.add_argument("--json", action="store_true", default=argparse.SUPPRESS, help="print results as JSON")
    shared.add_argument("--api-url", default=argparse.SUPPRESS, help="use a running Cutout Studio server")
    sub = parser.add_subparsers(dest="command", required=True)

    def cmd(name: str, **kw) -> argparse.ArgumentParser:
        return sub.add_parser(name, parents=[shared], **kw)

    cmd("models", help="list detection models")

    p = cmd("info", help="show image size")
    p.add_argument("input")

    p = cmd("remove-bg", help="keep the subject, remove the background")
    p.add_argument("input")
    p.add_argument("--bg-color", help="'#rrggbb' instead of transparency")
    _common(p)

    p = cmd("remove-fg", help="remove the subject, keep the background")
    p.add_argument("input")
    p.add_argument("--fill", choices=["inpaint", "transparent"], default="inpaint")
    p.add_argument("--bg-color", help="colour for the removed area (with --fill transparent)")
    _common(p)

    p = cmd("mask", help="save the AI subject mask (white = subject)")
    p.add_argument("input")
    _common(p, edges=False)

    p = cmd("apply-mask", help="cut out / remove using your own mask")
    p.add_argument("input")
    p.add_argument("mask", help="grayscale PNG, white = subject")
    p.add_argument("--mode", choices=["remove_background", "remove_foreground"], default="remove_background")
    p.add_argument("--fill", choices=["inpaint", "transparent"], default="inpaint")
    p.add_argument("--bg-color")
    _common(p, model=False)

    p = cmd("erase", help="erase boxes from the image with AI inpainting")
    p.add_argument("input")
    p.add_argument("--region", type=_region, action="append", required=True, metavar="x0,y0,x1,y1",
                   help="area to erase; repeat for several")
    p.add_argument("--shape", choices=["rect", "ellipse"], default="rect")
    p.add_argument("--units", choices=["px", "fraction"], default="px")
    p.add_argument("--fill", choices=["inpaint", "transparent"], default="inpaint")
    p.add_argument("--grow", type=int, default=0, help="enlarge the erased area by N px")
    _common(p, model=False, edges=False)

    p = cmd("batch", help="process many images")
    p.add_argument("inputs", nargs="+", help="files, directories or glob patterns")
    p.add_argument("--operation", choices=list(agent_ops.BATCH_OPERATIONS), default="remove_background")
    p.add_argument("--fill", choices=["inpaint", "transparent"], default="inpaint")
    p.add_argument("--bg-color")
    _common(p)

    p = cmd("warmup", help="download and load models")
    p.add_argument("--model")
    p.add_argument("--no-inpaint", action="store_true", help="skip the inpainting model")

    cmd("mcp", help="run the MCP server on stdio (for AI agents)")

    p = cmd("serve", help="run the web UI and HTTP API")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    return parser


def run(args: argparse.Namespace, backend) -> dict:
    c = args.command
    if c == "models":
        return backend.list_models()
    if c == "info":
        return agent_ops.image_info(args.input)
    if c == "remove-bg":
        return agent_ops.remove_background(
            args.input, args.output, args.bg_color, args.model, args.feather, args.shift, args.overwrite, backend
        )
    if c == "remove-fg":
        return agent_ops.remove_foreground(
            args.input, args.output, args.fill, args.bg_color, args.model, args.feather, args.shift,
            args.overwrite, backend,
        )
    if c == "mask":
        return agent_ops.get_mask(args.input, args.output, args.model, args.overwrite, backend)
    if c == "apply-mask":
        return agent_ops.apply_mask(
            args.input, args.mask, args.output, args.mode, args.fill, args.bg_color, args.feather, args.shift,
            args.overwrite, backend,
        )
    if c == "erase":
        return agent_ops.erase_regions(
            args.input, args.region, args.output, args.shape, args.units, args.fill, args.grow, args.overwrite, backend
        )
    if c == "batch":
        progress = None if args.json else (lambda i, n: print(f"[{i}/{n}]", end="\r", file=sys.stderr, flush=True))
        return agent_ops.batch(
            args.inputs, args.output, args.operation, args.fill, args.bg_color, args.model, args.feather,
            args.shift, args.overwrite, backend, progress,
        )
    if c == "warmup":
        return backend.warmup(args.model, not args.no_inpaint)
    raise AssertionError(c)


def _human(command: str, result: dict) -> str:
    if command == "models":
        return "\n".join(
            f"{'*' if m['id'] == result['default'] else ' '} {m['id']:<24} {m['label']}" for m in result["models"]
        )
    if command == "batch":
        lines = [f"{r['input_path']} -> {r['output_path']}" for r in result["results"]]
        lines += [f"FAILED {e['input_path']}: {e['error']}" for e in result["errors"]]
        lines.append(f"{result['processed']} done, {result['failed']} failed in {result['elapsed_s']} s")
        return "\n".join(lines)
    if "output_path" in result:
        return f"{result['output_path']} ({result['width']}x{result['height']}, {result['elapsed_s']} s)"
    return json.dumps(result, indent=2)


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)

    if args.command == "mcp":
        if args.api_url:
            import os

            os.environ["CUTOUT_API_URL"] = args.api_url
        from .mcp_server import main as mcp_main

        mcp_main()
        return 0
    if args.command == "serve":
        import uvicorn

        uvicorn.run("app.main:app", host=args.host, port=args.port)
        return 0

    try:
        result = run(args, get_backend(args.api_url))
    except (ValueError, OSError, BackendError) as exc:
        if args.json:
            print(json.dumps({"error": str(exc)}))
        else:
            print(f"error: {exc}", file=sys.stderr)
        return 1

    print(json.dumps(result, indent=2) if args.json else _human(args.command, result))
    return 1 if args.command == "batch" and result["failed"] and not result["processed"] else 0


if __name__ == "__main__":
    sys.exit(main())
