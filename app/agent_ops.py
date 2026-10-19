"""File-path level operations for AI agents (used by the CLI and the MCP server).

Every operation reads image files from disk, runs them through a backend and
writes PNG results, returning a JSON-serialisable dict that describes what was
written.
"""

import glob
import io
import os
import time
from pathlib import Path
from typing import Iterable, Optional

from PIL import Image, ImageDraw, ImageOps

from .backends import get_backend
from .pipeline import Options

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}

SUFFIXES = {
    "remove_background": "no-background",
    "remove_foreground": "no-foreground",
    "mask": "mask",
    "apply_mask": "masked",
    "erase": "erased",
}


class AgentError(ValueError):
    """Bad arguments or paths; the message is meant to be shown to the agent."""


# ---------------------------------------------------------------- paths


def _allowed_dirs() -> list[Path]:
    raw = os.getenv("CUTOUT_ALLOWED_DIRS", "")
    return [Path(p).expanduser().resolve() for p in raw.split(os.pathsep) if p.strip()]


def _check_allowed(path: Path) -> None:
    allowed = _allowed_dirs()
    if allowed and not any(path == d or d in path.parents for d in allowed):
        raise AgentError(f"{path} is outside the allowed directories (CUTOUT_ALLOWED_DIRS)")


def resolve_input(path: str) -> Path:
    p = Path(path).expanduser().resolve()
    _check_allowed(p)
    if not p.is_file():
        raise AgentError(f"Input file not found: {p}")
    return p


def resolve_output(input_path: Path, output: Optional[str], kind: str, overwrite: bool = False) -> Path:
    """Pick where to write a result.

    `output` may be a file path, an existing directory (or one ending in a path
    separator), or None for "next to the input". Results are always PNG. Unless
    `overwrite` is set, an existing file is never replaced; -2, -3, ... is added.
    """
    default_name = f"{input_path.stem}-{SUFFIXES[kind]}.png"
    if output is None:
        out = input_path.with_name(default_name)
    else:
        out = Path(output).expanduser()
        if out.is_dir() or output.endswith(("/", os.sep)):
            out = out / default_name
        elif out.suffix.lower() != ".png":
            out = out.with_suffix(".png")
        out = out.resolve()
    _check_allowed(out)
    out.parent.mkdir(parents=True, exist_ok=True)

    if not overwrite:
        base, n = out, 2
        while out.exists():
            out = base.with_name(f"{base.stem}-{n}{base.suffix}")
            n += 1
    return out


def _write(out: Path, png: bytes, op: str, src: Path, started: float, **extra) -> dict:
    out.write_bytes(png)
    with Image.open(io.BytesIO(png)) as img:
        size, mode = img.size, img.mode
    return {
        "operation": op,
        "input_path": str(src),
        "output_path": str(out),
        "width": size[0],
        "height": size[1],
        "has_transparency": mode in ("RGBA", "LA"),
        "elapsed_s": round(time.monotonic() - started, 2),
        **extra,
    }


# ---------------------------------------------------------------- operations


def image_info(input_path: str) -> dict:
    """Size and format of an image, as the models see it (EXIF rotation applied)."""
    src = resolve_input(input_path)
    with Image.open(src) as img:
        fmt = img.format
        img = ImageOps.exif_transpose(img)
        return {"input_path": str(src), "width": img.width, "height": img.height, "format": fmt, "mode": img.mode}


def remove_background(
    input_path: str,
    output_path: Optional[str] = None,
    bg_color: Optional[str] = None,
    model: Optional[str] = None,
    feather: float = 0,
    shift: int = 0,
    overwrite: bool = False,
    backend=None,
) -> dict:
    started = time.monotonic()
    src = resolve_input(input_path)
    out = resolve_output(src, output_path, "remove_background", overwrite)
    opts = Options(mode="remove_background", bg_color=bg_color, feather=feather, shift=shift, model=model)
    png = (backend or get_backend()).process(src.read_bytes(), opts)
    return _write(out, png, "remove_background", src, started)


def remove_foreground(
    input_path: str,
    output_path: Optional[str] = None,
    fill: str = "inpaint",
    bg_color: Optional[str] = None,
    model: Optional[str] = None,
    feather: float = 0,
    shift: int = 0,
    overwrite: bool = False,
    backend=None,
) -> dict:
    started = time.monotonic()
    src = resolve_input(input_path)
    out = resolve_output(src, output_path, "remove_foreground", overwrite)
    opts = Options(mode="remove_foreground", fill=fill, bg_color=bg_color, feather=feather, shift=shift, model=model)
    png = (backend or get_backend()).process(src.read_bytes(), opts)
    return _write(out, png, "remove_foreground", src, started, fill=fill)


def get_mask(
    input_path: str,
    output_path: Optional[str] = None,
    model: Optional[str] = None,
    overwrite: bool = False,
    backend=None,
) -> dict:
    started = time.monotonic()
    src = resolve_input(input_path)
    out = resolve_output(src, output_path, "mask", overwrite)
    png = (backend or get_backend()).segment(src.read_bytes(), model)
    return _write(out, png, "mask", src, started, note="white = subject, black = background")


def apply_mask(
    input_path: str,
    mask_path: str,
    output_path: Optional[str] = None,
    mode: str = "remove_background",
    fill: str = "inpaint",
    bg_color: Optional[str] = None,
    feather: float = 0,
    shift: int = 0,
    overwrite: bool = False,
    backend=None,
) -> dict:
    started = time.monotonic()
    src = resolve_input(input_path)
    mask = resolve_input(mask_path)
    out = resolve_output(src, output_path, "apply_mask", overwrite)
    opts = Options(mode=mode, fill=fill, bg_color=bg_color, feather=feather, shift=shift)
    png = (backend or get_backend()).compose(src.read_bytes(), mask.read_bytes(), opts)
    return _write(out, png, "apply_mask", src, started, mask_path=str(mask), mode=mode)


def regions_mask(size: tuple[int, int], regions: Iterable, shape: str = "rect", units: str = "px") -> Image.Image:
    """Build a mask (white = remove) from [x0, y0, x1, y1] boxes, origin top-left."""
    if shape not in ("rect", "ellipse"):
        raise AgentError("shape must be 'rect' or 'ellipse'")
    if units not in ("px", "fraction"):
        raise AgentError("units must be 'px' or 'fraction'")
    w, h = size
    mask = Image.new("L", size, 0)
    draw = ImageDraw.Draw(mask)
    count = 0
    for region in regions:
        try:
            x0, y0, x1, y1 = (float(v) for v in region)
        except (TypeError, ValueError):
            raise AgentError(f"Each region must be [x0, y0, x1, y1], got {region!r}") from None
        if units == "fraction":
            x0, x1, y0, y1 = x0 * w, x1 * w, y0 * h, y1 * h
        x0, x1 = sorted((max(0, min(x0, w)), max(0, min(x1, w))))
        y0, y1 = sorted((max(0, min(y0, h)), max(0, min(y1, h))))
        if x1 - x0 < 1 or y1 - y0 < 1:
            raise AgentError(f"Region {region!r} is empty or outside the {w}x{h} image")
        (draw.rectangle if shape == "rect" else draw.ellipse)((x0, y0, x1 - 1, y1 - 1), fill=255)
        count += 1
    if not count:
        raise AgentError("Give at least one region")
    return mask


def erase_regions(
    input_path: str,
    regions: list,
    output_path: Optional[str] = None,
    shape: str = "rect",
    units: str = "px",
    fill: str = "inpaint",
    grow: int = 0,
    overwrite: bool = False,
    backend=None,
) -> dict:
    started = time.monotonic()
    src = resolve_input(input_path)
    out = resolve_output(src, output_path, "erase", overwrite)
    info = image_info(str(src))
    regions = list(regions)
    mask = regions_mask((info["width"], info["height"]), regions, shape, units)
    buf = io.BytesIO()
    mask.save(buf, "PNG")
    opts = Options(mode="remove_foreground", fill=fill, shift=grow)
    png = (backend or get_backend()).compose(src.read_bytes(), buf.getvalue(), opts)
    return _write(out, png, "erase", src, started, regions=len(regions), fill=fill)


def expand_inputs(inputs: Iterable[str]) -> list[Path]:
    """Files, directories (their images, non-recursive) and glob patterns -> image paths."""
    found: list[Path] = []
    for item in inputs:
        p = Path(item).expanduser()
        if any(c in item for c in "*?["):
            matches = [Path(m) for m in sorted(glob.glob(str(p), recursive=True))]
        elif p.is_dir():
            matches = sorted(c for c in p.iterdir() if c.is_file())
        else:
            matches = [p]
        for m in matches:
            if m.suffix.lower() in IMAGE_EXTENSIONS or m == p:
                m = m.resolve()
                if m not in found:
                    found.append(m)
    return found


BATCH_OPERATIONS = ("remove_background", "remove_foreground", "mask")


def batch(
    inputs: list[str],
    output_dir: Optional[str] = None,
    operation: str = "remove_background",
    fill: str = "inpaint",
    bg_color: Optional[str] = None,
    model: Optional[str] = None,
    feather: float = 0,
    shift: int = 0,
    overwrite: bool = False,
    backend=None,
    on_progress=None,
) -> dict:
    """Run one operation over many images; a failure on one file doesn't stop the rest."""
    if operation not in BATCH_OPERATIONS:
        raise AgentError(f"operation must be one of {', '.join(BATCH_OPERATIONS)}")
    files = expand_inputs(inputs)
    if not files:
        raise AgentError("No input images found")
    out_dir = None
    if output_dir:
        out_dir = str(Path(output_dir).expanduser()) + os.sep
        _check_allowed(Path(out_dir).resolve())
        Path(out_dir).mkdir(parents=True, exist_ok=True)

    backend = backend or get_backend()
    started = time.monotonic()
    results, failed = [], []
    for i, f in enumerate(files, 1):
        try:
            if operation == "remove_background":
                r = remove_background(str(f), out_dir, bg_color, model, feather, shift, overwrite, backend)
            elif operation == "remove_foreground":
                r = remove_foreground(str(f), out_dir, fill, bg_color, model, feather, shift, overwrite, backend)
            else:
                r = get_mask(str(f), out_dir, model, overwrite, backend)
            results.append(r)
        except Exception as exc:  # keep going; report per-file errors
            failed.append({"input_path": str(f), "error": str(exc)})
        if on_progress:
            on_progress(i, len(files))
    return {
        "operation": operation,
        "processed": len(results),
        "failed": len(failed),
        "elapsed_s": round(time.monotonic() - started, 2),
        "results": results,
        "errors": failed,
    }


def make_preview(path: str, max_side: int = 512) -> bytes:
    """A small PNG of an image for an agent to look at; transparency shown as a checkerboard."""
    with Image.open(path) as img:
        img = ImageOps.exif_transpose(img)
        img.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
        if img.mode in ("RGBA", "LA", "P"):
            rgba = img.convert("RGBA")
            board = Image.new("RGB", rgba.size, (255, 255, 255))
            draw = ImageDraw.Draw(board)
            cell = 12
            for y in range(0, rgba.height, cell):
                for x in range((y // cell) % 2 * cell, rgba.width, cell * 2):
                    draw.rectangle((x, y, x + cell - 1, y + cell - 1), fill=(204, 204, 204))
            board.paste(rgba, mask=rgba.getchannel("A"))
            img = board
        else:
            img = img.convert("RGB")
        buf = io.BytesIO()
        img.save(buf, "PNG")
        return buf.getvalue()
