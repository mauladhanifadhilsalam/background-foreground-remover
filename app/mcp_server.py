"""MCP server exposing Cutout Studio to AI agents (Claude Code, Claude Desktop, Cursor, ...).

Run over stdio:  python -m app.mcp_server   (or the `cutout-mcp` script)

Models run in this process unless CUTOUT_API_URL points at a running
Cutout Studio web server, in which case requests are forwarded there.
"""

import json
from typing import Annotated, Literal, Optional

from mcp.server import MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from mcp.server.mcpserver.utilities.types import Image as MCPImage
from mcp_types import CallToolResult, TextContent
from pydantic import Field

from . import agent_ops
from .backends import BackendError, get_backend

INSTRUCTIONS = """\
Cutout Studio removes image backgrounds and objects on this machine.
- All paths are local file paths; results are always written as PNG files and the
  tool result gives `output_path`. Existing files are never overwritten unless
  overwrite=true; by default results go next to the input (<name>-no-background.png etc).
- Masks: white = subject / area to act on, black = everything else.
- To erase specific things (watermark, logo, person), call image_info (optionally with
  preview=true) to learn the size, then erase_regions with pixel boxes [x0, y0, x1, y1]
  (origin top-left) or units="fraction" for 0..1 coordinates.
- Pass preview=true to get a small image of the result to check it visually.
- Speed on CPU: detection ~1-3 s with the default model, AI inpainting ~5 s; the
  first call can take longer while models download. Call warmup first if needed.
"""

server = MCPServer("cutout-studio", instructions=INSTRUCTIONS)
_backend = None


def backend():
    global _backend
    if _backend is None:
        _backend = get_backend()
    return _backend


def _reply(result: dict, preview_path: Optional[str] = None) -> CallToolResult:
    content = [TextContent(type="text", text=json.dumps(result, indent=2))]
    if preview_path:
        content.append(MCPImage(data=agent_ops.make_preview(preview_path), format="png").to_image_content())
    return CallToolResult(content=content, structured_content=result)


def _run(fn, *args, **kwargs):
    """Call an operation, turning expected failures into tool errors the agent can read."""
    try:
        return fn(*args, **kwargs)
    except (ValueError, OSError, BackendError) as exc:
        raise ToolError(str(exc)) from exc


# Shared parameter types (descriptions end up in the tool schemas).
InputPath = Annotated[str, Field(description="Path of the source image (jpg, png, webp, ...)")]
OutputPath = Annotated[
    Optional[str],
    Field(description="Where to write the PNG result: a file path or a directory. Default: next to the input."),
]
BgColor = Annotated[Optional[str], Field(description="Fill colour '#rrggbb' instead of transparency")]
Model = Annotated[Optional[str], Field(description="Detection model id from list_models; default is fast ISNet")]
Feather = Annotated[float, Field(ge=0, le=50, description="Soften mask edges by this many pixels")]
Shift = Annotated[int, Field(ge=-50, le=50, description="Grow (+) or shrink (-) the mask edge in pixels")]
Fill = Annotated[
    Literal["inpaint", "transparent"],
    Field(description="inpaint = AI fills the removed area; transparent = leave a transparent hole"),
]
Overwrite = Annotated[bool, Field(description="Replace an existing output file instead of adding -2, -3, ...")]
Preview = Annotated[bool, Field(description="Also return a small preview image of the result")]


@server.tool()
def list_models() -> CallToolResult:
    """List the available subject-detection models and the default one."""
    return _reply(_run(backend().list_models))


@server.tool()
def image_info(input_path: InputPath, preview: Preview = False) -> CallToolResult:
    """Get an image's width/height (after EXIF rotation). Use before erase_regions to pick coordinates."""
    info = _run(agent_ops.image_info, input_path)
    return _reply(info, info["input_path"] if preview else None)


@server.tool()
def remove_background(
    input_path: InputPath,
    output_path: OutputPath = None,
    bg_color: BgColor = None,
    model: Model = None,
    feather: Feather = 0,
    shift: Shift = 0,
    overwrite: Overwrite = False,
    preview: Preview = False,
) -> CallToolResult:
    """Keep the main subject and remove the background (transparent PNG, or bg_color)."""
    r = _run(agent_ops.remove_background, input_path, output_path, bg_color, model, feather, shift, overwrite, backend())
    return _reply(r, r["output_path"] if preview else None)


@server.tool()
def remove_foreground(
    input_path: InputPath,
    output_path: OutputPath = None,
    fill: Fill = "inpaint",
    bg_color: BgColor = None,
    model: Model = None,
    feather: Feather = 0,
    shift: Shift = 0,
    overwrite: Overwrite = False,
    preview: Preview = False,
) -> CallToolResult:
    """Remove the main subject and keep the background, filling the hole with AI inpainting by default."""
    r = _run(
        agent_ops.remove_foreground, input_path, output_path, fill, bg_color, model, feather, shift, overwrite, backend()
    )
    return _reply(r, r["output_path"] if preview else None)


@server.tool()
def get_mask(
    input_path: InputPath,
    output_path: OutputPath = None,
    model: Model = None,
    overwrite: Overwrite = False,
    preview: Preview = False,
) -> CallToolResult:
    """Save the AI subject mask as a grayscale PNG (white = subject). Edit it and pass it to apply_mask."""
    r = _run(agent_ops.get_mask, input_path, output_path, model, overwrite, backend())
    return _reply(r, r["output_path"] if preview else None)


@server.tool()
def apply_mask(
    input_path: InputPath,
    mask_path: Annotated[str, Field(description="Grayscale mask PNG, white = subject; resized to the image if needed")],
    output_path: OutputPath = None,
    mode: Annotated[
        Literal["remove_background", "remove_foreground"],
        Field(description="remove_background keeps the white area; remove_foreground removes it"),
    ] = "remove_background",
    fill: Fill = "inpaint",
    bg_color: BgColor = None,
    feather: Feather = 0,
    shift: Shift = 0,
    overwrite: Overwrite = False,
    preview: Preview = False,
) -> CallToolResult:
    """Cut out or remove using your own mask instead of AI detection (the manual-brush equivalent)."""
    r = _run(
        agent_ops.apply_mask,
        input_path, mask_path, output_path, mode, fill, bg_color, feather, shift, overwrite, backend(),
    )
    return _reply(r, r["output_path"] if preview else None)


@server.tool()
def erase_regions(
    input_path: InputPath,
    regions: Annotated[
        list[list[float]],
        Field(description="Boxes to erase as [x0, y0, x1, y1], origin top-left", min_length=1),
    ],
    output_path: OutputPath = None,
    shape: Annotated[Literal["rect", "ellipse"], Field(description="Erase the box or the ellipse inside it")] = "rect",
    units: Annotated[
        Literal["px", "fraction"], Field(description="px = pixels; fraction = 0..1 of width/height")
    ] = "px",
    fill: Fill = "inpaint",
    grow: Annotated[int, Field(ge=0, le=50, description="Enlarge the erased area by this many pixels")] = 0,
    overwrite: Overwrite = False,
    preview: Preview = False,
) -> CallToolResult:
    """Erase objects inside the given boxes (watermarks, logos, people...) and fill them with AI inpainting."""
    r = _run(
        agent_ops.erase_regions, input_path, regions, output_path, shape, units, fill, grow, overwrite, backend()
    )
    return _reply(r, r["output_path"] if preview else None)


@server.tool()
def batch_process(
    inputs: Annotated[
        list[str], Field(description="Image files, directories (non-recursive) or glob patterns", min_length=1)
    ],
    output_dir: Annotated[Optional[str], Field(description="Directory for results; default: next to each input")] = None,
    operation: Annotated[
        Literal["remove_background", "remove_foreground", "mask"], Field(description="What to do to every image")
    ] = "remove_background",
    fill: Fill = "inpaint",
    bg_color: BgColor = None,
    model: Model = None,
    feather: Feather = 0,
    shift: Shift = 0,
    overwrite: Overwrite = False,
) -> CallToolResult:
    """Process many images with the same settings. Per-file errors are reported without stopping the batch."""
    r = _run(
        agent_ops.batch,
        inputs, output_dir, operation, fill, bg_color, model, feather, shift, overwrite, backend(),
    )
    return _reply(r)


@server.tool()
def warmup(
    model: Model = None,
    inpaint: Annotated[bool, Field(description="Also load the inpainting model")] = True,
) -> CallToolResult:
    """Download and load models now, so later calls don't time out on the first run."""
    return _reply(_run(backend().warmup, model, inpaint))


def main() -> None:
    import logging

    logging.getLogger("httpx").setLevel(logging.WARNING)  # one INFO line per request is noise
    server.run("stdio")


if __name__ == "__main__":
    main()
