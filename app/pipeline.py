"""Model pipeline shared by the HTTP API, the CLI and the MCP server.

Everything here raises ValueError (or a subclass) for bad input, so callers can
map it to their own error type (HTTP 400, CLI message, MCP tool error).
"""

import threading
from dataclasses import asdict, dataclass
from typing import Optional

from PIL import Image

from . import compose as composer
from . import config, imaging, inpainting, segmentation

_inference = threading.BoundedSemaphore(max(config.INFERENCE_CONCURRENCY, 1))


@dataclass
class Options:
    mode: str = "remove_background"  # or "remove_foreground"
    fill: str = "inpaint"  # or "transparent" (remove_foreground only)
    bg_color: Optional[str] = None  # "#rrggbb"; None = transparent
    feather: float = 0
    shift: int = 0
    model: Optional[str] = None  # None = config.DEFAULT_MODEL

    def form_fields(self) -> dict[str, str]:
        """The options as multipart form fields for the HTTP API."""
        return {k: str(v) for k, v in asdict(self).items() if v is not None}


def list_models() -> dict:
    return {
        "default": config.DEFAULT_MODEL,
        "models": [{"id": k, "label": v} for k, v in segmentation.MODELS.items()],
    }


def segment_image(img: Image.Image, model: Optional[str] = None) -> Image.Image:
    with _inference:
        return segmentation.predict_mask(img, model or config.DEFAULT_MODEL)


def compose_image(img: Image.Image, mask: Image.Image, opts: Options) -> Image.Image:
    with _inference:
        return composer.compose(
            img,
            mask,
            mode=opts.mode,
            fill=opts.fill,
            bg_color=composer.parse_color(opts.bg_color),
            feather=max(0.0, min(float(opts.feather), 50.0)),
            shift=max(-50, min(int(opts.shift), 50)),
            inpaint=inpainting.inpainter.inpaint,
        )


def segment(image: bytes, model: Optional[str] = None) -> bytes:
    """Image bytes -> mask PNG (white = subject)."""
    return imaging.to_png(segment_image(imaging.load_image(image), model))


def compose(image: bytes, mask: bytes, opts: Options) -> bytes:
    """Image bytes + mask bytes -> result PNG."""
    img = imaging.load_image(image)
    return imaging.to_png(compose_image(img, imaging.load_mask(mask, img.size), opts))


def process(image: bytes, opts: Options) -> bytes:
    """Image bytes -> result PNG, detecting the subject first."""
    img = imaging.load_image(image)
    return imaging.to_png(compose_image(img, segment_image(img, opts.model), opts))


def warmup(model: Optional[str] = None, inpaint: bool = True) -> dict:
    """Download and load models ahead of time so the first real request is fast."""
    name = model or config.DEFAULT_MODEL
    segmentation.get_session(name)
    loaded = {"segmentation_model": name, "inpainting": False}
    if inpaint:
        inpainting.inpainter.load_session()
        loaded["inpainting"] = True
    return loaded
