"""FastAPI app: JSON/PNG API under /api and the static web UI at /."""

import threading
from typing import Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import Response
from fastapi.staticfiles import StaticFiles
from PIL import Image

from . import compose as composer
from . import config, imaging, inpainting, segmentation

app = FastAPI(title="Background / Foreground Remover")
_inference = threading.BoundedSemaphore(max(config.INFERENCE_CONCURRENCY, 1))


def _read_upload(upload: UploadFile) -> bytes:
    limit = int(config.MAX_UPLOAD_MB * 1024 * 1024)
    data = upload.file.read(limit + 1)
    if len(data) > limit:
        raise HTTPException(413, f"File too large (max {config.MAX_UPLOAD_MB:g} MB)")
    if not data:
        raise HTTPException(400, "Empty upload")
    return data


def _load_image(upload: UploadFile) -> Image.Image:
    try:
        return imaging.load_image(_read_upload(upload))
    except imaging.ImageError as exc:
        raise HTTPException(400, str(exc)) from exc


def _segment(img: Image.Image, model: Optional[str]) -> Image.Image:
    try:
        with _inference:
            return segmentation.predict_mask(img, model or config.DEFAULT_MODEL)
    except segmentation.UnknownModelError as exc:
        raise HTTPException(400, str(exc)) from exc


def _compose(img, mask, mode, fill, bg_color, feather, shift) -> Response:
    try:
        with _inference:
            out = composer.compose(
                img,
                mask,
                mode=mode,
                fill=fill,
                bg_color=composer.parse_color(bg_color),
                feather=max(0.0, min(feather, 50.0)),
                shift=max(-50, min(shift, 50)),
                inpaint=inpainting.inpainter.inpaint,
            )
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    return Response(imaging.to_png(out), media_type="image/png")


@app.get("/api/models")
def list_models():
    return {
        "default": config.DEFAULT_MODEL,
        "models": [{"id": k, "label": v} for k, v in segmentation.MODELS.items()],
    }


@app.post("/api/segment")
def segment(image: UploadFile = File(...), model: Optional[str] = Form(None)):
    """Return the AI subject mask (white = subject) as a grayscale PNG."""
    img = _load_image(image)
    mask = _segment(img, model)
    return Response(
        imaging.to_png(mask),
        media_type="image/png",
        headers={"X-Image-Width": str(img.width), "X-Image-Height": str(img.height)},
    )


@app.post("/api/compose")
def compose(
    image: UploadFile = File(...),
    mask: UploadFile = File(...),
    mode: composer.Mode = Form("remove_background"),
    fill: composer.Fill = Form("inpaint"),
    bg_color: Optional[str] = Form(None),
    feather: float = Form(0),
    shift: int = Form(0),
):
    """Apply a client-supplied (e.g. hand-edited) mask to the image."""
    img = _load_image(image)
    try:
        mask_img = imaging.load_mask(_read_upload(mask), img.size)
    except imaging.ImageError as exc:
        raise HTTPException(400, str(exc)) from exc
    return _compose(img, mask_img, mode, fill, bg_color, feather, shift)


@app.post("/api/process")
def process(
    image: UploadFile = File(...),
    model: Optional[str] = Form(None),
    mode: composer.Mode = Form("remove_background"),
    fill: composer.Fill = Form("inpaint"),
    bg_color: Optional[str] = Form(None),
    feather: float = Form(0),
    shift: int = Form(0),
):
    """Segment and compose in one request (used for batch processing)."""
    img = _load_image(image)
    mask = _segment(img, model)
    return _compose(img, mask, mode, fill, bg_color, feather, shift)


app.mount("/", StaticFiles(directory=config.ROOT_DIR / "static", html=True), name="static")
