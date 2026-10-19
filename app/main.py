"""FastAPI app: JSON/PNG API under /api and the static web UI at /."""

from typing import Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import Response
from fastapi.staticfiles import StaticFiles

from . import compose as composer
from . import config, imaging, pipeline

app = FastAPI(title="Background / Foreground Remover")


def _read_upload(upload: UploadFile) -> bytes:
    limit = int(config.MAX_UPLOAD_MB * 1024 * 1024)
    data = upload.file.read(limit + 1)
    if len(data) > limit:
        raise HTTPException(413, f"File too large (max {config.MAX_UPLOAD_MB:g} MB)")
    if not data:
        raise HTTPException(400, "Empty upload")
    return data


def _bad_request(fn, *args):
    """Run a pipeline step, turning input errors into HTTP 400."""
    try:
        return fn(*args)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc


def _png(img) -> Response:
    return Response(imaging.to_png(img), media_type="image/png")


@app.get("/api/models")
def list_models():
    return pipeline.list_models()


@app.post("/api/segment")
def segment(image: UploadFile = File(...), model: Optional[str] = Form(None)):
    """Return the AI subject mask (white = subject) as a grayscale PNG."""
    img = _bad_request(imaging.load_image, _read_upload(image))
    mask = _bad_request(pipeline.segment_image, img, model)
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
    img = _bad_request(imaging.load_image, _read_upload(image))
    mask_img = _bad_request(imaging.load_mask, _read_upload(mask), img.size)
    opts = pipeline.Options(mode=mode, fill=fill, bg_color=bg_color, feather=feather, shift=shift)
    return _png(_bad_request(pipeline.compose_image, img, mask_img, opts))


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
    img = _bad_request(imaging.load_image, _read_upload(image))
    opts = pipeline.Options(mode=mode, fill=fill, bg_color=bg_color, feather=feather, shift=shift, model=model)
    mask = _bad_request(pipeline.segment_image, img, model)
    return _png(_bad_request(pipeline.compose_image, img, mask, opts))


app.mount("/", StaticFiles(directory=config.ROOT_DIR / "static", html=True), name="static")
