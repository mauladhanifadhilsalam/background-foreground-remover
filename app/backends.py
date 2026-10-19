"""Where the models run: in this process, or on a running Cutout Studio server.

Both backends take and return encoded image bytes, so the CLI and MCP server
don't care which one they use. Set CUTOUT_API_URL (e.g. http://127.0.0.1:8000)
to send requests to a running web server instead of loading models locally.
"""

import io
import os
from typing import Optional

from PIL import Image

from . import pipeline
from .pipeline import Options


class BackendError(RuntimeError):
    """The backend could not complete a request (server down, server-side error...)."""


class LocalBackend:
    name = "local"

    def list_models(self) -> dict:
        return pipeline.list_models()

    def segment(self, image: bytes, model: Optional[str] = None) -> bytes:
        return pipeline.segment(image, model)

    def compose(self, image: bytes, mask: bytes, opts: Options) -> bytes:
        return pipeline.compose(image, mask, opts)

    def process(self, image: bytes, opts: Options) -> bytes:
        return pipeline.process(image, opts)

    def warmup(self, model: Optional[str] = None, inpaint: bool = True) -> dict:
        return pipeline.warmup(model, inpaint)


class HttpBackend:
    name = "http"

    def __init__(self, base_url: str, timeout: float = 600, client=None):
        import httpx

        self.base_url = base_url.rstrip("/")
        self._client = client or httpx.Client(base_url=self.base_url, timeout=timeout)

    def _request(self, method: str, path: str, **kwargs):
        import httpx

        try:
            resp = self._client.request(method, path, **kwargs)
        except httpx.HTTPError as exc:
            raise BackendError(f"Cannot reach Cutout Studio at {self.base_url} ({exc}). Is the server running?") from exc
        if resp.status_code >= 400:
            try:
                detail = resp.json().get("detail", resp.text)
            except ValueError:
                detail = resp.text
            raise BackendError(f"Server error {resp.status_code}: {detail}")
        return resp

    def list_models(self) -> dict:
        return self._request("GET", "/api/models").json()

    def segment(self, image: bytes, model: Optional[str] = None) -> bytes:
        data = {"model": model} if model else {}
        return self._request("POST", "/api/segment", files={"image": ("image", image)}, data=data).content

    def compose(self, image: bytes, mask: bytes, opts: Options) -> bytes:
        files = {"image": ("image", image), "mask": ("mask.png", mask)}
        return self._request("POST", "/api/compose", files=files, data=opts.form_fields()).content

    def process(self, image: bytes, opts: Options) -> bytes:
        return self._request("POST", "/api/process", files={"image": ("image", image)}, data=opts.form_fields()).content

    def warmup(self, model: Optional[str] = None, inpaint: bool = True) -> dict:
        # Run a tiny image through the server so it loads (and downloads) the models.
        def png(img: Image.Image) -> bytes:
            buf = io.BytesIO()
            img.save(buf, "PNG")
            return buf.getvalue()

        image = png(Image.new("RGB", (64, 64), (128, 128, 128)))
        self.segment(image, model)
        if inpaint:
            mask = Image.new("L", (64, 64), 0)
            mask.paste(255, (16, 16, 48, 48))  # non-empty, so LaMa actually runs
            self.compose(image, png(mask), Options(mode="remove_foreground", fill="inpaint"))
        return {"segmentation_model": model or self.list_models()["default"], "inpainting": inpaint}


def get_backend(api_url: Optional[str] = None):
    url = api_url or os.getenv("CUTOUT_API_URL")
    return HttpBackend(url) if url else LocalBackend()
