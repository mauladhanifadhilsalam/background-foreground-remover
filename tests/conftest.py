import io

import pytest
from PIL import Image

from app import inpainting, segmentation


@pytest.fixture
def fake_models(monkeypatch):
    """Replace the ML models with cheap fakes: the subject is the centre box, inpainting paints green."""

    def fake_predict(img, model):
        if model not in segmentation.MODELS:
            raise segmentation.UnknownModelError(f"Unknown model '{model}'")
        mask = Image.new("L", img.size, 0)
        w, h = img.size
        mask.paste(255, (w // 4, h // 4, 3 * w // 4, 3 * h // 4))
        return mask

    def fake_inpaint(img, mask):
        out = img.copy()
        out.paste((0, 255, 0), mask=mask)
        return out

    monkeypatch.setattr(segmentation, "predict_mask", fake_predict)
    monkeypatch.setattr(inpainting.inpainter, "inpaint", fake_inpaint)


def png_bytes(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, "PNG")
    return buf.getvalue()


@pytest.fixture
def photo_path(tmp_path):
    path = tmp_path / "photo.jpg"
    Image.new("RGB", (64, 48), (200, 50, 50)).save(path, "JPEG", quality=95)
    return path
