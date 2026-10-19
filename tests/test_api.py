import io

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app import config, main


@pytest.fixture
def client(fake_models):
    """App with the ML models replaced by cheap fakes (no downloads)."""
    return TestClient(main.app)


def png(img):
    buf = io.BytesIO()
    img.save(buf, "PNG")
    return buf.getvalue()


def photo(w=64, h=48):
    return png(Image.new("RGB", (w, h), (200, 50, 50)))


def open_png(resp):
    assert resp.status_code == 200, resp.text
    assert resp.headers["content-type"] == "image/png"
    return Image.open(io.BytesIO(resp.content))


def test_models(client):
    data = client.get("/api/models").json()
    assert data["default"] in [m["id"] for m in data["models"]]


def test_index_served(client):
    resp = client.get("/")
    assert resp.status_code == 200
    assert "Cutout Studio" in resp.text


def test_segment_returns_mask(client):
    mask = open_png(client.post("/api/segment", files={"image": ("a.png", photo())}))
    assert mask.mode == "L"
    assert mask.size == (64, 48)


def test_process_remove_background(client):
    out = open_png(client.post("/api/process", files={"image": ("a.png", photo())}))
    assert out.mode == "RGBA"
    assert out.getpixel((0, 0))[3] == 0
    assert out.getpixel((32, 24))[3] == 255


def test_process_remove_background_color(client):
    out = open_png(
        client.post("/api/process", files={"image": ("a.png", photo())}, data={"bg_color": "#0000ff"})
    )
    assert out.mode == "RGB"
    assert out.getpixel((0, 0)) == (0, 0, 255)


def test_process_remove_foreground_inpaint(client):
    out = open_png(
        client.post(
            "/api/process",
            files={"image": ("a.png", photo())},
            data={"mode": "remove_foreground", "fill": "inpaint"},
        )
    )
    assert out.mode == "RGB"
    assert out.getpixel((32, 24)) == (0, 255, 0)


def test_compose_with_edited_mask(client):
    mask = Image.new("L", (32, 24), 0)  # half size: server scales it up
    mask.paste(255, (0, 0, 16, 24))
    out = open_png(
        client.post(
            "/api/compose",
            files={"image": ("a.png", photo()), "mask": ("m.png", png(mask))},
            data={"mode": "remove_foreground", "fill": "transparent"},
        )
    )
    assert out.size == (64, 48)
    assert out.getpixel((5, 5))[3] == 0  # painted as subject -> removed
    assert out.getpixel((60, 40))[3] == 255


def test_unknown_model(client):
    resp = client.post("/api/segment", files={"image": ("a.png", photo())}, data={"model": "nope"})
    assert resp.status_code == 400


def test_bad_image(client):
    resp = client.post("/api/process", files={"image": ("a.png", b"garbage")})
    assert resp.status_code == 400


def test_bad_mode(client):
    resp = client.post("/api/process", files={"image": ("a.png", photo())}, data={"mode": "explode"})
    assert resp.status_code == 422


def test_too_large(client, monkeypatch):
    monkeypatch.setattr(config, "MAX_UPLOAD_MB", 0.001)
    resp = client.post("/api/process", files={"image": ("a.png", b"x" * 5000)})
    assert resp.status_code == 413
