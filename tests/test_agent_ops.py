import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app import agent_ops, main
from app.backends import BackendError, HttpBackend, LocalBackend

from .conftest import png_bytes


@pytest.fixture(params=["local", "http"])
def backend(request, fake_models):
    """Run every test against both backends; HTTP goes through the real FastAPI app."""
    if request.param == "local":
        return LocalBackend()
    return HttpBackend("http://testserver", client=TestClient(main.app))


def test_remove_background_default_output_and_no_overwrite(photo_path, backend):
    r1 = agent_ops.remove_background(str(photo_path), backend=backend)
    r2 = agent_ops.remove_background(str(photo_path), backend=backend)
    assert r1["output_path"].endswith("photo-no-background.png")
    assert r2["output_path"].endswith("photo-no-background-2.png")
    assert r1["has_transparency"] and (r1["width"], r1["height"]) == (64, 48)
    with Image.open(r1["output_path"]) as img:
        assert img.getpixel((0, 0))[3] == 0 and img.getpixel((32, 24))[3] == 255


def test_overwrite_and_output_options(photo_path, tmp_path, backend):
    out = tmp_path / "res" / "cut.jpg"  # wrong extension is turned into .png
    r = agent_ops.remove_background(str(photo_path), str(out), bg_color="#ffffff", backend=backend)
    assert r["output_path"] == str(out.with_suffix(".png"))
    assert not r["has_transparency"]
    r = agent_ops.remove_background(str(photo_path), str(out), overwrite=True, backend=backend)
    assert r["output_path"] == str(out.with_suffix(".png"))

    out_dir = tmp_path / "dir"
    out_dir.mkdir()
    r = agent_ops.remove_foreground(str(photo_path), str(out_dir), backend=backend)
    assert r["output_path"] == str(out_dir / "photo-no-foreground.png")


def test_remove_foreground_inpaint(photo_path, backend):
    r = agent_ops.remove_foreground(str(photo_path), fill="inpaint", backend=backend)
    with Image.open(r["output_path"]) as img:
        assert img.getpixel((32, 24)) == (0, 255, 0)


def test_mask_and_apply_mask(photo_path, tmp_path, backend):
    m = agent_ops.get_mask(str(photo_path), backend=backend)
    with Image.open(m["output_path"]) as mask:
        assert mask.mode == "L" and mask.getpixel((32, 24)) == 255

    custom = tmp_path / "custom.png"
    mask = Image.new("L", (64, 48), 0)
    mask.paste(255, (0, 0, 10, 10))
    custom.write_bytes(png_bytes(mask))
    r = agent_ops.apply_mask(str(photo_path), str(custom), mode="remove_background", backend=backend)
    with Image.open(r["output_path"]) as img:
        assert img.getpixel((5, 5))[3] == 255 and img.getpixel((32, 24))[3] == 0


def test_erase_regions(photo_path, backend):
    r = agent_ops.erase_regions(str(photo_path), [[0, 0, 10, 10]], backend=backend)
    with Image.open(r["output_path"]) as img:
        assert img.getpixel((4, 4)) == (0, 255, 0)
        assert img.getpixel((50, 40)) != (0, 255, 0)


def test_regions_mask_fraction_and_ellipse():
    mask = agent_ops.regions_mask((100, 50), [[0.5, 0, 1, 1]], units="fraction")
    assert mask.getpixel((75, 25)) == 255 and mask.getpixel((10, 25)) == 0
    ell = agent_ops.regions_mask((100, 100), [[0, 0, 100, 100]], shape="ellipse")
    assert ell.getpixel((50, 50)) == 255 and ell.getpixel((1, 1)) == 0
    with pytest.raises(agent_ops.AgentError):
        agent_ops.regions_mask((100, 50), [[200, 200, 300, 300]])
    with pytest.raises(agent_ops.AgentError):
        agent_ops.regions_mask((100, 50), [[1, 2, 3]])


def test_batch_reports_bad_files(photo_path, tmp_path, backend):
    (tmp_path / "broken.png").write_bytes(b"nope")
    (tmp_path / "notes.txt").write_text("ignored")
    r = agent_ops.batch([str(tmp_path)], str(tmp_path / "out"), backend=backend)
    assert r["processed"] == 1 and r["failed"] == 1
    assert "decode" in r["errors"][0]["error"]
    assert r["results"][0]["output_path"] == str(tmp_path / "out" / "photo-no-background.png")


def test_batch_glob_and_mask_operation(photo_path, tmp_path, backend):
    r = agent_ops.batch([str(tmp_path / "*.jpg")], operation="mask", backend=backend)
    assert r["processed"] == 1 and r["results"][0]["operation"] == "mask"
    with pytest.raises(agent_ops.AgentError):
        agent_ops.batch([str(tmp_path / "*.gif")], backend=backend)


def test_missing_input(tmp_path):
    with pytest.raises(agent_ops.AgentError):
        agent_ops.remove_background(str(tmp_path / "missing.jpg"), backend=LocalBackend())


def test_allowed_dirs(photo_path, tmp_path, monkeypatch, fake_models):
    monkeypatch.setenv("CUTOUT_ALLOWED_DIRS", str(tmp_path))
    agent_ops.remove_background(str(photo_path), backend=LocalBackend())
    with pytest.raises(agent_ops.AgentError, match="outside"):
        agent_ops.remove_background(str(photo_path), "/tmp/elsewhere/x.png", backend=LocalBackend())


def test_http_backend_errors(photo_path, fake_models):
    backend = HttpBackend("http://testserver", client=TestClient(main.app))
    with pytest.raises(BackendError, match="Unknown model"):
        agent_ops.remove_background(str(photo_path), model="nope", backend=backend)
    with pytest.raises(BackendError, match="Cannot reach"):
        HttpBackend("http://127.0.0.1:9", timeout=2).list_models()


def test_preview_is_small_png(photo_path, backend):
    r = agent_ops.remove_background(str(photo_path), backend=backend)
    import io

    with Image.open(io.BytesIO(agent_ops.make_preview(r["output_path"], max_side=32))) as img:
        assert img.format == "PNG" and max(img.size) == 32 and img.mode == "RGB"
