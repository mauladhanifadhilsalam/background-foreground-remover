import io

import numpy as np
import pytest
from PIL import Image

from app import compose, imaging


def make_image(w=40, h=30, color=(10, 120, 200)):
    return Image.new("RGB", (w, h), color)


def make_mask(w=40, h=30, box=(10, 5, 30, 25)):
    mask = Image.new("L", (w, h), 0)
    mask.paste(255, box)
    return mask


def test_parse_color():
    assert compose.parse_color("#ff8000") == (255, 128, 0)
    assert compose.parse_color("0f0") == (0, 255, 0)
    assert compose.parse_color("") is None
    assert compose.parse_color("transparent") is None
    with pytest.raises(ValueError):
        compose.parse_color("#12")


def test_remove_background_transparent():
    out = compose.compose(make_image(), make_mask())
    assert out.mode == "RGBA"
    alpha = np.asarray(out.getchannel("A"))
    assert alpha[15, 20] == 255  # inside subject
    assert alpha[0, 0] == 0  # background


def test_remove_background_solid_color():
    out = compose.compose(make_image(), make_mask(), bg_color=(255, 255, 255))
    assert out.mode == "RGB"
    assert out.getpixel((0, 0)) == (255, 255, 255)
    assert out.getpixel((20, 15)) == (10, 120, 200)


def test_remove_foreground_transparent_inverts_mask():
    out = compose.compose(make_image(), make_mask(), mode="remove_foreground", fill="transparent")
    alpha = np.asarray(out.getchannel("A"))
    assert alpha[15, 20] == 0
    assert alpha[0, 0] == 255


def test_remove_foreground_inpaint_uses_grown_hole():
    seen = {}

    def fake_inpaint(img, hole):
        seen["hole"] = hole
        return Image.new("RGB", img.size, (1, 2, 3))

    out = compose.compose(make_image(), make_mask(), mode="remove_foreground", fill="inpaint", inpaint=fake_inpaint)
    assert out.getpixel((0, 0)) == (1, 2, 3)
    hole = np.asarray(seen["hole"])
    # The hole is larger than the subject so soft edges get replaced too.
    assert (hole > 0).sum() > (np.asarray(make_mask()) > 0).sum()


def test_grow_and_shrink():
    mask = make_mask()
    area = (np.asarray(mask) > 127).sum()
    assert (np.asarray(compose.grow_mask(mask, 4)) > 127).sum() > area
    assert (np.asarray(compose.grow_mask(mask, -4)) > 127).sum() < area


def test_invalid_mode():
    with pytest.raises(ValueError):
        compose.compose(make_image(), make_mask(), mode="nope")


def test_load_image_flattens_alpha_and_limits_size():
    rgba = Image.new("RGBA", (300, 100), (255, 0, 0, 0))
    buf = io.BytesIO()
    rgba.save(buf, "PNG")
    img = imaging.load_image(buf.getvalue(), max_side=150)
    assert img.mode == "RGB"
    assert img.size == (150, 50)
    assert img.getpixel((0, 0)) == (255, 255, 255)


def test_load_image_rejects_garbage():
    with pytest.raises(imaging.ImageError):
        imaging.load_image(b"not an image")


def test_load_mask_resizes_and_reads_alpha():
    m = Image.new("RGBA", (20, 10), (0, 0, 0, 0))
    m.paste((0, 0, 0, 255), (0, 0, 10, 10))
    buf = io.BytesIO()
    m.save(buf, "PNG")
    mask = imaging.load_mask(buf.getvalue(), (40, 20))
    assert mask.size == (40, 20)
    assert mask.getpixel((2, 2)) == 255 and mask.getpixel((38, 18)) == 0
