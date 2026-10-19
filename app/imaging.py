"""Decoding uploads and encoding results."""

import io

from PIL import Image, ImageOps, UnidentifiedImageError

from . import config


class ImageError(ValueError):
    """The uploaded bytes are not a usable image."""


def load_image(data: bytes, max_side: int = config.MAX_SIDE) -> Image.Image:
    """Decode an upload into an RGB image, honouring EXIF rotation and size limits."""
    try:
        img = Image.open(io.BytesIO(data))
        img = ImageOps.exif_transpose(img)
    except (UnidentifiedImageError, OSError) as exc:
        raise ImageError("Could not decode image") from exc

    if img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in img.info):
        # Flatten existing transparency onto white so the models see a normal photo.
        rgba = img.convert("RGBA")
        bg = Image.new("RGB", rgba.size, (255, 255, 255))
        bg.paste(rgba, mask=rgba.getchannel("A"))
        img = bg
    else:
        img = img.convert("RGB")

    if max(img.size) > max_side:
        img.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
    return img


def load_mask(data: bytes, size: tuple[int, int]) -> Image.Image:
    """Decode a client-supplied mask (white = subject) and match it to the image size."""
    try:
        mask = Image.open(io.BytesIO(data))
    except (UnidentifiedImageError, OSError) as exc:
        raise ImageError("Could not decode mask") from exc

    if mask.mode in ("RGBA", "LA"):
        # Canvas exports may carry the mask in alpha; use it when it isn't flat.
        alpha = mask.getchannel("A")
        if alpha.getextrema() != (255, 255):
            mask = alpha
    mask = mask.convert("L")
    if mask.size != size:
        mask = mask.resize(size, Image.Resampling.BILINEAR)
    return mask


def to_png(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=False, compress_level=6)
    return buf.getvalue()
