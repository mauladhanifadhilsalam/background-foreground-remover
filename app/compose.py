"""Turning an image and a subject mask into the final output."""

from typing import Callable, Literal, Optional

from PIL import Image, ImageFilter

Mode = Literal["remove_background", "remove_foreground"]
Fill = Literal["inpaint", "transparent"]
RGB = tuple[int, int, int]


def grow_mask(mask: Image.Image, px: int) -> Image.Image:
    """Grow (px > 0) or shrink (px < 0) the white area of a mask by roughly `px` pixels."""
    if px == 0:
        return mask
    blurred = mask.filter(ImageFilter.GaussianBlur(abs(px) / 2))
    if px > 0:
        return blurred.point(lambda v: 255 if v > 8 else 0)
    return blurred.point(lambda v: 255 if v > 247 else 0)


def refine_mask(mask: Image.Image, feather: float = 0, shift: int = 0) -> Image.Image:
    """Optionally grow/shrink the mask edge, then soften it by `feather` pixels."""
    mask = mask.convert("L")
    if shift:
        # Grow/shrink works on a hard edge; keep soft model edges otherwise.
        mask = grow_mask(mask.point(lambda v: 255 if v > 127 else 0), shift)
    if feather > 0:
        mask = mask.filter(ImageFilter.GaussianBlur(feather))
    return mask


def parse_color(value: Optional[str]) -> Optional[RGB]:
    """Parse '#rrggbb' (or 'rrggbb'); empty / 'transparent' means no colour."""
    if not value or value.lower() == "transparent":
        return None
    v = value.lstrip("#")
    if len(v) == 3:
        v = "".join(c * 2 for c in v)
    if len(v) != 6:
        raise ValueError(f"Invalid colour '{value}'")
    return int(v[0:2], 16), int(v[2:4], 16), int(v[4:6], 16)


def cutout(img: Image.Image, keep: Image.Image, bg_color: Optional[RGB] = None) -> Image.Image:
    """Keep the white part of `keep`; the rest becomes transparent or `bg_color`."""
    img = img.convert("RGB")
    if bg_color is None:
        out = img.convert("RGBA")
        out.putalpha(keep.convert("L"))
        return out
    return Image.composite(img, Image.new("RGB", img.size, bg_color), keep.convert("L"))


def compose(
    img: Image.Image,
    mask: Image.Image,
    mode: Mode = "remove_background",
    fill: Fill = "inpaint",
    bg_color: Optional[RGB] = None,
    feather: float = 0,
    shift: int = 0,
    inpaint: Optional[Callable[[Image.Image, Image.Image], Image.Image]] = None,
) -> Image.Image:
    """Apply a subject mask (white = subject) to `img`.

    remove_background keeps the subject. remove_foreground removes it, either by
    inpainting the hole or by leaving it transparent.
    """
    if mode == "remove_background":
        return cutout(img, refine_mask(mask, feather, shift), bg_color)

    if mode != "remove_foreground":
        raise ValueError(f"Unknown mode '{mode}'")

    if fill == "inpaint":
        if inpaint is None:
            raise ValueError("No inpainter available")
        # Grow the hole a little so the subject's soft edge/halo is also replaced.
        hole = grow_mask(mask.convert("L").point(lambda v: 255 if v > 32 else 0), max(shift, 0) + 6)
        return inpaint(img, hole)

    if fill == "transparent":
        subject = refine_mask(mask, feather, shift)
        return cutout(img, subject.point(lambda v: 255 - v), bg_color)

    raise ValueError(f"Unknown fill '{fill}'")
