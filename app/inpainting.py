"""Object removal with the LaMa inpainting model (ONNX, fixed 512x512 input)."""

import threading

import numpy as np
from PIL import Image, ImageFilter

from . import config

LAMA_REPO = "Carve/LaMa-ONNX"
LAMA_FILE = "lama_fp32.onnx"
LAMA_SIZE = 512


class LamaInpainter:
    def __init__(self) -> None:
        self._session = None
        self._lock = threading.Lock()

    def _get_session(self):
        with self._lock:
            if self._session is None:
                import onnxruntime as ort
                from huggingface_hub import hf_hub_download

                path = hf_hub_download(LAMA_REPO, LAMA_FILE, cache_dir=str(config.MODELS_DIR / "hf"))
                providers = [
                    p
                    for p in ("CUDAExecutionProvider", "CPUExecutionProvider")
                    if p in ort.get_available_providers()
                ]
                opts = ort.SessionOptions()
                opts.enable_cpu_mem_arena = False
                self._session = ort.InferenceSession(path, sess_options=opts, providers=providers)
            return self._session

    def _run(self, img: Image.Image, mask: Image.Image) -> Image.Image:
        """Inpaint a 512x512 RGB image where the 512x512 mask is white."""
        session = self._get_session()
        image_in = np.asarray(img, dtype=np.float32).transpose(2, 0, 1)[None] / 255.0
        mask_in = (np.asarray(mask, dtype=np.float32)[None, None] > 127).astype(np.float32)

        inputs = {}
        for inp in session.get_inputs():
            inputs[inp.name] = mask_in if "mask" in inp.name.lower() else image_in
        out = session.run(None, inputs)[0][0].transpose(1, 2, 0)
        if out.max() <= 1.5:  # some exports return 0..1, others 0..255
            out = out * 255.0
        return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8), "RGB")

    def inpaint(self, img: Image.Image, mask: Image.Image, pad_ratio: float = 0.5) -> Image.Image:
        """Fill the white area of `mask` in `img`, returning a full-size RGB image.

        Only a padded crop around the mask goes through the model, and only the
        masked pixels are written back, so the rest of the photo keeps its
        original resolution.
        """
        img = img.convert("RGB")
        mask = mask.convert("L")
        binary = mask.point(lambda v: 255 if v > 127 else 0)
        bbox = binary.getbbox()
        if bbox is None:
            return img.copy()

        x0, y0, x1, y1 = bbox
        w, h = img.size
        # Square crop around the mask with context on every side, clamped to the image.
        side = int(max(x1 - x0, y1 - y0) * (1 + 2 * pad_ratio))
        side = max(side, min(LAMA_SIZE, w, h))
        side = min(side, max(w, h))
        cx, cy = (x0 + x1) // 2, (y0 + y1) // 2
        left = min(max(cx - side // 2, 0), max(w - side, 0))
        top = min(max(cy - side // 2, 0), max(h - side, 0))
        box = (left, top, min(left + side, w), min(top + side, h))

        crop = img.crop(box)
        crop_mask = binary.crop(box)
        result = self._run(
            crop.resize((LAMA_SIZE, LAMA_SIZE), Image.Resampling.LANCZOS),
            crop_mask.resize((LAMA_SIZE, LAMA_SIZE), Image.Resampling.NEAREST),
        ).resize(crop.size, Image.Resampling.LANCZOS)

        # Blend back with a slightly softened mask to hide the seam.
        blend = crop_mask.filter(ImageFilter.MaxFilter(3)).filter(ImageFilter.GaussianBlur(2))
        out = img.copy()
        out.paste(Image.composite(result, crop, blend), box[:2])
        return out


inpainter = LamaInpainter()
