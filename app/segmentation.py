"""Subject segmentation with rembg models."""

import threading
from collections import OrderedDict

from PIL import Image

from . import config  # also sets U2NET_HOME before rembg is imported

# name -> human readable description shown in the UI
MODELS: dict[str, str] = {
    "isnet-general-use": "ISNet - fast, general purpose",
    "u2net_human_seg": "U2Net human - fast, people",
    "isnet-anime": "ISNet anime - illustrations",
    "bria-rmbg": "BRIA RMBG 2.0 - best quality, slow on CPU (non-commercial)",
    "birefnet-general": "BiRefNet - sharp edges, slow on CPU",
    "birefnet-general-lite": "BiRefNet lite - high quality, slow on CPU",
    "birefnet-portrait": "BiRefNet portrait - people, slow on CPU",
}

_sessions: "OrderedDict[str, object]" = OrderedDict()
_lock = threading.Lock()


class UnknownModelError(ValueError):
    pass


def get_session(name: str):
    if name not in MODELS:
        raise UnknownModelError(f"Unknown model '{name}'")
    with _lock:
        if name in _sessions:
            _sessions.move_to_end(name)
            return _sessions[name]

        import onnxruntime as ort
        from rembg import new_session

        # Without this, onnxruntime's CPU arena keeps the peak allocation of the
        # large models (several GB) around after every request.
        opts = ort.SessionOptions()
        opts.enable_cpu_mem_arena = False
        _sessions[name] = new_session(name, sess_opts=opts)
        while len(_sessions) > max(config.MAX_LOADED_MODELS, 1):
            _sessions.popitem(last=False)
        return _sessions[name]


def predict_mask(img: Image.Image, model: str) -> Image.Image:
    """Return an 8-bit mask the size of `img`, white where the subject is."""
    session = get_session(model)
    mask = session.predict(img.convert("RGB"))[0].convert("L")
    if mask.size != img.size:
        mask = mask.resize(img.size, Image.Resampling.LANCZOS)
    return mask
