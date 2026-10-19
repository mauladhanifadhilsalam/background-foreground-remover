"""Runtime settings, read from environment variables."""

import os
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent

MODELS_DIR = Path(os.getenv("MODELS_DIR", ROOT_DIR / "models")).resolve()
# isnet-general-use is ~1s/image on CPU; bria-rmbg / birefnet-* are better but
# take 30-50s and ~8 GB RAM per image on CPU, so prefer them on a GPU box.
DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", "isnet-general-use")
# How many segmentation models may stay loaded at once (least recently used is dropped).
MAX_LOADED_MODELS = int(os.getenv("MAX_LOADED_MODELS", "2"))
# Requests allowed to run model inference at the same time (bounds peak memory).
INFERENCE_CONCURRENCY = int(os.getenv("INFERENCE_CONCURRENCY", "1"))
MAX_UPLOAD_MB = float(os.getenv("MAX_UPLOAD_MB", "20"))
MAX_SIDE = int(os.getenv("MAX_SIDE", "4096"))

# rembg reads its model cache location from U2NET_HOME.
os.environ.setdefault("U2NET_HOME", str(MODELS_DIR))
