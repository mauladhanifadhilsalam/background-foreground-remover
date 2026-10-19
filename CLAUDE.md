# Cutout Studio

Background / foreground remover: FastAPI web app (`app/main.py`, `static/`), shared model
pipeline (`app/pipeline.py`), CLI (`app/cli.py`) and MCP server (`app/mcp_server.py`).

- To *use* the tool on images, see [AGENTS.md](AGENTS.md) (MCP tools and `cutout` CLI).
- Run tests with `.venv/bin/python -m pytest` (models are faked; no downloads).
- Keep model-free logic testable: anything touching rembg/LaMa goes through
  `segmentation.predict_mask` / `inpainting.inpainter.inpaint`, which the tests monkeypatch.
