# Cutout Studio: background / foreground remover

A self-hosted web app that:

- **Removes the background.** It keeps the subject and outputs a transparent PNG or a solid background colour.
- **Removes the foreground.** It deletes the subject and fills the hole with AI inpainting (LaMa), or leaves the hole transparent.
- **Has a manual mask brush.** You can fix the AI's mask with *Keep* / *Remove* brushes, with undo and reset. You can also paint any object you want gone, without relying on detection.
- **Does batch processing.** Drop many images, process them all with the same settings, touch up any of them in the editor, and download everything as a ZIP.

All AI runs on your server with ONNX Runtime. Images never go to a third-party service.

## Tech

| Part | Tech |
| --- | --- |
| API / server | Python 3.11+, FastAPI, uvicorn |
| Subject detection | [rembg](https://github.com/danielgatis/rembg) models: ISNet (default), BRIA RMBG 2.0, BiRefNet, U2Net |
| Inpainting | [LaMa](https://github.com/advimman/lama), ONNX export [`Carve/LaMa-ONNX`](https://huggingface.co/Carve/LaMa-ONNX) |
| Image handling | Pillow, NumPy |
| Frontend | Plain HTML/CSS/JS (no build step), Canvas mask editor, JSZip (vendored) |

Models download automatically on first use into `./models` (or `MODELS_DIR`). ISNet is about 180 MB and LaMa about 200 MB. BRIA and BiRefNet range from 220 MB to 1 GB.

## Run it

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 8000
# open http://localhost:8000
```

Or with Docker:

```bash
docker build -t cutout-studio .
docker run -p 8000:8000 -v cutout-models:/models cutout-studio
```

**GPU:** install `onnxruntime-gpu` instead of `onnxruntime`, together with matching CUDA/cuDNN libraries. Sessions use CUDA automatically when it's available.

## Use with AI agents

Local AI agents can drive Cutout Studio in two ways:
- **MCP server (`cutout-mcp`).** For Claude Code, Claude Desktop, Cursor and other MCP clients. It provides tools such as `remove_background`, `remove_foreground`, `erase_regions` (erase boxes) and `batch_process`.
- **`cutout` CLI, with `--json` output.** For any agent that can run shell commands.

```bash
pip install -e .
claude mcp add cutout-studio -- "$PWD/.venv/bin/cutout-mcp"   # Claude Code
cutout remove-bg photo.jpg --json                              # any shell agent
```

Setup for each client, the tool reference and conventions are in **[AGENTS.md](AGENTS.md)**.

## How to use

1. Drop, browse or paste (Ctrl+V) one or more images.
2. Pick **Remove background** or **Remove foreground** in the left panel. For foreground removal, choose **AI inpaint** or **Transparent**.
3. The first image opens in the editor and is processed automatically. On the **Mask brush** tab, the red tint shows what will be removed:
   - **Keep** (K) / **Remove** (R) brush; `[` and `]` change the size, Ctrl+Z undoes.
   - **Reset** goes back to the AI mask. **Re-detect** runs the selected model again.
   - Press **Apply** to render the result. The **Result** tab has a before/after slider and a PNG download.
4. For batches, press **Process all**. Click any image in the list to touch up its mask. Hand-edited masks are kept when you re-run with other settings. **Download ZIP** saves all results.

## API

All endpoints take `multipart/form-data` and return `image/png`.

| Endpoint | Fields | Returns |
| --- | --- | --- |
| `GET /api/models` | n/a | JSON list of models and the default |
| `POST /api/segment` | `image`, `model?` | Grayscale mask, white = subject |
| `POST /api/compose` | `image`, `mask`, options | Result built from your own (e.g. edited) mask |
| `POST /api/process` | `image`, `model?`, options | Detection and compose in one call |

Options are:
- `mode`: `remove_background` | `remove_foreground`
- `fill`: `inpaint` | `transparent`
- `bg_color`: `#rrggbb`, or empty for transparent
- `feather`: edge blur in px
- `shift`: grow (+) or shrink (−) the mask in px

```bash
curl -F image=@photo.jpg -F mode=remove_background http://localhost:8000/api/process -o cutout.png
curl -F image=@photo.jpg -F mode=remove_foreground -F fill=inpaint http://localhost:8000/api/process -o erased.png
```

## Configuration (env vars)

| Variable | Default | Meaning |
| --- | --- | --- |
| `MODELS_DIR` | `./models` | Where model weights are cached |
| `DEFAULT_MODEL` | `isnet-general-use` | Detection model used when none is given |
| `MAX_LOADED_MODELS` | `2` | Detection models kept in memory (LRU) |
| `INFERENCE_CONCURRENCY` | `1` | Requests running models at once (bounds RAM) |
| `MAX_UPLOAD_MB` | `20` | Upload size limit |
| `MAX_SIDE` | `4096` | Larger images are downscaled to this |
| `CUTOUT_API_URL` | n/a | CLI/MCP only: send requests to this running server instead of loading models in-process |
| `CUTOUT_ALLOWED_DIRS` | n/a | CLI/MCP only: restrict readable/writable folders (`:`-separated; `;` on Windows) |

## Performance notes (4-core CPU, no GPU)

| Step | Time per image | Peak RAM |
| --- | --- | --- |
| ISNet detection | ~1–3 s | ~1 GB |
| LaMa inpainting | ~5 s (first call ~20 s while loading) | n/a |
| BRIA RMBG 2.0 / BiRefNet | 30–50 s | 7–8 GB |

On CPU, use ISNet. Switch to BRIA or BiRefNet for the best edges (hair, fur) when you have a GPU.

## Licences

The code in this repo is yours. Model licences vary:
- **BRIA RMBG 2.0** is CC BY-NC 4.0 (non-commercial only).
- ISNet, BiRefNet and U2Net use permissive licences (Apache-2.0 / MIT).
- LaMa is Apache-2.0.

Check the licence of the model you deploy before using it commercially.

## Development

```bash
pip install -e ".[dev]"
pytest
```

The tests replace the models with fakes, so they run in a couple of seconds without downloading anything. They cover the HTTP API, CLI and MCP tools, and run the agent operations through both the in-process and the HTTP backend.
