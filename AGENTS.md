# Using Cutout Studio from AI agents

Cutout Studio can be driven by local AI agents in two ways:

- **MCP server.** Use this for Claude Code, Claude Desktop, Cursor, Windsurf, Cline and any other MCP client. The agent gets tools such as `remove_background` and `erase_regions`.
- **`cutout` CLI.** Use this for any agent that can run shell commands. Add `--json` for machine-readable output.

Everything works on **local file paths**. Agents pass an input path and get back the path of the PNG that was written, plus metadata. MCP tools can also return a small preview image so the agent can check the result.

## 1. Install (once)

```bash
cd /path/to/background-foreground-remover
python -m venv .venv
.venv/bin/pip install -e .          # Windows: .venv\Scripts\pip install -e .
.venv/bin/cutout warmup             # optional: download models now (~400 MB)
```

This installs two commands into the venv: `cutout` (CLI) and `cutout-mcp` (MCP server on stdio).

## 2. Connect your agent (MCP)

**Claude Code.** The repo ships a `.mcp.json`, so starting `claude` in this folder offers the `cutout-studio` server. To use it from any folder:

```bash
claude mcp add cutout-studio --scope user -- /path/to/background-foreground-remover/.venv/bin/cutout-mcp
```

**Claude Desktop** (`claude_desktop_config.json`), **Cursor** (`~/.cursor/mcp.json`) and most other clients use the same JSON:

```json
{
  "mcpServers": {
    "cutout-studio": {
      "command": "/path/to/background-foreground-remover/.venv/bin/cutout-mcp"
    }
  }
}
```

On Windows the command is `C:\\path\\to\\background-foreground-remover\\.venv\\Scripts\\cutout-mcp.exe`.

### Faster: share models with the web server

By default each MCP server or CLI process loads the models itself. A CLI call therefore pays the model load time, around 5–30 s on CPU. If the web app is already running, point agents at it, so models load once and stay warm:

```bash
cutout serve --port 8000            # or: uvicorn app.main:app --port 8000
```

```json
{
  "mcpServers": {
    "cutout-studio": {
      "command": "/path/to/.venv/bin/cutout-mcp",
      "env": { "CUTOUT_API_URL": "http://127.0.0.1:8000" }
    }
  }
}
```

For the CLI, use `export CUTOUT_API_URL=http://127.0.0.1:8000` or `cutout --api-url http://127.0.0.1:8000 ...`.

## 3. MCP tools

| Tool | What it does |
| --- | --- |
| `list_models` | Detection models and the default |
| `image_info` | Width and height (EXIF rotation applied); `preview=true` returns the image |
| `remove_background` | Keep the subject; transparent PNG or `bg_color` |
| `remove_foreground` | Remove the subject; `fill="inpaint"` (AI) or `"transparent"` |
| `get_mask` | Save the AI mask PNG (white = subject) |
| `apply_mask` | Use your own mask file instead of AI detection (the brush equivalent) |
| `erase_regions` | Erase boxes `[x0, y0, x1, y1]` (pixels, or `units="fraction"`) with inpainting |
| `batch_process` | Run one operation on files, folders or globs; reports per-file errors |
| `warmup` | Download and load models so later calls are fast |

Common parameters:
- **`output_path`:** a file or a directory. The default is next to the input, e.g. `photo-no-background.png`.
- **`overwrite`:** existing files are never replaced unless this is set; `-2`, `-3`... is added instead.
- **`preview`:** returns a ≤512 px image of the result.
- **`model`:** see `list_models`. The fast default is ISNet; BRIA/BiRefNet are best quality but slow on CPU.
- **`feather`:** soften edges by N px.
- **`shift`:** grow (+) or shrink (−) the mask by N px.

Every tool returns JSON like:

```json
{
  "operation": "remove_background",
  "input_path": "/home/me/photo.jpg",
  "output_path": "/home/me/photo-no-background.png",
  "width": 640, "height": 800, "has_transparency": true, "elapsed_s": 2.1
}
```

Errors such as a missing file, a bad region or a server that's down come back as tool errors with a readable message.

### Example prompts

- "Remove the background from ~/Pictures/product.jpg and put it on white."
- "Erase the watermark in the bottom-right corner of scan.png." The agent calls `image_info` with `preview=true`, then `erase_regions`.
- "Cut out every photo in ~/shoot/ into ~/shoot/cutouts."
- "Remove the person from beach.jpg and fill in the background."

## 4. CLI

```bash
cutout models
cutout info photo.jpg
cutout remove-bg photo.jpg                          # -> photo-no-background.png
cutout remove-bg photo.jpg -o out/ --bg-color '#ffffff'
cutout remove-fg photo.jpg --fill inpaint           # remove the subject, AI-fill
cutout mask photo.jpg                               # -> photo-mask.png (white = subject)
cutout apply-mask photo.jpg my-mask.png --mode remove_foreground
cutout erase photo.jpg --region 120,40,380,200 --region 0.8,0.9,1,1 --units fraction
cutout batch ./photos "more/*.png" -o ./out --operation remove_background
cutout warmup
cutout mcp                                          # MCP server on stdio
cutout serve --port 8000                            # web UI + HTTP API
```

Output and exit codes:
- **`--json`** works before or after the subcommand and prints the same JSON as the MCP tools.
- **Exit code 0:** success.
- **Exit code 1:** error. The message goes to stderr, or prints as `{"error": "..."}` with `--json`.
- **`batch`:** exits 1 only if every file failed. Check `failed` / `errors` in the JSON.

## 5. HTTP API

Agents that only speak HTTP can call the web server directly with `multipart/form-data`. See the README's API section, or the interactive OpenAPI docs at `http://127.0.0.1:8000/docs`.

```bash
curl -F image=@photo.jpg -F mode=remove_background http://127.0.0.1:8000/api/process -o cutout.png
```

## Conventions

- **Masks:** white = subject (or the area to act on), black = the rest. Masks of another size are resized to the image.
- **Coordinates:** pixels, origin top-left, `[x0, y0, x1, y1]` with x1/y1 exclusive. They refer to the image as displayed, i.e. after EXIF rotation, which is what `image_info` reports.
- **Size limit:** images larger than `MAX_SIDE` (4096 px) are downscaled before processing.
- **Sandboxing (optional):** set `CUTOUT_ALLOWED_DIRS` (paths separated by `:`, or `;` on Windows) to limit which folders the tools may read and write.

## Performance (4-core CPU)

| Step | Time |
| --- | --- |
| Detection (ISNet) | ~1–3 s |
| Inpainting | ~5 s |
| Model load per fresh process | 5–30 s |

Use `CUTOUT_API_URL` or one long-lived MCP server to avoid paying the load time repeatedly. Use a GPU (`onnxruntime-gpu`) for the BRIA/BiRefNet models.
