import anyio
import pytest
from mcp import Client

from app import mcp_server
from app.backends import LocalBackend


@pytest.fixture
def mcp(fake_models, monkeypatch):
    monkeypatch.setattr(mcp_server, "_backend", LocalBackend())
    return mcp_server.server


def call(server, name, args):
    async def go():
        async with Client(server) as client:
            return await client.call_tool(name, args)

    return anyio.run(go)


def test_lists_all_tools(mcp):
    async def go():
        async with Client(mcp) as client:
            return {t.name for t in (await client.list_tools()).tools}

    assert anyio.run(go) == {
        "list_models", "image_info", "remove_background", "remove_foreground", "get_mask",
        "apply_mask", "erase_regions", "batch_process", "warmup",
    }


def test_remove_background_with_preview(mcp, photo_path):
    r = call(mcp, "remove_background", {"input_path": str(photo_path), "preview": True})
    assert not r.is_error
    assert r.structured_content["output_path"].endswith("photo-no-background.png")
    assert [c.type for c in r.content] == ["text", "image"]


def test_erase_regions(mcp, photo_path):
    r = call(mcp, "erase_regions", {"input_path": str(photo_path), "regions": [[0.1, 0.1, 0.4, 0.4]], "units": "fraction"})
    assert not r.is_error and r.structured_content["regions"] == 1


def test_errors_are_readable(mcp, tmp_path):
    r = call(mcp, "remove_background", {"input_path": str(tmp_path / "missing.jpg")})
    assert r.is_error and "Input file not found" in r.content[0].text


def test_batch(mcp, photo_path, tmp_path):
    r = call(mcp, "batch_process", {"inputs": [str(tmp_path)], "output_dir": str(tmp_path / "out")})
    assert not r.is_error and r.structured_content["processed"] == 1
