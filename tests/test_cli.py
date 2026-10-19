import json

from app import cli


def run(capsys, *argv):
    code = cli.main(list(argv))
    out = capsys.readouterr()
    return code, out.out, out.err


def test_json_flag_before_or_after_command(capsys, photo_path, fake_models):
    code, out, _ = run(capsys, "--json", "remove-bg", str(photo_path))
    assert code == 0 and json.loads(out)["output_path"].endswith("photo-no-background.png")
    code, out, _ = run(capsys, "remove-fg", str(photo_path), "--fill", "transparent", "--json")
    assert code == 0 and json.loads(out)["fill"] == "transparent"


def test_human_output(capsys, photo_path, fake_models):
    code, out, _ = run(capsys, "erase", str(photo_path), "--region", "0,0,10,10", "--region", "20,20,30,30")
    assert code == 0 and "photo-erased.png" in out


def test_errors(capsys, tmp_path, fake_models):
    code, out, _ = run(capsys, "--json", "remove-bg", str(tmp_path / "missing.jpg"))
    assert code == 1 and "not found" in json.loads(out)["error"]
    code, _, err = run(capsys, "remove-bg", str(tmp_path / "missing.jpg"))
    assert code == 1 and err.startswith("error:")


def test_batch_and_models(capsys, photo_path, tmp_path, fake_models):
    code, out, _ = run(capsys, "batch", str(tmp_path), "-o", str(tmp_path / "out"), "--json")
    assert code == 0 and json.loads(out)["processed"] == 1
    code, out, _ = run(capsys, "models")
    assert code == 0 and "* isnet-general-use" in out
