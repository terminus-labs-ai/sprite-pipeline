"""Empty-set tolerance for `sprite_pipeline.pixelize` (sprite-pipeline-59)."""
import subprocess, sys, tempfile
from pathlib import Path

from PIL import Image


def make_frames(d, n=2, size=(16, 32)):
    d = Path(d); d.mkdir(parents=True, exist_ok=True)
    for i in range(n):
        im = Image.new("RGBA", size, (200, 60, 60, 255))
        im.save(d / f"{i:03d}.png")
    return d


def run_pixelize(args, cwd):
    return subprocess.run([sys.executable, "-m", "sprite_pipeline.pixelize"] + args,
                          cwd=cwd, capture_output=True, text=True)


def test_empty_set_exit_1_with_scale(tmp_path):
    ok = make_frames(tmp_path / "ok")
    bad = tmp_path / "bad"; bad.mkdir()
    ref = make_frames(tmp_path / "ref", n=1) / "000.png"
    r = run_pixelize(["--set", f"ok={ok}", "--set", f"bad={bad}",
                      "--ref", str(ref), "--scale", "0.5",
                      "--out", str(tmp_path / "out")], cwd=Path(".").resolve())
    assert r.returncode == 1
    assert "bad" in r.stdout
    assert not (tmp_path / "out" / "bad_sheet.png").exists()
    assert (tmp_path / "out" / "ok_sheet.png").exists()
    assert "Traceback" not in r.stderr


def test_happy_path_exit_0(tmp_path):
    ok = make_frames(tmp_path / "ok")
    r = run_pixelize(["--set", f"ok={ok}", "--height", "64",
                      "--out", str(tmp_path / "out")], cwd=Path(".").resolve())
    assert r.returncode == 0
    assert (tmp_path / "out" / "ok_sheet.png").exists()


def test_empty_set_height_only(tmp_path):
    ok = make_frames(tmp_path / "ok")
    bad = tmp_path / "bad"; bad.mkdir()
    r = run_pixelize(["--set", f"ok={ok}", "--set", f"bad={bad}",
                      "--height", "64", "--out", str(tmp_path / "out")],
                     cwd=Path(".").resolve())
    assert r.returncode == 1
    assert "bad" in r.stdout
    assert "IndexError" not in r.stderr


def test_all_empty_no_ref_exit_1(tmp_path):
    bad = tmp_path / "bad"; bad.mkdir()
    r = run_pixelize(["--set", f"a={bad}", "--set", f"b={bad}",
                      "--out", str(tmp_path / "out")], cwd=Path(".").resolve())
    assert r.returncode == 1
    assert "a" in r.stdout and "b" in r.stdout
    assert "Traceback" not in r.stderr
