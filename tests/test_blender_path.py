"""sprite-pipeline-37: the configured blender executable.

AC1 (README Config): `blender = "blender"` is a machine location — the
drive step spawns the configured executable, and an unset key keeps the
PATH lookup. AC2 (spec FR 12): the same holds for the mannequin-profile
render, which plans through the same `plan_pair`. The rest of the argv
(-b, --python render_drive.py -- + params) and the "blender" log labels
are unchanged. No network, no GPU, no Blender: subprocess.run is faked
and the config home is isolated per test.
"""
import io
import os
import subprocess
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest import mock

from PIL import Image

import mannequins

from sprite_pipeline import cli, config
from test_stages import Env, FakeRun, ProjectFixture


class TestDriveBlenderPath(Env):
    """AC1-r1/r2: `go --stage drive` with the subprocess runner faked spawns
    the configured blender executable (plan_pair's real argv, not a stubbed
    plan, reaches the fake)."""

    def setUp(self):
        super().setUp()
        self.f = ProjectFixture(self.tmp)
        self.f.approve()

    def _produce(self, argv):
        """The fake blender writes its frames; the fake ffmpeg writes its
        output (pattern: tests/test_stages.py TestExecution._produce)."""
        if "render_drive.py" in " ".join(argv):
            frames = Path(argv[argv.index("--") + 3])
            frames.mkdir(parents=True, exist_ok=True)
            Image.new("RGBA", (4, 4), (0, 0, 0, 0)).save(frames / "f0000.png")
        elif argv[0] == "ffmpeg":
            dst = Path(argv[-1])
            dst.parent.mkdir(parents=True, exist_ok=True)
            dst.write_bytes(b"drive")

    def _drive(self, fake):
        return self.run_cli(["go", str(self.f.project), "--animations", "run",
                             "--dirs", "s", "--stage", "drive"], fake=fake)

    def test_configured_blender_is_spawned(self):
        # AC1-r1: the host config points at a non-PATH blender; the spawned
        # argv starts with it, and the plan's remaining steps (the other four
        # union directions, ffmpeg, the in-process copy) are unchanged
        self.home.mkdir(parents=True, exist_ok=True)
        (self.home / "config.toml").write_text('blender = "/opt/blender-5.1"\n')
        fake = FakeRun(after={"render_drive.py": self._produce, "ffmpeg": self._produce})
        code, out = self._drive(fake)
        self.assertEqual(code, 0, out)
        # the five union directions, then the ffmpeg of the `s` drive: the
        # planned argv (plan_pair's real output, not a stub) reached the fake
        blender = [c for c in fake.calls if "render_drive.py" in " ".join(c)]
        self.assertEqual(len(blender), 5, out)
        for c in blender:
            self.assertEqual(c[0], "/opt/blender-5.1")
            self.assertEqual(c[1], "-b")
        self.assertEqual(len(fake.calls), 6)  # the copy is Python, not a subprocess
        copied = self.home / "comfyui" / "input"
        self.assertEqual(len(list(copied.glob("drive-courier-run-s-*.mp4"))), 1)

    def test_unset_key_keeps_path_lookup(self):
        # AC1-r2: the default config (no key written) still runs "blender"
        c = config.load()
        self.assertEqual(c["blender"], "blender")
        fake = FakeRun(after={"render_drive.py": self._produce, "ffmpeg": self._produce})
        code, out = self._drive(fake)
        self.assertEqual(code, 0, out)
        blender = [c for c in fake.calls if "render_drive.py" in " ".join(c)]
        self.assertEqual(len(blender), 5, out)
        for c in blender:
            self.assertEqual(c[0], "blender")


class TestMannequinBlenderPath(unittest.TestCase):
    """AC1-r3 (AC2): `cli.render_mannequin` spawns the configured binary;
    the measured-frame flow (f*.png in the plan's frames dir -> A_TPose.png)
    is unchanged. Pattern: tests/test_arm_drop.py r1."""

    def setUp(self):
        self._t = tempfile.TemporaryDirectory()
        self.addCleanup(self._t.cleanup)
        self.tmp = Path(self._t.name)
        self._old_home = os.environ.get("SPRITE_PIPELINE_HOME")
        os.environ["SPRITE_PIPELINE_HOME"] = str(self.tmp / "home")
        self.addCleanup(self._restore_home)

    def _restore_home(self):
        if self._old_home is None:
            os.environ.pop("SPRITE_PIPELINE_HOME", None)
        else:
            os.environ["SPRITE_PIPELINE_HOME"] = self._old_home

    def test_mannequin_render_uses_configured_blender(self):
        calls = []

        class Res:
            returncode = 0
            stdout = ""
            stderr = ""

        def fake_run(argv, **kw):
            argv = [str(a) for a in argv]
            calls.append(argv)
            if any("render_drive.py" in a for a in argv):
                frames = Path(argv[argv.index("--") + 3])
                frames.mkdir(parents=True, exist_ok=True)
                mannequins.tpose_render().save(frames / "f0001.png")
            elif argv[0] == "ffmpeg":
                dst = Path(argv[-1])
                dst.parent.mkdir(parents=True, exist_ok=True)
                dst.write_bytes(b"drive")
            return Res()

        old_run = subprocess.run
        subprocess.run = fake_run
        self.addCleanup(setattr, subprocess, "run", old_run)
        home = self.tmp / "home"
        home.mkdir(parents=True, exist_ok=True)
        (home / "config.toml").write_text('blender = "/opt/blender-5.1"\n')
        glb = self.tmp / "ual-female.glb"
        glb.write_bytes(b"stub glb bytes")
        mannequins.register(glb.read_bytes(), glb_name="ual-female.glb")
        old_argv = list(sys.argv)
        self.addCleanup(setattr, sys, "argv", old_argv)
        sys.argv = ["sprite-pipeline"]  # Project(path=Path.cwd())
        # config.load is patched to the home's defaults plus the written
        # blender key: render_mannequin's StageCtx reads cfg["blender"] from it
        cfg = dict(config.defaults(home))
        cfg["blender"] = "/opt/blender-5.1"
        with mock.patch.object(config, "load", return_value=cfg):
            buf, err = io.StringIO(), io.StringIO()
            with redirect_stdout(buf), redirect_stderr(err):
                shot = cli.render_mannequin(glb, "s")
        blender = [c for c in calls if any("render_drive.py" in a for a in c)]
        self.assertEqual(len(blender), 1, calls)
        self.assertEqual(blender[0][0], "/opt/blender-5.1")
        self.assertEqual(blender[0][1], "-b")
        # the measured-frame flow: the frame the fake wrote in the plan's
        # frames dir becomes the render's A_TPose.png
        frames = Path(blender[0][blender[0].index("--") + 3])
        self.assertEqual(Path(shot).parent, frames)
        self.assertTrue((frames / "A_TPose.png").is_file())


if __name__ == "__main__":
    unittest.main()
