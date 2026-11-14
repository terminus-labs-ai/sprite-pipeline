"""Tests for run_cmds failure output (AC: stderr tail on step failure).

When a step fails, `go` prints the failing command and its stderr tail, so
the cause is visible without rerunning by hand.

No network, no GPU, no Blender, no ComfyUI — subprocesses faked at
subprocess.run, config home isolated per test.
"""
import io
import os
import subprocess
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
EXAMPLE = ROOT / "examples" / "isometric"


def _make_tiny_png():
    """A valid 1x1 RGBA PNG."""
    return bytes.fromhex(
        "89504e470d0a1a0a0000000d494844520000000200000002"
        "080600000072b60d080000001d4944415408d763f8ffff3f"
        "01078895a300000000000000db12c6c60000000049454e44"
        "ae426082")


# --------------------------------------------------------------------------- fakes

class _Result:
    def __init__(self, returncode=0, stdout="", stderr=""):
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


class FakeRun:
    """subprocess.run stand-in: records every call and returns a controlled
    result.

    `fails` maps a substring of the joined argv to (returncode, stderr).
    """

    def __init__(self, fails=None):
        self.calls = []
        self.fails = fails or {}

    def __call__(self, argv, **kwargs):
        argv = [str(a) for a in argv]
        self.calls.append(argv)
        joined = " ".join(argv)
        for match, (code, stderr) in self.fails.items():
            if match in joined:
                return _Result(returncode=code, stderr=stderr)
        return _Result(0)


# --------------------------------------------------------------------------- env

class Env(unittest.TestCase):
    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmpdir.cleanup)
        self.tmp = Path(self._tmpdir.name)
        self.home = self.tmp / "home"
        self._old_home = os.environ.get("SPRITE_PIPELINE_HOME")
        os.environ["SPRITE_PIPELINE_HOME"] = str(self.home)
        self.addCleanup(self._restore_home)
        self._old_run = subprocess.run

    def _restore_home(self):
        if self._old_home is None:
            os.environ.pop("SPRITE_PIPELINE_HOME", None)
        else:
            os.environ["SPRITE_PIPELINE_HOME"] = self._old_home

    def tearDown(self):
        subprocess.run = self._old_run

    def _make_project(self):
        """A stubbed working copy of examples/isometric."""
        base = self.tmp / "isometric"
        import shutil
        shutil.copytree(EXAMPLE, base)
        for rel in ("stills/courier.png", "anims/ual.glb"):
            f = base / rel
            f.parent.mkdir(parents=True, exist_ok=True)
            f.write_bytes(_make_tiny_png())
        # copy workflow files
        wf_dir = base / "workflows"
        wf_dir.mkdir(parents=True, exist_ok=True)
        for name in ("edit.json", "animate.json"):
            src = ROOT / "src" / "sprite_pipeline" / "workflows" / name
            (wf_dir / name).write_bytes(src.read_bytes())
        return base / "project.yaml"

    def run_cli(self, argv, fake=None):
        if fake is not None:
            subprocess.run = fake
        buf = io.StringIO()
        err = io.StringIO()
        try:
            from sprite_pipeline import cli
            with redirect_stdout(buf), redirect_stderr(err):
                code = cli.main(argv)
        except SystemExit as e:
            code = e.code if isinstance(e.code, int) else 1
            buf.write(str(e) + "\n")
        buf.write(err.getvalue())
        return code, buf.getvalue()


# ----------------------------------------------------------------------- tests

class TestRunCmdsFailureOutput(Env):
    """When a step fails, `go` prints the failing command and its stderr tail."""

    def test_failed_pair_stage_prints_command_and_stderr(self):
        """Fake sprite-gen video-frames to exit 2 with "unrecognized arguments:
        --foo" on stderr; go's output must contain the command and that line.

        This targets the pair-stage call site (run_cmds → main)."""
        import shutil
        project = self._make_project()
        # approve so pair stages run
        master = self.tmp / "isometric" / "out" / "courier" / "master"
        master.mkdir(parents=True)
        for d in ["s", "se", "e", "ne", "n"]:
            (master / f"{d}.png").write_bytes(_make_tiny_png())
        with open(project) as f:
            proj_text = f.read()
        # approve via interim marker
        from sprite_pipeline import cli
        code = cli.main(["approve", "--interim", str(project), "courier"])
        self.assertEqual(code, 0)

        fake = FakeRun(fails={
            "video-frames": (2, "unrecognized arguments: --foo\n")
        })
        code, out = self.run_cli([
            "go", str(project), "--stage", "key", "--force",
            "--animations", "run", "--dirs", "s"
        ], fake=fake)

        self.assertEqual(code, 1, out)
        # must show the failing command
        self.assertIn("video-frames", out)
        # must show the stderr line
        self.assertIn("unrecognized arguments: --foo", out)
        # must show the exit code
        self.assertIn("exited 2", out)

    def test_failed_pair_stage_prints_command_and_label(self):
        """The output includes the label (e.g. 'sprite-gen') so the tool is
        identifiable."""
        project = self._make_project()
        master = self.tmp / "isometric" / "out" / "courier" / "master"
        master.mkdir(parents=True)
        for d in ["s", "se", "e", "ne", "n"]:
            (master / f"{d}.png").write_bytes(_make_tiny_png())
        from sprite_pipeline import cli
        code = cli.main(["approve", "--interim", str(project), "courier"])
        self.assertEqual(code, 0)

        fake = FakeRun(fails={
            "video-frames": (1, "some error\n")
        })
        code, out = self.run_cli([
            "go", str(project), "--stage", "key", "--force",
            "--animations", "run", "--dirs", "s"
        ], fake=fake)

        self.assertEqual(code, 1, out)
        # label present
        self.assertIn("sprite-gen", out)
        # command present
        self.assertIn("video-frames", out)

    def test_successful_steps_remain_quiet(self):
        """When no step fails, no extra failure output appears."""
        project = self._make_project()
        master = self.tmp / "isometric" / "out" / "courier" / "master"
        master.mkdir(parents=True)
        for d in ["s", "se", "e", "ne", "n"]:
            (master / f"{d}.png").write_bytes(_make_tiny_png())
        from sprite_pipeline import cli
        code = cli.main(["approve", "--interim", str(project), "courier"])
        self.assertEqual(code, 0)

        fake = FakeRun()  # everything succeeds
        code, out = self.run_cli([
            "go", str(project), "--stage", "key", "--force",
            "--animations", "run", "--dirs", "s"
        ], fake=fake)

        self.assertEqual(code, 0, out)
        # no failure lines at all
        self.assertNotIn("exit", out)
        self.assertNotIn("FAILED:", out)


if __name__ == "__main__":
    unittest.main()
