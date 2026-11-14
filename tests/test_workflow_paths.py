"""Tests for workflow path resolution in the project model (spec AC6, AC10;
FR 1, FR 3, FR 9, amendment 1).

An animation's `workflow` replaces `workflows.animate` for that animation
(loaded, not set by hand); a missing `workflows` key or a missing individual
key defaults to the packaged workflow file; a given path resolves against the
project file. The example project carries no `workflows` key and its resolved
workflows are the packaged files. No network, GPU, Blender or ComfyUI.
"""
import json
import os
import shutil
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from importlib import resources
from pathlib import Path

import yaml
from PIL import Image

from sprite_pipeline import config, project, stages

HERE = Path(__file__).resolve().parent
FIXTURES = HERE / "fixtures" / "workflow_paths"
STILL = (FIXTURES / "still.png").read_bytes()
PKG_WF = resources.files("sprite_pipeline") / "workflows"


def base_project(workflows=None, workflow_field=False):
    """A minimal valid project; `workflows` is the full key (or None to omit
    it); `workflow_field` puts a per-animation `workflow` on the alt entry."""
    wf = ""
    if workflows is not None:
        wf = "workflows: {" + ", ".join(f"{k}: {v}" for k, v in workflows.items()) + "}" + "\n"
    alt_wf = ", workflow: workflows/alt.json" if workflow_field else ""
    return f"""style: "2D game sprite art"
view: isometric
directions:
  generate: [s, se]
  mirror: {{}}
sprite: {{scale: 0.2, colors: 40, accent_boost: 3, fps: 30, frame_size: 704}}
naming: "{{animation}}_{{dir}}"
export: godot
seed: 7
pose_strength: 0.75
{wf}characters:
  - {{name: courier, still: stills/courier.png, describe: "a courier", pronoun: she}}
animations:
  - {{name: run, source: "anims/ual.glb#Jog_Fwd_Loop", frames: 22, start: 1, clip_frames: 45, loop: true, prompt: "{{who}} jogging"}}
  - {{name: alt, source: "anims/ual.glb#Jog_Fwd_Loop", frames: 22, start: 1, clip_frames: 45, loop: true, prompt: "{{who}} jogging"{alt_wf}}}
"""


def make_project(tmp, text, with_alt=True):
    """Write project.yaml plus the stub still, GLB and (optionally) the
    per-animation workflow file it may reference. Returns the yaml path."""
    base = Path(tmp)
    (base / "stills").mkdir(parents=True, exist_ok=True)
    (base / "stills" / "courier.png").write_bytes(STILL)
    (base / "anims").mkdir(parents=True, exist_ok=True)
    (base / "anims" / "ual.glb").write_bytes(STILL)
    if with_alt:
        wf_dir = base / "workflows"
        wf_dir.mkdir(parents=True, exist_ok=True)
        (wf_dir / "alt.json").write_text(json.dumps(
            {"1": {"class_type": "Test", "inputs": {},
                   "_meta": {"title": "sp:prompt"}}}))
    (base / "project.yaml").write_text(text)
    return base / "project.yaml"


def animate_argv(ctx, anim, d="s"):
    """The --workflow value of the planned animate command for anim."""
    argv = ctx.plan_pair("courier", anim, d, "animate")[0][0]
    return argv[argv.index("--workflow") + 1]


def stubbed_example(tmp):
    """A working copy of examples/isometric with stubbed still and GLB.
    Returns (project.yaml path, base dir)."""
    base = Path(tmp) / "isometric"
    shutil.copytree(HERE.parent / "examples" / "isometric", base)
    (base / "anims").mkdir(parents=True, exist_ok=True)
    (base / "anims" / "ual.glb").write_bytes(STILL)
    (base / "stills").mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (64, 80), (10, 200, 10)).save(base / "stills" / "courier.png")
    return base / "project.yaml", base


class Env(unittest.TestCase):
    """Isolated config home; run_cli() invokes the real entry point (the
    dry run plans in-process and runs no subprocess)."""

    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmpdir.cleanup)
        self.tmp = Path(self._tmpdir.name)
        self.home = self.tmp / "home"
        self._old_home = os.environ.get("SPRITE_PIPELINE_HOME")
        os.environ["SPRITE_PIPELINE_HOME"] = str(self.home)
        self.addCleanup(self._restore_home)

    def _restore_home(self):
        if self._old_home is None:
            os.environ.pop("SPRITE_PIPELINE_HOME", None)
        else:
            os.environ["SPRITE_PIPELINE_HOME"] = self._old_home

    def run_cli(self, argv):
        import io
        import sys
        from sprite_pipeline import cli
        buf = io.StringIO()
        err = io.StringIO()
        try:
            with redirect_stdout(buf), redirect_stderr(err):
                code = cli.main(argv)
        except SystemExit as e:
            code = e.code if isinstance(e.code, int) else 1
            buf.write(str(e) + "\n")
        buf.write(err.getvalue())
        return code, buf.getvalue()


def cfg(home):
    return config.defaults(home)


class TestAnimationWorkflowField(unittest.TestCase):
    """AC6-r9: `animations[].workflow` is loaded, resolved against the
    project file, and replaces `workflows.animate` for that animation."""

    def test_r9_loaded_and_resolved(self):
        with tempfile.TemporaryDirectory() as t:
            p = make_project(t, base_project(
                workflows={"edit": "workflows/edit.json",
                           "animate": "workflows/animate.json"},
                workflow_field=True))
            base = p.parent
            proj = project.load_project(p)
            # the field is loaded: animation 1 (alt) carries the absolute path,
            # animation 0 (run) is unset
            self.assertEqual(proj.animations[1].workflow, base / "workflows" / "alt.json")
            self.assertIsNone(proj.animations[0].workflow)
            # the planned animate command names alt.json for the override and
            # the default animate workflow for the other
            ctx = stages.StageCtx(proj, cfg(Path("/nonexistent-home")))
            self.assertEqual(animate_argv(ctx, proj.animation("alt")),
                             str(base / "workflows" / "alt.json"))
            self.assertEqual(animate_argv(ctx, proj.animation("run")),
                             str(base / "workflows" / "animate.json"))


class TestPackagedDefaults(unittest.TestCase):
    """AC6-r10, r11: a missing `workflows` key (or individual key) defaults
    to the packaged workflow file; a given path resolves against the file."""

    def test_r10_missing_workflows_key_defaults_to_packaged(self):
        with tempfile.TemporaryDirectory() as t:
            p = make_project(t, base_project(), with_alt=False)
            proj = project.load_project(p)
            self.assertEqual(str(proj.workflows["edit"]),
                             str(PKG_WF / "edit.json"))
            self.assertEqual(str(proj.workflows["animate"]),
                             str(PKG_WF / "animate.json"))
            # both packaged files exist (FR 9)
            self.assertTrue((PKG_WF / "edit.json").is_file())
            self.assertTrue((PKG_WF / "animate.json").is_file())
            # and the plan uses them
            ctx = stages.StageCtx(proj, cfg(Path("/nonexistent-home")))
            self.assertEqual(animate_argv(ctx, proj.animation("run")),
                             str(PKG_WF / "animate.json"))

    def test_r11_partial_workflows_defaults_only_the_missing_key(self):
        with tempfile.TemporaryDirectory() as t:
            p = make_project(t, base_project(workflows={"animate": "mine.json"}),
                             with_alt=False)
            base = p.parent
            (base / "mine.json").write_text(json.dumps(
                {"1": {"class_type": "Test", "inputs": {},
                       "_meta": {"title": "sp:prompt"}}}))
            proj = project.load_project(p)
            self.assertEqual(str(proj.workflows["animate"]), str(base / "mine.json"))
            self.assertEqual(str(proj.workflows["edit"]), str(PKG_WF / "edit.json"))
            ctx = stages.StageCtx(proj, cfg(Path("/nonexistent-home")))
            self.assertEqual(animate_argv(ctx, proj.animation("run")),
                             str(base / "mine.json"))


class TestDryRunPackagedEdit(Env):
    """AC6-r12: `go --dry-run` on a project with no `workflows` key and one
    unapproved character prints each direction-edit line naming the packaged
    edit.json path; exit 0."""

    def test_r12_dry_run_names_the_packaged_edit(self):
        p, _ = stubbed_example(self.tmp)
        d = yaml.safe_load(p.read_text())
        d.pop("workflows", None)  # the example must not name one (AC10-r2)
        p.write_text(yaml.safe_dump(d, sort_keys=False))
        code, out = self.run_cli(["go", str(p), "--dry-run"])
        self.assertEqual(code, 0, out)
        edit_lines = [l for l in out.splitlines() if "direction edit" in l]
        self.assertTrue(edit_lines, out)
        for l in edit_lines:
            self.assertIn(str(PKG_WF / "edit.json"), l)


class TestExampleProject(unittest.TestCase):
    """AC10-r2: the example still validates with its 13 animations unchanged,
    has no `workflows` key and resolves to the packaged workflow files."""

    def test_r2_example_uses_packaged_workflows(self):
        with tempfile.TemporaryDirectory() as t:
            p, _ = stubbed_example(t)
            d = yaml.safe_load(p.read_text())
            self.assertNotIn("workflows", d)
            proj = project.load_project(p)
            self.assertEqual(len(proj.animations), 13)
            self.assertEqual([a.name for a in proj.animations], [
                "run", "punch", "idle", "walk", "sneak", "crouch_idle", "hack",
                "interact", "jab", "pistol_shoot", "hit", "death", "roll"])
            self.assertEqual(str(proj.workflows["edit"]), str(PKG_WF / "edit.json"))
            self.assertEqual(str(proj.workflows["animate"]), str(PKG_WF / "animate.json"))
            # none of the 13 animations carries a per-animation override
            for a in proj.animations:
                self.assertIsNone(a.workflow)


if __name__ == "__main__":
    unittest.main()
