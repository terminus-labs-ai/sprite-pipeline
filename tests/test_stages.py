"""Tests for `sprite-pipeline go` on the project model: the AC1 dry run, stage
planning (drive, animate, key, loop, pixel, export), skip-unless-force, the
sprite-gen pin and argv resolution (AC8), plus the export .tres writer and the
mirror mapping ported from tests/test_build.py.

No network, no GPU, no Blender, no ComfyUI at test runtime: subprocesses are
faked at subprocess.run, the config home is isolated per test.
"""
import io
import os
import re
import shutil
import subprocess
import sys
import tempfile
import tomllib
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

from sprite_pipeline import cli, config, project, stages

ROOT = Path(__file__).resolve().parent.parent
EXAMPLE = ROOT / "examples" / "isometric"

# 2x2 RGBA PNG, every pixel (2, 3, 5, 255): a valid stub still/GLB byte source
TINY_PNG = bytes.fromhex(
    "89504e470d0a1a0a0000000d4948445200000002000000020806000000"
    "72b60d080000001d4944415408d763f8ffff3f01078895a30000000000"
    "000000db12c6c60000000049454e44ae426082")

ANIM_NAMES = ["run", "punch", "idle", "walk", "sneak", "crouch_idle", "hack",
              "interact", "jab", "pistol_shoot", "hit", "death", "roll"]
GEN_DIRS = ["s", "se", "e", "ne", "n"]


# --------------------------------------------------------------------------- fakes

class _Result:
    def __init__(self, returncode=0):
        self.returncode = returncode
        self.stdout = ""
        self.stderr = ""


class FakeRun:
    """subprocess.run stand-in that executes nothing and records every argv.

    `fails` maps a substring of the joined argv to a nonzero exit code.
    `after` maps a substring to a hook(argv) run after recording, to materialise
    a stage's outputs so the next stage proceeds."""

    def __init__(self, fails=None, after=None):
        self.calls = []
        self.fails = fails or {}
        self.after = after or {}

    def __call__(self, argv, **kwargs):
        argv = [str(a) for a in argv]
        self.calls.append(argv)
        joined = " ".join(argv)
        for match, hook in self.after.items():
            if match in joined:
                hook(argv)
        for match, code in self.fails.items():
            if match in joined:
                return _Result(code)
        return _Result(0)


class Env(unittest.TestCase):
    """Base: isolated config home; self.run_cli() invokes the real entry point."""

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

    def run_cli(self, argv, fake=None):
        """Invoke cli.main capturing stdout and stderr; a config SystemExit
        becomes its exit code and its message joins the captured output."""
        if fake is not None:
            subprocess.run = fake
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


# ------------------------------------------------------------------------ fixtures

class ProjectFixture:
    """A stubbed working copy of examples/isometric: the still and the GLB are
    stub bytes (the example ships only project.yaml and its README), and an LSR
    stub stands in for the machine-local drive LSR named by config.toml."""

    def __init__(self, tmp):
        self.tmp = Path(tmp)
        self.base = self.tmp / "isometric"
        shutil.copytree(EXAMPLE, self.base)
        for rel in ("stills/courier.png", "anims/ual.glb"):
            f = self.base / rel
            f.parent.mkdir(parents=True, exist_ok=True)
            f.write_bytes(TINY_PNG)
        self.lsr = self.tmp / "machine" / "lsr.blend"
        self.lsr.parent.mkdir(parents=True)
        self.lsr.write_bytes(TINY_PNG)
        self.project = self.base / "project.yaml"

    def approve(self, character="courier", dirs=GEN_DIRS):
        """Approved via the interim marker: `approve --interim` records the
        still names (slice 2's rule); the hash gate accepts it while every
        named still exists. The hash rule itself is tested in test_review.py."""
        master = self.base / "out" / character / "master"
        master.mkdir(parents=True, exist_ok=True)
        for d in dirs:
            (master / f"{d}.png").write_bytes(TINY_PNG)
        code = cli.main(["approve", "--interim", str(self.project), character])
        assert code == 0, f"approve --interim exited {code}"


# ---------------------------------------------------------------- AC1-r1: dry run

class TestDryRunNeedsNothing(Env):
    """AC1-r1: go examples/isometric/project.yaml --dry-run with an empty config
    home: exit 0, still preparation and four direction-edit lines per character;
    no ComfyUI, Blender or sprite_gen config key is read."""

    def setUp(self):
        super().setUp()
        self.f = ProjectFixture(self.tmp)

    def test_exit_zero_still_preparation_and_summary(self):
        code, out = self.run_cli(["go", str(self.f.project), "--dry-run"])
        self.assertEqual(code, 0, out)
        self.assertIn(str(self.f.base / "out" / "courier" / "master" / "se.png"), out)
        self.assertIn("[courier] still preparation", out)
        self.assertIn("dry run complete", out)
        self.assertIn("summary", out)

    def test_four_direction_edit_lines(self):
        code, out = self.run_cli(["go", str(self.f.project), "--dry-run"])
        for d in ("s", "e", "ne", "n"):
            self.assertIn(f"[courier/{d}] direction edit", out)
        self.assertNotIn("[courier/se] direction edit", out)  # the still's own dir
        self.assertEqual(len(re.findall(r"courier/\w+\] direction edit", out)), 4)

    def test_no_animation_lines_for_an_unapproved_character(self):
        code, out = self.run_cli(["go", str(self.f.project), "--dry-run"])
        for stage in ("drive", "animate", "key", "loop", "pixel", "export"):
            self.assertNotIn(f"stage={stage}", out, stage)
        self.assertIn("unapproved", out)

    def test_dry_run_executes_nothing(self):
        fake = FakeRun()
        code, out = self.run_cli(["go", str(self.f.project), "--dry-run"], fake=fake)
        self.assertEqual(code, 0, out)
        self.assertEqual(fake.calls, [])

    def test_empty_config_home_has_no_sprite_gen_or_anim_src_key(self):
        c = config.load()
        self.assertNotIn("sprite_gen", c)
        self.assertNotIn("anim_src", c)
        self.assertEqual(c["motion_cache"], str(self.home / "motion-cache"))


# ---------------------------------------------------------------- AC1-r2: approved

class TestDryRunApproved(Env):
    """AC1-r2: with its five direction stills present the dry run also prints
    drive, animate, key, loop per animation x generated direction and one pixel
    and one export line; every printed path is under the project dir or motion_cache."""

    def setUp(self):
        super().setUp()
        self.f = ProjectFixture(self.tmp)
        self.f.approve()

    def test_lines_per_animation_and_direction(self):
        code, out = self.run_cli(["go", str(self.f.project), "--dry-run"])
        self.assertEqual(code, 0, out)
        for stage in ("animate", "key", "loop"):
            self.assertEqual(len(re.findall(rf"\] stage={stage} \(cmd\)  ", out)), 13 * 5, stage)
        # the drive stage plans three steps per pair: blender, ffmpeg, python copy
        self.assertEqual(len(re.findall(r"\] stage=drive \(cmd\)  ", out)), 13 * 5 * 2)
        self.assertEqual(len(re.findall(r"\] stage=drive \(python copy\)  ", out)), 13 * 5)
        self.assertEqual(len(re.findall(r"stage=pixel", out)), 1)
        self.assertEqual(len(re.findall(r"stage=export", out)), 1)
        for d in GEN_DIRS:
            for a in ANIM_NAMES:
                self.assertIn(f"{a}/{d}", out)

    def test_paths_under_the_output_layout(self):
        code, out = self.run_cli(["go", str(self.f.project), "--dry-run"])
        out_dir = self.f.base / "out" / "courier"
        self.assertIn(str(out_dir / "clips" / "run-s.mp4"), out)
        self.assertIn(str(out_dir / "work" / "frames-run-s" / "keyed"), out)
        self.assertIn(str(out_dir / "loops" / "run-s"), out)
        self.assertIn(str(out_dir / "sprites"), out)

    def test_every_printed_absolute_path_is_under_project_or_motion_cache(self):
        # the drive LSR is a machine location in config.toml (like `blender`
        # and the ComfyUI dirs); the home itself holds motion_cache and comfyui
        self.home.mkdir(parents=True, exist_ok=True)
        (self.home / "config.toml").write_text(f'drive_lsr = "{self.f.lsr}"\n')
        code, out = self.run_cli(["go", str(self.f.project), "--dry-run"])
        self.assertEqual(code, 0, out)
        # every printed absolute path starts at one of the three known roots;
        # a window per root skips prose fragments like "stills/courier.png"
        roots = [str(self.f.base), str(self.home), str(self.f.lsr)]
        found = 0
        for root in roots:
            i = 0
            while True:
                i = out.find(root, i)
                if i < 0:
                    break
                w = re.match(r"\S*", out[max(0, i - 1):]).group(0)  # whole token
                if i > 0 and out[i - 1] not in " \t([":
                    i += 1
                    continue  # mid-token occurrence, e.g. inside a longer path
                self.assertTrue(out[i - 1] in " \t([" or i == 0)
                found += 1
                i += len(root)
        self.assertGreater(found, 50)  # the dry run really did print its paths
        # and no *argv token* (the part after "(cmd)  ") escapes those roots
        for line in out.splitlines():
            if "(cmd)  " in line or "(python copy)  " in line:
                argv_part = line.split("  ", 2)[-1]
                for tok in argv_part.split():
                    tok = tok.strip("'")
                    if tok.startswith("/") and len(tok) > 4:
                        allowed = roots + [str(Path(cli.__file__).parent),  # render_drive.py
                                           str(Path(sys.executable).parent)]  # venv python
                        self.assertTrue(any(tok.startswith(a) for a in allowed),
                                        f"argv path outside the known roots: {tok}")

    def test_drive_planned_with_the_slice1_content_key(self):
        code, out = self.run_cli(["go", str(self.f.project), "--dry-run",
                                  "--animations", "run", "--dirs", "s,se"])
        self.assertEqual(code, 0, out)
        glb = (self.f.base / "anims" / "ual.glb").read_bytes()
        p = project.load_project(self.f.project)
        key = p.drive_key(p.animation("run"), glb)
        self.assertIn(f"courier-run-s-{key}.mp4", out)
        self.assertIn(f"drive-courier-run-se-{key}.mp4", out)


# ------------------------------------------------------------------- AC1-r3: flags

class TestNarrowingFlags(Env):
    def setUp(self):
        super().setUp()
        self.f = ProjectFixture(self.tmp)
        self.f.approve()

    def test_r3_animations_and_dirs_limit_the_plan(self):
        code, out = self.run_cli(["go", str(self.f.project), "--animations", "run",
                                  "--dirs", "se", "--dry-run"])
        self.assertEqual(code, 0, out)
        for stage in ("animate", "key", "loop"):
            self.assertEqual(len(re.findall(rf"\] stage={stage} \(cmd\)  ", out)), 1, stage)
        self.assertEqual(len(re.findall(r"\] stage=drive \(cmd\)  ", out)), 2)
        self.assertEqual(len(re.findall(r"\] stage=drive \(python copy\)  ", out)), 1)
        self.assertIn("[run/se]", out)
        for d in ("s", "e", "ne", "n"):
            self.assertNotIn(f"[run/{d}]", out)
        pixel = [l for l in out.splitlines() if "] stage=pixel (cmd)  " in l][0]
        self.assertIn("--set se-run=", pixel)
        self.assertNotIn("--set s-run", pixel)
        self.assertNotIn("--set se-punch", pixel)

    def test_characters_flag_selects_and_rejects(self):
        # an unknown name is rejected naming the flag (the example has one
        # character, so selecting others plans nothing of courier's either way)
        code, out = self.run_cli(["go", str(self.f.project), "--characters", "nobody",
                                  "--dry-run"])
        self.assertEqual(code, 2, out)
        self.assertIn("--characters", out)
        code, out = self.run_cli(["go", str(self.f.project), "--characters", "courier",
                                  "--dry-run"])
        self.assertEqual(code, 0, out)
        self.assertIn("[courier]", out)

    def test_unknown_narrowing_names_exit_2(self):
        code, out = self.run_cli(["go", str(self.f.project), "--animations", "nope",
                                  "--dry-run"])
        self.assertEqual(code, 2, out)
        self.assertIn("--animations", out)
        code, out = self.run_cli(["go", str(self.f.project), "--dirs", "sw", "--dry-run"])
        self.assertEqual(code, 2, out)
        self.assertIn("--dirs", out)

    def test_stage_flag_runs_only_that_stage(self):
        code, out = self.run_cli(["go", str(self.f.project), "--stage", "loop",
                                  "--dry-run", "--animations", "run", "--dirs", "se"])
        self.assertEqual(code, 0, out)
        self.assertIn("stage=loop", out)
        for stage in ("drive", "animate", "key", "pixel", "export"):
            self.assertNotIn(f"stage={stage}", out, stage)


# ------------------------------------------------------------- AC1-r4 / FR 10 skip

class TestSkipUnlessForce(Env):
    def setUp(self):
        super().setUp()
        self.f = ProjectFixture(self.tmp)
        self.f.approve()

    def _place_drive(self, anim, d):
        p = project.load_project(self.f.project)
        key = p.drive_key(p.animation(anim), p.animation(anim).source.read_bytes())
        cache = self.home / "motion-cache"
        inp = self.home / "comfyui" / "input"
        cache.mkdir(parents=True, exist_ok=True)
        inp.mkdir(parents=True, exist_ok=True)
        (cache / f"courier-{anim}-{d}-{key}.mp4").write_bytes(b"x")
        (inp / f"drive-courier-{anim}-{d}-{key}.mp4").write_bytes(b"x")

    def test_drive_skip_without_force_planned_with_force(self):
        self._place_drive("run", "s")
        code, out = self.run_cli(["go", str(self.f.project), "--dry-run", "--stage", "drive",
                                  "--animations", "run", "--dirs", "s"])
        self.assertEqual(code, 0, out)
        self.assertRegex(out, r"run/s\] drive=skip")
        self.assertNotIn("stage=drive", out)
        code, out = self.run_cli(["go", str(self.f.project), "--dry-run", "--force",
                                  "--stage", "drive", "--animations", "run", "--dirs", "s"])
        self.assertEqual(code, 0, out)
        self.assertIn("stage=drive", out)
        self.assertRegex(out, r"run/s\] drive=dry")

    def test_key_skip_and_loop_planned(self):
        keyed = self.f.base / "out" / "courier" / "work" / "frames-run-s" / "keyed"
        keyed.mkdir(parents=True)
        (keyed / "000.png").write_bytes(b"x")
        code, out = self.run_cli(["go", str(self.f.project), "--dry-run", "--stage", "key",
                                  "--animations", "run", "--dirs", "s"])
        self.assertEqual(code, 0, out)
        self.assertRegex(out, r"run/s\] key=skip")
        code, out = self.run_cli(["go", str(self.f.project), "--dry-run", "--force",
                                  "--stage", "key", "--animations", "run", "--dirs", "s"])
        self.assertEqual(code, 0, out)
        self.assertIn("stage=key", out)

    def test_pixel_skip_when_sprites_exist(self):
        sprites = self.f.base / "out" / "courier" / "sprites"
        sprites.mkdir(parents=True)
        for a in ANIM_NAMES:
            for d in GEN_DIRS:
                (sprites / f"{d}-{a}_000.png").write_bytes(b"x")
        code, out = self.run_cli(["go", str(self.f.project), "--dry-run"])
        self.assertEqual(code, 0, out)
        self.assertRegex(out, r"\[pixel\] skip")
        code, out = self.run_cli(["go", str(self.f.project), "--dry-run", "--force"])
        self.assertEqual(code, 0, out)
        self.assertNotIn("[pixel] skip", out)
        self.assertIn("stage=pixel", out)


# -------------------------------------------------------------------- AC8-r1: pin

class TestSpriteGenPin(unittest.TestCase):
    """AC8-r1: sprite-gen pinned at git+https://github.com/aldegad/sprite-gen@v4.3.0
    in pyproject.toml and uv.lock (FR 12, dec-616)."""

    def test_pyproject_declares_the_pinned_git_dependency(self):
        d = tomllib.loads((ROOT / "pyproject.toml").read_text())
        matches = [r for r in d["project"]["dependencies"] if "sprite-gen" in r]
        self.assertEqual(len(matches), 1)
        # uv pins the git dependency through [tool.uv.sources]; the requirement
        # and the source together are the pinned git URL
        src = d["tool"]["uv"]["sources"]["sprite-gen"]
        self.assertEqual(src, {"git": "https://github.com/aldegad/sprite-gen",
                               "rev": "v4.3.0"})

    def test_uv_lock_pins_the_git_source_at_the_tag(self):
        d = tomllib.loads((ROOT / "uv.lock").read_text())  # uv.lock is TOML
        pkg = [p for p in d["package"] if p["name"] == "sprite-gen"]
        self.assertEqual(len(pkg), 1)
        src = pkg[0]["source"]["git"]
        base, _, rest = src.partition("?")
        self.assertEqual(base, "https://github.com/aldegad/sprite-gen")
        self.assertIn("rev=v4.3.0", rest)  # uv records a git tag pin as rev=<tag>
        self.assertEqual(pkg[0]["version"], "4.3.0")

    def test_uv_lock_lists_sprite_gen_as_a_root_dependency(self):
        d = tomllib.loads((ROOT / "uv.lock").read_text())
        root = [p for p in d["package"] if p["name"] == "sprite-pipeline"]
        self.assertEqual(len(root), 1)
        self.assertIn({"name": "sprite-gen"}, root[0]["dependencies"])


# ---------------------------------------------------------- AC8-r2: sprite-gen argv

class TestSpriteGenArgv(Env):
    """AC8-r2: planned key, loop and cutout commands run sprite-gen through the
    running interpreter environment: argv[0] is sys.executable, not a config value."""

    def setUp(self):
        super().setUp()
        self.f = ProjectFixture(self.tmp)
        self.f.approve()

    def test_sprite_gen_cmds_argv0_is_the_running_envs_console_script(self):
        # checked on install: sprite_gen ships no __main__, so the pinned
        # package is called through its console script next to sys.executable
        cmds = stages.sprite_gen_cmds("video-frames", "--clip", "x")
        self.assertEqual(len(cmds), 1)
        argv, _label = cmds[0]
        self.assertEqual(argv, [str(Path(sys.executable).parent / "sprite-gen"),
                                "video-frames", "--clip", "x"])
        self.assertTrue(Path(argv[0]).is_file(), f"{argv[0]} not installed")

    def test_planned_key_and_loop_commands(self):
        code, out = self.run_cli(["go", str(self.f.project), "--dry-run",
                                  "--animations", "run", "--dirs", "s"])
        self.assertEqual(code, 0, out)
        key = [l for l in out.splitlines() if "] stage=key (cmd)  " in l][0].split("  ", 1)[1].split()
        self.assertEqual(key[:2], [str(stages.sprite_gen_exe()), "video-frames"])
        self.assertEqual(key[key.index("--clip") + 1],
                         str(self.f.base / "out" / "courier" / "clips" / "run-s.mp4"))
        self.assertEqual(key[key.index("--out-dir") + 1],
                         str(self.f.base / "out" / "courier" / "work" / "frames-run-s"))
        self.assertEqual(key[key.index("--spill") + 1], "small")
        self.assertIn("--allow-subject-edge-contact", key)
        loop = [l for l in out.splitlines() if "] stage=loop (cmd)  " in l][0].split("  ", 1)[1].split()
        self.assertEqual(loop[:2], [str(stages.sprite_gen_exe()), "video-loop"])
        self.assertEqual(loop[loop.index("--frames-dir") + 1],
                         str(self.f.base / "out" / "courier" / "work" / "frames-run-s" / "keyed"))
        self.assertEqual(loop[loop.index("--out-dir") + 1],
                         str(self.f.base / "out" / "courier" / "loops" / "run-s"))
        self.assertEqual(loop[loop.index("--name") + 1], "s-run")

    def test_loop_windows_from_the_project(self):
        code, out = self.run_cli(["go", str(self.f.project), "--dry-run",
                                  "--animations", "run,punch", "--dirs", "s"])
        self.assertEqual(code, 0, out)
        loops = {l.split("] ")[0][1:]: l.split(")  ", 1)[1]
                 for l in out.splitlines() if "] stage=loop (cmd)  " in l}
        run_loop = loops["run/s"].split()
        # run: looped -> fixed cycle over its 1..22 window; seam_max 0 -> no --seam-max
        self.assertEqual(run_loop[run_loop.index("--cycle") + 1], "fixed")
        self.assertEqual(run_loop[run_loop.index("--start") + 1], "1")
        self.assertEqual(run_loop[run_loop.index("--length") + 1], "22")
        self.assertEqual(run_loop[run_loop.index("--state") + 1], "run")
        self.assertNotIn("--seam-max", run_loop)
        punch = loops["punch/s"].split()
        # punch: unlooped -> play the whole clip once
        self.assertEqual(punch[punch.index("--cycle") + 1], "once")
        self.assertEqual(punch[punch.index("--start") + 1], "0")
        self.assertEqual(punch[punch.index("--length") + 1], "33")

    def test_seam_max_from_the_project_reaches_the_loop_command(self):
        code, out = self.run_cli(["go", str(self.f.project), "--dry-run",
                                  "--animations", "sneak", "--dirs", "s"])
        self.assertEqual(code, 0, out)
        loop = [l for l in out.splitlines() if "] stage=loop (cmd)  " in l][0].split("  ", 1)[1].split()
        self.assertEqual(loop[loop.index("--seam-max") + 1], "3")

    def test_argv0_is_not_a_config_value(self):
        # FR 12: host config keeps only machine locations; sprite_gen is gone,
        # so no config key can point sprite-gen anywhere
        self.assertNotIn("sprite_gen", config.defaults(config.home()))
        self.home.mkdir(parents=True, exist_ok=True)
        (self.home / "config.toml").write_text('sprite_gen = "/somewhere/else"\n')
        code, out = self.run_cli(["go", str(self.f.project), "--dry-run", "--stage", "key",
                                  "--animations", "run", "--dirs", "s"])
        # the run refuses to start: the config load exits with the error message
        self.assertEqual(code, 1, out)
        self.assertIn("unknown key", out)
        self.assertIn("sprite_gen", out)
        self.assertNotIn("/somewhere/else", out.split("config.toml: unknown key")[0])

    def test_cutout_plan_exists_and_uses_the_same_entry_point(self):
        # FR 6's SE-still cutout, planned here; slice 4 wires it into still prep
        still = self.f.base / "stills" / "courier.png"
        out_dir = self.f.base / "out" / "courier" / "master"
        cmds = stages.plan_cutout(still, out_dir)
        self.assertEqual(len(cmds), 1)
        argv, _label = cmds[0]
        self.assertEqual(argv[0], str(stages.sprite_gen_exe()))
        self.assertEqual(argv[1], "cutout")
        self.assertEqual(argv[argv.index("--source") + 1], str(still))
        self.assertEqual(argv[argv.index("--out-dir") + 1], str(out_dir))
        self.assertEqual(argv[argv.index("--key") + 1], "measured")


# ------------------------------------------------------- AC8-r3: no cp/mv/rm argv

class TestNoPosixShellingOut(Env):
    """AC8-r3: no planned subprocess argv across all stages of the example
    project starts with cp, mv or rm; the drive copy is done in Python."""

    def test_no_planned_argv_starts_with_a_posix_file_command(self):
        f = ProjectFixture(self.tmp)
        f.approve()
        code, out = self.run_cli(["go", str(f.project), "--dry-run"])
        self.assertEqual(code, 0, out)
        cmd_lines = [l for l in out.splitlines() if "(cmd)  " in l]
        self.assertGreaterEqual(len(cmd_lines), 13 * 5 * 5)  # all stages planned
        for line in cmd_lines:
            rest = line.split("(cmd)  ", 1)[1]
            first = Path(rest.split(" ")[0]).name
            self.assertNotIn(first, ("cp", "mv", "rm"), line)

    def test_drive_copy_step_is_python_not_subprocess(self):
        src = self.tmp / "cache" / "courier-run-s-abc.mp4"
        src.parent.mkdir(parents=True)
        src.write_bytes(b"drive bytes")
        dst = self.home / "comfyui" / "input" / "drive-courier-run-s-abc.mp4"
        log = []
        stages.copy_file(str(src), str(dst), log)
        self.assertEqual(dst.read_bytes(), b"drive bytes")
        self.assertIn("python", log[-1])
        # and the planned drive stage carries no third (cp) command any more
        f = ProjectFixture(self.tmp)
        f.approve()
        code, out = self.run_cli(["go", str(f.project), "--dry-run", "--stage", "drive",
                                  "--animations", "run", "--dirs", "s"])
        self.assertEqual(code, 0, out)
        drive_cmds = [l for l in out.splitlines() if "] stage=drive (cmd)  " in l]
        self.assertEqual(len(drive_cmds), 2)  # blender + ffmpeg only
        self.assertTrue(all(" cp " not in l for l in drive_cmds))
        copy_lines = [l for l in out.splitlines() if "] stage=drive (python copy)  " in l]
        self.assertEqual(len(copy_lines), 1)


# ------------------------------------------------------------------- AC8-r4: config

class TestConfigRemovedKeys(Env):
    """AC8-r4: a config.toml with a removed key errors naming the unknown key
    (FR 12: host config keeps only machine locations)."""

    def _write(self, text):
        self.home.mkdir(parents=True, exist_ok=True)
        (self.home / "config.toml").write_text(text)

    def test_sprite_gen_key_rejected_naming_it(self):
        self._write('sprite_gen = "/usr/local/bin/sprite-gen"\n')
        with self.assertRaises(SystemExit) as cm:
            config.load()
        msg = str(cm.exception)
        self.assertIn("unknown key", msg)
        self.assertIn("sprite_gen", msg)

    def test_anim_src_key_rejected_naming_it(self):
        self._write('anim_src = "/srv/anim"\n')
        with self.assertRaises(SystemExit) as cm:
            config.load()
        self.assertIn("anim_src", str(cm.exception))

    def test_machine_locations_only(self):
        self.assertEqual(set(config.defaults(config.home())),
                         {"comfy_url", "comfy_input", "comfy_output", "blender",
                          "motion_cache", "drive_lsr", "extra_path"})

    def test_defaults_and_overrides_ported(self):
        # ported from test_build.py TestConfig: defaults under the home, overrides win
        c = config.load()
        self.assertEqual(c["motion_cache"], str(self.home / "motion-cache"))
        self.assertEqual(c["comfy_input"], str(self.home / "comfyui" / "input"))
        self.assertEqual(c["comfy_output"], str(self.home / "comfyui" / "output"))
        self.assertEqual(c["blender"], "blender")
        self.assertEqual(c["comfy_url"], "http://127.0.0.1:8188")
        self._write('blender = "/opt/blender"\nmotion_cache = "/srv/cache"\n')
        c = config.load()
        self.assertEqual((c["blender"], c["motion_cache"]), ("/opt/blender", "/srv/cache"))
        self._write("bogus = 1\n")
        with self.assertRaises(SystemExit):
            config.load()

    def test_glb_paths_now_resolve_against_the_project_file(self):
        # ported from test_build.py's anim_src test with the anim_src rule
        # replaced by the project-model rule (FR 1)
        f = ProjectFixture(self.tmp)
        p = project.load_project(f.project)
        self.assertEqual(p.animation("run").source, f.base / "anims" / "ual.glb")
        text = f.project.read_text().replace(
            'source: "anims/ual.glb#Jog_Fwd_Loop"', 'source: "/x/ual.glb#Jog_Fwd_Loop"')
        f.project.write_text(text)
        with self.assertRaises(project.ProjectError) as cm:
            project.load_project(f.project)  # absolute paths are kept; missing -> named
        self.assertIn("source", str(cm.exception))
        self.assertIn("not found", str(cm.exception))


# ------------------------------------------- ported from test_build.py: .tres writer

class TestTresWriter(unittest.TestCase):
    def test_write_and_parse_back(self):
        with tempfile.TemporaryDirectory() as tmp:
            f = Path(tmp) / "sf.tres"
            stages.write_tres(f, "atlas.png",
                              [("run_s", True, [(0, 0, 10, 10), (10, 0, 10, 10)]),
                               ("punch_se", False, [(0, 10, 8, 8)])])
            text = f.read_text()
            self.assertIn('[gd_resource type="SpriteFrames" load_steps=5 format=3]',
                          text.splitlines()[0])  # self + 1 ext + 3 subs
            self.assertIn('[ext_resource type="Texture2D" path="atlas.png" id="1_atlas"]', text)
            subs = {m[0]: tuple(m[1:]) for m in re.findall(
                r'\[sub_resource type="AtlasTexture" id="([^"]+)"\]\natlas = ExtResource\("1_atlas"\)\nregion = Rect2\((\d+), (\d+), (\d+), (\d+)\)',
                text)}
            anims = re.findall(
                r'\{\n"frames": \[(.*?)\],\n"loop": (true|false),\n"name": &"([^"]+)",\n"speed": ([0-9.]+)\n\}',
                text, re.S)
            self.assertEqual([a[2] for a in anims], ["run_s", "punch_se"])
            self.assertEqual([a[1] == "true" for a in anims], [True, False])
            self.assertEqual([a[3] for a in anims], ["30.0", "30.0"])
            self.assertEqual(re.findall(r'SubResource\("([^"]+)"\)', anims[0][0]),
                             ["AtlasTexture_run_s_f0", "AtlasTexture_run_s_f1"])
            self.assertEqual(subs["AtlasTexture_run_s_f0"], ("0", "0", "10", "10"))
            self.assertEqual(subs["AtlasTexture_run_s_f1"], ("10", "0", "10", "10"))
            self.assertEqual(subs["AtlasTexture_punch_se_f0"], ("0", "10", "8", "8"))
            self.assertIn('[resource]', text)
            self.assertIn('\nanimations = [{', text)  # Godot 4: untyped array of dicts
            self.assertEqual(text.count('"duration": 1.0'), 3)  # one frame dict per frame


# ------------------------------------------ ported from test_build.py: mirror table

class TestMirrorMapping(unittest.TestCase):
    def test_mirror_table_and_order(self):
        self.assertEqual(stages.MIRROR_OF, {"sw": "se", "w": "e", "nw": "ne"})
        self.assertEqual(stages.EIGHT_DIRS, ["n", "ne", "e", "se", "s", "sw", "w", "nw"])
        for d in ("s", "se", "e", "ne", "n"):
            self.assertNotIn(d, stages.MIRROR_OF)

    def test_mirror_flips_horizontally(self):
        from PIL import Image
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            row = [(255, 0, 0, 255), (0, 255, 0, 255), (0, 0, 255, 255), (255, 255, 0, 255)]
            src = tmp / "se-run_000.png"
            im = Image.new("RGBA", (4, 1))
            im.putdata(row)
            im.save(src)
            flipped = Image.open(src).convert("RGBA").transpose(Image.FLIP_LEFT_RIGHT)
            self.assertEqual(list(flipped.get_flattened_data()), list(reversed(row)))


# --------------------------------------------- ported from test_build.py: do_export

class TestExportProgram(Env):
    def _ctx(self, tmp, anim_names, dirs):
        p = project.load_project(ProjectFixture(tmp).project)
        ctx = stages.StageCtx(p, config.defaults(Path(tmp) / "home"))
        moves = [p.animation(n) for n in anim_names]
        return ctx, moves, list(dirs), ctx.sprite_out("courier")

    def test_export_packs_available_dirs_and_flags_missing(self):
        from PIL import Image
        with tempfile.TemporaryDirectory() as tmp:
            ctx, moves, dirs, out = self._ctx(tmp, ("run",), ("s", "se"))
            out.mkdir(parents=True)
            row = [(255, 0, 0, 255), (0, 255, 0, 255), (0, 0, 255, 255), (255, 255, 0, 255)]
            for d in ("s", "se"):
                for i in range(2):
                    im = Image.new("RGBA", (4, 1))
                    im.putdata(row if i == 0 else list(reversed(row)))
                    im.save(out / f"{d}-run_{i:03d}.png")
            log = []
            ok = stages.do_export(ctx, "courier", moves, dirs, log, dry=False)
            self.assertTrue(ok, log)  # export is driven by the project: generate+mirror only
            # sw mirrors se, so it is present; no direction beyond the project's
            # set (generate + mirror) is projected, so none is reported missing
            self.assertNotIn("missing", log[-1])
            self.assertTrue((out / "atlas.png").is_file())
            self.assertTrue((out / "contact-sheet.png").is_file())
            text = (out / "sprite_frames.tres").read_text()
            self.assertIn('"name": &"run_s"', text)
            self.assertIn('"name": &"run_se"', text)
            self.assertNotIn('"name": &"run_n"', text)
            self.assertIn('loop": true', text)  # run is a fixed loop
            # mirrored sw flips se
            self.assertTrue((out / "sw-run_000.png").is_file())
            sw0 = list(Image.open(out / "sw-run_000.png").convert("RGBA").get_flattened_data())
            se0 = list(Image.open(out / "se-run_000.png").convert("RGBA").get_flattened_data())
            self.assertEqual(sw0, list(reversed(se0)))

    def test_export_ignores_pixelize_sheet_strips(self):
        """pixelize.py writes <dir>-<move>_sheet.png next to the frames; export must not pack them."""
        from PIL import Image
        with tempfile.TemporaryDirectory() as tmp:
            ctx, moves, dirs, out = self._ctx(tmp, ("run",), ("s", "se", "e", "ne", "n"))
            out.mkdir(parents=True)
            for d in ("s", "se", "e", "ne", "n"):
                for i in range(2):
                    Image.new("RGBA", (4, 4), (255, 0, 0, 255)).save(out / f"{d}-run_{i:03d}.png")
                Image.new("RGBA", (12, 4)).save(out / f"{d}-run_sheet.png")
            log = []
            ok = stages.do_export(ctx, "courier", moves, dirs, log, dry=False)
            self.assertTrue(ok, log)
            self.assertEqual(Image.open(out / "atlas.png").size, (8, 32))


# ------------------------------------------------------------------ execution (faked)

class TestExecution(Env):
    """The non-dry path with the runner faked: planned commands actually run in
    stage order; a failure is recorded and the exit code is 1."""

    def setUp(self):
        super().setUp()
        self.f = ProjectFixture(self.tmp)
        self.f.approve()

    def _produce(self, argv):
        """The fake blender writes its frames; the fake ffmpeg writes its output."""
        if "render_drive.py" in " ".join(argv):
            frames = Path(argv[argv.index("--") + 3])
            frames.mkdir(parents=True, exist_ok=True)
            (frames / "f0000.png").write_bytes(TINY_PNG)
        elif argv[0] == "ffmpeg":
            dst = Path(argv[-1])
            dst.parent.mkdir(parents=True, exist_ok=True)
            dst.write_bytes(b"drive")

    def test_drive_runs_and_the_copy_lands_in_comfy_input(self):
        # blender and ffmpeg produce their outputs; the copy step is in-process
        fake = FakeRun(after={"render_drive.py": self._produce, "ffmpeg": self._produce})
        code, out = self.run_cli(["go", str(self.f.project), "--animations", "run",
                                  "--dirs", "s", "--stage", "drive"], fake=fake)
        self.assertEqual(code, 0, out)
        self.assertRegex(out, r"run/s\] drive=ok")
        self.assertEqual(Path(fake.calls[0][0]).name, "blender")
        self.assertEqual(fake.calls[1][0], "ffmpeg")
        self.assertEqual(len(fake.calls), 2)  # the copy is Python, not a subprocess
        copied = self.home / "comfyui" / "input"
        self.assertEqual(len(list(copied.glob("drive-courier-run-s-*.mp4"))), 1)
        self.assertEqual(list(copied.glob("drive-courier-run-s-*.mp4"))[0].read_bytes(),
                         b"drive")

    def test_failed_stage_marks_FAIL_and_exits_1(self):
        fake = FakeRun(fails={"video-frames": 1})
        code, out = self.run_cli(["go", str(self.f.project), "--stage", "key", "--force",
                                  "--animations", "run", "--dirs", "s"], fake=fake)
        self.assertEqual(code, 1, out)
        self.assertRegex(out, r"run/s\] key=FAIL")
        self.assertIn("FAILED: 1", out)
        self.assertEqual(len(fake.calls), 1)

    def test_animate_subprocess_shape(self):
        fake = FakeRun()
        clip = self.f.base / "out" / "courier" / "clips"
        clip.mkdir(parents=True)
        (clip / "run-s.mp4").write_bytes(b"x")
        code, out = self.run_cli(["go", str(self.f.project), "--stage", "animate",
                                  "--animations", "run", "--dirs", "s"], fake=fake)
        # animate is done (clip exists) -> skip
        self.assertEqual(code, 0, out)
        self.assertRegex(out, r"run/s\] animate=skip")
        self.assertEqual(fake.calls, [])
        # forced: the comfy_run call is planned and executed
        code, out = self.run_cli(["go", str(self.f.project), "--stage", "animate", "--force",
                                  "--animations", "run", "--dirs", "s"], fake=fake)
        self.assertEqual(code, 0, out)
        argv = fake.calls[0]
        self.assertEqual(argv[:4],
                         [sys.executable, "-m", "sprite_pipeline.comfy", "animate"])
        self.assertTrue(argv[argv.index("--video") + 1].startswith("drive-courier-run-s"))
        self.assertEqual(argv[argv.index("--copy-to") + 1], str(clip / "run-s.mp4"))
        self.assertEqual(argv[argv.index("--pose-strength") + 1], "0.75")
        self.assertIn("head up", argv[argv.index("--pose-prompt") + 1])


# ----------------------------------------------------------------- CLI surface shape

class TestCliSurface(Env):
    def test_dry_run_of_the_example_needs_no_config_home_at_all(self):
        # AC1: "needs no ComfyUI, Blender or sprite_gen config key" — a dry run
        # never even creates the config home
        f = ProjectFixture(self.tmp)
        code, out = self.run_cli(["go", str(f.project), "--dry-run"])
        self.assertEqual(code, 0, out)
        self.assertFalse(self.home.exists())

    def test_invalid_project_exits_2_naming_the_field(self):
        f = ProjectFixture(self.tmp)
        f.project.write_text(f.project.read_text().replace("view: isometric", "view: badview"))
        code, out = self.run_cli(["go", str(f.project), "--dry-run"])
        self.assertEqual(code, 2, out)
        self.assertIn("view", out)

    def test_removed_kit_cli_positional_usage_is_gone(self):
        # `sprite-pipeline go <project.yaml>` is the interface; a bare kit file
        # without `go` is a usage error (exit 2)
        with self.assertRaises(SystemExit) as cm:
            cli.main([str(EXAMPLE / "project.yaml")])
        self.assertEqual(cm.exception.code, 2)

    def test_old_build_module_is_removed(self):
        import importlib.util
        self.assertIsNone(importlib.util.find_spec("sprite_pipeline.build"))
        self.assertFalse((ROOT / "src" / "sprite_pipeline" / "motions.yaml").exists())


if __name__ == "__main__":
    unittest.main()
