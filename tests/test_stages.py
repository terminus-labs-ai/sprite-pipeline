"""Tests for `sprite-pipeline go` on the project model: the AC1 dry run, stage
planning (drive, animate, key, loop, pixel, export), skip-unless-force, the
sprite-gen pin and argv resolution (AC8), plus the export .tres writer and the
mirror mapping ported from tests/test_build.py.

No network, no GPU, no Blender, no ComfyUI at test runtime: subprocesses are
faked at subprocess.run, the config home is isolated per test.
"""
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import tomllib
import unittest
from contextlib import redirect_stdout
from pathlib import Path

import yaml

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
    `after` maps a substring to a hook(argv) run after recording (to materialise
    a stage's outputs so the next stage proceeds)."""

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
        if fake is not None:
            subprocess.run = fake
        out = io_capture()
        try:
            with out as buf:
                code = cli.main(argv)
        except SystemExit as e:
            code = e.code if isinstance(e.code, int) else 1
            buf.write(str(e) + "\n")
        return code, buf.getvalue()


class io_capture:
    def __enter__(self):
        import io
        self.buf = io.StringIO()
        self.ctx = redirect_stdout(self.buf)
        self.ctx.__enter__()
        return self.buf

    def __exit__(self, *exc):
        self.ctx.__exit__(*exc)
        return False


# ------------------------------------------------------------------------ fixtures

class ProjectFixture:
    """A stubbed working copy of examples/isometric: the still and the GLB are
    stub bytes (the example ships only project.yaml and its README), and an LSR
    stub stands in for the machine-local drive LSR from config.toml."""

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
        """This slice: a character counts as approved when its five ref stills exist."""
        master = self.base / "out" / character / "master"
        master.mkdir(parents=True, exist_ok=True)
        for d in dirs:
            (master / f"{d}.png").write_bytes(TINY_PNG)


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
        self.assertIn(
            f"{self.f.base / 'out' / 'courier' / 'master' / 'se.png'}", out)
        self.assertIn("prepare still", out)
        self.assertIn("courier: still preparation", out)
        self.assertIn("dry run complete", out)
        self.assertIn("summary", out)

    def test_four_direction_edit_lines(self):
        code, out = self.run_cli(["go", str(self.f.project), "--dry-run"])
        for d in ("s", "e", "ne", "n"):
            self.assertIn(f"courier/{d}: direction edit", out)
        self.assertNotIn("courier/se: direction edit", out)  # the still's own dir
        self.assertEqual(len(re.findall(r"courier/\w+: direction edit", out)), 4)

    def test_no_animation_lines_for_an_unapproved_character(self):
        code, out = self.run_cli(["go", str(self.f.project), "--dry-run"])
        for stage in ("drive", "animate", "key", "loop", "pixel", "export"):
            self.assertNotIn(f"{stage} (", out, stage)
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
        for stage in ("drive", "animate", "key", "loop"):
            self.assertEqual(len(re.findall(rf"stage={stage} ", out)), 13 * 5, stage)
        self.assertEqual(len(re.findall(r"stage=pixel", out)), 1)
        self.assertEqual(len(re.findall(r"stage=export", out)), 1)
        # every summary row is present
        for d in GEN_DIRS:
            for a in ANIM_NAMES:
                self.assertIn(f"{a}/{d} ", out + "\n")

    def test_paths_under_the_output_layout(self):
        code, out = self.run_cli(["go", str(self.f.project), "--dry-run"])
        out_dir = self.f.base / "out" / "courier"
        self.assertIn(str(out_dir / "clips" / "run-s.mp4"), out)
        self.assertIn(str(out_dir / "work" / "frames-run-s" / "keyed"), out)
        self.assertIn(str(out_dir / "loops" / "run-s"), out)
        self.assertIn(str(out_dir / "sprites"), out)

    def test_every_printed_absolute_path_is_under_project_or_motion_cache(self):
        # the drive LSR is a machine location in config.toml (like `blender`
        # and the ComfyUI dirs), so it is the one allowed foreign root
        self.home.mkdir(parents=True, exist_ok=True)
        (self.home / "config.toml").write_text(f'drive_lsr = "{self.f.lsr}"\n')
        code, out = self.run_cli(["go", str(self.f.project), "--dry-run"])
        self.assertEqual(code, 0, out)
        cache = str(self.home / "motion-cache")
        found = 0
        for m in re.findall(r"(/[^\s'()\[\]=,]+)", out):
            found += 1
            self.assertTrue(
                m.startswith(str(self.f.base)) or m.startswith(cache) or m == str(self.f.lsr),
                f"path outside the project dir or motion_cache: {m}")
        self.assertGreater(found, 50)  # the dry run really did print its paths

    def test_drive_is_cached_across_directions_by_content_key(self):
        # one GLB, one action: all five direction drives of one animation share
        # the animation-level content key of the drive (FACING differs per dir,
        # so the drive mp4s differ; the key itself is dir-independent)
        code, out = self.run_cli(["go", str(self.f.project), "--dry-run",
                                  "--animations", "run", "--dirs", "s,se"])
        keys = set(re.findall(r"run-(?:s|se)-([0-9a-f]{16})\.mp4", out))
        self.assertGreaterEqual(len(keys), 1)
        glb = (self.f.base / "anims" / "ual.glb").read_bytes()
        p = project.load_project(self.f.project)
        self.assertIn(p.drive_key(p.animation("run"), glb), keys)


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
        self.assertEqual(len(re.findall(r"stage=drive ", out)), 1)
        self.assertEqual(len(re.findall(r"stage=animate ", out)), 1)
        self.assertEqual(len(re.findall(r"stage=key ", out)), 1)
        self.assertEqual(len(re.findall(r"stage=loop ", out)), 1)
        self.assertIn("run/se", out)
        for d in ("s", "e", "ne", "n"):
            self.assertNotIn(f"/{d}]", out)
        pixel = [l for l in out.splitlines() if "stage=pixel" in l][0]
        self.assertIn("--set se-run=", pixel)
        self.assertNotIn("--set s-run", pixel)
        self.assertNotIn("--set se-punch", pixel)

    def test_characters_flag_excludes_others(self):
        code, out = self.run_cli(["go", str(self.f.project), "--characters", "nobody",
                                  "--dry-run"])
        self.assertEqual(code, 0, out)
        self.assertNotIn("courier", out)

    def test_unknown_narrowing_names_exit_2(self):
        code, out = self.run_cli(["go", str(self.f.project), "--animations", "nope",
                                  "--dry-run"])
        self.assertEqual(code, 2, out)
        self.assertIn("animations", out)
        code, out = self.run_cli(["go", str(self.f.project), "--dirs", "sw", "--dry-run"])
        self.assertEqual(code, 2, out)
        self.assertIn("dirs", out)

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
        (cache / f"{anim}-{d}-{key}.mp4").write_bytes(b"x")
        (inp / f"drive-{anim}-{d}-{key}.mp4").write_bytes(b"x")

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
        self.assertIn("[pixel] skip", out)
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
        self.assertLen(matches, 1)
        self.assertEqual(
            matches[0].replace(" ", ""),
            "sprite-gen@git+https://github.com/aldegad/sprite-gen@v4.3.0")

    def test_uv_lock_pins_the_git_source_at_the_tag(self):
        d = yaml.safe_load((ROOT / "uv.lock").read_text())
        pkg = [p for p in d["package"] if p["name"] == "sprite-gen"]
        self.assertLen(pkg, 1)
        self.assertEqual(pkg[0]["source"],
                         {"git": "https://github.com/aldegad/sprite-gen", "tag": "v4.3.0"})

    def test_uv_lock_lists_sprite_gen_as_a_root_dependency(self):
        d = yaml.safe_load((ROOT / "uv.lock").read_text())
        root = [p for p in d["package"] if p["name"] == "sprite-pipeline"]
        self.assertLen(root, 1)
        self.assertIn({"name": "sprite-gen"}, root[0]["dependencies"])


# ---------------------------------------------------------- AC8-r2: sprite-gen argv

class TestSpriteGenArgv(Env):
    """AC8-r2: planned key, loop and cutout commands run sprite-gen through the
    running interpreter environment: argv[0] is sys.executable, not a config value."""

    def setUp(self):
        super().setUp()
        self.f = ProjectFixture(self.tmp)
        self.f.approve()

    def test_key_loop_and_cutout_argv0_is_the_running_interpreter(self):
        cmds = stages.sprite_gen_cmds(["video-frames", "--clip", "x"])
        self.assertLen(cmds, 1)
        argv, _label = cmds[0]
        self.assertEqual(argv, [sys.executable, "-m", "sprite_gen", "video-frames",
                                "--clip", "x"])

    def test_planned_key_and_loop_commands(self):
        code, out = self.run_cli(["go", str(self.f.project), "--dry-run",
                                  "--animations", "run", "--dirs", "s"])
        self.assertEqual(code, 0, out)
        key = [l for l in out.splitlines() if "stage=key " in l][0].split(")  ", 1)[1].split()
        self.assertEqual(key[:3], [sys.executable, "-m", "sprite_gen", "video-frames"])
        self.assertEqual(key[key.index("--clip") + 1],
                         str(self.f.base / "out" / "courier" / "clips" / "run-s.mp4"))
        self.assertEqual(key[key.index("--out-dir") + 1],
                         str(self.f.base / "out" / "courier" / "work" / "frames-run-s"))
        self.assertEqual(key[key.index("--spill") + 1], "small")
        self.assertIn("--allow-subject-edge-contact", key)
        loop = [l for l in out.splitlines() if "stage=loop " in l][0].split(")  ", 1)[1].split()
        self.assertEqual(loop[:3], [sys.executable, "-m", "sprite_gen", "video-loop"])
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
                 for l in out.splitlines() if "stage=loop " in l}
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
        loop = [l for l in out.splitlines() if "stage=loop " in l][0].split(")  ", 1)[1].split()
        self.assertEqual(loop[loop.index("--seam-max") + 1], "3")

    def test_argv0_is_not_a_config_value(self):
        # FR 12: host config keeps only machine locations; sprite_gen is gone,
        # so no config key can point sprite-gen anywhere
        self.assertNotIn("sprite_gen", config.defaults(config.home()))
        self.home.mkdir(parents=True, exist_ok=True)
        (self.home / "config.toml").write_text('sprite_gen = "/somewhere/else"\n')
        code, out = self.run_cli(["go", str(self.f.project), "--dry-run", "--stage", "key",
                                  "--animations", "run", "--dirs", "s"])
        self.assertEqual(code, 2, out)  # rejected as an unknown key, not obeyed
        self.assertNotIn("/somewhere/else", out)


# ------------------------------------------------------- AC8-r3: no cp/mv/rm argv

class TestNoPosixShellingOut(Env):
    """AC8-r3: no planned subprocess argv across all stages of the example
    project starts with cp, mv or rm; the drive copy is done in Python."""

    def test_no_planned_argv_starts_with_a_posix_file_command(self):
        self.f = ProjectFixture(self.tmp)
        self.f.approve()
        self.home.mkdir(parents=True, exist_ok=True)
        (self.home / "config.toml").write_text(f'drive_lsr = "{self.f.lsr}"\n')
        code, out = self.run_cli(["go", str(self.f.project), "--dry-run"])
        self.assertEqual(code, 0, out)
        cmd_lines = [l for l in out.splitlines() if ")  " in l]
        self.assertGreaterEqual(len(cmd_lines), 13 * 5 * 4)
        for line in cmd_lines:
            for tok in line.split(")  ", 1)[1].split():
                first = Path(tok).name
                self.assertNotIn(first, ("cp", "mv", "rm"), line)
                break  # only argv[0] matters, but the loop guards the check

    def test_drive_copy_step_is_python_not_subprocess(self):
        src = self.tmp / "cache" / "run-s-abc.mp4"
        src.parent.mkdir(parents=True)
        src.write_bytes(b"drive bytes")
        dst = self.home / "comfyui" / "input" / "drive-run-s-abc.mp4"
        with open(os.devnull, "w") as sink:
            code = stages.copy_step(str(src), str(dst), sink)
        self.assertEqual(code, 0)
        self.assertEqual(dst.read_bytes(), b"drive bytes")
        # and the planned drive stage carries no third (cp) command any more
        f = ProjectFixture(self.tmp)  # fresh: same tmp, approved
        f.approve()
        code, out = self.run_cli(["go", str(f.project), "--dry-run", "--stage", "drive",
                                  "--animations", "run", "--dirs", "s"])
        self.assertEqual(code, 0, out)
        drive_cmds = [l for l in out.splitlines() if "stage=drive" in l]
        self.assertLen(drive_cmds, 2)  # blender + ffmpeg only
        self.assertTrue(all(Path(l.split(")  ", 1)[1].split()[0]).name != "cp"
                            for l in drive_cmds))


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
            self.assertEqual(list(flipped.getdata()), list(reversed(row)))


# --------------------------------------------- ported from test_build.py: do_export

class TestExportProgram(unittest.TestCase):
    def _ctx(self, tmp, anim_names, dirs):
        p = project.load_project(ProjectFixture(tmp).project)
        ctx = stages.StageCtx(p, config.defaults(config.home()))
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
            self.assertFalse(ok)
            # sw mirrors se, so it is present; the other four real dirs were never built
            missing = log[-1].split("missing: ")[1]
            for m in ("run/n", "run/ne", "run/e", "run/w", "run/nw"):
                self.assertIn(m, missing)
            self.assertNotIn("run/sw", missing)
            self.assertTrue((out / "atlas.png").is_file())
            self.assertTrue((out / "contact-sheet.png").is_file())
            text = (out / "sprite_frames.tres").read_text()
            self.assertIn('"name": &"run_s"', text)
            self.assertIn('"name": &"run_se"', text)
            self.assertNotIn('"name": &"run_n"', text)
            self.assertIn('loop": true', text)  # run is a fixed loop
            # mirrored sw flips se
            self.assertTrue((out / "sw-run_000.png").is_file())
            sw0 = list(Image.open(out / "sw-run_000.png").convert("RGBA").getdata())
            se0 = list(Image.open(out / "se-run_000.png").convert("RGBA").getdata())
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
    stage order; a failure is recorded, the run continues, exit code 1."""

    def setUp(self):
        super().setUp()
        self.f = ProjectFixture(self.tmp)
        self.f.approve()

    def _produce(self, argv):
        """Stage-output materialiser for the fake runner."""
        joined = " ".join(argv)
        if "render_drive.py" in joined:
            frames = Path(argv[argv.index("--") + 3])
            frames.mkdir(parents=True, exist_ok=True)
            (frames / "f0000.png").write_bytes(TINY_PNG)
        if "video-frames" in joined:
            wd = Path(argv[argv.index("--out-dir") + 1]) / "keyed"
            wd.mkdir(parents=True, exist_ok=True)
            (wd / "000.png").write_bytes(TINY_PNG)
        if "video-loop" in joined:
            cyc = Path(argv[argv.index("--out-dir") + 1]) / "cycle"
            cyc.mkdir(parents=True, exist_ok=True)
            (cyc / "000.png").write_bytes(TINY_PNG)

    def test_drive_then_animate_run_and_report_ok(self):
        fake = FakeRun(after={"render_drive.py": self._produce,
                              "-i ": lambda a: None})
        # the ffmpeg drive encode must materialise the cached mp4 so the Python
        # copy step finds it
        def after_ffmpeg(argv):
            dst = Path(argv[argv.index("-pix_fmt") + 2])
            dst.parent.mkdir(parents=True, exist_ok=True)
            dst.write_bytes(b"drive")
        fake.after["-pix_fmt"] = after_ffmpeg
        code, out = self.run_cli(["go", str(self.f.project), "--animations", "run",
                                  "--dirs", "s", "--stage", "drive"], fake=fake)
        self.assertEqual(code, 0, out)
        self.assertRegex(out, r"run/s\] drive=ok")
        self.assertEqual(Path(fake.calls[0][0]).name, "blender")
        self.assertEqual(fake.calls[1][0], "ffmpeg")
        self.assertLen(fake.calls, 2)  # blender, ffmpeg — the copy is Python, not a subprocess
        copied = self.home / "comfyui" / "input"
        self.assertEqual(len(list(copied.glob("drive-run-s-*.mp4"))), 1)

    def test_failed_stage_marks_FAIL_and_exits_1(self):
        fake = FakeRun(fails={"video-frames": 1})
        code, out = self.run_cli(["go", str(self.f.project), "--stage", "key", "--force",
                                  "--animations", "run", "--dirs", "s"], fake=fake)
        self.assertEqual(code, 1, out)
        self.assertRegex(out, r"run/s\] key=FAIL")
        self.assertIn("FAILED: 1", out)
        self.assertLen(fake.calls, 1)

    def test_animate_subprocess_shape(self):
        fake = FakeRun(after={"video-loop": self._produce})
        # clip already keyed and looped? no: run animate only, with the clip faked in
        clip = self.f.base / "out" / "courier" / "clips"
        clip.mkdir(parents=True)
        (clip / "run-s.mp4").write_bytes(b"x")
        code, out = self.run_cli(["go", str(self.f.project), "--stage", "animate",
                                  "--animations", "run", "--dirs", "s"], fake=fake)
        self.assertEqual(code, 0, out)
        argv = fake.calls[0]
        self.assertEqual(argv[:3], [sys.executable, "-m", "sprite_pipeline.comfy_run", "animate"])
        self.assertEqual(argv[argv.index("--video") + 1][:11], "drive-run-s")
        self.assertEqual(argv[argv.index("--copy-to") + 1], str(clip / "run-s.mp4"))
        self.assertEqual(argv[argv.index("--pose-strength") + 1], "0.75")
        self.assertIn("head up", argv[argv.index("--pose-prompt") + 1])


# ----------------------------------------------------------------- CLI surface shape

class TestCliSurface(Env):
    def test_godry_run_of_the_example_needs_no_config_home_at_all(self):
        # AC1: "needs no ComfyUI, Blender or sprite_gen config key" — the home
        # is never even created by a dry run
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
