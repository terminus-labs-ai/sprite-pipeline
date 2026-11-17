"""Anchored still preparation, the A/T-pose rule, the profile gate and the
`mannequin-profile` subcommand (spec AC15, FR 4, FR 6, FR 14; amendments 3 and 4).

- r1: `anchor` on a synthetic 500x900 still (flat background, body bbox 300 px
  tall x 200 wide, feet 10 px from the frame bottom) yields the 832x1024 master
  on flat chroma green (31, 203, 25) whose measured body height is the profile
  target (the frame_size 704 rule: profile body_h x 1024 / frame_size, floor_y
  525 likewise), centred on x 416.
- r2: the same source at frame_size 1024 measures 324 +/- 1 and floor 685 +/- 1.
- r3: a non-standing still (lying / too wide / too small / floating) makes `go`
  exit 2 naming the file and the failed rule.
- r4: a project whose source GLB has no mannequin profile exits 2 printing the
  sha256 prefix and the `sprite-pipeline mannequin-profile <glb>` command.
- r5: with ComfyUI faked, the edit step receives the anchored master as
  `--image` and the keyed file is `master/s.keyed.png`.
- r7 (amendment 4): `mannequin-profile <glb>` with the render measurement faked
  prints the measured JSON and appends the entry to the user file
  `~/.sprite-pipeline/mannequin_profiles.json`.
- r8: re-run on the same GLB, it replaces the existing entry for that hash (no
  duplicate).

The anchor rows state their expected numbers independently of production code,
as the literal FR 6 formula over the profile, each +/- 1: 324 x 1024 / 704 =
471.3 and 525 x 1024 / 704 = 763.6 at frame_size 704 (AC15's 471 / 764,
amendment 7), and at frame_size 1024:
324 x 1024 / 1024 = 324, (525 + (1024 - 704) // 2) = 685.

No network, GPU, Blender or ComfyUI: subprocesses are faked at subprocess.run,
the config home is isolated per test, the mannequin profiles are injected.
"""
import hashlib
import io
import json
import os
import shutil
import subprocess
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

from PIL import Image

import mannequins
from sprite_pipeline import cli, config, metrics, project, stages, stills

ROOT = Path(__file__).resolve().parent.parent
EXAMPLE = ROOT / "examples" / "isometric"

# the flat chroma green of an anchored master (FR 6), stated independently
CHROMA = (31, 203, 25)
# the default drive resolution and the packaged female mannequin's geometry
FRAME = 704
BODY_H, FLOOR_Y = 324, 525

# a valid 2x2 PNG: the stub GLB bytes (its profile is registered per test)
STUB_GLB = Image.new("RGB", (2, 2), (2, 3, 5))


def _stub_glb_bytes():
    buf = io.BytesIO()
    STUB_GLB.save(buf, format="PNG")
    return buf.getvalue()


def profile(body_h=BODY_H, floor_y=FLOOR_Y, resolution=FRAME):
    """A mannequin profile in the FR 14 schema."""
    return {"glb_name": "test-mannequin.glb", "resolution": resolution,
            "body_h": body_h, "floor_y": floor_y}


def expected_target(frame_size, prof=None):
    """FR 6's target on the 832x1024 edit canvas, stated independently of
    production code: body_h x 1024 / frame_size."""
    prof = prof or profile()
    return prof["body_h"] * 1024 / frame_size


def expected_floor(frame_size, prof=None):
    """FR 6/FR 14's floor line on the edit canvas: the 704 floor shifted by
    (max(frame_size, 704) - 704) // 2, then x 1024 / frame_size."""
    prof = prof or profile()
    canvas = max(frame_size, FRAME)
    return (prof["floor_y"] + (canvas - FRAME) // 2) * 1024 / frame_size


def spec_still():
    """The AC15 source: a 500x900 still on a flat background whose figure's
    bbox is 300 px tall x 200 px wide, its feet 10 px from the frame bottom."""
    return mannequins.pose_frame(500, 900, body_h=300, body_w=200,
                                 feet_from_bottom=10)


class AnchorCase(unittest.TestCase):
    """Unit rows over `stills.anchor` / `stills.check`."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.tmp = Path(self._tmp.name)

    def _anchor(self, im, frame_size, prof=None):
        src = self.tmp / "src.png"
        dst = self.tmp / "master.png"
        im.save(src)
        stills.anchor(src, dst, frame_size, prof or profile())
        return dst

    def _assert_geometry(self, dst, frame_size, prof=None):
        m = metrics.measure(dst)
        self.assertAlmostEqual(m["body_h"], expected_target(frame_size, prof), delta=1)
        self.assertAlmostEqual(m["floor_y"], expected_floor(frame_size, prof), delta=1)
        # centred horizontally: the measured centre sits on the canvas centre
        self.assertAlmostEqual(m["centre_x"], 416, delta=1)
        return m

    # ------------------------------------------------------------------ AC15-r1
    def test_r1_anchor_is_832x1024_on_flat_green_at_the_profile_target(self):
        """AC15-r1: the 500x900 still anchors to the 832x1024 master on flat
        chroma green at the FR 6 profile target, centre x 416 (704)."""
        dst = self._anchor(spec_still(), 704)
        out = Image.open(dst)
        self.assertEqual(out.size, (832, 1024))
        # every pixel outside the figure is the flat chroma green
        self.assertEqual(out.getpixel((0, 0)), CHROMA)
        self.assertEqual(out.getpixel((831, 0)), CHROMA)
        self.assertEqual(out.getpixel((0, 1023)), CHROMA)
        self.assertEqual(out.getpixel((831, 1023)), CHROMA)
        # the FR 6 formula at 704: 324 x 1024 / 704 = 471.3 and
        # 525 x 1024 / 704 = 763.6, each +/- 1 (AC15's 471 / 764,
        # amendment 7)
        self._assert_geometry(dst, 704)

    def test_r1_green_is_the_flat_chroma_colour_everywhere_off_the_figure(self):
        """Every background pixel of the master is (31, 203, 25) — a corner
        fill or a gradient is not acceptable."""
        import numpy

        dst = self._anchor(spec_still(), 704)
        arr = numpy.asarray(Image.open(dst).convert("RGB"), dtype=numpy.int16)
        corner = arr[0, 0]
        self.assertEqual(tuple(int(c) for c in corner), CHROMA)
        off_figure = (abs(arr - corner) <= 40).all(axis=2)
        self.assertTrue((arr[off_figure] == numpy.asarray(CHROMA)).all())

    def test_r1_check_returns_the_measurement_of_a_standing_still(self):
        """`stills.check` accepts the AC15 source and reports its bbox."""
        src = self.tmp / "src.png"
        spec_still().save(src)
        self.assertEqual(stills.check(src),
                         {"body_h": 300, "floor_y": 889, "centre_x": 249.5})

    # ------------------------------------------------------------------ AC15-r2
    def test_r2_frame_size_1024_measures_324_at_floor_685(self):
        """AC15-r2: the same source at frame_size 1024 gives body 324 +/- 1 and
        floor 685 +/- 1 (525 + (1024 - 704) // 2), still centred on x 416."""
        dst = self._anchor(spec_still(), 1024)
        self.assertEqual(Image.open(dst).size, (832, 1024))
        m = self._assert_geometry(dst, 1024)
        self.assertAlmostEqual(m["body_h"], 324, delta=1)
        self.assertAlmostEqual(m["floor_y"], 685, delta=1)

    def test_r2_other_frame_sizes_follow_the_rule(self):
        """The rule at two more frame sizes (1152, 1408), corrected to the
        FR 6/FR 14 formula: body_h x 1024 / frame_size and the shifted floor
        floor_y(frame_size) x 1024 / frame_size — 288 / 665.8 at 1152, 235.6 /
        637.8 at 1408 (the original 198 / 162, 472 / 509 were self-authored
        values that contradicted the formula)."""
        for frame_size, want_h, want_floor in (
                (1152, 324 * 1024 / 1152, (525 + (1152 - 704) // 2) * 1024 / 1152),
                (1408, 324 * 1024 / 1408, (525 + (1408 - 704) // 2) * 1024 / 1408)):
            dst = self._anchor(spec_still(), frame_size)
            m = metrics.measure(dst)
            self.assertAlmostEqual(m["body_h"], want_h, delta=1.5)
            self.assertAlmostEqual(m["floor_y"], want_floor, delta=2.0)
            self.assertAlmostEqual(m["centre_x"], 416, delta=1.5)

    def test_r2_larger_mannequin_anchors_larger(self):
        """A profile with a bigger body (the male mannequin, 345/538) anchors
        the same still bigger and lower: 502 / 784 at frame_size 704."""
        prof = profile(body_h=345, floor_y=538)
        dst = self._anchor(spec_still(), 704, prof)
        m = metrics.measure(dst)
        self.assertAlmostEqual(m["body_h"], 345 * 1024 / 704, delta=1.5)
        self.assertAlmostEqual(m["floor_y"], 538 * 1024 / 704, delta=1.5)

    def test_r1_anchor_rejects_a_non_standing_still(self):
        """`anchor` itself refuses a figure that fails FR 6, naming the rule."""
        src = self.tmp / "src.png"
        mannequins.pose_frame(704, 704, body_h=100, body_w=100,
                              feet_from_bottom=10).save(src)
        with self.assertRaises(metrics.FigureError) as cm:
            stills.anchor(src, self.tmp / "m.png", 704, profile())
        self.assertEqual(cm.exception.reason, "too small")

    def test_no_1280_canvas_survives_the_module(self):
        """FR 6: the 1280 reference canvas and its fill-fit are retired."""
        self.assertFalse(hasattr(stills, "REF_SIZE"))
        self.assertFalse(hasattr(stills, "REF_OFFSET"))
        self.assertFalse(hasattr(stills, "fit"))
        self.assertFalse(hasattr(stills, "place"))
        self.assertEqual(stills.STILL_SIZE, (832, 1024))
        self.assertEqual(stills.CHROMA_GREEN, CHROMA)


def _make_project_dir(base):
    """`examples/isometric` copied to an asset-free working copy: the project
    file's relative paths (stills/, anims/, workflows/) get directories to
    hold them, filled by the test (the source still included, so the FR 4
    missing-still rule does not fire before the profile gate). Used by the
    AC15 gate rows, which write their own stub still and GLB bytes."""
    (base / "stills").mkdir(parents=True, exist_ok=True)
    (base / "anims").mkdir(parents=True, exist_ok=True)
    (base / "workflows").mkdir(parents=True, exist_ok=True)
    spec_still().save(base / "stills" / "courier.png")
    # the project file's workflow paths resolve against it; the packaged
    # workflow files stand in for the copy a user keeps in the project
    for name in ("edit.json", "animate.json"):
        shutil.copyfile(ROOT / "src" / "sprite_pipeline" / "workflows" / name,
                        base / "workflows" / name)
    return base


class GoEnv(unittest.TestCase):
    """`go` over a stubbed copy of the example project with faked subprocesses."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.tmp = Path(self._tmp.name)
        self.home = self.tmp / "home"
        self._old_home = os.environ.get("SPRITE_PIPELINE_HOME")
        os.environ["SPRITE_PIPELINE_HOME"] = str(self.home)
        self.addCleanup(self._restore_home)
        self._old_run = subprocess.run
        self.base = self.tmp / "isometric"
        shutil.copytree(EXAMPLE, self.base)
        _make_project_dir(self.base)
        self.glb_bytes = _stub_glb_bytes()
        (self.base / "anims" / "ual.glb").write_bytes(self.glb_bytes)
        self.project = self.base / "project.yaml"
        self.master = self.base / "out" / "courier" / "master"

    def _restore_home(self):
        if self._old_home is None:
            os.environ.pop("SPRITE_PIPELINE_HOME", None)
        else:
            os.environ["SPRITE_PIPELINE_HOME"] = self._old_home

    def tearDown(self):
        subprocess.run = self._old_run

    # profiles for `go`: the registered stub GLB plus the packaged file
    def go_profiles(self):
        return {**metrics.load_profiles(user_path=Path(str(self.home / "nope"))),
                **mannequins.profiles()}

    def run_cli(self, argv, fake=None):
        if fake is not None:
            subprocess.run = fake
        buf, err = io.StringIO(), io.StringIO()
        try:
            with redirect_stdout(buf), redirect_stderr(err):
                code = cli.main(argv, profiles=self.go_profiles())
        except SystemExit as e:
            code = e.code if isinstance(e.code, int) else 1
            buf.write(str(e) + "\n")
        buf.write(err.getvalue())
        return code, buf.getvalue()

    def edit_calls(self, fake):
        return [c for c in fake.calls
                if len(c) > 3 and c[1:3] == ["-m", "sprite_pipeline.comfy"]
                and c[3] == "edit"]


class FakeRun:
    """subprocess.run stand-in: records argv, materialises each step's output
    through a hook (`after` maps an argv substring to a hook)."""

    def __init__(self, after=None, fails=None):
        self.calls = []
        self.after = after or {}
        self.fails = fails or {}

    def __call__(self, argv, **kwargs):
        argv = [str(a) for a in argv]
        self.calls.append(argv)
        joined = " ".join(argv)
        for match, hook in self.after.items():
            if match in joined:
                hook(argv)
        for match, code in self.fails.items():
            if match in joined:
                return _Res(code)
        return _Res(0)


class _Res:
    def __init__(self, returncode=0):
        self.returncode = returncode
        self.stdout = ""
        self.stderr = ""


# ----------------------------------------------------------- AC15-r3: still rule

class TestStandingRuleRejects(GoEnv):
    """AC15-r3: a non-standing still makes `go` exit 2, naming the file and the
    failed rule (`too wide`, `too small`, `feet not at the bottom`)."""

    CASES = {
        "too wide": dict(body_h=200, body_w=500, feet_from_bottom=10),
        "too small": dict(body_h=100, body_w=60, feet_from_bottom=10),
        "feet not at the bottom": dict(body_h=300, body_w=200,
                                       feet_from_bottom=400),
    }

    def run_case(self, name):
        mannequins.pose_frame(500, 900, **self.CASES[name]).save(
            self.base / "stills" / "courier.png")
        fake = FakeRun()
        code, out = self.run_cli(["go", str(self.project)], fake=fake)
        self.assertEqual(code, 2, out)
        self.assertIn(str(self.base / "stills" / "courier.png"), out)
        self.assertIn(name, out)
        return out

    def test_r3_lying_figure_exit_2_names_file_and_too_wide(self):
        out = self.run_case("too wide")
        self.assertNotIn("still preparation: anchor", out)

    def test_r3_small_figure_exit_2_names_file_and_too_small(self):
        self.run_case("too small")

    def test_r3_floating_figure_exit_2_names_file_and_feet_rule(self):
        self.run_case("feet not at the bottom")

    def test_r3_a_still_that_is_not_an_image_exits_2(self):
        """A source that cannot be measured is a project error (exit 2), not a
        traceback."""
        (self.base / "stills" / "courier.png").write_bytes(b"not a png")
        code, out = self.run_cli(["go", str(self.project)], fake=FakeRun())
        self.assertEqual(code, 2, out)
        self.assertIn(str(self.base / "stills" / "courier.png"), out)

    def test_r3_nothing_is_written_for_a_rejected_still(self):
        """The check precedes the output tree: no master, no fingerprint."""
        self.run_case("too small")
        self.assertFalse(self.master.exists())


# ------------------------------------------------------- AC15-r4: profile gate

class TestProfileGate(GoEnv):
    """AC15-r4: a project whose source GLB has no mannequin profile exits 2,
    printing the GLB's sha256 prefix and the `mannequin-profile` command."""

    def test_r4_unprofiled_glb_exit_2_prints_prefix_and_command(self):
        """The load-level gate (FR 4): the message names the GLB, prints the
        sha256 prefix and the command that measures a profile. The unprofiled
        GLB is written before any other GLB byte is seen, so it is the first
        source the gate meets."""
        payload = b"glb bytes no profile knows"
        (self.base / "anims" / "ual.glb").write_bytes(payload)
        digest = hashlib.sha256(payload).hexdigest()
        with self.assertRaises(project.ProjectError) as cm:
            project.load_project(self.project, profiles={})  # no profiles at all
        msg = str(cm.exception)
        self.assertIn("no mannequin profile", msg)
        self.assertIn(digest[:16], msg)
        # the runnable command, naming the GLB to measure
        glb = (self.base / "anims" / "ual.glb").resolve()
        self.assertIn(f"sprite-pipeline mannequin-profile {glb}", msg)

    def test_r4_go_exit_2_names_the_glb_and_the_command(self):
        """`go` exits 2 (the project file is invalid) with the sha256 prefix and
        the `sprite-pipeline mannequin-profile` command in its output."""
        payload = b"glb bytes no profile knows"
        (self.base / "anims" / "ual.glb").write_bytes(payload)
        digest = hashlib.sha256(payload).hexdigest()
        # the still must exist (FR 4's missing-still rule fires before the
        # profile gate and would shadow it): a standing figure, so the run
        # reaches the profile gate and reports the unprofiled GLB
        spec_still().save(self.base / "stills" / "courier.png")
        code, out = self.run_cli(["go", str(self.project)], fake=FakeRun())
        self.assertEqual(code, 2, out)
        self.assertIn(digest[:16], out)
        glb = (self.base / "anims" / "ual.glb").resolve()
        self.assertIn(f"sprite-pipeline mannequin-profile {glb}", out)

    def test_r4_a_profiled_glb_passes_the_gate(self):
        """The same run with the GLB's profile registered gets past the gate."""
        mannequins.register(self.glb_bytes, glb_name="ual.glb")
        spec_still().save(self.base / "stills" / "courier.png")
        fake = FakeRun(after={"sprite_pipeline.comfy edit": self._write_still,
                             "cutout": self._write_keyed})
        code, out = self.run_cli(["go", str(self.project)], fake=fake)
        self.assertEqual(code, 0, out)

    def _write_still(self, argv):
        dst = Path(argv[argv.index("--copy-to") + 1])
        dst.parent.mkdir(parents=True, exist_ok=True)
        spec_still().save(dst)

    def _write_keyed(self, argv):
        out = Path(argv[argv.index("--out") + 1])
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(_stub_glb_bytes())

    def test_r4_dry_run_needs_no_profiled_glb(self):
        """AC1-r1 keeps working: `--dry-run` plans from the project file."""
        code, out = self.run_cli(["go", str(self.project), "--dry-run"])
        self.assertEqual(code, 0, out)
        self.assertIn("still preparation", out)


# ------------------------------------------------- AC15-r5: edit + keyed master

class TestEditConsumesTheAnchoredMaster(GoEnv):
    """AC15-r5: with ComfyUI faked, the edit step receives the anchored master
    as `--image`, and the keyed file is `master/s.keyed.png` (the master's
    direction is `s`)."""

    def setUp(self):
        super().setUp()
        mannequins.register(self.glb_bytes, glb_name="ual.glb")
        spec_still().save(self.base / "stills" / "courier.png")
        self.fake = FakeRun(after={"sprite_pipeline.comfy edit": self._write_still,
                                   "cutout": self._write_keyed})

    def _write_still(self, argv):
        dst = Path(argv[argv.index("--copy-to") + 1])
        dst.parent.mkdir(parents=True, exist_ok=True)
        spec_still().save(dst)

    def _write_keyed(self, argv):
        out = Path(argv[argv.index("--out") + 1])
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(_stub_glb_bytes())

    def test_r5_edit_image_is_the_anchored_master(self):
        code, out = self.run_cli(["go", str(self.project)], fake=self.fake)
        self.assertEqual(code, 0, out)
        edits = self.edit_calls(self.fake)
        self.assertEqual(len(edits), 4, "\n".join(" ".join(c) for c in self.fake.calls))
        s = self.master / "s.png"
        for e in edits:
            self.assertEqual(e[e.index("--image") + 1], str(s))
            self.assertEqual(Image.open(s).size, (832, 1024))
        # the master itself is anchored: flat green, profile target, feet on the
        # floor line, centred
        m = metrics.measure(s)
        self.assertAlmostEqual(m["body_h"], expected_target(704), delta=1.5)
        self.assertAlmostEqual(m["floor_y"], expected_floor(704), delta=1.5)
        self.assertAlmostEqual(m["centre_x"], 416, delta=1.5)
        self.assertEqual(Image.open(s).getpixel((0, 0)), CHROMA)

    def test_r5_keyed_file_is_master_s_keyed_png(self):
        """The keyed master is `master/s.keyed.png` (AC15): the still's own
        direction is `s` (the master), and the cutout of the master keys to
        that name."""
        code, out = self.run_cli(["go", str(self.project)], fake=self.fake)
        self.assertEqual(code, 0, out)
        cutouts = [c for c in self.fake.calls if Path(c[0]).name == "sprite-gen"]
        self.assertEqual(len(cutouts), 1, "\n".join(" ".join(c) for c in self.fake.calls))
        self.assertEqual(cutouts[0][cutouts[0].index("--out") + 1],
                         str(self.master / "s.keyed.png"))
        # the keyed input is the anchored master (the master's own direction, s)
        self.assertEqual(cutouts[0][2], str(self.master / "s.png"))

    def test_r5_no_fit_file_is_prepared_anymore(self):
        """The 832x1024 `s.fit.png` companion of the old fill-fit flow is gone:
        the master itself is the edit input."""
        code, out = self.run_cli(["go", str(self.project)], fake=self.fake)
        self.assertEqual(code, 0, out)
        self.assertFalse((self.master / "s.fit.png").exists())
        self.assertFalse((self.master / "se.fit.png").exists())

    def test_r5_direction_stills_keep_the_edit_canvas(self):
        """A direction still is the edit's own output, landing on the 832x1024
        edit canvas — no 1280 padding is applied to it. The fake edit here
        copies the edit's own input (the anchored master), so each direction
        still is the edit output at the master's size."""
        code, out = self.run_cli(["go", str(self.project)], fake=self._copy_fake())
        self.assertEqual(code, 0, out)
        for d in ("se", "e", "ne", "n"):
            self.assertEqual(Image.open(self.master / f"{d}.png").size, (832, 1024), d)

    def _copy_fake(self):
        """A fake edit that copies its --image to --copy-to (the edit output
        is the anchored master's own bytes)."""
        def _copy(argv):
            src = argv[argv.index("--image") + 1]
            dst = Path(argv[argv.index("--copy-to") + 1])
            dst.parent.mkdir(parents=True, exist_ok=True)
            dst.write_bytes(Path(src).read_bytes())
        def _keyed(argv):
            out = Path(argv[argv.index("--out") + 1])
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_bytes(_stub_glb_bytes())
        return FakeRun(after={"sprite_pipeline.comfy edit": _copy,
                              "cutout": _keyed})


# ---------------------------------------- AC15-r7/r8 (amendment 4): subcommand

class TestMannequinProfileCommand(unittest.TestCase):
    """AC15 amendment 4: `sprite-pipeline mannequin-profile <glb>` with the
    render measurement faked prints the measured JSON and appends the entry to
    the user file (r7), replacing an existing entry for the same hash (r8)."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.tmp = Path(self._tmp.name)
        self.home = self.tmp / "home"
        self.user = self.home / "mannequin_profiles.json"
        self._old_home = os.environ.get("SPRITE_PIPELINE_HOME")
        os.environ["SPRITE_PIPELINE_HOME"] = str(self.home)
        self.addCleanup(self._restore_home)
        self.glb = self.tmp / "new-mannequin.glb"
        self.glb.write_bytes(b"synthetic glb bytes for the profile command")
        self.digest = hashlib.sha256(self.glb.read_bytes()).hexdigest()
        self._old_render = cli.render_mannequin
        # the render measurement, faked: a frame whose mannequin is 300 tall
        # with its feet row at 500
        self.frame = mannequins.tpose_render(body_h=300, floor_y=500)
        cli.render_mannequin = lambda glb: self._write_frame(glb)
        self.addCleanup(self._restore_render)

    def _restore_home(self):
        if self._old_home is None:
            os.environ.pop("SPRITE_PIPELINE_HOME", None)
        else:
            os.environ["SPRITE_PIPELINE_HOME"] = self._old_home

    def _restore_render(self):
        cli.render_mannequin = self._old_render

    def _write_frame(self, glb):
        shot = self.tmp / "render.png"
        self.frame.save(shot)
        return shot

    def run_cli(self, argv):
        buf, err = io.StringIO(), io.StringIO()
        try:
            with redirect_stdout(buf), redirect_stderr(err):
                code = cli.main(argv)
        except SystemExit as e:
            code = e.code if isinstance(e.code, int) else 1
            buf.write(str(e) + "\n")
        buf.write(err.getvalue())
        return code, buf.getvalue()

    def _doc(self):
        return json.loads(self.user.read_text())

    def test_r7_prints_the_measured_json(self):
        code, out = self.run_cli(["mannequin-profile", str(self.glb)])
        self.assertEqual(code, 0, out)
        printed = json.loads(out.strip().splitlines()[-1])
        self.assertEqual(printed, {"glb_name": self.glb.name, "resolution": 704,
                                   "body_h": 300, "floor_y": 500})

    def test_r7_appends_the_entry_to_the_user_file(self):
        """With no SPRITE_PIPELINE_HOME, the entry lands in FR 14's user file
        `~/.sprite-pipeline/mannequin_profiles.json`, keyed by the GLB's
        sha256 (HOME is injected so the real one is never touched)."""
        os.environ.pop("SPRITE_PIPELINE_HOME", None)
        user_home = self.tmp / "user-home"
        old_home = os.environ.get("HOME")
        os.environ["HOME"] = str(user_home)
        self.addCleanup(lambda: os.environ.__setitem__("HOME", old_home)
                        if old_home is not None else os.environ.pop("HOME", None))
        code, out = self.run_cli(["mannequin-profile", str(self.glb)])
        self.assertEqual(code, 0, out)
        user = user_home / ".sprite-pipeline" / "mannequin_profiles.json"
        doc = json.loads(user.read_text())
        self.assertEqual(doc["profiles"][self.digest],
                         {"glb_name": self.glb.name, "resolution": 704,
                          "body_h": 300, "floor_y": 500})

    def test_r8_a_second_run_for_the_same_hash_replaces_no_duplicate(self):
        """The same GLB measured twice leaves exactly one entry for its hash —
        the second measurement — and an unrelated entry survives. The raw
        file is checked too: a JSON object with the hash written twice would
        parse to one key and hide the duplicate."""
        other = {"glb_name": "other.glb", "resolution": 704, "body_h": 111,
                 "floor_y": 222}
        self.user.parent.mkdir(parents=True, exist_ok=True)
        self.user.write_text(json.dumps({"profiles": {"a" * 64: other}}))
        self.frame = mannequins.tpose_render(body_h=300, floor_y=500)
        self.run_cli(["mannequin-profile", str(self.glb)])
        self.frame = mannequins.tpose_render(body_h=310, floor_y=510)
        code, out = self.run_cli(["mannequin-profile", str(self.glb)])
        self.assertEqual(code, 0, out)
        self.assertEqual(self.user.read_text().count(self.digest), 1)
        doc = self._doc()
        self.assertEqual(doc["profiles"],
                         {"a" * 64: other,
                          self.digest: {"glb_name": self.glb.name, "resolution": 704,
                                        "body_h": 310, "floor_y": 510}})

    def test_r7_keeps_other_entries_of_the_user_file(self):
        other = {"glb_name": "other.glb", "resolution": 704, "body_h": 111,
                 "floor_y": 222}
        self.user.parent.mkdir(parents=True, exist_ok=True)
        self.user.write_text(json.dumps({"profiles": {"a" * 64: other}}))
        code, out = self.run_cli(["mannequin-profile", str(self.glb)])
        self.assertEqual(code, 0, out)
        doc = self._doc()
        self.assertEqual(doc["profiles"]["a" * 64], other)
        self.assertEqual(len(doc["profiles"]), 2)

    def test_r7_the_written_profile_reaches_the_default_loader(self):
        """FR 14: what the command wrote is what the merged loader reads from
        its own default user path (the writer and the reader agree on the
        file), alongside the packaged profiles."""
        self.run_cli(["mannequin-profile", str(self.glb)])
        merged = metrics.load_profiles()
        self.assertEqual(merged[self.digest],
                         {"glb_name": self.glb.name, "resolution": 704,
                          "body_h": 300, "floor_y": 500})
        # the packaged ual-female.glb profile survives the merge
        self.assertEqual(
            merged["b97d76c65bd2805e1b04d049fd42cbb9268018f24678858fdafa6891b3f2ddf4"]["body_h"],
            324)

    def test_r7_an_invalid_user_file_is_replaced_not_merged(self):
        self.user.parent.mkdir(parents=True, exist_ok=True)
        self.user.write_text("{not json")
        code, out = self.run_cli(["mannequin-profile", str(self.glb)])
        self.assertEqual(code, 0, out)
        self.assertEqual(list(self._doc()["profiles"]), [self.digest])

    def test_r7_a_missing_glb_exits_2(self):
        code, out = self.run_cli(["mannequin-profile", str(self.tmp / "nope.glb")])
        self.assertEqual(code, 2, out)
        self.assertIn(str(self.tmp / "nope.glb"), out)

    def test_r7_measures_the_frame_the_drive_render_writes(self):
        """The real render path with Blender faked at subprocess.run: the
        A_TPose render at 704 writes its frames where the drive plan points,
        and the command measures the first one."""
        cli.render_mannequin = self._old_render
        old_run = subprocess.run
        self.addCleanup(setattr, subprocess, "run", old_run)
        calls = []

        def fake_run(argv, **kw):
            argv = [str(a) for a in argv]
            calls.append((argv, kw.get("env") or {}))
            if argv[0] == "blender":
                rest = argv[argv.index("--") + 1:]
                frames = Path(rest[2])
                frames.mkdir(parents=True, exist_ok=True)
                mannequins.tpose_render(body_h=300, floor_y=500).save(frames / "f0001.png")
                mannequins.tpose_render(body_h=280, floor_y=480).save(frames / "f0002.png")
            return _Res(0)

        subprocess.run = fake_run
        code, out = self.run_cli(["mannequin-profile", str(self.glb)])
        self.assertEqual(code, 0, out)
        blender = [(a, e) for a, e in calls if a[0] == "blender"]
        self.assertEqual(len(blender), 1)
        argv, env = blender[0]
        rest = argv[argv.index("--") + 1:]
        self.assertEqual(rest[:2], [str(self.glb), "A_TPose"])
        self.assertEqual(rest[3], "0")  # facing s
        self.assertEqual((env.get("RW"), env.get("RH")), ("704", "704"))
        self.assertEqual(json.loads(out.strip().splitlines()[-1]),
                         {"glb_name": self.glb.name, "resolution": 704,
                          "body_h": 300, "floor_y": 500})

    def test_render_mannequin_fails_loudly_when_no_frame_lands(self):
        """The real render path (Blender faked at subprocess.run to write
        nothing) reports a failed render instead of measuring nothing."""
        cli.render_mannequin = self._old_render
        old_run = subprocess.run
        self.addCleanup(setattr, subprocess, "run", old_run)
        subprocess.run = lambda argv, **kw: _Res(0)
        with self.assertRaises(SystemExit):
            cli.render_mannequin(self.glb)


if __name__ == "__main__":
    unittest.main()
