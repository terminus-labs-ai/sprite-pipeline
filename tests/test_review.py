"""Tests for the direction stills, the review stop, `approve` and --reroll
(spec AC4, AC5; FR 6-8).

With ComfyUI faked at subprocess.run: `go` on an unapproved character prepares
the still, makes one edit per generated direction other than `se` with the
view's turn prompt in the character's pronoun, writes refs-sheet.png, makes no
animate call and exits 0 printing the approve command (AC4). `approve` writes
refs.approved holding the hash of the direction stills; `go` then continues to
the animation stages, while a changed still, a --reroll or a missing approval
stops it again; --no-stop continues without approval (AC5).

No network, GPU, Blender or real ComfyUI: subprocesses are faked at
subprocess.run, the config home is isolated per test. The expected prompts are
computed here from views/isometric.yaml plus independently written pronoun
tables — not from production code.
"""
import hashlib
import io
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from importlib import resources
from pathlib import Path

import yaml
from PIL import Image

from sprite_pipeline import cli, config, project, stages, stills

ROOT = Path(__file__).resolve().parent.parent
EXAMPLE = ROOT / "examples" / "isometric"

# 2x2 RGBA PNG, every pixel (2, 3, 5, 255): a valid stub still byte source
TINY_PNG = bytes.fromhex(
    "89504e470d0a1a0a0000000d4948445200000002000000020806000000"
    "72b60d080000001d4944415408d763f8ffff3f01078895a30000000000"
    "000000db12c6c60000000049454e44ae426082")
OTHER_PNG = bytes.fromhex(  # every pixel (9, 9, 9, 255): a *changed* still
    "89504e470d0a1a0a0000000d4948445200000002000000020806000000"
    "72b60d080000001d4944415408d763e2dfdfdf030118ece30000000000"
    "0000e104e5dc2b0000000049454e44ae426082")

GEN_DIRS = ["s", "se", "e", "ne", "n"]
EDIT_DIRS = ["s", "e", "ne", "n"]  # the generated directions that are not se

# the review prompt's framing, copied from the new-kit pattern (FR 7): the
# AC row pins the whole string, so it is stated here independently of stills.py
STYLE = "hand-painted cel shading, bold dark outlines, crisp edges"
TAIL = " Full body, plain flat pure green background, no shadow."
OBJ = {"he": "him", "she": "her", "they": "them"}
POS = {"he": "his", "she": "her", "they": "their"}


def expected_prompt(turn, d, pronoun):
    """The AC4-r1/r2 pinned prompt: the framing plus the view's turn text with
    the pronoun substituted (the real turn texts hold {pos} / {obj} fields)."""
    obj = OBJ[pronoun]
    pos = POS[pronoun]
    return ("The exact same character with the exact same outfit, colors, proportions "
            f"and art style, seen from the exact same high isometric camera angle "
            f"looking down at {obj}, standing in the same relaxed pose, but now "
            + turn[d].format(obj=obj, pos=pos) + "." + TAIL)


class _Result:
    def __init__(self, returncode=0):
        self.returncode = returncode
        self.stdout = ""
        self.stderr = ""


class FakeRun:
    """subprocess.run stand-in that records every argv. `fails` maps a substring
    of the joined argv to a nonzero exit code; `after` maps a substring to a
    hook(argv) run after recording, to materialise a step's output."""

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

    def edit_calls(self):
        return [c for c in self.calls
                if len(c) > 2 and c[1:3] == ["-m", "sprite_pipeline.comfy"]
                and "edit" in c and "animate" not in c]

    def animate_calls(self):
        return [c for c in self.calls
                if len(c) > 2 and c[1:3] == ["-m", "sprite_pipeline.comfy"]
                and "animate" in c]


class Env(unittest.TestCase):
    """Base: isolated config home; run_cli() invokes the real entry point."""

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


def make_stub_project(tmp, pronoun="she"):
    """A stubbed working copy of examples/isometric: real project.yaml, stub
    GLB bytes and a real (valid PNG) source still — the run prepares it for
    real. Returns the project.yaml path."""
    base = Path(tmp) / "isometric"
    shutil.copytree(EXAMPLE, base)
    (base / "anims").mkdir(parents=True, exist_ok=True)
    (base / "anims" / "ual.glb").write_bytes(TINY_PNG)
    (base / "stills").mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (64, 80), (10, 200, 10)).save(base / "stills" / "courier.png")
    if pronoun != "she":
        pf = base / "project.yaml"
        pf.write_text(pf.read_text().replace("pronoun: she", f"pronoun: {pronoun}"))
    return base / "project.yaml"


class ReviewFixture(Env):
    """The example project with the edit calls faked: a faked edit writes the
    requested --copy-to destination, so a run produces the direction stills."""

    def setUp(self, pronoun="she"):
        super().setUp()
        self.pronoun = pronoun
        self.project = make_stub_project(self.tmp, pronoun)
        self.base = self.project.parent
        self.out_char = self.base / "out" / "courier"
        self.master = self.out_char / "master"
        self.turn = (yaml.safe_load(
            (ROOT / "src" / "sprite_pipeline" / "views" / "isometric.yaml")
            .read_text()) or {})["turn"]
        self.fake = self._fresh_fake()

    def _fresh_fake(self):
        return FakeRun(after={"sprite_pipeline.comfy edit": self._make_still,
                              "cutout": self._produce_still,
                              "render_drive.py": self._produce_frames,
                              "ffmpeg": self._produce_drive})

    def _produce_frames(self, argv):
        # the fake blender writes its frames; the fake ffmpeg writes its output
        frames = Path(argv[argv.index("--") + 3])
        frames.mkdir(parents=True, exist_ok=True)
        (frames / "f0000.png").write_bytes(TINY_PNG)

    def _produce_drive(self, argv):
        dst = Path(argv[-1])
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_bytes(b"drive")

    def _produce_still(self, argv):
        """The faked sprite-gen cutout writes its --out-dir output."""
        out = Path(argv[argv.index("--out-dir") + 1])
        out.mkdir(parents=True, exist_ok=True)
        (out / "se.keyed.png").write_bytes(TINY_PNG)

    def _make_still(self, argv):
        # the faked edit writes a real PNG to its --copy-to destination
        dst = Path(argv[argv.index("--copy-to") + 1])
        dst.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (32, 32), (10, 200, 10)).save(dst)

    def seed_stills(self):
        # real PNG bytes: a sheet over them must build (the sheet is part of AC4/AC5)
        im = Image.new("RGB", (32, 32), (10, 200, 10))
        self.master.mkdir(parents=True, exist_ok=True)
        for d in GEN_DIRS:
            im.save(self.master / f"{d}.png")
        (self.master / "se.keyed.png").write_bytes(TINY_PNG)

    def approve_cmd(self):
        return ["approve", str(self.project), "courier"]

    def stopped_lines(self, out):
        return [l for l in out.splitlines() if "stopped for review" in l]

    def assert_stopped_with_approve_command(self, code, out):
        self.assertEqual(code, 0, out)
        lines = self.stopped_lines(out)
        self.assertEqual(len(lines), 1, out)
        self.assertTrue(lines[-1].endswith(
            f"sprite-pipeline approve {self.project} courier"), lines[-1])


# ---------------------------------------------------------------- AC4-r1: first run

class TestFirstRunStopsForReview(ReviewFixture):
    """AC4-r1: fake ComfyUI, example project (pronoun she), no approval: 4 edit
    calls (s, e, ne, n) with the pinned turn prompts, refs-sheet.png written,
    0 animate calls, exit 0 printing the approve command."""

    def test_r1_four_edit_calls_pinned_prompts_exit_zero_with_approve_command(self):
        code, out = self.run_cli(["go", str(self.project)], fake=self.fake)
        self.assertEqual(code, 0, out)  # the review stop exits 0
        edits = self.fake.edit_calls()
        self.assertEqual(len(edits), 4, "\n".join(" ".join(c) for c in self.fake.calls))
        got = {e[e.index("--seed") + 1] for e in edits}
        for d in EDIT_DIRS:
            matches = [e for e in edits if e[e.index("--copy-to") + 1]
                       == str(self.master / f"{d}.png")]
            self.assertEqual(len(matches), 1, d)
            e = matches[0]
            # the pinned FR 7 prompt: the view's turn text in her/her
            self.assertEqual(e[e.index("--prompt") + 1],
                             expected_prompt(self.turn, d, "she"))
            self.assertEqual(e[e.index("--image") + 1], str(self.master / "se.png"))
        self.assertEqual(self.fake.animate_calls(), [])
        # prepared SE still (1280x1280, AC3), palette key, direction stills on disk
        se = self.master / "se.png"
        self.assertTrue(se.is_file())
        self.assertEqual(Image.open(se).size, (1280, 1280))
        cut = [c for c in self.fake.calls if Path(c[0]).name == "sprite-gen"]
        self.assertEqual(len(cut), 1, "\n".join(" ".join(c) for c in self.fake.calls))
        self.assertEqual(cut[0][1], "cutout")
        self.assertEqual(cut[0][cut[0].index("--source") + 1], str(se))
        self.assertEqual(cut[0][cut[0].index("--key") + 1], "measured")
        for d in EDIT_DIRS:
            self.assertTrue((self.master / f"{d}.png").is_file(), d)
        sheet = self.out_char / "refs-sheet.png"
        self.assertTrue(sheet.is_file())
        self.assertEqual(Image.open(sheet).size[1], 340 + 24)  # the labelled sheet
        # the run ends with the approve command and exit 0
        self.assert_stopped_with_approve_command(code, out)
        self.assertIn(str(sheet), out)

    def test_r1_no_se_edit_call(self):
        code, out = self.run_cli(["go", str(self.project)], fake=self.fake)
        self.assertEqual(code, 0, out)
        for e in self.fake.edit_calls():
            self.assertNotEqual(e[e.index("--copy-to") + 1], str(self.master / "se.png"))

    def test_r1_default_seed_is_the_project_seed(self):
        code, out = self.run_cli(["go", str(self.project)], fake=self.fake)
        self.assertEqual(code, 0, out)
        self.assertEqual({e[e.index("--seed") + 1] for e in self.fake.edit_calls()},
                         {"7"})  # examples project: seed: 7

    def test_r1_edit_targets_the_edit_workflow(self):
        code, out = self.run_cli(["go", str(self.project)], fake=self.fake)
        self.assertEqual(code, 0, out)
        for e in self.fake.edit_calls():
            self.assertEqual(e[:3], [sys.executable, "-m", "sprite_pipeline.comfy"])
            self.assertEqual(e[e.index("--workflow") + 1],
                             str(resources.files("sprite_pipeline") / "workflows" / "edit.json"))


# ----------------------------------------------------------------- AC4-r2: they

class TestPronounThey(ReviewFixture):
    """AC4-r2: pronoun they — the edit prompts use them/their."""

    def setUp(self):
        super().setUp(pronoun="they")

    def test_r2_prompts_use_them_their(self):
        code, out = self.run_cli(["go", str(self.project)], fake=self.fake)
        self.assertEqual(code, 0, out)
        edits = self.fake.edit_calls()
        self.assertEqual(len(edits), 4)
        for e in edits:
            d = Path(e[e.index("--copy-to") + 1]).stem
            prompt = e[e.index("--prompt") + 1]
            self.assertEqual(prompt, expected_prompt(self.turn, d, "they"))
            self.assertIn("looking down at them,", prompt)
            if "{pos}" in self.turn[d]:
                self.assertIn("their", prompt)
            self.assertNotIn("{pos}", prompt)


# ----------------------------------------------------- AC4-r3: rerun, stills exist

class TestRerunSkipsExistingStills(ReviewFixture):
    """AC4-r3: rerun with all direction stills present but unapproved: the
    edits skip (0 edit calls) and the run stops again."""

    def test_r3_zero_edit_calls_stops_again(self):
        self.seed_stills()
        fake = self._fresh_fake()
        code, out = self.run_cli(["go", str(self.project)], fake=fake)
        self.assertEqual(code, 0, out)
        self.assertEqual(fake.edit_calls(), [], "\n".join(" ".join(c) for c in fake.calls))
        self.assertEqual(fake.calls, [])  # every still step skip-if-exists
        self.assertTrue((self.out_char / "refs-sheet.png").is_file())
        self.assert_stopped_with_approve_command(code, out)

    def test_r3_force_redoes_the_edits(self):
        # the stop alone never re-runs existing still work; --force overrides
        self.seed_stills()
        fake = self._fresh_fake()
        code, out = self.run_cli(["go", str(self.project), "--force"], fake=fake)
        self.assertEqual(code, 0, out)  # stopped again before the animation stages
        self.assertEqual(len(fake.edit_calls()), 4, "\n".join(" ".join(c) for c in fake.calls))
        self.assert_stopped_with_approve_command(code, out)


# -------------------------------------------------------------------- AC5-r1: approve

class TestApproveThenContinue(ReviewFixture):
    """AC5-r1: approve writes refs.approved holding sha256 over the 5 stills'
    bytes in generate order; the next go continues to the animation stages."""

    def expected_hash(self):
        h = hashlib.sha256()
        for d in GEN_DIRS:
            h.update((self.master / f"{d}.png").read_bytes())
            h.update(b"\0")
        return h.hexdigest()

    def test_r1_approve_writes_the_stills_hash(self):
        self.seed_stills()
        code, out = self.run_cli(self.approve_cmd())
        self.assertEqual(code, 0, out)
        f = self.out_char / "refs.approved"
        self.assertTrue(f.is_file(), out)
        self.assertEqual(f.read_text().strip(), self.expected_hash())

    def test_r1_next_go_continues_to_the_animation_stages(self):
        self.seed_stills()
        code, _ = self.run_cli(self.approve_cmd())
        self.assertEqual(code, 0)
        fake = FakeRun()
        code, out = self.run_cli(["go", str(self.project), "--dry-run"], fake=fake)
        self.assertEqual(code, 0, out)
        for stage in ("drive", "animate", "key", "loop", "pixel", "export"):
            self.assertIn(f"stage={stage}", out, stage)
        self.assertEqual(self.stopped_lines(out), [], out)
        self.assertIn("[courier] still preparation: skip", out)

    def test_r1_an_existing_run_made_of_go_approve_go(self):
        # the user story: go stops, approve records, go continues
        code, out = self.run_cli(["go", str(self.project)], fake=self.fake)
        self.assert_stopped_with_approve_command(code, out)
        code, out = self.run_cli(self.approve_cmd())
        self.assertEqual(code, 0, out)
        code, out = self.run_cli(["go", str(self.project), "--dry-run"])
        self.assertEqual(code, 0, out)
        self.assertIn("stage=animate", out)
        self.assertEqual(self.stopped_lines(out), [], out)


# ------------------------------------------------------ AC5-r2: a still changed

class TestChangedStillStopsAgain(ReviewFixture):
    """AC5-r2: approved, then one direction still's bytes change: the hash no
    longer matches and go stops at the sheet again."""

    def test_r2_changed_still_bytes_stop_the_run(self):
        self.seed_stills()
        code, _ = self.run_cli(self.approve_cmd())
        self.assertEqual(code, 0)
        (self.master / "n.png").write_bytes(OTHER_PNG)  # one still's bytes changed
        fake = FakeRun()
        code, out = self.run_cli(["go", str(self.project), "--dry-run"], fake=fake)
        self.assertEqual(code, 0, out)
        self.assertIn("unapproved", out)
        self.assertEqual(self.stopped_lines(out),
                         [f"[courier] stopped for review: sprite-pipeline approve "
                          f"{self.project} courier"], out)
        for stage in ("drive", "animate", "key", "loop"):
            self.assertNotIn(f"stage={stage}", out, stage)


# ----------------------------------------------------------------- AC5-r3: --reroll

class TestReroll(ReviewFixture):
    """AC5-r3: approved, then --reroll courier:n=2: one edit call for n with
    seed 2 and the run stops again."""

    def test_r3_reroll_one_edit_seed_2_stops_again(self):
        self.seed_stills()
        code, _ = self.run_cli(self.approve_cmd())
        self.assertEqual(code, 0)
        code, out = self.run_cli(["go", str(self.project), "--reroll", "courier:n=2"],
                                 fake=self.fake)
        self.assertEqual(code, 0, out)
        edits = self.fake.edit_calls()
        self.assertEqual(len(edits), 1, "\n".join(" ".join(c) for c in self.fake.calls))
        self.assertEqual(edits[0][edits[0].index("--copy-to") + 1],
                         str(self.master / "n.png"))
        self.assertEqual(edits[0][edits[0].index("--seed") + 1], "2")
        # stops again: the sheet path and the approve command end the run
        self.assertTrue((self.out_char / "refs-sheet.png").is_file())
        self.assertIn("reroll", out)
        self.assert_stopped_with_approve_command(code, out)

    def test_r5_unknown_reroll_direction_exits_2(self):
        code, out = self.run_cli(["go", str(self.project), "--reroll", "courier:x=2"],
                                 fake=self.fake)
        self.assertEqual(code, 2, out)
        self.assertIn("--reroll", out)
        self.assertIn("x", out)
        self.assertEqual(self.fake.calls, [])


# ------------------------------------------------------------------ AC5-r4: --no-stop

class TestNoStop(ReviewFixture):
    """AC5-r4: unapproved with --no-stop: the run continues to the animation
    stages and writes no refs.approved."""

    def test_r4_no_stop_continues_without_approval(self):
        fake = FakeRun()
        code, out = self.run_cli(["go", str(self.project), "--dry-run", "--no-stop"],
                                 fake=fake)
        self.assertEqual(code, 0, out)
        for stage in ("drive", "animate", "key", "loop", "pixel", "export"):
            self.assertIn(f"stage={stage}", out, stage)
        self.assertEqual(self.stopped_lines(out), [], out)
        self.assertFalse((self.out_char / "refs.approved").exists())
        self.assertNotIn("sprite-pipeline approve", out)


# ------------------------------------------------------------ AC5-r5: bad approve

class TestApproveErrors(ReviewFixture):
    """AC5-r5: approve before the stills exist: exit 2 naming the missing
    stills (the unknown --reroll direction case lives in TestReroll)."""

    def test_r5_approve_before_the_stills_exist_exits_2_naming_them(self):
        code, out = self.run_cli(self.approve_cmd())
        self.assertEqual(code, 2, out)
        self.assertIn("missing", out)
        for d in GEN_DIRS:
            self.assertIn(str(self.master / f"{d}.png"), out)
        self.assertFalse((self.out_char / "refs.approved").exists())

    def test_r5_approve_with_a_partial_set_names_the_missing_ones(self):
        self.seed_stills()
        (self.master / "ne.png").unlink()
        code, out = self.run_cli(self.approve_cmd())
        self.assertEqual(code, 2, out)
        self.assertIn(str(self.master / "ne.png"), out)
        self.assertNotIn(str(self.master / "s.png"), out)
        self.assertFalse((self.out_char / "refs.approved").exists())


# ------------------------------------------------- review-stop rules of the unit

class TestApprovalGate(unittest.TestCase):
    """The StageCtx approval predicate itself: is_approved is true exactly
    when refs.approved holds the hash of the current stills; absence,
    staleness or a changed still fails. (In-process only: no subprocess is
    spawned or intercepted here.)"""

    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmpdir.cleanup)
        self.ctx = stages.StageCtx(project.load_project(
            make_stub_project(self._tmpdir.name)), config.defaults(Path(self._tmpdir.name) / "home"))

    def test_unapproved_until_the_file_holds_the_matching_hash(self):
        master = self.ctx.master("courier")
        master.mkdir(parents=True)
        for d in GEN_DIRS:
            (master / f"{d}.png").write_bytes(TINY_PNG)
        self.assertFalse(self.ctx.is_approved("courier"))  # no refs.approved yet
        self.ctx.write_refs_approved("courier")
        self.assertTrue(self.ctx.is_approved("courier"))
        (master / "e.png").write_bytes(OTHER_PNG)  # change one still
        self.assertFalse(self.ctx.is_approved("courier"))

    def test_hash_order_and_content_are_the_generate_order(self):
        master = self.ctx.master("courier")
        master.mkdir(parents=True)
        blobs = {d: b"still " + d.encode() for d in GEN_DIRS}
        for d in GEN_DIRS:
            (master / f"{d}.png").write_bytes(blobs[d])
        want = hashlib.sha256()
        for d in GEN_DIRS:  # the project's generate order, each still once
            want.update(blobs[d])
            want.update(b"\0")
        self.assertEqual(self.ctx.refs_hash("courier"), want.hexdigest())

    def test_missing_still_hash_is_none(self):
        master = self.ctx.master("courier")
        master.mkdir(parents=True)
        for d in GEN_DIRS:
            (master / f"{d}.png").write_bytes(TINY_PNG)
        (master / "s.png").unlink()
        self.assertIsNone(self.ctx.refs_hash("courier"))
        self.assertEqual(self.ctx.missing_stills("courier"), ["s"])


class TestDirPromptAndSheet(unittest.TestCase):
    """stills.dir_prompt per pronoun (bad pronoun is an error, never a wrong
    prompt) and stills.review_sheet (labelled row, width is the sum)."""

    def test_pronoun_tables_he_she_they(self):
        # the possessive reaches the prompt through the turn text, verbatim:
        # the whole pinned string per pronoun (object in the camera clause)
        turn = {"n": "so we see only {pos} back"}
        for pronoun, (obj, pos) in (("he", ("him", "his")),
                                    ("she", ("her", "her")),
                                    ("they", ("them", "their"))):
            want = ("The exact same character with the exact same outfit, colors, "
                    "proportions and art style, seen from the exact same high "
                    f"isometric camera angle looking down at {obj}, standing in the "
                    f"same relaxed pose, but now so we see only {pos} back." + TAIL)
            self.assertEqual(stills.dir_prompt(turn, "n", pronoun), want)

    def test_bad_pronoun_raises(self):
        with self.assertRaises(ValueError):
            stills.dir_prompt({"n": "x"}, "n", "it")

    def test_direction_without_turn_prompt_raises(self):
        with self.assertRaises(ValueError):
            stills.dir_prompt({"s": "x"}, "n", "she")

    def test_review_sheet_layout(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            paths = []
            for i, d in enumerate(("s", "se", "e", "ne", "n")):
                p = tmp / f"{d}.png"
                Image.new("RGB", (60 + 20 * i, 100), (i * 40, 0, 0)).save(p)
                paths.append(p)
            dst = tmp / "refs-sheet.png"
            stills.review_sheet(paths, dst, height=100)
            im = Image.open(dst)
            # height 100 over sources of height 100: widths unchanged, stacked
            self.assertEqual(im.size, (60 + 80 + 100 + 120 + 140, 124))
            self.assertEqual(im.getpixel((5, 50)), (0, 0, 0))  # first tile's colour
            self.assertEqual(im.getpixel((140, 50)), (80, 0, 0))  # second tile
            self.assertEqual(im.getpixel((im.width - 5, 120)), (40, 42, 52))  # below tiles


class TestParseRerolls(unittest.TestCase):
    """--reroll parsing: <char>:<dir>=<seed>, bare <dir>=<seed> spread over the
    run's characters, and malformed entries raising (exit 2 upstream)."""

    def test_parses_char_dir_seed(self):
        self.assertEqual(cli.parse_rerolls("courier:n=2"), {"courier": {"n": 2}})

    def test_bare_dir_applies_to_the_selected_characters(self):
        parsed = cli.parse_rerolls("n=2")
        self.assertEqual(parsed, {"": {"n": 2}})
        self.assertEqual(cli.rerolls_for(parsed, ["courier", "guard"]),
                         {"courier": {"n": 2}, "guard": {"n": 2}})

    def test_malformed_entries_raise(self):
        for bad in ("n", "courier:n", "courier:n=two", "=2"):
            with self.assertRaises(ValueError):
                cli.parse_rerolls(bad)


if __name__ == "__main__":
    unittest.main()
