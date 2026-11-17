"""Tests for the review stop's advisory size report (spec AC16; FR 8 extended
by amendments 3 and 9, FR 6 measurement).

With ComfyUI (and Blender, for the FR 7 guide renders) faked at
subprocess.run, the faked edit writes synthetic raw edits
(`master/<d>.edit.png`) on the 832x1024 edit canvas whose figures measure a
chosen ratio of their own direction's target (amendment 10: the stub
mannequin's `dirs[d].body_h` at frame_size 704 x 832 / 704 — se 433.7, e
471.5, ne 418.4, n 361.6; the `s` master against 391.2). At the stop, `go`
logs one `[courier/<dir>] ref size <ratio>x` line per direction measured on
the raw edit (or on `<d>.png` when no raw edit exists), the refs sheet labels
append the two-decimal ratio and `CHECK` below 0.85 or above 1.15, and
`approve` succeeds whatever the ratios — the report never blocks. A
direction still (`<d>.png`) that changes after approval still invalidates the
approval exactly as before; a changed raw edit alone does not.

The expected ratios, labels and log lines are stated literally here; the
target only sizes the synthetic inputs.
"""
import hashlib
import re
import shutil
from pathlib import Path

from PIL import Image, ImageDraw

import mannequins
from test_review import GEN_DIRS, ReviewFixture

from sprite_pipeline import config, project, stages, stills

# the anchor target on the 832x1024 canvas for the stub profile at 704
# (FR 8, amendments 9 and 10): 331 * 832 / 704 = 391.2 px (the spec's "391
# px target") — stills.body_target(704, profile, "s")
TARGET = mannequins.BODY_H * 832 / 704
# amendment 10: each direction's own target, from the stub profile's dirs
DIR_TARGET = {d: v["body_h"] * 832 / 704 for d, v in mannequins.DIRS.items()}
# AC16's synthetic direction stills; ne sits below the band to cover the low
# side of the CHECK rule
AC16_RATIOS = {"se": 0.97, "e": 1.06, "ne": 0.84, "n": 1.30}
SIZE_LINE = re.compile(r"^\[courier/(\w+)\] ref size (.*)$")
# the sheet layout the labels sit in (FR 8, unchanged): tiles 340 px high,
# an 832x1024 still is 276 px wide there, the label strip is 24 px below
TILE_W = round(832 * 340 / 1024)
STRIP_H = 24


def direction_still(path, ratio, marker=None, d="s"):
    """A synthetic direction still on the 832x1024 edit canvas: one standing
    figure `ratio` x direction `d`'s target px tall (feet 10 px above the
    bottom edge)."""
    im = mannequins.pose_frame(832, 1024, body_h=round(ratio * DIR_TARGET[d]),
                               body_w=200, feet_from_bottom=10)
    if marker is not None:
        # inside the figure: a 1.00 figure spans rows 623-1013, x 316-515
        im.paste(marker, (400, 800, 420, 820))
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    im.save(path)


def label_strip(text):
    """The expected label strip of one sheet tile reading `text`."""
    strip = Image.new("RGB", (TILE_W, STRIP_H), (40, 42, 52))
    ImageDraw.Draw(strip).text((8, 5), text, fill=(230, 230, 230))
    return strip


class SizeFixture(ReviewFixture):
    """The review fixture whose faked edit writes each direction's still at
    `self.ratios[dir]` of the target."""

    def setUp(self):
        super().setUp()
        self.ratios = dict(AC16_RATIOS)

    def _make_still(self, argv):
        # the raw edit `master/<d>.edit.png` at its direction's ratio
        dst = Path(argv[argv.index("--copy-to") + 1])
        d = dst.name.split(".")[0]
        direction_still(dst, self.ratios[d], d=d)

    def size_lines(self, out):
        return [m.groups() for m in map(SIZE_LINE.match, out.splitlines()) if m]

    def stills_hash(self):
        h = hashlib.sha256()
        for d in GEN_DIRS:
            h.update((self.master / f"{d}.png").read_bytes())
            h.update(b"\0")
        return h.hexdigest()


# ------------------------------------------------- AC16-r1: one log line per dir

class TestSizeLogLines(SizeFixture):
    """AC16-r1: one `ref size` line per direction with two-decimal ratios."""

    def test_r1_one_ref_size_line_per_direction_two_decimals(self):
        code, out = self.run_cli(["go", str(self.project)], fake=self.fake)
        self.assertEqual(code, 0, out)
        lines = self.size_lines(out)
        # exactly one line per generated direction, nothing twice
        self.assertEqual(sorted(d for d, _ in lines), sorted(GEN_DIRS), out)
        self.assertEqual(dict(lines), {
            "s": "1.00x",          # the anchored master itself
            "se": "0.97x",
            "e": "1.06x",
            "ne": "0.84x CHECK",   # below 0.85
            "n": "1.30x CHECK",    # above 1.15
        }, out)
        # the report sits at the stop: it precedes the stop verdict
        self.assert_stopped_with_approve_command(code, out)

    def test_r1_ratio_target_is_391_at_704(self):
        """FR 8 (amendments 9 and 10): the ratio's denominator is FR 6's
        dirs.s.body_h x 832 / frame_size — 391.2 at 704 for the stub
        mannequin."""
        self.assertAlmostEqual(stills.body_target(704, mannequins.profile(), "s"),
                               331 * 832 / 704)

    def test_r1_band_edges_are_inside(self):
        # 0.85 and 1.15 are inside the band; just past either prints CHECK
        self.assertEqual(stages.size_note(0.85), "0.85")
        self.assertEqual(stages.size_note(1.15), "1.15")
        self.assertEqual(stages.size_note(0.84), "0.84 CHECK")
        self.assertEqual(stages.size_note(1.16), "1.16 CHECK")
        self.assertEqual(stages.size_note(1.0), "1.00")


# ----------------------------------------------------- AC16-r2: the sheet labels

class TestSheetLabels(SizeFixture):
    """AC16-r2: the refs sheet labels read `se 0.97`, `e 1.06`, `n 1.30
    CHECK` (and the master's and ne's ratios likewise)."""

    def test_r2_labels_append_the_ratio_and_check(self):
        code, out = self.run_cli(["go", str(self.project)], fake=self.fake)
        self.assertEqual(code, 0, out)
        sheet = Image.open(self.out_char / "refs-sheet.png").convert("RGB")
        self.assertEqual(sheet.size, (TILE_W * len(GEN_DIRS), 340 + STRIP_H))
        want = {"s": "s 1.00", "se": "se 0.97", "e": "e 1.06",
                "ne": "ne 0.84 CHECK", "n": "n 1.30 CHECK"}
        # the strip comparison discriminates the CHECK flag and the ratio
        self.assertNotEqual(label_strip("n 1.30").tobytes(),
                            label_strip("n 1.30 CHECK").tobytes())
        self.assertNotEqual(label_strip("se 0.97").tobytes(),
                            label_strip("se").tobytes())
        for i, d in enumerate(GEN_DIRS):  # tiles in generate order
            got = sheet.crop((i * TILE_W, 340, (i + 1) * TILE_W, 340 + STRIP_H))
            self.assertEqual(got.tobytes(), label_strip(want[d]).tobytes(),
                             f"{d}: label is not {want[d]!r}")


# ------------------------------------------------ AC16-r3: approve never blocked

class TestApproveIsWarnOnly(SizeFixture):
    """AC16-r3: `approve` after the stop succeeds in every ratio case, CHECK
    or not, and the next `go` continues past the stop."""

    def test_r3_approve_succeeds_in_every_ratio_case(self):
        for ratio in (0.97, 1.06, 1.30, 0.70):
            with self.subTest(ratio=ratio):
                shutil.rmtree(self.base / "out", ignore_errors=True)
                self.ratios = {d: ratio for d in AC16_RATIOS}
                code, out = self.run_cli(["go", str(self.project)],
                                         fake=self._fresh_fake())
                self.assert_stopped_with_approve_command(code, out)
                flagged = [d for d, s in self.size_lines(out) if s.endswith("CHECK")]
                self.assertEqual(sorted(flagged),
                                 sorted(AC16_RATIOS) if ratio in (1.30, 0.70) else [],
                                 out)
                code, out = self.run_cli(self.approve_cmd())
                self.assertEqual(code, 0, out)
                self.assertEqual((self.out_char / "refs.approved").read_text().strip(),
                                 self.stills_hash())
                code, out = self.run_cli(["go", str(self.project), "--dry-run"])
                self.assertEqual(code, 0, out)
                self.assertEqual(self.stopped_lines(out), [], out)
                self.assertIn("stage=animate", out)


# --------------------------------------- AC16-r4: a changed still re-stops as before

class TestChangedStillInvalidates(SizeFixture):
    """AC16-r4: approved, then one direction still changes (here to an
    in-band ratio): the approval no longer matches and `go` stops again,
    reporting the new size."""

    def test_r4_changed_direction_still_invalidates_the_approval(self):
        code, out = self.run_cli(["go", str(self.project)], fake=self.fake)
        self.assert_stopped_with_approve_command(code, out)
        code, out = self.run_cli(self.approve_cmd())
        self.assertEqual(code, 0, out)
        approved = (self.out_char / "refs.approved").read_text()
        direction_still(self.master / "n.png", 1.00, marker=(200, 20, 20))
        code, out = self.run_cli(["go", str(self.project), "--dry-run"])
        self.assertEqual(code, 0, out)
        self.assertIn("unapproved", out)
        self.assertEqual(len(self.stopped_lines(out)), 1, out)
        self.assertNotIn("stage=animate", out)
        # the real run stops at the sheet again, reporting the size of n's
        # raw edit (amendment 10: n.edit.png is still there, so it is what is
        # measured), and leaves the old approval file as it was
        code, out = self.run_cli(["go", str(self.project)], fake=self._fresh_fake())
        self.assert_stopped_with_approve_command(code, out)
        self.assertIn(("n", "1.30x CHECK"), self.size_lines(out), out)
        self.assertEqual((self.out_char / "refs.approved").read_text(), approved)
        self.assertNotEqual(approved.strip(), self.stills_hash())


# ------------------------- AC16-r2 / r3 / r5 (amendment 10): what is measured

class TestMeasuredFile(SizeFixture):
    """AC16-r2: with `<d>.edit.png` absent the ratio is measured on
    `<d>.png`. AC16-r3: the `s` master's ratio is against the FR 6 target
    (391.2 at 704), not another direction's. AC16-r5: a change to the raw
    edit alone keeps the approval (the hash covers `<d>.png` only)."""

    def test_ac16_r2_without_a_raw_edit_the_ratio_is_measured_on_the_still(self):
        code, out = self.run_cli(["go", str(self.project)], fake=self.fake)
        self.assertIn(("e", "1.06x"), self.size_lines(out), out)
        # no raw edit for e; e.png itself at 0.90 of the e target (the
        # re-anchored e.png measured 1.00, the deleted raw edit 1.06)
        (self.master / "e.edit.png").unlink()
        direction_still(self.master / "e.png", 0.90, d="e")
        fake = self._fresh_fake()
        code, out = self.run_cli(["go", str(self.project)], fake=fake)
        self.assert_stopped_with_approve_command(code, out)
        self.assertIn(("e", "0.90x"), self.size_lines(out), out)
        self.assertEqual(fake.edit_calls(), [])  # e.png exists: no re-edit

    def test_ac16_r2_ref_size_reads_the_raw_edit_then_the_still(self):
        ctx = stages.StageCtx(project.load_project(self.project), config.load())
        direction_still(self.master / "e.png", 1.06, d="e")
        self.assertEqual(ctx.ref_size("courier", "e"), 1.06)
        direction_still(self.master / "e.edit.png", 1.30, d="e")
        self.assertEqual(ctx.ref_size("courier", "e"), 1.30)

    def test_ac16_r3_the_s_master_is_measured_against_the_fr6_target(self):
        ctx = stages.StageCtx(project.load_project(self.project), config.load())
        # 430 px: 430 / 391.2 = 1.10 (against the e target 471.5 it would
        # read 0.91, against se 433.7 0.99)
        self.master.mkdir(parents=True, exist_ok=True)
        mannequins.pose_frame(832, 1024, body_h=430, body_w=200,
                              feet_from_bottom=10).save(self.master / "s.png")
        self.assertEqual(ctx.ref_size("courier", "s"), 1.10)

    def test_ac16_r5_a_changed_raw_edit_alone_keeps_the_approval(self):
        code, out = self.run_cli(["go", str(self.project)], fake=self.fake)
        self.assert_stopped_with_approve_command(code, out)
        code, out = self.run_cli(self.approve_cmd())
        self.assertEqual(code, 0, out)
        approved = (self.out_char / "refs.approved").read_text()
        before = (self.master / "n.edit.png").read_bytes()
        direction_still(self.master / "n.edit.png", 1.00, marker=(200, 20, 20), d="n")
        self.assertNotEqual((self.master / "n.edit.png").read_bytes(), before)
        code, out = self.run_cli(["go", str(self.project), "--dry-run"])
        self.assertEqual(code, 0, out)
        self.assertEqual(self.stopped_lines(out), [], out)
        self.assertIn("stage=animate", out)
        self.assertEqual((self.out_char / "refs.approved").read_text(), approved)
        self.assertEqual(approved.strip(), self.stills_hash())
