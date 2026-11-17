"""Tests for the `s` review stop (spec AC20, FR 8 amendment 12).

With Blender and ComfyUI faked and generate `[s, se, e, ne, n]`, a first
`go` runs only the `s` guide, edit, re-anchor and cutout and stops at
`s-sheet.png` (three panels: anchored still, `s` guide, guided `s`);
`approve` writes `s.approved` holding the sha256 of `master/s.png`; the next
`go` edits the other four from `master/s.png` and stops at the full review
stop, where `approve` writes refs.approved. `--no-stop` on a fresh output
runs all five edits in one run. An `s` redo (`--reroll <char>:s=<seed>`, a
missing `master/s.png` or a changed `s.png` the approval no longer matches)
deletes the other four `master/<d>.png` and `master/<d>.edit.png` and stops
at the `s` stop again; a non-`s` `--reroll` with `s` unapproved runs no edit
and prints that it waits for the `s` approval. When the `s` edit fails, no
cutout and no other edit runs and the other stills stay untouched. A
project generating only `s` has no `s` stop: it stops at the full review
stop and `approve` writes refs.approved.
"""
import hashlib
import unittest
from pathlib import Path

from PIL import Image

from test_dir_anchor import GuidedFixture
from test_review import EDIT_DIRS, FakeRun

# an 832x1024 tile at the sheet height 340 (same layout as refs-sheet.png)
TILE_W = round(832 * 340 / 1024)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class SStopFixture(GuidedFixture):
    """GuidedFixture plus helpers for the two-stop build."""

    def go(self, *extra, fake=None):
        return self.run_cli(["go", str(self.project), *extra],
                            fake=fake or self.fake)

    def approve(self):
        return self.run_cli(self.approve_cmd())

    def build_s(self):
        """Run `go` once: the s work, then the s stop."""
        code, out = self.go()
        self.assert_stopped_with_approve_command(code, out)
        return code, out

    def build_all(self):
        """The full two-stop build, ending with refs.approved written."""
        self.build_s()
        code, out = self.approve()
        self.assertEqual(code, 0, out)
        code, out = self.go()
        self.assert_stopped_with_approve_command(code, out)
        code, out = self.approve()
        self.assertEqual(code, 0, out)

    def edits(self, fake=None):
        return (fake or self.fake).edit_calls()


# --------------------------------------------------------- AC20-r1/r2: the s stop

class TestFirstRunStopsAtTheSSheet(SStopFixture):
    """AC20-r1: a first `go` runs only the `s` guide, edit, re-anchor and
    cutout, writes `s-sheet.png` with three panels (anchored still, `s`
    guide, guided `s`), runs no other edit and prints the approve command.
    AC20-r2: `approve` then writes `s.approved` with the sha256 of
    `master/s.png` and no refs.approved."""

    def test_ac20_r1_only_the_s_work_runs_then_the_s_stop(self):
        code, out = self.go()
        self.assertEqual(code, 0, out)
        # one edit, for s; no other direction's guide, edit or still exists
        edits = self.edits()
        self.assertEqual(len(edits), 1, "\n".join(" ".join(c) for c in self.fake.calls))
        e = edits[0]
        self.assertEqual(e[e.index("--copy-to") + 1], str(self.master / "s.edit.png"))
        self.assertEqual(e[e.index("--image") + 1], str(self.master / "still.png"))
        self.assertEqual(e[e.index("--pose-image") + 1],
                         str(self.master / "s.guide.png"))
        for p in ("still.png", "s.guide.png", "s.png", "s.keyed.png"):
            self.assertTrue((self.master / p).is_file(), p)
        for d in EDIT_DIRS:
            for suffix in (".guide.png", ".edit.png", ".png"):
                self.assertFalse((self.master / f"{d}{suffix}").exists(),
                                 f"{d}{suffix}")
        # one cutout, from s.png to s.keyed.png
        cutouts = [c for c in self.fake.calls if "cutout" in " ".join(c)]
        self.assertEqual(len(cutouts), 1)
        # the s sheet holds its three panels
        sheet = self.out_char / "s-sheet.png"
        self.assertTrue(sheet.is_file())
        self.assertEqual(Image.open(sheet).size, (TILE_W * 3, 340 + 24))
        self.assert_stopped_with_approve_command(code, out)
        self.assertIn(str(sheet), out)

    def test_ac20_r2_approve_writes_the_s_hash_only(self):
        self.build_s()
        code, out = self.approve()
        self.assertEqual(code, 0, out)
        f = self.out_char / "s.approved"
        self.assertTrue(f.is_file())
        self.assertEqual(f.read_text().strip(), sha256(self.master / "s.png"))
        self.assertFalse((self.out_char / "refs.approved").exists())


# ------------------------------------------------- AC20-r3/r4: the second `go`

class TestSecondRunEditsTheRest(SStopFixture):
    """AC20-r3: the next `go` after the s approval edits se, e, ne, n (each
    `--image master/s.png`), runs no `s` edit, writes refs-sheet.png and
    stops at the full review stop. AC20-r4: `approve` then writes
    refs.approved and the next `go` proceeds past both stops."""

    def test_ac20_r3_four_edits_from_the_guided_master(self):
        self.build_s()
        code, out = self.approve()
        self.assertEqual(code, 0, out)
        fake = self._fresh_fake()
        code, out = self.go(fake=fake)
        self.assertEqual(code, 0, out)
        edits = fake.edit_calls()
        self.assertEqual(len(edits), 4, "\n".join(" ".join(c) for c in fake.calls))
        for e in edits:
            d = Path(e[e.index("--copy-to") + 1]).name.split(".")[0]
            self.assertIn(d, EDIT_DIRS)
            self.assertEqual(e[e.index("--image") + 1], str(self.master / "s.png"))
            self.assertEqual(e[e.index("--pose-image") + 1],
                             str(self.master / f"{d}.guide.png"))
        # no s edit: s.png and s.edit.png are the first run's bytes
        self.assertEqual([e for e in edits if "s.edit.png" in
                          e[e.index("--copy-to") + 1]], [])
        self.assertTrue((self.out_char / "refs-sheet.png").is_file())
        self.assert_stopped_with_approve_command(code, out)

    def test_ac20_r4_second_approve_writes_refs_and_go_proceeds(self):
        self.build_all()
        f = self.out_char / "refs.approved"
        self.assertTrue(f.is_file())
        code, out = self.go("--dry-run")
        self.assertEqual(code, 0, out)
        self.assertEqual(self.stopped_lines(out), [], out)
        self.assertIn("stage=animate", out)


# ----------------------------------------------------------------- AC20-r5: --no-stop

class TestNoStopRunsAllFive(SStopFixture):
    """AC20-r5: `go --no-stop` on a fresh output runs all five edits in one
    invocation, `s` first, and no approval file is written."""

    def test_ac20_r5_no_stop_edits_s_then_the_four(self):
        # --stage keeps the assertion on the still work (the faked pipeline
        # cannot satisfy export)
        code, out = self.go("--no-stop", "--stage", "pixel")
        self.assertEqual(code, 0, out)
        edits = self.edits()
        self.assertEqual(len(edits), 5)
        order = [Path(e[e.index("--copy-to") + 1]).name.split(".")[0]
                 for e in edits]
        self.assertEqual(order, ["s", "se", "e", "ne", "n"])
        self.assertEqual(edits[0][edits[0].index("--image") + 1],
                         str(self.master / "still.png"))
        for e in edits[1:]:
            self.assertEqual(e[e.index("--image") + 1], str(self.master / "s.png"))
        self.assertFalse((self.out_char / "s.approved").exists())
        self.assertFalse((self.out_char / "refs.approved").exists())


# --------------------------------------------------- AC20-r6/r7/r8: --reroll and s

class TestSReroll(SStopFixture):
    """AC20-r6: `go --reroll courier:s=9` over a fully approved set runs the
    `s` edit with seed 9, its re-anchor and cutout, deletes the other four
    `master/<d>.png` and `master/<d>.edit.png` and stops at the `s` stop —
    no other edit runs. AC20-r7: `--reroll courier:e=3` with `s` approved
    runs only the `e` edit. AC20-r8: the same with `s` unapproved runs no
    edit and prints that `e` waits for the `s` approval."""

    def test_ac20_r6_s_reroll_reedits_s_deletes_the_rest_and_stops(self):
        self.build_all()
        before = {d: (self.master / f"{d}.png").read_bytes() for d in EDIT_DIRS}
        fake = self._fresh_fake()
        code, out = self.run_cli(
            ["go", str(self.project), "--reroll", "courier:s=9"], fake=fake)
        self.assertEqual(code, 0, out)
        edits = fake.edit_calls()
        self.assertEqual(len(edits), 1, "\n".join(" ".join(c) for c in fake.calls))
        e = edits[0]
        self.assertEqual(e[e.index("--copy-to") + 1], str(self.master / "s.edit.png"))
        self.assertEqual(e[e.index("--seed") + 1], "9")
        # the other four stills and raw edits are deleted
        for d in EDIT_DIRS:
            self.assertFalse((self.master / f"{d}.png").exists(), d)
            self.assertFalse((self.master / f"{d}.edit.png").exists(), d)
            self.assertNotEqual(before.get(d), None)
        # the s guide stays (it is not the still); the cutout re-ran
        self.assertTrue((self.master / "s.guide.png").is_file())
        self.assertTrue((self.master / "s.keyed.png").is_file())
        self.assertTrue((self.out_char / "s-sheet.png").is_file())
        self.assert_stopped_with_approve_command(code, out)

    def test_ac20_r7_other_reroll_with_s_approved_edits_only_it(self):
        self.build_all()
        fake = self._fresh_fake()
        code, out = self.run_cli(
            ["go", str(self.project), "--reroll", "courier:e=3"], fake=fake)
        self.assertEqual(code, 0, out)
        edits = fake.edit_calls()
        self.assertEqual([Path(e[e.index("--copy-to") + 1]).name for e in edits],
                         ["e.edit.png"])
        self.assertEqual(edits[0][edits[0].index("--seed") + 1], "3")
        self.assertEqual(edits[0][edits[0].index("--image") + 1],
                         str(self.master / "s.png"))

    def test_ac20_r8_other_reroll_with_s_unapproved_waits(self):
        self.build_s()  # s built but not approved
        fake = self._fresh_fake()
        code, out = self.run_cli(
            ["go", str(self.project), "--reroll", "courier:e=3"], fake=fake)
        self.assertEqual(code, 0, out)
        self.assertEqual(fake.edit_calls(), [])
        self.assertIn("e waits for the s approval", out)
        self.assert_stopped_with_approve_command(code, out)


# ------------------------------------------------------------ AC20-r9: s failure

class TestSEditFailure(SStopFixture):
    """AC20-r9: with the `s` edit faked to fail (non-zero exit) and the
    other stills present from a prior approved set, no cutout and no other
    edit runs, the other `master/<d>.png` files are byte-identical to before
    and `go` exits 1."""

    def test_ac20_r9_a_failed_s_edit_touches_nothing_else(self):
        self.build_all()
        before = {d: (self.master / f"{d}.png").read_bytes() for d in EDIT_DIRS}
        (self.master / "s.png").unlink()  # force the s redo
        fake = FakeRun(fails={"sprite_pipeline.comfy edit": 1},
                       after={"render_drive.py": self._produce_frames,
                              "ffmpeg": self._produce_drive})
        code, out = self.go(fake=fake)
        self.assertEqual(code, 1, out)
        self.assertEqual(len(fake.edit_calls()), 1)  # the s edit only
        self.assertEqual([c for c in fake.calls if "cutout" in " ".join(c)], [])
        for d in EDIT_DIRS:
            self.assertEqual((self.master / f"{d}.png").read_bytes(), before[d], d)
        self.assertFalse((self.master / "s.png").exists())


# ------------------------------------------------- AC20-r10: generate [s] only

class TestSOnlyProjectHasNoSStop(GuidedFixture):
    """AC20-r10: a project with generate `[s]` only has no `s` stop: `go`
    stops at the full review stop and `approve` writes refs.approved — no
    s-sheet.png, no s.approved."""

    def setUp(self):
        super().setUp()
        pf = self.project
        pf.write_text(pf.read_text()
                      .replace("generate: [s, se, e, ne, n]", "generate: [s]")
                      .replace("mirror: {sw: se, w: e, nw: ne}", "mirror: {}"))

    def test_ac20_r10_s_only_project_stops_at_the_refs_sheet(self):
        code, out = self.run_cli(["go", str(self.project)], fake=self.fake)
        self.assertEqual(code, 0, out)
        self.assertEqual(len(self.fake.edit_calls()), 1)  # the s edit only
        self.assertFalse((self.out_char / "s-sheet.png").exists())
        self.assertTrue((self.out_char / "refs-sheet.png").is_file())
        self.assert_stopped_with_approve_command(code, out)
        code, out = self.run_cli(self.approve_cmd())
        self.assertEqual(code, 0, out)
        self.assertFalse((self.out_char / "s.approved").exists())
        self.assertTrue((self.out_char / "refs.approved").is_file())


# ------------------------------------------- AC20-r11: a changed s.png re-stops

class TestChangedSRestops(SStopFixture):
    """AC20-r11: `s.approved` present but `master/s.png` bytes changed —
    `go` stops at the `s` stop again and edits no other direction; the other
    stills are stale and deleted, remade from the changed `s` once it is
    approved again."""

    def test_ac20_r11_changed_s_stops_at_the_s_stop_and_invalidates(self):
        self.build_all()
        # s.png changed out-of-band: the approval no longer matches
        (self.master / "s.png").write_bytes(
            (self.master / "n.png").read_bytes())
        fake = self._fresh_fake()
        code, out = self.go(fake=fake)
        self.assertEqual(code, 0, out)
        self.assertEqual(fake.edit_calls(), [])
        for d in EDIT_DIRS:  # stale stills deleted
            self.assertFalse((self.master / f"{d}.png").exists(), d)
            self.assertFalse((self.master / f"{d}.edit.png").exists(), d)
        self.assertTrue((self.out_char / "s-sheet.png").is_file())
        self.assert_stopped_with_approve_command(code, out)
        # approve records the changed s; the next run remakes the other four
        # from it and reaches the review stop
        code, out = self.approve()
        self.assertEqual(code, 0, out)
        self.assertEqual((self.out_char / "s.approved").read_text().strip(),
                         sha256(self.master / "s.png"))
        fake = self._fresh_fake()
        code, out = self.go(fake=fake)
        self.assertEqual(code, 0, out)
        edits = fake.edit_calls()
        self.assertEqual(len(edits), 4, "\n".join(" ".join(c) for c in fake.calls))
        for e in edits:
            self.assertEqual(e[e.index("--image") + 1], str(self.master / "s.png"))
        self.assertTrue((self.out_char / "refs-sheet.png").is_file())
        self.assert_stopped_with_approve_command(code, out)


if __name__ == "__main__":
    unittest.main()
