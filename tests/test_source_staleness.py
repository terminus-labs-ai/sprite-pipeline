"""Source-still fingerprint and stale-source rebuild (spec FR 10 amendment 2,
AC13).

`go` writes the lowercase sha256 of each character's source still bytes at
out/<char>/master/source.sha256 whenever the stills are prepared. Before
running or planning any work for a character whose out/<char>/ exists, it
compares that file with the source's current bytes: a difference (changed
file or repointed path) deletes the whole out/<char>/ and rebuilds the
character from still preparation through the review stop — which invalidates
refs.approved and every derived clip, loop and export. A --dry-run prints the
stale-source line but deletes nothing. An output directory without
source.sha256 is unknown, not stale: the hash of the current bytes is written
and the outputs are kept. Other characters' outputs and approvals are
untouched.

Same fake-ComfyUI fixture style as tests/test_review.py: the edit calls are
faked at subprocess.run, the config home is isolated per test.
"""
import hashlib
import unittest
from pathlib import Path

from PIL import Image

import mannequins

from sprite_pipeline import config, project, stages
from test_review import (GEN_DIRS, TINY_PNG, ReviewFixture,
                         make_stub_project)

STILL = "stills/courier.png"
# a *different* valid 2x2 RGBA PNG (every pixel (9,9,9,255)): changed source
CHANGED_PNG = bytes.fromhex(
    "89504e470d0a1a0a0000000d4948445200000002000000020806000000"
    "72b60d240000001449444154789c63e4e4e4fccfc0c0c0c0c40005000f"
    "d8011ec61914650000000049454e44ae426082")


def sha256(b):
    return hashlib.sha256(b).hexdigest()


def _standing_other():
    """A *different* valid standing-figure still (changed source bytes): flat
    background, figure 320 px tall x 180 wide, feet 12 px from the bottom."""
    import io

    from mannequins import pose_frame
    buf = io.BytesIO()
    pose_frame(520, 920, body_h=320, body_w=180, feet_from_bottom=12,
               corner=(20, 90, 200), colour=(240, 60, 60)).save(buf, format="PNG")
    return buf.getvalue()


def _add_guard(tmp, still):
    """Add a second character `guard` with its own still to the stub project
    (YAML round-trip; the naming pattern gains {character} so two characters
    produce unique frame names)."""
    import yaml
    base = Path(tmp) / "isometric"
    (base / "stills" / still).parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (64, 80), (200, 10, 10)).save(base / "stills" / still)
    pf = base / "project.yaml"
    d = yaml.safe_load(pf.read_text())
    if any(c.get("name") == "guard" for c in d.get("characters") or []):
        return
    d["characters"].append({"name": "guard", "still": f"stills/{still}",
                            "describe": "a guard in a cyberpunk city",
                            "pronoun": "he"})
    d["naming"] = "{character}_{animation}_{dir}"
    pf.write_text(yaml.safe_dump(d, sort_keys=False))


class StalenessFixture(ReviewFixture):
    """ReviewFixture plus a fingerprint assertion helper and a second
    character (`guard`) for the two-character rows."""

    def source_hash_file(self):
        return self.master / "source.sha256"

    def assert_fingerprint(self, expected_bytes):
        self.assertTrue(self.source_hash_file().is_file())
        self.assertEqual(self.source_hash_file().read_text().strip(),
                         sha256(expected_bytes))

    def still_path(self):
        return self.base / STILL

    def ctx(self):
        return stages.StageCtx(project.load_project(self.project),
                               config.load())


class TestSourceFingerprint(StalenessFixture):
    """AC13: the fingerprint is the lowercase sha256 of the source still
    bytes, written at out/<char>/master/source.sha256 every time the stills
    are prepared (FR 10 amendment 2)."""

    def test_first_run_writes_the_source_fingerprint(self):
        code, out = self.run_cli(["go", str(self.project)], fake=self.fake)
        self.assertEqual(code, 0, out)
        self.assert_fingerprint(self.still_path().read_bytes())

    def test_rerun_after_source_change_rewrites_the_fingerprint(self):
        self.run_cli(["go", str(self.project)], fake=self.fake)
        self.still_path().write_bytes(_standing_other())
        code, out = self.run_cli(["go", str(self.project)], fake=self.fake)
        self.assertEqual(code, 0, out)
        self.assert_fingerprint(self.still_path().read_bytes())

    def test_fingerprint_survives_a_matching_rerun(self):
        # matching bytes: the fingerprint stays, and the run skips the stills
        self.run_cli(["go", str(self.project)], fake=self.fake)
        fake = self._fresh_fake()
        code, out = self.run_cli(["go", str(self.project)], fake=fake)
        self.assertEqual(code, 0, out)
        self.assertEqual(fake.calls, [])
        self.assert_fingerprint(self.still_path().read_bytes())


class TestStaleSourceRebuild(StalenessFixture):
    """AC13-r1: changing a built, approved character's source still bytes
    makes a non-dry go delete its output directory, redo its stills, write
    refs-sheet.png and stop for approval again (exit 0 printing the approve
    command); source.sha256 is rewritten (FR 10 amendment 2)."""

    def build_and_approve(self):
        # amendment 12: the full build is go -> approve (s stop) -> go ->
        # approve (refs stop)
        code, out = self.run_cli(["go", str(self.project)], fake=self.fake)
        self.assert_stopped_with_approve_command(code, out)
        code, out = self.run_cli(self.approve_cmd())
        self.assertEqual(code, 0, out)
        code, out = self.run_cli(["go", str(self.project)], fake=self.fake)
        self.assert_stopped_with_approve_command(code, out)
        code, out = self.run_cli(self.approve_cmd())
        self.assertEqual(code, 0, out)
        self.assertTrue(self.ctx().is_approved("courier"))

    def test_r1_changed_source_bytes_rebuild_and_stop_again(self):
        self.build_and_approve()
        fingerprint = self.source_hash_file().read_text().strip()
        self.still_path().write_bytes(_standing_other())  # changed file
        fake = self._fresh_fake()
        code, out = self.run_cli(["go", str(self.project)], fake=fake)
        # the whole character output directory was deleted then rebuilt
        self.assertIn(f"[courier] stale source still: {self.still_path()}", out)
        self.assertNotIn(fingerprint, out)
        # the rebuild stops at the s sheet first (amendment 12): only the s
        # work ran, s.png sits on the edit canvas
        self.assert_fingerprint(self.still_path().read_bytes())
        self.assertEqual(Image.open(self.master / "s.png").size, (832, 1024))
        self.assertTrue((self.out_char / "s-sheet.png").is_file())
        # the old approval is gone with the directory: the run stops again
        self.assertFalse((self.out_char / "refs.approved").exists())
        self.assertFalse(self.ctx().is_approved("courier"))
        self.assert_stopped_with_approve_command(code, out)
        # approved, the next run edits the other directions from the new s
        # and stops at the refs sheet
        code, out = self.run_cli(self.approve_cmd())
        self.assertEqual(code, 0, out)
        code, out = self.run_cli(["go", str(self.project)], fake=fake)
        for d in GEN_DIRS:
            self.assertTrue((self.master / f"{d}.png").is_file(), d)
        self.assertTrue((self.out_char / "refs-sheet.png").is_file())
        self.assert_stopped_with_approve_command(code, out)

    def test_r3_repointed_still_path_is_the_same_trigger(self):
        self.build_and_approve()
        # the repointed still: a different standing figure (FR 6 accepts it)
        mannequins.pose_frame(560, 880, body_h=280, body_w=160,
                              feet_from_bottom=8).save(self.base / "stills" / "courier2.png")
        pf = self.base / "project.yaml"
        pf.write_text(pf.read_text().replace("still: stills/courier.png",
                                             "still: stills/courier2.png"))
        fake = self._fresh_fake()
        code, out = self.run_cli(["go", str(self.project)], fake=fake)
        self.assertIn(f"[courier] stale source still: {self.base}/stills/courier2.png", out)
        self.assert_fingerprint((self.base / "stills" / "courier2.png").read_bytes())
        self.assertTrue((self.out_char / "s-sheet.png").is_file())
        self.assertFalse((self.out_char / "refs.approved").exists())
        self.assert_stopped_with_approve_command(code, out)

    def test_r7_stale_line_names_the_source_file_path(self):
        self.build_and_approve()
        self.still_path().write_bytes(_standing_other())
        code, out = self.run_cli(["go", str(self.project)], fake=self._fresh_fake())
        line = next(l for l in out.splitlines() if "stale source still" in l)
        self.assertEqual(line, f"[courier] stale source still: {self.still_path()}")


class TestDryRunStale(StalenessFixture):
    """AC13-r2: a --dry-run on a stale project prints the stale-source line
    but deletes nothing (FR 10 dry-run clause)."""

    def test_r2_dry_run_prints_and_deletes_nothing(self):
        code, out = self.run_cli(["go", str(self.project)], fake=self.fake)
        self.assert_stopped_with_approve_command(code, out)
        code, out = self.run_cli(self.approve_cmd())
        self.assertEqual(code, 0, out)
        code, out = self.run_cli(["go", str(self.project)], fake=self.fake)
        self.assert_stopped_with_approve_command(code, out)
        code, out = self.run_cli(self.approve_cmd())
        self.assertEqual(code, 0, out)
        before = {p: p.read_bytes() for p in self.out_char.rglob("*") if p.is_file()}
        self.still_path().write_bytes(_standing_other())
        fake = self._fresh_fake()
        code, out = self.run_cli(["go", str(self.project), "--dry-run"], fake=fake)
        self.assertEqual(code, 0, out)
        self.assertIn(f"[courier] stale source still: {self.still_path()}", out)
        self.assertIn("no deletion", out)
        # nothing deleted, nothing run: the tree is byte-identical
        after = {p: p.read_bytes() for p in self.out_char.rglob("*") if p.is_file()}
        self.assertEqual(before, after)
        self.assertEqual(fake.calls, [])
        self.assertTrue(self.ctx().is_approved("courier"))


class TestMatchingBytes(StalenessFixture):
    """AC13-r4: an approved character whose source still bytes are unchanged
    proceeds past the review stop unchanged — no stale line, no deletion, no
    still work (FR 10 amendment 2)."""

    def test_r4_matching_bytes_change_nothing(self):
        # the two-stop build: go -> approve (s) -> go -> approve (refs)
        self.run_cli(["go", str(self.project)], fake=self.fake)
        code, out = self.run_cli(self.approve_cmd())
        self.assertEqual(code, 0, out)
        self.run_cli(["go", str(self.project)], fake=self.fake)
        code, out = self.run_cli(self.approve_cmd())
        self.assertEqual(code, 0, out)
        before = {p: p.read_bytes() for p in self.out_char.rglob("*") if p.is_file()}
        # touch the source file (mtime changes) without changing its bytes
        import time
        time.sleep(0.01)
        self.still_path().write_bytes(self.still_path().read_bytes())
        fake = self._fresh_fake()
        code, out = self.run_cli(["go", str(self.project), "--dry-run"],
                                 fake=fake)
        self.assertEqual(code, 0, out)
        self.assertNotIn("stale source still", out)
        self.assertNotIn("no source fingerprint", out)
        # proceeds past the review stop into the animation stages
        for stage in ("drive", "animate", "key", "loop", "pixel", "export"):
            self.assertIn(f"stage={stage}", out, stage)
        after = {p: p.read_bytes() for p in self.out_char.rglob("*") if p.is_file()}
        self.assertEqual(before, after)
        self.assertEqual(fake.calls, [])


class TestLegacyDirectory(StalenessFixture):
    """AC13-r5: an output directory without source.sha256 is unknown, not
    stale: go writes the hash of the current bytes, notes it once, and keeps
    the outputs (FR 10 hashless-legacy-directory clause)."""

    def seed_legacy(self):
        # a built+approved tree that carries no fingerprint
        self.seed_stills()
        code, out = self.run_cli(self.approve_cmd())
        self.assertEqual(code, 0, out)
        (self.out_char / "refs-sheet.png").write_bytes(TINY_PNG)
        self.assertTrue(self.ctx().is_approved("courier"))
        self.assertFalse(self.source_hash_file().exists())

    def test_r5_legacy_hashless_dir_gets_the_hash_and_keeps_outputs(self):
        self.seed_legacy()
        before = {p: p.read_bytes() for p in self.out_char.rglob("*") if p.is_file()}
        fake = self._fresh_fake()
        code, out = self.run_cli(["go", str(self.project), "--dry-run"], fake=fake)
        self.assertEqual(code, 0, out)
        self.assertNotIn("stale source still", out)
        self.assertIn("no source fingerprint", out)
        self.assert_fingerprint(self.still_path().read_bytes())
        # outputs kept: the tree is identical except for the new fingerprint
        after = {p: b for p, b in
                 {p: p.read_bytes() for p in self.out_char.rglob("*") if p.is_file()
                  }.items() if p != self.source_hash_file()}
        self.assertEqual(before, after)
        self.assertEqual(fake.calls, [])
        self.assertTrue(self.ctx().is_approved("courier"))

    def test_blank_hash_file_is_unknown_not_stale(self):
        self.seed_legacy()
        self.master.mkdir(parents=True, exist_ok=True)
        self.source_hash_file().write_text("   \n")
        fake = self._fresh_fake()
        code, out = self.run_cli(["go", str(self.project), "--dry-run"], fake=fake)
        self.assertEqual(code, 0, out)
        self.assertNotIn("stale source still", out)
        self.assertIn("no source fingerprint", out)
        self.assert_fingerprint(self.still_path().read_bytes())
        self.assertTrue(self.ctx().is_approved("courier"))
        self.assertEqual(fake.calls, [])

    def test_legacy_then_changed_source_is_stale(self):
        # once the fingerprint exists, a later change is stale, not unknown
        self.seed_legacy()
        self.run_cli(["go", str(self.project)], fake=self._fresh_fake())
        self.still_path().write_bytes(_standing_other())
        code, out = self.run_cli(["go", str(self.project)], fake=self._fresh_fake())
        self.assertEqual(code, 0, out)
        self.assertIn(f"[courier] stale source still: {self.still_path()}", out)
        self.assert_fingerprint(self.still_path().read_bytes())


class TestTwoCharacterIsolation(StalenessFixture):
    """AC13-r6: in a two-character project, staling one character deletes and
    rebuilds only its outputs; the other character's outputs and approval are
    untouched (FR 10 amendment 2)."""

    def setUp(self):
        super().setUp()
        _add_guard(self.tmp, "guard.png")
        self.proj = project.load_project(self.project)  # reload with guard
        self.ctx_guard = stages.StageCtx(self.proj, config.load())
        self.out_guard = self.base / "out" / "guard"
        self.master_guard = self.out_guard / "master"

    def build_both(self):
        # Seed both characters' direction stills and approve them; only guard's
        # source fingerprint goes on record (see the test).
        for ch, color in (("courier", (10, 200, 10)), ("guard", (200, 10, 10))):
            master = self.base / "out" / ch / "master"
            master.mkdir(parents=True, exist_ok=True)
            im = Image.new("RGB", (32, 32), color)
            for d in GEN_DIRS:
                im.save(master / f"{d}.png")
            (master / "s.keyed.png").write_bytes(TINY_PNG)
            (self.base / "out" / ch / "refs-sheet.png").write_bytes(TINY_PNG)
            # amendment 12: two approve calls — s.approved, then refs.approved
            for _ in range(2):
                code, out = self.run_cli(["approve", str(self.project), ch])
                self.assertEqual(code, 0, out)

    def test_r6_only_the_stale_character_is_rebuilt(self):
        self.build_both()
        # The fingerprint is written by `go`, never by `approve`: guard's goes
        # on record and matches its source bytes — steady state. courier's
        # source still is then rewritten, so courier is stale: the non-dry run
        # deletes and rebuilds courier only.
        cs_guard = next(c for c in self.proj.characters if c.name == "guard")
        self.ctx_guard.write_source_hash("guard", cs_guard)
        self.assertTrue(self.ctx_guard.is_approved("guard"))
        self.assertFalse(self.ctx_guard.source_stale("guard", cs_guard))
        # courier needs a fingerprint on record before its source can go stale
        # (a hashless directory is unknown, not stale): write the old one
        cs_courier = next(c for c in self.proj.characters if c.name == "courier")
        self.ctx().write_source_hash("courier", cs_courier)
        self.still_path().write_bytes(_standing_other())
        self.assertTrue(self.ctx().source_stale("courier", cs_courier))
        guard_before = {p: p.read_bytes()
                        for p in self.out_guard.rglob("*") if p.is_file()}
        # narrow to guard's already-complete stage set (drive..loop): the row
        # is about which character the purge touched, not about driving the
        # faked pipeline through an export the stub frames never satisfy
        fake = self._fresh_fake()
        code, out = self.run_cli(["go", str(self.project), "--stage", "drive",
                                  "--animations", "run", "--dirs", "s"],
                                 fake=fake)
        self.assertEqual(code, 0, out)
        # courier: purged, its stills redone from the new source, stopped for
        # the review again with the approve command
        self.assertIn(f"[courier] stale source still: {self.still_path()}", out)
        self.assert_stopped_with_approve_command(code, out)
        self.assertFalse((self.out_char / "refs.approved").exists())
        self.assertEqual((self.master / "source.sha256").read_text().strip(),
                         sha256(self.still_path().read_bytes()))
        # guard: its directory, fingerprint and approval byte-identical
        guard_after = {p: p.read_bytes()
                       for p in self.out_guard.rglob("*") if p.is_file()}
        self.assertEqual(guard_before, guard_after)
        self.assertTrue(self.ctx_guard.is_approved("guard"))
        self.assertFalse(self.ctx_guard.source_stale("guard", cs_guard))


class TestDropCharOutputs(unittest.TestCase):
    """The delete itself: drop_char_outputs removes the whole character
    directory; a path that resolves outside the project output directory is
    refused (never deleted)."""

    def _ctx(self, tmp, output="out"):
        return stages.StageCtx(
            project.load_project(make_stub_project(tmp)),
            config.defaults(Path(tmp) / "home"))

    def test_delete_removes_the_character_directory(self):
        with __import__("tempfile").TemporaryDirectory() as td:
            ctx = self._ctx(td)
            out_char = ctx.project.output_root() / "courier"
            (out_char / "master").mkdir(parents=True)
            (out_char / "master" / "s.png").write_bytes(TINY_PNG)
            (out_char / "refs.approved").write_text("x\n")
            ctx.drop_char_outputs("courier")
            self.assertFalse(out_char.exists())

    def test_delete_refused_outside_the_output_directory(self):
        # a character directory whose resolved path is not inside the project
        # output directory is refused (never deleted)
        with __import__("tempfile").TemporaryDirectory() as td:
            td = Path(td)
            ctx = self._ctx(td)
            out_root = ctx.project.output_root()
            out_root.mkdir(parents=True)
            # an absolute character name resolves outside the output root
            outside = td / "elsewhere"
            outside.mkdir()
            (outside / "marker.png").write_bytes(TINY_PNG)
            with self.assertRaises(ValueError):
                ctx.drop_char_outputs(str(outside))
            self.assertTrue((outside / "marker.png").exists())
            # sanity: a normal character directory inside the root is deleted
            (out_root / "courier" / "master").mkdir(parents=True)
            ctx.drop_char_outputs("courier")
            self.assertFalse((out_root / "courier").exists())


if __name__ == "__main__":
    unittest.main()
