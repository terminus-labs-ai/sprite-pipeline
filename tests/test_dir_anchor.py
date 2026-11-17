"""Tests for the mannequin-guided direction stills (spec AC18, AC19; FR 7,
FR 9, amendment 10).

Per guided direction `go` renders the anchor GLB's A_TPose facing it
(Blender faked at subprocess.run by `mannequins.fake_blender`, cached in the
configured motion_cache), places it as `master/<d>.guide.png`, runs the edit
with `--image master/s.png --pose-image master/<d>.guide.png` and the guide
line appended to the prompt (ComfyUI faked at subprocess.run), keeps the raw
output as `master/<d>.edit.png` and re-anchors it to `master/<d>.png`
without the standing-figure rule.

The expected targets are stated literally from the stub profile (the female
values in tests/mannequins.py) on the 832x1024 canvas at frame size 704:
body / floor se 433.7 / 724.7, e 471.5 / 725.9, ne 418.4 / 727.1, n 361.6 /
715.3; centre x 416; chroma green (31, 203, 25).
"""
import io
import json
import tempfile
import unittest
from pathlib import Path

import numpy
from PIL import Image

import mannequins
from test_review import EDIT_DIRS, FakeRun, ReviewFixture

from sprite_pipeline import metrics, stills

ROOT = Path(__file__).resolve().parent.parent
GREEN = (31, 203, 25)
# body / floor on the 832x1024 canvas at 704, per direction (case table)
TARGET = {"se": (433.7, 724.7), "e": (471.5, 725.9), "ne": (418.4, 727.1),
          "n": (361.6, 715.3)}
CENTRE_X = 416
EDIT_BG = (10, 200, 10)
GUIDE_LINE = {
    "he": ("Image 2 is a plain grey mannequin used only as a guide: draw him in "
           "exactly the mannequin's pose, facing direction, size and position in the "
           "frame, with his feet exactly where the mannequin's feet are and his arms "
           "held out like the mannequin's. Do not draw the mannequin itself."),
    "she": ("Image 2 is a plain grey mannequin used only as a guide: draw her in "
            "exactly the mannequin's pose, facing direction, size and position in the "
            "frame, with her feet exactly where the mannequin's feet are and her arms "
            "held out like the mannequin's. Do not draw the mannequin itself."),
    "they": ("Image 2 is a plain grey mannequin used only as a guide: draw them in "
             "exactly the mannequin's pose, facing direction, size and position in "
             "the frame, with their feet exactly where the mannequin's feet are and "
             "their arms held out like the mannequin's. Do not draw the mannequin "
             "itself."),
}


def png_bytes(im):
    buf = io.BytesIO()
    im.save(buf, format="PNG")
    return buf.getvalue()


def raw_edit(d, ratio=1.30, feet_low=60, centre_x=440):
    """A raw edit for direction `d` on the 832x1024 canvas: a figure `ratio`
    x the direction's target tall, feet `feet_low` px below its floor,
    centred at `centre_x`."""
    body, floor = TARGET[d]
    return mannequins.pose_frame(832, 1024, body_h=round(ratio * body), body_w=200,
                                 bottom=round(floor) + feet_low, centre_x=centre_x,
                                 corner=EDIT_BG)


def t_pose(d, arm_w=600):
    """A raw edit wider than tall: a torso at the direction's target height
    with T-pose arms `arm_w` px wide."""
    body, floor = TARGET[d]
    im = mannequins.pose_frame(832, 1024, body_h=round(body), body_w=80,
                               bottom=round(floor), corner=EDIT_BG)
    top = round(floor) - round(body) + 1
    im.paste(mannequins.STILL_FIGURE, (416 - arm_w // 2, top + 40,
                                       416 + arm_w // 2, top + 70))
    return im


class GuidedFixture(ReviewFixture):
    """The review fixture whose faked edit writes `self.raw[d]` (a PIL
    image) as the raw edit and records the exact bytes written."""

    def setUp(self):
        super().setUp()
        self.raw = {d: raw_edit(d) for d in EDIT_DIRS}
        self.written = {}

    def _make_still(self, argv):
        dst = Path(argv[argv.index("--copy-to") + 1])
        d = dst.name.split(".")[0]
        data = png_bytes(self.raw[d])
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_bytes(data)
        self.written[d] = data

    def renders(self, fake):
        """The faked Blender A_TPose renders, by facing degrees."""
        return [c[c.index("--") + 4] for c in fake.calls
                if "render_drive.py" in " ".join(c) and "A_TPose" in c]

    def assert_on_target(self, path, d):
        body, floor = TARGET[d]
        m = metrics.measure(path)
        self.assertAlmostEqual(m["body_h"], body, delta=1, msg=f"{path} body")
        self.assertAlmostEqual(m["floor_y"], floor, delta=1, msg=f"{path} floor")
        self.assertAlmostEqual(m["centre_x"], CENTRE_X, delta=1, msg=f"{path} centre")
        im = Image.open(path).convert("RGB")
        self.assertEqual(im.size, (832, 1024))
        self.assertEqual(im.getpixel((0, 0)), GREEN)
        self.assertEqual(im.getpixel((831, 1023)), GREEN)


# ------------------------------------------------------------ AC18-r1: edit argv

class TestGuidedEditArgv(GuidedFixture):
    """AC18-r1: each direction edit runs with `--image master/s.png` and
    `--pose-image master/<d>.guide.png`; there is no edit for s."""

    def test_ac18_r1_each_edit_gets_the_master_and_its_guide(self):
        code, out = self.run_cli(["go", str(self.project)], fake=self.fake)
        self.assert_stopped_with_approve_command(code, out)
        edits = self.fake.edit_calls()
        by_dir = {Path(e[e.index("--copy-to") + 1]).name: e for e in edits}
        self.assertEqual(sorted(by_dir), sorted(f"{d}.edit.png" for d in EDIT_DIRS))
        self.assertEqual(len(edits), 4)
        for d in EDIT_DIRS:
            e = by_dir[f"{d}.edit.png"]
            self.assertEqual(e[e.index("--image") + 1], str(self.master / "s.png"))
            self.assertEqual(e[e.index("--pose-image") + 1],
                             str(self.master / f"{d}.guide.png"))
            self.assertEqual(e[e.index("--copy-to") + 1],
                             str(self.master / f"{d}.edit.png"))
            self.assertTrue((self.master / f"{d}.guide.png").is_file(), d)
        self.assertFalse((self.master / "s.guide.png").exists())
        self.assertFalse((self.master / "s.edit.png").exists())
        # one A_TPose render per guided direction, none for s
        self.assertEqual(sorted(self.renders(self.fake)), ["135", "180", "45", "90"])


# --------------------------------------------------- AC18-r2 / r3: guide placement

def grey_render(bg, fig, body_h, floor_y, centre_x=300, res=704):
    """A synthetic grey mannequin render: a torso `body_h` tall with its feet
    on `floor_y` and arms out, off-centre at `centre_x`, on flat grey `bg`."""
    im = Image.new("RGB", (res, res), bg)
    top = floor_y - body_h + 1
    im.paste(fig, (centre_x - 40, top, centre_x + 40, floor_y + 1))
    im.paste(fig, (centre_x - 150, top + 60, centre_x + 150, top + 90))
    return im


class TestGuidePlacement(GuidedFixture):
    """AC18-r2: a synthetic grey mannequin render for e places as
    `master/e.guide.png` — 832x1024 on chroma green, body 471.5 ± 1, floor
    725.9 ± 1, centre x 416 ± 1 — both through `stills.guide` and through a
    `go` run's real render path (Blender faked). AC18-r3: a figure only 6
    levels off its grey background pastes whole."""

    def test_ac18_r2_guide_from_a_grey_render_measures_the_e_target(self):
        with tempfile.TemporaryDirectory() as tmp:
            render = Path(tmp) / "render.png"
            # the stub e mannequin: 399 px, feet on 533 at 704
            grey_render((128, 128, 128), (90, 90, 90), 399, 533).save(render)
            dst = Path(tmp) / "e.guide.png"
            self.assertEqual(stills.guide(render, dst, 704, mannequins.profile(), "e"), dst)
            self.assert_on_target(dst, "e")
            arr = numpy.asarray(Image.open(dst).convert("RGB"))
            # every pixel is chroma green or the render's grey figure
            grey = (arr[..., 0] == arr[..., 1]) & (arr[..., 1] == arr[..., 2])
            green = (arr == GREEN).all(axis=2)
            self.assertTrue((grey | green).all())
            self.assertTrue(grey.any())

    def test_ac18_r2_go_writes_each_guide_at_its_direction_target(self):
        code, out = self.run_cli(["go", str(self.project)], fake=self.fake)
        self.assertEqual(code, 0, out)
        for d in EDIT_DIRS:
            self.assert_on_target(self.master / f"{d}.guide.png", d)

    def test_ac18_r3_a_figure_6_levels_off_pastes_whole(self):
        with tempfile.TemporaryDirectory() as tmp:
            render = Path(tmp) / "render.png"
            grey_render((120, 120, 120), (126, 126, 126), 399, 533).save(render)
            dst = Path(tmp) / "e.guide.png"
            stills.guide(render, dst, 704, mannequins.profile(), "e")
            self.assert_on_target(dst, "e")
            arr = numpy.asarray(Image.open(dst).convert("RGB"))
            green = (arr == GREEN).all(axis=2)
            # the torso column, placed: scale 471.5 / 399, centred on 416,
            # rows from the top of the figure to the floor; 2 px inside its
            # edges no pixel is left green
            m = metrics.measure(dst)
            half = round(40 * 471.5 / 399)
            top = m["floor_y"] - m["body_h"] + 1
            inner = green[top + 2:m["floor_y"] - 1, 416 - half + 2:416 + half - 2]
            self.assertGreater(inner.size, 0)
            self.assertFalse(inner.any(), "green pixels inside the silhouette")
            self.assertTrue((arr[top + 2:m["floor_y"] - 1, 416] == (126, 126, 126)).all())

    def test_ac18_r3_a_render_without_a_silhouette_is_a_figure_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            render = Path(tmp) / "flat.png"
            Image.new("RGB", (704, 704), (120, 120, 120)).save(render)
            with self.assertRaises(metrics.FigureError) as cm:
                stills.guide(render, Path(tmp) / "g.png", 704, mannequins.profile(), "e")
            self.assertIn(str(render), str(cm.exception))


# ------------------------------------------------------------ AC18-r4: the prompt

class TestGuideLine(unittest.TestCase):
    """AC18-r4: dir_prompt ends with the guide line in the pronoun."""

    def test_ac18_r4_dir_prompt_ends_with_the_guide_line(self):
        turn = {"e": "turned to face the right side of the frame"}
        for pronoun, line in GUIDE_LINE.items():
            with self.subTest(pronoun=pronoun):
                prompt = stills.dir_prompt(turn, "e", pronoun)
                self.assertTrue(prompt.endswith(
                    " Full body, plain flat pure green background, no shadow. " + line),
                    prompt)
        he = stills.dir_prompt(turn, "e", "he")
        self.assertIn("draw him", he)
        self.assertIn("his feet", he)
        self.assertIn("his arms", he)


# ------------------------------------------------------ AC18-r5: packaged edit.json

class TestPackagedEditWorkflow(unittest.TestCase):
    """AC18-r5 (FR 9): a LoadImage titled sp:pose_image -> ImageScaleToTotalPixels
    -> VAEEncode -> a ReferenceLatent on the positive and one on the negative
    conditioning, both feeding the CFGGuider."""

    def test_ac18_r5_pose_image_chain_feeds_both_conditionings(self):
        g = json.loads((ROOT / "src" / "sprite_pipeline" / "workflows" / "edit.json")
                       .read_text())
        titled = {(n.get("_meta") or {}).get("title"): k for k, n in g.items()}
        pose = titled["sp:pose_image"]
        self.assertEqual(g[pose]["class_type"], "LoadImage")
        self.assertIn("image", g[pose]["inputs"])
        scale = [k for k, n in g.items() if n["class_type"] == "ImageScaleToTotalPixels"
                 and n["inputs"]["image"] == [pose, 0]]
        self.assertEqual(len(scale), 1)
        enc = [k for k, n in g.items() if n["class_type"] == "VAEEncode"
               and n["inputs"]["pixels"] == [scale[0], 0]]
        self.assertEqual(len(enc), 1)
        refs = [k for k, n in g.items() if n["class_type"] == "ReferenceLatent"
                and n["inputs"]["latent"] == [enc[0], 0]]
        self.assertEqual(len(refs), 2)
        guider = next(n for n in g.values() if n["class_type"] == "CFGGuider")
        pos, neg = guider["inputs"]["positive"][0], guider["inputs"]["negative"][0]
        self.assertEqual(sorted([pos, neg]), sorted(refs))

        def source(k):
            # follow the conditioning chain back through ReferenceLatents
            while g[k]["class_type"] == "ReferenceLatent":
                k = g[k]["inputs"]["conditioning"][0]
            return k
        self.assertEqual(source(pos), titled["sp:prompt"])
        self.assertEqual(g[source(neg)]["class_type"], "ConditioningZeroOut")
        # each chain also carries the sp:ref_image reference
        ref_enc = next(k for k, n in g.items() if n["class_type"] == "VAEEncode"
                       and g[n["inputs"]["pixels"][0]]["inputs"]["image"]
                       == [titled["sp:ref_image"], 0])
        for k in (pos, neg):
            inner = g[k]["inputs"]["conditioning"][0]
            self.assertEqual(g[inner]["class_type"], "ReferenceLatent")
            self.assertEqual(g[inner]["inputs"]["latent"], [ref_enc, 0])


# ------------------------------------------------------------ AC18-r7: --dry-run

class TestDryRunPlan(GuidedFixture):
    """AC18-r7: `--dry-run` on an unapproved character prints, per guided
    direction, a guide, a direction edit and a re-anchor line, runs no
    subprocess (no Blender) and writes nothing."""

    def test_ac18_r7_dry_run_prints_guide_edit_reanchor_per_direction(self):
        fake = FakeRun()
        code, out = self.run_cli(["go", str(self.project), "--dry-run"], fake=fake)
        self.assertEqual(code, 0, out)
        self.assertEqual(fake.calls, [])
        lines = out.splitlines()
        for d in EDIT_DIRS:
            mine = [l for l in lines if l.startswith(f"[courier/{d}] ")]
            kinds = [l.split("] ", 1)[1].split(" ")[0] for l in mine]
            self.assertEqual(kinds, ["guide", "direction", "re-anchor"], mine)
            self.assertIn(str(self.master / f"{d}.guide.png"), mine[0])
            self.assertIn("direction edit (edit)", mine[1])
            self.assertIn(str(self.master / f"{d}.edit.png"), mine[2])
            self.assertIn(str(self.master / f"{d}.png"), mine[2])
        self.assertFalse([l for l in lines if l.startswith("[courier/s] ")])
        self.assertFalse(self.out_char.exists())
        self.assertFalse((self.home / "motion-cache").exists())


# --------------------------------------------------------- AC18-r8: render cache

class TestGuideRenderCache(GuidedFixture):
    """AC18-r8: once the guide render for a GLB and direction is in
    motion_cache, a later run (here: the character rebuilt from scratch)
    makes no second Blender render for it."""

    def test_ac18_r8_a_cached_render_skips_blender(self):
        code, out = self.run_cli(["go", str(self.project)], fake=self.fake)
        self.assertEqual(code, 0, out)
        self.assertEqual(len(self.renders(self.fake)), 4)
        first = (self.master / "e.guide.png").read_bytes()
        import shutil
        shutil.rmtree(self.out_char)
        fake = self._fresh_fake()
        code, out = self.run_cli(["go", str(self.project)], fake=fake)
        self.assert_stopped_with_approve_command(code, out)
        self.assertEqual(self.renders(fake), [])
        self.assertEqual([c for c in fake.calls if "render_drive.py" in " ".join(c)], [])
        self.assertEqual(len(fake.edit_calls()), 4)
        self.assertEqual((self.master / "e.guide.png").read_bytes(), first)


# ------------------------------------------------------------ AC19-r1: re-anchor

class TestReanchor(GuidedFixture):
    """AC19-r1: the edit returns a still at 1.30x its direction's target,
    feet 60 px low, centre x 440: `master/<d>.edit.png` is byte-identical
    to it and `master/<d>.png` sits on the direction's target, floor and
    centre. AC19-r2: a raw edit wider than tall is re-anchored, not
    rejected."""

    def test_ac19_r1_raw_edit_kept_and_still_reanchored(self):
        code, out = self.run_cli(["go", str(self.project)], fake=self.fake)
        self.assert_stopped_with_approve_command(code, out)
        for d in EDIT_DIRS:
            raw = self.master / f"{d}.edit.png"
            self.assertEqual(raw.read_bytes(), self.written[d], d)
            # the input really is off target
            m = metrics.measure(raw)
            self.assertAlmostEqual(m["body_h"], 1.30 * TARGET[d][0], delta=1)
            self.assertEqual(m["centre_x"], 440 - 0.5)
            self.assert_on_target(self.master / f"{d}.png", d)

    def test_ac19_r2_a_raw_edit_wider_than_tall_is_reanchored(self):
        self.raw = {d: t_pose(d) for d in EDIT_DIRS}
        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp) / "ne.edit.png"
            self.raw["ne"].save(src)
            with self.assertRaises(metrics.FigureError) as cm:
                metrics.check_standing(src)  # the input breaks the FR 6 rule
            self.assertEqual(cm.exception.reason, "too wide")
            dst = Path(tmp) / "ne.png"
            stills.reanchor(src, dst, 704, mannequins.profile(), "ne")
            self.assert_on_target(dst, "ne")
        code, out = self.run_cli(["go", str(self.project)], fake=self.fake)
        self.assert_stopped_with_approve_command(code, out)
        self.assertNotIn("too wide", out)
        for d in EDIT_DIRS:
            self.assert_on_target(self.master / f"{d}.png", d)


# ------------------------------------------------------ AC19-r3: direction failure

class TestDirectionFailure(GuidedFixture):
    """AC19-r3: a raw edit with no figure makes `go` exit 1 naming
    `master/<d>.edit.png`, and the other directions still run. A failed
    guide render is that direction's failure the same way."""

    def test_ac19_r3_flat_raw_edit_fails_its_direction_only(self):
        self.raw["e"] = Image.new("RGB", (832, 1024), EDIT_BG)
        code, out = self.run_cli(["go", str(self.project)], fake=self.fake)
        self.assertEqual(code, 1, out)
        self.assertIn(str(self.master / "e.edit.png"), out)
        self.assertIn("FAILED", out)
        self.assertFalse((self.master / "e.png").exists())
        self.assertEqual(len(self.fake.edit_calls()), 4)
        for d in ("se", "ne", "n"):  # before and after e, in generate order
            self.assert_on_target(self.master / f"{d}.png", d)

    def test_ac19_r3_failed_guide_render_fails_its_direction_only(self):
        def no_ne(argv):
            if argv[argv.index("--") + 4] != "135":
                self._produce_frames(argv)
        fake = FakeRun(after={"sprite_pipeline.comfy edit": self._make_still,
                              "cutout": self._produce_still,
                              "render_drive.py": no_ne,
                              "ffmpeg": self._produce_drive})
        code, out = self.run_cli(["go", str(self.project)], fake=fake)
        self.assertEqual(code, 1, out)
        self.assertIn(str(self.master / "ne.guide.png"), out)
        self.assertIn("ual.glb", out)
        self.assertFalse((self.master / "ne.png").exists())
        edited = sorted(Path(e[e.index("--copy-to") + 1]).name for e in fake.edit_calls())
        self.assertEqual(edited, ["e.edit.png", "n.edit.png", "se.edit.png"])
        for d in ("se", "e", "n"):
            self.assert_on_target(self.master / f"{d}.png", d)


# ------------------------------------------------------- AC19-r4 / r5: what reruns

class TestRerun(GuidedFixture):
    """AC19-r4: `--reroll courier:ne=3` on a built character edits only ne
    (seed 3) and re-anchors it. AC19-r5: with `master/e.edit.png` present
    but `master/e.png` missing, `go` re-runs e's edit and re-anchor."""

    def build(self):
        code, out = self.run_cli(["go", str(self.project)], fake=self.fake)
        self.assert_stopped_with_approve_command(code, out)
        code, out = self.run_cli(self.approve_cmd())
        self.assertEqual(code, 0, out)
        return {d: (self.master / f"{d}.png").read_bytes() for d in EDIT_DIRS}

    def test_ac19_r4_reroll_redoes_only_that_directions_edit_and_reanchor(self):
        before = self.build()
        self.raw["ne"] = raw_edit("ne", ratio=0.80, feet_low=-30, centre_x=380)
        fake = self._fresh_fake()
        code, out = self.run_cli(["go", str(self.project), "--reroll", "courier:ne=3"],
                                 fake=fake)
        self.assert_stopped_with_approve_command(code, out)
        edits = fake.edit_calls()
        self.assertEqual(len(edits), 1, "\n".join(" ".join(c) for c in fake.calls))
        self.assertEqual(edits[0][edits[0].index("--copy-to") + 1],
                         str(self.master / "ne.edit.png"))
        self.assertEqual(edits[0][edits[0].index("--seed") + 1], "3")
        self.assertEqual((self.master / "ne.edit.png").read_bytes(), self.written["ne"])
        self.assertNotEqual((self.master / "ne.png").read_bytes(), before["ne"])
        self.assert_on_target(self.master / "ne.png", "ne")
        for d in ("se", "e", "n"):
            self.assertEqual((self.master / f"{d}.png").read_bytes(), before[d], d)

    def test_ac19_r5_missing_still_with_a_raw_edit_reruns_edit_and_reanchor(self):
        self.build()
        (self.master / "e.png").unlink()
        self.assertTrue((self.master / "e.edit.png").is_file())
        fake = self._fresh_fake()
        code, out = self.run_cli(["go", str(self.project)], fake=fake)
        self.assert_stopped_with_approve_command(code, out)
        edits = fake.edit_calls()
        self.assertEqual([e[e.index("--copy-to") + 1] for e in edits],
                         [str(self.master / "e.edit.png")])
        self.assert_on_target(self.master / "e.png", "e")


if __name__ == "__main__":
    unittest.main()
