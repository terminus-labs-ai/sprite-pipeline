"""Tests for the mannequin-guided direction stills (spec AC15, AC18, AC19;
FR 6, FR 7, FR 9, amendments 10 and 12).

Per guided direction — `s` first (amendment 12) — `go` renders the anchor
GLB's A_TPose facing it (Blender faked at subprocess.run by
`mannequins.fake_blender`, cached in the configured motion_cache), places it
as `master/<d>.guide.png`, runs the edit with `--image` the anchored still
`master/still.png` for `s` and the guided master `master/s.png` for every
other direction, `--pose-image master/<d>.guide.png` and the guide line
appended to the prompt (ComfyUI faked at subprocess.run), keeps the raw
output as `master/<d>.edit.png` and re-anchors it to `master/<d>.png`
without the standing-figure rule. The guide's silhouette includes every
region it fully encloses (amendment 11): no chroma green is left inside the
mannequin.

The expected targets are stated literally from the stub profile (the female
values in tests/mannequins.py) on the 832x1024 canvas at frame size 704:
body / floor s 391.2 / 715.3, se 433.7 / 724.7, e 471.5 / 725.9, ne 418.4 /
727.1, n 361.6 / 715.3; centre x 416; chroma green (31, 203, 25).
"""
import io
import json
import tempfile
import unittest
from collections import deque
from pathlib import Path

import numpy
from PIL import Image

import mannequins
from prompt_text import GUIDE_LINE
from test_review import EDIT_DIRS, FakeRun, ReviewFixture

from sprite_pipeline import metrics, stills

ROOT = Path(__file__).resolve().parent.parent
GREEN = (31, 203, 25)
# body / floor on the 832x1024 canvas at 704, per direction (case table):
# dirs.<d>.body_h x 832 / 704 and dirs.<d>.floor_y x 832 / 704 + 96 over the
# stub profile (mannequins.DIRS)
TARGET = {"s": (331 * 832 / 704, 524 * 832 / 704 + 96),
          "se": (391.2, 724.7), "e": (384.4, 725.9), "ne": (375.6, 727.1),
          "n": (361.6, 715.3)}
ALL_DIRS = ["s", "se", "e", "ne", "n"]
CENTRE_X = 416
EDIT_BG = (10, 200, 10)


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
        self.raw = {d: raw_edit(d) for d in ALL_DIRS}
        self.written = {}

    def build_all(self, fake=None):
        """The amendment 12 two-stop build: `go` edits `s` and stops;
        `approve` writes s.approved; the next `go` edits the rest and stops
        at the full review stop. Returns the last run's (code, out)."""
        fake = fake or self.fake
        code, out = self.run_cli(["go", str(self.project)], fake=fake)
        self.assert_stopped_with_approve_command(code, out)
        code, out = self.run_cli(self.approve_cmd())
        self.assertEqual(code, 0, out)
        code, out = self.run_cli(["go", str(self.project)], fake=fake)
        self.assert_stopped_with_approve_command(code, out)
        return code, out

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
    """AC18-r1/r11/r12 (amendment 12): `go --no-stop` edits `s` first with
    `--image master/still.png --pose-image master/s.guide.png`, then each
    other generated direction with `--image master/s.png --pose-image
    master/<d>.guide.png`."""

    def test_ac18_r1_each_edit_gets_its_reference_and_its_guide(self):
        # --stage keeps the assertion on the still work (the faked pipeline
        # cannot satisfy export)
        code, out = self.run_cli(["go", str(self.project), "--no-stop",
                                  "--stage", "pixel"], fake=self.fake)
        self.assertEqual(code, 0, out)
        edits = self.fake.edit_calls()
        by_dir = {Path(e[e.index("--copy-to") + 1]).name: e for e in edits}
        self.assertEqual(sorted(by_dir), sorted(f"{d}.edit.png" for d in ALL_DIRS))
        self.assertEqual(len(edits), 5)
        # s first, edited from the anchored still (AC18-r11)
        first = edits[0]
        self.assertEqual(first[first.index("--copy-to") + 1],
                         str(self.master / "s.edit.png"))
        self.assertEqual(first[first.index("--image") + 1],
                         str(self.master / "still.png"))
        self.assertEqual(first[first.index("--pose-image") + 1],
                         str(self.master / "s.guide.png"))
        # every other direction edited from the guided master (AC18-r12)
        for d in EDIT_DIRS:
            e = by_dir[f"{d}.edit.png"]
            self.assertEqual(e[e.index("--image") + 1], str(self.master / "s.png"))
            self.assertEqual(e[e.index("--pose-image") + 1],
                             str(self.master / f"{d}.guide.png"))
            self.assertEqual(e[e.index("--copy-to") + 1],
                             str(self.master / f"{d}.edit.png"))
        for d in ALL_DIRS:
            self.assertTrue((self.master / f"{d}.guide.png").is_file(), d)
        # one A_TPose render per direction, s included (facing 0)
        self.assertEqual(sorted(self.renders(self.fake)),
                         ["0", "135", "180", "45", "90"])


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
    `master/e.guide.png` — 832x1024 on chroma green, body 384.4 ± 1, floor
    725.9 ± 1, centre x 416 ± 1 — both through `stills.guide` and through a
    `go` run's real render path (Blender faked). AC18-r3: a figure only 6
    levels off its grey background pastes whole."""

    def test_ac18_r2_guide_from_a_grey_render_measures_the_e_target(self):
        with tempfile.TemporaryDirectory() as tmp:
            render = Path(tmp) / "render.png"
            # the stub e mannequin: 325 px, feet on 533 at 704
            grey_render((128, 128, 128), (90, 90, 90), 325, 533).save(render)
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
        self.build_all()
        for d in ALL_DIRS:
            self.assert_on_target(self.master / f"{d}.guide.png", d)

    def test_ac18_r3_a_figure_6_levels_off_pastes_whole(self):
        with tempfile.TemporaryDirectory() as tmp:
            render = Path(tmp) / "render.png"
            grey_render((120, 120, 120), (126, 126, 126), 325, 533).save(render)
            dst = Path(tmp) / "e.guide.png"
            stills.guide(render, dst, 704, mannequins.profile(), "e")
            self.assert_on_target(dst, "e")
            arr = numpy.asarray(Image.open(dst).convert("RGB"))
            green = (arr == GREEN).all(axis=2)
            # the torso column, placed: scale 384.4 / 399, centred on 416,
            # rows from the top of the figure to the floor; 2 px inside its
            # edges no pixel is left green
            m = metrics.measure(dst)
            half = round(40 * 384.4 / 399)
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


# ------------------------------------------- AC18-r9 / r10: enclosed holes filled

FIXTURES = ROOT / "tests" / "fixtures"
# FR 14's packaged (body_h, floor_y) per GLB and direction at 704, stated
# independently of the packaged file
FR14 = {
    "ual-female.glb": {"s": (331, 524), "se": (331, 532), "e": (325, 533),
                       "ne": (318, 534), "n": (306, 524)},
    "ual.glb": {"s": (354, 536), "se": (350, 539), "e": (328, 528),
                "ne": (320, 526), "n": (308, 518)},
}
FIXTURE_GLB = {"female": "ual-female.glb", "male": "ual.glb"}


def enclosed_green(arr):
    """The case table's oracle: the number of chroma-green pixels (within 40
    of GREEN in every channel) not connected to the canvas border through
    such pixels (4-connected breadth-first flood fill from the border)."""
    green = (numpy.abs(arr.astype(numpy.int16) - numpy.array(GREEN)) <= 40).all(axis=2)
    h, w = green.shape
    seen = numpy.zeros_like(green)
    queue = deque()
    border = ([(y, x) for y in range(h) for x in (0, w - 1)]
              + [(y, x) for x in range(w) for y in (0, h - 1)])
    for y, x in border:
        if green[y, x] and not seen[y, x]:
            seen[y, x] = True
            queue.append((y, x))
    while queue:
        y, x = queue.popleft()
        for yy, xx in ((y + 1, x), (y - 1, x), (y, x + 1), (y, x - 1)):
            if 0 <= yy < h and 0 <= xx < w and green[yy, xx] and not seen[yy, xx]:
                seen[yy, xx] = True
                queue.append((yy, xx))
    return int((green & ~seen).sum())


class TestGuideHolesFilled(unittest.TestCase):
    """AC18-r9: a guide built from each committed fixture render with the
    packaged profile for its GLB and direction (frame size 704) has no
    enclosed chroma-green pixel and sits on that direction's target, floor
    and centre x 416 +/- 1. AC18-r10: a region inside the figure exactly
    equal to the background colour is pasted as figure (amendment 11)."""

    def test_ac18_r9_fixture_guides_have_no_enclosed_green(self):
        pkg = json.loads(metrics.PACKAGED_PROFILES.read_text())["profiles"]
        with tempfile.TemporaryDirectory() as tmp:
            for sex, glb in FIXTURE_GLB.items():
                profile = next(p for p in pkg.values() if p["glb_name"] == glb)
                for d, (body_h, floor_y) in FR14[glb].items():
                    with self.subTest(glb=glb, d=d):
                        render = FIXTURES / f"mannequin_apose_{sex}_{d}_704.png"
                        dst = Path(tmp) / f"{sex}_{d}.guide.png"
                        stills.guide(render, dst, 704, profile, d)
                        arr = numpy.asarray(Image.open(dst).convert("RGB"))
                        self.assertEqual(arr.shape, (1024, 832, 3))
                        self.assertEqual(enclosed_green(arr), 0)
                        # FR 6 targets at 704: body x 832 / 704, floor x 832 / 704 + 96
                        m = metrics.measure(dst)
                        self.assertAlmostEqual(m["body_h"], body_h * 832 / 704, delta=1)
                        self.assertAlmostEqual(m["floor_y"], floor_y * 832 / 704 + 96,
                                               delta=1)
                        self.assertAlmostEqual(m["centre_x"], CENTRE_X, delta=1)

    def test_ac18_r10_background_coloured_patch_inside_pastes_as_figure(self):
        bg, fig = (120, 120, 120), (126, 126, 126)
        with tempfile.TemporaryDirectory() as tmp:
            render = Path(tmp) / "render.png"
            # the stub e mannequin (399 px, feet on 533, torso x 260-339)
            # with a 60x60 patch of exactly the background colour inside the
            # torso, below the arms' band (rows 195-224): source x 270-330,
            # rows 250-310
            im = grey_render(bg, fig, 399, 533)
            im.paste(bg, (270, 250, 330, 310))
            im.save(render)
            dst = Path(tmp) / "e.guide.png"
            stills.guide(render, dst, 704, mannequins.profile(), "e")
            arr = numpy.asarray(Image.open(dst).convert("RGB"))
            self.assertEqual(enclosed_green(arr), 0)
            # the patch, placed: scale 384.4 / 399 about the torso's centre x
            # (299.5 -> 416) and feet row (533 -> the guide's floor)
            m = metrics.measure(dst)
            s = TARGET["e"][0] / 399
            x0, x1 = round(416 + (270 - 299.5) * s), round(416 + (330 - 299.5) * s)
            y0 = round(m["floor_y"] - (533 - 250) * s)
            y1 = round(m["floor_y"] - (533 - 310) * s)
            patch = arr[y0:y1, x0:x1]
            self.assertGreater(patch.shape[0] * patch.shape[1], 30 * 30)
            self.assertFalse((patch == GREEN).all(axis=2).any(),
                             "green pixels in the background-coloured patch")
            # its interior holds the render's own pixels: the background grey
            inner = arr[y0 + 5:y1 - 5, x0 + 5:x1 - 5]
            self.assertGreater(inner.size, 0)
            self.assertTrue((inner == bg).all(), "patch interior is not render pixels")


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
    """AC18-r7/r14 (amendment 12): `--dry-run` on an unapproved character
    prints, per direction — `s` first, its edit naming `still.png` — a
    guide, a direction edit and a re-anchor line (the `s` block ends with
    the palette cutout), runs no subprocess (no Blender) and writes
    nothing."""

    def test_ac18_r7_dry_run_prints_guide_edit_reanchor_per_direction(self):
        fake = FakeRun()
        code, out = self.run_cli(["go", str(self.project), "--dry-run"], fake=fake)
        self.assertEqual(code, 0, out)
        self.assertEqual(fake.calls, [])
        lines = out.splitlines()
        mine = [l for l in lines if l.startswith("[courier/s] ")]
        kinds = [l.split("] ", 1)[1].split(" ")[0] for l in mine]
        self.assertEqual(kinds, ["guide", "direction", "re-anchor", "palette"], mine)
        self.assertIn(str(self.master / "s.guide.png"), mine[0])
        self.assertIn("direction edit (edit)", mine[1])
        self.assertIn("still.png", mine[1])  # the s edit's input, not s.png
        self.assertIn(str(self.master / "s.edit.png"), mine[2])
        self.assertIn(str(self.master / "s.png"), mine[2])
        self.assertIn(str(self.master / "s.keyed.png"), mine[3])
        for d in EDIT_DIRS:
            mine = [l for l in lines if l.startswith(f"[courier/{d}] ")]
            kinds = [l.split("] ", 1)[1].split(" ")[0] for l in mine]
            self.assertEqual(kinds, ["guide", "direction", "re-anchor"], mine)
            self.assertIn(str(self.master / f"{d}.guide.png"), mine[0])
            self.assertIn("direction edit (edit)", mine[1])
            self.assertIn(str(self.master / f"{d}.edit.png"), mine[2])
            self.assertIn(str(self.master / f"{d}.png"), mine[2])
        # the s block precedes every other direction's lines
        s_first = lines.index(next(l for l in lines if l.startswith("[courier/s] ")))
        others_first = min(lines.index(l) for d in EDIT_DIRS for l in lines
                           if l.startswith(f"[courier/{d}] "))
        self.assertLess(s_first, others_first)
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
        self.assertEqual(self.renders(self.fake), ["0"])  # the s render only
        code, out = self.run_cli(self.approve_cmd())
        self.assertEqual(code, 0, out)
        code, out = self.run_cli(["go", str(self.project)], fake=self.fake)
        self.assertEqual(code, 0, out)
        self.assertEqual(sorted(self.renders(self.fake)),
                         ["0", "135", "180", "45", "90"])
        first = (self.master / "s.guide.png").read_bytes()
        import shutil
        shutil.rmtree(self.out_char)
        fake = self._fresh_fake()
        code, out = self.run_cli(["go", str(self.project)], fake=fake)
        self.assert_stopped_with_approve_command(code, out)
        self.assertEqual(self.renders(fake), [])
        self.assertEqual([c for c in fake.calls if "render_drive.py" in " ".join(c)], [])
        self.assertEqual(len(fake.edit_calls()), 1)  # the s stop edits only s
        self.assertEqual((self.master / "s.guide.png").read_bytes(), first)


# ------------------------------------------------------------ AC19-r1: re-anchor

class TestReanchor(GuidedFixture):
    """AC19-r1: the edit returns a still at 1.30x its direction's target,
    feet 60 px low, centre x 440: `master/<d>.edit.png` is byte-identical
    to it and `master/<d>.png` sits on the direction's target, floor and
    centre. AC19-r2: a raw edit wider than tall is re-anchored, not
    rejected."""

    def test_ac19_r1_raw_edit_kept_and_still_reanchored(self):
        self.build_all()
        for d in ALL_DIRS:
            raw = self.master / f"{d}.edit.png"
            self.assertEqual(raw.read_bytes(), self.written[d], d)
            # the input really is off target
            m = metrics.measure(raw)
            self.assertAlmostEqual(m["body_h"], 1.30 * TARGET[d][0], delta=1)
            self.assertEqual(m["centre_x"], 440 - 0.5)
            self.assert_on_target(self.master / f"{d}.png", d)

    def test_ac19_r2_a_raw_edit_wider_than_tall_is_reanchored(self):
        self.raw = {d: t_pose(d) for d in EDIT_DIRS} | {"s": self.raw["s"]}
        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp) / "ne.edit.png"
            self.raw["ne"].save(src)
            with self.assertRaises(metrics.FigureError) as cm:
                metrics.check_standing(src)  # the input breaks the FR 6 rule
            self.assertEqual(cm.exception.reason, "too wide")
            dst = Path(tmp) / "ne.png"
            stills.reanchor(src, dst, 704, mannequins.profile(), "ne")
            self.assert_on_target(dst, "ne")
        self.build_all()
        for d in ALL_DIRS:
            self.assert_on_target(self.master / f"{d}.png", d)


# ------------------------------------------------------ AC19-r3: direction failure

class TestDirectionFailure(GuidedFixture):
    """AC19-r3: a raw edit with no figure makes `go` exit 1 naming
    `master/<d>.edit.png`, and the other directions still run. A failed
    guide render is that direction's failure the same way."""

    def _approve_s(self):
        """Pass the s stop: `go` edits s, `approve` writes s.approved."""
        code, out = self.run_cli(["go", str(self.project)], fake=self.fake)
        self.assert_stopped_with_approve_command(code, out)
        code, out = self.run_cli(self.approve_cmd())
        self.assertEqual(code, 0, out)

    def test_ac19_r3_flat_raw_edit_fails_its_direction_only(self):
        self._approve_s()
        self.raw["e"] = Image.new("RGB", (832, 1024), EDIT_BG)
        fake = self._fresh_fake()
        code, out = self.run_cli(["go", str(self.project)], fake=fake)
        self.assertEqual(code, 1, out)
        self.assertIn(str(self.master / "e.edit.png"), out)
        self.assertIn("FAILED", out)
        self.assertFalse((self.master / "e.png").exists())
        self.assertEqual(len(fake.edit_calls()), 4)
        for d in ("se", "ne", "n"):  # before and after e, in generate order
            self.assert_on_target(self.master / f"{d}.png", d)

    def test_ac19_r3_failed_guide_render_fails_its_direction_only(self):
        self._approve_s()
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
        # sprite-pipeline-51: the stage-failure line names the guide file and
        # the source GLB — it carries no mannequin-profile command prefix
        self.assertNotIn("mannequin-profile", out)
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
        # the full two-stop build (amendment 12): go -> approve -> go ->
        # approve, ending with the whole set approved
        code, out = self.run_cli(["go", str(self.project)], fake=self.fake)
        self.assert_stopped_with_approve_command(code, out)
        code, out = self.run_cli(self.approve_cmd())
        self.assertEqual(code, 0, out)
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


# ------------------------------------- AC15-r1 / r2 (amendment 12): anchored still

class TestAnchoredStillIsTheSEditInput(GuidedFixture):
    """AC15-r1: the `s` edit's `--image` is `master/still.png`, the anchored
    still measuring body 391, floor 715, centre x 416 at frame size 704.
    AC15-r2: the cutout reads the re-anchored `master/s.png` (not the
    anchored still) and writes `master/s.keyed.png`."""

    def test_ac15_r1_s_edit_reads_the_anchored_still(self):
        code, out = self.run_cli(["go", str(self.project)], fake=self.fake)
        self.assert_stopped_with_approve_command(code, out)
        edits = self.fake.edit_calls()
        self.assertEqual(len(edits), 1)
        anchored = self.master / "still.png"
        self.assertEqual(edits[0][edits[0].index("--image") + 1], str(anchored))
        self.assert_on_target(anchored, "s")  # 391.2 / 715.3, centre x 416

    def test_ac15_r2_cutout_reads_the_reanchored_s(self):
        read = []

        def spy_cutout(argv):
            read.append(Path(argv[argv.index("cutout") + 1]).read_bytes())
            self._produce_still(argv)
        fake = FakeRun(after={"sprite_pipeline.comfy edit": self._make_still,
                              "cutout": spy_cutout,
                              "render_drive.py": self._produce_frames,
                              "ffmpeg": self._produce_drive})
        code, out = self.run_cli(["go", str(self.project)], fake=fake)
        self.assert_stopped_with_approve_command(code, out)
        cutouts = [c for c in fake.calls if Path(c[0]).name == "sprite-gen"]
        self.assertEqual(len(cutouts), 1)
        argv = cutouts[0]
        self.assertEqual(argv[argv.index("cutout") + 1],
                         str(self.master / "s.png"))
        self.assertEqual(argv[argv.index("--out") + 1],
                         str(self.master / "s.keyed.png"))
        # the bytes the cutout read are the re-anchored s, which only exists
        # after the s re-anchor — never the anchored still's
        self.assertEqual(len(read), 1)
        self.assertEqual(read[0], (self.master / "s.png").read_bytes())
        self.assertNotEqual(read[0], (self.master / "still.png").read_bytes())


# ------------------------------------------ AC18-r13 (amendment 12): the framing

class TestMannequinCameraFraming(unittest.TestCase):
    """AC18-r13: dir_prompt for every direction opens with the amendment 12
    mannequin-camera sentence, holds no `relaxed pose`, and ends with the
    guide line in the pronoun."""

    def test_ac18_r13_every_direction_uses_the_mannequin_camera_framing(self):
        turn = {d: f"turn {d} {{{'obj'}}}" for d in
                ("s", "se", "e", "ne", "n")}
        for d in turn:
            with self.subTest(d=d):
                p = stills.dir_prompt(turn, d, "she")
                self.assertTrue(p.startswith(
                    "The exact same character with the exact same outfit, colors, "
                    "proportions and art style, seen from the same high isometric "
                    "camera angle as the mannequin in image 2, looking down at her "
                    "at about 45 degrees, but now "), p)
                self.assertNotIn("relaxed pose", p)
                self.assertTrue(p.endswith(GUIDE_LINE["she"]), p)
        he = stills.dir_prompt(turn, "n", "he")
        self.assertIn("looking down at him at about 45 degrees", he)
        self.assertTrue(he.endswith(GUIDE_LINE["he"]), he)


if __name__ == "__main__":
    unittest.main()
