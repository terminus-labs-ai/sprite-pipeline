"""Per-animation canvas: the auto canvas of death-repro FR 4 (AC 3, case rows
AC3-r1..r5), sized from the union of the scaled mannequin over every frame of
every direction of the animation, plus the reference padding the animate
stage keeps until slice 3.

No network, GPU, Blender or ComfyUI: plans are inspected, images are small
PNGs written by the test.
"""
import tempfile
import unittest
from pathlib import Path

import numpy
from PIL import Image

import mannequins
from test_workflow_paths import cfg, stubbed_example

from sprite_pipeline import config, metrics, project, stages, stills


class TestAC3CanvasFor(unittest.TestCase):
    """The pure FR 4 rule: H = ceil16(margin x union h), W = ceil16(max(margin
    x union w, H x aspect)), then H = max(H, ceil16(W x 9 / 16)). Expected
    values are worked by hand."""

    def test_r1_tall_union_pads_width_to_16_9(self):
        # H = ceil16(390) = 400; W = ceil16(max(260, 711.1)) = 720;
        # H = max(400, ceil16(405)) = 416
        self.assertEqual(stages.canvas_for((200, 300), 1.3, (16, 9)), (720, 416))

    def test_r2_measured_sp64_unions_at_0_605(self):
        """The prototype's c1152 drives (sp62/dirs/rgba-drives/c1152-<d>):
        raw union px at 1152, scaled by round(1152 x 0.605) / 1152 = 697 /
        1152."""
        k = 697 / 1152
        raw = {"se": (490, 389), "s": (244, 396), "e": (499, 383),
               "ne": (454, 541), "n": (244, 563)}
        want = {"se": (576, 336), "s": (576, 336), "e": (544, 320),
                "ne": (768, 432), "n": (800, 464)}
        for d, (w, h) in raw.items():
            with self.subTest(d=d):
                self.assertEqual(stages.canvas_for((w * k, h * k), 1.3, (16, 9)), want[d])

    def test_r3_wide_union_width_follows_the_margin(self):
        # H = ceil16(130) = 144; W = ceil16(max(1300, 256)) = 1312;
        # H = max(144, ceil16(738)) = 752
        self.assertEqual(stages.canvas_for((1000, 100), 1.3, (16, 9)), (1312, 752))


# AC3-r4/r5 drives: 200 px square transparent frames, one opaque block each,
# at drive_scale 0.5 (k = 100 / 200). Single-direction unions differ; the
# union over s, se, e is x 40-160, y 0-200 -> (60, 100) scaled.
BLOCKS = {"s": ((90, 20), (20, 160)), "se": ((40, 100), (120, 80)),
          "e": ((60, 0), (40, 200))}


def _with_death(path, extra):
    text = path.read_text().replace("- {name: death, source:", f"- {{name: death, {extra}, source:", 1)
    path.write_text(text)
    return path


class DriveCase(unittest.TestCase):
    def setUp(self):
        self._t = tempfile.TemporaryDirectory()
        self.addCleanup(self._t.cleanup)
        self.tmp = Path(self._t.name)
        self.path, self.base = stubbed_example(self._t.name)
        _with_death(self.path, "directions: [s, se, e], drive_scale: 0.5")

    def ctx(self):
        p = project.load_project(self.path)
        return stages.StageCtx(p, cfg(self.tmp / "home")), p

    def render(self, ctx, anim, dirs=BLOCKS):
        """The fake drive render: valid Pillow PNGs in each direction's frames dir."""
        for d, (at, size) in dirs.items():
            frames = ctx.frames_dir("courier", anim, d)
            frames.mkdir(parents=True, exist_ok=True)
            im = Image.new("RGBA", (200, 200), (0, 0, 0, 0))
            im.paste(Image.new("RGBA", size, (255, 0, 0, 255)), at)
            im.save(frames / "f0000.png")


class TestAC3OneCanvasPerAnimation(DriveCase):
    def test_r4_all_directions_share_the_union_canvas(self):
        ctx, p = self.ctx()
        death = p.animation("death")
        # each direction alone: s (10, 80) -> (208, 128); se (60, 40) -> (128, 80)
        self.assertEqual(stages.canvas_for((10, 80), 1.3, (16, 9)), (208, 128))
        self.render(ctx, death)
        # the union (60, 100): H = ceil16(130) = 144, W = ceil16(max(78, 256)) = 256
        self.assertEqual(ctx.canvas("courier", death), (256, 144))
        for d in BLOCKS:
            with self.subTest(d=d):
                step = next(a for a, _ in ctx.plan_pair("courier", death, d, "drive")
                            if a[0] == "composite_drive")
                dst = self.tmp / "out" / d
                stages.composite_drive(ctx.frames_dir("courier", death, d), dst,
                                       step[3], step[4:7], step[7], step[8:10], step[10:])
                self.assertEqual(Image.open(dst / "f0000.png").size, (256, 144))
                argv = ctx.plan_pair("courier", death, d, "animate")[-1][0]
                self.assertEqual(argv[argv.index("--w") + 1:argv.index("--h") + 2],
                                 ["256", "--h", "144"])

    def test_r4_the_drive_plan_renders_the_directions_still_missing(self):
        ctx, p = self.ctx()
        death = p.animation("death")
        steps = ctx.plan_pair("courier", death, "se", "drive")
        renders = [a for a, label in steps if label == "blender"]
        self.assertEqual([r[r.index("--") + 3] for r in renders],
                         [str(ctx.frames_dir("courier", death, d)) for d in ("se", "s", "e")])
        self.assertEqual([r[r.index("--") + 4] for r in renders], ["45", "0", "90"])
        self.render(ctx, death, {"s": BLOCKS["s"], "e": BLOCKS["e"]})
        steps = ctx.plan_pair("courier", death, "se", "drive")
        self.assertEqual([label for _, label in steps].count("blender"), 1)

    def test_r5_margin_and_aspect_drive_the_canvas_and_the_key(self):
        ctx, p = self.ctx()
        k0 = p.drive_key(p.animation("death"), b"x")
        text = self.path.read_text().replace("\nanimations:", "\ndrive: {margin: 1.5, aspect: [4, 3]}\nanimations:", 1)
        self.path.write_text(text)
        ctx, p = self.ctx()
        death = p.animation("death")
        self.assertEqual((p.drive.margin, p.drive.aspect), (1.5, (4, 3)))
        self.render(ctx, death)
        # union (60, 100): H = ceil16(150) = 160, W = ceil16(max(90, 213.3)) = 224,
        # H = max(160, ceil16(168)) = 176
        self.assertEqual(ctx.canvas("courier", death), (224, 176))
        self.assertNotEqual(p.drive_key(death, b"x"), k0)
        for block in ("{margin: 1.5}", "{aspect: [4, 3]}"):
            with self.subTest(block=block):
                self.path.write_text(text.replace("drive: {margin: 1.5, aspect: [4, 3]}", f"drive: {block}"))
                q = project.load_project(self.path)
                self.assertNotEqual(q.drive_key(q.animation("death"), b"x"), k0)


def _bbox(path):
    """(x0, y0, x1, y1) inclusive of the pixels differing from the top-left
    colour by more than 40 in any channel (FR 6's figure mask)."""
    arr = numpy.asarray(Image.open(path).convert("RGB"), dtype=numpy.int16)
    ys, xs = numpy.nonzero((abs(arr - arr[0, 0]) > 40).any(axis=2))
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


class TestPadStill(unittest.TestCase):
    def test_pad_keeps_pixel_size_and_centres_on_key_colour(self):
        with tempfile.TemporaryDirectory() as t:
            src, dst = Path(t) / "s.png", Path(t) / "out" / "p.png"
            im = Image.new("RGB", (40, 50), (0, 200, 0))
            im.putpixel((20, 25), (255, 0, 0))
            im.save(src)
            stages.pad_ref(str(src), str(dst), 20, 32, [])
            out = Image.open(dst).convert("RGB")
            self.assertEqual(out.size, (32, 32))
            self.assertEqual(out.getpixel((0, 0)), (0, 200, 0))
            # amendment 9: the still is centre-cropped to a square (the 50 px
            # side keeps rows 5-44, as ImageScale crop=center does), scaled to
            # frame x frame, then centred: its centre lands at the canvas centre
            ref = im.crop((0, 5, 40, 45)).resize((20, 20), Image.LANCZOS)
            self.assertEqual(out.crop((6, 6, 26, 26)).tobytes(), ref.tobytes())


class TestPadAnchoredReference(unittest.TestCase):
    """AC17 (amendment 9): the larger-canvas reference of an anchored master
    is the animate workflow's crop-and-scale (832x1024 -> rows 96-927 ->
    frame x frame) centred on the canvas, so the figure keeps the mannequin's
    pixel size and aspect. Profile ual-female `s` (amendment 10): body_h 331,
    floor_y 524 at 704; canvas 1152 shifts the floor by (1152 - 704) // 2 =
    224."""

    PROFILE = mannequins.profile()

    def setUp(self):
        self._t = tempfile.TemporaryDirectory()
        self.addCleanup(self._t.cleanup)
        self.tmp = Path(self._t.name)

    def _assert_mannequin_at_1152(self, padded):
        self.assertEqual(Image.open(padded).size, (1152, 1152))
        m = metrics.measure(padded)
        self.assertAlmostEqual(m["body_h"], 331, delta=1)
        self.assertAlmostEqual(m["floor_y"], 524 + (1152 - 704) // 2, delta=1)  # 748
        self.assertAlmostEqual(m["centre_x"], 576, delta=1)

    def test_ac17_r1_padded_anchored_master_measures_the_mannequin(self):
        """AC17-r1 through the real producer: `stills.anchor` at frame 704
        (master body 391, floor 715 on the 832x1024 canvas, FR 6), then
        `pad_ref` to 1152: 1152x1152, body 331 +/- 1, floor 748 +/- 1,
        centre x 576 +/- 1."""
        src, master = self.tmp / "src.png", self.tmp / "s.png"
        mannequins.standing_still().save(src)
        stills.anchor(src, master, 704, self.PROFILE, "s")
        m = metrics.measure(master)
        self.assertAlmostEqual(m["body_h"], 331 * 832 / 704, delta=1)
        self.assertAlmostEqual(m["floor_y"], 524 * 832 / 704 + 96, delta=1)
        padded = self.tmp / "padded.png"
        stages.pad_ref(str(master), str(padded), "704", "1152", [])
        self._assert_mannequin_at_1152(padded)

    def test_ac17_r1_pad_ref_maps_an_fr6_master_to_the_mannequin(self):
        """AC17-r1 on `pad_ref` alone: a synthetic 832x1024 master at the FR 6
        geometry (body 391, feet row 715, centre x 416, flat chroma green)
        pads to body 331, floor 748, centre x 576 — a squash to 704x704 would
        give body 269."""
        master, padded = self.tmp / "s.png", self.tmp / "padded.png"
        mannequins.pose_frame(832, 1024, body_h=391, body_w=200, bottom=715,
                              centre_x=416, corner=(31, 203, 25)).save(master)
        stages.pad_ref(str(master), str(padded), "704", "1152", [])
        self._assert_mannequin_at_1152(padded)

    def test_ac17_r2_a_square_block_stays_square(self):
        """AC17-r2: a 40x40 block in an 832x1024 source stays square +/- 1 px
        through `pad_ref` frame 704 on canvas 1152 (no aspect distortion;
        uniform 704 / 832 scale: about 34 x 34)."""
        src, padded = self.tmp / "block.png", self.tmp / "padded.png"
        im = Image.new("RGB", (832, 1024), (31, 203, 25))
        im.paste(Image.new("RGB", (40, 40), (140, 80, 60)), (396, 492))
        im.save(src)
        stages.pad_ref(str(src), str(padded), "704", "1152", [])
        x0, y0, x1, y1 = _bbox(padded)
        w, h = x1 - x0 + 1, y1 - y0 + 1
        self.assertAlmostEqual(w, h, delta=1)
        self.assertAlmostEqual(w, 40 * 704 / 832, delta=1.5)


class TestExportMixedSizes(unittest.TestCase):
    def test_smaller_frames_are_centred_in_the_largest_cell(self):
        with tempfile.TemporaryDirectory() as t:
            path, base = stubbed_example(t)
            text = path.read_text()
            text = text.replace("generate: [s, se, e, ne, n]", "generate: [s]")
            text = text.replace("mirror: {sw: se, w: e, nw: ne}", "mirror: {}")
            path.write_text(text)
            p = project.load_project(path)
            ctx = stages.StageCtx(p, config.defaults(Path(t) / "home"))
            out = base / "out" / "courier" / "sprites"
            out.mkdir(parents=True)
            small = Image.new("RGBA", (4, 4), (0, 0, 0, 0))
            small.putpixel((1, 1), (255, 0, 0, 255))
            small.save(out / "s-run_000.png")
            Image.new("RGBA", (8, 8), (0, 0, 255, 255)).save(out / "s-punch_000.png")
            log = []
            ok = stages.StageCtx.run_program(ctx, "export", "courier", ["run", "punch"], [], log, False)
            self.assertTrue(ok, log)
            atlas = Image.open(out / "atlas.png").convert("RGBA")
            self.assertEqual(atlas.size, (8, 16))
            self.assertEqual(atlas.getpixel((3, 3)), (255, 0, 0, 255))  # (1,1) + (2,2) centring offset
            self.assertEqual(atlas.getpixel((0, 0))[3], 0)


if __name__ == "__main__":
    unittest.main()
