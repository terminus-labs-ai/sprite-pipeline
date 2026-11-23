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
