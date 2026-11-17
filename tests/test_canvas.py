"""Per-animation canvas: a wide pose (a fall) renders on a larger square
canvas at the same pixel scale, so the character stays inside the frame.

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


def _with_canvas(path, canvas):
    text = path.read_text().replace(
        '- {name: punch, source: "anims/ual.glb#Punch_Cross",',
        f'- {{name: punch, canvas: {canvas}, source: "anims/ual.glb#Punch_Cross",', 1)
    path.write_text(text)
    return path


class Case(unittest.TestCase):
    def setUp(self):
        self._t = tempfile.TemporaryDirectory()
        self.addCleanup(self._t.cleanup)
        self.path, self.base = stubbed_example(self._t.name)

    def load(self, canvas=None):
        if canvas is not None:
            _with_canvas(self.path, canvas)
        return project.load_project(self.path)


class TestLoad(Case):
    def test_canvas_defaults_to_frame_size(self):
        p = self.load()
        self.assertEqual(p.animation("punch").canvas, 704)

    def test_canvas_loaded(self):
        self.assertEqual(self.load(960).animation("punch").canvas, 960)
        self.assertEqual(project.load_project(self.path).animation("run").canvas, 704)

    def test_canvas_smaller_than_frame_size_rejected(self):
        with self.assertRaises(project.ProjectError) as cm:
            self.load(512)
        self.assertIn("animations[1].canvas", str(cm.exception))

    def test_canvas_not_multiple_of_16_rejected(self):
        with self.assertRaises(project.ProjectError) as cm:
            self.load(970)
        self.assertIn("multiple of 16", str(cm.exception))


class TestPlans(Case):
    def ctx(self, canvas):
        p = self.load(canvas)
        return stages.StageCtx(p, cfg(Path("/nonexistent-home"))), p

    def test_drive_renders_wider_at_the_same_pixel_scale(self):
        ctx, p = self.ctx(960)
        argv = ctx.plan_pair("courier", p.animation("punch"), "s", "drive")[0][0]
        self.assertIn("res=960", argv)
        ortho = [x for x in argv if x.startswith("ortho=")][0]
        self.assertAlmostEqual(float(ortho[6:]), 3.0 * 960 / 704, places=4)
        run = ctx.plan_pair("courier", p.animation("run"), "s", "drive")[0][0]
        self.assertFalse([x for x in run if x.startswith(("res=", "ortho="))])

    def test_drive_key_changes_with_canvas(self):
        a = self.load().animation("punch")
        b = project.load_project(_with_canvas(self.path, 960)).animation("punch")
        p = project.load_project(self.path)
        self.assertNotEqual(p.drive_key(a, b"x"), p.drive_key(b, b"x"))

    def test_animate_uses_a_padded_reference_at_canvas_size(self):
        ctx, p = self.ctx(960)
        cmds = ctx.plan_pair("courier", p.animation("punch"), "s", "animate")
        pad, (argv, _) = cmds[0][0], cmds[1]
        self.assertEqual(pad[0], "pad_ref")
        self.assertEqual(pad[1], str(ctx.dir_still("courier", "s")))
        self.assertEqual(argv[argv.index("--ref") + 1], pad[2])
        self.assertEqual(argv[argv.index("--w") + 1], "960")
        self.assertEqual(argv[argv.index("--h") + 1], "960")
        self.assertEqual(pad[3:], ("704", "960"))

    def test_default_canvas_animate_is_unchanged(self):
        ctx, p = self.ctx(960)
        cmds = ctx.plan_pair("courier", p.animation("run"), "s", "animate")
        self.assertEqual(len(cmds), 1)
        argv = cmds[0][0]
        self.assertEqual(argv[argv.index("--ref") + 1], str(ctx.dir_still("courier", "s")))


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
    pixel size and aspect. Profile ual-female: body_h 324, floor_y 525 at
    704; canvas 1152 shifts the floor by (1152 - 704) // 2 = 224."""

    PROFILE = {"glb_name": "test.glb", "resolution": 704, "body_h": 324,
               "floor_y": 525}

    def setUp(self):
        self._t = tempfile.TemporaryDirectory()
        self.addCleanup(self._t.cleanup)
        self.tmp = Path(self._t.name)

    def _assert_mannequin_at_1152(self, padded):
        self.assertEqual(Image.open(padded).size, (1152, 1152))
        m = metrics.measure(padded)
        self.assertAlmostEqual(m["body_h"], 324, delta=1)
        self.assertAlmostEqual(m["floor_y"], 525 + (1152 - 704) // 2, delta=1)  # 749
        self.assertAlmostEqual(m["centre_x"], 576, delta=1)

    def test_ac17_r1_padded_anchored_master_measures_the_mannequin(self):
        """AC17-r1 through the real producer: `stills.anchor` at frame 704
        (master body 383, floor 716 on the 832x1024 canvas, FR 6), then
        `pad_ref` to 1152: 1152x1152, body 324 +/- 1, floor 749 +/- 1,
        centre x 576 +/- 1."""
        src, master = self.tmp / "src.png", self.tmp / "s.png"
        mannequins.standing_still().save(src)
        stills.anchor(src, master, 704, self.PROFILE)
        m = metrics.measure(master)
        self.assertAlmostEqual(m["body_h"], 324 * 832 / 704, delta=1)
        self.assertAlmostEqual(m["floor_y"], 525 * 832 / 704 + 96, delta=1)
        padded = self.tmp / "padded.png"
        stages.pad_ref(str(master), str(padded), "704", "1152", [])
        self._assert_mannequin_at_1152(padded)

    def test_ac17_r1_pad_ref_maps_an_fr6_master_to_the_mannequin(self):
        """AC17-r1 on `pad_ref` alone: a synthetic 832x1024 master at the FR 6
        geometry (body 383, feet row 716, centre x 416, flat chroma green)
        pads to body 324, floor 749, centre x 576 — a squash to 704x704 would
        give body 263."""
        master, padded = self.tmp / "s.png", self.tmp / "padded.png"
        mannequins.pose_frame(832, 1024, body_h=383, body_w=200, bottom=716,
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
