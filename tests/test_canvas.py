"""Per-animation canvas: a wide pose (a fall) renders on a larger square
canvas at the same pixel scale, so the character stays inside the frame.

No network, GPU, Blender or ComfyUI: plans are inspected, images are small
PNGs written by the test.
"""
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from test_workflow_paths import cfg, stubbed_example

from sprite_pipeline import config, project, stages


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
        self.assertEqual(pad[0], "pad_still")
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


class TestPadStill(unittest.TestCase):
    def test_pad_keeps_pixel_size_and_centres_on_key_colour(self):
        with tempfile.TemporaryDirectory() as t:
            src, dst = Path(t) / "s.png", Path(t) / "out" / "p.png"
            im = Image.new("RGB", (40, 50), (0, 200, 0))
            im.putpixel((20, 25), (255, 0, 0))
            im.save(src)
            stages.pad_still(str(src), str(dst), 20, 32, [])
            out = Image.open(dst).convert("RGB")
            self.assertEqual(out.size, (32, 32))
            self.assertEqual(out.getpixel((0, 0)), (0, 200, 0))
            # the still is scaled to frame x frame, then centred: its centre lands at the canvas centre
            ref = im.resize((20, 20), Image.LANCZOS)
            self.assertEqual(out.crop((6, 6, 26, 26)).tobytes(), ref.tobytes())


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
