"""Death repro slice 1 (sprite-pipeline-67): the green drive at render size
and drive scale (spec death-repro FR 1-3, AC 1-2; case rows AC1-r1..r3,
AC2-r1..r5).

No Blender, ffmpeg or ComfyUI: plans are inspected and the in-process
composite step runs on small PNGs written by the test.
"""
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from test_workflow_paths import cfg, stubbed_example

from sprite_pipeline import project, stages

GREEN = (30, 202, 24)


def _with_drive(path, block):
    """Insert a top-level `drive:` block into the stubbed example project."""
    text = path.read_text().replace("\nanimations:", f"\ndrive: {block}\nanimations:", 1)
    path.write_text(text)
    return path


def _with_death_scale(path, value):
    text = path.read_text().replace(
        '- {name: death, source:', f'- {{name: death, drive_scale: {value}, source:', 1)
    path.write_text(text)
    return path


class Case(unittest.TestCase):
    def setUp(self):
        self._t = tempfile.TemporaryDirectory()
        self.addCleanup(self._t.cleanup)
        self.tmp = Path(self._t.name)
        self.path, _ = stubbed_example(self._t.name)

    def load(self):
        return project.load_project(self.path)

    def drive_steps(self, p, anim="punch"):
        ctx = stages.StageCtx(p, cfg(Path("/nonexistent-home")))
        return ctx.plan_pair("courier", p.animation(anim), "s", "drive")

    def composite_step(self, p, anim="punch"):
        return next(argv for argv, _ in self.drive_steps(p, anim) if argv[0] == "composite_drive")

    def run_composite(self, step, size, pixel=None):
        """Write one transparent `size` square frame (one opaque red pixel at
        its centre) into the step's source dir, run the step in process and
        return the composited image."""
        src = self.tmp / "src"
        dst = self.tmp / "dst"
        src.mkdir(exist_ok=True)
        im = Image.new("RGBA", (size, size), (0, 0, 0, 0))
        if pixel is not None:
            im.putpixel(pixel, (255, 0, 0, 255))
        im.save(src / "f0000.png")
        # the step's own scale, background, margin and aspect; the union is
        # the test's frame dir (FR 4)
        stages.composite_drive(src, dst, step[3], step[4:7], step[7], step[8:10], [src])
        return Image.open(dst / "f0000.png")


class TestAC1GreenDrive(Case):
    def test_r1_default_background_keys_every_transparent_pixel(self):
        p = self.load()
        self.assertEqual(p.drive.background, GREEN)
        step = self.composite_step(p)
        self.assertEqual(list(step[4:7]), [str(c) for c in GREEN])
        out = self.run_composite(step, 64, pixel=(32, 32))
        self.assertEqual(out.mode, "RGB")
        colours = {c for _, c in out.getcolors(out.width * out.height)}
        # the opaque pixel survives (resampled); everything else is the key
        self.assertIn(GREEN, colours)
        self.assertEqual(out.getpixel((0, 0)), GREEN)
        self.assertEqual(out.getpixel((out.width - 1, out.height - 1)), GREEN)
        self.assertTrue(any(c != GREEN for c in colours))

    def test_r2_background_override_changes_the_pixels_and_the_key(self):
        default = self.load()
        k0 = default.drive_key(default.animation("punch"), b"x")
        _with_drive(self.path, "{background: [0, 0, 255]}")
        p = self.load()
        self.assertEqual(p.drive.background, (0, 0, 255))
        out = self.run_composite(self.composite_step(p), 64)
        self.assertEqual(out.getpixel((0, 0)), (0, 0, 255))
        self.assertEqual(out.getcolors(), [(out.width * out.height, (0, 0, 255))])
        self.assertNotEqual(p.drive_key(p.animation("punch"), b"x"), k0)

    def test_r3_short_background_names_the_key(self):
        _with_drive(self.path, "{background: [30, 202]}")
        with self.assertRaises(project.ProjectError) as cm:
            self.load()
        self.assertIn("drive.background", str(cm.exception))


class TestAC2RenderSizeAndScale(Case):
    def test_r1_default_render_size_argv_and_transparent_film(self):
        p = self.load()
        self.assertEqual(p.sprite.frame_size, 704)
        argv = self.drive_steps(p)[0][0]
        self.assertIn("res=1152", argv)
        self.assertIn("ortho=4.909091", argv)
        src = (Path(stages.__file__).parent / "render_drive.py").read_text()
        self.assertIn("film_transparent = True", src)
        self.assertNotIn("film_transparent = False", src)

    def test_r2_render_size_override_and_key(self):
        default = self.load()
        k0 = default.drive_key(default.animation("punch"), b"x")
        _with_drive(self.path, "{render_size: 1408}")
        p = self.load()
        argv = self.drive_steps(p)[0][0]
        self.assertIn("res=1408", argv)
        self.assertIn("ortho=6.000000", argv)
        self.assertNotEqual(p.drive_key(p.animation("punch"), b"x"), k0)

    def test_r3_default_scale_shrinks_a_1152_frame_to_697(self):
        p = self.load()
        self.assertAlmostEqual(p.drive_scale(p.animation("punch")), 0.605)
        step = self.composite_step(p)
        self.assertEqual(float(step[3]), 0.605)
        # an opaque 1152 frame: the scaled mannequin area is round(1152*0.605)
        src = self.tmp / "src"
        src.mkdir()
        Image.new("RGBA", (1152, 1152), (255, 0, 0, 255)).save(src / "f0000.png")
        dst = self.tmp / "dst"
        stages.composite_drive(src, dst, step[3], step[4:7], step[7], step[8:10], [src])
        out = Image.open(dst / "f0000.png")
        # FR 4 auto canvas of a 697 x 697 union at margin 1.3, 16:9:
        # H = ceil16(906.1) = 912, W = ceil16(max(906.1, 1621.3)) = 1632,
        # H = max(912, ceil16(918)) = 928
        self.assertEqual(out.size, (1632, 928))
        red = [(x, y) for y in range(out.height) for x in range(out.width)
               if out.getpixel((x, y)) != GREEN]
        xs = [x for x, _ in red]
        ys = [y for _, y in red]
        self.assertEqual(max(xs) - min(xs) + 1, 697)
        self.assertEqual(max(ys) - min(ys) + 1, 697)

    def test_r4_animation_drive_scale_overrides_the_project_scale(self):
        default = self.load()
        k0 = default.drive_key(default.animation("death"), b"x")
        _with_drive(self.path, "{scale: 0.5}")
        _with_death_scale(self.path, 0.4)
        p = self.load()
        self.assertEqual(p.drive_scale(p.animation("death")), 0.4)
        self.assertEqual(float(self.composite_step(p, "death")[3]), 0.4)
        for a in p.animations:
            if a.name != "death":
                self.assertEqual(p.drive_scale(a), 0.5, a.name)
                self.assertEqual(float(self.composite_step(p, a.name)[3]), 0.5, a.name)
        self.assertNotEqual(p.drive_key(p.animation("death"), b"x"), k0)

    def test_r5_non_positive_drive_scale_names_the_animation(self):
        index = [a.name for a in self.load().animations].index("death")
        text = self.path.read_text()
        for bad in (0, -0.5):
            with self.subTest(drive_scale=bad):
                self.path.write_text(text)
                _with_death_scale(self.path, bad)
                with self.assertRaises(project.ProjectError) as cm:
                    self.load()
                self.assertIn(f"animations[{index}].drive_scale", str(cm.exception))


class TestCompositeStepIsInProcess(Case):
    def test_drive_plan_composites_between_blender_and_ffmpeg(self):
        steps = self.drive_steps(self.load())
        labels = [label for _, label in steps]
        # FR 4: before any direction is rendered, the plan also renders the
        # other four directions the canvas spans
        self.assertEqual(labels, ["blender"] * 5 + ["composite-drive", "ffmpeg", "copy-drive"])
        composite = steps[5][0]
        ffmpeg = steps[6][0]
        self.assertIn(f"{composite[2]}/f%04d.png", ffmpeg)


if __name__ == "__main__":
    unittest.main()
