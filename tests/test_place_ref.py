"""death-repro AC 6 (FR 8): the reference is the direction master scaled to
`reference.height_ratio` x the scaled mannequin's f0 body height, feet on the
f0 bottom row, x-centred on the f0 bbox, on a drive-sized canvas filled with
`drive.background`. Expected values are worked by hand; the placed figure is
measured with the FR 6 measure (`metrics.measure`)."""
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from test_workflow_paths import cfg, stubbed_example

from sprite_pipeline import cli, metrics, project, stages

GREEN = (30, 202, 24)
F0 = (250, 100, 350, 300)  # inclusive: body height 201, bottom row 300, x-centre 300


def master(path):
    """The case-table master: an opaque 100x400 rect on transparent."""
    im = Image.new("RGBA", (300, 600), (0, 0, 0, 0))
    im.paste(Image.new("RGBA", (100, 400), (200, 50, 50, 255)), (100, 100))
    im.save(path)


def _set(path, old, new):
    path.write_text(path.read_text().replace(old, new, 1))


class Case(unittest.TestCase):
    def setUp(self):
        self._t = tempfile.TemporaryDirectory()
        self.addCleanup(self._t.cleanup)
        self.tmp = Path(self._t.name)
        self.path, _ = stubbed_example(self._t.name)

    def place(self, ratio, background=GREEN, name="ref.png"):
        src, dst = self.tmp / "master.png", self.tmp / "out" / name
        master(src)
        stages.place_ref(src, dst, F0, (576, 336), ratio, background, [])
        return dst

    def assert_placed(self, dst, body_h, background=GREEN):
        im = Image.open(dst).convert("RGB")
        self.assertEqual(im.size, (576, 336))
        self.assertEqual(im.getpixel((0, 0)), background)
        self.assertEqual(im.getpixel((575, 335)), background)
        m = metrics.measure(dst)
        self.assertAlmostEqual(m["body_h"], body_h, delta=1)
        self.assertEqual(m["floor_y"], 300)
        self.assertAlmostEqual(m["centre_x"], 300, delta=1)


class TestAC6PlaceRef(Case):
    def test_r1_default_ratio_1_07(self):
        p = project.load_project(self.path)  # the example has no reference: block
        self.assertEqual(p.reference.height_ratio, 1.07)
        self.assertEqual(tuple(p.drive.background), GREEN)
        self.assert_placed(self.place(p.reference.height_ratio, p.drive.background), 214)

    def test_r2_height_ratio_1_2_and_its_own_work_path(self):
        _set(self.path, "output: out", "output: out\nreference: {height_ratio: 1.2}")
        p = project.load_project(self.path)
        self.assertEqual(p.reference.height_ratio, 1.2)
        self.assert_placed(self.place(p.reference.height_ratio), 240)

    def test_r3_background_comes_from_drive(self):
        _set(self.path, "output: out", "output: out\ndrive: {background: [0, 0, 255]}")
        p = project.load_project(self.path)
        self.assert_placed(self.place(p.reference.height_ratio, p.drive.background), 214,
                           background=(0, 0, 255))

    def test_rgb_master_on_a_flat_background(self):
        """A master without transparency: the figure is what differs from its corner."""
        src, dst = self.tmp / "rgb.png", self.tmp / "out" / "rgb-ref.png"
        im = Image.new("RGB", (300, 600), (31, 203, 25))
        im.paste(Image.new("RGB", (100, 400), (200, 50, 50)), (100, 100))
        im.save(src)
        stages.place_ref(src, dst, F0, (576, 336), 1.07, GREEN, [])
        self.assert_placed(dst, 214)

    def test_bad_height_ratio_is_a_project_error(self):
        _set(self.path, "output: out", "output: out\nreference: {height_ratio: 0}")
        with self.assertRaisesRegex(project.ProjectError, r"project\.yaml: reference\.height_ratio"):
            project.load_project(self.path)


# Drives for the real-producer seam: 200 px transparent frames at drive_scale
# 0.5; union over s, se, e is x 40-160, y 0-200 -> canvas_for((60, 100)) =
# 256x144, paste offset (78, 22). The s f0 block (90, 20) 20x160 scales to
# (45, 10) 10x80, so its f0 bbox is x 123-132, y 32-111 (inclusive).
BLOCKS = {"s": ((90, 20), (20, 160)), "se": ((40, 100), (120, 80)),
          "e": ((60, 0), (40, 200))}


class TestAC6ThroughThePlan(Case):
    """The animate plan's place_ref step, built from real drive frames and
    executed by the cli's in-process dispatch."""

    def setUp(self):
        super().setUp()
        _set(self.path, "- {name: death, source:",
             "- {name: death, directions: [s, se, e], drive_scale: 0.5, source:")

    def run_plan(self):
        p = project.load_project(self.path)
        ctx = stages.StageCtx(p, cfg(self.tmp / "home"))
        anim = p.animation("death")
        for d, (at, size) in BLOCKS.items():
            frames = ctx.frames_dir("courier", anim, d)
            frames.mkdir(parents=True, exist_ok=True)
            im = Image.new("RGBA", (200, 200), (0, 0, 0, 0))
            im.paste(Image.new("RGBA", size, (255, 0, 0, 255)), at)
            im.save(frames / "f0000.png")
        ctx.dir_still("courier", "s").parent.mkdir(parents=True, exist_ok=True)
        master(ctx.dir_still("courier", "s"))
        cmds = ctx.plan_pair("courier", anim, "s", "animate")
        step = cmds[0][0]
        self.assertEqual(step[0], "place_ref")
        argv = cmds[-1][0]
        self.assertEqual(argv[argv.index("--ref") + 1], step[2])
        self.assertTrue(cli.run_cmds(ctx, cmds[:1], []))
        return ctx, anim, step

    def test_f0_is_the_composited_mannequin(self):
        ctx, anim, step = self.run_plan()
        f0 = tuple(int(v) for v in step[3:7])
        for got, want in zip(f0, (123, 32, 132, 111)):
            self.assertAlmostEqual(got, want, delta=1)
        # the same box the composite stage draws on the background
        out = self.tmp / "comp"
        p = ctx.project
        stages.composite_drive(ctx.frames_dir("courier", anim, "s"), out, 0.5, p.drive.background,
                               p.drive.margin, p.drive.aspect,
                               [ctx.frames_dir("courier", anim, d) for d in BLOCKS])
        m = metrics.measure(out / "f0000.png")
        self.assertEqual(m["floor_y"], f0[3])
        self.assertEqual(m["body_h"], f0[3] - f0[1] + 1)

    def test_r1_placed_on_f0_through_the_plan(self):
        _, _, step = self.run_plan()
        f0 = tuple(int(v) for v in step[3:7])
        im = Image.open(step[2]).convert("RGB")
        self.assertEqual(im.size, (256, 144))
        self.assertEqual(im.getpixel((0, 0)), GREEN)
        m = metrics.measure(step[2])
        self.assertAlmostEqual(m["body_h"], 1.07 * (f0[3] - f0[1] + 1), delta=1)
        self.assertEqual(m["floor_y"], f0[3])
        self.assertAlmostEqual(m["centre_x"], (f0[0] + f0[2]) / 2, delta=1)

    def test_r2_ratio_enters_the_work_path(self):
        _, _, step = self.run_plan()
        _set(self.path, "output: out", "output: out\nreference: {height_ratio: 1.2}")
        _, _, step2 = self.run_plan()
        self.assertEqual(step2[9], "1.2")
        self.assertNotEqual(step2[2], step[2])
        m = metrics.measure(step2[2])
        self.assertAlmostEqual(m["body_h"], 1.2 * 80, delta=1)

    def test_r3_background_through_the_plan(self):
        _set(self.path, "output: out", "output: out\ndrive: {background: [0, 0, 255]}")
        _, _, step = self.run_plan()
        self.assertEqual(step[10:13], ("0", "0", "255"))
        self.assertEqual(Image.open(step[2]).convert("RGB").getpixel((0, 0)), (0, 0, 255))


if __name__ == "__main__":
    unittest.main()
