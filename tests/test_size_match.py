"""death-repro AC 13, 14 (FR 17-19): the per-set automatic size factor and
its size log line. Expected values are worked by hand."""
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from test_workflow_paths import cfg, stubbed_example

from sprite_pipeline import project, stages


def figure(path, h, canvas_h=400):
    """A keyed cycle frame whose alpha bbox is `h` px tall (0: no figure)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    im = Image.new("RGBA", (60, canvas_h), (0, 0, 0, 0))
    if h:
        im.paste(Image.new("RGBA", (20, h), (200, 50, 50, 255)), (20, 10))
    im.save(path)
    return path


class TestAC13SizeRatio(unittest.TestCase):
    def setUp(self):
        self._t = tempfile.TemporaryDirectory()
        self.addCleanup(self._t.cleanup)
        self.tmp = Path(self._t.name)

    def cycle(self, heights):
        return [figure(self.tmp / f"frame-{k:03d}.png", h) for k, h in enumerate(heights)]

    def test_r1_median_of_matched_ratios(self):
        # mannequin 200 px on every drive frame: ratios 1.2, 1.4, 1.3
        r = stages.size_ratio(self.cycle([240, 280, 260]), 0, lambda i: 200)
        self.assertAlmostEqual(r, 1.3, places=9)
        self.assertAlmostEqual(1.07 / r, 1.07 / 1.3, places=9)

    def test_r2_frame_without_figure_is_skipped(self):
        r = stages.size_ratio(self.cycle([240, 0, 280, 260]), 0, lambda i: 200)
        self.assertAlmostEqual(r, 1.3, places=9)  # not median(0, 1.2, 1.3, 1.4)
        # a drive frame with no mannequin is skipped too
        r = stages.size_ratio(self.cycle([240, 999, 280, 260]), 0,
                              {0: 200, 2: 200, 3: 200}.get)
        self.assertAlmostEqual(r, 1.3, places=9)

    def test_r3_cycle_frame_k_matches_drive_frame_start_plus_k(self):
        asked = []
        drive = {i: 100 for i in range(5)} | {5: 200, 6: 200, 7: 200}

        def mh(i):
            asked.append(i)
            return drive.get(i)

        r = stages.size_ratio(self.cycle([240, 260, 280]), 5, mh)
        self.assertEqual(asked, [5, 6, 7])
        self.assertAlmostEqual(r, 1.3, places=9)


class SeamCase(unittest.TestCase):
    """The real StageCtx measures drive frames and cycle frames on disk."""

    def setUp(self):
        self._t = tempfile.TemporaryDirectory()
        self.addCleanup(self._t.cleanup)
        self.tmp = Path(self._t.name)
        self.path, _ = stubbed_example(self._t.name)
        text = self.path.read_text()
        # drive scale 0.5: a full-height 400 px drive frame is 200 px on the canvas
        self.path.write_text(text.replace("- {name: death, source:",
                                          "- {name: death, drive_scale: 0.5, source:", 1))

    def ctx(self, height_scale=None):
        if height_scale:
            text = self.path.read_text()
            self.path.write_text(text.replace("pronoun: she}",
                                              f"pronoun: she, height_scale: {height_scale}}}", 1))
        p = project.load_project(self.path)
        ctx = stages.StageCtx(p, cfg(self.tmp / "home"))
        death = p.animation("death")
        for d in death.directions:  # every direction rendered: the canvas exists
            fd = ctx.frames_dir("courier", death, d)
            fd.mkdir(parents=True, exist_ok=True)
            for i in range(3):
                Image.new("RGBA", (100, 400), (90, 90, 90, 255)).save(fd / f"f{i:04d}.png")
        return p, ctx, death

    def cycle(self, ctx, death, d, heights):
        for k, h in enumerate(heights):
            figure(ctx.loops_dir("courier", death, d) / "cycle" / f"frame-{k:03d}.png", h)

    def scales(self, ctx, dirs):
        argv = ctx.plan_kit_stage("courier", "pixel", ["death"], dirs)[0][0]
        return {argv[i + 1].split("=")[0]: float(argv[i + 1].split("=")[1])
                for i, a in enumerate(argv) if a == "--set-scale"}


class TestAC13SetScale(SeamCase):
    def test_r4_no_height_scale_is_base_times_auto(self):
        p, ctx, death = self.ctx()
        self.cycle(ctx, death, "s", [240, 280, 260])
        auto, r = ctx.size_match("courier", death, "s")
        self.assertAlmostEqual(r, 1.3, places=9)
        self.assertAlmostEqual(auto, 1.07 / 1.3, places=9)
        self.assertAlmostEqual(self.scales(ctx, ["s"])["s-death"],
                               0.2023 / 0.5 * 1.07 / 1.3, places=5)

    def test_r5_mirror_measures_source_and_takes_its_height_scale(self):
        p, ctx, death = self.ctx("{se: 0.9}")
        self.assertEqual(p.mirror["sw"], "se")
        self.cycle(ctx, death, "se", [240, 280, 260])  # only the source has frames
        self.assertAlmostEqual(self.scales(ctx, ["sw"])["sw-death"],
                               0.2023 / 0.5 * 1.07 / 1.3 * 0.9, places=5)


class TestAC14SizeLine(unittest.TestCase):
    def test_r1_in_range(self):
        self.assertEqual(stages.size_line("vexa", "se", "death", 1.07, 1.07),
                         "vexa se-death: size x1.00")

    def test_r2_out_of_range_appends_check(self):
        line = stages.size_line("vexa", "se", "death", 1.3, 1.0)
        self.assertEqual(line, "vexa se-death: size x0.77 CHECK")

    def test_r3_range_is_inclusive(self):
        self.assertEqual(stages.size_line("vexa", "s", "death", 0.85, 1.0),
                         "vexa s-death: size x1.18")
        self.assertEqual(stages.size_line("vexa", "s", "death", 1.15, 1.0),
                         "vexa s-death: size x0.87")
        self.assertTrue(stages.size_line("vexa", "s", "death", 0.84, 1.0).endswith(" CHECK"))
        self.assertTrue(stages.size_line("vexa", "s", "death", 1.16, 1.0).endswith(" CHECK"))


class TestAC14Unmeasurable(SeamCase):
    def test_r4_no_figure_gives_1_0_and_na_check(self):
        p, ctx, death = self.ctx()
        self.cycle(ctx, death, "s", [0, 0])
        self.assertEqual(ctx.size_match("courier", death, "s"), (1.0, None))
        self.assertEqual(stages.size_line("courier", "s", "death", None, 1.07),
                         "courier s-death: size n/a CHECK")
        self.assertAlmostEqual(self.scales(ctx, ["s"])["s-death"], 0.2023 / 0.5, places=5)

    def test_r4_cli_logs_the_line_and_continues(self):
        """The real CLI dry run prints one size line per set and plans the
        pixel stage anyway."""
        import os
        from test_workflow_paths import Env
        self.ctx()  # drives rendered, no cycle frames: unmeasurable
        old = os.environ.get("SPRITE_PIPELINE_HOME")
        self.addCleanup(lambda: os.environ.pop("SPRITE_PIPELINE_HOME", None) if old is None
                        else os.environ.__setitem__("SPRITE_PIPELINE_HOME", old))
        os.environ["SPRITE_PIPELINE_HOME"] = str(self.tmp / "home")
        code, out = Env.run_cli(None, ["go", str(self.path), "--dry-run", "--no-stop",
                                       "--stage", "pixel", "--animations", "death",
                                       "--dirs", "s"])
        self.assertEqual(code, 0, out)
        self.assertIn("courier s-death: size n/a CHECK\n", out)
        self.assertIn("sprite_pipeline.pixelize", out)

if __name__ == "__main__":
    unittest.main()
