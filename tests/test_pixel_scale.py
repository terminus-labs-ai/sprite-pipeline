"""death-repro AC 7 (FR 9, 10): pixelize receives `pixel_scale` when set,
else `sprite.scale / drive scale`, times the character's `height_scale` for
the direction (1.0 when absent; a mirrored direction takes its source's
factor). The example project has sprite.scale 0.2023 and the default drive
scale 0.605; expected values are worked by hand."""
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from test_workflow_paths import cfg, stubbed_example

from sprite_pipeline import project, stages

BASE = 0.334380  # 0.2023 / 0.605


def _set(path, old, new):
    text = path.read_text()
    assert old in text, old
    path.write_text(text.replace(old, new, 1))


class Case(unittest.TestCase):
    def setUp(self):
        self._t = tempfile.TemporaryDirectory()
        self.addCleanup(self._t.cleanup)
        self.tmp = Path(self._t.name)
        self.path, _ = stubbed_example(self._t.name)

    def height_scale(self, block):
        _set(self.path, "pronoun: she}", f"pronoun: she, height_scale: {block}}}")

    def death_extra(self, extra):
        _set(self.path, "- {name: death, source:", f"- {{name: death, {extra}, source:")

    def scales(self, dirs=("s", "se", "e", "ne", "n")):
        """The --set-scale value pixelize gets for each `<dir>-death` set."""
        p = project.load_project(self.path)
        ctx = stages.StageCtx(p, cfg(self.tmp / "home"))
        argv = ctx.plan_kit_stage("courier", "pixel", ["death"], list(dirs))[0][0]
        self.assertNotIn("--scale", argv)
        got = {}
        for i, a in enumerate(argv):
            if a == "--set-scale":
                name, v = argv[i + 1].split("=")
                got[name] = float(v)
        self.assertEqual(sorted(got), sorted(f"{d}-death" for d in dirs))
        return got


class TestAC7PixelScale(Case):
    def test_r1_sprite_scale_over_drive_scale(self):
        for name, v in self.scales().items():
            self.assertAlmostEqual(v, BASE, places=6, msg=name)

    def test_r1_drive_scale_of_the_animation(self):
        self.death_extra("drive_scale: 0.5")
        self.assertAlmostEqual(self.scales(["s"])["s-death"], 0.4046, places=6)

    def test_r2_pixel_scale_wins(self):
        self.death_extra("pixel_scale: 0.3408, drive_scale: 0.5")
        self.assertEqual(self.scales(["s"])["s-death"], 0.3408)

    def test_r3_height_scale_of_the_direction(self):
        self.height_scale("{s: 0.91}")
        self.assertAlmostEqual(self.scales()["s-death"], 0.304286, places=6)

    def test_r3_height_scale_times_pixel_scale(self):
        self.height_scale("{s: 0.91}")
        self.death_extra("pixel_scale: 0.3408")
        self.assertAlmostEqual(self.scales(["s"])["s-death"], 0.310128, places=6)

    def test_r4_unlisted_direction_is_1_0(self):
        self.height_scale("{s: 0.91}")
        got = self.scales()
        for d in ("se", "e", "ne", "n"):
            self.assertAlmostEqual(got[f"{d}-death"], BASE, places=6, msg=d)

    def test_r5_mirror_uses_its_source_factor(self):
        self.height_scale("{se: 0.96}")
        p = project.load_project(self.path)
        self.assertEqual(p.mirror["sw"], "se")
        cs, death = p.characters[0], p.animation("death")
        self.assertAlmostEqual(p.pixel_scale(death, cs, "sw"), BASE * 0.96, places=6)
        self.assertAlmostEqual(p.pixel_scale(death, cs, "se"), BASE * 0.96, places=6)
        self.assertAlmostEqual(self.scales(["se"])["se-death"], 0.321005, places=6)

    def test_r6_unknown_direction_is_a_project_error(self):
        self.height_scale("{up: 0.9}")
        with self.assertRaisesRegex(project.ProjectError,
                                    r"project\.yaml: characters\[0\]\.height_scale\.up"):
            project.load_project(self.path)

    def test_r6_non_positive_factor_is_a_project_error(self):
        for bad in ("0", "-0.5", "x"):
            with self.subTest(bad=bad):
                self.path, _ = stubbed_example(tempfile.mkdtemp(dir=self.tmp))
                self.height_scale(f"{{s: {bad}}}")
                with self.assertRaisesRegex(project.ProjectError,
                                            r"project\.yaml: characters\[0\]\.height_scale\.s"):
                    project.load_project(self.path)

    def test_r6_non_positive_pixel_scale_is_a_project_error(self):
        self.death_extra("pixel_scale: 0")
        with self.assertRaisesRegex(project.ProjectError, r"animations\[11\]\.pixel_scale"):
            project.load_project(self.path)


class TestPixelizeSetScale(unittest.TestCase):
    """The real pixelize module applies each set's --set-scale."""

    def test_each_set_is_shrunk_by_its_own_scale(self):
        with tempfile.TemporaryDirectory() as t:
            t = Path(t)
            for name in ("a", "b"):
                (t / name).mkdir()
                im = Image.new("RGBA", (40, 100), (0, 0, 0, 0))
                im.paste(Image.new("RGBA", (20, 80), (200, 50, 50, 255)), (10, 10))
                im.save(t / name / "f0000.png")
            r = subprocess.run([sys.executable, "-m", "sprite_pipeline.pixelize",
                                "--set", f"a={t / 'a'}", "--set", f"b={t / 'b'}",
                                "--set-scale", "a=0.5", "--set-scale", "b=0.25",
                                "--scale", "1.0", "--out", str(t / "out")],
                               capture_output=True, text=True)
            self.assertEqual(r.returncode, 0, r.stderr)
            self.assertEqual(Image.open(t / "out" / "a_000.png").size, (20, 50))
            self.assertEqual(Image.open(t / "out" / "b_000.png").size, (10, 25))


if __name__ == "__main__":
    unittest.main()
