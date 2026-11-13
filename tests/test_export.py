"""AC7: the Godot export is driven by the project — each `directions.mirror`
entry is a left-right flip of its source's frames, the direction order is
`generate` then `mirror` keys, and every Godot animation in
`sprite_frames.tres` is named by the project's `naming` pattern (spec FR 11).

No network, no GPU, no Blender, no ComfyUI at test runtime: the frames are
small PNGs written by the test itself.
"""
import re
import shutil
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from sprite_pipeline import config, project, stages
from sprite_pipeline.export_godot import anim_name, direction_order, write_tres

ROOT = Path(__file__).resolve().parent.parent
EXAMPLE = ROOT / "examples" / "isometric"

# one row of one pixel per colour: a left-right flip is a full reversal of the
# pixel row, so the flip is checked pixel-exactly, not just on the ends
ROW = [(255, 0, 0, 255), (0, 255, 0, 255), (0, 0, 255, 255), (255, 255, 0, 255)]
FLIPPED = list(reversed(ROW))


def _write_frames(out, anim, d, n):
    """n pixelised frames for one (animation, direction) in <dir>-<anim>_%03d.png."""
    for i in range(n):
        im = Image.new("RGBA", (4, 1))
        im.putdata(ROW if i == 0 else FLIPPED)
        im.save(out / f"{d}-{anim}_{i:03d}.png")


def _pixels(path):
    return list(Image.open(path).convert("RGBA").get_flattened_data())


def _anim_names(tres_text):
    """The Godot animation names, in the .tres's own order."""
    return re.findall(r'"name": &"([^"]+)"', tres_text)


def _write_project(tmp, generate, mirror, naming):
    """A valid isometric project with the given directions and naming pattern;
    the still and the GLB are stub bytes like ProjectFixture in test_stages.py."""
    base = tmp / "iso"
    shutil.copytree(EXAMPLE, base)
    (base / "stills").mkdir(parents=True, exist_ok=True)
    (base / "anims").mkdir(parents=True, exist_ok=True)
    png = Image.new("RGBA", (2, 2), (2, 3, 5, 255))
    png.save(base / "stills" / "courier.png")
    (base / "anims" / "ual.glb").write_bytes(png.tobytes())
    p = base / "project.yaml"
    text = p.read_text()
    text = text.replace("generate: [s, se, e, ne, n]", f"generate: [{', '.join(generate)}]")
    text = text.replace("mirror: {sw: se, w: e, nw: ne}",
                        "mirror: {" + ", ".join(f"{k}: {v}" for k, v in mirror.items()) + "}")
    text = re.sub(r'^naming: .*\n', f'naming: "{naming}"\n', text, count=1, flags=re.M)
    p.write_text(text)
    return base / "project.yaml"


class ExportCase(unittest.TestCase):
    """One project on disk, its StageCtx, and the character's sprites dir."""

    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmpdir.cleanup)
        self.tmp = Path(self._tmpdir.name)
        self.home = self.tmp / "home"
        self.base = self.tmp / "iso"

    def _ctx(self, path):
        p = project.load_project(path)
        return stages.StageCtx(p, config.defaults(self.home)), p

    def _export(self, ctx, char, anim_names, dry=False):
        log = []
        ok = stages.StageCtx.run_program(ctx, "export", char, anim_names, [], log, dry)
        return ok, log


# ----------------------------------------------------------------- AC7-r1 (unit)

class TestMirrorFlips(ExportCase):
    def test_r1_sw_is_a_left_right_flip_of_se(self):
        """generate [s, se, e, ne, n], mirror {sw: se, w: e, nw: ne}; frames
        exist for s and se: every sw frame is the se frame flipped left-right,
        and the missing report covers every other direction (the case table's
        view check is bypassed in this unit test: only se's frames are needed
        for the flip, s's stand in for the remaining generated directions)."""
        path = _write_project(self.tmp, ["s", "se", "e", "ne", "n"],
                              {"sw": "se", "w": "e", "nw": "ne"}, "{animation}_{dir}")
        out = self.base / "out" / "courier" / "sprites"
        out.mkdir(parents=True, exist_ok=True)
        _write_frames(out, "run", "se", 2)
        ctx, p = self._ctx(path)
        ok, log = self._export(ctx, "courier", ["run"], dry=False)
        self.assertFalse(ok, log)
        missing = log[-1].split("missing: ")[1]
        # se's mirror sw is present; every direction without frames is missing
        self.assertNotIn("run/sw", missing)
        for m in ("run/s", "run/e", "run/ne", "run/n", "run/w", "run/nw"):
            self.assertIn(m, missing, missing)
        # sw frames are the se frames flipped left-right, pixel-exact
        self.assertTrue((out / "sw-run_000.png").is_file())
        self.assertTrue((out / "sw-run_001.png").is_file())
        self.assertEqual(_pixels(out / "sw-run_000.png"), FLIPPED)
        self.assertEqual(_pixels(out / "se-run_000.png"), ROW)
        self.assertEqual(_pixels(out / "sw-run_001.png"), ROW)  # frame 1 was already reversed
        self.assertEqual(_pixels(out / "se-run_001.png"), FLIPPED)
        # the atlas packs se + sw rows of 4x1 frames; w/nw are not invented
        self.assertTrue((out / "atlas.png").is_file())
        self.assertEqual(Image.open(out / "atlas.png").size, (8, 2))  # 2 cols x 2 rows of 4x1
        text = (out / "sprite_frames.tres").read_text()
        self.assertEqual(_anim_names(text), ["run_se", "run_sw"])


class TestMirrorFromItsSource(ExportCase):
    def test_each_mirror_direction_flips_its_own_source(self):
        """All five generated directions present: w flips e, nw flips ne (not
        the whole table at once), and the .tres carries all eight directions
        in the generate-then-mirror order."""
        path = _write_project(self.tmp, ["s", "se", "e", "ne", "n"],
                              {"sw": "se", "w": "e", "nw": "ne"}, "{animation}_{dir}")
        out = self.base / "out" / "courier" / "sprites"
        out.mkdir(parents=True, exist_ok=True)
        for d in ("s", "se", "e", "ne", "n"):
            _write_frames(out, "run", d, 2)
        ctx, _p = self._ctx(path)
        ok, log = self._export(ctx, "courier", ["run"], dry=False)
        self.assertTrue(ok, log)
        self.assertEqual(Image.open(out / "atlas.png").size, (8, 8))  # 2 cols x 8 rows of 4x1
        text = (out / "sprite_frames.tres").read_text()
        self.assertEqual(_anim_names(text),
                         ["run_s", "run_se", "run_e", "run_ne", "run_n",
                          "run_sw", "run_w", "run_nw"])
        for d, src in (("sw", "se"), ("w", "e"), ("nw", "ne")):
            for i in range(2):
                self.assertEqual(_pixels(out / f"{d}-run_{i:03d}.png"),
                                 list(reversed(_pixels(out / f"{src}-run_{i:03d}.png"))),
                                 f"{d} is not the flip of {src} frame {i}")


# ----------------------------------------------------------------- AC7-r2 (unit)

class TestDirectionSetFromProject(ExportCase):
    def test_r2_side_layout_exports_only_the_projected_directions(self):
        """generate [e], mirror {w: e}: the export has e and w only; no other
        direction is reported missing (the direction set comes from the
        project, FR 1 directions explicit)."""
        path = _write_project(self.tmp, ["e"], {"w": "e"}, "{animation}_{dir}")
        out = self.base / "out" / "courier" / "sprites"
        out.mkdir(parents=True, exist_ok=True)
        _write_frames(out, "run", "e", 2)
        ctx, p = self._ctx(path)
        self.assertEqual(direction_order(p), ["e", "w"])
        ok, log = self._export(ctx, "courier", ["run"], dry=False)
        self.assertTrue(ok, log)
        self.assertNotIn("missing", log[-1])  # nothing beyond e and w is projected
        text = (out / "sprite_frames.tres").read_text()
        self.assertEqual(_anim_names(text), ["run_e", "run_w"])
        self.assertEqual(_pixels(out / "w-run_000.png"), FLIPPED)
        self.assertEqual(_pixels(out / "w-run_001.png"), ROW)
        self.assertTrue((out / "contact-sheet.png").is_file())


# ----------------------------------------------------------------- AC7-r3 (unit)

class TestNamingPattern(ExportCase):
    def test_r3_naming_pattern_names_the_animations(self):
        """naming "{character}_{animation}_{dir}": the .tres carries
        courier_run_se; the default naming gives run_se."""
        path = _write_project(self.tmp, ["e", "se"], {"w": "e"},
                              "{character}_{animation}_{dir}")
        out = self.base / "out" / "courier" / "sprites"
        out.mkdir(parents=True, exist_ok=True)
        _write_frames(out, "run", "e", 1)
        _write_frames(out, "run", "se", 1)
        ctx, p = self._ctx(path)
        self.assertEqual(anim_name(p, "courier", "run", "se"), "courier_run_se")
        ok, _log = self._export(ctx, "courier", ["run"], dry=False)
        self.assertTrue(ok, _log)
        text = (out / "sprite_frames.tres").read_text()
        self.assertIn('"name": &"courier_run_se"', text)
        self.assertIn('"name": &"courier_run_e"', text)
        self.assertNotIn('"name": &"run_se"', text)
        # the sub-resource ids follow the name, one per frame
        self.assertIn('[sub_resource type="AtlasTexture" id="AtlasTexture_courier_run_se_f0"]', text)

    def test_r3_default_naming_gives_animation_dir(self):
        path = _write_project(self.tmp, ["se"], {}, "{animation}_{dir}")
        out = self.base / "out" / "courier" / "sprites"
        out.mkdir(parents=True, exist_ok=True)
        _write_frames(out, "run", "se", 1)
        ctx, p = self._ctx(path)
        self.assertEqual(anim_name(p, "courier", "run", "se"), "run_se")
        ok, _log = self._export(ctx, "courier", ["run"], dry=False)
        self.assertTrue(ok, _log)
        text = (out / "sprite_frames.tres").read_text()
        self.assertIn('"name": &"run_se"', text)


# ----------------------------------------------------------------- AC7-r4 (unit)

class TestTresAtlasPath(ExportCase):
    def test_r4_ext_resource_path_is_atlas_relative_to_the_tres(self):
        """The .tres next to atlas.png references it as plain atlas.png —
        relative to the .tres, no directory component, no absolute path."""
        path = _write_project(self.tmp, ["se"], {}, "{animation}_{dir}")
        out = self.base / "out" / "courier" / "sprites"
        out.mkdir(parents=True, exist_ok=True)
        _write_frames(out, "run", "se", 1)
        ctx, _p = self._ctx(path)
        ok, _log = self._export(ctx, "courier", ["run"], dry=False)
        self.assertTrue(ok, _log)
        tres = out / "sprite_frames.tres"
        self.assertTrue(tres.is_file())
        text = tres.read_text()
        self.assertEqual(re.findall(r'\[ext_resource type="Texture2D" path="([^"]+)"', text),
                         ["atlas.png"])
        self.assertNotIn("sprites/", text)  # no directory component at all
        # the atlas really is the file the .tres points to
        self.assertEqual((tres.parent / "atlas.png").is_file(), True)

    def test_write_tres_keeps_the_given_relative_atlas_path(self):
        """write_tres carries the atlas path through verbatim: the exported
        .tres is always atlas.png relative to itself, and a different
        relative path is not silently rewritten to an absolute one."""
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            f = tmp / "sf.tres"
            write_tres(f, "atlas.png",
                       [("courier_run_se", True, [(0, 0, 4, 1), (4, 0, 4, 1)])], fps=30)
            text = f.read_text()
            self.assertEqual(re.findall(r'\[ext_resource type="Texture2D" path="([^"]+)"', text),
                             ["atlas.png"])
            self.assertNotIn(str(tmp), text)  # nothing absolute leaked in
            self.assertIn('[gd_resource type="SpriteFrames" load_steps=4 format=3]', text.splitlines()[0])


if __name__ == "__main__":
    unittest.main()
