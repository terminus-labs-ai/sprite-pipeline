"""Amendment 14 slice 19 hygiene (spec AC22-AC24): one centre-crop rule,
the guide/edit path helpers and one test-side guide line. No behaviour
change: expected values are stated here, independently of production code.
"""
import ast
import inspect
import tempfile
import unittest
from pathlib import Path

import numpy
from PIL import Image

from test_workflow_paths import cfg, stubbed_example

from sprite_pipeline import project, stages, stills

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src" / "sprite_pipeline"
TESTS = Path(__file__).resolve().parent
GUIDE_START = "Image 2 is a plain grey mannequin"


def synthetic(w, h):
    """A still whose every pixel differs, so any shifted crop shows."""
    yy, xx = numpy.mgrid[0:h, 0:w]
    a = numpy.stack([xx % 256, yy % 256, (xx * 7 + yy * 13) % 256], axis=-1)
    return Image.fromarray(a.astype(numpy.uint8), "RGB")


def expected_pad(im, frame, canvas):
    """The base commit's pad_ref, with its inline offsets."""
    w, h = im.size
    x = round((w - h) / 2) if w > h else 0
    y = round((h - w) / 2) if h > w else 0
    sq = im.crop((x, y, w - x, h - y)).resize((frame, frame), Image.LANCZOS)
    out = Image.new("RGB", (canvas, canvas), sq.getpixel((0, 0)))
    off = (canvas - frame) // 2
    out.paste(sq, (off, off))
    return out


class TestCentreCrop(unittest.TestCase):
    def test_ac22_r1_offsets(self):
        self.assertEqual(stills.centre_crop(832, 1024), (0, 96))
        self.assertEqual(stills.centre_crop(1024, 832), (96, 0))
        self.assertEqual(stills.centre_crop(832, 832), (0, 0))

    def test_ac22_r2_crop_y(self):
        self.assertEqual(stills.CROP_Y, 96)
        self.assertEqual(stills.CROP_Y,
                         stills.centre_crop(stills.CANVAS_W, stills.CANVAS_H)[1])

    def test_ac22_r3_pad_ref_pixel_identical(self):
        with tempfile.TemporaryDirectory() as t:
            for w, h in [(832, 1024), (1024, 832), (832, 832)]:
                for canvas in (704, 1152):
                    src = Path(t) / f"{w}x{h}.png"
                    dst = Path(t) / f"out-{w}x{h}-{canvas}.png"
                    im = synthetic(w, h)
                    im.save(src)
                    stages.pad_ref(src, dst, 704, canvas, [])
                    got = numpy.asarray(Image.open(dst).convert("RGB"))
                    want = numpy.asarray(expected_pad(im, 704, canvas))
                    self.assertTrue(numpy.array_equal(got, want),
                                    f"{w}x{h} on {canvas}")

    def test_ac22_r4_pad_ref_uses_centre_crop(self):
        src = inspect.getsource(stages.pad_ref)
        self.assertIn("stills.centre_crop(", src)
        self.assertNotIn("round((w - h) / 2)", src)
        self.assertNotIn("round((h - w) / 2)", src)


class TestGuideEditPaths(unittest.TestCase):
    def test_ac23_r1_methods(self):
        with tempfile.TemporaryDirectory() as t:
            proj = project.load_project(stubbed_example(t)[0])
            ctx = stages.StageCtx(proj, cfg(Path("/nonexistent-home")))
            master = proj.output_root() / "vexa" / "master"
            self.assertEqual(ctx.master("vexa"), master)
            self.assertEqual(ctx.guide_file("vexa", "se"), master / "se.guide.png")
            self.assertEqual(ctx.edit_file("vexa", "se"), master / "se.edit.png")

    def test_ac23_r2_no_literals_outside_methods(self):
        for name in ("cli.py", "stages.py"):
            tree = ast.parse((SRC / name).read_text())
            allowed, docs = set(), set()
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.ClassDef, ast.Module,
                                     ast.AsyncFunctionDef)):
                    body = node.body
                    if (body and isinstance(body[0], ast.Expr)
                            and isinstance(body[0].value, ast.Constant)
                            and isinstance(body[0].value.value, str)):
                        docs.add(id(body[0].value))
                    if (isinstance(node, ast.FunctionDef)
                            and node.name in ("guide_file", "edit_file")):
                        allowed.update(id(n) for n in ast.walk(node))
            hits = [n.lineno for n in ast.walk(tree)
                    if isinstance(n, ast.Constant) and isinstance(n.value, str)
                    and (".guide.png" in n.value or ".edit.png" in n.value)
                    and id(n) not in allowed and id(n) not in docs]
            self.assertEqual(hits, [], f"{name} lines {hits}")


class TestGuideLineText(unittest.TestCase):
    def test_ac24_r1_prompt_text_is_literal(self):
        src = (TESTS / "prompt_text.py").read_text()
        tree = ast.parse(src)
        imports = [n for n in ast.walk(tree)
                   if isinstance(n, (ast.Import, ast.ImportFrom))]
        self.assertEqual(imports, [])
        import prompt_text
        self.assertEqual(sorted(prompt_text.GUIDE_LINE), ["he", "she", "they"])
        self.assertEqual(
            prompt_text.GUIDE_LINE["they"],
            "Image 2 is a plain grey mannequin used only as a guide: draw them in "
            "exactly the mannequin's pose, facing direction, size and position in "
            "the frame, with their feet exactly where the mannequin's feet are and "
            "their arms held out like the mannequin's. Do not draw the mannequin "
            "itself.")

    def test_ac24_r2_tests_take_it_from_prompt_text(self):
        for name in ("test_dir_anchor.py", "test_review.py", "test_comfy_cli.py"):
            src = (TESTS / name).read_text()
            self.assertNotIn(GUIDE_START, src, name)
            self.assertIn("from prompt_text import GUIDE_LINE", src, name)


if __name__ == "__main__":
    unittest.main()
