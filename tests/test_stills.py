"""Tests for still preparation (spec AC3, FR 6).

A still of any size on a flat background is scaled to fit the reference
canvas, centred, and padded with its corner colour.

No network, GPU, Blender, or ComfyUI needed.
"""
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from sprite_pipeline import stills


def _make_still(w, h, corner=(10, 200, 10)):
    """Create a still of the given size with the given corner colour (all pixels)."""
    im = Image.new("RGB", (w, h), corner)
    return im


def _save_still(im, path):
    im.save(path)
    return path


class TestStillPreparation(unittest.TestCase):
    """AC3: a still of any size on a flat background is scaled to fit the
    reference canvas, centred, and padded with its corner colour."""

    def test_r1_500x700_still_scaled_and_padded(self):
        """AC3-r1: 500×700 still with corner colour (10,200,10) → 1280×1280
        output; still scaled to fit 832×1024 keeping aspect (731×1024) and
        centred; every pixel outside it is (10,200,10)."""
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            src = tmp / "src.png"
            dst = tmp / "dst.png"
            im = _make_still(500, 700, corner=(10, 200, 10))
            _save_still(im, src)
            stills.prepare(str(src), str(dst))
            out = Image.open(dst)
            # 1280×1280 output
            self.assertEqual(out.size, (1280, 1280))
            # scaled to fit 832×1024 keeping aspect: 500/700 = 0.714, 832/1024 = 0.8125
            # scale = min(832/500, 1024/700) = min(1.664, 1.463) = 1.463
            # new_w = round(500 * 1.463) = 731, new_h = round(700 * 1.463) = 1024
            self.assertEqual(out.size, (1280, 1280))
            # centred within 832×1024 at (224, 128)
            x = 224 + (832 - 731) // 2  # 224 + 50 = 274
            y = 128 + (1024 - 1024) // 2  # 128 + 0 = 128
            # every pixel outside the scaled still is (10,200,10)
            self.assertEqual(out.getpixel((0, 0)), (10, 200, 10))
            self.assertEqual(out.getpixel((1279, 1279)), (10, 200, 10))
            # the scaled still is at (274, 128) with size 731×1024
            # check a pixel inside the scaled still (should be the corner colour too,
            # since the whole still is one colour)
            self.assertEqual(out.getpixel((274 + 100, 128 + 100)), (10, 200, 10))

    def test_r2_832x1024_still_pasted_unscaled(self):
        """AC3-r2: exactly 832×1024 still → pasted unscaled at (224,128)."""
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            src = tmp / "src.png"
            dst = tmp / "dst.png"
            im = _make_still(832, 1024, corner=(50, 100, 150))
            _save_still(im, src)
            stills.prepare(str(src), str(dst))
            out = Image.open(dst)
            self.assertEqual(out.size, (1280, 1280))
            # pasted unscaled at (224, 128)
            # check a pixel at the top-left of the pasted still
            self.assertEqual(out.getpixel((224, 128)), (50, 100, 150))
            # check a pixel outside the pasted still (should be the corner colour)
            self.assertEqual(out.getpixel((0, 0)), (50, 100, 150))

    def test_r3_2000x1000_still_scaled(self):
        """AC3-r3: 2000×1000 still → scaled to 832×416, centred."""
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            src = tmp / "src.png"
            dst = tmp / "dst.png"
            im = _make_still(2000, 1000, corner=(200, 50, 100))
            _save_still(im, src)
            stills.prepare(str(src), str(dst))
            out = Image.open(dst)
            self.assertEqual(out.size, (1280, 1280))
            # scaled to fit 832×1024 keeping aspect: 2000/1000 = 2.0, 832/1024 = 0.8125
            # scale = min(832/2000, 1024/1000) = min(0.416, 1.024) = 0.416
            # new_w = round(2000 * 0.416) = 832, new_h = round(1000 * 0.416) = 416
            x = 224 + (832 - 832) // 2  # 224 + 0 = 224
            y = 128 + (1024 - 416) // 2  # 128 + 304 = 432
            # every pixel outside the scaled still is the corner colour
            self.assertEqual(out.getpixel((0, 0)), (200, 50, 100))
            self.assertEqual(out.getpixel((1279, 1279)), (200, 50, 100))
            # check a pixel inside the scaled still
            self.assertEqual(out.getpixel((224 + 100, 432 + 100)), (200, 50, 100))


def _legacy_prepare(src, dst):
    """The pre-refactor prepare, kept verbatim as the byte-for-byte oracle."""
    im = Image.open(src).convert("RGB")
    w, h = im.size
    scale = min(stills.STILL_SIZE[0] / w, stills.STILL_SIZE[1] / h)
    new_w, new_h = round(w * scale), round(h * scale)
    im = im.resize((new_w, new_h), Image.LANCZOS)
    canvas = Image.new("RGB", (stills.REF_SIZE, stills.REF_SIZE), im.getpixel((0, 0)))
    canvas.paste(im, (stills.REF_OFFSET[0] + (stills.STILL_SIZE[0] - new_w) // 2,
                      stills.REF_OFFSET[1] + (stills.STILL_SIZE[1] - new_h) // 2))
    canvas.save(dst)


class TestPrepareUnchangedAndFit(unittest.TestCase):
    def _src(self, tmp, w, h):
        im = Image.new("RGB", (w, h), (10, 200, 10))
        im.paste((200, 20, 20), (w // 4, h // 4, w // 2, h // 2))
        return _save_still(im, tmp / "src.png")

    def test_prepare_output_identical_to_pre_refactor(self):
        for w, h in ((600, 900), (1000, 700), (832, 1024)):
            with tempfile.TemporaryDirectory() as tmp:
                tmp = Path(tmp)
                src = self._src(tmp, w, h)
                _legacy_prepare(src, tmp / "old.png")
                stills.prepare(src, tmp / "new.png")
                self.assertEqual((tmp / "old.png").read_bytes(),
                                 (tmp / "new.png").read_bytes(), (w, h))

    def test_fit_is_832x1024_with_corner_fill(self):
        for w, h in ((600, 900), (1000, 700)):
            with tempfile.TemporaryDirectory() as tmp:
                tmp = Path(tmp)
                out = stills.fit(self._src(tmp, w, h), tmp / "fit.png")
                self.assertEqual(out.size, stills.STILL_SIZE)
                self.assertEqual(Image.open(tmp / "fit.png").size, stills.STILL_SIZE)
                self.assertEqual(out.getpixel((0, 0)), (10, 200, 10))
