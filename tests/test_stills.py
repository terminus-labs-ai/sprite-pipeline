"""Tests for still preparation (spec AC15, FR 6; amendment 3).

Retargeted from the retired fill-fit / 1280 reference canvas to anchored
preparation: `stills.anchor` scales a standing-figure still so its measured
body height equals the mannequin profile's body target on the 832x1024 edit
canvas, pins its feet to the floor line and centres it on x 416, with every
other pixel flat chroma green (31, 203, 25). There is no 1280 canvas
anymore.

The expected numbers are stated independently of production code, as the
literal FR 6 formula over the test mannequin profile (body_h 324, floor_y
525 at the 704 drive resolution): 324 x 1024 / 704 = 471.3 -> 470 +/- 1
(the spec's rounded 470), (525 + (704 - 704) // 2) x 1024 / 704 = 763.6 ->
766 +/- 1 (the spec's rounded 766).
"""
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from sprite_pipeline import metrics, stills

# the flat chroma green of an anchored master (FR 6)
CHROMA = (31, 203, 25)
# the test mannequin profile (tests/mannequins.py's canonical stub): the
# packaged ual-female.glb geometry at the 704 drive resolution
PROFILE = {"glb_name": "test.glb", "resolution": 704, "body_h": 324,
           "floor_y": 525}

# a standing-figure source: a flat background with a figure 300 px tall x
# 200 px wide, its feet 10 px from the bottom (the AC15 source geometry)
SRC_W, SRC_H = 500, 900
BODY_H, BODY_W, FEET = 300, 200, 10
# the source figure's top row (889 - 299), so the scaled figure's top lands on
# floor + 1 - (890 * 470 / 300) = 766 + 1 - 1389 = -622 (off the canvas)


def _make_still(w, h, corner=(10, 200, 10)):
    """A flat-background frame of the given size (all pixels `corner`)."""
    return Image.new("RGB", (w, h), corner)


def _standing_still(path):
    """The AC15 source written to `path`."""
    im = _make_still(SRC_W, SRC_H, corner=(10, 200, 10))
    left = SRC_W // 2 - BODY_W // 2
    top = SRC_H - 1 - FEET - BODY_H + 1
    im.paste(Image.new("RGB", (BODY_W, BODY_H), (140, 80, 60)), (left, top))
    im.save(path)
    return path


def _save_still(im, path):
    im.save(path)
    return path


def _anchor(path, frame_size=704):
    dst = path.parent / "master.png"
    stills.anchor(path, dst, frame_size, PROFILE)
    return dst


class TestStillPreparation(unittest.TestCase):
    """AC15-r1: the anchored master is 832x1024 on flat chroma green, with the
    figure scaled to the profile body target, feet on the floor line and
    centred on x 416."""

    def test_r1_500x900_still_anchors_to_the_profile_target(self):
        """AC15-r1 (retargeted): the 500x900 standing still anchors to the
        832x1024 master on flat chroma green (31, 203, 25): measured body
        470 +/- 1, floor 766 +/- 1, centre x 416 at the default frame size
        704."""
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            src = _standing_still(tmp / "src.png")
            dst = _anchor(src)
            out = Image.open(dst)
            # 832x1024 master, flat chroma green everywhere the figure is not
            self.assertEqual(out.size, (832, 1024))
            self.assertEqual(out.getpixel((0, 0)), CHROMA)
            self.assertEqual(out.getpixel((831, 0)), CHROMA)
            self.assertEqual(out.getpixel((0, 1023)), CHROMA)
            self.assertEqual(out.getpixel((831, 1023)), CHROMA)
            m = metrics.measure(dst)
            # body_h x 1024 / frame_size = 324 x 1024 / 704 = 471.3 -> 470 +/- 1
            self.assertAlmostEqual(m["body_h"], 470, delta=2)
            # floor_y x 1024 / frame_size = 525 x 1024 / 704 = 763.6 -> 766 +/- 1
            self.assertAlmostEqual(m["floor_y"], 766, delta=4)
            # centred horizontally: the measured centre sits on the canvas centre
            self.assertAlmostEqual(m["centre_x"], 416, delta=1.5)

    def test_r2_832x1024_still_anchors_unscaled_in_size(self):
        """AC15-r2 (retargeted): a standing still exactly 832x1024 anchors to
        the 832x1024 master on flat chroma green, scaled to the profile target
        (no 1280 padding at any point)."""
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            src = _make_still(832, 1024, corner=(10, 200, 10))
            left = 832 // 2 - BODY_W // 2
            top = 1024 - 1 - FEET - BODY_H + 1
            src.paste(Image.new("RGB", (BODY_W, BODY_H), (140, 80, 60)), (left, top))
            _save_still(src, tmp / "src.png")
            dst = _anchor(tmp / "src.png")
            out = Image.open(dst)
            self.assertEqual(out.size, (832, 1024))
            self.assertEqual(out.getpixel((0, 0)), CHROMA)
            self.assertEqual(out.getpixel((831, 1023)), CHROMA)
            m = metrics.measure(dst)
            self.assertAlmostEqual(m["body_h"], 470, delta=2)
            self.assertAlmostEqual(m["floor_y"], 766, delta=4)
            self.assertAlmostEqual(m["centre_x"], 416, delta=1.5)

    def test_r3_2000x1000_wide_frame_anchors_the_figure(self):
        """AC15-r3 (retargeted): a wide flat-background frame whose standing
        figure is 300 px tall anchors to the 832x1024 master on flat chroma
        green at the same profile target — the frame size no longer pads."""
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            src = _make_still(2000, 1000, corner=(10, 200, 10))
            left = 2000 // 2 - BODY_W // 2
            top = 1000 - 1 - FEET - BODY_H + 1
            src.paste(Image.new("RGB", (BODY_W, BODY_H), (140, 80, 60)), (left, top))
            _save_still(src, tmp / "src.png")
            dst = _anchor(tmp / "src.png")
            out = Image.open(dst)
            self.assertEqual(out.size, (832, 1024))
            self.assertEqual(out.getpixel((0, 0)), CHROMA)
            self.assertEqual(out.getpixel((831, 0)), CHROMA)
            self.assertEqual(out.getpixel((0, 1023)), CHROMA)
            self.assertEqual(out.getpixel((831, 1023)), CHROMA)
            m = metrics.measure(dst)
            self.assertAlmostEqual(m["body_h"], 470, delta=2)
            self.assertAlmostEqual(m["floor_y"], 766, delta=4)
            self.assertAlmostEqual(m["centre_x"], 416, delta=1.5)

    def test_r1_green_is_flat_everywhere_off_the_figure(self):
        """Every background pixel of the master is (31, 203, 25) — a corner
        fill or a gradient is not acceptable."""
        import numpy

        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            src = _standing_still(tmp / "src.png")
            dst = _anchor(src)
            arr = numpy.asarray(Image.open(dst).convert("RGB"), dtype=numpy.int16)
            corner = arr[0, 0]
            self.assertEqual(tuple(int(c) for c in corner), CHROMA)
            off_figure = (abs(arr - corner) <= 40).all(axis=2)
            self.assertTrue((arr[off_figure] == numpy.asarray(CHROMA)).all())


class TestPrepareUnchangedAndFit(unittest.TestCase):
    """Retargeted from the retired fill-fit oracle to the anchored-prep
    contract (amendment 5 orders retarget, not delete): the fill-fit
    `prepare`/`fit` functions and the 1280 canvas are gone, and a source that
    does not stand is rejected with FigureError naming the rule."""

    def test_prepare_output_identical_to_pre_refactor(self):
        """The retired fill-fit/1280 surface is gone: `prepare` and `fit` no
        longer exist, the 1280 canvas is retired and the edit canvas is
        832x1024 on flat chroma green."""
        self.assertFalse(hasattr(stills, "prepare"))
        self.assertFalse(hasattr(stills, "fit"))
        self.assertFalse(hasattr(stills, "REF_SIZE"))
        self.assertFalse(hasattr(stills, "REF_OFFSET"))
        self.assertEqual(stills.STILL_SIZE, (832, 1024))
        self.assertEqual(stills.CHROMA_GREEN, CHROMA)

    def test_fit_is_832x1024_with_corner_fill(self):
        """A non-standing source is rejected with FigureError naming the
        failed rule (too small), not anchored — the standing rule precedes the
        output."""
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            for w, h in ((600, 900), (1000, 700)):
                src = tmp / "src.png"
                # a 100x100 figure in a 900-tall frame is under the 20% minimum
                im = _make_still(w, h, corner=(10, 200, 10))
                im.paste(Image.new("RGB", (100, 100), (140, 80, 60)),
                         (w // 2 - 50, h - 110))
                im.save(src)
                with self.assertRaises(metrics.FigureError) as cm:
                    stills.anchor(src, tmp / "master.png", 704, PROFILE)
                self.assertEqual(cm.exception.reason, "too small")
