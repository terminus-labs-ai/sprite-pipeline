"""Tests for the mannequin metrics (spec AC14, FR 6 and FR 14).

Measurement (FR 14): the figure is the bounding box of the pixels differing
from the top-left corner colour by more than 40 in any channel; the ten
committed 704-res A_TPose fixtures (two GLBs x five directions) measure the
packaged per-direction `dirs` values (amendment 10). The standing
check (FR 6) accepts a standing figure and rejects a lying, too small or
floating figure with the named reason. Profile loading (FR 14) merges the
user file over the packaged profiles per hash, last-write-wins.

No network, GPU, Blender or ComfyUI needed.
"""
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from sprite_pipeline import metrics

FIXTURES = Path(__file__).parent / "fixtures"
# the isometric directions a profile carries (FR 14, amendment 10)
DIRS = ("s", "se", "e", "ne", "n")
# FR 14's measured values, stated independently of the packaged file:
# (body_h, floor_y) per GLB and direction at 704
FR14 = {
    "ual-female.glb": {"s": (331, 524), "se": (331, 532), "e": (325, 533),
                       "ne": (318, 534), "n": (306, 524)},
    "ual.glb": {"s": (354, 536), "se": (350, 539), "e": (328, 528),
                "ne": (320, 526), "n": (308, 518)},
}


def _dirs(body_h, floor_y):
    """A user-file `dirs` map with every direction at one geometry."""
    return {d: {"body_h": body_h, "floor_y": floor_y} for d in DIRS}

# the fixture's background colour (top-left pixel of both committed renders)
CORNER = (95, 189, 63)
# the synthetic figure's colour: differs from CORNER by 50 in every channel
# (> 40, so it is detected)
BODY = (145, 239, 113)
# a 324 px tall, 200 px wide body — tall enough for the standing rule
# (200 <= 324, 324 >= 20% of 704)
BODY_H = 324
BODY_W = 200


def _figure_frame(size, body_h, body_w, bottom, centre_x=None, colour=BODY):
    """A flat-corner frame with one solid rectangular figure of the given
    size whose bottom row is `bottom`, centred on `centre_x` (or the frame
    centre). Returns the in-memory PIL image."""
    size = (size, size)
    im = Image.new("RGB", size, CORNER)
    cx = size[0] // 2 if centre_x is None else centre_x
    x0, x1 = cx - body_w // 2, cx - body_w // 2 + body_w - 1
    y0, y1 = bottom - body_h + 1, bottom
    im.paste(Image.new("RGB", (body_w, body_h), colour), (x0, y0))
    return im


def _save(im, path):
    im.save(path)
    return path


class TestMeasure(unittest.TestCase):
    """AC14: corner-colour measurement (difference > 40) on the committed
    per-direction fixtures and a synthetic drive-rule frame (FR 14)."""

    def _assert_fixtures_match(self, sex, glb_name):
        """Each of the GLB's five fixtures measures the FR 14 value of its
        direction +/- 2, and the packaged `dirs` entry holds exactly that
        value."""
        pkg = json.loads(metrics.PACKAGED_PROFILES.read_text())["profiles"]
        packaged = next(p for p in pkg.values() if p["glb_name"] == glb_name)
        self.assertEqual(packaged["resolution"], 704)
        self.assertEqual(sorted(packaged["dirs"]), sorted(DIRS))
        for d in DIRS:
            want_h, want_floor = FR14[glb_name][d]
            with self.subTest(glb=glb_name, d=d):
                m = metrics.measure(FIXTURES / f"mannequin_apose_{sex}_{d}_704.png")
                self.assertAlmostEqual(m["body_h"], want_h, delta=2)
                self.assertAlmostEqual(m["floor_y"], want_floor, delta=2)
                self.assertEqual(packaged["dirs"][d],
                                 {"body_h": want_h, "floor_y": want_floor})

    def test_ac14_r1_female_fixtures_match_packaged_dirs(self):
        """AC14-r1: the five female 704 fixtures measure the packaged
        ual-female.glb `dirs` entries (s 331 / 524 ... n 306 / 524) +/- 2."""
        self._assert_fixtures_match("female", "ual-female.glb")

    def test_ac14_r1_male_fixtures_match_packaged_dirs(self):
        """AC14-r1: the five male 704 fixtures measure the packaged ual.glb
        `dirs` entries (s 354 / 536 ... n 308 / 518) +/- 2, read through the
        loader with no user file."""
        self._assert_fixtures_match("male", "ual.glb")
        profiles = metrics.load_profiles(user_path=Path("/nonexistent-mannequins"))
        male = next(p for p in profiles.values() if p["glb_name"] == "ual.glb")
        self.assertEqual(male["dirs"]["s"], {"body_h": 354, "floor_y": 536})

    def test_ac14_r2_synthetic_1152_drive_rule(self):
        """AC14-r2: a 1152 frame built to the drive rule from the female `s`
        render (the 704 frame pasted with everything shifted +224, the body at
        its own pixel size) measures body 331 +/- 2 and floor 748 +/- 2."""
        src = Image.open(FIXTURES / "mannequin_apose_female_s_704.png").convert("RGB")
        with tempfile.TemporaryDirectory() as tmp:
            im = Image.new("RGB", (1152, 1152), src.getpixel((0, 0)))
            im.paste(src, ((1152 - 704) // 2, (1152 - 704) // 2))
            p = _save(im, Path(tmp) / "frame1152.png")
            m = metrics.measure(p)
            self.assertAlmostEqual(m["body_h"], 331, delta=2)
            self.assertAlmostEqual(m["floor_y"], 748, delta=2)

    def test_ac14_r2_floor_for_canvas(self):
        """AC14-r2: the drive-rule floor conversion — body size unchanged,
        floor shifted (canvas-704)//2."""
        self.assertEqual(metrics.floor_for_canvas(524, 1152), 748)
        self.assertEqual(metrics.floor_for_canvas(524, 704), 524)
        self.assertEqual(metrics.floor_for_canvas(536, 1152), 760)
        self.assertEqual(metrics.floor_for_canvas(524, 1408), 876)

    def test_corner_difference_threshold(self):
        """A pixel differing from the corner by <= 40 in every channel is not
        the figure (the mask's > 40 rule)."""
        with tempfile.TemporaryDirectory() as tmp:
            im = _figure_frame(704, BODY_H, BODY_W, 500,
                               colour=(135, 229, 103))  # exactly +40 on all channels
            p = _save(im, Path(tmp) / "flat.png")
            with self.assertRaises(metrics.FigureError):
                metrics.measure(p)
            im = _figure_frame(704, BODY_H, BODY_W, 500,
                               colour=(136, 229, 103))  # +41 on the red channel
            p = _save(im, Path(tmp) / "fig.png")
            m = metrics.measure(p)
            self.assertEqual(m["body_h"], BODY_H)


class TestCheckStanding(unittest.TestCase):
    """AC14: the standing-figure rule (FR 6) accepts a standing figure and
    rejects a lying, too small or floating one with the named reason."""

    def _check(self, im):
        with tempfile.TemporaryDirectory() as tmp:
            p = _save(im, Path(tmp) / "fig.png")
            return metrics.check_standing(p)

    def test_ac14_r3_standing_figure_accepted(self):
        """AC14-r3: a standing figure (324x200 on 704, feet near the bottom)
        is accepted."""
        self.assertIsNone(self._check(_figure_frame(704, BODY_H, BODY_W, 672)))

    def test_ac14_r3_lying_figure_rejected(self):
        """AC14-r3: a lying figure (wider than tall, but over the height
        minimum so `too wide` is the rule that fails) is rejected `too wide`."""
        with self.assertRaises(metrics.FigureError) as ctx:
            self._check(_figure_frame(704, 200, 500, 672))  # 200 >= 140.8, 200 < 500
        self.assertEqual(ctx.exception.reason, "too wide")

    def test_ac14_r3_small_figure_rejected(self):
        """AC14-r3: a figure under 20% of the frame height is rejected
        `too small`."""
        with self.assertRaises(metrics.FigureError) as ctx:
            self._check(_figure_frame(704, 139, 100, 672))  # 139 < 0.2 * 704 = 140.8
        self.assertEqual(ctx.exception.reason, "too small")

    def test_ac14_r3_floating_figure_rejected(self):
        """AC14-r3: a figure whose bottom is above the lower 20% is rejected
        `feet not at the bottom`."""
        with self.assertRaises(metrics.FigureError) as ctx:
            self._check(_figure_frame(704, BODY_H, BODY_W, 500))  # 500 < 0.8 * 704 = 563.2
        self.assertEqual(ctx.exception.reason, "feet not at the bottom")

    def test_ac14_r3_rejections_are_isolated(self):
        """Each rejection names exactly one rule: the too-small, too-wide and
        floating figures each satisfy the other two rules, so a bug that
        reports the wrong reason (or a threshold that misclassifies) fails
        one of these."""
        # too small: wide (200 > 139) and bottom in the lower band (672)
        with self.assertRaises(metrics.FigureError) as ctx:
            self._check(_figure_frame(704, 139, 200, 672))
        self.assertEqual(ctx.exception.reason, "too small")
        # too wide: tall (200 >= 140.8) and bottom in the lower band (672)
        with self.assertRaises(metrics.FigureError) as ctx:
            self._check(_figure_frame(704, 200, 500, 672))
        self.assertEqual(ctx.exception.reason, "too wide")
        # floating: tall (324 >= 140.8) and taller than wide (324 > 200)
        with self.assertRaises(metrics.FigureError) as ctx:
            self._check(_figure_frame(704, BODY_H, BODY_W, 500))
        self.assertEqual(ctx.exception.reason, "feet not at the bottom")


class TestLoadProfiles(unittest.TestCase):
    """AC14: profile loading merges the user file over the packaged profiles
    per hash, last-write-wins (FR 14, amendment 4)."""

    def _packaged(self):
        return json.loads(metrics.PACKAGED_PROFILES.read_text())["profiles"]

    def test_ac14_r4_merge_last_write_wins(self):
        """AC14-r4: a user entry for a known hash replaces the packaged entry
        and unrelated packaged entries survive."""
        with tempfile.TemporaryDirectory() as tmp:
            user = Path(tmp) / "mannequin_profiles.json"
            known = next(k for k, p in self._packaged().items()
                         if p["glb_name"] == "ual-female.glb")
            other = next(k for k in self._packaged() if k != known)
            user.write_text(json.dumps({"profiles": {
                known: {"glb_name": "ual-female.glb", "resolution": 704,
                        "dirs": _dirs(320, 500)},
                "e" * 64: {"glb_name": "new.glb", "resolution": 704,
                           "dirs": _dirs(100, 200)},
            }}))
            profiles = metrics.load_profiles(user_path=user)
            # user entry wins over the packaged one, last-write-wins: the
            # whole entry is replaced, every direction included
            self.assertEqual(profiles[known]["dirs"], _dirs(320, 500))
            # the unrelated packaged entry survives
            self.assertEqual(profiles[other], self._packaged()[other])
            # the user's new hash appears
            self.assertEqual(profiles["e" * 64]["glb_name"], "new.glb")

    def test_ac14_r4_user_entry_replaces_packaged(self):
        """AC14-r4: a user entry for a known hash replaces the packaged entry
        (last-write-wins): the female profile's `dirs.s.body_h` becomes the
        user value while the rest of the entry is intact."""
        with tempfile.TemporaryDirectory() as tmp:
            user = Path(tmp) / "mannequin_profiles.json"
            known = next(k for k, p in self._packaged().items()
                         if p["glb_name"] == "ual-female.glb")
            user.write_text(json.dumps({"profiles": {
                known: {"glb_name": "ual-female.glb", "resolution": 704,
                        "dirs": _dirs(999, 500)}}}))
            merged = metrics.load_profiles(user_path=user)[known]
            self.assertEqual(merged["dirs"]["s"]["body_h"], 999)
            self.assertEqual(merged["glb_name"], "ual-female.glb")

    def test_ac14_r4_absent_user_file_changes_nothing(self):
        """An absent user file leaves the packaged profiles unchanged."""
        profiles = metrics.load_profiles(user_path=Path("/nonexistent-mannequins"))
        self.assertEqual(profiles, self._packaged())

    def test_ac14_r4_invalid_user_file_changes_nothing(self):
        """An invalid (unparseable / non-mapping) user file leaves the
        packaged profiles unchanged."""
        with tempfile.TemporaryDirectory() as tmp:
            bad = Path(tmp) / "bad.json"
            bad.write_text("{not json")
            self.assertEqual(metrics.load_profiles(user_path=bad), self._packaged())
            notmap = Path(tmp) / "notmap.json"
            notmap.write_text(json.dumps(["no", "profiles", "key"]))
            self.assertEqual(metrics.load_profiles(user_path=notmap), self._packaged())

    def test_profile_for_hashes_the_glb_bytes(self):
        """profile_for resolves by the sha256 of the GLB bytes: a file whose
        content hashes to a hash registered only in the user file resolves to
        that entry, and an unknown file raises KeyError naming the
        mannequin-profile command."""
        payload = b"synthetic glb bytes"
        digest = hashlib.sha256(payload).hexdigest()
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            user = tmp / "mannequin_profiles.json"
            entry = {"glb_name": "synthetic.glb", "resolution": 704,
                     "dirs": _dirs(333, 520)}
            user.write_text(json.dumps({"profiles": {digest: entry}}))
            glb = tmp / "synthetic.glb"
            glb.write_bytes(payload)
            self.assertEqual(metrics.profile_for(glb, user_path=user), entry)
            unknown = tmp / "unknown.glb"
            unknown.write_bytes(payload + b" (changed)")
            with self.assertRaises(KeyError) as ctx:
                metrics.profile_for(unknown, user_path=user)
            self.assertIn("mannequin-profile", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
