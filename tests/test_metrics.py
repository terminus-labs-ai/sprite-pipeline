"""Tests for the mannequin metrics (spec AC14, FR 6 and FR 14).

Measurement (FR 14): the figure is the bounding box of the pixels differing
from the top-left corner colour by more than 40 in any channel; the committed
704-res A_TPose fixtures measure the packaged profile values. The standing
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
FEMALE_FIXTURE = FIXTURES / "mannequin_tpose_female_s_704.png"
MALE_FIXTURE = FIXTURES / "mannequin_tpose_male_s_704.png"

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
    fixtures and a synthetic drive-rule frame (FR 14)."""

    def test_r1_female_fixture_matches_packaged_profile(self):
        """AC14-r1: the female 704 fixture measures 324 / 525 / 352 and equals
        the packaged ual-female.glb profile."""
        m = metrics.measure(FEMALE_FIXTURE)
        self.assertEqual(m["body_h"], 324)
        self.assertAlmostEqual(m["floor_y"], 525, delta=2)
        self.assertAlmostEqual(m["centre_x"], 352, delta=1)
        pkg = json.loads(metrics.PACKAGED_PROFILES.read_text())["profiles"]
        female = next(p for p in pkg.values() if p["glb_name"] == "ual-female.glb")
        self.assertEqual(female["body_h"], m["body_h"])
        self.assertAlmostEqual(female["floor_y"], m["floor_y"], delta=2)
        self.assertEqual(female["resolution"], 704)

    def test_r1_user_entry_replaces_packaged(self):
        """AC14-r8: a user entry for a known hash replaces the packaged entry
        (last-write-wins): the female profile's body_h becomes the user value
        while the rest of the profile is intact."""
        with tempfile.TemporaryDirectory() as tmp:
            user = Path(tmp) / "mannequin_profiles.json"
            pkg = json.loads(metrics.PACKAGED_PROFILES.read_text())["profiles"]
            known = next(k for k, p in pkg.items() if p["glb_name"] == "ual-female.glb")
            user.write_text(json.dumps({"profiles": {
                known: {"glb_name": "ual-female.glb", "resolution": 704,
                        "body_h": 999, "floor_y": 500}}}))
            merged = metrics.load_profiles(user_path=user)[known]
            self.assertEqual(merged["body_h"], 999)
            self.assertEqual(merged["glb_name"], "ual-female.glb")

    def test_r2_male_fixture_matches_packaged_profile(self):
        """AC14-r2: the male 704 fixture measures 345 / 538 and equals the
        packaged ual.glb profile."""
        m = metrics.measure(MALE_FIXTURE)
        self.assertEqual(m["body_h"], 345)
        self.assertAlmostEqual(m["floor_y"], 538, delta=2)
        profiles = metrics.load_profiles(user_path=Path("/nonexistent-mannequins"))
        profile = profiles[next(k for k, p in profiles.items()
                                if p["glb_name"] == "ual.glb")]
        self.assertEqual(profile["body_h"], m["body_h"])
        self.assertAlmostEqual(profile["floor_y"], m["floor_y"], delta=2)

    def test_r3_synthetic_1152_drive_rule(self):
        """AC14-r3: a 1152 frame built to the drive rule (same 324 px body,
        everything shifted +224) measures the same body height and floor 749."""
        with tempfile.TemporaryDirectory() as tmp:
            # the female body's measured geometry on 704: 324 px tall, 392 px
            # wide, bottom row 524, centre x 351.5. Drive rule: the body keeps
            # its pixel size and the floor line shifts (1152-704)//2 = 224.
            im = _figure_frame(1152, BODY_H, 392, 524 + 224, centre_x=351)
            p = _save(im, Path(tmp) / "frame1152.png")
            m = metrics.measure(p)
            self.assertEqual(m["body_h"], 324)
            self.assertAlmostEqual(m["floor_y"], 749, delta=2)

    def test_r3_floor_for_canvas(self):
        """AC14-r3: the drive-rule floor conversion — body size unchanged,
        floor shifted (canvas-704)//2, centre on the canvas."""
        self.assertEqual(metrics.floor_for_canvas(525, 1152), 525 + 224)
        self.assertEqual(metrics.floor_for_canvas(525, 704), 525)
        self.assertEqual(metrics.floor_for_canvas(538, 1152), 538 + 224)
        self.assertEqual(metrics.floor_for_canvas(525, 1408), 525 + 352)

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

    def test_r4_standing_figure_accepted(self):
        """AC14-r4: a standing figure (324x200 on 704, feet near the bottom)
        is accepted."""
        self.assertIsNone(self._check(_figure_frame(704, BODY_H, BODY_W, 672)))

    def test_r5_lying_figure_rejected(self):
        """AC14-r5: a lying figure (wider than tall, but over the height
        minimum so `too wide` is the rule that fails) is rejected `too wide`."""
        with self.assertRaises(metrics.FigureError) as ctx:
            self._check(_figure_frame(704, 200, 500, 672))  # 200 >= 140.8, 200 < 500
        self.assertEqual(ctx.exception.reason, "too wide")

    def test_r6_small_figure_rejected(self):
        """AC14-r6: a figure under 20% of the frame height is rejected
        `too small`."""
        with self.assertRaises(metrics.FigureError) as ctx:
            self._check(_figure_frame(704, 139, 100, 672))  # 139 < 0.2 * 704 = 140.8
        self.assertEqual(ctx.exception.reason, "too small")

    def test_r7_floating_figure_rejected(self):
        """AC14-r7: a figure whose bottom is above the lower 20% is rejected
        `feet not at the bottom`."""
        with self.assertRaises(metrics.FigureError) as ctx:
            self._check(_figure_frame(704, BODY_H, BODY_W, 500))  # 500 < 0.8 * 704 = 563.2
        self.assertEqual(ctx.exception.reason, "feet not at the bottom")

    def test_rejections_are_isolated(self):
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

    def test_r8_merge_last_write_wins(self):
        """AC14-r8: a user entry for a known hash replaces the packaged entry
        and unrelated packaged entries survive."""
        with tempfile.TemporaryDirectory() as tmp:
            user = Path(tmp) / "mannequin_profiles.json"
            known = next(k for k, p in self._packaged().items()
                         if p["glb_name"] == "ual-female.glb")
            other = next(k for k in self._packaged() if k != known)
            user.write_text(json.dumps({"profiles": {
                known: {"glb_name": "ual-female.glb", "resolution": 704,
                        "body_h": 320, "floor_y": 500},
                "e" * 64: {"glb_name": "new.glb", "resolution": 704,
                           "body_h": 100, "floor_y": 200},
            }}))
            profiles = metrics.load_profiles(user_path=user)
            # user entry wins over the packaged one, last-write-wins
            self.assertEqual(profiles[known]["body_h"], 320)
            self.assertEqual(profiles[known]["floor_y"], 500)
            # the unrelated packaged entry survives
            self.assertEqual(profiles[other], self._packaged()[other])
            # the user's new hash appears
            self.assertEqual(profiles["e" * 64]["glb_name"], "new.glb")

    def test_r8_absent_user_file_changes_nothing(self):
        """An absent user file leaves the packaged profiles unchanged."""
        profiles = metrics.load_profiles(user_path=Path("/nonexistent-mannequins"))
        self.assertEqual(profiles, self._packaged())

    def test_r8_invalid_user_file_changes_nothing(self):
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
                     "body_h": 333, "floor_y": 520}
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
