"""Mannequin metrics: still measurement, the standing-figure check and the
per-GLB body-geometry profiles (spec FR 6, FR 14).

A still's figure is the bounding box of the pixels that differ from the
top-left corner colour by more than CORNER_DIFF (40) in any channel; `measure`
reports that bbox as `body_h`, `floor_y` (the bottom row — the feet line) and
`centre_x` (the bbox's horizontal centre). Under the drive camera the body
keeps its pixel size at larger canvases and the floor line shifts by
`(canvas - 704) // 2`, so `floor_for_canvas` converts a 704-res floor to
another drive canvas.

`check_standing` (FR 6) is the standing-figure rule: the bbox height is at
least MIN_HEIGHT_FRACTION (20%) of the image height, at least the bbox width,
and the bbox bottom lies in the lower LOWER_BAND_FRACTION (20%) of the image;
otherwise it raises FigureError naming the file and the failed rule (`too
wide`, `too small` or `feet not at the bottom`). A standing figure yields None.

Profiles are packaged data (`data/mannequin_profiles.json`, keyed by the
sha256 of the source GLB bytes). An entry holds `glb_name`, `resolution`
(704) and `dirs`: `body_h` and `floor_y` per isometric direction s, se, e,
ne, n (amendment 10), read as `profile["dirs"][d]["body_h"]`; the project
load gate (FR 4) rejects an entry lacking a generated direction. A user file
with the same schema, merged last per hash, lives in the host home
(`config.home() / "mannequin_profiles.json"`) and its path is injectable for
tests. An absent or invalid user file changes nothing.
"""
import hashlib
import json
from pathlib import Path

import numpy
from PIL import Image

from sprite_pipeline import config

# the drive camera's resolution: fixtures and profiles are measured at 704
# (spec FR 14); the body keeps its pixel size at larger canvases
DRIVE_RES = 704
# FR 6: pixels differing from the corner colour by more than this in any
# channel belong to the figure
CORNER_DIFF = 40
# FR 6 standing rule: the minimum body-height fraction of the image and the
# lower band the feet line must sit in
MIN_HEIGHT_FRACTION = 0.20
LOWER_BAND_FRACTION = 0.20

PACKAGED_PROFILES = Path(__file__).resolve().parent / "data" / "mannequin_profiles.json"
USER_PROFILE_NAME = "mannequin_profiles.json"


class FigureError(ValueError):
    """A still is not a standing figure (FR 6); `reason` names the failed rule."""

    def __init__(self, path, reason):
        self.path = path
        self.reason = reason
        super().__init__(f"{path}: {reason}")


def _figure_bbox(arr, corner):
    """(min_x, min_y, max_x, max_y) of the pixels differing from `corner` by
    more than CORNER_DIFF in any channel, or None when no pixel does."""
    diff = (abs(arr - corner) > CORNER_DIFF).any(axis=2)
    ys, xs = numpy.nonzero(diff)
    if len(ys) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


def measure(path):
    """Measure the figure of a still on a flat background (FR 6, FR 14).

    The figure is the bounding box of the pixels that differ from the top-left
    corner colour by more than CORNER_DIFF in any channel. Returns `body_h`
    (the bbox height in pixels), `floor_y` (the bbox's bottom row — the feet
    line) and `centre_x` (the bbox's horizontal centre). Raises FigureError
    when no pixel differs from the corner (nothing to measure).
    """
    im = Image.open(path).convert("RGB")
    arr = numpy.asarray(im, dtype=numpy.int16)
    corner = arr[0, 0]
    box = _figure_bbox(arr, corner)
    if box is None:
        raise FigureError(str(path), "nothing to measure (flat image)")
    min_x, min_y, max_x, max_y = box
    return {
        "body_h": max_y - min_y + 1,
        "floor_y": max_y,
        "centre_x": (min_x + max_x) / 2,
    }


def check_standing(path):
    """The standing-figure rule (FR 6) for one still.

    Returns None when the figure stands — bbox height at least
    MIN_HEIGHT_FRACTION of the image height, bbox height at least the bbox
    width, bbox bottom in the lower LOWER_BAND_FRACTION of the image.
    Otherwise raises FigureError naming the file and the failed rule
    (`too small`, `too wide` or `feet not at the bottom`).
    """
    im = Image.open(path).convert("RGB")
    arr = numpy.asarray(im, dtype=numpy.int16)
    h, w = arr.shape[:2]
    box = _figure_bbox(arr, arr[0, 0])
    if box is None:
        raise FigureError(str(path), "too small")
    min_x, min_y, max_x, max_y = box
    height = max_y - min_y + 1
    width = max_x - min_x + 1
    if height < MIN_HEIGHT_FRACTION * h:
        raise FigureError(str(path), "too small")
    if height < width:
        raise FigureError(str(path), "too wide")
    if max_y < h * (1 - LOWER_BAND_FRACTION):
        raise FigureError(str(path), "feet not at the bottom")
    return None


def floor_for_canvas(floor_y, canvas):
    """The drive-rule floor line at `canvas` for a 704-res `floor_y` (FR 14):
    the body keeps its pixel size and the floor line shifts by
    `(canvas - 704) // 2`."""
    return floor_y + (canvas - DRIVE_RES) // 2


def glb_sha256(path):
    """The profile key of a source GLB: the sha256 of its bytes (FR 14)."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _profiles_from(data):
    """The `{hash: profile}` map of one profiles document (the `profiles` key
    of the JSON schema)."""
    return {str(h): dict(p) for h, p in (data.get("profiles") or {}).items()}


def load_profiles(user_path=None):
    """The mannequin profiles keyed by GLB sha256 (FR 14).

    The packaged `data/mannequin_profiles.json` merged with the user file
    (`config.home() / mannequin_profiles.json`, or `user_path` when injected)
    per hash, last-write-wins: a user entry replaces the packaged entry for
    the same hash and untouched hashes survive. An absent or invalid user
    file changes nothing.
    """
    if user_path is None:
        user_path = config.home() / USER_PROFILE_NAME
    profiles = {k: dict(p) for k, p in _profiles_from(json.loads(PACKAGED_PROFILES.read_text())).items()}
    user_path = Path(user_path)
    if user_path.is_file():
        try:
            user = json.loads(user_path.read_text())
        except (json.JSONDecodeError, OSError):
            return profiles
        if not isinstance(user, dict):
            return profiles
        profiles.update(_profiles_from(user))
    return profiles


def profile_for(path, user_path=None):
    """The profile for the source GLB at `path`, or KeyError naming the
    command that measures one (FR 4, amendment 3)."""
    digest = glb_sha256(path)
    profiles = load_profiles(user_path)
    if digest not in profiles:
        raise KeyError(
            f"no mannequin profile for {Path(path).name} (sha256 {digest[:16]}…); "
            "run `sprite-pipeline mannequin-profile` for it")
    return profiles[digest]
