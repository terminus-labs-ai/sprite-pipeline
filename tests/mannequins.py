"""Test-side mannequin profiles and standing-figure images (AC15 / FR 4, FR 6, FR 14).

`project.load_project`'s profile gate (FR 4) rejects a source GLB with no
mannequin profile, and every stub GLB in this suite hashes to an unknown
value. Tests that load a project or run `go` need their stub bytes
registered — the test-visible stand-in for the packaged
`data/mannequin_profiles.json`; `tests/conftest.py` merges this registry into
`project.anchor_profiles`. Each fixture that writes stub GLB bytes registers
those exact bytes with `register`; nothing is discovered from the filesystem.
A test that wants the gate to *reject* writes bytes no fixture registers.

`standing_still` and `pose_frame` build the images the FR 6 rules are tested
against: a flat-background standing figure, and a figure placed by rule
(too small, too wide, floating) in a frame of a given size.
"""
import hashlib

from PIL import Image

# the canonical test mannequin at resolution 704: the packaged ual-female.glb
# per-direction values (FR 14, amendment 10), body_h / floor_y per direction;
# BODY_H / FLOOR_Y are its `s` (master) values, 331 / 524
DIRS = {
    "s": {"body_h": 331, "floor_y": 524},
    "se": {"body_h": 367, "floor_y": 532},
    "e": {"body_h": 399, "floor_y": 533},
    "ne": {"body_h": 354, "floor_y": 534},
    "n": {"body_h": 306, "floor_y": 524},
}
BODY_H = DIRS["s"]["body_h"]
FLOOR_Y = DIRS["s"]["floor_y"]
RESOLUTION = 704
# the flat background a drive render comes back on, and the mannequin's own
# colour (differing from it by more than 40 in every channel, FR 6's mask)
CORNER = (95, 189, 63)
FIGURE = (145, 239, 113)
# the flat background of a test still, and the colour of its figure
STILL_CORNER = (10, 200, 10)
STILL_FIGURE = (140, 80, 60)

_REGISTRY = {}


def profile(dirs=None, *, glb_name="test.glb", resolution=RESOLUTION):
    """A mannequin profile in the FR 14 `dirs` schema: `dirs` (default the
    canonical DIRS) maps each direction to its `body_h` / `floor_y`."""
    dirs = DIRS if dirs is None else dirs
    return {"glb_name": glb_name, "resolution": resolution,
            "dirs": {d: dict(v) for d, v in dirs.items()}}


def register(payload, *, glb_name="test.glb", dirs=None, resolution=RESOLUTION):
    """Register the mannequin profile for a source GLB's bytes, keyed by their
    sha256, so the profile gate (FR 4) accepts them. Returns the digest."""
    digest = hashlib.sha256(payload).hexdigest()
    _REGISTRY.setdefault(digest, profile(dirs, glb_name=glb_name,
                                         resolution=resolution))
    return digest


def profiles():
    """The registered profiles in the `{hash: profile}` schema."""
    return {k: profile(v["dirs"], glb_name=v["glb_name"], resolution=v["resolution"])
            for k, v in _REGISTRY.items()}


def tpose_render(body_h=BODY_H, floor_y=FLOOR_Y, resolution=RESOLUTION):
    """A drive-camera A_TPose render of a mannequin measuring exactly
    `body_h` with its feet row at `floor_y`: one solid figure centred
    horizontally on the frame."""
    im = Image.new("RGB", (resolution, resolution), CORNER)
    width = max(1, body_h // 2)
    left = resolution // 2 - width // 2
    im.paste(Image.new("RGB", (width, body_h), FIGURE),
             (left, floor_y - body_h + 1))
    return im


def tpose_png_bytes(**kwargs):
    """The A_TPose render as PNG bytes."""
    import io

    buf = io.BytesIO()
    tpose_render(**kwargs).save(buf, format="PNG")
    return buf.getvalue()


def pose_frame(width, height, *, body_h, body_w, feet_from_bottom=None,
               bottom=None, centre_x=None, corner=STILL_CORNER,
               colour=STILL_FIGURE):
    """A flat-background frame with one solid rectangular figure: `body_h` px
    tall, `body_w` px wide, its bottom row `bottom` (or `height - 1 -
    feet_from_bottom`), centred on `centre_x` (or the frame's centre)."""
    if bottom is None:
        bottom = height - 1 - feet_from_bottom
    im = Image.new("RGB", (width, height), corner)
    cx = width // 2 if centre_x is None else centre_x
    left = cx - body_w // 2
    im.paste(Image.new("RGB", (body_w, body_h), colour), (left, bottom - body_h + 1))
    return im


def standing_still(width=500, height=900, *, body_h=300, body_w=200,
                   feet_from_bottom=10, **kwargs):
    """A still the FR 6 standing rule accepts: a figure at least 20% of the
    frame height, no wider than tall, feet inside the lower 20% band."""
    return pose_frame(width, height, body_h=body_h, body_w=body_w,
                      feet_from_bottom=feet_from_bottom, **kwargs)
