"""Anchored still preparation, direction-edit prompts and the review sheet
(spec FR 6-8, amendments 3, 9, 10 and 11).

A character's still must be an A- or T-pose standing figure on a flat
background. `check` (FR 6) measures it with `metrics` and rejects a figure that
does not stand, naming the file and the failed rule (`too wide`, `too small` or
`feet not at the bottom`).

`anchor` then prepares the anchored still `master/still.png` against the
anchor profile's `s`
direction (FR 14 `dirs`): the still is scaled uniformly so its measured body
height equals the target on the 832x1024 edit canvas — `dirs.s.body_h * 832 /
frame_size` px (391.2 for the packaged ual-female.glb profile at 704) — with
its feet pinned to floor `floor_y(frame_size) * 832 / frame_size + 96` (715.27
there) and the figure centred horizontally (x 416). No mannequin is a code
default: the profile and direction are always passed in. Every other pixel
is flat chroma green (CHROMA_GREEN). The 832 and 96 are the packaged animate
workflow's reference mapping: it centre-crops the 832x1024 still to 832x832
(rows 96-927, CROP_Y = (CANVAS_H - CANVAS_W) // 2) and scales it by
`frame_size / 832`. Anchoring reproduces the drive-video geometry after that
centre crop and downscale, so Wan Animate receives a ref framed like the
mannequin it follows. The anchored still replaces the old fill-fit plus 1280
reference canvas: there is no 1280 canvas anymore.

`guide` (FR 7) places a mannequin render facing a direction on the edit
canvas at that direction's target (silhouette: difference > GUIDE_DIFF from
the render's corner, plus every region it fully encloses) and `reanchor`
scales and pastes a direction's raw edit the same way, without the
standing-figure rule; `anchor`, `guide` and `reanchor` share one
scale-and-paste (`_place`).

`dir_prompt` (FR 7) is the edit-workflow prompt for one direction: the view's
turn prompt for that direction, wrapped in the character's pronoun, plus the
guide line (GUIDE_LINE). `review_sheet`
(FR 8) writes the review sheet `refs-sheet.png`: the direction stills side by
side, each labelled underneath.
"""
from pathlib import Path

import numpy

from sprite_pipeline import metrics

STILL_SIZE = (832, 1024)
# the drive camera's canvas and the edit canvas it is composited onto (FR 6)
DRIVE_RES = 704
CANVAS_W = 832
CANVAS_H = 1024
# the animate workflow's centre crop of the edit canvas to a square (FR 6,
# amendment 9): rows CROP_Y .. CROP_Y + CANVAS_W - 1 (96-927) are kept
CROP_Y = (CANVAS_H - CANVAS_W) // 2
# the flat chroma background of an anchored master (FR 6)
CHROMA_GREEN = (31, 203, 25)

# FR 7: a mannequin render's silhouette is every pixel differing from its
# corner colour by more than this (its grey shading sits close to its grey
# background)
GUIDE_DIFF = 4
# FR 7: the guide line appended to a direction edit's prompt (dec-732)
GUIDE_LINE = ("Image 2 is a plain grey mannequin used only as a guide: draw {obj} in "
              "exactly the mannequin's pose, facing direction, size and position in "
              "the frame, with {pos} feet exactly where the mannequin's feet are and "
              "{pos} arms held out like the mannequin's. Do not draw the mannequin "
              "itself.")

# the edit prompt's framing, from the new-kit branch (dir_prompt): the turn
# prompt keeps the character, camera and style; only the facing changes
STYLE = "hand-painted cel shading, bold dark outlines, crisp edges"
TAIL = " Full body, plain flat pure green background, no shadow."

# pronoun -> (subject, object, possessive, verb suffix: "keep" + "s" for he/she)
PRONOUNS = {
    "he": ("he", "him", "his", "s"),
    "she": ("she", "her", "her", "s"),
    "they": ("they", "them", "their", ""),
}


def pronoun_forms(pronoun):
    """The view-line placeholders for a pronoun: {subj} (sentence start),
    {sub} (mid-sentence), {obj}, {pos}, {pos_cap} (sentence start) and {s}."""
    if pronoun not in PRONOUNS:
        raise ValueError(f"pronoun {pronoun!r} (he, she or they)")
    subj, obj, pos, s = PRONOUNS[pronoun]
    return {"subj": subj.capitalize(), "sub": subj, "obj": obj, "pos": pos, "pos_cap": pos.capitalize(), "s": s}


def check(src):
    """FR 6: the still must be an A- or T-pose standing figure. Returns the
    measurement (`body_h`, `floor_y`, `centre_x`) when it stands; raises
    metrics.FigureError naming the file and the failed rule when it does not."""
    metrics.check_standing(src)
    return metrics.measure(src)


def anchor_floor(floor_y, frame_size):
    """FR 6 + FR 14: the floor row in px of the anchored master on the
    832x1024 edit canvas for a 704-res `floor_y` and the drive `frame_size`:
    FR 14's drive rule shifts the 704-res floor by
    `(max(frame_size, 704) - 704) // 2`, and the animate workflow's mapping
    (centre crop from row CROP_Y, then scale by frame_size / 832) is undone:
    x 832 / frame_size + 96 — 524 -> 715.27 at 704, 524 -> 684 -> 651.75 at
    1024."""
    canvas = max(int(frame_size), DRIVE_RES)
    return (metrics.floor_for_canvas(floor_y, canvas) * CANVAS_W / int(frame_size)
            + CROP_Y)


def body_target(frame_size, profile, d):
    """FR 6 / FR 8: the body height in px a figure facing `d` measures on the
    832x1024 edit canvas — the profile's `dirs[d].body_h` at the drive
    resolution scaled by 832 / frame_size, the inverse of the animate
    workflow's 832 -> frame_size downscale (s 331 -> 391.2 at frame_size 704,
    268.9 at 1024). No mannequin is a code default: the profile is required."""
    return profile["dirs"][d]["body_h"] * CANVAS_W / frame_size


def floor_line(frame_size, profile, d):
    """FR 6: the floor row in px of a figure facing `d` on the 832x1024 edit
    canvas — `anchor_floor` of the profile's 704-res `dirs[d].floor_y`
    (s 524 -> 715.27 at frame_size 704, 524 -> 651.75 at 1024)."""
    return anchor_floor(profile["dirs"][d]["floor_y"], frame_size)


def _figure_mask(im, diff):
    """The figure of an RGB image: every pixel differing from the top-left
    corner colour by more than `diff` in any channel."""
    arr = numpy.asarray(im, dtype=numpy.int16)
    return (abs(arr - arr[0, 0]) > diff).any(axis=2)


def _fill_holes(mask):
    """FR 7 (amendment 11): `mask` plus every region of other pixels it fully
    encloses — the non-mask pixels a 4-connected flood fill from the image
    border does not reach."""
    from PIL import Image, ImageDraw

    # a one-pixel non-mask frame joins the whole border into one seed region
    im = Image.fromarray(numpy.pad(mask, 1).astype(numpy.uint8) * 255).copy()
    ImageDraw.floodfill(im, (0, 0), 128)
    # the fill marks every border-reached pixel 128: the rest is silhouette
    return numpy.asarray(im)[1:-1, 1:-1] != 128


def _place(src, dst, frame_size, profile, d, diff, fill_holes=False):
    """The FR 6 scale-and-paste shared by `anchor`, `reanchor` and `guide`.

    The figure of `src` (every pixel differing from its corner colour by
    more than `diff`, plus the regions it encloses when `fill_holes`; its
    bounding box is the measured body) is scaled uniformly so its body
    height equals `body_target(frame_size, profile, d)` and pasted on an
    832x1024 canvas filled with CHROMA_GREEN: its feet row lands on
    `floor_line(frame_size, profile, d)` and its centre x on the canvas
    centre (x 416). Returns `dst`. Raises metrics.FigureError
    naming `src` when it cannot be read as an image or has no figure.
    """
    from PIL import Image

    try:
        im = Image.open(src).convert("RGB")
    except Exception as e:
        # an unreadable or malformed source: FigureError naming the file,
        # never a traceback
        raise metrics.FigureError(str(src), "not a readable image") from e
    mask = _figure_mask(im, diff)
    if fill_holes:
        mask = _fill_holes(mask)
    rows = numpy.nonzero(mask.any(axis=1))[0]
    if len(rows) == 0:
        raise metrics.FigureError(str(src), "nothing to measure (flat image)")
    scale = body_target(frame_size, profile, d) / (int(rows.max()) - int(rows.min()) + 1)
    new_w = max(1, round(im.width * scale))
    new_h = max(1, round(im.height * scale))
    # the whole source frame scales as one image and the figure mask with
    # it (the mask is taken on the source, so resampling ringing near a
    # low threshold never grows the figure); the masked pixels are pasted
    # on the chroma canvas: the background is flat (31, 203, 25) wherever
    # the figure is not
    scaled = im.resize((new_w, new_h), Image.LANCZOS)
    arr = numpy.asarray(scaled, dtype=numpy.uint8)
    fig = numpy.asarray(Image.fromarray(mask.astype(numpy.uint8) * 255)
                        .resize((new_w, new_h), Image.BILINEAR)) >= 128
    if not fig.any():
        raise metrics.FigureError(str(src), "nothing to measure (flat image)")
    # the figure bbox in the scaled frame: the paste pins its bottom row
    # (floor + 1) on the floor line and its centre x on the canvas centre
    ys, xs = numpy.nonzero(fig)
    bx0, bx1 = int(xs.min()), int(xs.max()) + 1
    by0, by1 = int(ys.min()), int(ys.max()) + 1
    left = CANVAS_W // 2 - round((bx0 + bx1 - 1) / 2)
    top = round(floor_line(frame_size, profile, d) + 1 - by1)
    canvas = Image.new("RGB", STILL_SIZE, CHROMA_GREEN)
    car = numpy.array(canvas)  # writable copy of the flat chroma canvas
    # clip the paste rect to the canvas: only figure pixels move, the
    # background stays CHROMA_GREEN everywhere the figure is not
    dx0, dx1 = max(0, bx0 + left), min(CANVAS_W, bx1 + left)
    dy0, dy1 = max(0, by0 + top), min(CANVAS_H, by1 + top)
    if dx1 > dx0 and dy1 > dy0:
        sub = fig[dy0 - top:dy1 - top, dx0 - left:dx1 - left]
        reg = car[dy0:dy1, dx0:dx1]
        reg[sub] = arr[dy0 - top:dy1 - top, dx0 - left:dx1 - left][sub]
    canvas = Image.fromarray(car)
    Path(dst).parent.mkdir(parents=True, exist_ok=True)
    canvas.save(dst)
    return dst


def anchor(src, dst, frame_size, profile, d):
    """FR 6: prepare the anchored still (`master/still.png`) of a
    standing-figure still against the profile's direction `d`.

    The still is scaled uniformly by `body_target(frame_size, profile, d) /
    measured body height` and its figure pasted on an 832x1024 canvas filled
    with CHROMA_GREEN: its feet row lands on `floor_line(frame_size, profile,
    d)` and its measured centre x lands on the canvas centre (x 416). Returns
    `dst`. Raises metrics.FigureError when the still is not a standing figure
    (or cannot be read as an image at all).
    """
    try:
        metrics.check_standing(src)
    except metrics.FigureError:
        raise
    except Exception as e:
        # an unreadable or malformed source is a standing-rule failure too:
        # FigureError naming the file, never a traceback
        raise metrics.FigureError(str(src), "not a readable image") from e
    return _place(src, dst, frame_size, profile, d, metrics.CORNER_DIFF)


def reanchor(src, dst, frame_size, profile, d):
    """FR 7: re-anchor a direction's raw edit (`master/<d>.edit.png`) to
    `master/<d>.png` — `anchor`'s scale-and-paste with direction `d`'s
    target and floor, centred at x 416, without the standing-figure rule (a
    T-pose wider than tall is re-anchored, not rejected). Returns `dst`.
    Raises metrics.FigureError naming `src` when it has no figure."""
    return _place(src, dst, frame_size, profile, d, metrics.CORNER_DIFF)


def guide(render, dst, frame_size, profile, d):
    """FR 7: place a mannequin render facing `d` on the edit canvas as the
    direction's guide (`master/<d>.guide.png`). The render's silhouette is
    every pixel differing from its corner colour by more than GUIDE_DIFF,
    plus every region of other pixels those fully enclose (not connected to
    the image border through non-silhouette pixels; amendment 11), so no
    chroma green is left inside the mannequin; it is scaled to direction
    `d`'s target, feet on its floor line, centred at x 416, on CHROMA_GREEN.
    Returns `dst`. Raises metrics.FigureError naming `render` when it has no
    silhouette."""
    return _place(render, dst, frame_size, profile, d, GUIDE_DIFF, fill_holes=True)


def size_master(src, dst_dir, size=STILL_SIZE):
    """Hand a direction still the master's canvas: the edit's output is
    expected on the 832x1024 edit canvas, so anything else is scaled to fit it
    and centred on flat chroma green. Returns the destination path."""
    from PIL import Image

    dst = Path(dst_dir) / Path(src).name
    im = Image.open(src).convert("RGB")
    if im.size != tuple(size):
        scale = min(size[0] / im.width, size[1] / im.height)
        new = (max(1, round(im.width * scale)), max(1, round(im.height * scale)))
        fitted = Image.new("RGB", tuple(size), CHROMA_GREEN)
        fitted.paste(im.resize(new, Image.LANCZOS),
                     ((size[0] - new[0]) // 2, (size[1] - new[1]) // 2))
        im = fitted
    Path(dst).parent.mkdir(parents=True, exist_ok=True)
    im.save(dst)
    return dst


def dir_prompt(turn, d, pronoun):
    """FR 7 (amendment 12, dec-745): the direction edit's prompt — the
    mannequin-camera framing, then the view's turn prompt `turn[d]` for
    direction `d` worded in the character's pronoun, then GUIDE_LINE in
    that pronoun. Raises ValueError for a pronoun or direction the preset
    does not know."""
    forms = pronoun_forms(pronoun)
    if d not in turn:
        raise ValueError(f"view has no turn prompt for direction {d!r}")
    obj = forms["obj"]
    turn_text = turn[d].format(**forms)
    return ("The exact same character with the exact same outfit, colors, proportions "
            f"and art style, seen from the same high isometric camera angle as the "
            f"mannequin in image 2, looking down at {obj} at about 45 degrees, "
            f"but now "
            + turn_text + "." + TAIL + " " + GUIDE_LINE.format(**forms))


def animate_prompt(view, style, char, anim, d):
    """The animate step's --prompt: the character's look (describe, project
    style, the view's camera line), the view's facing line for the direction,
    the view's face line for a face direction only, the view's background
    line, then the character's override `extra` for the `<animation>-<dir>`
    pair. The look is the character's describe text for direction `d`."""
    forms = pronoun_forms(char.pronoun)
    face = d in char.face_dirs
    look = char.look(d)
    parts = [look.rstrip(". "), style.rstrip(". "), view["camera"].format(**forms),
             view["facing"][d].format(**forms)]
    parts = [x if x.endswith(".") else x + "." for x in parts if x]
    if face:
        parts.append(view["face"].format(**forms))
    parts.append(view["background"])
    extra = (char.overrides.get(f"{anim.name}-{d}") or {}).get("extra")
    if extra:
        parts.append(extra)
    return " ".join(parts)


def review_sheet(images, dst, height=340, labels=None):
    """FR 8: the review sheet — the direction stills side by side at one
    height, each labelled underneath (from the new-kit branch's
    contact_sheet): `labels[i]` when given (the size report's `se 0.97`,
    amendment 3), otherwise the still's direction (its file stem). Returns
    the destination path."""
    from PIL import Image, ImageDraw

    ims = [Image.open(p).convert("RGB") for p in images]
    ims = [im.resize((round(im.width * height / im.height), height), Image.LANCZOS)
           for im in ims]
    if labels is None:
        labels = [Path(p).stem for p in images]
    sheet = Image.new("RGB", (sum(im.width for im in ims), height + 24), (40, 42, 52))
    draw, x = ImageDraw.Draw(sheet), 0
    for im, label in zip(ims, labels):
        sheet.paste(im, (x, 0))
        draw.text((x + 8, height + 5), label, fill=(230, 230, 230))
        x += im.width
    Path(dst).parent.mkdir(parents=True, exist_ok=True)
    sheet.save(dst)
    return dst
