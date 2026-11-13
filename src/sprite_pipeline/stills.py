"""Still preparation, direction-edit prompts and the review sheet (spec FR 6-8).

`prepare` scales a still to fit the reference canvas, centres it and pads with
the still's corner colour. The reference canvas is 1280x1280: the still is
scaled to fit within 832x1024 (keeping aspect ratio), then pasted centred
within that 832x1024 area at (224, 128) on the canvas. Every pixel outside the
scaled still is filled with the still's corner colour (top-left pixel).

`dir_prompt` (FR 7) is the edit-workflow prompt for one direction: the view's
turn prompt for that direction, wrapped in the character's pronoun. `review_sheet`
(FR 8) writes the review sheet `refs-sheet.png`: the direction stills side by
side, each labelled underneath.
"""
import shutil
from pathlib import Path

STILL_SIZE = (832, 1024)
REF_SIZE = 1280
REF_OFFSET = (224, 128)

# the edit prompt's framing, from the new-kit branch (dir_prompt): the turn
# prompt keeps the character, camera and style; only the facing changes
STYLE = "hand-painted cel shading, bold dark outlines, crisp edges"
TAIL = " Full body, plain flat pure green background, no shadow."

# pronoun -> (object, possessive)
PRONOUNS = {
    "he": ("him", "his"),
    "she": ("her", "her"),
    "they": ("them", "their"),
}


def prepare(src, dst):
    """Scale a still of any size to fit the reference canvas, centre it, and
    pad with its corner colour.

    The still is scaled to fit within STILL_SIZE (832x1024) keeping aspect
    ratio, then pasted centred within that area at REF_OFFSET (224, 128) on
    a 1280x1280 canvas filled with the still's top-left pixel colour.

    Returns the destination path.
    """
    from PIL import Image

    im = Image.open(src).convert("RGB")
    w, h = im.size

    # scale to fit within STILL_SIZE, keeping aspect ratio
    scale = min(STILL_SIZE[0] / w, STILL_SIZE[1] / h)
    new_w = round(w * scale)
    new_h = round(h * scale)
    im = im.resize((new_w, new_h), Image.LANCZOS)

    # corner colour (top-left pixel)
    corner = im.getpixel((0, 0))

    # canvas
    canvas = Image.new("RGB", (REF_SIZE, REF_SIZE), corner)

    # centre the scaled still within the STILL_SIZE area at REF_OFFSET
    x = REF_OFFSET[0] + (STILL_SIZE[0] - new_w) // 2
    y = REF_OFFSET[1] + (STILL_SIZE[1] - new_h) // 2
    canvas.paste(im, (x, y))

    Path(dst).parent.mkdir(parents=True, exist_ok=True)
    canvas.save(dst)
    return dst


def dir_prompt(turn, d, pronoun):
    """FR 7: the direction edit's prompt — the view's turn prompt `turn[d]`
    for direction `d` worded in the character's pronoun. Raises ValueError
    for a pronoun or direction the preset does not know."""
    if pronoun not in PRONOUNS:
        raise ValueError(f"pronoun {pronoun!r} (he, she or they)")
    if d not in turn:
        raise ValueError(f"view has no turn prompt for direction {d!r}")
    obj, pos = PRONOUNS[pronoun]
    turn_text = turn[d].format(obj=obj, pos=pos)
    return ("The exact same character with the exact same outfit, colors, proportions "
            f"and art style, seen from the exact same high isometric camera angle "
            f"looking down at {obj}, standing in the same relaxed pose, but now "
            + turn_text + "." + TAIL)


def review_sheet(images, dst, height=340):
    """FR 8: the review sheet — the direction stills side by side at one
    height, each labelled with its direction (from the new-kit branch's
    contact_sheet). Returns the destination path."""
    from PIL import Image, ImageDraw

    ims = [Image.open(p).convert("RGB") for p in images]
    ims = [im.resize((round(im.width * height / im.height), height), Image.LANCZOS)
           for im in ims]
    sheet = Image.new("RGB", (sum(im.width for im in ims), height + 24), (40, 42, 52))
    draw, x = ImageDraw.Draw(sheet), 0
    for im, label in zip(ims, [Path(p).stem for p in images]):
        sheet.paste(im, (x, 0))
        draw.text((x + 8, height + 5), label, fill=(230, 230, 230))
        x += im.width
    Path(dst).parent.mkdir(parents=True, exist_ok=True)
    sheet.save(dst)
    return dst
