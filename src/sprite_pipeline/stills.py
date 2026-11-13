"""Still preparation: scale to fit the reference canvas, centre, and pad with
the still's corner colour (spec FR 6, AC3).

The reference canvas is 1280x1280. The still is scaled to fit within 832x1024
(keeping aspect ratio), then pasted centred within that 832x1024 area at
(224, 128) on the canvas. Every pixel outside the scaled still is filled with
the still's corner colour (top-left pixel).
"""
from pathlib import Path

STILL_SIZE = (832, 1024)
REF_SIZE = 1280
REF_OFFSET = (224, 128)


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
