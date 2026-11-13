"""Pixelize keyed RGBA frames. With --ref, one master frame fixes the scale and the palette for every set."""
import argparse, glob
from pathlib import Path
from PIL import Image

ap = argparse.ArgumentParser()
ap.add_argument("--set", action="append", required=True, help="name=dir[:i,j,k]  (optional frame indexes)")
ap.add_argument("--ref", help="master frame (same resolution as the clips' keyed frames)")
ap.add_argument("--height", type=int, default=128, help="figure height in game pixels")
ap.add_argument("--colors", type=int, default=32)
ap.add_argument("--out", required=True)
ap.add_argument("--view", type=int, default=3)
ap.add_argument("--fps", type=int, default=16)
ap.add_argument("--scale", type=float, help="fixed source-to-game pixel ratio for every set (overrides --height)")
ap.add_argument("--accent-boost", type=int, default=0, help="repeat saturated, bright pixels N times when building the palette")
a = ap.parse_args()


def figure_h(im):
    b = im.getchannel("A").point(lambda v: 255 if v > 127 else 0).getbbox()
    return b[3] - b[1]


def shrink(im, scale):
    s = im.resize((round(im.width * scale), round(im.height * scale)), Image.BOX)
    s.putalpha(s.getchannel("A").point(lambda v: 255 if v > 110 else 0))
    return s


sets = {}
for spec in a.set:
    name, rest = spec.split("=", 1)
    d, _, idx = rest.partition(":")
    files = sorted(glob.glob(f"{d}/*.png"))
    if idx:
        files = [files[int(i)] for i in idx.split(",")]
    sets[name] = [Image.open(f).convert("RGBA") for f in files]

ref = Image.open(a.ref).convert("RGBA") if a.ref else None
small = {}
for name, frames in sets.items():
    scale = a.scale or a.height / figure_h(frames[0])  # --scale keeps every move at one size; --ref drives the palette
    small[name] = [shrink(f, scale) for f in frames]

palette_src = [shrink(ref, a.height / figure_h(ref))] if ref else [s for fr in small.values() for s in fr]
import colorsys
pixels = []
for s in palette_src:
    for p in s.get_flattened_data():
        if not p[3]:
            continue
        pixels.append(p[:3])
        if a.accent_boost:
            h, sat, v = colorsys.rgb_to_hsv(*(c / 255 for c in p[:3]))
            if sat > 0.45 and v > 0.35:
                pixels += [p[:3]] * a.accent_boost
strip = Image.new("RGB", (len(pixels), 1)); strip.putdata(pixels)
pal = strip.quantize(colors=a.colors, method=Image.Quantize.MEDIANCUT)

outdir = Path(a.out); outdir.mkdir(parents=True, exist_ok=True)
for name, frames in small.items():
    done = []
    for i, s in enumerate(frames):
        q = s.convert("RGB").quantize(palette=pal, dither=Image.Dither.NONE).convert("RGBA")
        out = Image.new("RGBA", q.size, (0, 0, 0, 0)); out.paste(q, mask=s.getchannel("A"))
        out.save(outdir / f"{name}_{i:03d}.png")
        done.append(out)
    view = [Image.alpha_composite(Image.new("RGBA", f.size, (52, 56, 72, 255)), f).resize((f.width * a.view, f.height * a.view), Image.NEAREST) for f in done]
    if len(view) > 1:
        view[0].save(outdir / f"{name}.gif", save_all=True, append_images=view[1:], duration=1000 // a.fps, loop=0, disposal=2)
    pick = view if len(view) <= 8 else view[::max(1, len(view) // 6)][:6]
    sheet = Image.new("RGBA", (sum(v.width for v in pick), pick[0].height))
    x = 0
    for v in pick:
        sheet.paste(v, (x, 0)); x += v.width
    sheet.save(outdir / f"{name}_sheet.png")
    print(name, len(done), "frames", done[0].size)
