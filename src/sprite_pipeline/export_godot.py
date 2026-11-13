#!/usr/bin/env python3
"""Godot 4 export of a character's pixelised frames (spec FR 11, AC7).

Driven by the project: the direction order is `directions.generate` then the
`directions.mirror` keys; each mirrored direction is a left-right flip of its
`mirror` source's frames, and every animation in `sprite_frames.tres` is named
by the project's `naming` pattern (`{character}`, `{animation}`, `{dir}`). The
atlas is referenced as `atlas.png`, relative to the `.tres` it sits next to.
"""
from pathlib import Path

from PIL import Image


def direction_order(proj):
    """The directions a character's frames can cover: generated first, then
    the mirrored ones in the project's own order."""
    return list(proj.generate) + list(proj.mirror)


def anim_name(proj, char, anim_name, d):
    """The Godot animation name for (character, animation, direction): the
    project's naming pattern with the three fields substituted."""
    return (proj.naming.replace("{character}", char)
                      .replace("{animation}", anim_name)
                      .replace("{dir}", d))


def export_char(ctx, char, anim_names, log, dry):
    """Flip each `mirror` direction from its source, pack the atlas, write the
    Godot 4 SpriteFrames .tres and a contact sheet. Returns True when nothing
    is missing."""
    p = ctx.project
    out = ctx.sprite_out(char)
    out.mkdir(parents=True, exist_ok=True)  # writing mirror flips needs the dir
    anims = [p.animation(n) for n in anim_names]
    mirror_src = dict(p.mirror)  # mirrored direction -> its flip source

    frames, missing = {}, []
    for anim in anims:
        for d in direction_order(p):
            src = mirror_src.get(d, d)
            # not the _sheet.png strips pixelize writes next to the frames
            files = sorted(out.glob(f"{src}-{anim.name}_[0-9][0-9][0-9].png"))
            if not files:
                missing.append(f"{anim.name}/{d}")
                continue
            if d != src:
                flips = []
                for f in files:
                    m = out / f"{d}-{anim.name}_{f.name.rsplit('_', 1)[1]}"
                    if dry or not m.is_file():
                        Image.open(f).convert("RGBA").transpose(Image.FLIP_LEFT_RIGHT).save(m)
                    flips.append(m)
                files = flips
            frames[(anim.name, d)] = files

    sizes = {im.size for fs in frames.values() for im in (Image.open(f) for f in fs)}
    if not sizes:
        return False
    if len(sizes) != 1:
        log.append(f"export: mixed frame sizes {sorted(sizes)} — cannot pack one atlas")
        return False
    w, h = sizes.pop()

    rows = [(a, d) for a in anims for d in direction_order(p) if (a.name, d) in frames]
    ncol = max(len(frames[(a.name, d)]) for a, d in rows)
    if not dry:
        atlas = Image.new("RGBA", (ncol * w, len(rows) * h), (0, 0, 0, 0))
        region = {}
        for ri, (a, d) in enumerate(rows):
            for i, f in enumerate(frames[(a.name, d)]):
                im = Image.open(f).convert("RGBA")
                atlas.paste(im, (i * w, ri * h), im)
                region[(a.name, d, i)] = (i * w, ri * h, w, h)
        atlas.save(out / "atlas.png")
        anims_out = []
        for (a, d) in rows:
            regions = [region[(a.name, d, i)] for i in range(len(frames[(a.name, d)]))]
            anims_out.append((anim_name(p, char, a.name, d), a.loop, regions))
        write_tres(out / "sprite_frames.tres", "atlas.png", anims_out, fps=p.sprite.fps)
        dirs = direction_order(p)
        sheet = Image.new("RGBA", (len(dirs) * w * 4, len(anims) * h * 4), (40, 42, 52, 255))
        for ri, a in enumerate(anims):
            for ci, d in enumerate(dirs):
                first = frames.get((a.name, d))
                if not first:
                    continue
                im = Image.open(first[0]).convert("RGBA").resize((w * 4, h * 4), Image.NEAREST)
                sheet.paste(im, (ci * w * 4, ri * h * 4), im)
        sheet.save(out / "contact-sheet.png")
    log.append(f"export: {len(rows)} of {len(anims) * len(direction_order(p))} animations packed"
               + (f", missing: {', '.join(missing)}" if missing else ""))
    return not missing


def do_export(ctx, char, anims, dirs, log, dry):
    """Backward-compatible entry (old do_export contract): ``anims`` are
    Animation objects and ``dirs`` is the explicit list of directions
    available to pixelize. Runs the same export as ``export_char`` with the
    project's naming and mirror table. Returns True when nothing is missing.
    A direction in ``dirs`` that is neither generated nor mirrored is simply
    not projected, matching the old behaviour."""
    p = ctx.project
    names = [a.name for a in anims]
    available = set(dirs)
    gen = list(p.generate)
    mir = dict(p.mirror)
    p.generate = [d for d in gen if d in available]
    p.mirror = {m: s for m, s in mir.items() if s in available}
    try:
        return export_char(ctx, char, names, log, dry)
    finally:
        p.generate, p.mirror = gen, mir


def write_tres(path, atlas_rel, anims, fps=30):
    """Godot 4 SpriteFrames (format=3) over one atlas: one AtlasTexture per frame."""
    sub_id = {}
    subs = []
    for name, loop, regions in anims:
        for i, (x, y, fw, fh) in enumerate(regions):
            sid = f"AtlasTexture_{name}_f{i}"
            sub_id[(name, i)] = sid
            subs.append(f'[sub_resource type="AtlasTexture" id="{sid}"]\n'
                        f'atlas = ExtResource("1_atlas")\n'
                        f'region = Rect2({x}, {y}, {fw}, {fh})')
    anim_blocks = []
    for name, loop, regions in anims:
        frames = ", ".join('{\n"duration": 1.0,\n"texture": SubResource("%s")\n}' % sub_id[(name, i)]
                           for i in range(len(regions)))
        anim_blocks.append('{\n'
                           f'"frames": [{frames}],\n'
                           f'"loop": {"true" if loop else "false"},\n'
                           f'"name": &"{name}",\n'
                           f'"speed": {float(fps)}\n'
                           '}')
    body = ('[gd_resource type="SpriteFrames" load_steps=%d format=3]\n\n'
            '[ext_resource type="Texture2D" path="%s" id="1_atlas"]\n\n'  # relative to the .tres
            % (2 + len(subs), atlas_rel))  # 1 (this resource) + 1 ext + subs
    body += "\n\n".join(subs)
    body += "\n\n[resource]\nanimations = [%s]\n" % ", ".join(anim_blocks)
    Path(path).write_text(body)
