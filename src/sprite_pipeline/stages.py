#!/usr/bin/env python3
"""Build stages of `sprite-pipeline go`, run from the project model (spec FR 10).

Per approved character, animation and generated direction: drive (Blender
render, cached by the project's drive key) -> animate (Wan Animate 2 via
comfy_run) -> key (sprite-gen video-frames) -> loop (sprite-gen video-loop);
then per character pixel (pixelize over all sets) -> export (mirrored frames,
atlas.png, Godot 4 SpriteFrames .tres, contact-sheet.png). Stages whose output
exists are skipped unless --force.

sprite-gen is a pinned package dependency (FR 12, dec-616): every sprite-gen
command runs through the console script installed next to sys.executable —
the running interpreter environment's own entry point — never through a
configured path. No stage shells out to POSIX-only commands such as cp
(dec-617): the drive copy is a Python step.
"""
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

from sprite_pipeline import config, project

PACKAGE = Path(__file__).resolve().parent

FACING = {"s": 0, "se": 45, "e": 90, "ne": 135, "n": 180}
MIRROR_OF = {"sw": "se", "w": "e", "nw": "ne"}
EIGHT_DIRS = ["n", "ne", "e", "se", "s", "sw", "w", "nw"]
PAIR_STAGES = ["drive", "animate", "key", "loop"]
KIT_STAGES = ["pixel", "export"]
STAGES = PAIR_STAGES + KIT_STAGES

# the drive scene's camera-height node; LSR-modeled from the view preset's 45°
DRIVE_ELEV = 45


class StageCtx:
    """Every location a stage touches for one project, resolved once per run."""

    def __init__(self, proj, cfg):
        self.project = proj
        self.cfg = cfg
        self.paths = type("Paths", (), {})()
        self.paths.motion_cache = Path(cfg["motion_cache"])
        self.paths.comfy_input = Path(cfg["comfy_input"])
        self.blender = cfg["blender"]
        self.drive_lsr = cfg.get("drive_lsr") or ""
        self.env = {**os.environ}
        if cfg["extra_path"]:
            self.env["PATH"] = f"{cfg['extra_path']}{os.pathsep}{self.env['PATH']}"
        self.env["RW"] = str(proj.sprite.frame_size)
        self.env["RH"] = str(proj.sprite.frame_size)
        self._drive_keys = {}

    def drive_key(self, anim):
        """The slice-1 cache key: GLB content, action and drive params, never the path."""
        if anim.name not in self._drive_keys:
            self._drive_keys[anim.name] = self.project.drive_key(anim, anim.source.read_bytes())
        return self._drive_keys[anim.name]

    # -- per (character, animation, dir) locations
    def cache_mp4(self, char, anim, d):
        return self.paths.motion_cache / f"{char}-{anim.name}-{d}-{self.drive_key(anim)}.mp4"

    def input_mp4(self, char, anim, d):
        return self.paths.comfy_input / f"drive-{char}-{anim.name}-{d}-{self.drive_key(anim)}.mp4"

    def frames_dir(self, char, anim, d):
        return self.paths.motion_cache / f"{char}-{anim.name}-{d}-{self.drive_key(anim)}" / "frames"

    def master(self, char):
        return self.project.output_root() / char / "master"

    def dir_still(self, char, d):
        return self.master(char) / f"{d}.png"

    def clip(self, char, anim, d):
        return self.project.output_root() / char / "clips" / f"{anim.name}-{d}.mp4"

    def work_dir(self, char, anim, d):
        return self.project.output_root() / char / "work" / f"frames-{anim.name}-{d}"

    def loops_dir(self, char, anim, d):
        return self.project.output_root() / char / "loops" / f"{anim.name}-{d}"

    def sprite_out(self, char):
        return self.project.output_root() / char / "sprites"

    # -- approval: this slice a character counts as approved when its five
    # direction stills exist (slices 4-5 add still preparation and refs.approved)
    def is_approved(self, char):
        return all(self.dir_still(char, d).is_file() for d in self.project.generate)

    # -- stage output predicates (skip-if-exists)
    def stage_done(self, stage, char=None, anim=None, d=None):
        if stage == "drive":
            return self.cache_mp4(char, anim, d).is_file() and self.input_mp4(char, anim, d).is_file()
        if stage == "animate":
            return self.clip(char, anim, d).is_file()
        if stage == "key":
            keyed = self.work_dir(char, anim, d) / "keyed"
            if keyed.is_dir() and any(keyed.glob("*.png")):
                return True
            return self.stage_done("loop", char, anim, d)  # keyed frames are disposable once the loop is cut
        if stage == "loop":
            cyc = self.loops_dir(char, anim, d) / "cycle"
            return cyc.is_dir() and any(cyc.glob("*.png"))
        raise ValueError(stage)

    # -- command planning: returns list of (argv, log-label); argv[0] ==
    # "copy_file" marks the in-process copy step (no subprocess, no cp)
    def plan_pair(self, char, anim, d, stage):
        p = self.project
        if stage == "drive":
            frames = self.frames_dir(char, anim, d)
            argv = ["blender", "-b"]
            if self.drive_lsr:
                argv += [self.drive_lsr]
            argv += ["--python", str(PACKAGE / "render_drive.py"), "--",
                     str(anim.source), anim.action, str(frames), str(FACING[d]),
                     str(DRIVE_ELEV), str(anim.repeats), str(anim.pre), str(anim.post)]
            if anim.head_lift:
                argv.append(str(anim.head_lift))
            cache = self.cache_mp4(char, anim, d)
            cmds = [(argv, "blender"),
                    (["ffmpeg", "-v", "error", "-y", "-framerate", str(p.sprite.fps),
                      "-i", f"{frames}/f%04d.png", "-c:v", "libx264", "-crf", "12",
                      "-pix_fmt", "yuv420p", str(cache)], "ffmpeg")]
            # the copy into ComfyUI's input/ is an in-process step: the marker
            # argv[0] "copy_file" is executed by Python, never spawned (AC8-r3)
            cmds.append((("copy_file", str(cache), str(self.input_mp4(char, anim, d))),
                         "copy-drive"))
            return cmds
        if stage == "animate":
            pose = anim.prompt.replace("{who}", "a person")
            argv = [sys.executable, "-m", "sprite_pipeline.comfy_run", "animate",
                    "--ref", str(self.dir_still(char, d)), "--video",
                    self.input_mp4(char, anim, d).name,
                    "--w", str(p.sprite.frame_size), "--h", str(p.sprite.frame_size),
                    "--length", str(anim.clip_frames), "--seed", str(p.seed),
                    "--prefix", f"{char}/dir8-{anim.name}-{d}",
                    "--copy-to", str(self.clip(char, anim, d)),
                    "--pose-prompt", pose]
            if p.pose_strength != 1.0:
                argv += ["--pose-strength", str(p.pose_strength)]
            return [(argv, "animate")]
        if stage == "key":
            return sprite_gen_cmds(
                "video-frames", "--clip", str(self.clip(char, anim, d)),
                "--out-dir", str(self.work_dir(char, anim, d)), "--spill", "small",
                "--allow-subject-edge-contact")
        if stage == "loop":
            argv = ["video-loop",
                    "--frames-dir", str(self.work_dir(char, anim, d) / "keyed"),
                    "--out-dir", str(self.loops_dir(char, anim, d)),
                    "--fps", str(p.sprite.fps), "--state", anim.state,
                    "--cycle", "fixed" if anim.loop else "once",
                    "--start", str(anim.start), "--length", str(anim.frames)]
            if anim.loop:
                argv += ["--seam-pop-max", "off"]
            if anim.seam_max:
                argv += ["--seam-max", f"{anim.seam_max:g}"]
            argv += ["--name", f"{d}-{anim.name}"]
            return sprite_gen_cmds(*argv)
        raise ValueError(stage)

    def pixel_done(self, char, anim_names, dirs):
        out = self.sprite_out(char)
        return all((out / f"{d}-{n}_000.png").is_file() for n in anim_names for d in dirs)

    def export_done(self, char):
        out = self.sprite_out(char)
        return all((out / f).is_file() for f in ("atlas.png", "sprite_frames.tres", "contact-sheet.png"))

    def plan_kit_stage(self, char, stage, anim_names, dirs):
        p = self.project
        if stage == "pixel":
            argv = [sys.executable, "-m", "sprite_pipeline.pixelize",
                    "--ref", str(self.master(char) / "se.keyed.png"),
                    "--scale", str(p.sprite.scale), "--colors", str(p.sprite.colors),
                    "--accent-boost", str(p.sprite.accent_boost), "--fps", str(p.sprite.fps),
                    "--out", str(self.sprite_out(char))]
            for d in dirs:  # dir-major, matching batch_dir8.sh's set order
                for n in anim_names:
                    argv += ["--set", f"{d}-{n}={self.loops_dir(char, p.animation(n), d) / 'cycle'}"]
            return [(argv, "pixel")]
        if stage == "export":
            return []  # pure python, no subprocess
        raise ValueError(stage)

    # -- programmatic in-process steps (never subprocesses)
    def run_program(self, stage, char, anim_names, dirs, log, dry):
        if stage == "export":
            return do_export(self, char, [self.project.animation(n) for n in anim_names],
                             dirs, log, dry)
        raise ValueError(stage)


def sprite_gen_exe():
    """The sprite-gen entry point of the running interpreter's environment
    (FR 12, dec-616): the console script installed next to sys.executable by
    the pinned dependency — never a config value. No resolve(): the venv bin
    dir is what holds the script, and python3 there is a symlink to the base
    interpreter."""
    return Path(sys.executable).parent / "sprite-gen"


def sprite_gen_cmds(*args):
    """sprite-gen through the tool's own environment: argv[0] is the console
    script of the running interpreter's venv, where the pin installed it."""
    return [([str(sprite_gen_exe()), *args], "sprite-gen")]


def copy_file(src, dst, log):
    """The drive copy, done in Python — no `cp` subprocess exists anywhere (AC8)."""
    Path(dst).parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src, dst)
    log.append(f"$ copy {src} -> {dst} (python)")
    return 0


def plan_cutout(still_path, out_dir):
    """Key the prepared SE still with `sprite-gen cutout --key measured` (FR 6).
    Planned here; wired into the still-preparation stage in slice 4."""
    return sprite_gen_cmds("cutout", "--source", str(still_path), "--out-dir", str(out_dir),
                           "--key", "measured")


def quote(a):
    a = str(a)
    return a if a and all(c.isalnum() or c in "/-_.=:+,@%" for c in a) else "'" + a.replace("'", "'\\''") + "'"


# --------------------------------------------------------------------------- export

def do_export(ctx, char, anims, dirs, log, dry):
    """Mirror sw/w/nw frames, pack the atlas, write the Godot 4 SpriteFrames .tres
    and a contact sheet. Export keeps today's behaviour this slice (slice 3
    adds naming and mirror config). Returns True when nothing is missing."""
    from PIL import Image
    out = ctx.sprite_out(char)
    frames, missing = {}, []
    for anim in anims:
        for d in EIGHT_DIRS:
            src = MIRROR_OF.get(d, d)
            if src not in dirs:
                missing.append(f"{anim.name}/{d}")
                continue
            files = sorted(out.glob(f"{src}-{anim.name}_[0-9][0-9][0-9].png"))  # not the _sheet.png strips
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

    rows = [(a, d) for a in anims for d in EIGHT_DIRS if (a.name, d) in frames]
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
            anims_out.append((f"{a.name}_{d}", a.loop, regions))
        write_tres(out / "sprite_frames.tres", "atlas.png", anims_out, fps=ctx.project.sprite.fps)
        sheet = Image.new("RGBA", (8 * w * 4, len(anims) * h * 4), (40, 42, 52, 255))
        for ri, a in enumerate(anims):
            for ci, d in enumerate(EIGHT_DIRS):
                first = frames.get((a.name, d))
                if not first:
                    continue
                im = Image.open(first[0]).convert("RGBA").resize((w * 4, h * 4), Image.NEAREST)
                sheet.paste(im, (ci * w * 4, ri * h * 4), im)
        sheet.save(out / "contact-sheet.png")
    log.append(f"export: {len(rows)} of {len(anims) * 8} animations packed"
               + (f", missing: {', '.join(missing)}" if missing else ""))
    return not missing


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
