#!/usr/bin/env python3
"""Build stages of `sprite-pipeline go`, run from the project model (spec FR 10).

Per approved character, animation and generated direction: drive (Blender
render, cached by the project's drive key) -> animate (Wan Animate 2 via
comfy) -> key (sprite-gen video-frames) -> loop (sprite-gen video-loop);
then per character pixel (pixelize over all sets) -> export (mirrored frames,
atlas.png, Godot 4 SpriteFrames .tres, contact-sheet.png). Stages whose output
exists are skipped unless --force.

sprite-gen is a pinned package dependency (FR 12, dec-616): every sprite-gen
command runs through the console script installed next to sys.executable —
the running interpreter environment's own entry point — never through a
configured path. No stage shells out to POSIX-only commands such as cp
(dec-617): the drive copy is a Python step.
"""
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

from sprite_pipeline import config, project, stills

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

    # -- review stop (FR 8): refs.approved holds the hash of the direction
    # stills approve wrote; the character is approved only while it matches
    def refs_sheet(self, char):
        return self.project.output_root() / char / "refs-sheet.png"

    def refs_approved(self, char):
        return self.project.output_root() / char / "refs.approved"

    def missing_stills(self, char):
        """The generated directions whose still file does not exist yet."""
        return [d for d in self.project.generate
                if not self.dir_still(char, d).is_file()]

    def refs_hash(self, char):
        """FR 8: sha256 over the direction stills' bytes, in generate order;
        None while any of them is missing."""
        if self.missing_stills(char):
            return None
        h = hashlib.sha256()
        for d in self.project.generate:
            h.update(self.dir_still(char, d).read_bytes())
            h.update(b"\0")
        return h.hexdigest()

    def write_refs_approved(self, char, interim=False):
        """`approve` writes refs.approved: the hash of the direction stills,
        or (interim, slice-2 compatibility) the still filenames the approval
        covers — the gate accepts it only while every named still exists."""
        dst = self.refs_approved(char)
        dst.parent.mkdir(parents=True, exist_ok=True)
        if interim:
            dst.write_text("interim\n" + "".join(f"{d}\n" for d in self.project.generate))
        else:
            dst.write_text(self.refs_hash(char) + "\n")

    def is_approved(self, char):
        h = self.refs_hash(char)
        f = self.refs_approved(char)
        if h is None or not f.is_file():
            return False
        text = f.read_text().strip()
        if text.startswith("interim"):
            # slice-2 rule: approved while the named five stills exist
            named = text.splitlines()[1:]
            return (sorted(named) == sorted(self.project.generate)
                    and all(self.dir_still(char, d).is_file()
                            for d in self.project.generate))
        return text == h

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
            cs = next(c for c in p.characters if c.name == char)
            pose = anim.prompt.replace("{who}", cs.who)
            # FR 2: a character override `<animation>-<dir>` may set the seed
            seed = (cs.overrides.get(f"{anim.name}-{d}") or {}).get("seed", p.seed)
            # per-animation workflow override (FR 3): anim.workflow replaces
            # p.workflows["animate"] for that animation; both are absolute,
            # resolved at load (FR 1)
            wf = anim.workflow or p.workflows["animate"]
            argv = [sys.executable, "-m", "sprite_pipeline.comfy", "animate",
                    "--workflow", str(wf),
                    "--ref", str(self.dir_still(char, d)), "--video",
                    self.input_mp4(char, anim, d).name,
                    "--w", str(p.sprite.frame_size), "--h", str(p.sprite.frame_size),
                    "--length", str(anim.clip_frames), "--seed", str(seed),
                    "--prefix", f"{char}/dir8-{anim.name}-{d}",
                    "--copy-to", str(self.clip(char, anim, d)),
                    "--prompt", stills.animate_prompt(
                        project.load_view(p.view), p.style, cs, anim, d),
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
            # driven by the project: the generate+mirror direction order, the
            # mirror flips and the naming pattern all come from it (FR 11)
            from sprite_pipeline import export_godot
            return export_godot.export_char(self, char, anim_names, log, dry)
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
    out_path = Path(out_dir) / (Path(still_path).stem + ".keyed.png")
    return sprite_gen_cmds("cutout", str(still_path), "--out", str(out_path),
                           "--key", "measured")


def quote(a):
    a = str(a)
    return a if a and all(c.isalnum() or c in "/-_.=:+,@%" for c in a) else "'" + a.replace("'", "'\\''") + "'"


# --------------------------------------------------------------------------- export
#
# The Godot 4 export (mirror flips, atlas.png, SpriteFrames .tres named by the
# project's naming pattern, contact sheet) lives in export_godot.py, driven by
# the project's directions.generate, directions.mirror and naming (FR 11, AC7);
# StageCtx.run_program dispatches "export" to it. These re-exports keep
# stages.do_export / stages.write_tres resolving for existing callers.
from sprite_pipeline.export_godot import do_export, write_tres  # noqa: E402,F401
