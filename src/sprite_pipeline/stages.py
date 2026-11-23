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
import os
import shutil
import subprocess
import sys
from pathlib import Path

from sprite_pipeline import config, metrics, project, stills

PACKAGE = Path(__file__).resolve().parent

# FR 8 (amendment 3): the review stop's advisory size band — a direction
# still whose size ratio lies outside it prints CHECK; nothing ever blocks
SIZE_CHECK_LOW = 0.85
SIZE_CHECK_HIGH = 1.15

FACING = {"s": 0, "se": 45, "e": 90, "ne": 135, "n": 180}
# the mannequin A-pose (amendment 13, dec: dec-799): degrees the mannequin
# render (the FR 7 guide and `mannequin-profile`) lowers both upper arms.
# A package constant used only by that render — not a project key, and
# animation drives never carry it.
MANNEQUIN_ARM_DROP = 45
MIRROR_OF = {"sw": "se", "w": "e", "nw": "ne"}
EIGHT_DIRS = ["n", "ne", "e", "se", "s", "sw", "w", "nw"]
PAIR_STAGES = ["drive", "animate", "key", "loop"]
KIT_STAGES = ["pixel", "export"]
STAGES = PAIR_STAGES + KIT_STAGES

# the drive scene's camera-height node; LSR-modeled from the view preset's 45°.
# One definition lives in project.py (drive_key hashes it); re-exported here
# so stages.DRIVE_ELEV keeps resolving and plan_pair shares that definition.
DRIVE_ELEV = project.DRIVE_ELEV
DRIVE_ORTHO = 3.0  # render_drive.py camera ortho_scale at sprite.frame_size


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

    def guide_file(self, char, d):
        return self.master(char) / f"{d}.guide.png"

    def edit_file(self, char, d):
        return self.master(char) / f"{d}.edit.png"

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

    # -- source-still fingerprint (FR 10, amendment 2): source.sha256 holds
    # the sha256 of the character's source still bytes; a changed source makes
    # the character's whole output tree stale
    def source_hash_file(self, char):
        return self.master(char) / "source.sha256"

    def source_hash(self, cs):
        """sha256 of the character's source still bytes (lowercase hex)."""
        return hashlib.sha256(cs.still.read_bytes()).hexdigest()

    def write_source_hash(self, char, cs):
        """Write the source-still fingerprint at out/<char>/master/source.sha256."""
        dst = self.source_hash_file(char)
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_text(self.source_hash(cs) + "\n")

    def recorded_source_hash(self, char):
        """The fingerprint on record, or None when there is none: no file, an
        unreadable file or a blank one is unknown, never stale."""
        try:
            recorded = self.source_hash_file(char).read_text(errors="replace").strip()
        except OSError:
            return None
        return recorded or None

    def source_stale(self, char, cs):
        """The character is stale when a fingerprint is on record and it
        differs from the current source still bytes (a changed file or a
        repointed path). Unknown (no usable hash on record) is never stale."""
        recorded = self.recorded_source_hash(char)
        if recorded is None:
            return False
        return recorded != self.source_hash(cs)

    def drop_char_outputs(self, char):
        """Stale-source rebuild: delete the character's whole output directory.
        The resolved path must be inside the project output directory."""
        out_root = self.project.output_root().resolve()
        char_dir = (self.project.output_root() / char).resolve()
        if char_dir.parent != out_root:
            raise ValueError(
                f"refusing to delete {char_dir}: not inside the project output "
                f"directory {out_root}")
        shutil.rmtree(char_dir)

    def write_refs_approved(self, char):
        """`approve` writes refs.approved: the hash of the direction stills."""
        dst = self.refs_approved(char)
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_text(self.refs_hash(char) + "\n")

    def is_approved(self, char):
        h = self.refs_hash(char)
        f = self.refs_approved(char)
        if h is None or not f.is_file():
            return False
        text = f.read_text().strip()
        if text.startswith("interim"):
            # pre-slice-3 interim files (written by the removed
            # `approve --interim`): approved while the named five stills exist
            named = text.splitlines()[1:]
            return (sorted(named) == sorted(self.project.generate)
                    and all(self.dir_still(char, d).is_file()
                            for d in self.project.generate))
        return text == h

    # -- s stop (FR 8, amendment 12): s.approved holds the sha256 of the
    # guided master `master/s.png`; the other direction edits wait while it
    # does not match
    def s_sheet(self, char):
        return self.project.output_root() / char / "s-sheet.png"

    def s_approved_file(self, char):
        return self.project.output_root() / char / "s.approved"

    def s_hash(self, char):
        """sha256 of the guided master `master/s.png`; None while it is missing."""
        p = self.dir_still(char, "s")
        return hashlib.sha256(p.read_bytes()).hexdigest() if p.is_file() else None

    def s_approved(self, char):
        """True while `s.approved` holds the sha256 of `master/s.png`."""
        h = self.s_hash(char)
        f = self.s_approved_file(char)
        return h is not None and f.is_file() and f.read_text().strip() == h

    def write_s_approved(self, char):
        """`approve` writes s.approved: the sha256 of `master/s.png`."""
        dst = self.s_approved_file(char)
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_text(self.s_hash(char) + "\n")

    def ref_size(self, char, d):
        """FR 8 (amendments 3 and 10): direction `d`'s size ratio — the body
        height of its raw edit (`master/<d>.edit.png`, or `master/<d>.png`
        when no raw edit exists) measured the way the master is (FR 6,
        `metrics.measure`) over that direction's target on the 832x1024 edit
        canvas (`stills.body_target`; the `s` master by the FR 6 target),
        rounded to two decimals. None when there is no figure to measure: the
        report is advisory and never raises."""
        raw = self.edit_file(char, d)
        try:
            body_h = metrics.measure(raw if raw.is_file()
                                     else self.dir_still(char, d))["body_h"]
        except Exception:
            return None
        target = stills.body_target(self.project.sprite.frame_size,
                                    project.anchor_profile(self.project), d)
        return round(body_h / target, 2)

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
                # the bone names ride only with the pitch (FR 3, rig.neck/rig.head)
                argv += [str(anim.head_lift), "--neck", p.rig["neck"], "--head", p.rig["head"]]
            if anim.arm_drop:
                # the A-pose drop rides only with the mannequin render (amendment 13)
                argv += ["--arm-drop", str(anim.arm_drop)]
            # FR 2: render_size widens the camera by the same ratio, so the
            # mannequin keeps the frame-size pixel scale and no limb clips
            size = p.drive.render_size
            argv += [f"res={size}", f"ortho={DRIVE_ORTHO * size / p.sprite.frame_size:.6f}"]
            # FR 1, 3: the transparent frames, scaled by the drive scale and
            # composited on drive.background (a Python step, run as a module)
            keyed = frames.parent / "composited"
            composite = [sys.executable, "-m", "sprite_pipeline.stages", "composite-drive",
                         str(frames), str(keyed), str(p.sprite.frame_size),
                         f"{p.drive_scale(anim):g}", *(str(c) for c in p.drive.background)]
            cache = self.cache_mp4(char, anim, d)
            cmds = [(argv, "blender"), (composite, "composite-drive"),
                    (["ffmpeg", "-v", "error", "-y", "-framerate", str(p.sprite.fps),
                      "-i", f"{keyed}/f%04d.png", "-c:v", "libx264", "-crf", "12",
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
            canvas = anim.canvas or p.sprite.frame_size
            ref, cmds = str(self.dir_still(char, d)), []
            if canvas != p.sprite.frame_size:
                # the reference still at the usual frame size, centred on the
                # larger canvas: the character keeps its pixel size
                padded = str(self.work_dir(char, anim, d) / f"ref-{canvas}.png")
                cmds.append((("pad_ref", ref, padded, str(p.sprite.frame_size), str(canvas)), "pad-ref"))
                ref = padded
            argv = [sys.executable, "-m", "sprite_pipeline.comfy", "animate",
                    "--workflow", str(wf),
                    "--ref", ref, "--video",
                    self.input_mp4(char, anim, d).name,
                    "--w", str(canvas), "--h", str(canvas),
                    "--length", str(anim.clip_frames), "--seed", str(seed),
                    "--prefix", f"{char}/dir8-{anim.name}-{d}",
                    "--copy-to", str(self.clip(char, anim, d)),
                    "--prompt", stills.animate_prompt(
                        project.load_view(p.view), p.style, cs, anim, d),
                    "--pose-prompt", pose]
            if p.pose_strength != 1.0:
                argv += ["--pose-strength", str(p.pose_strength)]
            return cmds + [(argv, "animate")]
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
                    "--ref", str(self.master(char) / f"{p.still_dir}.keyed.png"),
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


def size_note(ratio):
    """FR 8 (amendment 3): the size report of one direction still — the
    two-decimal ratio (`0.97`), plus ` CHECK` below SIZE_CHECK_LOW or above
    SIZE_CHECK_HIGH; an unmeasurable still (None) reads `n/a CHECK`."""
    if ratio is None:
        return "n/a CHECK"
    flag = " CHECK" if not SIZE_CHECK_LOW <= ratio <= SIZE_CHECK_HIGH else ""
    return f"{ratio:.2f}{flag}"


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


def pad_ref(src, dst, frame, canvas, log):
    """FR 10 (amendment 9): map a still to frame x frame the way the animate
    workflow does (ImageScale crop=center: the longer side is cropped
    symmetrically to a square, then the square is scaled to frame x frame,
    so an 832x1024 still keeps rows 96-927), and centre it on a canvas x
    canvas image filled with its top-left colour. FR 6 retires the 1280
    still canvas (`pad_still`); this remains for a larger animation canvas,
    where the reference keeps the mannequin's pixel size and aspect."""
    from PIL import Image

    frame, canvas = int(frame), int(canvas)
    im = Image.open(src).convert("RGB")
    w, h = im.size
    x, y = stills.centre_crop(w, h)
    im = im.crop((x, y, w - x, h - y)).resize((frame, frame), Image.LANCZOS)
    out = Image.new("RGB", (canvas, canvas), im.getpixel((0, 0)))
    off = (canvas - frame) // 2
    out.paste(im, (off, off))
    Path(dst).parent.mkdir(parents=True, exist_ok=True)
    out.save(dst)
    log.append(f"pad_ref {src} -> {dst} ({frame} on {canvas})")


def composite_drive(src_dir, dst_dir, frame_size, scale, background):
    """FR 1, 3: each transparent drive frame `src_dir/f*.png`, resized by
    `scale` (rounded), centred on a frame_size square of the flat
    `background` colour and written as RGB to `dst_dir` under the same name.
    (Slice 2's auto canvas replaces the frame_size square.)"""
    from PIL import Image

    dst_dir = Path(dst_dir)
    dst_dir.mkdir(parents=True, exist_ok=True)
    frame_size, scale, background = int(frame_size), float(scale), tuple(int(c) for c in background)
    for f in sorted(Path(src_dir).glob("f*.png")):
        im = Image.open(f).convert("RGBA")
        im = im.resize((round(im.width * scale), round(im.height * scale)), Image.LANCZOS)
        layer = Image.new("RGBA", (frame_size, frame_size), (0, 0, 0, 0))
        layer.paste(im, ((frame_size - im.width) // 2, (frame_size - im.height) // 2))
        out = Image.alpha_composite(Image.new("RGBA", layer.size, background + (255,)), layer)
        out.convert("RGB").save(dst_dir / f.name)


def main(argv=None):
    """`python -m sprite_pipeline.stages composite-drive SRC DST FRAME SCALE R G B`."""
    a = sys.argv[1:] if argv is None else argv
    if len(a) != 8 or a[0] != "composite-drive":
        sys.exit("usage: python -m sprite_pipeline.stages composite-drive SRC DST FRAME SCALE R G B")
    composite_drive(a[1], a[2], a[3], a[4], a[5:8])
    return 0


def copy_file(src, dst, log):
    """The drive copy, done in Python — no `cp` subprocess exists anywhere (AC8)."""
    Path(dst).parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src, dst)
    log.append(f"$ copy {src} -> {dst} (python)")
    return 0


def plan_cutout(still_path, out_dir):
    """Key the guided `s` master with `sprite-gen cutout --key measured`
    (FR 6, amendment 12): `out_dir/<stem>.keyed.png`, so the re-anchored
    `master/s.png` keys to `master/s.keyed.png`."""
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


if __name__ == "__main__":
    sys.exit(main())
