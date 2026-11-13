#!/usr/bin/env python3
"""Data-driven sprite-set builder: one character kit x the shared motion library.

    sprite-pipeline <kit.yaml> [--moves run,punch] [--dirs s,se] [--stage STAGE]
                    [--dry-run] [--force]

Stages per (move, dir): drive (Blender render, cached and shared across kits) ->
animate (Wan Animate 2 via comfy_run.py) -> key (sprite-gen video-frames) -> loop
(sprite-gen video-loop). Then per kit: pixel (one pixelize.py call over all sets)
-> export (mirrored frames for sw/w/nw, atlas.png + Godot 4 SpriteFrames .tres,
contact-sheet.png). Stages whose output already exists are skipped unless --force.
--stage runs only that one stage. A failed (move, dir) is recorded and the build
moves on; a summary table is printed at the end (exit 1 if anything failed).

Manifests are plain YAML; parsed with PyYAML when importable, otherwise the
minimal loader below (same subset).
"""
import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass, field, replace
from pathlib import Path

from sprite_pipeline import config

TOOLS = Path(__file__).resolve().parent

FACING = {"s": 0, "se": 45, "e": 90, "ne": 135, "n": 180}
MIRROR_OF = {"sw": "se", "w": "e", "nw": "ne"}
EIGHT_DIRS = ["n", "ne", "e", "se", "s", "sw", "w", "nw"]
VALID_STATES = {"idle", "walk", "run", "jump", "attack"}
PAIR_STAGES = ["drive", "animate", "key", "loop"]
KIT_STAGES = ["pixel", "export"]
STAGES = PAIR_STAGES + KIT_STAGES


# --------------------------------------------------------------------------- manifests

def _scalar(s):
    s = s.strip()
    if s.startswith('"') and s.endswith('"') and len(s) >= 2:
        out, i = [], 1
        while i < len(s) - 1:
            c = s[i]
            if c == "\\" and i + 1 < len(s) - 1:
                out.append({"n": "\n", "t": "\t", '"': '"', "\\": "\\"}.get(s[i + 1], s[i + 1]))
                i += 2
            else:
                out.append(c)
                i += 1
        return "".join(out)
    if s.startswith("'") and s.endswith("'") and len(s) >= 2:
        return s[1:-1].replace("''", "'")
    if s in ("true", "True"):
        return True
    if s in ("false", "False"):
        return False
    if s in ("null", "~", ""):
        return None
    try:
        return int(s)
    except ValueError:
        pass
    try:
        return float(s)
    except ValueError:
        pass
    return s


def _split_top(s, sep=","):
    parts, depth, quote, cur = [], 0, None, []
    for c in s:
        if quote:
            cur.append(c)
            if c == quote:
                quote = None
        elif c in "\"'":
            quote = c
            cur.append(c)
        elif c in "[{":
            depth += 1
            cur.append(c)
        elif c in "]}":
            depth -= 1
            cur.append(c)
        elif c == sep and depth == 0:
            parts.append("".join(cur))
            cur = []
        else:
            cur.append(c)
    parts.append("".join(cur))
    return parts


def _flow_list(s):
    inner = s.strip()[1:-1].strip()
    if not inner:
        return []
    return [_scalar(p) for p in _split_top(inner)]


def mini_yaml_loads(text):
    """Minimal YAML subset: nested maps by 2-space indent, block lists of scalars,
    flow lists, quoted/plain scalars. Inline comments are NOT supported — put
    comments on their own line."""
    lines = []
    for raw in text.splitlines():
        if not raw.strip() or raw.lstrip().startswith("#"):
            continue
        if "\t" in raw[: len(raw) - len(raw.lstrip())]:
            raise ValueError("tabs are not allowed for indentation")
        lines.append((len(raw) - len(raw.lstrip(" ")), raw.strip()))
    pos = 0

    def parse_block(indent):
        # lines entries are (indent, text)
        nonlocal pos
        if lines[pos][0] != indent:
            raise ValueError(f"bad indent near: {lines[pos][1]!r}")
        if lines[pos][1].startswith("- "):
            out = []
            while pos < len(lines) and lines[pos][0] == indent and lines[pos][1].startswith("- "):
                val = lines[pos][1][2:].strip()
                out.append(_flow_list(val) if val.startswith("[") else _scalar(val))
                pos += 1
            return out
        out = {}
        while pos < len(lines) and lines[pos][0] == indent and not lines[pos][1].startswith("- "):
            line = lines[pos][1]
            key, sep, rest = line.partition(":")
            if not sep:
                raise ValueError(f"expected 'key:' at: {line!r}")
            key = _scalar(key)
            rest = rest.strip()
            pos += 1
            if rest:
                out[key] = _flow_list(rest) if rest.startswith("[") else _scalar(rest)
            elif pos < len(lines) and lines[pos][0] > indent:
                out[key] = parse_block(lines[pos][0])
            elif pos < len(lines) and lines[pos][0] == indent and lines[pos][1].startswith("- "):
                out[key] = parse_block(indent)
            else:
                out[key] = None
        return out

    if not lines:
        return {}
    return parse_block(lines[0][0])


def load_yaml(path):
    text = Path(path).read_text()
    try:
        import yaml
    except ModuleNotFoundError:
        return mini_yaml_loads(text)
    return yaml.safe_load(text)


# --------------------------------------------------------------------------- models

@dataclass
class Paths:
    """Every external location build.py touches. Overridable in tests."""
    cache: Path
    comfy_input: Path
    blender: str
    sprite_gen: str
    extra_path: str = ""
    tools: Path = TOOLS

    @classmethod
    def from_config(cls, c):
        return cls(cache=Path(c["motion_cache"]), comfy_input=Path(c["comfy_input"]),
                   blender=c["blender"], sprite_gen=c["sprite_gen"], extra_path=c["extra_path"])


@dataclass
class MotionLib:
    default_glb: str
    elev: int
    fps: int
    moves: dict  # name -> Move


@dataclass
class Move:
    name: str
    action: str
    glb: str
    repeats: int = 1
    pre: int = 0
    post: int = 0
    head_pitch: float = 0.0
    clip_length: int = 0
    loop_mode: str = "once"
    loop_start: int = 0
    loop_length: int = 0
    seam_max: float = 0.0
    state: str = "attack"
    pose_prompt: str = ""
    directions: list = field(default_factory=lambda: list(FACING))
    status: str = "untested"

    @classmethod
    def from_dict(cls, name, d, default_glb):
        return cls(
            name=name, action=d.get("action"), glb=d.get("glb") or default_glb,
            repeats=int(d.get("repeats", 1)), pre=int(d.get("pre", 0)), post=int(d.get("post", 0)),
            head_pitch=float(d.get("head_pitch", 0) or 0), clip_length=int(d["clip_length"]),
            loop_mode=d.get("loop_mode", "once"), loop_start=int(d.get("loop_start", 0)),
            loop_length=int(d.get("loop_length", 0)), seam_max=float(d.get("seam_max", 0) or 0), state=d.get("state", "attack"),
            pose_prompt=d.get("pose_prompt", ""),
            directions=d.get("directions") or list(FACING), status=d.get("status", "untested"))


@dataclass
class Kit:
    name: str
    output_root: Path
    seed: int
    frame_size: int
    pixel_scale: float
    pixel_colors: int
    accent_boost: int
    palette: str
    look: str
    face: str
    face_dirs: list
    background: str
    refs: dict
    pose_who: str = ""
    pose_strength: float = 1.0
    ref_strength: float = 1.0
    overrides: dict = field(default_factory=dict)

    @classmethod
    def from_dict(cls, d):
        for k in ("name", "output_root", "seed", "frame_size", "pixel_scale", "pixel_colors",
                  "accent_boost", "palette", "look", "face", "face_dirs", "background", "refs"):
            if k not in d or d[k] is None:
                raise ValueError(f"kit missing field {k!r}")
        return cls(name=d["name"], output_root=Path(d["output_root"]), seed=int(d["seed"]),
                   frame_size=int(d["frame_size"]), pixel_scale=float(d["pixel_scale"]),
                   pixel_colors=int(d["pixel_colors"]), accent_boost=int(d["accent_boost"]),
                   palette=d["palette"], look=d["look"], face=d["face"], face_dirs=list(d["face_dirs"]),
                   background=d["background"], refs=dict(d["refs"]),
                   pose_who=d.get("pose_who") or "", pose_strength=float(d.get("pose_strength", 1.0)),
                   ref_strength=float(d.get("ref_strength", 1.0)), overrides=dict(d.get("overrides") or {}))

    def override(self, move, d):
        """Per (move, dir) reroll settings, keyed "<move>-<dir>": seed and an extra prompt line."""
        return self.overrides.get(f"{move}-{d}") or {}

    def prompt_for(self, d, move=None):
        extra = self.override(move, d).get("extra", "") if move else ""
        return self.look + (self.face if d in self.face_dirs else "") + (" " + extra if extra else "") + self.background

    def pose_prompt(self, m):
        return m.pose_prompt.replace("{who}", self.pose_who or "a person")


def load_motion_lib(path):
    d = load_yaml(path)
    moves = {}
    for name, md in (d.get("moves") or {}).items():
        moves[name] = Move.from_dict(name, md, d.get("default_glb", ""))
    return MotionLib(default_glb=d.get("default_glb", ""), elev=int(d.get("elev", 45)),
                     fps=int(d.get("fps", 30)), moves=moves)


def validate(lib, kit, move_names, dirs):
    for n in move_names:
        if n not in lib.moves:
            raise ValueError(f"unknown move {n!r} (have: {', '.join(sorted(lib.moves))})")
    for d in dirs:
        if d not in FACING:
            raise ValueError(f"unknown generation direction {d!r} (valid: {', '.join(FACING)})")
    for d in kit.face_dirs:
        if d not in FACING:
            raise ValueError(f"face_dirs entry {d!r} is not a real direction")
    for n in move_names:
        m = lib.moves[n]
        if not m.action or not m.glb:
            raise ValueError(f"move {n}: missing action or glb")
        if m.clip_length % 4 != 1:
            raise ValueError(f"move {n}: clip_length {m.clip_length} is not 4n+1 (Wan requirement)")
        if m.loop_mode not in ("fixed", "once"):
            raise ValueError(f"move {n}: loop_mode {m.loop_mode!r} not in fixed|once")
        if m.state not in VALID_STATES:
            raise ValueError(f"move {n}: state {m.state!r} not one of {sorted(VALID_STATES)}")
        if not (1 <= m.repeats and m.pre >= 0 and m.post >= 0):
            raise ValueError(f"move {n}: bad repeats/pre/post")
        if not (1 <= m.loop_length and m.loop_start >= 0 and m.loop_start + m.loop_length <= m.clip_length):
            raise ValueError(f"move {n}: loop start/length out of the clip")
        bad = [x for x in m.directions if x not in FACING]
        if bad:
            raise ValueError(f"move {n}: unknown directions {bad}")
    missing = [d for d in dirs if d not in kit.refs]
    if missing:
        raise ValueError(f"kit has no ref for direction(s) {missing}")


def resolve_glbs(lib, anim_src):
    """Relative GLB paths in motions.yaml are relative to the configured anim_src."""
    def res(g):
        return g if not g or Path(g).is_absolute() else str(Path(anim_src) / g)
    moves = {n: replace(m, glb=res(m.glb)) for n, m in lib.moves.items()}
    return replace(lib, default_glb=res(lib.default_glb), moves=moves)


# --------------------------------------------------------------------------- planning

def drive_hash(m, elev, frame_size):
    """Cache key over the move params that change the rendered drive."""
    payload = json.dumps(
        {"glb": m.glb, "action": m.action, "repeats": m.repeats, "pre": m.pre, "post": m.post,
         "head_pitch": m.head_pitch, "elev": elev, "frame_size": frame_size},
        sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


class Ctx:
    """Everything a stage command needs, resolved once per build."""

    def __init__(self, lib, kit, move_names, dirs, paths, force):
        self.lib, self.kit, self.paths, self.force = lib, kit, paths, force
        self.move_names, self.dirs = move_names, dirs
        self.env = {**os.environ}
        if paths.extra_path:
            self.env["PATH"] = f"{paths.extra_path}{os.pathsep}{self.env.get('PATH', '')}"
        self.env["RW"] = str(kit.frame_size)
        self.env["RH"] = str(kit.frame_size)
        self.hashes = {n: drive_hash(lib.moves[n], lib.elev, kit.frame_size) for n in move_names}

    # -- per (move, dir) locations
    def cache_mp4(self, m, d):
        return self.paths.cache / f"{m.name}-{d}-{self.hashes[m.name]}.mp4"

    def input_mp4(self, m, d):
        return self.paths.comfy_input / f"drive-{m.name}-{d}-{self.hashes[m.name]}.mp4"

    def frames_dir(self, m, d):
        return self.paths.cache / f"{m.name}-{d}-{self.hashes[m.name]}" / "frames"

    def clip(self, m, d):
        return self.kit.output_root / "clips" / f"{m.name}-{d}.mp4"

    def work_dir(self, m, d):
        return self.kit.output_root / "work" / f"frames-{m.name}-{d}"

    def loops_dir(self, m, d):
        return self.kit.output_root / "loops" / f"{m.name}-{d}"

    def sprite_out(self):
        return self.kit.output_root / "sprites" / self.kit.name

    # -- stage output predicates (skip-if-exists)
    def stage_done(self, stage, m=None, d=None):
        k = self.kit
        if stage == "drive":
            return self.cache_mp4(m, d).is_file() and self.input_mp4(m, d).is_file()
        if stage == "animate":
            return self.clip(m, d).is_file()
        if stage == "key":
            keyed = self.work_dir(m, d) / "keyed"
            if keyed.is_dir() and any(keyed.glob("*.png")):
                return True
            return self.stage_done("loop", m, d)  # keyed frames are disposable once the loop is cut
        if stage == "loop":
            cyc = self.loops_dir(m, d) / "cycle"
            return cyc.is_dir() and any(cyc.glob("*.png"))
        if stage == "pixel":
            return all((self.sprite_out() / f"{d}-{n}_000.png").is_file()
                       for n in self.move_names for d in self.dirs)
        if stage == "export":
            out = self.sprite_out()
            return all((out / f).is_file() for f in ("atlas.png", "sprite_frames.tres", "contact-sheet.png"))
        raise ValueError(stage)

    # -- command planning: returns list of (argv, log-label)
    def plan_pair(self, m, d, stage):
        k, p, lib = self.kit, self.paths, self.lib
        if stage == "drive":
            frames = self.frames_dir(m, d)
            argv = [p.blender, "-b", "--python", str(p.tools / "render_drive.py"), "--",
                    m.glb, m.action, str(frames), str(FACING[d]), str(lib.elev),
                    str(m.repeats), str(m.pre), str(m.post)]
            if m.head_pitch:
                argv.append(f"{m.head_pitch:g}")
            cmds = [(argv, "blender"),
                    (["ffmpeg", "-v", "error", "-y", "-framerate", str(lib.fps),
                      "-i", f"{frames}/f%04d.png", "-c:v", "libx264", "-crf", "12",
                      "-pix_fmt", "yuv420p", str(self.cache_mp4(m, d))], "ffmpeg"),
                    (["cp", "-f", str(self.cache_mp4(m, d)), str(self.input_mp4(m, d))], "copy-drive")]
            return cmds
        if stage == "animate":
            argv = [sys.executable, "-m", "sprite_pipeline.comfy_run", "animate",
                    "--ref", k.refs[d], "--video", self.input_mp4(m, d).name,
                    "--w", str(k.frame_size), "--h", str(k.frame_size),
                    "--length", str(m.clip_length), "--seed", str(k.override(m.name, d).get("seed", k.seed)),
                    "--prefix", f"{k.name}/dir8-{m.name}-{d}",
                    "--copy-to", str(self.clip(m, d)),
                    "--pose-prompt", k.pose_prompt(m), "--prompt", k.prompt_for(d, m.name)]
            if k.pose_strength != 1.0:
                argv += ["--pose-strength", str(k.pose_strength)]
            if k.ref_strength != 1.0:
                argv += ["--ref-strength", str(k.ref_strength)]
            return [(argv, "animate")]
        if stage == "key":
            argv = [p.sprite_gen, "video-frames", "--clip", str(self.clip(m, d)),
                    "--out-dir", str(self.work_dir(m, d)), "--spill", "small",
                    "--allow-subject-edge-contact"]
            return [(argv, "key")]
        if stage == "loop":
            argv = [p.sprite_gen, "video-loop", "--frames-dir", str(self.work_dir(m, d) / "keyed"),
                    "--out-dir", str(self.loops_dir(m, d)), "--fps", str(lib.fps),
                    "--state", m.state, "--cycle", m.loop_mode,
                    "--start", str(m.loop_start), "--length", str(m.loop_length)]
            if m.loop_mode == "fixed":
                argv += ["--seam-pop-max", "off"]
            if m.seam_max:
                argv += ["--seam-max", f"{m.seam_max:g}"]
            argv += ["--name", f"{d}-{m.name}"]
            return [(argv, "loop")]
        raise ValueError(stage)

    def plan_kit_stage(self, stage):
        k, p = self.kit, self.paths
        if stage == "pixel":
            argv = [sys.executable, "-m", "sprite_pipeline.pixelize", "--ref", k.palette,
                    "--scale", str(k.pixel_scale), "--colors", str(k.pixel_colors),
                    "--accent-boost", str(k.accent_boost), "--fps", str(self.lib.fps),
                    "--out", str(self.sprite_out())]
            for d in self.dirs:  # dir-major, matching batch_dir8.sh's set order
                for n in self.move_names:
                    argv += ["--set", f"{d}-{n}={self.loops_dir(self.lib.moves[n], d) / 'cycle'}"]
            return [(argv, "pixel")]
        if stage == "export":
            return []  # pure python, no subprocess
        raise ValueError(stage)


# --------------------------------------------------------------------------- execution

def run_cmd(argv, ctx, label, log):
    log.append("$ " + " ".join(_quote(a) for a in argv))
    r = subprocess.run(argv, env=ctx.env, capture_output=True, text=True)
    if r.stdout.strip():
        log.append(r.stdout.strip())
    if r.stderr.strip():
        log.append(r.stderr.strip())
    return r.returncode


def _quote(a):
    return a if all(c.isalnum() or c in "/-_.=:+,@%" for c in a) else "'" + a.replace("'", "'\\''") + "'"


def do_export(ctx, log, dry):
    """Mirror sw/w/nw frames, pack the atlas, write the Godot 4 SpriteFrames .tres
    and a contact sheet. Returns (ok, missing)."""
    from PIL import Image
    k, out = ctx.kit, ctx.sprite_out()
    frames, missing = {}, []
    for n in ctx.move_names:
        for d in EIGHT_DIRS:
            src = MIRROR_OF.get(d, d)
            if src not in ctx.dirs:
                missing.append(f"{n}/{d}")
                continue
            files = sorted(out.glob(f"{src}-{n}_[0-9][0-9][0-9].png"))  # not the _sheet.png strips
            if not files:
                missing.append(f"{n}/{d}")
                continue
            if d != src:
                flips = []
                for f in files:
                    m = out / f"{d}-{n}_{f.name.rsplit('_', 1)[1]}"
                    if dry or not m.is_file():
                        Image.open(f).convert("RGBA").transpose(Image.FLIP_LEFT_RIGHT).save(m)
                    flips.append(m)
                files = flips
            frames[(n, d)] = files

    sizes = {im.size for fs in frames.values() for im in (Image.open(f) for f in fs)}
    if not sizes:
        return False, missing
    if len(sizes) != 1:
        log.append(f"export: mixed frame sizes {sorted(sizes)} — cannot pack one atlas")
        return False, missing
    w, h = sizes.pop()

    rows = [(n, d) for n in ctx.move_names for d in EIGHT_DIRS if (n, d) in frames]
    ncol = max(len(frames[r]) for r in rows)
    if not dry:
        atlas = Image.new("RGBA", (ncol * w, len(rows) * h), (0, 0, 0, 0))
        region = {}
        for ri, (n, d) in enumerate(rows):
            for i, f in enumerate(frames[(n, d)]):
                im = Image.open(f).convert("RGBA")
                atlas.paste(im, (i * w, ri * h), im)
                region[(n, d, i)] = (i * w, ri * h, w, h)
        atlas.save(out / "atlas.png")
        anims = []
        for (n, d) in rows:
            regions = [region[(n, d, i)] for i in range(len(frames[(n, d)]))]
            anims.append((f"{n}_{d}", ctx.lib.moves[n].loop_mode == "fixed", regions))
        write_tres(out / "sprite_frames.tres", "atlas.png", anims, fps=ctx.lib.fps)
        sheet = Image.new("RGBA", (8 * w * 4, len(ctx.move_names) * h * 4), (40, 42, 52, 255))
        for ri, n in enumerate(ctx.move_names):
            for ci, d in enumerate(EIGHT_DIRS):
                first = frames.get((n, d))
                if not first:
                    continue
                im = Image.open(first[0]).convert("RGBA").resize((w * 4, h * 4), Image.NEAREST)
                sheet.paste(im, (ci * w * 4, ri * h * 4), im)
        sheet.save(out / "contact-sheet.png")
    log.append(f"export: {len(rows)} of {len(ctx.move_names) * 8} animations packed"
               + (f", missing: {', '.join(missing)}" if missing else ""))
    return not missing, missing


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


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("kit", help="path to kits/<name>.yaml")
    ap.add_argument("--moves", help="comma-separated move names (default: all in the kit's dirs)")
    ap.add_argument("--dirs", help="comma-separated directions (default: from the moves)")
    ap.add_argument("--stage", choices=STAGES, help="run only this stage")
    ap.add_argument("--dry-run", action="store_true", help="print commands, run nothing")
    ap.add_argument("--force", action="store_true", help="rerun stages whose output exists")
    ap.add_argument("--motions", default=str(TOOLS / "motions.yaml"), help="motion library path")
    a = ap.parse_args(argv)

    lib = load_motion_lib(a.motions)
    kit = Kit.from_dict(load_yaml(a.kit))
    move_names = [x for x in (a.moves.split(",") if a.moves else list(lib.moves))]
    dirs = [x for x in (a.dirs.split(",") if a.dirs else sorted({d for n in move_names for d in lib.moves[n].directions}, key=list(FACING).index))]
    try:
        validate(lib, kit, move_names, dirs)
    except ValueError as e:
        print(f"manifest error: {e}", file=sys.stderr)
        return 2
    c = config.load()
    lib = resolve_glbs(lib, Path(c["anim_src"]))
    ctx = Ctx(lib, kit, move_names, dirs, Paths.from_config(c), force=a.force)

    stages = [a.stage] if a.stage else STAGES
    pair_stages = [s for s in stages if s in PAIR_STAGES]
    log, status = [], {}
    failed = []

    for d in dirs:
        for n in move_names:
            m = lib.moves[n]
            row = {}
            for s in pair_stages:
                if ctx.stage_done(s, m, d) and not ctx.force:
                    row[s] = "skip"
                    continue
                if a.dry_run:
                    for argv, label in ctx.plan_pair(m, d, s):
                        print(f"[{n}/{d}] {s} ({label})  " + " ".join(_quote(x) for x in argv))
                    row[s] = "dry"
                    continue
                ok = True
                if s == "key":
                    shutil.rmtree(ctx.work_dir(m, d), ignore_errors=True)
                    shutil.rmtree(ctx.loops_dir(m, d), ignore_errors=True)
                if s == "drive":
                    shutil.rmtree(ctx.frames_dir(m, d).parent, ignore_errors=True)
                    ctx.paths.cache.mkdir(parents=True, exist_ok=True)
                    ctx.paths.comfy_input.mkdir(parents=True, exist_ok=True)
                for argv, label in ctx.plan_pair(m, d, s):
                    if run_cmd(argv, ctx, label, log) != 0:
                        ok = False
                        break
                if not ok:
                    row[s] = "FAIL"
                    failed.append(f"{n}/{d} {s}")
                    break
                row[s] = "ok"
            status[(n, d)] = row
            print(f"[{n}/{d}] " + "  ".join(f"{s}={row[s]}" for s in pair_stages))

    for s in stages:
        if s not in KIT_STAGES:
            continue
        if s == "pixel" and ctx.stage_done("pixel") and not ctx.force:
            print("[pixel] skip")
            continue
        if a.dry_run:
            for argv, label in ctx.plan_kit_stage(s):
                print(f"[kit] {s}  " + " ".join(_quote(x) for x in argv))
            if s == "export":
                print("[kit] export  (mirrors sw<-se w<-e nw<-ne; atlas.png + sprite_frames.tres + contact-sheet.png)")
            continue
        if s == "export":
            ok, missing = do_export(ctx, log, dry=False)
            print(f"[export] {'ok' if ok else 'incomplete'}" + (f": {log[-1]}" if log else ""))
            if missing:
                failed.append(f"export missing: {', '.join(missing)}")
        else:
            shutil.rmtree(ctx.sprite_out(), ignore_errors=True)  # batch_dir8.sh clears the output first
            for argv, label in ctx.plan_kit_stage(s):
                if run_cmd(argv, ctx, label, log) != 0:
                    print("[pixel] FAIL")
                    failed.append("pixel")
                    break
            else:
                print("[pixel] ok")

    # summary table
    print("\nsummary")
    header = "move/dir".ljust(14) + "".join(s.ljust(9) for s in PAIR_STAGES)
    print(header)
    for (n, d), row in status.items():
        print(f"{n}/{d}".ljust(14) + "".join(row.get(s, "-").ljust(9) for s in PAIR_STAGES))
    if failed:
        print(f"\nFAILED: {len(failed)}: {', '.join(failed)}")
        return 1
    print("all stages ok" if not a.dry_run else "dry run complete")
    return 0


if __name__ == "__main__":
    sys.exit(main())
