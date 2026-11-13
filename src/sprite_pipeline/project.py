#!/usr/bin/env python3
"""Project-file model: load and validate a sprite pipeline `project.yaml`.

One project file describes a whole sprite job; every relative path in it
resolves against the directory the file lives in, so a game folder can be
copied to another machine (spec FR 1). Views (FR 5) are package data:
`views/<view>.yaml` holds the camera prompt line, the face line and the
turn prompt for each direction. `load_project()` raises ProjectError, a
ValueError carrying the `<project file>: <field>: <reason>` message, for
every FR 4 case.
"""
import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

import yaml

VIEWS_DIR = Path(__file__).resolve().parent / "views"

KNOWN_DIRS = {"n", "ne", "e", "se", "s", "sw", "w", "nw"}
VALID_STATES = {"idle", "walk", "run", "jump", "attack"}
PRONOUNS = {"he", "she", "they"}


class ProjectError(ValueError):
    """The project file is invalid; the message names the offending field."""


@dataclass
class Sprite:
    scale: float
    colors: int
    accent_boost: int
    fps: int
    frame_size: int

    @classmethod
    def from_dict(cls, d):
        return cls(scale=float(d["scale"]), colors=int(d["colors"]),
                   accent_boost=int(d["accent_boost"]), fps=int(d["fps"]),
                   frame_size=int(d["frame_size"]))

@dataclass
class Character:
    name: str
    still: Path
    describe: str
    pronoun: str
    who: str = ""
    face_dirs: list = field(default_factory=list)
    overrides: dict = field(default_factory=dict)


@dataclass
class Animation:
    name: str
    source: Path          # .glb path, resolved against the project file
    action: str           # the #<Action> part of the source
    frames: int
    loop: bool
    prompt: str
    start: int = 0
    clip_frames: int = 0
    repeats: int = 1
    pre: int = 1
    post: int = 0
    look_down: bool = False
    state: str = "attack"
    seam_max: float = 0.0
    directions: list = field(default_factory=list)

    @property
    def head_lift(self):
        """Degrees the head is lifted in the drive: 0 for moves that look down, 25 else."""
        return 0 if self.look_down else 25


@dataclass
class Project:
    path: Path
    style: str
    view: str
    generate: list
    mirror: dict
    sprite: Sprite
    naming: str
    export: str
    seed: int
    pose_strength: float
    output: Path
    rig: dict
    workflows: dict
    characters: list
    animations: list

    def output_root(self):
        return self.path.parent / self.output

    def animation(self, name):
        for a in self.animations:
            if a.name == name:
                return a
        raise ProjectError(f"animation {name!r} not found")

    def drive_key(self, anim, glb_bytes):
        """Cache key over the source file content, the action and the drive params (FR 10, dec: dec-629).

        The key hashes the GLB bytes and never names the file path, so copying
        the GLB elsewhere leaves the cached drive valid. elev is 45, the value
        in the current motion library.
        """
        payload = json.dumps(
            {"glb": hashlib.sha256(glb_bytes).hexdigest(), "action": anim.action,
             "repeats": anim.repeats, "pre": anim.pre, "post": anim.post,
             "head_lift": anim.head_lift, "rig": {"neck": self.rig["neck"], "head": self.rig["head"]},
             "elev": 45, "frame_size": self.sprite.frame_size},
            sort_keys=True)
        return hashlib.sha256(payload.encode()).hexdigest()[:16]


def load_view(name, views_dir=VIEWS_DIR):
    """Package-data view preset: camera prompt, face line, turn prompt per direction."""
    path = Path(views_dir) / f"{name}.yaml"
    if not path.is_file():
        raise ProjectError(f"view {name!r} has no preset views/{name}.yaml")
    d = yaml.safe_load(path.read_text()) or {}
    return d


def _clip_frames_default(start, frames):
    """start + frames rounded up to 4n+1 (FR 3)."""
    n = start + frames
    if n % 4 == 1:
        return n
    return n + (5 - n % 4) % 4  # next 4n+1, e.g. 23 -> 25


def load_project(path, views_dir=VIEWS_DIR):
    """Load, resolve paths against the project file and validate (FR 1-5).

    Raises ProjectError (a ValueError) naming the field for every FR 4 case.
    """
    path = Path(path).resolve()
    try:
        d = yaml.safe_load(path.read_text()) or {}
    except yaml.YAMLError as e:
        raise ProjectError(f"{path.name}: invalid YAML: {e}") from e
    view = d.get("view")
    if not view:
        raise ProjectError(f"{path.name}: view: missing")
    try:
        preset = load_view(view, views_dir)
    except ProjectError as e:
        raise ProjectError(f"{path.name}: view: {e}") from e
    turn = preset.get("turn") or {}

    # directions (FR 4: unknown direction; mirror source not in generate)
    dirs = d.get("directions") or {}
    generate = list(dirs.get("generate") or [])
    if not generate:
        raise ProjectError(f"{path.name}: directions.generate: empty")
    for i, x in enumerate(generate):
        if x not in KNOWN_DIRS:
            raise ProjectError(f"{path.name}: directions.generate[{i}]: unknown direction {x!r} for view {view}")
    if len(generate) != len(set(generate)):
        raise ProjectError(f"{path.name}: directions.generate: directions not unique")
    mirror = dict(dirs.get("mirror") or {})
    for m in mirror:
        if m not in KNOWN_DIRS:
            raise ProjectError(f"{path.name}: directions.mirror.{m}: unknown direction {m!r} for view {view}")
        if m in generate:
            raise ProjectError(f"{path.name}: directions.mirror.{m}: {m!r} is both generated and mirrored")
        if mirror[m] not in generate:
            if mirror[m] not in KNOWN_DIRS:
                raise ProjectError(f"{path.name}: directions.mirror.{m}: unknown direction {mirror[m]!r} for view {view}")
            raise ProjectError(f"{path.name}: directions.mirror.{m}: mirror source {mirror[m]!r} not in generate")

    # FR 4: a view with no turn prompt for a generated direction
    for x in generate:
        if x not in turn:
            raise ProjectError(f"{path.name}: directions.generate: view {view} has no turn prompt for direction {x!r} "
                               f"(missing from views/{view}.yaml)")

    sprite = Sprite.from_dict(d.get("sprite") or {})
    naming = d.get("naming") or "{animation}_{dir}"
    export = d.get("export") or "godot"
    if export != "godot":
        raise ProjectError(f"{path.name}: export: {export!r} (only godot is supported)")
    seed = int(d.get("seed", 0))
    pose_strength = float(d.get("pose_strength", 1.0))
    output = Path(d.get("output") or "out")
    rig = d.get("rig") or {}
    rig = {"neck": rig.get("neck") or "DEF-neck", "head": rig.get("head") or "DEF-head"}
    workflows = dict(d.get("workflows") or {})
    for k in ("edit", "animate"):
        if not workflows.get(k):
            raise ProjectError(f"{path.name}: workflows.{k}: missing")

    proj = Project(path=path, style=d.get("style") or "", view=view, generate=generate, mirror=mirror,
                   sprite=sprite, naming=naming, export=export, seed=seed,
                   pose_strength=pose_strength, output=output, rig=rig, workflows=workflows,
                   characters=[], animations=[])

    # characters (FR 2; FR 4: missing still)
    for i, c in enumerate(d.get("characters") or []):
        name = c.get("name")
        if not name:
            raise ProjectError(f"{path.name}: characters[{i}].name: missing")
        still = c.get("still")
        if not still:
            raise ProjectError(f"{path.name}: characters[{i}].still: missing")
        still_p = Path(still)
        if not still_p.is_absolute():
            still_p = path.parent / still_p
        if not still_p.is_file():
            raise ProjectError(f"{path.name}: characters[{i}].still: {still} not found")
        if c.get("pronoun") not in PRONOUNS:
            raise ProjectError(f"{path.name}: characters[{i}].pronoun: {c.get('pronoun')!r} (he, she or they)")
        face_dirs = c.get("face_dirs") or []
        for x in face_dirs:
            if x not in generate:
                raise ProjectError(f"{path.name}: characters[{i}].face_dirs: {x!r} not in directions.generate")
        proj.characters.append(Character(name=name, still=still_p, describe=c.get("describe") or "",
                                         pronoun=c["pronoun"], who=c.get("who") or "",
                                         face_dirs=list(face_dirs),
                                         overrides=dict(c.get("overrides") or {})))

    # animations (FR 3; FR 4: missing source, no #<Action>, clip_frames, naming unique)
    for i, a in enumerate(d.get("animations") or []):
        name = a.get("name")
        if not name:
            raise ProjectError(f"{path.name}: animations[{i}].name: missing")
        source = a.get("source")
        if not source:
            raise ProjectError(f"{path.name}: animations[{i}].source: missing")
        file_part, sep, action = source.partition("#")
        if not sep or not file_part or not action:
            raise ProjectError(f"{path.name}: animations[{i}].source: missing #<Action>")
        src_p = Path(file_part)
        if not src_p.is_absolute():
            src_p = path.parent / src_p
        if not src_p.is_file():
            raise ProjectError(f"{path.name}: animations[{i}].source: {file_part} not found")
        if src_p.suffix != ".glb":
            raise ProjectError(f"{path.name}: animations[{i}].source: {file_part} is not a .glb")
        frames = int(a.get("frames") or 0)
        if frames <= 0:
            raise ProjectError(f"{path.name}: animations[{i}].frames: must be positive")
        start = int(a.get("start") or 0)
        clip = a.get("clip_frames")
        too_short = False
        if clip is None:
            clip = _clip_frames_default(start, frames)
        else:
            clip = int(clip)
            if clip % 4 != 1:
                raise ProjectError(f"{path.name}: animations[{i}].clip_frames: {clip} is not 4n+1")
            too_short = clip < start + frames
        if too_short:
            # an unlooped 'once' clip may end before start + frames (its cycle
            # truncates); a looped one must hold the whole cycle
            if a.get("loop", False):
                raise ProjectError(f"{path.name}: animations[{i}].clip_frames: {clip} is shorter than start + frames "
                                   f"({start} + {frames} = {start + frames})")
            elif clip < frames:
                raise ProjectError(f"{path.name}: animations[{i}].clip_frames: {clip} is shorter than frames "
                                   f"({frames})")
        anim_dirs = a.get("directions") or list(generate)
        for x in anim_dirs:
            if x not in generate:
                raise ProjectError(f"{path.name}: animations[{i}].directions: {x!r} not in directions.generate")
        state = a.get("state") or "attack"
        if state not in VALID_STATES:
            raise ProjectError(f"{path.name}: animations[{i}].state: {state!r} not one of {sorted(VALID_STATES)}")
        proj.animations.append(Animation(name=name, source=src_p, action=action, frames=frames,
                                         loop=bool(a.get("loop", False)), prompt=a.get("prompt") or "",
                                         start=start, clip_frames=clip,
                                         repeats=int(a.get("repeats") or 1), pre=int(a.get("pre", 1)),
                                         post=int(a.get("post") or 0), look_down=bool(a.get("look_down", False)),
                                         state=state, seam_max=float(a.get("seam_max") or 0),
                                         directions=list(anim_dirs)))

    # FR 4: a naming pattern that does not produce unique names
    seen = {}
    for c in proj.characters:
        for a in proj.animations:
            for x in a.directions:
                n = (naming.replace("{character}", c.name)
                          .replace("{animation}", a.name)
                          .replace("{dir}", x))
                if n in seen:
                    raise ProjectError(f"{path.name}: naming: {naming!r} gives duplicate name {n!r} ({seen[n]} and {c.name}/{a.name}/{x})")
                seen[n] = f"{c.name}/{a.name}/{x}"

    return proj
