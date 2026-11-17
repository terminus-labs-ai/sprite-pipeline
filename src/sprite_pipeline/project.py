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
from importlib import resources
from pathlib import Path

import yaml

from sprite_pipeline import metrics

VIEWS_DIR = Path(__file__).resolve().parent / "views"

# the host profile file, where `mannequin-profile` appends measured profiles
# (FR 14); tests inject SPRITE_PIPELINE_HOME
USER_PROFILES_FILE = "mannequin_profiles.json"

KNOWN_DIRS = {"n", "ne", "e", "se", "s", "sw", "w", "nw"}
VALID_STATES = {"idle", "walk", "run", "jump", "attack"}

# the drive scene's camera elevation: the one definition, used by the drive
# plan (stages.plan_pair) and by its cache key (Project.drive_key)
DRIVE_ELEV = 45
# pronoun -> the pose subject when a character gives no `who` (FR 2)
DEFAULT_WHO = {"he": "a man", "she": "a woman", "they": "a person"}
PRONOUNS = set(DEFAULT_WHO)


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
    describe: object  # one string for every direction, or a map direction -> text
    pronoun: str
    who: str = ""
    face_dirs: list = field(default_factory=list)
    overrides: dict = field(default_factory=dict)

    def look(self, d):
        """The character's description for direction `d`."""
        return self.describe[d] if isinstance(self.describe, dict) else self.describe


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
    workflow: Path = None  # per-animation override of workflows.animate (FR 3)
    canvas: int = 0       # square render size in px; 0 until load sets sprite.frame_size

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

    @property
    def still_dir(self):
        """FR 6: the still's own direction is the master (AC15): the source
        still is anchored to `master/s.png`, keyed to `master/s.keyed.png`
        and is the edit input of every other generated direction."""
        return "s"

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
        the GLB elsewhere leaves the cached drive valid. elev is DRIVE_ELEV,
        the value in the current motion library.
        """
        payload = json.dumps(
            {"glb": hashlib.sha256(glb_bytes).hexdigest(), "action": anim.action,
             "repeats": anim.repeats, "pre": anim.pre, "post": anim.post,
             "head_lift": anim.head_lift, "rig": {"neck": self.rig["neck"], "head": self.rig["head"]},
             "elev": DRIVE_ELEV, "frame_size": self.sprite.frame_size,
             **({"canvas": anim.canvas} if anim.canvas and anim.canvas != self.sprite.frame_size else {})},
            sort_keys=True)
        return hashlib.sha256(payload.encode()).hexdigest()[:16]


def load_view(name, views_dir=VIEWS_DIR):
    """Package-data view preset: camera prompt, face line, turn prompt per direction."""
    path = Path(views_dir) / f"{name}.yaml"
    if not path.is_file():
        raise ProjectError(f"view {name!r} has no preset views/{name}.yaml")
    d = yaml.safe_load(path.read_text()) or {}
    return d


def user_profiles_path():
    """The host mannequin-profile file (FR 14): `<config home>/mannequin_profiles.json`,
    where `mannequin-profile` appends measured profiles."""
    from sprite_pipeline import config

    return config.home() / USER_PROFILES_FILE


def anchor_profiles():
    """FR 14: the profile map a load validates against and `go` anchors with:
    the packaged profiles merged with the host file, last-write-wins per
    hash. (A test injects its own map through `load_project(profiles=…)`
    instead of writing the host file.)"""
    return metrics.load_profiles()


def anchor_profile(proj, profiles=None):
    """The project's anchor profile (FR 6, FR 14): the profile of the
    first-listed animation's source GLB. Load checked every distinct source GLB
    against `profiles` (FR 4), so the key is there; an explicit `profiles` map
    (a caller that merged the host file) wins over the packaged file."""
    digest = metrics.glb_sha256(proj.animations[0].source)
    if profiles is not None and digest in profiles:
        return profiles[digest]
    return anchor_profiles()[digest]


def _clip_frames_default(start, frames):
    """start + frames rounded up to 4n+1 (FR 3)."""
    n = start + frames
    if n % 4 == 1:
        return n
    return n + (5 - n % 4) % 4  # next 4n+1, e.g. 23 -> 25


def load_project(path, views_dir=VIEWS_DIR, profiles=None):
    """Load, resolve paths against the project file and validate (FR 1-5).

    `profiles` is the mannequin-profile map the profile gate (FR 4) reads:
    by default the packaged profiles merged with the host file, keyed by
    GLB sha256 (FR 14). Raises ProjectError (a ValueError) naming the field for every FR 4 case.
    """
    if profiles is None:
        profiles = anchor_profiles()
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

    # a view with no facing line for a generated direction
    facing = preset.get("facing") or {}
    for x in generate:
        if x not in facing:
            raise ProjectError(f"{path.name}: directions.generate: view {view} has no facing line for direction {x!r} "
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
    # workflows (FR 1, amendment 1): each missing key defaults to the
    # packaged workflow file; a given path resolves against the project file
    packaged = resources.files("sprite_pipeline") / "workflows"
    workflows = {}
    for k in ("edit", "animate"):
        wf = (d.get("workflows") or {}).get(k)
        if not wf:
            workflows[k] = packaged / f"{k}.json"
        else:
            wf_p = Path(wf)
            workflows[k] = (wf_p if wf_p.is_absolute() else path.parent / wf_p)

    proj = Project(path=path, style=d.get("style") or "", view=view, generate=generate, mirror=mirror,
                   sprite=sprite, naming=naming, export=export, seed=seed,
                   pose_strength=pose_strength, output=output, rig=rig, workflows=workflows,
                   characters=[], animations=[])

    # the view's camera-facing directions (FR 2): the preset's face_dirs
    # intersected with directions.generate, in generate order; a preset
    # without the key contributes nothing
    preset_face = [x for x in generate if x in (preset.get("face_dirs") or [])]

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
        # face_dirs (FR 2): omitting the key defaults to the view's camera-
        # facing directions that are generated (preset face_dirs ∩ generate,
        # in generate order); an explicit list, including an explicit [],
        # is kept as given — [] means no face line.
        if "face_dirs" in c:
            face_dirs = list(c["face_dirs"] or [])
        else:
            face_dirs = list(preset_face)
        for x in face_dirs:
            if x not in generate:
                raise ProjectError(f"{path.name}: characters[{i}].face_dirs: {x!r} not in directions.generate")
        if "describe_back" in c:
            raise ProjectError(f"{path.name}: characters[{i}].describe_back: removed; "
                               f"give a describe map with one entry per direction instead")
        describe = c.get("describe") or ""
        if isinstance(describe, dict):
            for x in describe:
                if x not in generate:
                    raise ProjectError(f"{path.name}: characters[{i}].describe.{x}: not in directions.generate")
            for x in generate:
                if not describe.get(x):
                    raise ProjectError(f"{path.name}: characters[{i}].describe: no entry for direction {x!r}")
            describe = {x: str(v) for x, v in describe.items()}
        proj.characters.append(Character(name=name, still=still_p, describe=describe,
                                         pronoun=c["pronoun"], who=c.get("who") or DEFAULT_WHO[c["pronoun"]],
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
        canvas = int(a.get("canvas") or sprite.frame_size)
        if canvas < sprite.frame_size:
            raise ProjectError(f"{path.name}: animations[{i}].canvas: {canvas} is smaller than sprite.frame_size "
                               f"({sprite.frame_size})")
        if canvas % 16:
            raise ProjectError(f"{path.name}: animations[{i}].canvas: {canvas} is not a multiple of 16")
        # FR 4 amendment 3: every source GLB needs a mannequin profile (FR 14)
        # to anchor the stills it drives; the error prints the sha256 prefix and
        # the command that measures one
        if metrics.glb_sha256(src_p) not in profiles:
            raise ProjectError(f"{path.name}: animations[{i}].source: no mannequin profile "
                               f"for {src_p.name} (sha256 {metrics.glb_sha256(src_p)[:16]}…); "
                               f"run: sprite-pipeline mannequin-profile {src_p}")
        state = a.get("state") or "attack"
        if state not in VALID_STATES:
            raise ProjectError(f"{path.name}: animations[{i}].state: {state!r} not one of {sorted(VALID_STATES)}")
        wf = a.get("workflow")  # FR 3: per-animation override of workflows.animate
        if wf is not None:
            wf_p = Path(wf)
            if not wf_p.is_absolute():
                wf_p = path.parent / wf_p
        else:
            wf_p = None
        proj.animations.append(Animation(name=name, source=src_p, action=action, frames=frames,
                                         loop=bool(a.get("loop", False)), prompt=a.get("prompt") or "",
                                         start=start, clip_frames=clip,
                                         repeats=int(a.get("repeats") or 1), pre=int(a.get("pre", 1)),
                                         post=int(a.get("post") or 0), look_down=bool(a.get("look_down", False)),
                                         state=state, seam_max=float(a.get("seam_max") or 0),
                                         directions=list(anim_dirs), workflow=wf_p, canvas=canvas))

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
