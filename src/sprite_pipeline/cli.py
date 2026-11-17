#!/usr/bin/env python3
"""`sprite-pipeline go <project.yaml>`: run the sprite job described by a project file.

    sprite-pipeline go <project.yaml> [--characters a,b] [--animations run,punch]
                       [--dirs s,se] [--stage drive|animate|key|loop|pixel|export]
                       [--reroll <char>:<dir>=<seed>] [--no-stop] [--force] [--dry-run]
    sprite-pipeline approve <project.yaml> <character>

Per character: prepare the still (pad, then key the SE still for the palette),
then run the edit workflow once per generated direction other than the still's
own, with the view's turn prompt in the character's pronoun (FR 6, FR 7); each
step is skip-if-exists unless --force or --reroll names it. `approve` writes
refs.approved (hash of the direction stills); the review stop (FR 8) keeps the
character's animation stages away until that hash matches, --no-stop skips the
stop for one run. Exit codes: 0 done or stopped for review, 1 a stage failed,
2 the project file is invalid.
"""
import argparse
from pathlib import Path
import shutil
import subprocess
import sys

from sprite_pipeline import config, metrics, project, stages, stills


def _narrow(values, allowed, flag):
    bad = [x for x in values if x not in allowed]
    if bad:
        raise ValueError(f"{flag}: unknown {', '.join(repr(x) for x in bad)} "
                         f"(valid: {', '.join(allowed)})")
    return values


def parse_rerolls(s):
    """--reroll <char>:<dir>=<seed>[,...]; {char: {dir: seed}}. A bare
    <dir>=<seed> entry applies to every character named by --characters (or
    all characters when --characters is absent)."""
    out = {}
    for part in filter(None, (s or "").split(",")):
        spec, sep, seed = part.rpartition("=")
        if not sep or not spec or not seed.lstrip("-").isdigit():
            raise ValueError(f"--reroll entry {part!r}: expected [<character>:]<dir>=<seed>")
        ch, colon, d = spec.partition(":")
        if not colon:
            ch, d = "", spec
        out.setdefault(ch, {})[d] = int(seed)
    return out


def rerolls_for(rerolls, chars):
    """Rerolls of one run's characters: {char: {dir: seed}}. A bare entry
    (no character) applies to every character in `chars`; an entry naming a
    character outside them simply matches nothing."""
    out = {}
    for ch, entries in rerolls.items():
        targets = chars if ch == "" else [ch]
        for t in targets:
            out.setdefault(t, {}).update(entries)
    return out


def resolve_run(project_path, args):
    """The project (paths relative to the file), the narrowed character,
    animation and direction sets, the rerolls and the host config.
    The profile gate (FR 4) reads the packaged mannequin profiles merged with
    the host file (FR 14, last-write-wins per hash). Raises ValueError (exit 2)."""
    proj = project.load_project(project_path, profiles=args.get("profiles"))
    if args["characters"]:
        # excluding every character is valid (nothing runs), only unknown *and*
        # unmatched names are errors
        known = {c.name for c in proj.characters}
        names = args["characters"].split(",")
        unknown = [x for x in names if x not in known]
        if unknown and not any(x in known for x in names):
            raise ValueError(f"--characters: unknown {', '.join(repr(x) for x in unknown)} "
                             f"(valid: {', '.join(sorted(known))})")
        chars = [x for x in names if x in known]
    else:
        chars = [c.name for c in proj.characters]
    if args["animations"]:
        names = _narrow(args["animations"].split(","), [a.name for a in proj.animations],
                        "--animations")
        anims = [a for a in proj.animations if a.name in names]
    else:
        anims = list(proj.animations)
    dirs = list(proj.generate)
    if args["dirs"]:
        dirs = _narrow(args["dirs"].split(","), dirs, "--dirs")
    # an animation's own direction subset still applies
    dirs = [d for d in dirs if any(d in a.directions for a in anims)]
    rerolls = parse_rerolls(args.get("reroll"))
    known = {c.name for c in proj.characters}
    for ch in sorted(rerolls):
        if ch and ch not in known:
            raise ValueError(f"--reroll: unknown character {ch!r} "
                             f"(valid: {', '.join(sorted(known))})")
        _narrow(list(rerolls[ch]), proj.generate, f"--reroll {ch}")
    rerolls = rerolls_for(rerolls, chars)
    return proj, config.load(), chars, [a.name for a in anims], dirs, rerolls


def _prepares_stills(args):
    """Whether this run does any still work at all (FR 6, FR 10): `--stage`
    over a pair or kit stage runs only that stage, and a character already
    approved continues past the still work (its skip-if-exists holds) unless
    --force names it again. Used for the FR 6 standing-figure gate and to
    scope the FR 10 stale-source purge: a run that does no still work
    neither measures a source image nor deletes a character's outputs."""
    if args["stage"] is not None:
        return False
    if args["no_stop"]:
        return args["force"]
    return True


def _print_plan_line(char, stage, argv, out):
    """One planned command: the `stage=<stage>` field names the stage and the
    `(label)` field names the tool; the full argv follows after two spaces."""
    if argv[0] == "copy_file":  # in-process step, never a subprocess
        print(f"[{char}] stage={stage} (python copy)  {argv[1]} -> {argv[2]}", file=out)
        return
    if argv[0] == "pad_ref":  # in-process step, never a subprocess
        print(f"[{char}] stage={stage} (python pad {argv[3]} on {argv[4]})  {argv[1]} -> {argv[2]}", file=out)
        return
    print(f"[{char}] stage={stage} (cmd)  "
          + " ".join(stages.quote(a) for a in argv), file=out)


def still_plan(proj, char):
    """The still work one character needs, in run order: the prepare step
    first (the anchored master — the still's own direction, `generate[0]`,
    FR 6 — and its keyed copy), then one direction edit per generated
    direction other than the master's — the still's own direction keeps the
    anchored image. The shared definition of that loop behind
    `_print_still_plan` (the dry-run plan) and `run_stills` (the real run),
    so the two can never disagree about which steps exist or their order."""
    cs = next(c for c in proj.characters if c.name == char)
    still_dir = proj.still_dir
    steps = [{"kind": "prepare", "char": char, "src": cs.still,
              "dst": None, "turn": "", "dir": still_dir}]
    turn = project.load_view(proj.view)["turn"]
    for d in proj.generate:
        if d == still_dir:
            continue
        steps.append({"kind": "edit", "char": char, "src": None,
                      "dst": None, "turn": turn[d], "dir": d})
    return steps


def _print_still_plan(proj, ctx, char, reroll, force, out):
    """FR 1 dry-run lines: the still anchoring and one line per direction
    edit (skip marked; --reroll and --force plan the step again)."""
    cs = next(c for c in proj.characters if c.name == char)
    still_dir = proj.still_dir
    master = ctx.dir_still(char, still_dir)
    for step in still_plan(proj, char):
        if step["kind"] == "prepare":
            pre = ("anchor" if (force or reroll.get(still_dir) is not None
                                or not master.is_file()) else "skip")
            print(f"[{char}] still preparation: {pre} {cs.still} -> {master} "
                  f"(python; then sprite-gen cutout --key measured)", file=out)
        else:
            d = step["dir"]
            edit = "edit" if (force or reroll.get(d) is not None
                              or not ctx.dir_still(char, d).is_file()) else "skip"
            print(f"[{char}/{d}] direction edit ({edit}): {proj.workflows['edit']} on the "
                  f"anchored master ({master.name}) ({step['turn'][:48]}...)", file=out)


def run_stills(ctx, turn, char, reroll, force, log, out):
    """FR 6 + FR 7: anchor the still to the mannequin profile onto the master
    (`master/<still_dir>.png`, the first generated direction), key the master
    for the palette, then one edit per generated direction other than the
    master's, each run on the anchored master (skip-if-exists; a reroll
    forces its direction). Walks `still_plan`, the shared loop behind the
    dry-run plan. Returns False if a step failed."""
    p = ctx.project
    cs = next(c for c in p.characters if c.name == char)
    # FR 6: the master is the anchored still: the edit input for every other
    # direction and the palette source (its keyed cutout is
    # `master/<still_dir>.keyed.png`). No direction ever receives the source
    # still under its own name: each stays its own edit's output.
    sd = p.still_dir
    master = ctx.dir_still(char, sd)
    keyed = ctx.master(char) / f"{sd}.keyed.png"
    ok = True
    # FR 14: the profile of the first-listed animation's GLB anchors the
    # stills: the packaged profiles merged with the host file
    profile = project.anchor_profile(p)

    def forced(path):
        return force or reroll.get(sd) is not None or not path.is_file()

    for step in still_plan(p, char):
        if step["kind"] == "prepare":
            if forced(master):
                print(f"[{char}] still preparation: anchor {cs.still} -> {master}",
                      file=out)
                stills.anchor(cs.still, master, p.sprite.frame_size, profile)
            if force or reroll.get(sd) is not None or not keyed.is_file():
                cmds = stages.plan_cutout(master, ctx.master(char))
                ok = run_cmds(ctx, cmds, log, out) and ok
            # FR 10 amendment 2: fingerprint the source still whenever the
            # stills are prepared (also refreshes a missing/legacy hash)
            ctx.write_source_hash(char, cs)
        else:
            d = step["dir"]
            dst = ctx.dir_still(char, d)
            seed = reroll.get(d, p.seed)
            if dst.is_file() and reroll.get(d) is None and not force:
                continue  # skip-if-exists (FR 10)
            prompt = stills.dir_prompt(turn, d, cs.pronoun)
            # the edit input is the anchored 832x1024 master itself (FR 6):
            # absolute, resolved at load (FR 1, amendment 1)
            argv = [sys.executable, "-m", "sprite_pipeline.comfy", "edit",
                    "--workflow", str(p.workflows["edit"]), "--image", str(master),
                    "--prompt", prompt, "--seed", str(seed),
                    "--prefix", f"{char}/dir-{d}", "--copy-to", str(dst)]
            print(f"[{char}/{d}] direction edit: seed {seed}", file=out)
            if run_cmds(ctx, [(argv, "edit")], log, out):
                # the edit ran on the 832x1024 master, so its output comes
                # back on that canvas and is already the direction's still:
                # nothing is copied a second time (skip-if-exists, FR 10,
                # and `se.png` is written once by the anchor step above)
                pass
            else:
                ok = False
    return ok


def run_cmds(ctx, cmds, log, out=None):
    """Execute planned (argv, label) steps; copy_file runs in-process (AC8).

    On non-zero exit, prints to *out*: the label, the quoted command, the
    exit code, and the last ~20 lines of stderr (stdout if stderr is empty).
    Successful steps stay quiet.  When *out* is None the failure info goes
    to stdout (sys.stdout).
    """
    ok = True
    for cmd_argv, label in cmds:
        if cmd_argv[0] == "copy_file":
            stages.copy_file(cmd_argv[1], cmd_argv[2], log)
            continue
        if cmd_argv[0] == "pad_ref":
            stages.pad_ref(*cmd_argv[1:], log)
            continue
        log.append("$ " + " ".join(stages.quote(x) for x in cmd_argv))
        r = subprocess.run(cmd_argv, env=ctx.env, capture_output=True, text=True)
        if r.stdout.strip():
            log.append(r.stdout.strip())
        if r.stderr.strip():
            log.append(r.stderr.strip())
        if r.returncode != 0:
            ok = False
            quoted = " ".join(stages.quote(x) for x in cmd_argv)
            print(f"[{label}] {quoted} exited {r.returncode}", file=out)
            tail_lines = (r.stderr or r.stdout).splitlines()
            tail = tail_lines[-20:] if len(tail_lines) > 20 else tail_lines
            for line in tail:
                print(f"  {line}", file=out)
            break
    return ok


def cmd_approve(args, out, **load_kwargs):
    """`sprite-pipeline approve <project.yaml> <character>`: record the review
    (FR 8). refs.approved holds the hash of the direction stills; exit 2 with
    the missing stills named if any does not exist yet."""
    try:
        proj = project.load_project(args["project_file"], profiles=load_kwargs.get("profiles"))
    except ValueError as e:
        print(f"project error: {e}", file=sys.stderr)
        return 2
    if args["character"] not in {c.name for c in proj.characters}:
        known = ", ".join(c.name for c in proj.characters)
        print(f"approve: unknown character {args['character']!r} (valid: {known})",
              file=sys.stderr)
        return 2
    ctx = stages.StageCtx(proj, config.load())
    missing = ctx.missing_stills(args["character"])
    if missing:
        print(f"approve: direction stills missing, run go first: "
              f"{', '.join(str(ctx.dir_still(args['character'], d)) for d in missing)}",
              file=sys.stderr)
        return 2
    ctx.write_refs_approved(args["character"])
    print(f"[{args['character']}] approved {ctx.refs_approved(args['character'])}",
          file=out)
    return 0


def render_mannequin(glb):
    """Render the GLB's `A_TPose` at the drive resolution 704 with the drive
    camera (facing `s`, elevation 45°, ortho 3.0 — the same camera and scene
    the drive plan uses). Returns the path of the rendered frame. Needs
    Blender and the host config (FR 14)."""
    from PIL import Image

    # a stand-in project for the drive plan: the drive resolution, the
    # drive-key's rig defaults, and Project.drive_key itself (the cache key
    # over the GLB bytes, action and drive params)
    from sprite_pipeline.project import Project
    ctx = stages.StageCtx(Project(path=Path.cwd(), style="", view="isometric",
                                  generate=["s"], mirror={},
                                  sprite=project.Sprite(scale=1.0, colors=40,
                                                        accent_boost=3, fps=30,
                                                        frame_size=metrics.DRIVE_RES),
                                  naming="{animation}_{dir}", export="godot",
                                  seed=0, pose_strength=1.0, output=Path("out"),
                                  rig={"neck": "DEF-neck", "head": "DEF-head"},
                                  workflows={}, characters=[], animations=[]),
                          config.load())
    anim = project.Animation(name="A_TPose", source=Path(glb), action="A_TPose",
                             frames=1, loop=False, prompt="")
    # the plan's own frames dir: where the blender step writes, and where the
    # measurement reads back. The plan's copy_file step (the mp4 upload to
    # ComfyUI's input/) is dropped here — the render keeps its own frames and
    # there is no upload. When no frame lands, the glob below finds nothing
    # and this raises SystemExit, never a traceback.
    frames = ctx.frames_dir("mannequin", anim, "s")
    cmds = [(c, k) for c, k in ctx.plan_pair("mannequin", anim, "s", "drive")
            if k != "copy-drive"]
    log = []
    if not run_cmds(ctx, cmds, log):
        raise SystemExit(f"mannequin-profile: rendering {glb} failed")
    shots = sorted(frames.glob("f*.png"))
    if not shots:
        # the fake renderer wrote nothing where the plan pointed: nothing to
        # measure, so the render failed
        raise SystemExit(f"mannequin-profile: no render of {glb} landed in {frames}")
    # a drive is many frames; the profile's pose is the first held frame
    img = Image.open(shots[0])
    shot = frames / "A_TPose.png"
    img.save(shot)
    return shot


def cmd_mannequin_profile(args, out):
    """`sprite-pipeline mannequin-profile <glb>` (FR 14, amendment 4): render
    the GLB's A_TPose at 704 with the drive camera, measure it and append the
    entry to the host profile file, replacing an existing entry for the same
    hash. Prints the measured JSON."""
    import json

    glb = Path(args["glb"])
    if not glb.is_file():
        print(f"mannequin-profile: {glb} not found", file=sys.stderr)
        return 2
    shot = render_mannequin(glb)
    m = metrics.measure(shot)
    entry = {"glb_name": glb.name, "resolution": metrics.DRIVE_RES,
             "body_h": m["body_h"], "floor_y": m["floor_y"]}
    digest = metrics.glb_sha256(glb)
    path = project.user_profiles_path()
    try:
        doc = __import__("json").loads(path.read_text())
        if not isinstance(doc, dict):
            doc = {}
    except (OSError, ValueError):
        doc = {}
    profiles = doc.get("profiles")
    if not isinstance(profiles, dict):
        profiles = {}
    profiles[digest] = entry  # last write wins: an existing same-hash entry is replaced
    doc["profiles"] = profiles
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc, indent=2) + "\n")
    print(json.dumps(entry), file=out)
    return 0


def main(argv=None, out=None, **load_kwargs):
    """Run one `sprite-pipeline` command. `load_kwargs.profiles` passes through
    to `project.load_project`: the profile map the gate (FR 4) reads, when a
    caller supplies one instead of the packaged file."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                 prog="sprite-pipeline")
    sub = ap.add_subparsers(dest="command", required=True)

    g = sub.add_parser("go", help="run the sprite job of a project file")
    g.add_argument("project_file", help="path to project.yaml")
    g.add_argument("--characters", help="comma-separated character names")
    g.add_argument("--animations", help="comma-separated animation names")
    g.add_argument("--dirs", help="comma-separated directions")
    g.add_argument("--stage", choices=stages.STAGES, help="run only this stage")
    g.add_argument("--reroll", help="redo one still: <character>:<dir>=<seed> "
                                    "[;<character>:<dir>=<seed>...]")
    g.add_argument("--no-stop", action="store_true", dest="no_stop",
                   help="skip the review stop for this run")
    g.add_argument("--force", action="store_true", help="rerun stages whose output exists")
    g.add_argument("--dry-run", action="store_true", dest="dry_run",
                   help="print the plan, run nothing")
    # `approve <project.yaml> <character>` takes no options: REMAINDER keeps
    # every word after the command name verbatim in approve_args.
    p = sub.add_parser("approve", help="approve a character's direction stills")
    p.add_argument("approve_args", nargs=argparse.REMAINDER)
    mp = sub.add_parser("mannequin-profile",
                        help="render + measure a GLB's A_TPose at 704 and append "
                             "its mannequin profile to the host profile file")
    mp.add_argument("glb", help="path to the source .glb")
    a = ap.parse_args(argv)
    args = vars(a)

    if a.command == "approve":
        rest = list(args["approve_args"])
        if len(rest) != 2 or any(x.startswith("-") for x in rest):
            ap.error("usage: sprite-pipeline approve <project.yaml> <character>")
        return cmd_approve({"project_file": rest[0], "character": rest[1]}, out, **load_kwargs)

    if a.command == "mannequin-profile":
        try:
            return cmd_mannequin_profile(args, out)
        except SystemExit:
            raise
        except (ValueError, OSError) as e:
            print(f"mannequin-profile: {e}", file=sys.stderr)
            return 2

    return _go(args, out, load_kwargs)


def _go(args, out, load_kwargs=None):
    """The `go` command (sprite-pipeline-32's scope): the plan, the FR 6
    standing-figure and FR 4 profile gates, the still work and the stages."""
    if load_kwargs is None:
        load_kwargs = {}
    out = out or sys.stdout

    force = args["force"]
    args["profiles"] = load_kwargs.get("profiles")
    try:
        proj, cfg, chars, anim_names, dirs, rerolls = resolve_run(args["project_file"], args)
        # FR 6: a character's still must be an A- or T-pose standing figure —
        # checked before any output is touched, so a bad still exits 2 naming
        # the file and the failed rule. Only when this run prepares stills: a
        # narrowed run (--stage, or --no-stop over approved stills) never
        # touches the source image.
        if not args["dry_run"] and _prepares_stills(args):
            # skip-if-exists (FR 10): a character whose direction stills all
            # exist does no still work this run, so its source image is not
            # measured — only the characters that would prepare stills are
            check_ctx = stages.StageCtx(proj, cfg)
            for name in chars:
                if not check_ctx.missing_stills(name):
                    continue
                cs = next(c for c in proj.characters if c.name == name)
                # FigureError (a failed rule, naming the file) is a ValueError;
                # any other read error (not an image at all) is reported as
                # the same exit-2 project error, never a traceback
                try:
                    stills.check(cs.still)
                except ValueError:
                    raise
                except Exception as e:
                    raise ValueError(f"{cs.still}: {e.__class__.__name__}: "
                                     f"{e}") from e
    except (ValueError, metrics.FigureError) as e:
        print(f"project error: {e}", file=sys.stderr)
        return 2
    selected = stages.STAGES if not args["stage"] else [args["stage"]]
    pair_stages = [s for s in selected if s in stages.PAIR_STAGES]
    kit_stages = [s for s in selected if s in stages.KIT_STAGES]
    dry = args["dry_run"]
    prepares = _prepares_stills(args)
    ctx = stages.StageCtx(proj, cfg)
    log, status, failed, stopped = [], {}, [], []

    for char in chars:
        cs = next(c for c in proj.characters if c.name == char)
        # FR 10 amendment 2: a stale source still (changed bytes or repointed
        # path) invalidates the character's whole output tree, including its
        # approval and every derived clip, loop and export.
        char_out = ctx.project.output_root() / char
        if char_out.is_dir() and ctx.source_stale(char, cs):
            # FR 10: the character is stale whenever its source bytes changed.
            # A run that does still work (no --stage) purges the outputs —
            # a --dry-run plans the purge and deletes nothing; a narrowed run
            # (--stage) never touches another run's outputs or prints, and
            # simply rebuilds what that stage needs from the new source.
            if prepares:
                print(f"[{char}] stale source still: {cs.still}", file=out)
                if not dry:
                    ctx.drop_char_outputs(char)
                else:
                    print(f"[{char}] stale: rebuild planned, no deletion "
                          f"(--dry-run)", file=out)
        elif prepares and char_out.is_dir() and ctx.recorded_source_hash(char) is None:
            # a legacy output directory carries no usable fingerprint (no file,
            # unreadable, or blank): record the current source's hash and keep
            # the outputs (unknown, not stale)
            ctx.write_source_hash(char, cs)
            print(f"[{char}] no source fingerprint: wrote {ctx.source_hash_file(char).name} "
                  f"for {cs.still}", file=out)
        reroll = rerolls.get(char, {})
        # a reroll makes its direction's still again, so the review stop and
        # the still-prep skips treat the character as unapproved this run
        reroll = rerolls.get(char, {})
        rerolled = bool(reroll)  # a reroll redoes its still and stops for review
        five = not ctx.missing_stills(char)  # every direction still exists
        # The review stop (FR 8) precedes the animation stages: once the
        # stills exist, go writes refs-sheet.png and stops until approve
        # recorded the stills' hash; --reroll stops the run again, --no-stop
        # skips the stop for one run (nothing is approved by it).
        if dry:
            # --dry-run writes nothing: the FR 1 plan lines plus the verdict
            _print_still_plan(proj, ctx, char, reroll, force, out)
            if not args["no_stop"] and (not ctx.is_approved(char) or rerolled):
                stopped.append(char)
                missing = ctx.missing_stills(char)
                print(f"[{char}] review stop: unapproved"
                      f"{': stills missing ' + ', '.join(missing) if missing else ''}; "
                      f"after checking {ctx.refs_sheet(char)} run: "
                      f"sprite-pipeline approve {args['project_file']} {char}", file=out)
                continue
        elif not (args["no_stop"] and ctx.is_approved(char) and not rerolled):
            # FR 6 + FR 7 + FR 8: still preparation, palette cutout, direction
            # edits, then the review sheet and the stop verdict.
            stop_now = not ctx.is_approved(char) or rerolled
            # --force alone over an approved complete set re-runs the chosen
            # stage's own commands, never the reviewed stills; redoing a still
            # would invalidate the approval it passed (a reroll does redo it)
            redo_stills = not five or not ctx.is_approved(char) or rerolled
            if redo_stills:
                if not run_stills(ctx, project.load_view(proj.view)["turn"], char,
                                  reroll, force, log, out):
                    failed.append(f"{char} stills")
                    continue
                # FR 8 review stop: the sheet is the review artefact
                sheet = ctx.refs_sheet(char)
                missing = ctx.missing_stills(char)
                stills.review_sheet([ctx.dir_still(char, d) for d in proj.generate
                                     if ctx.dir_still(char, d).is_file()], sheet)
                print(f"[{char}] review sheet {sheet}", file=out)
            if stop_now and not args["no_stop"]:
                stopped.append(char)
                missing = ctx.missing_stills(char)
                if missing:
                    print(f"[{char}] unapproved: direction stills missing "
                          f"{', '.join(missing)}; rerun go to make them", file=out)
                else:
                    print(f"[{char}] unapproved: check the sheet, then run: "
                          f"sprite-pipeline approve {args['project_file']} {char}",
                          file=out)
                    if rerolled:
                        # --reroll redid a still: even while refs.approved
                        # still matches (same bytes), this run ends at the
                        # review so the sheet is looked at again
                        print(f"[{char}] reroll: review {ctx.refs_sheet(char)} again",
                              file=out)
                continue  # stop: no animation stages this run

        for d in dirs:
            for n in anim_names:
                anim = proj.animation(n)
                if d not in anim.directions:
                    continue
                row = {}
                for s in pair_stages:
                    if ctx.stage_done(s, char, anim, d) and not args["force"]:
                        row[s] = "skip"
                        continue
                    cmds = ctx.plan_pair(char, anim, d, s)
                    if dry:
                        for cmd_argv, label in cmds:
                            _print_plan_line(f"{n}/{d}", s, cmd_argv, out)
                        row[s] = "dry"
                        continue
                    if s == "key":
                        shutil.rmtree(ctx.work_dir(char, anim, d), ignore_errors=True)
                        shutil.rmtree(ctx.loops_dir(char, anim, d), ignore_errors=True)
                    if s == "drive":
                        shutil.rmtree(ctx.frames_dir(char, anim, d), ignore_errors=True)
                        ctx.paths.motion_cache.mkdir(parents=True, exist_ok=True)
                        ctx.paths.comfy_input.mkdir(parents=True, exist_ok=True)
                    ok = run_cmds(ctx, cmds, log, out)
                    if not ok:
                        row[s] = "FAIL"
                        failed.append(f"{n}/{d} {s}")
                        break
                    row[s] = "ok"
                status[(n, d)] = row
                print(f"[{n}/{d}] " + "  ".join(f"{s}={row.get(s, '-')}" for s in pair_stages), file=out)

        for s in kit_stages:
            done = ctx.pixel_done(char, anim_names, dirs) if s == "pixel" else ctx.export_done(char)
            if done and not args["force"]:
                print(f"[{char}] [{s}] skip", file=out)
                continue
            if dry:
                for cmd_argv, label in ctx.plan_kit_stage(char, s, anim_names, dirs):
                    _print_plan_line(char, s, cmd_argv, out)
                if s == "export":
                    print(f"[{char}] stage=export (python)  mirrors sw<-se w<-e nw<-ne; "
                          f"atlas.png + sprite_frames.tres + contact-sheet.png", file=out)
                continue
            if s == "export":
                ok = ctx.run_program("export", char, anim_names, dirs, log, dry=False)
                print(f"[export] {'ok' if ok else 'incomplete'}"
                      + (f": {log[-1]}" if log else ""), file=out)
                if not ok:
                    failed.append(f"{char} export")
            else:
                shutil.rmtree(ctx.sprite_out(char), ignore_errors=True)
                ok = True
                for cmd_argv, label in ctx.plan_kit_stage(char, s, anim_names, dirs):
                    if not run_cmds(ctx, [(cmd_argv, label)], log, out):
                        ok = False
                        break
                if ok:
                    print(f"[{s}] ok", file=out)
                else:
                    failed.append(f"{char} {s}")

    print("\nsummary", file=out)
    if pair_stages:
        print("move/dir".ljust(14) + "".join(s.ljust(9) for s in pair_stages), file=out)
        for (n, d), row in status.items():
            print(f"{n}/{d}".ljust(14) + "".join(row.get(s, "-").ljust(9) for s in pair_stages),
                  file=out)
    if failed:
        print(f"\nFAILED: {len(failed)}: {', '.join(failed)}", file=out)
        return 1
    for char in stopped:
        # the review stop exits 0: the sheet path and the approve command end the run
        print(f"[{char}] stopped for review: sprite-pipeline approve "
              f"{args['project_file']} {char}", file=out)
    print("dry run complete" if dry else "all stages ok", file=out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
