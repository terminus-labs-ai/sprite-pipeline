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

from sprite_pipeline import config, project, stages, stills


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
    Raises ValueError (exit 2)."""
    proj = project.load_project(project_path)
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


def _print_cmd(prefix, argv, label, out):
    print(f"{prefix} ({label})  " + " ".join(stages.quote(a) for a in argv), file=out)


def _print_plan_line(char, stage, argv, out):
    """One planned command: the `stage=<stage>` field names the stage and the
    `(label)` field names the tool; the full argv follows after two spaces."""
    if argv[0] == "copy_file":  # in-process step, never a subprocess
        print(f"[{char}] stage={stage} (python copy)  {argv[1]} -> {argv[2]}", file=out)
        return
    print(f"[{char}] stage={stage} (cmd)  "
          + " ".join(stages.quote(a) for a in argv), file=out)


def _print_still_plan(proj, ctx, char, reroll, force, out):
    """FR 1 dry-run lines: the still preparation and one line per direction
    edit (skip marked; --reroll and --force plan the step again)."""
    cs = next(c for c in proj.characters if c.name == char)
    se = ctx.dir_still(char, "se")
    pre = "prepare" if (force or reroll.get("se") is not None or not se.is_file()) else "skip"
    print(f"[{char}] still preparation: {pre} {cs.still} -> {se} "
          f"(python; then sprite-gen cutout --key measured)", file=out)
    for d in proj.generate:
        if d == "se":
            continue  # the still's own direction keeps the prepared image
        turn = project.load_view(proj.view)["turn"][d]
        edit = "edit" if (force or reroll.get(d) is not None
                          or not ctx.dir_still(char, d).is_file()) else "skip"
        print(f"[{char}/{d}] direction edit ({edit}): {proj.workflows['edit']} on the "
              f"prepared still ({turn[:48]}...)", file=out)


def run_stills(ctx, turn, char, reroll, force, log, out):
    """FR 6 + FR 7: prepare the still, key the SE still for the palette, then
    one edit per generated direction other than se, each padded onto the
    reference canvas like the SE still (skip-if-exists; a reroll forces its
    direction). Returns False if a step failed."""
    p = ctx.project
    cs = next(c for c in p.characters if c.name == char)
    se = ctx.dir_still(char, "se")
    ok = True

    def forced(path):
        return force or not path.is_file()

    if forced(se):
        print(f"[{char}] still preparation: prepare {cs.still} -> {se}", file=out)
        stills.prepare(cs.still, se)
    if forced(ctx.master(char) / "se.keyed.png"):
        cmds = stages.plan_cutout(se, ctx.master(char))
        ok = run_cmds(ctx, cmds, log) and ok
    for d in p.generate:
        if d == "se":
            continue  # the still's own direction keeps the prepared image
        dst = ctx.dir_still(char, d)
        seed = reroll.get(d, p.seed)
        if dst.is_file() and reroll.get(d) is None and not force:
            continue  # skip-if-exists (FR 10)
        prompt = stills.dir_prompt(turn, d, cs.pronoun)
        # absolute, resolved at load (FR 1, amendment 1)
        argv = [sys.executable, "-m", "sprite_pipeline.comfy", "edit",
                "--workflow", str(p.workflows["edit"]), "--image", str(se),
                "--prompt", prompt, "--seed", str(seed),
                "--prefix", f"{char}/dir-{d}", "--copy-to", str(dst)]
        print(f"[{char}/{d}] direction edit: seed {seed}", file=out)
        if run_cmds(ctx, [(argv, "edit")], log):
            # same canvas and scale as the prepared SE still (padding in place)
            stills.prepare(dst, dst)
        else:
            ok = False
    return ok


def run_cmds(ctx, cmds, log):
    """Execute planned (argv, label) steps; copy_file runs in-process (AC8)."""
    ok = True
    for cmd_argv, label in cmds:
        if cmd_argv[0] == "copy_file":
            stages.copy_file(cmd_argv[1], cmd_argv[2], log)
            continue
        log.append("$ " + " ".join(stages.quote(x) for x in cmd_argv))
        r = subprocess.run(cmd_argv, env=ctx.env, capture_output=True, text=True)
        if r.stdout.strip():
            log.append(r.stdout.strip())
        if r.stderr.strip():
            log.append(r.stderr.strip())
        if r.returncode != 0:
            ok = False
            break
    return ok


def cmd_approve(args, out, interim=False):
    """`sprite-pipeline approve <project.yaml> <character>`: record the review
    (FR 8). refs.approved holds the hash of the direction stills; exit 2 with
    the missing stills named if any does not exist yet. `interim` writes the
    slice-2 marker (still filenames, one per line) instead of the hash: the
    hash gate then treats those stills as approved while their names and
    locations stand, and any later change invalidates it."""
    try:
        proj = project.load_project(args["project_file"])
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
    ctx.write_refs_approved(args["character"], interim=interim)
    print(f"[{args['character']}] approved {ctx.refs_approved(args['character'])}",
          file=out)
    return 0


def main(argv=None, out=None):
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
    # --interim writes the slice-2 marker (the still names, not the hash) so
    # approvals recorded by earlier-slice tooling keep working.
    p = sub.add_parser("approve", help="approve a character's direction stills")
    p.add_argument("--interim", action="store_true",
                   help="record the interim slice-2 approval (still names, not a hash)")
    p.add_argument("approve_args", nargs=argparse.REMAINDER)
    a = ap.parse_args(argv)
    args = vars(a)

    if a.command == "approve":
        rest = list(args["approve_args"])
        interim = args["interim"]
        if len(rest) != 2 or any(x.startswith("-") for x in rest):
            ap.error("usage: sprite-pipeline approve [--interim] <project.yaml> <character>")
        return cmd_approve({"project_file": rest[0], "character": rest[1]}, out,
                           interim=interim)

    force = args["force"]
    try:
        proj, cfg, chars, anim_names, dirs, rerolls = resolve_run(args["project_file"], args)
    except ValueError as e:
        print(f"project error: {e}", file=sys.stderr)
        return 2
    selected = stages.STAGES if not args["stage"] else [args["stage"]]
    pair_stages = [s for s in selected if s in stages.PAIR_STAGES]
    kit_stages = [s for s in selected if s in stages.KIT_STAGES]
    dry = args["dry_run"]
    ctx = stages.StageCtx(proj, cfg)
    log, status, failed, stopped = [], {}, [], []

    for char in chars:
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
                    ok = run_cmds(ctx, cmds, log)
                    if not ok:
                        row[s] = "FAIL"
                        failed.append(f"{n}/{d} {s}")
                        break
                    row[s] = "ok"
                status[(n, d)] = row
                print(f"[{n}/{d}] " + "  ".join(f"{s}={row[s]}" for s in pair_stages), file=out)

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
                bad = False
                for cmd_argv, label in ctx.plan_kit_stage(char, s, anim_names, dirs):
                    log.append("$ " + " ".join(stages.quote(x) for x in cmd_argv))
                    r = subprocess.run(cmd_argv, env=ctx.env, capture_output=True, text=True)
                    if r.returncode != 0:
                        bad = True
                        break
                if bad:
                    print(f"[{s}] FAIL", file=out)
                    failed.append(f"{char} {s}")
                else:
                    print(f"[{s}] ok", file=out)

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
