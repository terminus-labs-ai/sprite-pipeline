#!/usr/bin/env python3
"""`sprite-pipeline go <project.yaml>`: run the sprite job described by a project file.

    sprite-pipeline go <project.yaml> [--characters a,b] [--animations run,punch]
                       [--dirs s,se] [--stage drive|animate|key|loop|pixel|export]
                       [--force] [--dry-run]

Per character: the still preparation and the direction edits print as plan
lines this slice (slices 4-5 implement them); a character counts as approved
when its direction stills exist. Per approved character, animation and
generated direction: drive -> animate -> key -> loop, then per character
pixel -> export. Stages whose output exists are skipped unless --force.
Exit codes: 0 done, 1 a stage failed, 2 the project file is invalid.
"""
import argparse
import shutil
import subprocess
import sys

from sprite_pipeline import config, project, stages


def _narrow(values, allowed, flag):
    bad = [x for x in values if x not in allowed]
    if bad:
        raise ValueError(f"{flag}: unknown {', '.join(repr(x) for x in bad)} "
                         f"(valid: {', '.join(allowed)})")
    return values


def resolve_run(project_path, args):
    """The project (paths relative to the file), the narrowed character,
    animation and direction sets, and the host config. Raises ValueError (exit 2)."""
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
    return proj, config.load(), chars, [a.name for a in anims], dirs


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
    g.add_argument("--force", action="store_true", help="rerun stages whose output exists")
    g.add_argument("--dry-run", action="store_true", dest="dry_run",
                   help="print the plan, run nothing")
    a = ap.parse_args(argv)
    args = vars(a)

    try:
        proj, cfg, chars, anim_names, dirs = resolve_run(args["project_file"], args)
    except ValueError as e:
        print(f"project error: {e}", file=sys.stderr)
        return 2
    selected = stages.STAGES if not args["stage"] else [args["stage"]]
    pair_stages = [s for s in selected if s in stages.PAIR_STAGES]
    kit_stages = [s for s in selected if s in stages.KIT_STAGES]
    dry = args["dry_run"]
    ctx = stages.StageCtx(proj, cfg)
    log, status, failed = [], {}, []

    for char in chars:
        # still preparation and the direction edits: plan lines only this slice,
        # implemented by slices 4-5 (workflow binding, padding, review stop)
        still = next(c for c in proj.characters if c.name == char).still
        print(f"[{char}] still preparation: prepare {still} -> "
              f"{ctx.dir_still(char, 'se')} (python; then sprite-gen cutout --key measured)",
              file=out)
        for d in proj.generate:
            if d == "se":
                continue  # the still's own direction keeps the prepared image
            turn = project.load_view(proj.view)["turn"][d]
            print(f"[{char}/{d}] direction edit: {proj.workflows['edit']} on the "
                  f"prepared still ({turn[:48]}...)", file=out)
        if not ctx.is_approved(char):
            print(f"[{char}] unapproved: direction stills missing, no animation "
                  f"stages this run", file=out)
            continue

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
    print("dry run complete" if dry else "all stages ok", file=out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
