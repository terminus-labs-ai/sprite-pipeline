"""Hygiene bead sprite-pipeline-10: the invariants behind F1a/F1c/F1d, F2,
F4, F5 and F6, one test class per case-table row.

- r1 (F1a/F1c): stages.py has no `import json`, cli.py no `_print_cmd`.
- r2 (F1d): the drive elevation 45 is defined once (project.DRIVE_ELEV,
  re-exported by stages), the drive plan passes 45 through, and the drive
  cache key for the example project equals the base's literal-payload key,
  pinned per animation for the exact GLB stub bytes used here.
- r3 (F4): `do_export(dirs=subset)` narrows the export to the subset and
  leaves the caller's project object untouched, before, during and after
  the call; the existing do_export tests in test_stages.py stay unmodified.
- r4 (F5): `comfy.post(timeout=...)` bounds the poll wait with a
  SystemExit naming the prompt and the bound; the default (None) keeps
  today's unbounded call path.
- r5 (F2): the pyproject dependency list is uniformly 4-space indented
  and parses to the same data as before.
- r6 (F6): `cli.still_plan` is the one direction loop behind the dry-run
  plan and the real run; the real run walks it in the base's order. The
  byte-identity of the dry-run and real-run output against the base is
  verified out-of-band for the handoff (a shared helper cannot be loaded
  from a base checkout alongside the head one).
- r7 (F6): `approve --interim` is rejected as an unknown flag; a
  pre-existing interim refs.approved file still reads as approved.

No network, no GPU, no Blender, no ComfyUI: the comfy server is a local
fake, subprocesses are faked at subprocess.run, the config home is
isolated per test.
"""
import ast
import hashlib
import http.server
import inspect
import io
import json
import os
import re
import shutil
import subprocess
import tempfile
import threading
import tomllib
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

from sprite_pipeline import cli, comfy, config, project, stages

ROOT = Path(__file__).resolve().parent.parent
EXAMPLE = ROOT / "examples" / "isometric"

# 2x2 RGBA PNG, every pixel (2, 3, 5, 255): the stub still/GLB byte source
TINY_PNG = bytes.fromhex(
    "89504e470d0a1a0a0000000d494844520000000200000002080600000072b60d"
    "240000001149444154789c63606266fd0fc20c3006001f640425fc82658c0000"
    "000049454e44ae426082")


def run_cli(argv, fake=None):
    """Invoke cli.main capturing stdout and stderr; an argparse SystemExit
    becomes its exit code and its message joins the captured output."""
    if fake is not None:
        subprocess.run = fake
    buf, err = io.StringIO(), io.StringIO()
    try:
        with redirect_stdout(buf), redirect_stderr(err):
            code = cli.main(argv)
    except SystemExit as e:
        code = e.code if isinstance(e.code, int) else 1
        buf.write(str(e) + "\n")
    buf.write(err.getvalue())
    return code, buf.getvalue()


def fixture(tmp):
    """A stubbed working copy of examples/isometric (stub still and GLB)."""
    base = Path(tmp) / "isometric"
    shutil.copytree(EXAMPLE, base)
    for rel in ("stills/courier.png", "anims/ual.glb"):
        f = base / rel
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_bytes(TINY_PNG)
    return base


# ------------------------------------------------------------- r1: dead code gone

class TestDeadCodeRemoved(unittest.TestCase):
    """F1a/F1c, static: the removed constructs do not come back."""

    def test_stages_has_no_import_json(self):
        self.assertNotIn("import json",
                         (ROOT / "src" / "sprite_pipeline" / "stages.py").read_text())

    def test_cli_has_no_print_cmd(self):
        text = (ROOT / "src" / "sprite_pipeline" / "cli.py").read_text()
        self.assertNotIn("_print_cmd", text)
        # the plan printer that stays is a different function
        self.assertIn("_print_plan_line", text)


# ----------------------------------------------------------- r2: one elevation

class TestOneElevationDefinition(unittest.TestCase):
    """F1d: 45 has one definition, referenced from both sites; the drive cache
    key for the example project is byte-identical to the base's payload."""

    def test_stages_reexports_the_project_definition(self):
        self.assertIs(stages.DRIVE_ELEV, project.DRIVE_ELEV)
        self.assertEqual(project.DRIVE_ELEV, 45)
        # the single definition lives in project.py (it imports no sibling
        # module, so both sites can reach it without a cycle); stages may
        # only re-export the name, never rebind a value
        def assignments(source):
            return [n for n in ast.parse(source).body
                    if isinstance(n, ast.Assign)
                    for t in n.targets
                    if getattr(t, "id", None) == "DRIVE_ELEV"]

        st = assignments((ROOT / "src" / "sprite_pipeline" / "stages.py").read_text())
        self.assertEqual(len(st), 1)
        self.assertIsInstance(st[0].value, ast.Attribute)  # project.DRIVE_ELEV
        pj = assignments((ROOT / "src" / "sprite_pipeline" / "project.py").read_text())
        self.assertEqual(len(pj), 1)
        self.assertEqual(pj[0].value.value, 45)

    def test_drive_key_hashes_the_shared_definition_not_a_second_literal(self):
        pj = (ROOT / "src" / "sprite_pipeline" / "project.py").read_text()
        self.assertNotIn('"elev": 45', pj)
        self.assertIn('"elev": DRIVE_ELEV', pj)

    def test_plan_pair_passes_elevation_45(self):
        with tempfile.TemporaryDirectory() as td:
            home = Path(td) / "home"
            base = fixture(td)
            p = project.load_project(base / "project.yaml")
            ctx = stages.StageCtx(p, config.defaults(home))
            argv = ctx.plan_pair("courier", p.animation("run"), "s", "drive")[0][0]
            # argv after `--`: glb action out_dir facing elev repeats pre post;
            # facing is 0 for s — pin the elev slot byte-exactly
            tail = argv[argv.index("--") + 1:]
            self.assertEqual(tail[3], "0")
            self.assertEqual(tail[4], "45")

    def test_drive_key_identical_to_the_base_payload(self):
        """Base's drive_key hashed a payload with the literal "elev": 45; the
        pinned keys below were taken at the base commit for the example
        project with these exact GLB stub bytes. Rebuilding the same payload
        by hand with 45, and calling drive_key at head, must both reproduce
        them."""
        with tempfile.TemporaryDirectory() as td:
            base = fixture(td)
            p = project.load_project(base / "project.yaml")
            glb = p.animations[0].source.read_bytes()

            def base_key(proj, anim, glb_bytes):
                payload = json.dumps(
                    {"glb": hashlib.sha256(glb_bytes).hexdigest(),
                     "action": anim.action, "repeats": anim.repeats,
                     "pre": anim.pre, "post": anim.post,
                     "head_lift": anim.head_lift,
                     "rig": {"neck": proj.rig["neck"], "head": proj.rig["head"]},
                     "elev": 45, "frame_size": proj.sprite.frame_size,
                     **({"canvas": anim.canvas} if anim.canvas
                        and anim.canvas != proj.sprite.frame_size else {})},
                    sort_keys=True)
                return hashlib.sha256(payload.encode()).hexdigest()[:16]

            pinned = {  # drive_key per animation, captured at the base commit
                "run": "d10a9fa3b7c2ca22", "punch": "af493d1482ff68b6",
                "idle": "1e10e8d03b0fe447", "walk": "0e2367000fd45664",
                "sneak": "d0401da6c31b3ef8", "crouch_idle": "c94ac4321f6739f3",
                "hack": "74b25e3d1e6a612f", "interact": "6cf5dbdfdfb1beaa",
                "jab": "be3f13aa8e524d9f", "pistol_shoot": "f820df3ea51e1330",
                "hit": "4d8dfb5d489a1306", "death": "1329fba138e6fccc",
                "roll": "41d262d0423e92ee",
            }
            self.assertEqual(set(pinned), {a.name for a in p.animations})
            for anim in p.animations:
                self.assertEqual(p.drive_key(anim, glb), pinned[anim.name],
                                 anim.name)
                self.assertEqual(p.drive_key(anim, glb),
                                 base_key(p, anim, glb), anim.name)


# ------------------------------------------------------- r3 (F4) and r5 (F2)

class TestDoExportNoMutation(unittest.TestCase):
    """F4: do_export honours `dirs` and never touches the caller's project."""

    def test_narrows_to_subset_and_leaves_the_project_unchanged(self):
        from PIL import Image
        from sprite_pipeline import export_godot

        with tempfile.TemporaryDirectory() as td:
            base = fixture(td)
            p = project.load_project(base / "project.yaml")
            ctx = stages.StageCtx(p, config.defaults(Path(td) / "home"))
            out = ctx.sprite_out("courier")
            out.mkdir(parents=True)
            # frames only for the subset dirs s and se
            for d in ("s", "se"):
                Image.new("RGBA", (4, 4), (255, 0, 0, 255)).save(out / f"{d}-run_000.png")
            moves = [p.animation("run")]
            subset = ["s", "se"]

            generate_before, mirror_before = p.generate, p.mirror
            real_export_char = export_godot.export_char

            def watching(ctx_, char, anim_names, log, dry):
                # observe the ctx do_export chose for the run, while it runs
                watching.seen_project = ctx_.project
                watching.identical_ctx = ctx_ is ctx
                watching.gen_shared_during = ctx_.project.generate is generate_before
                watching.mirror_shared_during = ctx_.project.mirror is mirror_before
                watching.gen_during = list(ctx_.project.generate)
                watching.mirror_during = dict(ctx_.project.mirror)
                return real_export_char(ctx_, char, anim_names, log, dry)

            export_godot.export_char = watching
            try:
                log = []
                ok = export_godot.do_export(ctx, "courier", moves, subset, log, dry=False)
            finally:
                export_godot.export_char = real_export_char

            self.assertTrue(ok, log)
            # the export was narrowed to the subset: se present, e (in the
            # project's full generate) not projected
            tres = (out / "sprite_frames.tres").read_text()
            self.assertIn('"name": &"run_s"', tres)
            self.assertIn('"name": &"run_se"', tres)
            self.assertIn('"name": &"run_sw"', tres)  # mirrors se, a subset member
            self.assertNotIn('"name": &"run_e"', tres)
            self.assertNotIn('"name": &"run_n"', tres)
            # the narrowing happened, on a private view: during the call the
            # ctx and project handed to export_char are not the caller's,
            # and the view carries exactly the subset...
            self.assertIsNot(watching.seen_project, p)
            self.assertFalse(watching.identical_ctx)
            self.assertEqual(watching.gen_during, subset)
            self.assertEqual(watching.mirror_during, {"sw": "se"})
            # ...while the caller's project object never held narrowed lists:
            # the very list and dict it held before the call are what it
            # holds now, unchanged, and they were never the exported view's
            self.assertFalse(watching.gen_shared_during)
            self.assertFalse(watching.mirror_shared_during)
            self.assertIs(p.generate, generate_before)
            self.assertIs(p.mirror, mirror_before)
            self.assertEqual(p.generate, ["s", "se", "e", "ne", "n"])
            self.assertEqual(p.mirror, {"sw": "se", "w": "e", "nw": "ne"})
            self.assertIs(p, ctx.project)

    def test_full_dirs_export_matches_the_project(self):
        """Honours `dirs` in the other direction too: passing everything is
        the plain project export."""
        from PIL import Image
        from sprite_pipeline import export_godot

        with tempfile.TemporaryDirectory() as td:
            base = fixture(td)
            p = project.load_project(base / "project.yaml")
            ctx = stages.StageCtx(p, config.defaults(Path(td) / "home"))
            out = ctx.sprite_out("courier")
            out.mkdir(parents=True)
            for d in p.generate:
                Image.new("RGBA", (4, 4), (255, 0, 0, 255)).save(out / f"{d}-run_000.png")
            moves = [p.animation("run")]
            log = []
            ok = export_godot.do_export(ctx, "courier", moves, list(p.generate), log, dry=False)
            self.assertTrue(ok, log)
            tres = (out / "sprite_frames.tres").read_text()
            for d in p.generate:
                self.assertIn(f'"name": &"run_{d}"', tres)


class TestPyprojectIndent(unittest.TestCase):
    """F2/r5: the dependency list is uniformly 4-space indented."""

    def test_dependency_list_four_space(self):
        text = (ROOT / "pyproject.toml").read_text()
        lines = text.splitlines()
        # the *project* dependency list: the one holding the sprite-gen pin
        start = next(i for i, line in enumerate(lines)
                     if line.startswith("dependencies = [")
                     and "sprite-gen" in "".join(lines[i:i + 6]))
        body = []
        for line in lines[start + 1:]:
            if line == "]":
                break
            body.append(line)
        self.assertTrue(body)
        for line in body:
            self.assertTrue(re.fullmatch(r'    "[^"]+",', line),
                            f"not 4-space indented: {line!r}")

    def test_still_parses_with_the_same_dependencies(self):
        # indent-only: the parsed data is unchanged
        d = tomllib.loads((ROOT / "pyproject.toml").read_text())
        self.assertEqual(d["project"]["dependencies"],
                         ["pillow>=11", "pyyaml>=6", "sprite-gen"])


# --------------------------------------------------------------------- r4: F5

class NeverReturnsHandler(http.server.BaseHTTPRequestHandler):
    """A fake ComfyUI: /prompt accepts the graph, /history never has a result."""

    def log_message(self, *args):
        pass

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        self.rfile.read(length)
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps({"prompt_id": "sp-test-pid"}).encode())

    def do_GET(self):
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(b"{}")  # the result never lands


class LandsEventuallyHandler(http.server.BaseHTTPRequestHandler):
    """A fake ComfyUI whose /history result lands only from the 6th poll."""

    def log_message(self, *args):
        pass

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        self.rfile.read(length)
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps({"prompt_id": "sp-test-pid"}).encode())

    def do_GET(self):
        self.server.polls["n"] += 1
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        if self.server.polls["n"] >= 6:  # finally lands, past any deadline
            body = {"sp-test-pid": {"status": {"status_str": "success"},
                                    "outputs": {"9": {"images": [
                                        {"filename": "x_00001.png",
                                         "subfolder": "", "type": "output"}]}}}}
            self.wfile.write(json.dumps(body).encode())
        else:
            self.wfile.write(b"{}")


class TestPostTimeout(unittest.TestCase):
    """F5/r4: post() gains a bounded wait; the default path is unchanged."""

    def _serve(self, handler):
        server = http.server.HTTPServer(("127.0.0.1", 0), handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        self.addCleanup(server.shutdown)
        return f"http://127.0.0.1:{server.server_address[1]}", server

    def test_timeout_bound_raises_system_exit_naming_prompt_and_bound(self):
        api, _ = self._serve(NeverReturnsHandler)
        with self.assertRaises(SystemExit) as cm:
            comfy.post(api, {}, "/nonexistent", poll_interval=0.01, timeout=0.05)
        msg = str(cm.exception)
        self.assertIn("sp-test-pid", msg)  # names the prompt that never landed
        self.assertIn("0.05", msg)  # and the bound that was exceeded

    def test_default_parameter_polls_past_a_deadline_the_bound_would_cross(self):
        """The unchanged default path: a fake clock that leaps 10_000 s after
        the third poll (any bound would have fired) and a result that lands
        only on the 6th poll must still end in a successful return when no
        timeout is given — the loop stops on the result, as at base."""
        self.assertIsNone(inspect.signature(comfy.post).parameters["timeout"].default)
        api, srv = self._serve(LandsEventuallyHandler)
        srv.polls = {"n": 0}

        ticks = iter([0.0, 0.0, 0.0] + [10_000.0] * 1000)
        real_monotonic, real_sleep = comfy.time.monotonic, comfy.time.sleep
        comfy.time.monotonic = lambda: next(ticks)
        comfy.time.sleep = lambda s: None
        try:
            files = comfy.post(api, {}, "/out")  # default: no bound
        finally:
            comfy.time.monotonic, comfy.time.sleep = real_monotonic, real_sleep
        self.assertEqual(files, [Path("/out") / "x_00001.png"])
        self.assertGreaterEqual(srv.polls["n"], 6)  # polled past the fake leap

    def test_timeout_default_is_none_in_the_signature(self):
        self.assertIsNone(inspect.signature(comfy.post).parameters["timeout"].default)


# ---------------------------------------------------------------------- r6: F6

class TestSharedStillPlan(unittest.TestCase):
    """F6/r6: one helper backs the dry-run plan and the real run."""

    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmpdir.cleanup)
        self.tmp = Path(self._tmpdir.name)
        self._old_home = os.environ.get("SPRITE_PIPELINE_HOME")
        os.environ["SPRITE_PIPELINE_HOME"] = str(self.tmp / "home")
        self.addCleanup(self._restore_home)
        self._old_run = subprocess.run

    def _restore_home(self):
        if self._old_home is None:
            os.environ.pop("SPRITE_PIPELINE_HOME", None)
        else:
            os.environ["SPRITE_PIPELINE_HOME"] = self._old_home

    def tearDown(self):
        subprocess.run = self._old_run

    def test_steps_are_prepare_then_generated_minus_se_in_order(self):
        with tempfile.TemporaryDirectory() as td:
            base = fixture(td)
            p = project.load_project(base / "project.yaml")
            steps = cli.still_plan(p, "courier")
            self.assertEqual([s["kind"] for s in steps],
                             ["prepare", "edit", "edit", "edit", "edit"])
            self.assertEqual([s["dir"] for s in steps], ["se", "s", "e", "ne", "n"])
            self.assertEqual(steps[0]["src"], base / "stills" / "courier.png")
            turn = project.load_view(p.view)["turn"]
            for s in steps[1:]:
                self.assertEqual(s["turn"], turn[s["dir"]])

    def test_both_plan_printers_use_the_shared_helper(self):
        """The dry-run printer and the real run each call `still_plan` — the
        duplication cannot silently come back."""
        tree = ast.parse((ROOT / "src" / "sprite_pipeline" / "cli.py").read_text())
        callers = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                for sub in ast.walk(node):
                    if (isinstance(sub, ast.Call)
                            and getattr(sub.func, "id", None) == "still_plan"):
                        callers.add(node.name)
        self.assertEqual(callers, {"_print_still_plan", "run_stills"})

    def test_prepare_happens_once_before_the_edits_in_a_real_run(self):
        """The real run walks the shared plan: the prepare line prints once,
        ahead of exactly the generated-minus-se edit lines, in plan order.
        The subprocesses are faked to materialise their declared outputs —
        the comfy edit copies its --image to --copy-to, the cutout keys its
        input — so the run proceeds like the real pipeline."""
        base = fixture(self.tmp)
        proj = base / "project.yaml"

        class Result:
            def __init__(self):
                self.returncode, self.stdout, self.stderr = 0, "", ""

        calls = []

        def fake_run(argv, **kwargs):
            argv = [str(a) for a in argv]
            joined = " ".join(argv)
            calls.append(joined)
            if "sprite_pipeline.comfy edit" in joined:
                dst = Path(argv[argv.index("--copy-to") + 1])
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(argv[argv.index("--image") + 1], dst)
            elif "cutout" in joined:  # sprite-gen cutout: write the keyed file
                out = Path(argv[argv.index("--out") + 1])
                out.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(argv[argv.index("cutout") + 1], out)
            return Result()

        code, out = run_cli(["go", str(proj), "--dry-run"])
        self.assertEqual(code, 0, out)
        code, out = run_cli(["go", str(proj), "--stage", "pixel",
                             "--no-stop", "--animations", "run", "--dirs", "se"],
                            fake=fake_run)
        self.assertEqual(code, 0, out)
        lines = out.splitlines()
        preps = [i for i, l in enumerate(lines) if l.startswith("[courier] still preparation:")]
        edits = [i for i, l in enumerate(lines) if "] direction edit:" in l]
        self.assertEqual(len(preps), 1)
        self.assertEqual([i for i in edits if i < preps[0]], [])
        self.assertEqual(len(edits), 4)  # s, e, ne, n — never se
        self.assertNotIn("[courier/se] direction edit", out)
        # the direction edits ran in plan order, one subprocess each
        edit_calls = [c for c in calls if "sprite_pipeline.comfy edit" in c]
        self.assertEqual(len(edit_calls), 4)
        order = [re.search(r"--prefix courier/dir-(\w+) ", c).group(1) for c in edit_calls]
        self.assertEqual(order, ["s", "e", "ne", "n"])
        # and the palette cutout of the prepared SE still happened once
        self.assertEqual(len([c for c in calls if "cutout" in c]), 1)


# ---------------------------------------------------------------------- r7: F6

class TestInterimFlagRemoved(unittest.TestCase):
    """F6/r7: the flag is unknown; pre-existing interim files stay valid."""

    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmpdir.cleanup)
        self.tmp = Path(self._tmpdir.name)
        self._old_home = os.environ.get("SPRITE_PIPELINE_HOME")
        os.environ["SPRITE_PIPELINE_HOME"] = str(self.tmp / "home")
        self.addCleanup(self._restore_home)

    def _restore_home(self):
        if self._old_home is None:
            os.environ.pop("SPRITE_PIPELINE_HOME", None)
        else:
            os.environ["SPRITE_PIPELINE_HOME"] = self._old_home

    def test_approve_interim_is_rejected_as_unknown(self):
        base = fixture(self.tmp)
        proj = base / "project.yaml"
        code, out = run_cli(["approve", "--interim", str(proj), "courier"])
        self.assertEqual(code, 2)
        self.assertIn("unrecognized arguments: --interim", out)

    def test_write_refs_approved_has_no_interim_parameter(self):
        self.assertNotIn("interim",
                         inspect.signature(stages.StageCtx.write_refs_approved).parameters)
        self.assertNotIn("interim", inspect.signature(cli.cmd_approve).parameters)

    def test_preexisting_interim_file_still_reads_as_approved(self):
        base = fixture(self.tmp)
        p = project.load_project(base / "project.yaml")
        ctx = stages.StageCtx(p, config.load())
        master = ctx.master("courier")
        master.mkdir(parents=True)
        for d in p.generate:
            (master / f"{d}.png").write_bytes(TINY_PNG)
        f = ctx.refs_approved("courier")
        f.write_text("interim\n" + "".join(f"{d}\n" for d in p.generate))
        self.assertTrue(ctx.is_approved("courier"))
        # the interim rule is names-and-locations, not content: still true
        # with changed bytes, and false once a named still disappears
        (master / "e.png").write_bytes(b"changed")
        self.assertTrue(ctx.is_approved("courier"))
        (master / "n.png").unlink()
        self.assertFalse(ctx.is_approved("courier"))


if __name__ == "__main__":
    unittest.main()
