"""Tests for the comfy CLI at the HTTP seam (spec AC4, AC6; FR 7-9).

The real `python -m sprite_pipeline.comfy` subprocess runs against a fake
ComfyUI HTTP server on 127.0.0.1 (the spec's Non-Functional Requirements:
ComfyUI is faked at the HTTP client, subprocesses at the runner). No
subprocess.run fake for the comfy calls themselves: the `go` row lets the real
`comfy edit` subprocess execute; the other tools (sprite-gen cutout, blender,
ffmpeg) are faked at the runner, as in test_review.py.

AC4-r5: `go` on an unapproved character makes one real edit per generated
direction other than `se`, each posting the view's turn prompt in the
character's pronoun, writes the direction stills and refs-sheet.png, makes
no animate post and exits 0 printing the approve command.

AC6-r6: `comfy edit` with the shipped `edit.json` posts a graph whose
`sp:ref_image` input is the name the upload returned, `sp:seed` holds the
seed, `sp:save` holds the prefix; the output image is copied to --copy-to;
exit 0. AC6-r7: `comfy animate` fills `sp:ref_image` with the uploaded
reference name, not the drive video name. AC6-r8: a retitled `sp:seed` node
is an error naming the workflow file and the title, with no `/prompt` post.
"""
import io
import json
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import unittest
from contextlib import redirect_stderr, redirect_stdout
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import yaml
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
EXAMPLE = ROOT / "examples" / "isometric"
WF_EDIT = ROOT / "src" / "sprite_pipeline" / "workflows" / "edit.json"
WF_ANIMATE = ROOT / "src" / "sprite_pipeline" / "workflows" / "animate.json"

# a valid 1x1 PNG the fake server serves as the render output
TINY_PNG = bytes.fromhex(
    "89504e470d0a1a0a0000000d4948445200000001000000010806000000"
    "1f15c4890000000d49444154789c63e03ac1f51f00039801dc8b1f920b"
    "0000000049454e44ae426082")

EDIT_DIRS = ["s", "e", "ne", "n"]  # the generated directions that are not se

# the review prompt's framing, copied from the new-kit pattern (FR 7): the
# whole string is stated independently of production code, as in test_review.py
TAIL = " Full body, plain flat pure green background, no shadow."
OBJ = {"he": "him", "she": "her", "they": "them"}
POS = {"he": "his", "she": "her", "they": "their"}


def expected_prompt(turn, d, pronoun):
    """The FR 7 direction edit prompt: the fixed framing plus the view's turn
    text with the pronoun substituted."""
    obj, pos = OBJ[pronoun], POS[pronoun]
    return ("The exact same character with the exact same outfit, colors, proportions "
            f"and art style, seen from the exact same high isometric camera angle "
            f"looking down at {obj}, standing in the same relaxed pose, but now "
            + turn[d].format(obj=obj, pos=pos) + "." + TAIL)


class FakeComfyUI(BaseHTTPRequestHandler):
    """A fake ComfyUI answering /upload/image, /prompt and /history on
    127.0.0.1. Uploads are stored under `inputs` and the file's own name is
    returned (ComfyUI's overwrite behaviour); renders write a real PNG under
    `output/<prefix>/` and /history reports it on the first poll."""

    def log_message(self, *args):
        pass

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length)
        if self.path == "/upload/image":
            # recover the filename from the multipart Content-Disposition
            name = "upload"
            for token in body.split(b"\r\n"):
                if b"filename=\"" in token:
                    name = token.split(b"filename=\"")[1].split(b"\"")[0].decode()
            # the file's bytes are what follows the part's headers and
            # precedes the final boundary
            boundary = self.headers.get("Content-Type", "")
            boundary = boundary.split("boundary=")[-1]
            (self.server.inputs / name).write_bytes(
                body.split(b"\r\n\r\n", 1)[1]
                    .split(f"\r\n--{boundary}".encode())[0])
            resp = {"name": name, "subfolder": "", "type": "input"}
        elif self.path == "/prompt":
            graph = json.loads(body)["prompt"]
            self.server.prompts.append(graph)
            save_prefix = ""
            for node in graph.values():
                if (node.get("_meta") or {}).get("title") == "sp:save":
                    save_prefix = node["inputs"]["filename_prefix"]
            # ComfyUI's SaveImage subfolder is the prefix's own directory
            subfolder = Path(save_prefix).parent
            out_dir = self.server.output / subfolder
            out_dir.mkdir(parents=True, exist_ok=True)
            out_file = out_dir / f"{save_prefix.rsplit('/', 1)[-1]}_00001.png"
            out_file.write_bytes(TINY_PNG)
            pid = f"pid{len(self.server.prompts)}"
            self.server.histories[pid] = {
                "status": {"status_str": "success"},
                "outputs": {"13": {"images": [{
                    "filename": out_file.name,
                    "subfolder": str(subfolder), "type": "output"}]}},
            }
            resp = {"prompt_id": pid}
        else:
            self.send_response(404)
            self.end_headers()
            return
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(resp).encode())

    def do_GET(self):
        if self.path.startswith("/history/"):
            pid = self.path.split("/")[-1]
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(
                {pid: self.server.histories.get(pid, {})}).encode())
        else:
            self.send_response(404)
            self.end_headers()


class FakeRun:
    """subprocess.run stand-in: the comfy argv is executed as a real
    subprocess (the HTTP server is the only fake for it); every other tool
    is faked at the runner, per the spec's Non-Functional Requirements."""

    def __init__(self, real_run, after=None):
        self.calls = []
        self.after = after or {}
        self._real_run = real_run

    def __call__(self, argv, **kwargs):
        argv = [str(a) for a in argv]
        self.calls.append(argv)
        joined = " ".join(argv)
        if argv[1:3] == ["-m", "sprite_pipeline.comfy"]:
            # the real comfy subprocess runs against the fake HTTP server
            r = self._real_run(argv, capture_output=True, text=True)
            if r.stdout.strip():
                print(r.stdout.strip())
            if r.stderr.strip():
                print(r.stderr.strip())
            return type("R", (), {"returncode": r.returncode,
                                  "stdout": r.stdout, "stderr": r.stderr})()
        for match, hook in self.after.items():
            if match in joined:
                hook(argv)
        return type("R", (), {"returncode": 0, "stdout": "", "stderr": ""})()

    def comfy_calls(self):
        return [c for c in self.calls
                if len(c) > 3 and c[1:3] == ["-m", "sprite_pipeline.comfy"]
                and c[3] in ("edit", "animate")]


def make_project(tmp, pronoun="she"):
    """A stubbed working copy of examples/isometric: real project.yaml, stub
    GLB bytes and a valid PNG source still (the run prepares it for real)."""
    base = Path(tmp) / "isometric"
    shutil.copytree(EXAMPLE, base)
    # the example project's workflow paths are relative to the file; the
    # packaged workflow files stand in for the copy the user keeps there
    (base / "workflows").mkdir(parents=True, exist_ok=True)
    for name in ("edit.json", "animate.json"):
        shutil.copyfile(ROOT / "src" / "sprite_pipeline" / "workflows" / name,
                        base / "workflows" / name)
    (base / "anims").mkdir(parents=True, exist_ok=True)
    (base / "anims" / "ual.glb").write_bytes(TINY_PNG)
    (base / "stills").mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (64, 80), (10, 200, 10)).save(base / "stills" / "courier.png")
    if pronoun != "she":
        pf = base / "project.yaml"
        pf.write_text(pf.read_text().replace("pronoun: she", f"pronoun: {pronoun}"))
    return base / "project.yaml"


class Env(unittest.TestCase):
    """Isolated config home pointing at the fake server's URL and dirs; the
    real `sprite_pipeline` entry points are invoked in-process, and the real
    `python -m sprite_pipeline.comfy` subprocess inherits the config home
    through the environment."""

    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmpdir.cleanup)
        self.tmp = Path(self._tmpdir.name)
        self.home = self.tmp / "home"
        self.home.mkdir()
        self._old_home = os.environ.get("SPRITE_PIPELINE_HOME")
        os.environ["SPRITE_PIPELINE_HOME"] = str(self.home)
        self.addCleanup(self._restore_home)

        self.inputs = self.tmp / "comfy_input"
        self.output = self.tmp / "comfy_output"
        self.inputs.mkdir(parents=True)
        self.output.mkdir(parents=True)

        self.server = HTTPServer(("127.0.0.1", 0), FakeComfyUI)
        self.server.inputs = self.inputs
        self.server.output = self.output
        self.server.prompts = []
        self.server.histories = {}
        self.port = self.server.server_address[1]
        (self.home / "config.toml").write_text(
            f'comfy_url = "http://127.0.0.1:{self.port}"\n'
            f'comfy_input = "{self.inputs}"\n'
            f'comfy_output = "{self.output}"\n')
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.addCleanup(self.server.shutdown)

        self._old_run = subprocess.run  # restored in tearDown
        self._real_run = self._old_run  # the un-faked run, kept for the real comfy calls

    def _restore_home(self):
        if self._old_home is None:
            os.environ.pop("SPRITE_PIPELINE_HOME", None)
        else:
            os.environ["SPRITE_PIPELINE_HOME"] = self._old_home

    def tearDown(self):
        subprocess.run = self._old_run

    def run_cli(self, argv, fake):
        """Invoke cli.main with subprocess.run faked at the runner; stdout and
        stderr are captured. The fake lets the real comfy subprocess through."""
        subprocess.run = fake
        from sprite_pipeline import cli
        buf, err = io.StringIO(), io.StringIO()
        code = None
        try:
            with redirect_stdout(buf), redirect_stderr(err):
                code = cli.main(argv)
        except SystemExit as e:
            code = e.code if isinstance(e.code, int) else 1
            buf.write(str(e) + "\n")
        return code, buf.getvalue() + err.getvalue()

    def run_module(self, argv):
        """Run `python -m sprite_pipeline.comfy <argv>` for real (no fake)."""
        r = self._real_run([sys.executable, "-m", "sprite_pipeline.comfy"] + argv,
                           capture_output=True, text=True, cwd=str(ROOT),
                           env={**os.environ, "PYTHONPATH": str(ROOT / "src")})
        return r.returncode, r.stdout + r.stderr


# ------------------------------------------------------------- AC4-r5: `go`

class TestGoRealEditAtHTTPSeam(Env):
    """AC4-r5: the real `comfy edit` subprocess runs against the fake ComfyUI
    HTTP server; everything else is faked at the runner."""

    @staticmethod
    def _cutout():
        def fake_cutout(argv):
            out = Path(argv[argv.index("--out") + 1])
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_bytes(TINY_PNG)

        return {"cutout": fake_cutout}

    def test_r5_exactly_four_edits_and_no_animate(self):
        project = make_project(self.tmp)
        fake = FakeRun(self._real_run, after=self._cutout())
        code, out = self.run_cli(["go", str(project)], fake)
        self.assertEqual(code, 0, out)
        # exactly 4 edit posts at the real HTTP seam, no animate post
        self.assertEqual(len([c for c in fake.comfy_calls() if c[3] == "edit"]), 4, out)
        self.assertEqual([c for c in fake.comfy_calls() if c[3] == "animate"], [], out)
        self.assertEqual(len(self.server.prompts), 4)

    def test_r5_exit_zero_prints_approve_command(self):
        project = make_project(self.tmp)
        base = project.parent
        turn = yaml.safe_load(
            (ROOT / "src" / "sprite_pipeline" / "views" / "isometric.yaml")
            .read_text())["turn"]
        master = base / "out" / "courier" / "master"
        fake = FakeRun(self._real_run, after=self._cutout())
        code, out = self.run_cli(["go", str(project)], fake)
        self.assertEqual(code, 0, out)

        # exactly 4 edits (s, e, ne, n), no animate, at the real HTTP seam
        edits = [c for c in fake.comfy_calls() if c[3] == "edit"]
        self.assertEqual(len(edits), 4, out)
        self.assertEqual([c for c in fake.comfy_calls() if c[3] == "animate"], [], out)
        self.assertEqual(len(self.server.prompts), 4)

        # each post holds the view's turn prompt for that direction in her
        # pronoun; the image is the fitted SE still; the prefix is char/dir-<d>
        for d in EDIT_DIRS:
            argv = next((c for c in edits
                         if c[c.index("--copy-to") + 1] == str(master / f"{d}.png")), d)
            self.assertEqual(argv[argv.index("--prompt") + 1],
                             expected_prompt(turn, d, "she"), d)
            self.assertEqual(argv[argv.index("--image") + 1], str(master / "se.fit.png"), d)
            self.assertEqual(argv[argv.index("--prefix") + 1], f"courier/dir-{d}", d)
            g = next((g for g in self.server.prompts
                      if any((n.get("_meta") or {}).get("title") == "sp:prompt"
                             and n["inputs"]["text"] == expected_prompt(turn, d, "she")
                             for n in g.values())), None)
            self.assertIsNotNone(g, d)
            ref = next(n for n in g.values()
                       if (n.get("_meta") or {}).get("title") == "sp:ref_image")
            self.assertEqual(ref["inputs"]["image"], "se.fit.png", d)
            self.assertEqual(next(n for n in g.values()
                                  if (n.get("_meta") or {}).get("title") == "sp:seed")
                             ["inputs"]["noise_seed"], 7, d)  # examples project seed

        # the 4 direction stills and the review sheet are written; the run ends
        # at 0 with the approve command (the review stop is the last line)
        for d in EDIT_DIRS:
            self.assertTrue((master / f"{d}.png").is_file(), d)
        sheet = base / "out" / "courier" / "refs-sheet.png"
        self.assertTrue(sheet.is_file())
        lines = [l for l in out.splitlines() if "stopped for review" in l]
        self.assertEqual(len(lines), 1, out)
        self.assertTrue(lines[-1].endswith(
            f"sprite-pipeline approve {project} courier"), lines[-1])
        self.assertIn(str(sheet), out)


# ------------------------------------------------------------- AC6-r6: comfy edit

class TestComfyEdit(Env):
    """AC6-r6: `comfy edit` with the shipped `edit.json`: the posted graph's
    sp:ref_image is the upload's returned name, sp:seed holds 3, sp:save the
    prefix; the output image is copied to --copy-to; exit 0."""

    def test_r6_edit_fills_slots_copies_output(self):
        img = self.tmp / "se.png"
        Image.new("RGB", (32, 32), (10, 200, 10)).save(img)
        dst = self.tmp / "out" / "n.png"
        rc, text = self.run_module([
            "edit", "--workflow", str(WF_EDIT), "--image", str(img),
            "--prompt", "turn the character north", "--seed", "3",
            "--prefix", "c/dir-n", "--copy-to", str(dst)])
        self.assertEqual(rc, 0, text)
        self.assertEqual(len(self.server.prompts), 1)
        g = self.server.prompts[0]
        by_title = {((n.get("_meta") or {}).get("title")): n for n in g.values()}
        self.assertEqual(by_title["sp:ref_image"]["inputs"]["image"], "se.png")
        self.assertEqual(by_title["sp:seed"]["inputs"]["noise_seed"], 3)
        self.assertEqual(by_title["sp:save"]["inputs"]["filename_prefix"], "c/dir-n")
        self.assertEqual(by_title["sp:prompt"]["inputs"]["text"],
                         "turn the character north")
        # the fake server wrote the render under output/c/dir-n/; the CLI
        # copied it to --copy-to
        self.assertTrue(dst.is_file())
        self.assertEqual(dst.read_bytes(), TINY_PNG)


# ----------------------------------------------------------- AC6-r7: comfy animate

class TestComfyAnimateRefName(Env):
    """AC6-r7: `comfy animate` fills sp:ref_image with the uploaded reference
    name, not the drive video name; sp:drive_video holds the video."""

    def test_r7_ref_image_is_the_uploaded_name(self):
        ref = self.tmp / "ref.png"
        Image.new("RGB", (32, 32), (10, 200, 10)).save(ref)
        clip = self.tmp / "clip.mp4"
        # no ffmpeg is needed: the seam assertions hold on the posted graph
        rc, text = self.run_module([
            "animate", "--workflow", str(WF_ANIMATE), "--ref", str(ref),
            "--video", "drive.mp4", "--prompt", "jogging in place",
            "--length", "45", "--seed", "3", "--copy-to", str(clip)])
        self.assertEqual(len(self.server.prompts), 1, text)
        g = self.server.prompts[0]
        by_title = {}
        for n in g.values():
            title = (n.get("_meta") or {}).get("title")
            for part in (title or "").split(","):
                if part.startswith("sp:"):
                    by_title[part] = n
        self.assertEqual(by_title["sp:ref_image"]["inputs"]["image"], "ref.png")
        self.assertNotEqual(by_title["sp:ref_image"]["inputs"]["image"], "drive.mp4")
        self.assertEqual(by_title["sp:drive_video"]["inputs"]["file"], "drive.mp4")


# --------------------------------------------------------- AC6-r8: missing slot

class TestComfyEditMissingSlot(Env):
    """AC6-r8: an `edit.json` copy whose `sp:seed` node is retitled →
    non-zero exit, the error names the workflow file and the title, and no
    /prompt post is made."""

    def test_r8_missing_sp_seed_is_an_error(self):
        graph = json.loads(WF_EDIT.read_text())
        graph["10"]["_meta"]["title"] = "sp:something_else"
        wf = self.tmp / "edit.json"
        wf.write_text(json.dumps(graph))
        img = self.tmp / "se.png"
        Image.new("RGB", (32, 32), (10, 200, 10)).save(img)
        rc, text = self.run_module([
            "edit", "--workflow", str(wf), "--image", str(img),
            "--prompt", "p", "--seed", "3",
            "--prefix", "c/dir-n", "--copy-to", str(self.tmp / "out.png")])
        self.assertNotEqual(rc, 0, text)
        self.assertIn("edit.json: missing node titled sp:seed", text)
        self.assertEqual(self.server.prompts, [])


if __name__ == "__main__":
    unittest.main()
