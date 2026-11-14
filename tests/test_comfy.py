"""Tests for ComfyUI workflow loading, slot filling, and the animate stage
(spec AC6, FR 9).

A loaded workflow gets every `sp:` slot the step needs filled; a missing slot
is an error naming the workflow file and the title; an animation's `workflow`
replaces `workflows.animate` for that animation.

ComfyUI is faked at the HTTP client (no real network).
"""
import json
import shutil
import tempfile
import threading
import unittest
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path

import sys
import os

from sprite_pipeline import comfy, project, stages, config

ROOT = Path(__file__).resolve().parent.parent
WORKFLOWS_DIR = ROOT / "src" / "sprite_pipeline" / "workflows"


class TestSlotFilling(unittest.TestCase):
    """AC6-r1, r2, r3: slot filling and missing slot error."""

    def test_r1_animate_workflow_slots_filled(self):
        """AC6-r1: shipped `animate.json` + values for prompt, negative,
        pose_prompt, seed, ref_image, drive_video, width, height, length,
        pose_strength, save → each `sp:` node's input holds its value; other
        nodes unchanged."""
        wf_path = WORKFLOWS_DIR / "animate.json"
        graph = comfy.load_workflow(wf_path)
        values = {
            "prompt": "a person jogging",
            "negative": "blurry, motion blur",
            "pose_prompt": "jogging in place",
            "seed": 42,
            "ref_image": "ref.png",
            "drive_video": "drive.mp4",
            "width": 704,
            "height": 704,
            "length": 45,
            "pose_strength": 0.75,
            "save": "anim/42",
        }
        result = comfy.fill_slots(graph, values, "animate.json")
        # check each sp: node
        self.assertEqual(result["7"]["inputs"]["text"], "a person jogging")
        self.assertEqual(result["8"]["inputs"]["text"], "blurry, motion blur")
        self.assertEqual(result["9"]["inputs"]["text"], "jogging in place")
        self.assertEqual(result["21"]["inputs"]["noise_seed"], 42)
        self.assertEqual(result["10"]["inputs"]["image"], "ref.png")
        self.assertEqual(result["12"]["inputs"]["file"], "drive.mp4")
        self.assertEqual(result["11"]["inputs"]["width"], 704)
        self.assertEqual(result["14"]["inputs"]["height"], 704)
        self.assertEqual(result["18"]["inputs"]["length"], 45)
        self.assertEqual(result["18"]["inputs"]["pose_strength"], 0.75)
        self.assertEqual(result["24"]["inputs"]["filename_prefix"], "anim/42")
        # other nodes unchanged (spot check)
        self.assertEqual(result["1"]["class_type"], "UNETLoader")
        self.assertEqual(result["2"]["class_type"], "LoraLoaderModelOnly")

    def test_r2_edit_workflow_slots_filled(self):
        """AC6-r2: shipped `edit.json` + prompt, seed, ref_image, save →
        filled likewise."""
        wf_path = WORKFLOWS_DIR / "edit.json"
        graph = comfy.load_workflow(wf_path)
        values = {
            "prompt": "replace the character",
            "seed": 7,
            "ref_image": "se.png",
            "save": "edit/7",
        }
        result = comfy.fill_slots(graph, values, "edit.json")
        self.assertEqual(result["4"]["inputs"]["text"], "replace the character")
        self.assertEqual(result["10"]["inputs"]["noise_seed"], 7)
        self.assertEqual(result["20"]["inputs"]["image"], "se.png")
        self.assertEqual(result["13"]["inputs"]["filename_prefix"], "edit/7")
        # other nodes unchanged
        self.assertEqual(result["1"]["class_type"], "UNETLoader")

    def test_r3_missing_slot_error(self):
        """AC6-r3: `animate.json` copy with the `sp:seed` node retitled →
        error `workflows/animate.json: missing node titled sp:seed`."""
        wf_path = WORKFLOWS_DIR / "animate.json"
        graph = comfy.load_workflow(wf_path)
        # retitle the sp:seed node
        graph["21"]["_meta"]["title"] = "sp:something_else"
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            alt_path = tmp / "animate.json"
            alt_path.write_text(json.dumps(graph))
            values = {"seed": 42}
            with self.assertRaises(comfy.WorkflowError) as cm:
                comfy.fill_slots(comfy.load_workflow(alt_path), values, "animate.json")
            self.assertIn("animate.json", str(cm.exception))
            self.assertIn("sp:seed", str(cm.exception))


class TestWorkflowOverride(unittest.TestCase):
    """AC6-r4: animation with `workflow: workflows/alt.json` → the animate
    stage for that animation posts alt.json's graph; others post the default."""

    def test_r4_per_animation_workflow_override(self):
        """The animate stage uses the animation's workflow if set, otherwise
        the project's default."""
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            # create a minimal project
            project_dir = tmp / "proj"
            project_dir.mkdir()
            stills_dir = project_dir / "stills"
            anims_dir = project_dir / "anims"
            stills_dir.mkdir()
            anims_dir.mkdir()
            (stills_dir / "courier.png").write_bytes(b"fake png")
            (anims_dir / "ual.glb").write_bytes(b"fake glb")
            # create an alt workflow
            workflows_dir = project_dir / "workflows"
            workflows_dir.mkdir()
            alt_wf = workflows_dir / "alt.json"
            alt_wf.write_text(json.dumps({"1": {"class_type": "Test", "inputs": {}, "_meta": {"title": "sp:prompt"}}}))
            # default workflow
            default_wf = workflows_dir / "animate.json"
            shutil.copy(WORKFLOWS_DIR / "animate.json", default_wf)
            (workflows_dir / "edit.json").write_text(json.dumps({"1": {"class_type": "Test", "inputs": {}, "_meta": {"title": "sp:prompt"}}}))

            project_text = f"""style: "test"
view: isometric
directions:
  generate: [s]
  mirror: {{}}
sprite: {{scale: 0.2, colors: 40, accent_boost: 3, fps: 30, frame_size: 704}}
naming: "{{animation}}_{{dir}}"
export: godot
seed: 7
pose_strength: 0.75
output: out
workflows: {{edit: workflows/edit.json, animate: workflows/animate.json}}
characters:
  - {{name: courier, still: stills/courier.png, describe: "test", pronoun: she}}
animations:
  - {{name: run, source: "anims/ual.glb#Jog", frames: 22, start: 1, clip_frames: 45, loop: true, prompt: "{{who}} jogging"}}
  - {{name: alt, source: "anims/ual.glb#Jog", frames: 22, start: 1, clip_frames: 45, loop: true, prompt: "{{who}} jogging", workflow: workflows/alt.json}}
"""
            project_path = project_dir / "project.yaml"
            project_path.write_text(project_text)
            proj = project.load_project(project_path)
            cfg = config.defaults(config.home())
            ctx = stages.StageCtx(proj, cfg)
            anim_run = proj.animation("run")
            anim_alt = proj.animation("alt")
            # set the workflow attribute on the alt animation
            anim_alt.workflow = "workflows/alt.json"
            # plan the animate stage for both animations
            cmds_run = ctx.plan_pair("courier", anim_run, "s", "animate")
            cmds_alt = ctx.plan_pair("courier", anim_alt, "s", "animate")
            # the default animation uses the default workflow
            argv_run = cmds_run[0][0]
            self.assertIn("--workflow", argv_run)
            wf_idx = argv_run.index("--workflow")
            self.assertEqual(argv_run[wf_idx + 1], str(project_dir / "workflows" / "animate.json"))
            # the alt animation uses the alt workflow
            argv_alt = cmds_alt[0][0]
            wf_idx = argv_alt.index("--workflow")
            self.assertEqual(argv_alt[wf_idx + 1], str(project_dir / "workflows" / "alt.json"))


class FakeComfyUIHandler(BaseHTTPRequestHandler):
    """A fake ComfyUI server for testing."""
    def log_message(self, *args):
        pass  # silence

    def do_POST(self):
        if self.path == "/prompt":
            length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(length)
            import uuid
            pid = str(uuid.uuid4())
            self.server.last_prompt = json.loads(body)
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"prompt_id": pid}).encode())
            self.server.prompt_id = pid
        elif self.path == "/upload/image":
            length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(length)
            self.server.last_upload = body
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"name": "uploaded.png"}).encode())
        else:
            self.send_response(404)
            self.end_headers()

    def do_GET(self):
        if self.path.startswith("/history/"):
            pid = self.path.split("/")[-1]
            self.server.history_calls += 1
            if self.server.history_calls >= self.server.history_threshold:
                # return success with one output file
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                out = {"outputs": {"13": {"images": [{"filename": "test_00001.png", "subfolder": "", "type": "output"}]}}}
                self.wfile.write(json.dumps({pid: {"status": {"status_str": "success"}, **out}}).encode())
            else:
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({}).encode())
        else:
            self.send_response(404)
            self.end_headers()


class TestComfyUIHTTP(unittest.TestCase):
    """AC6-r5: fake ComfyUI HTTP server answering /upload/image, /prompt,
    /history → outputs collected into the clip path; no real network."""

    def test_r5_outputs_collected_into_clip(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            # set up a fake ComfyUI server
            server = HTTPServer(("127.0.0.1", 0), FakeComfyUIHandler)
            server.prompt_id = None
            server.last_prompt = None
            server.last_upload = None
            server.history_calls = 0
            server.history_threshold = 1
            port = server.server_address[1]
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            self.addCleanup(server.shutdown)

            api = f"http://127.0.0.1:{port}"
            # create a fake ref image
            from PIL import Image
            ref_path = tmp / "ref.png"
            Image.new("RGB", (10, 10), (255, 0, 0)).save(ref_path)

            # create a fake workflow
            wf_path = tmp / "animate.json"
            wf_path.write_text(json.dumps({
                "7": {"class_type": "CLIPTextEncode", "inputs": {"text": ""}, "_meta": {"title": "sp:prompt"}},
                "10": {"class_type": "LoadImage", "inputs": {"image": ""}, "_meta": {"title": "sp:ref_image"}},
                "12": {"class_type": "LoadVideo", "inputs": {"file": ""}, "_meta": {"title": "sp:drive_video"}},
                "24": {"class_type": "SaveImage", "inputs": {"filename_prefix": ""}, "_meta": {"title": "sp:save"}},
            }))

            # create a fake output file (the server will report this path)
            out_root = tmp / "output"
            out_root.mkdir()
            (out_root / "test_00001.png").write_bytes(b"fake frame")

            cfg = {"comfy_url": api, "comfy_output": str(out_root), "comfy_input": str(tmp), "extra_path": ""}
            values = {"prompt": "test", "save": "test", "drive_video": "drive.mp4"}
            log = []
            out_clip = tmp / "clip.mp4"
            # mock subprocess.run to avoid requiring ffmpeg
            import subprocess
            original_run = subprocess.run
            mock_result = type('MockResult', (), {'returncode': 0, 'stdout': '', 'stderr': ''})
            subprocess.run = lambda *args, **kwargs: mock_result
            try:
                rc = comfy.animate(cfg, str(wf_path), values, str(ref_path), "drive.mp4", out_clip, log)
            finally:
                subprocess.run = original_run
            # the upload was called
            self.assertIsNotNone(server.last_upload)
            # the prompt was posted
            self.assertIsNotNone(server.last_prompt)
            # the log shows the frames were collected
            self.assertIn("1 frames from ComfyUI", " ".join(log))
            # the clip was written (mocked)
            self.assertEqual(rc, 0)
