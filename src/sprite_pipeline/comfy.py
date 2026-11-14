"""ComfyUI workflow loading, slot filling, and the HTTP client (upload, post, collect).

The build fills nodes titled `sp:<slot>` in the workflow's `_meta.title`.
Each slot maps to a fixed input name on that node. A slot the step needs but
the workflow lacks is an error naming the workflow file and the title.

ComfyUI is faked at the HTTP client in tests (no real network).

Can be run as a subprocess: `python -m sprite_pipeline.comfy animate ...`
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.request
import uuid
from pathlib import Path

from sprite_pipeline import config

# slot suffix -> input name on the node
SLOT_INPUTS = {
    "prompt": "text",
    "negative": "text",
    "pose_prompt": "text",
    "seed": "noise_seed",
    "ref_image": "image",
    "drive_video": "file",
    "width": "width",
    "height": "height",
    "length": "length",
    "pose_strength": "pose_strength",
    "save": "filename_prefix",
}


class WorkflowError(ValueError):
    """A workflow is missing a slot the step needs; the message names the
    workflow file and the title."""


def load_workflow(path):
    """Load a ComfyUI API-format workflow JSON file. Returns the graph dict."""
    path = Path(path)
    if not path.is_file():
        raise WorkflowError(f"{path.name}: workflow file not found")
    return json.loads(path.read_text())


def fill_slots(graph, values, workflow_name):
    """Fill `sp:<slot>` nodes in the graph with the given values.

    `values` maps slot names (without the `sp:` prefix) to values.
    For each slot, finds the node with title `sp:<slot>` and sets the input
    whose name is in SLOT_INPUTS. Raises WorkflowError if a slot is missing.
    Returns the modified graph.

    A node may carry multiple `sp:` titles (comma-separated in `_meta.title`),
    in which case each slot is filled on that node.
    """
    slot_to_node = {}
    for node_id, node in graph.items():
        title = (node.get("_meta") or {}).get("title")
        if not title or not title.startswith("sp:"):
            continue
        for part in title.split(","):
            part = part.strip()
            if not part.startswith("sp:"):
                continue
            slot = part[3:]
            if slot in SLOT_INPUTS:
                slot_to_node[slot] = node_id

    for slot, value in values.items():
        title = f"sp:{slot}"
        if slot not in slot_to_node:
            raise WorkflowError(f"{workflow_name}: missing node titled {title}")
        node = graph[slot_to_node[slot]]
        input_name = SLOT_INPUTS[slot]
        if input_name not in node["inputs"]:
            raise WorkflowError(f"{workflow_name}: node {slot_to_node[slot]} ({title}) "
                                f"has no input '{input_name}'")
        node["inputs"][input_name] = value

    return graph


def upload(api, path):
    """Upload an image to ComfyUI's /upload/image endpoint. Returns the filename."""
    data = Path(path).read_bytes()
    boundary = uuid.uuid4().hex
    name = Path(path).name
    body = (f"--{boundary}\r\nContent-Disposition: form-data; name=\"image\"; filename=\"{name}\"\r\n"
            f"Content-Type: image/png\r\n\r\n").encode() + data + f"\r\n--{boundary}\r\nContent-Disposition: form-data; name=\"overwrite\"\r\n\r\ntrue\r\n--{boundary}--\r\n".encode()
    req = urllib.request.Request(f"{api}/upload/image", body, {"Content-Type": f"multipart/form-data; boundary={boundary}"})
    return json.load(urllib.request.urlopen(req))["name"]


def post(api, graph, out_root, poll_interval=2.0):
    """Post the graph to ComfyUI's /prompt endpoint and poll /history until
    the result is ready. Returns the list of output files (under out_root)."""
    body = json.dumps({"prompt": graph, "client_id": str(uuid.uuid4())}).encode()
    req = urllib.request.Request(f"{api}/prompt", body, {"Content-Type": "application/json"})
    pid = json.load(urllib.request.urlopen(req))["prompt_id"]
    while True:
        hist = json.load(urllib.request.urlopen(f"{api}/history/{pid}"))
        if pid in hist:
            h = hist[pid]
            if h["status"].get("status_str") != "success":
                raise SystemExit(json.dumps(h["status"], indent=1)[:3000])
            files = []
            for out in h["outputs"].values():
                for key in ("images", "videos", "gifs"):
                    for f in [x for x in out.get(key, []) if x.get("type", "output") == "output"]:
                        files.append(Path(out_root) / f.get("subfolder", "") / f["filename"])
            return files
        time.sleep(poll_interval)


def animate(cfg, workflow_path, values, ref_image, drive_video, out_clip, log):
    """Run the animate workflow: fill slots, upload inputs, post, collect
    outputs, and write the clip to out_clip.

    Returns 0 on success, 1 on failure.
    """
    api = cfg["comfy_url"]
    out_root = Path(cfg["comfy_output"])

    # upload the reference image
    ref_name = upload(api, ref_image)
    log.append(f"$ upload {ref_image} -> {ref_name}")

    # fill slots (the ref_image slot gets the uploaded reference name;
    # anything the CLI placed in values["ref_image"] is replaced here)
    values = dict(values)
    values["ref_image"] = ref_name
    graph = load_workflow(workflow_path)
    fill_slots(graph, values, Path(workflow_path).name)

    # post and collect
    files = post(api, graph, str(out_root))
    log.append(f"$ {len(files)} frames from ComfyUI")

    if not files:
        log.append("no output frames")
        return 1

    # collect frames into the clip
    Path(out_clip).parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as td:
        for i, f in enumerate(sorted(files)):
            shutil.copyfile(f, f"{td}/f{i:04d}.png")
        ffmpeg = shutil.which("ffmpeg", path=os.pathsep.join(
            x for x in (cfg.get("extra_path", ""), os.environ.get("PATH", "")) if x)) or "ffmpeg"
        r = subprocess.run([ffmpeg, "-y", "-v", "error", "-framerate", "30",
                            "-i", f"{td}/f%04d.png", "-c:v", "libx264", "-crf", "10",
                            "-pix_fmt", "yuv444p", str(out_clip)],
                           capture_output=True, text=True)
        if r.returncode != 0:
            log.append(f"ffmpeg failed: {r.stderr}")
            return 1
    log.append(f"mp4 {out_clip}")
    return 0


def edit(cfg, workflow_path, prompt, seed, image, prefix, copy_to, log):
    """Run the edit workflow: fill slots, upload the image, post, collect
    the output image, and copy it to copy_to.

    Returns 0 on success, 1 on failure.
    """
    api = cfg["comfy_url"]
    out_root = Path(cfg["comfy_output"])

    # upload the image
    name = upload(api, image)
    log.append(f"$ upload {image} -> {name}")

    # fill slots
    graph = load_workflow(workflow_path)
    fill_slots(graph, {
        "prompt": prompt,
        "seed": seed,
        "ref_image": name,
        "save": prefix,
    }, Path(workflow_path).name)

    # post and collect
    files = post(api, graph, str(out_root))
    log.append(f"$ {len(files)} image(s) from ComfyUI")

    if not files:
        log.append("no output image")
        return 1

    Path(copy_to).parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(files[0], copy_to)
    log.append(f"png {copy_to}")
    return 0


def build_parser():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    n = sub.add_parser("edit")
    n.add_argument("--workflow", required=True, help="path to the workflow JSON file")
    n.add_argument("--image", required=True, help="path to the fitted still to edit")
    n.add_argument("--prompt", required=True)
    n.add_argument("--seed", type=int, default=0)
    n.add_argument("--prefix", required=True, help="filename_prefix for sp:save")
    n.add_argument("--copy-to", required=True, help="path to write the output image")
    n = sub.add_parser("animate")
    n.add_argument("--workflow", required=True, help="path to the workflow JSON file")
    n.add_argument("--ref", required=True, help="path to the reference image")
    n.add_argument("--video", required=True, help="name of the drive video in ComfyUI's input dir")
    n.add_argument("--prompt", required=True)
    n.add_argument("--pose-prompt", default="")
    n.add_argument("--negative", default="")
    n.add_argument("--w", type=int, default=704)
    n.add_argument("--h", type=int, default=704)
    n.add_argument("--length", type=int, required=True)
    n.add_argument("--seed", type=int, default=0)
    n.add_argument("--pose-strength", type=float, default=1.0)
    n.add_argument("--prefix", default="anim/a")
    n.add_argument("--copy-to", required=True, help="path to write the output clip")
    return ap


if __name__ == "__main__":
    a = build_parser().parse_args()

    cfg = config.load()
    log = []
    if a.cmd == "edit":
        rc = edit(cfg, a.workflow, a.prompt, a.seed, a.image, a.prefix, a.copy_to, log)
    else:
        values = {
            "prompt": a.prompt,
            "pose_prompt": a.pose_prompt,
            "negative": a.negative,
            "width": a.w,
            "height": a.h,
            "length": a.length,
            "seed": a.seed,
            "pose_strength": a.pose_strength,
            "save": f"{a.prefix}_{a.seed}",
            "ref_image": a.ref,  # the upload inside animate() replaces it
            "drive_video": a.video,
        }
        rc = animate(cfg, a.workflow, values, a.ref, a.video, a.copy_to, log)
    for line in log:
        print(line)
    sys.exit(rc)
