"""Drive the local ComfyUI API: `still` (FLUX.2-klein-4B), `edit` (FLUX.2-klein reference edit), `i2v` (Wan 2.2 14B + lightx2v 4-step) and `animate` (Wan Animate 2)."""
import argparse, json, os, shutil, time, urllib.request, uuid
from pathlib import Path

from sprite_pipeline import config

_C = config.load()
API = _C["comfy_url"]
OUT_ROOT = Path(_C["comfy_output"])
FFMPEG = shutil.which("ffmpeg", path=os.pathsep.join(x for x in (_C["extra_path"], os.environ.get("PATH", "")) if x)) or "ffmpeg"


def post(graph):
    body = json.dumps({"prompt": graph, "client_id": str(uuid.uuid4())}).encode()
    req = urllib.request.Request(f"{API}/prompt", body, {"Content-Type": "application/json"})
    pid = json.load(urllib.request.urlopen(req))["prompt_id"]
    while True:
        hist = json.load(urllib.request.urlopen(f"{API}/history/{pid}"))
        if pid in hist:
            h = hist[pid]
            if h["status"].get("status_str") != "success":
                raise SystemExit(json.dumps(h["status"], indent=1)[:3000])
            files = []
            for out in h["outputs"].values():
                for key in ("images", "videos", "gifs"):
                    for f in [x for x in out.get(key, []) if x.get("type", "output") == "output"]:
                        files.append(OUT_ROOT / f.get("subfolder", "") / f["filename"])
            return files
        time.sleep(2)


def upload(path):
    data = Path(path).read_bytes()
    boundary = uuid.uuid4().hex
    name = Path(path).name
    body = (f"--{boundary}\r\nContent-Disposition: form-data; name=\"image\"; filename=\"{name}\"\r\n"
            f"Content-Type: image/png\r\n\r\n").encode() + data + f"\r\n--{boundary}\r\nContent-Disposition: form-data; name=\"overwrite\"\r\n\r\ntrue\r\n--{boundary}--\r\n".encode()
    req = urllib.request.Request(f"{API}/upload/image", body, {"Content-Type": f"multipart/form-data; boundary={boundary}"})
    return json.load(urllib.request.urlopen(req))["name"]


def still_graph(prompt, w, h, seed, prefix):
    return {
        "1": {"class_type": "UNETLoader", "inputs": {"unet_name": "flux-2-klein-4b.safetensors", "weight_dtype": "default"}},
        "2": {"class_type": "CLIPLoader", "inputs": {"clip_name": "qwen_3_4b.safetensors", "type": "flux2"}},
        "3": {"class_type": "VAELoader", "inputs": {"vae_name": "flux2-vae.safetensors"}},
        "4": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["2", 0], "text": prompt}},
        "5": {"class_type": "ConditioningZeroOut", "inputs": {"conditioning": ["4", 0]}},
        "6": {"class_type": "CFGGuider", "inputs": {"model": ["1", 0], "positive": ["4", 0], "negative": ["5", 0], "cfg": 1.0}},
        "7": {"class_type": "EmptyFlux2LatentImage", "inputs": {"width": w, "height": h, "batch_size": 1}},
        "8": {"class_type": "Flux2Scheduler", "inputs": {"steps": 4, "width": w, "height": h}},
        "9": {"class_type": "KSamplerSelect", "inputs": {"sampler_name": "euler"}},
        "10": {"class_type": "RandomNoise", "inputs": {"noise_seed": seed}},
        "11": {"class_type": "SamplerCustomAdvanced", "inputs": {"noise": ["10", 0], "guider": ["6", 0], "sampler": ["9", 0], "sigmas": ["8", 0], "latent_image": ["7", 0]}},
        "12": {"class_type": "VAEDecode", "inputs": {"samples": ["11", 0], "vae": ["3", 0]}},
        "13": {"class_type": "SaveImage", "inputs": {"images": ["12", 0], "filename_prefix": prefix}},
    }


def edit_graph(image_name, prompt, w, h, seed, prefix):
    g = still_graph(prompt, w, h, seed, prefix)
    g["20"] = {"class_type": "LoadImage", "inputs": {"image": image_name}}
    g["21"] = {"class_type": "ImageScaleToTotalPixels", "inputs": {"image": ["20", 0], "upscale_method": "lanczos", "megapixels": 1.0, "resolution_steps": 16}}
    g["22"] = {"class_type": "VAEEncode", "inputs": {"pixels": ["21", 0], "vae": ["3", 0]}}
    g["23"] = {"class_type": "ReferenceLatent", "inputs": {"conditioning": ["4", 0], "latent": ["22", 0]}}
    g["24"] = {"class_type": "ReferenceLatent", "inputs": {"conditioning": ["5", 0], "latent": ["22", 0]}}
    g["6"]["inputs"]["positive"] = ["23", 0]
    g["6"]["inputs"]["negative"] = ["24", 0]
    return g


def animate_graph(ref_name, video_name, prompt, pose_prompt, w, h, length, seed, prefix, pose_strength=1.0, ref_strength=1.0):
    return {
        "1": {"class_type": "UNETLoader", "inputs": {"unet_name": "wan_animate_2_int8_convrot.safetensors", "weight_dtype": "default"}},
        "2": {"class_type": "LoraLoaderModelOnly", "inputs": {"model": ["1", 0], "lora_name": "lightx2v_I2V_14B_480p_cfg_step_distill_rank64_bf16.safetensors", "strength_model": 1.0}},
        "3": {"class_type": "ModelSamplingSD3", "inputs": {"model": ["2", 0], "shift": 5.0}},
        "4": {"class_type": "CLIPLoader", "inputs": {"clip_name": "umt5_xxl_fp8_e4m3fn_scaled.safetensors", "type": "wan"}},
        "5": {"class_type": "CLIPVisionLoader", "inputs": {"clip_name": "clip_vision_h.safetensors"}},
        "6": {"class_type": "VAELoader", "inputs": {"vae_name": "Wan2_1_VAE_bf16.safetensors"}},
        "7": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["4", 0], "text": prompt}},
        "8": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["4", 0], "text": NEG}},
        "9": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["4", 0], "text": pose_prompt}},
        "10": {"class_type": "LoadImage", "inputs": {"image": ref_name}},
        "11": {"class_type": "ImageScale", "inputs": {"image": ["10", 0], "upscale_method": "lanczos", "width": w, "height": h, "crop": "center"}},
        "12": {"class_type": "LoadVideo", "inputs": {"file": video_name}},
        "13": {"class_type": "GetVideoComponents", "inputs": {"video": ["12", 0]}},
        "14": {"class_type": "ImageScale", "inputs": {"image": ["13", 0], "upscale_method": "lanczos", "width": w, "height": h, "crop": "center"}},
        "15": {"class_type": "CLIPVisionEncode", "inputs": {"clip_vision": ["5", 0], "image": ["11", 0], "crop": "none"}},
        "16": {"class_type": "ImageFromBatch", "inputs": {"image": ["14", 0], "batch_index": 0, "length": 1}},
        "17": {"class_type": "CLIPVisionEncode", "inputs": {"clip_vision": ["5", 0], "image": ["16", 0], "crop": "none"}},
        "18": {"class_type": "WanAnimate2ToVideo", "inputs": {"positive": ["7", 0], "negative": ["8", 0], "vae": ["6", 0], "width": w, "height": h, "length": length, "batch_size": 1,
               "video_frame_offset": 0, "pose_strength": pose_strength, "pose_start_percent": 0.0, "pose_end_percent": 1.0, "reference_image_strength": ref_strength,
               "reference_image": ["11", 0], "pose_video": ["14", 0], "clip_vision_output": ["15", 0], "positive_pose": ["9", 0], "clip_vision_output_pose": ["17", 0]}},
        "19": {"class_type": "KSamplerSelect", "inputs": {"sampler_name": "lcm"}},
        "20": {"class_type": "BasicScheduler", "inputs": {"model": ["3", 0], "scheduler": "simple", "steps": 6, "denoise": 1.0}},
        "21": {"class_type": "SamplerCustom", "inputs": {"model": ["3", 0], "add_noise": True, "noise_seed": seed, "cfg": 1.0, "positive": ["18", 0], "negative": ["18", 1], "sampler": ["19", 0], "sigmas": ["20", 0], "latent_image": ["18", 2]}},
        "22": {"class_type": "TrimVideoLatent", "inputs": {"samples": ["21", 0], "trim_amount": ["18", 3]}},
        "23": {"class_type": "VAEDecode", "inputs": {"samples": ["22", 0], "vae": ["6", 0]}},
        "24": {"class_type": "SaveImage", "inputs": {"images": ["23", 0], "filename_prefix": prefix}},
    }


NEG = "blurry, motion blur, camera movement, zoom, pan, background change, extra limbs, deformed hands, text, watermark, scene cut"


def i2v_graph(image_name, prompt, w, h, length, seed, prefix, fps, pin_end=True):
    g = {
        "1": {"class_type": "UNETLoader", "inputs": {"unet_name": "wan2.2_i2v_high_noise_14B_fp8_scaled.safetensors", "weight_dtype": "default"}},
        "2": {"class_type": "UNETLoader", "inputs": {"unet_name": "wan2.2_i2v_low_noise_14B_fp8_scaled.safetensors", "weight_dtype": "default"}},
        "3": {"class_type": "LoraLoaderModelOnly", "inputs": {"model": ["1", 0], "lora_name": "wan2.2_i2v_lightx2v_4steps_lora_v1_high_noise.safetensors", "strength_model": 1.0}},
        "4": {"class_type": "LoraLoaderModelOnly", "inputs": {"model": ["2", 0], "lora_name": "wan2.2_i2v_lightx2v_4steps_lora_v1_low_noise.safetensors", "strength_model": 1.0}},
        "5": {"class_type": "ModelSamplingSD3", "inputs": {"model": ["3", 0], "shift": 5.0}},
        "6": {"class_type": "ModelSamplingSD3", "inputs": {"model": ["4", 0], "shift": 5.0}},
        "7": {"class_type": "CLIPLoader", "inputs": {"clip_name": "umt5_xxl_fp8_e4m3fn_scaled.safetensors", "type": "wan"}},
        "8": {"class_type": "VAELoader", "inputs": {"vae_name": "wan_2.1_vae.safetensors"}},
        "9": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["7", 0], "text": prompt}},
        "10": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["7", 0], "text": NEG}},
        "11": {"class_type": "LoadImage", "inputs": {"image": image_name}},
        "12": {"class_type": "WanFirstLastFrameToVideo", "inputs": {"positive": ["9", 0], "negative": ["10", 0], "vae": ["8", 0], "start_image": ["11", 0], "end_image": ["11", 0], "width": w, "height": h, "length": length, "batch_size": 1}},
        "13": {"class_type": "KSamplerAdvanced", "inputs": {"model": ["5", 0], "add_noise": "enable", "noise_seed": seed, "steps": 4, "cfg": 1.0, "sampler_name": "euler", "scheduler": "simple", "positive": ["12", 0], "negative": ["12", 1], "latent_image": ["12", 2], "start_at_step": 0, "end_at_step": 2, "return_with_leftover_noise": "enable"}},
        "14": {"class_type": "KSamplerAdvanced", "inputs": {"model": ["6", 0], "add_noise": "disable", "noise_seed": seed, "steps": 4, "cfg": 1.0, "sampler_name": "euler", "scheduler": "simple", "positive": ["12", 0], "negative": ["12", 1], "latent_image": ["13", 0], "start_at_step": 2, "end_at_step": 4, "return_with_leftover_noise": "disable"}},
        "15": {"class_type": "VAEDecode", "inputs": {"samples": ["14", 0], "vae": ["8", 0]}},
        "16": {"class_type": "SaveImage", "inputs": {"images": ["15", 0], "filename_prefix": prefix}},
    }
    if not pin_end:
        del g["12"]["inputs"]["end_image"]
    return g


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("still"); s.add_argument("--prompt", required=True); s.add_argument("--w", type=int, default=768); s.add_argument("--h", type=int, default=1024)
    s.add_argument("--seeds", default="1"); s.add_argument("--prefix", default="still/s")
    v = sub.add_parser("i2v"); v.add_argument("--image", required=True); v.add_argument("--prompt", required=True); v.add_argument("--w", type=int, required=True); v.add_argument("--h", type=int, required=True)
    v.add_argument("--length", type=int, default=81); v.add_argument("--seed", type=int, default=1); v.add_argument("--prefix", default="video/v"); v.add_argument("--fps", type=int, default=16)
    v.add_argument("--copy-to"); v.add_argument("--no-end", action="store_true")
    e = sub.add_parser("edit"); e.add_argument("--image", required=True); e.add_argument("--prompt", required=True); e.add_argument("--w", type=int, default=832); e.add_argument("--h", type=int, default=1024)
    e.add_argument("--seeds", default="1"); e.add_argument("--prefix", default="edit/e")
    n = sub.add_parser("animate"); n.add_argument("--ref", required=True); n.add_argument("--video", required=True); n.add_argument("--prompt", required=True); n.add_argument("--pose-prompt", default="")
    n.add_argument("--w", type=int, default=576); n.add_argument("--h", type=int, default=704); n.add_argument("--length", type=int, required=True); n.add_argument("--seed", type=int, default=1)
    n.add_argument("--prefix", default="anim/a"); n.add_argument("--pose-strength", type=float, default=1.0); n.add_argument("--ref-strength", type=float, default=1.0); n.add_argument("--fps", type=int, default=30); n.add_argument("--copy-to")
    a = ap.parse_args()
    if a.cmd == "still":
        for seed in a.seeds.split(","):
            print(*post(still_graph(a.prompt, a.w, a.h, int(seed), f"{a.prefix}_{seed}")))
    elif a.cmd == "animate":
        files = post(animate_graph(a.ref, a.video, a.prompt, a.pose_prompt, a.w, a.h, a.length, a.seed, a.prefix, a.pose_strength, a.ref_strength))
        print(len(files), "frames")
        if a.copy_to:
            import subprocess, tempfile
            with tempfile.TemporaryDirectory() as td:
                for i, f in enumerate(sorted(files)):
                    shutil.copyfile(f, f"{td}/f{i:04d}.png")
                subprocess.run([FFMPEG, "-y", "-v", "error", "-framerate", str(a.fps), "-i", f"{td}/f%04d.png",
                                "-c:v", "libx264", "-crf", "10", "-pix_fmt", "yuv444p", a.copy_to], check=True)
            print("mp4", a.copy_to)
    elif a.cmd == "edit":
        name = upload(a.image)
        for seed in a.seeds.split(","):
            print(*post(edit_graph(name, a.prompt, a.w, a.h, int(seed), f"{a.prefix}_{seed}")))
    else:
        files = post(i2v_graph(upload(a.image), a.prompt, a.w, a.h, a.length, a.seed, a.prefix, a.fps, pin_end=not a.no_end))
        print(*files)
        if a.copy_to:
            import subprocess, tempfile
            with tempfile.TemporaryDirectory() as td:
                for i, f in enumerate(sorted(files)):
                    shutil.copyfile(f, f"{td}/f{i:04d}.png")
                subprocess.run([FFMPEG, "-y", "-v", "error", "-framerate", str(a.fps), "-i", f"{td}/f%04d.png",
                                "-c:v", "libx264", "-crf", "10", "-pix_fmt", "yuv444p", a.copy_to], check=True)
            print("mp4", a.copy_to)
