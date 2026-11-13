"""Host-specific locations, read from $SPRITE_PIPELINE_HOME/config.toml (default ~/.sprite-pipeline).

Every key is optional; missing keys fall back to the defaults below. The config
keeps only machine locations (spec FR 12): where ComfyUI lives, where Blender
is, where rendered drives are cached. `sprite_gen` and `anim_src` were removed
with the project model — sprite-gen runs from the tool's own environment and
project paths resolve against the project file. An unknown key is an error.
"""
import os
import tomllib
from pathlib import Path


def home():
    return Path(os.environ.get("SPRITE_PIPELINE_HOME", Path.home() / ".sprite-pipeline")).expanduser()


def defaults(h):
    return {
        "comfy_url": "http://127.0.0.1:8188",     # ComfyUI API
        "comfy_input": str(h / "comfyui" / "input"),    # ComfyUI's input/ dir (refs and drive clips)
        "comfy_output": str(h / "comfyui" / "output"),  # ComfyUI's output dir
        "blender": "blender",
        "motion_cache": str(h / "motion-cache"),   # rendered drives, shared by every project
        "drive_lsr": "",                           # optional Blender file holding the drive scene
        "extra_path": "",                          # prepended to PATH for child tools (ffmpeg, ffprobe, img2webp)
    }


def load():
    h = home()
    c = defaults(h)
    f = h / "config.toml"
    if f.is_file():
        user = tomllib.loads(f.read_text())
        unknown = set(user) - set(c)
        if unknown:
            names = ", ".join(sorted(unknown))
            raise SystemExit(f"{f}: unknown key {names} (config keeps only machine locations)")
        c.update({k: v if k == "comfy_url" else os.path.expanduser(v) for k, v in user.items()})
    return c
