# sprite-pipeline

Turns one character still into a pixel-art sprite set for every move in a shared motion library: a Blender mannequin drive, motion transfer with Wan Animate 2 in ComfyUI, keying and looping with `sprite-gen`, pixelization at a fixed scale, and a Godot 4 `SpriteFrames` export. A new character needs a new kit, not new motion work.

## Install

```bash
uv tool install git+https://github.com/terminus-labs-ai/sprite-pipeline@main
```

External tools it calls: ComfyUI (with the FLUX.2-klein and Wan Animate 2 models), Blender 5.x, `sprite-gen`, and `ffmpeg`/`ffprobe`/`img2webp`.

## Configure

Host locations live in `$SPRITE_PIPELINE_HOME/config.toml` (default `~/.sprite-pipeline/config.toml`). Every key is optional; defaults are in `src/sprite_pipeline/config.py`.

```toml
comfy_url = "http://127.0.0.1:8188"
comfy_input = "/path/to/comfyui/input"     # refs and drive clips go here
comfy_output = "/path/to/comfyui/output"
blender = "/path/to/blender"
sprite_gen = "sprite-gen"
anim_src = "/path/to/anim-src"             # relative GLB paths in motions.yaml resolve here
motion_cache = "/path/to/motion-cache"     # rendered drives, shared by every kit
extra_path = "/path/to/ffmpeg-bin"         # prepended to PATH for child tools
```

The default motion library uses the Quaternius Universal Animation Library (CC0) at `<anim_src>/ual/Animation Library[Standard]/Godot/AnimationLibrary_Godot_Standard.glb`.

## Build a sprite set

```bash
sprite-pipeline <kit.yaml> --moves run,punch --dirs s,se --dry-run
sprite-pipeline <kit.yaml> --moves run,punch [--stage drive|animate|key|loop|pixel|export] [--force] [--motions <motions.yaml>]
```

Stages per (move, direction): `drive` (Blender render, cached and shared across kits), `animate` (Wan Animate 2), `key` (`sprite-gen video-frames`), `loop` (`sprite-gen video-loop`). Then per kit: `pixel` (one pixelize call over all sets) and `export`. Stages whose output exists are skipped unless `--force`. A failed pair is logged and the build continues, ending with a summary table (exit 1 if anything failed).

Output under the kit's `output_root`: `clips/<move>-<dir>.mp4`, `loops/<move>-<dir>/cycle/`, and `sprites/<kit>/` with the pixel frames, mirrored `sw`←`se`, `w`←`e`, `nw`←`ne`, `atlas.png`, `sprite_frames.tres` (Godot 4, one animation per `<move>_<dir>`, atlas path relative to the `.tres`) and `contact-sheet.png` (rows = moves, columns = 8 directions).

One bad clip: add `overrides: {<move>-<dir>: {seed: N, extra: "<prompt line>"}}` to the kit, rerun that pair with `--moves <move> --dirs <dir> --stage animate --force`, then the `key`, `loop`, `pixel` and `export` stages.

## Motion library (`motions.yaml`)

Per move: GLB action, repeats and pre/post holds, `head_pitch` (25 keeps faces visible; 0 for moves that look down), clip length (Wan needs 4n+1 frames), loop cycle (`fixed` or `once`), sprite-gen state, optional `seam_max`, pose prompt (`{who}` is filled from the kit) and the 5 generated directions. Drives are cached as `<motion_cache>/<move>-<dir>-<hash>.mp4` (hash of the move params) and copied to ComfyUI's input as `drive-<move>-<dir>-<hash>.mp4`.

## Character kit (`<kit>.yaml`)

`tests/fixtures/guard.yaml` is a complete example: `pose_who`, `pose_strength` (0.75), per-direction ComfyUI refs, look/face/background prompt lines (the face line applies only to `face_dirs`), palette key frame, pixel scale and colours, seed, `output_root` and optional `overrides`.

Making a kit by hand:

1. Swap the character into an existing kit's isometric SE still: `python -m sprite_pipeline.comfy_run edit --image <iso-se.png> --prompt "Replace the character with a different character, keeping the exact same high isometric camera angle, facing, pose, size in the frame and art style. The new character is ..."`. Pick a seed with empty hands and no drop shadow.
2. Directions: four more edits of that still (`s`, `e`, `ne`, `n`) with an "exact same character ... but now turned ..." prompt.
3. Pad each still to 1280×1280 with its own corner colour (832×1024 pasted at 224,128) as `<comfy_input>/<kit>-ref-<dir>.png`; key the SE still with `sprite-gen cutout --key measured` for the palette.
4. Write the kit YAML and run the build.

Avoid soft headwear (caps) that hands can touch in back-view attacks; the model takes it off.

## Other commands

- `python -m sprite_pipeline.comfy_run still|edit|i2v|animate ...`: run the ComfyUI graphs from the shell.
- `python -m sprite_pipeline.pixelize --ref <key.png> --scale <s> --colors 40 --out <dir> --set <name>=<frames-dir>`: pixelize keyed frames with a fixed scale and a palette from the reference.
- `blender -b --python src/sprite_pipeline/render_drive.py -- <glb> <Action> <out_dir> <facing> <elev> <repeats> <pre> <post> [head_pitch]`: render one drive (`RW`/`RH` set the size).

## Tests

```bash
uv sync --frozen
env -u PYTHONPATH uv run --frozen python -m pytest
```

No network, GPU or real clips needed.
