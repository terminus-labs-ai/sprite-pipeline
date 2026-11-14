# sprite-pipeline

Turns one character still into a pixel-art sprite set for a shared motion library. A project file (`project.yaml`) describes the whole job — art style, view and directions, characters and a list of Blender animations — and one command, `sprite-pipeline go project.yaml`, drives a Blender mannequin (the Quaternius Universal Animation Library), motion-transfers it with Wan Animate 2 in ComfyUI, keys and loops with `sprite-gen`, pixelizes at a fixed scale and exports a Godot 4 `SpriteFrames` set. All paths in the project file are relative to it, so the folder copies to another machine as-is.

## Requirements

- **ComfyUI** with the models and nodes the shipped workflows load. `src/sprite_pipeline/workflows/edit.json` (FLUX.2-klein 4B) loads:
  - Models: `flux-2-klein-4b.safetensors` (UNet), `qwen_3_4b.safetensors` (CLIP, flux2), `flux2-vae.safetensors` (VAE)
  - Nodes: `UNETLoader`, `CLIPLoader`, `VAELoader`, `CLIPTextEncode`, `ConditioningZeroOut`, `CFGGuider`, `EmptyFlux2LatentImage`, `Flux2Scheduler`, `KSamplerSelect`, `RandomNoise`, `SamplerCustomAdvanced`, `VAEDecode`, `SaveImage`, `LoadImage`, `ImageScaleToTotalPixels`, `VAEEncode`, `ReferenceLatent`
  `src/sprite_pipeline/workflows/animate.json` (Wan Animate 2 with the lightx2v LoRA) loads:
  - Models: `wan_animate_2_int8_convrot.safetensors` (UNet), `lightx2v_I2V_14B_480p_cfg_step_distill_rank64_bf16.safetensors` (LoRA), `umt5_xxl_fp8_e4m3fn_scaled.safetensors` (CLIP, wan), `clip_vision_h.safetensors` (CLIP Vision), `Wan2_1_VAE_bf16.safetensors` (VAE)
  - Nodes: `UNETLoader`, `LoraLoaderModelOnly`, `ModelSamplingSD3`, `CLIPLoader`, `CLIPVisionLoader`, `VAELoader`, `CLIPTextEncode`, `ImageScale`, `LoadVideo`, `GetVideoComponents`, `CLIPVisionEncode`, `ImageFromBatch`, `WanAnimate2ToVideo`, `KSamplerSelect`, `BasicScheduler`, `SamplerCustom`, `TrimVideoLatent`, `VAEDecode`, `SaveImage`, `LoadImage`
  (ComfyUI 0.3-style custom nodes; ComfyUI setup — installing it, its models and nodes — is manual.)
- **Blender 5.x** (renders the mannequin drives headless).
- **ffmpeg** and **ffprobe** (pack the rendered frames into drive clips).
- **img2webp** (the child tools' extra binary; the config's `extra_path` can point at it).
- **Quaternius Universal Animation Library (CC0)** — the motion library the drives come from; download it from Quaternius' itch.io page and use `Animation Library[Standard]/Godot/AnimationLibrary_Godot_Standard.glb`.
- **Python 3.12+** and [uv](https://docs.astral.sh/uv/) for the install.

## Install

```bash
uv tool install git+https://github.com/terminus-labs-ai/sprite-pipeline@main
```

This installs the `sprite-pipeline` command together with `sprite-gen` (pinned at `v4.3.0`), which the pipeline calls from its own environment for keying (`video-frames`), looping (`video-loop`) and the palette cutout (`cutout`).

## Config

Host locations live in `$SPRITE_PIPELINE_HOME/config.toml` (default `~/.sprite-pipeline/config.toml`). Every key is optional and unknown keys are an error; the config keeps only machine locations:

```toml
comfy_url = "http://127.0.0.1:8188"   # ComfyUI's API server (default shown)
comfy_input = "/path/to/comfyui/input"   # ComfyUI's input/ dir; refs and drive clips go here (default: $SPRITE_PIPELINE_HOME/comfyui/input)
comfy_output = "/path/to/comfyui/output" # ComfyUI's output dir (default: $SPRITE_PIPELINE_HOME/comfyui/output)
blender = "blender"                        # Blender executable (default: blender on PATH)
motion_cache = "/path/to/motion-cache"  # rendered drives, shared by every project (default: $SPRITE_PIPELINE_HOME/motion-cache)
drive_lsr = ""                             # optional Blender file holding the drive scene
extra_path = "/path/to/ffmpeg-bin"      # prepended to PATH for child tools (ffmpeg, ffprobe, img2webp)
```

## Project file

A `project.yaml` describes a whole sprite job. The complete reference in `examples/isometric/project.yaml`; every path in it is relative to the file.

Top level:

- `style` — the art style prompt line (no default; free text).
- `view` — `isometric` (the only view in this increment). The view preset in the package carries the camera prompt line, the face line and the turn prompt per direction.
- `directions.generate` — the list of directions rendered (`n ne e se s sw w nw`); required.
- `directions.mirror` — map of mirrored direction to its generate source (e.g. `{sw: se, w: e, nw: ne}`); the mirror is flipped at export. Default: none.
- `sprite.scale` — the pixelize scale (numeric, no default).
- `sprite.colors` — palette size for the pixelize (no default).
- `sprite.accent_boost` — accent-boost factor for the palette (no default).
- `sprite.fps` — frames per second for the drives and export (no default).
- `sprite.frame_size` — the square drive-render and animation frame size in pixels (no default).
- `naming` — the animation-name pattern, with `{character}`, `{animation}` and `{dir}`; default `{animation}_{dir}`. Must produce unique names.
- `export` — `godot` (the only supported target; default).
- `seed` — ComfyUI seed for edits and animate (default `0`).
- `pose_strength` — pose strength for the animate stage (default `1.0`).
- `output` — output directory, relative to the project file (default `out`).
- `rig.neck` — the rig's neck bone name (default `DEF-neck`, the Quaternius mannequin).
- `rig.head` — the rig's head bone name (default `DEF-head`).
- `workflows.edit` — path to the FLUX.2-klein edit workflow JSON (relative to the project file; default: the packaged `src/sprite_pipeline/workflows/edit.json`).
- `workflows.animate` — path to the Wan Animate 2 animate workflow JSON (relative to the project file; default: the packaged `src/sprite_pipeline/workflows/animate.json`).
- `characters` — list of characters (below).
- `animations` — list of animations (below).

`characters[]` (FR 2):

- `name` — the character name (used in `naming` and output paths).
- `still` — the isometric still of the character, relative to the project file (e.g. `stills/courier.png`).
- `describe` — a one-line description of the character.
- `pronoun` — `he`, `she` or `they` (fills the view's face line and turn prompts).
- `who` — optional pose subject for `{who}` in animation prompts; defaults from the pronoun (`a man`, `a woman`, `a person`).
- `face_dirs` — optional list of directions that face the camera; defaults to the generated directions that face the camera for the view.
- `overrides` — optional map `<animation>-<dir>` → `{seed: N, extra: "<prompt line>"}` for rerolls of a specific pair.

`animations[]` (FR 3):

- `name` — the animation name (used in `naming`).
- `source` — `<file>.glb#<Action>`; the GLB must exist and the action must be a real action in it (e.g. `anims/ual.glb#Jog_Fwd_Loop`).
- `frames` — the animation's cycle length in frames.
- `loop` — whether the loop stage cuts a fixed cycle (`true`) or a single pass (`false`).
- `prompt` — the pose prompt, with `{who}` substituted (e.g. `{who} jogging in place, head up`).
- `start` — first frame of the clip (default `0`).
- `clip_frames` — clip length (must be `4n+1`, ≥ `start + frames` when `loop: true`); default `start + frames` rounded up to the next `4n+1`.
- `repeats` — how many times the drive repeats the cycle (default `1`).
- `pre` — pre-hold frames before the cycle (default `1`).
- `post` — post-hold frames after the cycle (default `0`).
- `look_down` — whether the move looks down (`false` lifts the head 25° via `rig.neck` and `rig.head`; default `false`).
- `state` — sprite-gen state for the loop stage: `idle`, `walk`, `run`, `jump` or `attack` (default `attack`).
- `seam_max` — optional seam-pop cap for the loop stage.
- `directions` — optional subset of `directions.generate`; defaults to all of them.
- `workflow` — optional per-animation override of `workflows.animate` (relative to the project file, like every other path; a missing file fails at the animate stage, naming the path).

Workflow slots:

Each workflow JSON is a ComfyUI API graph. The build fills the nodes titled `sp:<slot>` with the step's values; a slot the step needs but the workflow lacks is an error naming the workflow and the title. The full slot list is `sp:prompt`, `sp:negative`, `sp:pose_prompt`, `sp:seed`, `sp:ref_image`, `sp:drive_video`, `sp:width`, `sp:height`, `sp:length`, `sp:pose_strength` and `sp:save`; the shipped `edit.json` carries `sp:prompt`, `sp:seed`, `sp:ref_image` and `sp:save`, and the shipped `animate.json` carries all eleven.

## Quickstart

The example in `examples/isometric/` carries the current motion library (13 animations, run/punch/sneak tested) and one character entry. Copy it and run:

```bash
cp -r examples/isometric my-game
cd my-game
# drop your own isometric still of your character into stills/courier.png
# and the Quaternius UAL Godot Standard GLB into anims/ual.glb
sprite-pipeline go project.yaml --dry-run   # prints the plan, runs nothing
sprite-pipeline go project.yaml             # prepares the still, makes the four other direction edits
# -> writes out/courier/refs-sheet.png and stops for review
# look at the sheet, then:
sprite-pipeline approve project.yaml courier
sprite-pipeline go project.yaml             # drive, animate, key, loop, pixel, export
```

`go` writes, under `<project dir>/out/<character>/`:

- `master/` — the prepared still, the four direction stills and the keyed SE still (the palette reference).
- `refs-sheet.png` — the direction stills side by side, one per direction (the review artefact).
- `refs.approved` — written by `approve`; holds the hash of the direction stills. A change to any of them (or a `--reroll`) invalidates it and `go` stops at the sheet again.
- `clips/<animation>-<dir>.mp4` — the animated clips.
- `loops/<animation>-<dir>/cycle/` — the looped frames.
- `sprites/` — the pixel frames (one per animation × direction), `atlas.png`, `sprite_frames.tres` (Godot 4 `SpriteFrames`, one animation per `naming` entry) and `contact-sheet.png`.

Narrow a run with `--characters a,b`, `--animations run,punch`, `--dirs s,se`, `--stage drive|animate|key|loop|pixel|export`; reroll one still with `--reroll <char>:<dir>=<seed>`; skip the review stop for one run with `--no-stop`; rerun stages whose output exists with `--force`; preview with `--dry-run`. A failed (animation, direction) is logged and the run continues; the summary ends with exit code 1 when anything failed. Exit code 0 means done or stopped for review (the stop prints the sheet path and the approve command); exit code 2 means the project file is invalid.

## Platforms

Linux with an NVIDIA GPU is tested; Windows is expected to work, is untested, and reports are welcome. (macOS is not tested.)

## License

MIT, see [LICENSE](LICENSE). The code only: models, animation libraries and generated art keep their own licences.
