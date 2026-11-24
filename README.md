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
- **Blender 5.x** (renders the mannequin drives headless, and the mannequin A-pose that `sprite-pipeline mannequin-profile` measures).
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
- `seed` — ComfyUI seed for edits and animate (default `0`); same as `animate.seed`, set only one.
- `pose_strength` — pose strength for the animate stage (default `1.0`); same as `animate.pose_strength`, set only one.
- `drive` — the drive render: `background` (flat `[r, g, b]` key colour, default `[30, 202, 24]`), `render_size` (square Blender render px, default `1152`), `scale` (frame scale when an animation sets no `drive_scale`, default `0.605`), `margin` (auto-canvas margin over the union body box, default `1.3`) and `aspect` (auto-canvas `[w, h]`, default `[16, 9]`). The canvas is automatic: the union box of the scaled mannequin over every frame of every direction, times `margin`, widened to `aspect`, rounded up to multiples of 16.
- `reference.height_ratio` — the placed reference's body height over the scaled mannequin's first-frame body height (default `1.07`).
- `animate` — values for the animate workflow's `sp:<key>` nodes: `unet`, `lora` (`none` removes the `sp:lora` node), `lora_strength`, `sampler`, `scheduler`, `steps`, `shift`, `cfg`, `seed`, `pose_strength`, `pose_start_percent`, `pose_end_percent`, `reference_image_strength`, `negative` (default `camera movement, zoom in, zoom out, pan, tilt, tracking shot, cropped limbs`) and `pose_prompt` (default: the animation's `prompt`). A key left out leaves the workflow's own value; a key set for a workflow without its slot is an error.
- `output` — output directory, relative to the project file (default `out`).
- `rig.neck` — the rig's neck bone name (default `DEF-neck`, the Quaternius mannequin).
- `rig.head` — the rig's head bone name (default `DEF-head`). A drive that lifts the head passes both names to `render_drive.py` as `--neck NAME --head NAME` after the head pitch (without them it uses `DEF-neck` and `DEF-head`); a named bone with no rotation keys on the action fails the drive with an error naming the bone. Drives cached by earlier versions for a rig with other names were rendered without the head lift; delete that project's drives (`<character>-<animation>-<dir>-*`) from `motion_cache` to render them again.
- `workflows.edit` — path to the FLUX.2-klein edit workflow JSON (relative to the project file; default: the packaged `src/sprite_pipeline/workflows/edit.json`).
- `workflows.animate` — path to the Wan Animate 2 animate workflow JSON (relative to the project file; default: the packaged `src/sprite_pipeline/workflows/animate.json`).
- `characters` — list of characters (below).
- `animations` — list of animations (below).

`characters[]` (FR 2):

- `name` — the character name (used in `naming` and output paths).
- `still` — the character's still, relative to the project file (e.g. `stills/courier.png`): an A- or T-pose standing figure facing front on a flat background (see [Stills](#stills)).
- `describe` — a one-line description of the character.
- `pronoun` — `he`, `she` or `they` (fills the view's face line and turn prompts).
- `who` — optional pose subject for `{who}` in animation prompts; defaults from the pronoun (`a man`, `a woman`, `a person`).
- `face_dirs` — optional list of directions that face the camera. Omit it to get the view's camera-facing directions (declared as `face_dirs` in `views/<view>.yaml`) that are in `directions.generate`; an explicit `face_dirs: []` means none (no face line).
- `height_scale` — optional map direction → pixelize factor (default `1.0` for any direction not listed); mirrored directions use their source's factor.
- `describe` may be one string for every direction or a map with one entry per generated direction (`{s: ..., se: ..., e: ..., ne: ..., n: ...}`), so each view can say what shows from that side (no face details on back views, for example). Mirrored directions reuse their source's entry. Every direction also gets the view's `facing` line from `views/<view>.yaml`, which keeps body, head and gaze pointing the same way.
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
- `drive_scale` — optional; overrides `drive.scale` for this animation.
- `pixel_scale` — optional pixelize scale (default `sprite.scale / drive scale`).
- `facing_line` — whether the animate prompt carries the view's facing line (default `true`).
- `animate` — optional block overriding the project `animate:` keys for this animation.
- `canvas` — removed; the canvas is automatic, and setting it is an error.
- `workflow` — optional per-animation override of `workflows.animate` (relative to the project file, like every other path; a missing file fails at the animate stage, naming the path).

Workflow slots:

Each workflow JSON is a ComfyUI API graph. The build fills the nodes titled `sp:<slot>` with the step's values; a slot the step needs but the workflow lacks is an error naming the workflow and the title. The shipped `animate.json` carries `sp:prompt`, `sp:negative`, `sp:pose_prompt`, `sp:seed`, `sp:ref_image`, `sp:drive_video`, `sp:width`, `sp:height`, `sp:length`, `sp:pose_strength`, `sp:save`, `sp:unet`, `sp:sampler`, `sp:scheduler`, `sp:steps`, `sp:shift`, `sp:cfg`, `sp:pose_start_percent`, `sp:pose_end_percent` and `sp:reference_image_strength`, with the distilled Wan Animate 2 model and no LoRA node (`lcm`, `simple`, 10 steps, shift 5, cfg 1.5, pose strength 0.85, seed 7). A workflow with a LoRA node titles it `sp:lora,sp:lora_strength`. The shipped `edit.json` carries five: `sp:prompt`, `sp:seed`, `sp:ref_image`, `sp:pose_image` (the direction guide, encoded like `sp:ref_image` and chained as a second reference on both conditionings) and `sp:save`; `sp:pose_image` is the twelfth slot, which `animate.json` lacks.

## Stills

A character's still must be an **A- or T-pose standing figure on a flat background**. `go` measures the figure as the bounding box of the pixels that differ from the top-left corner colour by more than 40 in any channel, and accepts it as a standing figure when the box is at least 20% of the image height, at least as tall as it is wide, and its bottom lies in the lower 20% of the image. Otherwise `go` exits 2 naming the file and the failed rule (`too wide`, `too small` or `feet not at the bottom`) before it writes anything.

The tool anchors the still to the mannequin. Wan Animate follows the Blender mannequin of the drive video, and a reference whose figure is a different size or stands at a different height than that mannequin drifts in size across animations. So `go` scales the still uniformly until its measured body height equals the front-facing (`s`) mannequin's on the 832×1024 edit canvas (`body_h × 832 / frame_size`: 391 px for the `ual-female.glb` mannequin at the default `frame_size` 704), pins its feet to that mannequin's floor line (`floor_y × 832 / frame_size + 96`: 715 for that mannequin at 704), centres it horizontally and fills every other pixel with flat chroma green `(31, 203, 25)`. The 832 and 96 come from the packaged animate workflow, which centre-crops the 832×1024 reference to 832×832 (rows 96–927) and scales it to `frame_size`. After the animate step's centre crop and downscale, the reference has exactly the drive video's framing: its figure measures the mannequin's body height with its feet on the drive floor line. This anchored still (`master/still.png`) is the input of the `s` direction edit.

Each generated direction — `s` first — is guided by the mannequin. `go` renders the first-listed animation's GLB in its mannequin A-pose — the `A_TPose` action with both upper arms lowered 45° (`--arm-drop 45`) — facing that direction with the drive camera (Blender; the render is cached in `motion_cache`, so later runs and other characters reuse it), and places its silhouette on the edit canvas at that direction's own mannequin size, floor line and centre as `master/<d>.guide.png`. The edit gets the reference as image 1 and the guide as image 2, with a prompt line asking for the mannequin's pose, facing, size and position. Its raw output is kept as `master/<d>.edit.png` and re-anchored to `master/<d>.png`: the same scale-and-paste as the anchored still, with that direction's target, and no standing-figure rule (the guided directions take the mannequin's near-T-pose arms, which can make the figure wider than tall). A raw edit with no figure, or a failed guide render, fails that direction; the other directions still run and `go` exits 1 — except for `s`: when its guide, edit or re-anchor fails, the palette cutout does not run and no other direction is touched.

The `s` edit reads `master/still.png`; its re-anchored output `master/s.png` is the guided master every other generated direction is edited from and, keyed as `master/s.keyed.png`, the palette reference. When the project generates directions besides `s`, `go` stops after the `s` edit: it writes `s-sheet.png` (anchored still, `s` guide, guided `s` side by side) and waits for `approve`, which writes `s.approved` (the sha256 of `master/s.png`). Only then does `go` edit the other directions and reach the full review stop, where `approve` writes `refs.approved`. Redoing `s` — a `--reroll` of it, or `--force` — deletes the other direction stills: they are edited again from the new `s` once it is approved. `--no-stop` skips both stops for one run.

The measured body height includes anything carried on the body: a headdress, a raised weapon or a held item makes the box taller than the body, so the figure comes out slightly smaller than the mannequin. The review stop covers this: each direction's size ratio (the measured body height of its raw edit `master/<d>.edit.png`, or of `master/<d>.png` when there is no raw edit, ÷ that direction's mannequin target; the master `s` against the `s` target) is printed as `[<character>/<dir>] ref size <ratio>x` and on the sheet label, and a ratio outside ±15% (below 0.85 or above 1.15) adds `CHECK`. The report is advisory; `approve` succeeds regardless.

## Mannequin profiles

The anchor needs the mannequin's body height and floor line, measured once per source GLB and per isometric direction (`s`, `se`, `e`, `ne`, `n`): under the drive camera the mannequin's height differs by direction. The package ships profiles for the Quaternius UAL mannequins (`ual.glb` and `ual-female.glb`), keyed by the sha256 of the GLB bytes. The first-listed animation's GLB anchors the project's stills, and every distinct source GLB in the project must have a profile carrying every generated direction: a GLB with none, or whose profile lacks a generated direction (such as a profile from before per-direction profiles), makes `go` exit 2, printing its sha256 prefix and the command that measures it.

```bash
sprite-pipeline mannequin-profile anims/ual.glb
```

renders the GLB's mannequin A-pose (its `A_TPose` action with both upper arms lowered 45° via `render_drive.py --arm-drop 45`) at 704 px with the drive camera (elevation 45°) facing each of `s`, `se`, `e`, `ne` and `n`, measures the body height and floor line of each, prints the measured profile as JSON (`{"glb_name": ..., "resolution": 704, "dirs": {"s": {"body_h": ..., "floor_y": ...}, ...}}`) and writes it to the user profile file `~/.sprite-pipeline/mannequin_profiles.json` (`$SPRITE_PIPELINE_HOME/mannequin_profiles.json` when that is set). If any direction's render fails, nothing is written. Re-running it on the same GLB replaces that GLB's entry; entries for other GLBs stay. The user file merges over the packaged profiles, last write wins per hash. The render needs Blender and the host config ([Config](#config)); a different download or export of the library has different bytes and needs its own profile.

## Quickstart

The example in `examples/isometric/` carries the current motion library (13 animations, run/punch/sneak tested) and one character entry. Copy it and run:

```bash
cp -r examples/isometric my-game
cd my-game
# drop your character's still (A/T-pose, standing, facing front, flat
# background) into stills/courier.png and the Quaternius UAL Godot Standard
# GLB into anims/ual.glb; if `go` reports no mannequin profile for it, run:
#   sprite-pipeline mannequin-profile anims/ual.glb
sprite-pipeline go project.yaml --dry-run   # prints the plan, runs nothing
sprite-pipeline go project.yaml             # anchors the still, makes the guided s edit
# -> writes out/courier/s-sheet.png and stops for review
# look at the sheet, then:
sprite-pipeline approve project.yaml courier
sprite-pipeline go project.yaml             # edits the other four directions from s
# -> writes out/courier/refs-sheet.png and stops for review again
sprite-pipeline approve project.yaml courier
sprite-pipeline go project.yaml             # drive, animate, key, loop, pixel, export
```

`go` writes, under `<project dir>/out/<character>/`:

- `master/` — the anchored still `still.png` (832×1024 on flat chroma green), the guided master `s.png` and the re-anchored direction stills (`se.png`, `e.png`, …), each one's guide (`<d>.guide.png`) and raw edit (`<d>.edit.png`), the keyed master `s.keyed.png` (the palette reference) and `source.sha256` (the fingerprint of the source still; changing the still rebuilds the character).
- `s-sheet.png` — the anchored still, the `s` guide and the guided `s` side by side (the `s` stop's review artefact).
- `s.approved` — written by `approve` at the `s` stop; holds the sha256 of `master/s.png`. Redoing `s` (or changing its bytes) invalidates it and `go` stops at the `s` sheet again before the other directions are edited.
- `refs-sheet.png` — the direction stills side by side, one per direction, each label carrying its size ratio (the review artefact).
- `refs.approved` — written by `approve`; holds the hash of the direction stills (`<d>.png`; the guides and raw edits are not in it). A change to any of them (or a `--reroll`) invalidates it and `go` stops at the sheet again.
- `clips/<animation>-<dir>.mp4` — the animated clips.
- `loops/<animation>-<dir>/cycle/` — the looped frames.
- `sprites/` — the pixel frames (one per animation × direction), `atlas.png`, `sprite_frames.tres` (Godot 4 `SpriteFrames`, one animation per `naming` entry) and `contact-sheet.png`.

Narrow a run with `--characters a,b`, `--animations run,punch`, `--dirs s,se`, `--stage drive|animate|key|loop|pixel|export`; reroll one still with `--reroll <char>:<dir>=<seed>` (redoes that direction's edit and re-anchor only — a reroll of `s` also deletes the other direction stills); skip both review stops for one run with `--no-stop`; rerun stages whose output exists with `--force`; preview with `--dry-run`. A failed (animation, direction) is logged and the run continues; the summary ends with exit code 1 when anything failed. Exit code 0 means done or stopped for review (the stop prints the sheet path and the approve command); exit code 2 means the project file is invalid (including a still that is not a standing figure and a source GLB with no mannequin profile).

## Platforms

Linux with an NVIDIA GPU is tested; Windows is expected to work, is untested, and reports are welcome. (macOS is not tested.)

## License

MIT, see [LICENSE](LICENSE). The code only: models, animation libraries and generated art keep their own licences.
