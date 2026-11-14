# Agent instructions: sprite-pipeline

- Full suite: `uv sync --frozen && env -u PYTHONPATH uv run --frozen python -m pytest`.
- Mandates: the fleet repository conventions (installable from GitHub with no sibling checkout, committed `uv.lock`, no host or user paths in code; host locations come from `~/.sprite-pipeline/config.toml`, see README).
- `sprite-gen` is a pinned package dependency (`v4.3.0` in `pyproject.toml`); its commands run through the console script installed next to `sys.executable` — the running interpreter environment's own entry point — never through a configured path.
- `render_drive.py` runs inside Blender's Python (`bpy`, `mathutils`): stdlib only, no package imports.
- The ComfyUI graphs are the shipped workflow JSON files (`src/sprite_pipeline/workflows/edit.json`, `animate.json`); the build fills nodes titled `sp:<slot>` and a slot the step needs but the workflow lacks is an error.
- No stage shells out to POSIX-only commands such as `cp`, `mv` or `rm`; file copies are in-process Python steps.
- Never commit generated art, clips or ComfyUI outputs; they belong to the consuming game's data directory.
