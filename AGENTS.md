# Agent instructions: sprite-pipeline

- Full suite: `uv sync --frozen && env -u PYTHONPATH uv run --frozen python -m pytest`.
- Mandates: the fleet repository conventions (installable from GitHub with no sibling checkout, committed `uv.lock`, no host or user paths in code; host locations come from `config.toml`, see README).
- `render_drive.py` runs inside Blender's Python (`bpy`, `mathutils`): stdlib only, no package imports.
- `motions.yaml` changes that alter a drive change its cache hash; existing kits then re-render that drive.
- Never commit generated art, clips or ComfyUI outputs; they belong to the consuming game's data directory.
