# Isometric example

The starting point for a sprite job. Copy the folder, then drop in your own
assets:

- `stills/courier.png` — one isometric still of your character facing the
  lower right (toward the camera), on a plain flat background. The other
  direction stills are made from it by the edit workflow.
- `anims/ual.glb` — the Quaternius Universal Animation Library (CC0),
  `AnimationLibrary_Godot_Standard.glb`, renamed to `ual.glb`; the animation
  sources in `project.yaml` reference it by `#<Action>`.

Then run `sprite-pipeline go project.yaml`.
