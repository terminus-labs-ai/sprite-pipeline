"""Render a Quaternius UAL action on its mannequin from an isometric camera. Args after --: glb action out_dir facing_deg elev_deg repeats pre post [head_pitch_deg]. head_pitch > 0 lifts the chin (split between neck and head)."""
import bpy, math, sys
from mathutils import Vector
args = sys.argv[sys.argv.index("--") + 1:]
glb, action, out, facing, elev, repeats, pre, post = args[:8]
head_pitch = float(args[8]) if len(args) > 8 else 0.0
facing, elev, repeats, pre, post = float(facing), float(elev), int(repeats), int(pre), int(post)
bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.import_scene.gltf(filepath=glb)
for o in list(bpy.data.objects):
    if o.name.startswith("Icosphere"):
        bpy.data.objects.remove(o)
rig = bpy.data.objects["Rig"]
act = bpy.data.actions[action]
rig.animation_data_create(); rig.animation_data.action = act
if hasattr(rig.animation_data, "action_slot") and act.slots:
    rig.animation_data.action_slot = act.slots[0]
def pitch_bone(act, bone, deg):
    from mathutils import Quaternion
    fcs = [f for f in act.layers[0].strips[0].channelbags[0].fcurves if f.data_path == f'pose.bones["{bone}"].rotation_quaternion']
    fcs.sort(key=lambda f: f.array_index)
    if len(fcs) != 4: return
    tilt = Quaternion((1, 0, 0), math.radians(deg))
    for k in range(len(fcs[0].keyframe_points)):
        q = Quaternion([f.keyframe_points[k].co[1] for f in fcs]) @ tilt
        for i, f in enumerate(fcs): f.keyframe_points[k].co[1] = q[i]
    for f in fcs: f.update()
if head_pitch:
    pitch_bone(act, "DEF-neck", -head_pitch / 2); pitch_bone(act, "DEF-head", -head_pitch / 2)
root = rig
while root.parent: root = root.parent
root.rotation_mode = "XYZ"; root.rotation_euler[2] += math.radians(facing)
a0, a1 = [int(x) for x in act.frame_range]
n = a1 - a0
sc = bpy.context.scene
# timeline: hold first pose `pre` frames, play `repeats` times, hold last pose `post` frames
frames = [a0] * pre + [a0 + (i % n) for i in range(n * repeats)] + [a1] * post
if repeats > 1: frames = [a0] * pre + [a0 + (i % n) for i in range(n * repeats)] + [a0] * post
cam_d = bpy.data.cameras.new("cam"); cam_d.type = "ORTHO"; cam_d.ortho_scale = 3.0
cam = bpy.data.objects.new("cam", cam_d); sc.collection.objects.link(cam); sc.camera = cam
target = Vector((0, 0, 0.92)); dist = 10
cam.location = target + Vector((0, -dist * math.cos(math.radians(elev)), dist * math.sin(math.radians(elev))))
cam.rotation_euler = (target - cam.location).to_track_quat("-Z", "Y").to_euler()
sc.render.engine = "BLENDER_WORKBENCH"
sh = sc.display.shading; sh.light = "STUDIO"; sh.color_type = "SINGLE"; sh.single_color = (0.75, 0.75, 0.78)
sh.show_cavity = True; sh.show_shadows = False
sc.world = bpy.data.worlds.new("w"); sc.world.color = (0.35, 0.35, 0.37)
sh.background_type = "WORLD" if hasattr(sh, "background_type") else sh.background_type
import os; sc.render.resolution_x, sc.render.resolution_y = int(os.environ.get("RW", 576)), int(os.environ.get("RH", 704))
sc.render.film_transparent = False
sc.render.image_settings.file_format = "PNG"
for i, f in enumerate(frames):
    sc.frame_set(f)
    sc.render.filepath = f"{out}/f{i:04d}.png"
    bpy.ops.render.render(write_still=True)
print("RENDERED", len(frames))
