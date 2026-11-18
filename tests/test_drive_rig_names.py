"""The drive stage honours the project's rig bone names (sprite-pipeline-21;
spec FR 1 rig defaults, FR 3 "the head lift uses rig.neck and rig.head").

stages.plan_pair appends `--neck <rig.neck> --head <rig.head>` to the
render_drive.py argv only when the animation pitches the head; render_drive.py
pitches those bones and fails loudly when one has no rotation fcurves on the
action. render_drive.py is stdlib + bpy only: it is exec'd by path with stub
`bpy`/`mathutils` modules injected into sys.modules. No Blender.
"""
import io
import runpy
import sys
import tempfile
import types
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from test_workflow_paths import cfg, stubbed_example

from sprite_pipeline import project, stages

RENDER_DRIVE = Path(stages.__file__).resolve().parent / "render_drive.py"
CUSTOM_RIG = "rig: {neck: neck_01, head: head_01}\n"


def _load(tmp, rig_line=None):
    path, _ = stubbed_example(tmp)
    if rig_line is not None:
        path.write_text(path.read_text().replace("output: out\n", "output: out\n" + rig_line, 1))
    proj = project.load_project(path)
    return stages.StageCtx(proj, cfg(Path("/nonexistent-home"))), proj


def _drive_argv(ctx, proj, anim_name, d="s"):
    return ctx.plan_pair("courier", proj.animation(anim_name), d, "drive")[0][0]


class TestDriveArgv(unittest.TestCase):
    """r1-r3: the planned blender argv."""

    def setUp(self):
        self._t = tempfile.TemporaryDirectory()
        self.addCleanup(self._t.cleanup)

    def _expected_head(self, ctx, proj, anim_name, action, repeats, pre, post):
        anim = proj.animation(anim_name)
        return ["blender", "-b", "--python", str(RENDER_DRIVE), "--",
                str(anim.source), action, str(ctx.frames_dir("courier", anim, "s")),
                "0", "45", repeats, pre, post]

    def test_r1_configured_names_follow_the_head_lift(self):
        ctx, proj = _load(self._t.name, CUSTOM_RIG)
        self.assertEqual(proj.rig, {"neck": "neck_01", "head": "head_01"})
        argv = _drive_argv(ctx, proj, "run")
        self.assertEqual(argv, self._expected_head(ctx, proj, "run", "Jog_Fwd_Loop", "2", "1", "0")
                         + ["25", "--neck", "neck_01", "--head", "head_01"])

    def test_r2_default_names_flow_through_stages(self):
        ctx, proj = _load(self._t.name)
        argv = _drive_argv(ctx, proj, "run")
        self.assertEqual(argv, self._expected_head(ctx, proj, "run", "Jog_Fwd_Loop", "2", "1", "0")
                         + ["25", "--neck", "DEF-neck", "--head", "DEF-head"])

    def test_r3_look_down_animation_carries_no_bone_flags(self):
        ctx, proj = _load(self._t.name, CUSTOM_RIG)
        argv = _drive_argv(ctx, proj, "hack")
        # byte-identical to the argv before the rig flags existed: no pitch,
        # no names, even though the project names custom bones
        self.assertEqual(argv, self._expected_head(ctx, proj, "hack", "Fixing_Kneeling", "1", "0", "0"))
        self.assertNotIn("--neck", argv)
        self.assertNotIn("--head", argv)
        self.assertNotIn("neck_01", argv)


# --------------------------------------------------------------- stub Blender

class _Quaternion:
    """Quaternion(axis, angle) is the tilt; Quaternion(values) @ tilt marks
    each component +100 so a pitched keyframe is recognisable."""

    def __init__(self, *a):
        self.v = list(a[0]) if len(a) == 1 else None

    def __matmul__(self, other):
        return [x + 100 for x in self.v]


def _fcurves(bone, keys=2):
    return [SimpleNamespace(data_path=f'pose.bones["{bone}"].rotation_quaternion', array_index=i,
                            keyframe_points=[SimpleNamespace(co=[k, float(i)]) for k in range(keys)],
                            update=lambda: None)
            for i in (3, 1, 0, 2)]


def _run_render_drive(after_dashdash, bones):
    """Exec render_drive.py with stub bpy/mathutils; `bones` are the bones whose
    rotation fcurves exist on the action. Returns (fcurves by bone, bpy stub,
    stdout, stderr, SystemExit or None)."""
    fcs = {b: _fcurves(b) for b in bones}
    action = after_dashdash[1]
    act = SimpleNamespace(name=action, slots=[], frame_range=(0, 2),
                          layers=[SimpleNamespace(strips=[SimpleNamespace(channelbags=[
                              SimpleNamespace(fcurves=[f for b in bones for f in fcs[b]])])])])
    rig = mock.MagicMock(); rig.name = "Rig"; rig.parent = None; rig.rotation_euler = [0.0, 0.0, 0.0]
    bpy = mock.MagicMock()
    bpy.data.objects.__iter__.side_effect = lambda: iter([rig])
    bpy.data.objects.__getitem__.side_effect = {"Rig": rig}.__getitem__
    bpy.data.actions = {action: act}
    mathutils = types.ModuleType("mathutils")
    mathutils.Vector = mock.MagicMock()
    mathutils.Quaternion = _Quaternion
    out, err, exc = io.StringIO(), io.StringIO(), None
    with mock.patch.dict(sys.modules, {"bpy": bpy, "mathutils": mathutils}), \
            mock.patch.object(sys, "argv", ["blender", "-b", "--python", str(RENDER_DRIVE), "--"] + after_dashdash), \
            redirect_stdout(out), redirect_stderr(err):
        try:
            runpy.run_path(str(RENDER_DRIVE), run_name="__main__")
        except SystemExit as e:
            exc = e
    return fcs, bpy, out.getvalue(), err.getvalue(), exc


def _pitched(fcs):
    return all(kp.co[1] >= 100 for f in fcs for kp in f.keyframe_points)


def _untouched(fcs):
    return all(kp.co[1] < 100 for f in fcs for kp in f.keyframe_points)


def _exit_text(exc, err):
    code = exc.code if exc is not None else None
    return (code if isinstance(code, str) else "") + err


class TestRenderDriveBones(unittest.TestCase):
    """r4-r5: render_drive.py resolves the named bones or fails loudly."""

    def setUp(self):
        self._t = tempfile.TemporaryDirectory()
        self.addCleanup(self._t.cleanup)
        ctx, proj = _load(self._t.name, CUSTOM_RIG)
        argv = _drive_argv(ctx, proj, "run")
        # the real producer: the argv stages plans, from after "--"
        self.planned = argv[argv.index("--") + 1:]
        self.assertIn("neck_01", self.planned)

    def test_r4_missing_configured_bone_exits_nonzero_naming_it(self):
        for present, missing in ((["DEF-neck", "DEF-head"], "neck_01"),
                                 (["neck_01", "DEF-head"], "head_01")):
            with self.subTest(missing=missing):
                fcs, bpy, out, err, exc = _run_render_drive(self.planned, present)
                self.assertIsNotNone(exc, "render_drive returned normally on a missing bone")
                self.assertNotIn(exc.code, (0, None))
                text = _exit_text(exc, err)
                self.assertIn(missing, text)
                self.assertIn("Jog_Fwd_Loop", text)
                self.assertIn("rotation_quaternion", text)
                # it stops before touching any keyframe and before rendering
                for b in present:
                    self.assertTrue(_untouched(fcs[b]), b)
                bpy.ops.render.render.assert_not_called()
                self.assertNotIn("RENDERED", out)

    def test_r5_configured_bones_resolve_and_the_render_proceeds(self):
        fcs, bpy, out, err, exc = _run_render_drive(self.planned, ["neck_01", "head_01", "DEF-neck", "DEF-head"])
        self.assertIsNone(exc, err)
        self.assertTrue(_pitched(fcs["neck_01"]))
        self.assertTrue(_pitched(fcs["head_01"]))
        self.assertTrue(_untouched(fcs["DEF-neck"]))
        self.assertTrue(_untouched(fcs["DEF-head"]))
        self.assertEqual(bpy.ops.render.render.call_count, 2 * 2 + 1)  # repeats 2, pre 1, post 0
        self.assertIn("RENDERED 5", out)

    def test_no_flags_default_to_def_bones(self):
        # the manual invocation without --neck/--head keeps the DEF- defaults
        args = [self.planned[i] for i in range(9)]  # glb action out facing elev repeats pre post pitch
        fcs, bpy, out, err, exc = _run_render_drive(args, ["DEF-neck", "DEF-head", "neck_01"])
        self.assertIsNone(exc, err)
        self.assertTrue(_pitched(fcs["DEF-neck"]))
        self.assertTrue(_pitched(fcs["DEF-head"]))
        self.assertTrue(_untouched(fcs["neck_01"]))
        self.assertIn("RENDERED", out)

    def test_no_pitch_never_looks_up_bones(self):
        args = [self.planned[i] for i in range(8)]  # no head_pitch
        fcs, bpy, out, err, exc = _run_render_drive(args, [])
        self.assertIsNone(exc, err)
        self.assertIn("RENDERED", out)


if __name__ == "__main__":
    unittest.main()
