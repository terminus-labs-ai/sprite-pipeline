"""The mannequin A-pose (amendment 13, AC21: FR 14 `--arm-drop`).

`render_mannequin` (the FR 7 guide and `mannequin-profile`) plans
render_drive.py with `--arm-drop 45` (the package constant
`stages.MANNEQUIN_ARM_DROP`); its drive cache key therefore differs from the
same action's key without the drop. No animation's drive plan carries the
flag and every animation's drive key equals its value before amendment 13.
render_drive.py with `--arm-drop 45` (stub bpy) multiplies every
`DEF-upper_arm.L` rotation key by a -45 deg rotation about the bone's local
Z axis and every `DEF-upper_arm.R` key by +45 deg, leaving every other
bone's keys untouched; without the flag it touches no arm key; an action
without rotation fcurves on either upper-arm bone makes it exit naming that
bone. No Blender.
"""
import dataclasses
import io
import math
import runpy
import sys
import tempfile
import types
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import mannequins
from test_hygiene import fixture
from test_workflow_paths import cfg

from sprite_pipeline import cli, config, metrics, project, stages

RENDER_DRIVE = Path(stages.__file__).resolve().parent / "render_drive.py"
ALL_BONES = ["DEF-neck", "DEF-head", "DEF-upper_arm.L", "DEF-upper_arm.R",
             "DEF-forearm.L", "DEF-forearm.R", "DEF-thigh.L", "DEF-thigh.R"]


def _mannequin_proj(glb_bytes=b"mannequin glb bytes"):
    """The stand-in project `cli.render_mannequin` builds its plan on, with
    a real source file (frames_dir reads its bytes)."""
    import dataclasses

    src = Path(tempfile.mkdtemp()) / "ual-female.glb"
    src.write_bytes(glb_bytes)
    proj = project.Project(path=Path.cwd(), style="", view="isometric",
                           generate=["s"], mirror={},
                           sprite=project.Sprite(scale=1.0, colors=40,
                                                 accent_boost=3, fps=30,
                                                 frame_size=704),
                           naming="{animation}_{dir}", export="godot",
                           seed=0, pose_strength=1.0, output=Path("out"),
                           rig={"neck": "DEF-neck", "head": "DEF-head"},
                           workflows={}, characters=[], animations=[])
    anim = project.Animation(name="A_TPose", source=src, action="A_TPose",
                             frames=1, loop=False, prompt="",
                             arm_drop=stages.MANNEQUIN_ARM_DROP)
    return dataclasses.replace(proj, animations=[anim]), anim, glb_bytes


class TestMannequinPlan(unittest.TestCase):
    """r1, r2, r3, r4: the plan and the drive cache key."""

    def setUp(self):
        self._t = tempfile.TemporaryDirectory()
        self.addCleanup(self._t.cleanup)
        self.tmp = Path(self._t.name)

    def test_r1_mannequin_plans_arm_drop_45(self):
        proj, anim, _ = _mannequin_proj()
        self.assertEqual(stages.MANNEQUIN_ARM_DROP, 45)
        self.assertEqual(anim.arm_drop, stages.MANNEQUIN_ARM_DROP)
        argv = stages.StageCtx(proj, cfg(Path("/nonexistent-home"))).plan_pair(
            "mannequin", anim, "s", "drive")[0][0]
        self.assertEqual(argv[argv.index("--arm-drop") + 1], "45")

    def test_r1_render_mannequin_plans_the_drop(self):
        """The real producer of the FR 7 guide render: `cli.render_mannequin`
        runs `render_drive.py` through `plan_pair` with the arm drop in its
        argv (subprocess.run faked; the planned argv, not a stubbed plan, is
        what the fake inspects)."""
        import subprocess

        calls = []

        class Res:
            returncode = 0
            stdout = ""
            stderr = ""

        def fake_run(argv, **kw):
            argv = [str(a) for a in argv]
            calls.append(argv)
            if argv[0] == "blender":
                rest = argv[argv.index("--") + 1:]
                frames = Path(rest[2])
                frames.mkdir(parents=True, exist_ok=True)
                mannequins.tpose_render().save(frames / "f0001.png")
            return Res()

        old_run = subprocess.run
        subprocess.run = fake_run
        self.addCleanup(setattr, subprocess, "run", old_run)
        home = self.tmp / "home"
        glb = self.tmp / "ual-female.glb"
        glb.write_bytes(b"stub glb bytes")
        old_argv = list(sys.argv)
        self.addCleanup(setattr, sys, "argv", old_argv)
        sys.argv = ["sprite-pipeline"]  # Project(path=Path.cwd())
        with mock.patch.object(config, "load", return_value=config.defaults(home)):
            shot = cli.render_mannequin(glb, "s")
        blender = [c for c in calls if c[0] == "blender"]
        self.assertEqual(len(blender), 1, calls)
        rest = blender[0][blender[0].index("--") + 1:]
        self.assertEqual(rest[1], "A_TPose")
        i = rest.index("--arm-drop")
        self.assertEqual(rest[i + 1], "45")
        # the drop rides after the positional head-pitch slot: the argv is
        # otherwise the plain drive argv (no head pitch on the mannequin)
        self.assertEqual(rest[:8], [str(glb), "A_TPose", rest[2], "0", "45",
                                    "1", "1", "0"])
        # the plan's head-lift rule applies: A_TPose is not look_down, so
        # the pitch rides, and the drop follows it
        self.assertEqual(rest[8:], ["25", "--neck", "DEF-neck",
                                    "--head", "DEF-head", "--arm-drop", "45",
                                    "res=1152", "ortho=4.909091"])
        self.assertTrue(Path(shot).is_file())

    def test_r2_drive_key_differs_with_the_drop(self):
        proj, anim, glb_bytes = _mannequin_proj()
        with_drop = proj.drive_key(anim, glb_bytes)
        without = proj.drive_key(
            dataclasses.replace(anim, arm_drop=0), glb_bytes)
        self.assertNotEqual(with_drop, without)
        # the key without the drop is the pre-amendment-13 key, computed
        # independently over the pre-amendment-13 payload (a drop-only
        # change to the payload cannot reproduce it)
        self.assertEqual(without, "aed88cd035dab391")

    def test_r3_animation_plans_carry_no_arm_drop(self):
        with tempfile.TemporaryDirectory() as t:
            path = fixture(t) / "project.yaml"
            proj = project.load_project(path)
            self.assertTrue(proj.animations)
            ctx = stages.StageCtx(proj, cfg(Path("/nonexistent-home")))
            for anim in proj.animations:
                for d in ("s", "se"):
                    argv = ctx.plan_pair("courier", anim, d, "drive")[0][0]
                    self.assertNotIn("--arm-drop", argv,
                                     f"{anim.name}/{d}")

    def test_r4_drive_key_equals_the_pre_amendment_13_value(self):
        """r4: every animation's drive key equals the hash of the exact
        pre-amendment-13 payload (no `arm_drop` member; death-repro FR 4's
        margin, aspect and directions). The keys were captured at the base commit
        e142de5 for the test_hygiene fixture (examples/isometric with its
        stills and GLB stubbed); test_hygiene pins the same keys."""
        with tempfile.TemporaryDirectory() as t:
            path = fixture(t) / "project.yaml"
            proj = project.load_project(path)
            glb = proj.animations[0].source.read_bytes()

            def pre_amendment_13_key(p, anim, glb_bytes):
                import hashlib
                import json

                payload = json.dumps(
                    {"glb": hashlib.sha256(glb_bytes).hexdigest(),
                     "action": anim.action, "repeats": anim.repeats,
                     "pre": anim.pre, "post": anim.post,
                     "head_lift": anim.head_lift,
                     "rig": {"neck": p.rig["neck"], "head": p.rig["head"]},
                     "elev": stages.DRIVE_ELEV,
                     "frame_size": p.sprite.frame_size,
                     "background": list(p.drive.background),
                     "render_size": p.drive.render_size,
                     "drive_scale": p.drive_scale(anim),
                     # death-repro FR 4: the auto canvas's inputs
                     "margin": 1.3, "aspect": [16, 9],
                     "directions": ["s", "se", "e", "ne", "n"]},
                    sort_keys=True)
                return hashlib.sha256(payload.encode()).hexdigest()[:16]

            pinned = {  # Project.drive_key per animation with the death-repro drive and canvas values
                "run": "5a151461d48db9b8", "punch": "36f3951e9daa8226",
                "idle": "e714381b915861b9", "walk": "24080037937a1790",
                "sneak": "9a2b0c0e9226f25d", "crouch_idle": "aecf7bd49b03076a",
                "hack": "66610f0437096705", "interact": "69ef6921df7bf8a1",
                "jab": "178f1683d0c8dd7b", "pistol_shoot": "e52c97ea3d1c84cb",
                "hit": "797f9bd98f427199", "death": "0da631f08769fd20",
                "roll": "1214ea51bf6f15a7",
            }
            self.assertEqual(set(pinned), {a.name for a in proj.animations})
            for anim in proj.animations:
                self.assertEqual(proj.drive_key(anim, glb), pinned[anim.name],
                                 anim.name)
                self.assertEqual(proj.drive_key(anim, glb),
                                 pre_amendment_13_key(proj, anim, glb),
                                 anim.name)


# --------------------------------------------------------------- stub Blender

class _Quaternion:
    """The stub quaternion: `Quaternion(axis, angle)` is a tilt (kept as
    (axis, angle), applied to a key vector by rotation about `axis`);
    `Quaternion(values)` is a key vector. The result marks each modified
    component +100 so a dropped keyframe is recognisable, the way the
    tilt-marking stub in test_drive_rig_names.py does for the head pitch."""

    def __init__(self, *a):
        if len(a) == 2:  # (axis, angle): a tilt
            self.tilt = (tuple(a[0]), a[1])
            self.v = None
        else:  # (values,): a key vector
            self.tilt = None
            self.v = list(a[0])

    def __matmul__(self, other):
        if other.tilt is not None:
            # pitch_bone writes each fcurve's real w[i] into slot i of this
            # stub vector, so the stub vector's component i IS the marker for
            # the real w[i]: a tilt about axis a moves w[a], so mark slot a.
            # (Z tilt -> slot 3, the w component; X tilt -> slot 0, the x
            # component — the way the real rotation changes those slots.)
            axis, angle = other.tilt
            v = list(self.v)
            if angle != 0:
                for i, a in enumerate(axis):
                    if a:
                        v[i] += 100.0
            return v
        # vector @ vector: not produced by pitch_bone; the stub never takes
        # this path
        raise TypeError("stub quaternion: vector @ vector is not supported")


def _fcurves(bone, keys=2):
    return [SimpleNamespace(data_path=f'pose.bones["{bone}"].rotation_quaternion',
                            array_index=i,
                            keyframe_points=[SimpleNamespace(co=[k, 0.0])
                                             for k in range(keys)],
                            update=lambda: None)
            for i in (3, 1, 0, 2)]


def _run_render_drive(after_dashdash, bones, action="A_TPose"):
    """Exec render_drive.py with stub bpy/mathutils; `bones` are the bones
    whose rotation fcurves exist on the action. Returns (fcurves by bone,
    stdout, stderr, SystemExit or None)."""
    fcs = {b: _fcurves(b) for b in bones}
    # one channelbag per bone: its fcurves list holds that bone's four
    # channels. (One shared list for every bag would let a bone's keys show
    # up as another bone's.)
    bags = [SimpleNamespace(fcurves=list(fcs[b])) for b in bones]
    act = SimpleNamespace(name=action, slots=[], frame_range=(0, 2),
                          layers=[SimpleNamespace(strips=[SimpleNamespace(channelbags=bags)])])
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
    return fcs, out.getvalue(), err.getvalue(), exc


def _dropped(fcs):
    """True when any keyframe of any of the bone's channels was modified
    (the stub marks a modified component +100; co[0] is the frame)."""
    return any(kp.co[1] >= 100 for f in fcs for kp in f.keyframe_points)


def _untouched(fcs):
    """True when no keyframe of any of the bone's channels was modified."""
    return not _dropped(fcs)


def _exit_text(exc, err):
    code = exc.code if exc is not None else None
    return (code if isinstance(code, str) else "") + err


class TestRenderDriveArmDrop(unittest.TestCase):
    """r5, r6, r7, r8: render_drive.py with the stub bpy."""

    PLANNED = ["mannequin.glb", "A_TPose", "frames", "0", "45", "1", "1", "0"]

    def test_r5_arm_drop_multiplies_the_upper_arm_keys_only(self):
        fcs, out, err, exc = _run_render_drive(
            ["run.glb", "Jog_Fwd_Loop", "frames", "0", "45", "1", "1", "0",
             "25", "--arm-drop", "45"],
            ALL_BONES, action="Jog_Fwd_Loop")
        self.assertIsNone(exc, err)
        self.assertIn("RENDERED", out)
        # both arms dropped: a Z-axis rotation (L by -45 deg, R by +45 deg,
        # per the plan) touches every key of both upper-arm bones
        for bone in ("DEF-upper_arm.L", "DEF-upper_arm.R"):
            with self.subTest(bone=bone):
                self.assertTrue(_dropped(fcs[bone]))
        # no other bone is touched (the neck and head are touched only by
        # the pitch — `test_r5_arm_drop_with_the_head_pitch` proves that)
        for bone in ("DEF-forearm.L", "DEF-forearm.R", "DEF-thigh.L", "DEF-thigh.R"):
            with self.subTest(bone=bone):
                self.assertTrue(_untouched(fcs[bone]))

    def test_r5_arm_drop_rotates_about_local_z_with_the_planned_signs(self):
        # the tilt applied to each upper arm is about the bone's local Z axis,
        # L by -45 deg and R by +45 deg (a local X tilt raises one arm)
        tilts = []

        class _Recording(_Quaternion):
            def __init__(self, *a):
                super().__init__(*a)
                if self.tilt is not None:
                    tilts.append(self.tilt)

        with mock.patch(f"{__name__}._Quaternion", _Recording):
            fcs, out, err, exc = _run_render_drive(self.PLANNED + ["--arm-drop", "45"],
                                                   ALL_BONES)
        self.assertIsNone(exc, err)
        self.assertEqual(sorted(set(tilts)),
                         [((0, 0, 1), math.radians(-45)), ((0, 0, 1), math.radians(45))])

    def test_r5_arm_drop_with_the_head_pitch(self):
        # the mannequin render's actual shape: pitch and drop together —
        # both the neck/head and the upper arms are modified
        fcs, out, err, exc = _run_render_drive(self.PLANNED + ["25", "--arm-drop", "45"],
                                               ALL_BONES)
        self.assertIsNone(exc, err)
        self.assertIn("RENDERED", out)
        for bone in ("DEF-upper_arm.L", "DEF-upper_arm.R", "DEF-neck", "DEF-head"):
            with self.subTest(bone=bone):
                self.assertTrue(_dropped(fcs[bone]))

    def test_r6_without_arm_drop_no_arm_key_is_touched(self):
        fcs, out, err, exc = _run_render_drive(
            ["run.glb", "Jog_Fwd_Loop", "frames", "0", "45", "1", "1", "0"],
            ALL_BONES, action="Jog_Fwd_Loop")
        self.assertIsNone(exc, err)
        self.assertIn("RENDERED", out)
        for bone in ALL_BONES:
            self.assertTrue(_untouched(fcs[bone]), bone)

    def test_r7_missing_upper_arm_fcurves_exit_names_the_bone(self):
        for missing in ("DEF-upper_arm.L", "DEF-upper_arm.R"):
            with self.subTest(missing=missing):
                present = [b for b in ALL_BONES if b != missing]
                fcs, out, err, exc = _run_render_drive(
                    ["run.glb", "Jog_Fwd_Loop", "frames", "0", "45", "1", "1", "0",
                     "--arm-drop", "45"],
                    present, action="Jog_Fwd_Loop")
                self.assertIsNotNone(exc, "render_drive returned normally "
                                          "on a missing arm bone")
                self.assertNotIn(exc.code, (0, None))
                text = _exit_text(exc, err)
                self.assertIn(missing, text)
                self.assertIn("Jog_Fwd_Loop", text)
                # it stops before rendering; the other arm bone, when present,
                # may already have been dropped — that is the order, not a leak
                self.assertNotIn("RENDERED", out)

    def test_r8_missing_value_exit_names_the_flag(self):
        fcs, out, err, exc = _run_render_drive(
            ["run.glb", "Jog_Fwd_Loop", "frames", "0", "45", "1", "1", "0",
             "--arm-drop"],
            ALL_BONES, action="Jog_Fwd_Loop")
        self.assertIsNotNone(exc, "render_drive returned normally on a bare "
                                  "--arm-drop")
        self.assertNotIn(exc.code, (0, None))
        text = _exit_text(exc, err)
        self.assertIn("--arm-drop", text)
        self.assertIn("render_drive:", text)
        self.assertNotIn("RENDERED", out)

    def test_missing_upper_arm_exit_carries_no_rig_hint(self):
        # sprite-pipeline-58 AC2: the upper-arm bones are not configurable
        # through rig: in project.yaml, so their missing-fcurve exit keeps
        # the plain message (the rig: pointer belongs to the head-lift
        # path — test_drive_rig_names.py pins that half)
        for missing in ("DEF-upper_arm.L", "DEF-upper_arm.R"):
            with self.subTest(missing=missing):
                present = [b for b in ALL_BONES if b != missing]
                fcs, out, err, exc = _run_render_drive(
                    ["run.glb", "Jog_Fwd_Loop", "frames", "0", "45", "1", "1", "0",
                     "--arm-drop", "45"],
                    present, action="Jog_Fwd_Loop")
                self.assertIsNotNone(exc, "render_drive returned normally on a "
                                          "missing arm bone")
                self.assertNotIn(exc.code, (0, None))
                text = _exit_text(exc, err)
                self.assertIn(missing, text)
                self.assertIn("Jog_Fwd_Loop", text)
                self.assertNotIn("set rig", text)
                self.assertNotIn("RENDERED", out)


class TestPackagedProfileNumbers(unittest.TestCase):
    """The packaged profiles are the A-pose values the fixtures measure
    (AC14-r1 states them independently in test_metrics.py; this pins them
    here too, where the A-pose constant is defined, so a revert of either
    half fails)."""

    def test_packaged_dirs_are_the_measured_a_pose_values(self):
        import json

        pkg = json.loads(metrics.PACKAGED_PROFILES.read_text())["profiles"]
        want = {
            "ual-female.glb": {"s": (331, 524), "se": (331, 532), "e": (325, 533),
                               "ne": (318, 534), "n": (306, 524)},
            "ual.glb": {"s": (354, 536), "se": (350, 539), "e": (328, 528),
                        "ne": (320, 526), "n": (308, 518)},
        }
        for name, dirs in want.items():
            entry = next(p for p in pkg.values() if p["glb_name"] == name)
            self.assertEqual(entry["resolution"], 704)
            self.assertEqual({d: (v["body_h"], v["floor_y"])
                              for d, v in entry["dirs"].items()}, dirs)


if __name__ == "__main__":
    unittest.main()
