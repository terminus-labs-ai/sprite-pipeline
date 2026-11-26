"""Tests for sprite_pipeline.project: FR 4 validation, FR 3 clip_frames default,
the drive cache key (FR 10) and the example project carrying the motion library
(FR 13 / AC10). No network, GPU, Blender or ComfyUI."""
import dataclasses
import hashlib
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

import yaml

import mannequins

from sprite_pipeline import project

HERE = Path(__file__).resolve().parent
FIXTURES = HERE / "fixtures" / "project"
EXAMPLES = HERE.parent / "examples" / "isometric"
# motions.yaml lived here until sprite-pipeline-4 removed the kit CLI; the
# cross-check reads it from the last commit that had it (b17638b)

ASSET = (FIXTURES / "still" / "a.png").read_bytes()


def base_project(generate="[s, se, e, ne, n]", mirror="{sw: se, w: e, nw: ne}",
                 naming="{animation}_{dir}", frame_size=704, animation_dirs=None,
                 characters="- {name: courier, still: stills/courier.png, describe: \"a courier\", pronoun: she}"):
    """A minimal valid project; the callers tweak one field per case.
    `animation_dirs` pins the animation's `directions:` (needed when the
    animation must cover fewer directions than generate, e.g. two characters
    sharing the default naming)."""
    anim_dirs = "" if animation_dirs is None else f", directions: {animation_dirs}"
    return f"""style: "2D game sprite art"
view: isometric
directions:
  generate: {generate}
  mirror: {mirror}
sprite: {{scale: 0.2023, colors: 40, accent_boost: 3, fps: 30, frame_size: {frame_size}}}
naming: "{naming}"
export: godot
seed: 7
pose_strength: 0.75
workflows: {{edit: workflows/edit.json, animate: workflows/animate.json}}
characters:
  {characters}
animations:
  - {{name: run, source: anims/ual.glb#Jog_Fwd_Loop, frames: 22, start: 1, clip_frames: 45, loop: true, prompt: "{{who}} jogging in place, head up"{anim_dirs}}}
"""


def write_project(tmp, text, with_still=True, with_glb=True):
    """Write project.yaml plus the stub still and GLB it references."""
    t = Path(tmp)
    p = t / "project.yaml"
    p.write_text(text)
    if with_still:
        f = t / "stills" / "courier.png"
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_bytes(ASSET)
        f = t / "stills" / "guard.png"
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_bytes(ASSET)
    if with_glb:
        f = t / "anims" / "ual.glb"
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_bytes(ASSET)
    mannequins.register(ASSET, glb_name="ual.glb")
    return p


class TestValidation(unittest.TestCase):
    """FR 4: each case is a ProjectError (a ValueError) naming the offending field."""

    def test_r1_unknown_direction_in_generate(self):
        with tempfile.TemporaryDirectory() as t:
            with self.assertRaises(project.ProjectError) as cm:
                project.load_project(write_project(t, base_project(generate="[s, se, e, ne, x]")))
            self.assertEqual(
                str(cm.exception),
                "project.yaml: directions.generate[4]: unknown direction 'x' for view isometric")

    def test_r2_mirror_source_not_in_generate(self):
        with tempfile.TemporaryDirectory() as t:
            with self.assertRaises(project.ProjectError) as cm:
                project.load_project(write_project(
                    t, base_project(generate="[s, se, e, ne]", mirror="{sw: sw, w: e, nw: ne}")))
            self.assertIn("project.yaml: directions.mirror.sw:", str(cm.exception))
            self.assertIn("not in generate", str(cm.exception))

    def test_r3_missing_still(self):
        with tempfile.TemporaryDirectory() as t:
            with self.assertRaises(project.ProjectError) as cm:
                project.load_project(write_project(t, base_project(), with_still=False))
            self.assertIn("project.yaml: characters[0].still", str(cm.exception))

    def test_r4_source_without_action(self):
        with tempfile.TemporaryDirectory() as t:
            with self.assertRaises(project.ProjectError) as cm:
                project.load_project(write_project(
                    t, base_project().replace("anims/ual.glb#Jog_Fwd_Loop", "anims/ual.glb")))
            self.assertEqual(
                str(cm.exception),
                "project.yaml: animations[0].source: missing #<Action>")

    def test_r4_missing_source_file(self):
        with tempfile.TemporaryDirectory() as t:
            with self.assertRaises(project.ProjectError) as cm:
                project.load_project(write_project(t, base_project(), with_glb=False))
            self.assertIn("project.yaml: animations[0].source", str(cm.exception))
            self.assertIn("not found", str(cm.exception))

    def test_r5_clip_frames_not_4n1(self):
        with tempfile.TemporaryDirectory() as t:
            with self.assertRaises(project.ProjectError) as cm:
                project.load_project(write_project(t, base_project().replace("clip_frames: 45", "clip_frames: 44")))
            self.assertIn("project.yaml: animations[0].clip_frames", str(cm.exception))
            self.assertIn("not 4n+1", str(cm.exception))

    def test_r5_clip_frames_shorter_than_start_plus_frames(self):
        # case table AC2-r5 second input: looped start 1, frames 22, clip_frames 21
        with tempfile.TemporaryDirectory() as t:
            text = base_project().replace("frames: 22, start: 1, clip_frames: 45",
                                          "frames: 22, start: 1, clip_frames: 21, loop: true")
            with self.assertRaises(project.ProjectError) as cm:
                project.load_project(write_project(t, text))
            self.assertIn("project.yaml: animations[0].clip_frames", str(cm.exception))
            self.assertIn("shorter than start + frames", str(cm.exception))

    def test_r6_view_without_turn_prompt_for_generated_direction(self):
        with tempfile.TemporaryDirectory() as t:
            (Path(t) / "notest.yaml").write_text("turn:\n  se: kept facing the lower right\n")
            p = write_project(t, base_project().replace("view: isometric", "view: notest")
                              .replace("  mirror: {sw: se, w: e, nw: ne}\n", ""))
            with self.assertRaises(project.ProjectError) as cm:
                project.load_project(p, views_dir=Path(t))
            msg = str(cm.exception)
            self.assertIn("view notest has no turn prompt for direction 's'", msg)
            self.assertIn("views/notest.yaml", msg)  # names the direction and the view

    def test_r6_view_without_facing_line_for_generated_direction(self):
        with tempfile.TemporaryDirectory() as t:
            (Path(t) / "notest.yaml").write_text(
                "turn:\n  s: a\n  se: b\n  e: c\n  ne: d\n  n: e\nfacing:\n  se: kept facing the lower right\n")
            p = write_project(t, base_project().replace("view: isometric", "view: notest")
                              .replace("  mirror: {sw: se, w: e, nw: ne}\n", ""))
            with self.assertRaises(project.ProjectError) as cm:
                project.load_project(p, views_dir=Path(t))
            self.assertIn("view notest has no facing line for direction 's'", str(cm.exception))

    def _char(self, describe):
        return base_project(characters="- {name: courier, still: stills/courier.png, describe: " + describe + ", pronoun: she}")

    def test_describe_string_applies_to_every_direction(self):
        with tempfile.TemporaryDirectory() as t:
            c = project.load_project(write_project(t, base_project())).characters[0]
            self.assertEqual({d: c.look(d) for d in ("s", "n")}, {"s": "a courier", "n": "a courier"})

    def test_describe_map_per_direction(self):
        with tempfile.TemporaryDirectory() as t:
            c = project.load_project(write_project(t, self._char("{s: front, se: fs, e: side, ne: tq, n: back}"))).characters[0]
            self.assertEqual(c.look("n"), "back")
            self.assertEqual(c.look("e"), "side")

    def test_describe_map_missing_generated_direction(self):
        with tempfile.TemporaryDirectory() as t:
            with self.assertRaises(project.ProjectError) as cm:
                project.load_project(write_project(t, self._char("{s: front, se: fs, e: side, ne: tq}")))
            self.assertIn("characters[0].describe: no entry for direction 'n'", str(cm.exception))

    def test_describe_map_direction_not_generated(self):
        with tempfile.TemporaryDirectory() as t:
            with self.assertRaises(project.ProjectError) as cm:
                project.load_project(write_project(t, self._char("{s: a, se: b, e: c, ne: d, n: e, sw: f}")))
            self.assertIn("characters[0].describe.sw: not in directions.generate", str(cm.exception))

    def test_describe_back_is_rejected(self):
        with tempfile.TemporaryDirectory() as t:
            text = base_project(characters="- {name: courier, still: stills/courier.png, describe: a, describe_back: b, pronoun: she}")
            with self.assertRaises(project.ProjectError) as cm:
                project.load_project(write_project(t, text))
            self.assertIn("describe_back", str(cm.exception))
            self.assertIn("describe map", str(cm.exception))

    def test_r6_unknown_view_has_no_preset(self):
        with tempfile.TemporaryDirectory() as t:
            p = write_project(t, base_project().replace("view: isometric", "view: badview"))
            with self.assertRaises(project.ProjectError) as cm:
                project.load_project(p, views_dir=Path(t))
            self.assertEqual(str(cm.exception), "project.yaml: view: view 'badview' has no preset views/badview.yaml")

    def test_r7_naming_not_unique_across_generated_directions(self):
        # {animation} alone names the two generated directions the same
        with tempfile.TemporaryDirectory() as t:
            with self.assertRaises(project.ProjectError) as cm:
                project.load_project(write_project(t, base_project(naming="{animation}")))
            self.assertIn("project.yaml: naming", str(cm.exception))
            self.assertIn("duplicate", str(cm.exception))

    def test_project_error_is_a_value_error(self):
        with tempfile.TemporaryDirectory() as t:
            with self.assertRaises(ValueError) as cm:
                project.load_project(write_project(t, base_project(generate="[s, x]")))
            self.assertIsInstance(cm.exception, project.ProjectError)


class TestLoading(unittest.TestCase):
    def test_paths_resolve_against_the_project_file(self):
        with tempfile.TemporaryDirectory() as t:
            proj = project.load_project(write_project(t, base_project()))
            base = Path(t)
            self.assertEqual(proj.path, (base / "project.yaml").resolve())
            self.assertEqual(proj.characters[0].still, base / "stills" / "courier.png")
            self.assertEqual(proj.animations[0].source, base / "anims" / "ual.glb")
            self.assertEqual(proj.animations[0].action, "Jog_Fwd_Loop")
            self.assertEqual(proj.output_root(), base / "out")

    def test_defaults(self):
        with tempfile.TemporaryDirectory() as t:
            proj = project.load_project(write_project(t, base_project()))
            a = proj.animations[0]
            self.assertEqual((a.start, a.repeats, a.pre, a.post), (1, 1, 1, 0))
            self.assertFalse(a.look_down)
            self.assertEqual(a.head_lift, 25)
            self.assertEqual(a.state, "attack")
            self.assertEqual(a.seam_max, 0.0)
            self.assertEqual(a.directions, ["s", "se", "e", "ne", "n"])
            self.assertTrue(a.loop)
            self.assertEqual(proj.rig, {"neck": "DEF-neck", "head": "DEF-head"})
            self.assertEqual(proj.naming, "{animation}_{dir}")
            self.assertEqual(proj.export, "godot")
            self.assertEqual(proj.seed, 7)
            self.assertEqual(proj.pose_strength, 0.75)
            self.assertEqual(proj.output, Path("out"))

    def test_look_down_zeroes_the_head_lift(self):
        with tempfile.TemporaryDirectory() as t:
            proj = project.load_project(
                write_project(t, base_project().replace("clip_frames: 45", "clip_frames: 45, look_down: true")))
            self.assertTrue(proj.animations[0].look_down)
            self.assertEqual(proj.animations[0].head_lift, 0)

    def test_r8_clip_frames_default_rounds_up_to_4n1(self):
        # start 1 + frames 22 = 23 -> 25
        with tempfile.TemporaryDirectory() as t:
            proj = project.load_project(write_project(t, base_project().replace(", clip_frames: 45", "")))
            self.assertEqual((proj.animations[0].start, proj.animations[0].frames), (1, 22))
            self.assertEqual(proj.animations[0].clip_frames, 25)


class TestDriveKey(unittest.TestCase):
    def _proj(self, frame_size=704):
        # the temp dir must outlive the call: drive_key reads the GLB from disk
        t = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, t)
        return project.load_project(write_project(t, base_project(frame_size=frame_size)))

    def test_r1_equal_when_only_the_path_changes(self):
        # same GLB bytes copied to two paths, same action and params -> equal key
        p1, p2 = self._proj(), self._proj()
        b = p1.animations[0].source.read_bytes()
        other = p1.animations[0].source.parent.parent / "copy" / "ual.glb"
        other.parent.mkdir(parents=True, exist_ok=True)
        other.write_bytes(b)
        self.assertNotEqual(p1.animations[0].source, other)
        self.assertNotIn("anims", p1.drive_key(p1.animations[0], b))  # the key is hex, not a path echo
        self.assertEqual(p1.drive_key(p1.animations[0], b),
                         p2.drive_key(p2.animations[0], other.read_bytes()))

    def test_r2_differs_when_content_action_or_params_change(self):
        p = self._proj()
        a, b = p.animations[0], p.animations[0].source.read_bytes()
        self.assertNotEqual(p.drive_key(a, b), p.drive_key(a, b + b"x"))  # GLB content
        self.assertNotEqual(p.drive_key(a, b),
                            p.drive_key(dataclasses.replace(a, action="Punch_Cross"), b))  # action
        for kw in ({"look_down": True}, {"repeats": 2}, {"pre": 4}, {"post": 5}):  # drive params
            self.assertNotEqual(p.drive_key(a, b), p.drive_key(dataclasses.replace(a, **kw), b), kw)
        p2 = self._proj(frame_size=1024)
        self.assertNotEqual(p.drive_key(a, b), p2.drive_key(p2.animations[0], b))  # frame_size
        self.assertEqual(p.drive_key(a, b), p.drive_key(a, b))  # stable for the same object


class TestExampleProject(unittest.TestCase):
    """AC10: the example validates and carries the 13 animations of the current
    motion library with their tuned values (FR 13; dec-631)."""

    EXPECTED = {  # name: (action, repeats, pre, post, head lift, clip, start, frames, loop, state, seam_max, prompt)
        "run": ("Jog_Fwd_Loop", 2, 1, 0, 25, 45, 1, 22, True, "run", 0.0, "{who} jogging in place, head up"),
        "punch": ("Punch_Cross", 1, 4, 5, 25, 33, 0, 33, False, "attack", 0.0, "{who} throwing a fast cross punch, head up"),
        "idle": ("Idle_Loop", 1, 1, 0, 25, 61, 1, 60, True, "idle", 0.0, "{who} standing still with a relaxed breathing cycle"),
        "walk": ("Walk_Loop", 1, 1, 0, 25, 33, 1, 32, True, "walk", 0.0, "{who} walking forward at an even pace in place"),
        "sneak": ("Crouch_Fwd_Loop", 1, 1, 0, 25, 49, 1, 48, True, "walk", 3.0, "{who} crouched low moving forward in place"),
        "crouch_idle": ("Crouch_Idle_Loop", 1, 3, 0, 25, 73, 3, 70, True, "idle", 0.0, "{who} crouched and holding still"),
        "hack": ("Fixing_Kneeling", 1, 0, 0, 0, 125, 0, 125, True, "idle", 0.0, "{who} kneeling and working with both hands on a device"),
        "interact": ("Interact", 1, 1, 0, 25, 49, 1, 49, False, "attack", 0.0, "{who} reaching out and pressing a control with one hand"),
        "jab": ("Punch_Jab", 1, 4, 4, 25, 29, 0, 29, False, "attack", 0.0, "{who} throwing a fast straight jab punch"),
        "pistol_shoot": ("Pistol_Shoot", 1, 3, 3, 25, 21, 0, 21, False, "attack", 0.0, "{who} firing a pistol with one hand"),
        "hit": ("Hit_Chest", 1, 2, 3, 0, 13, 0, 13, False, "attack", 0.0, "{who} taking a hit to the chest and flinching back"),
        "death": ("Death01", 1, 0, 3, 0, 65, 0, 61, False, "attack", 0.0, "{who} falling to the ground and staying down"),
        "roll": ("Roll", 1, 2, 0, 0, 37, 2, 35, True, "attack", 0.0, "{who} rolling forward on the ground"),
    }

    def _stubbed_example(self, t):
        # the still and GLB are stubbed by the test fixtures; the example folder
        # ships only project.yaml and the README
        base = Path(t) / "isometric"
        shutil.copytree(EXAMPLES, base)
        for rel in ("stills/courier.png", "anims/ual.glb"):
            f = base / rel
            f.parent.mkdir(parents=True, exist_ok=True)
            f.write_bytes(ASSET)
        mannequins.register(ASSET, glb_name="ual.glb")
        return project.load_project(base / "project.yaml"), base

    def test_matches_the_motion_library_at_base(self):
        # motions.yaml was removed with the kit CLI (sprite-pipeline-4); read
        # it from the last commit that had it
        text = subprocess.run(
            ["git", "-C", str(HERE.parent), "show",
             "b17638b:src/sprite_pipeline/motions.yaml"],
            capture_output=True, text=True, check=True).stdout
        motions = yaml.safe_load(text)
        # the table must track motions.yaml exactly (expected values, not code)
        for name, m in motions["moves"].items():
            exp = (m["action"], m["repeats"], m["pre"], m["post"], m["head_pitch"],
                   m["clip_length"], m["loop_start"], m["loop_length"],
                   m["loop_mode"] == "fixed", m["state"], float(m.get("seam_max") or 0),
                   m["pose_prompt"])
            if name == "death":  # dec-823: death clip_frames 61 -> 65 since the library
                exp = exp[:5] + (65,) + exp[6:]
            self.assertEqual(self.EXPECTED[name], exp, name)
        with tempfile.TemporaryDirectory() as t:
            proj, base = self._stubbed_example(t)
            self.assertEqual(proj.view, "isometric")
            self.assertEqual(proj.generate, ["s", "se", "e", "ne", "n"])
            self.assertEqual(proj.mirror, {"sw": "se", "w": "e", "nw": "ne"})
            self.assertEqual(len(proj.animations), 13)
            self.assertEqual([a.name for a in proj.animations], list(self.EXPECTED))
            self.assertEqual(len(proj.characters), 1)
            c = proj.characters[0]
            self.assertEqual((c.name, c.pronoun), ("courier", "she"))
            self.assertEqual(c.still, base / "stills" / "courier.png")
            for a in proj.animations:
                self.assertEqual(a.source, base / "anims" / "ual.glb")
                got = (a.action, a.repeats, a.pre, a.post, a.head_lift, a.clip_frames,
                       a.start, a.frames, a.loop, a.state, a.seam_max, a.prompt)
                self.assertEqual(got, self.EXPECTED[a.name], a.name)

    def test_directions_and_defaults(self):
        with tempfile.TemporaryDirectory() as t:
            proj, _ = self._stubbed_example(t)
            for a in proj.animations:
                self.assertEqual(a.directions, ["s", "se", "e", "ne", "n"], a.name)
            self.assertEqual(proj.naming, "{animation}_{dir}")
            self.assertEqual(proj.sprite.frame_size, 704)
            self.assertEqual(proj.sprite.fps, 30)
            self.assertEqual(proj.sprite.scale, 0.2023)
            self.assertEqual(proj.rig, {"neck": "DEF-neck", "head": "DEF-head"})
            self.assertEqual(proj.seed, 7)
            self.assertEqual(proj.pose_strength, 0.85)


class TestFaceDirs(unittest.TestCase):
    """FR 2: a character without face_dirs gets the view preset's camera-facing
    directions intersected with directions.generate, in generate order; an
    explicit list, including [], is kept as given."""

    def test_r1_face_dirs_omitted_defaults_to_view_face_dirs(self):
        # generate [s, se, e]: the default is the preset's camera-facing set
        # restricted to what is generated — expected independently of the code
        with tempfile.TemporaryDirectory() as t:
            text = base_project(generate="[s, se, e]", mirror="{}")
            proj = project.load_project(write_project(t, text))
            self.assertEqual(proj.characters[0].face_dirs, ["s", "se", "e"])

    # a stub GLB profile carrying sw and w too: the FR 4 gate (amendment 10)
    # needs a dirs entry for every generated direction
    SW_W_PROFILES = {hashlib.sha256(ASSET).hexdigest(): mannequins.profile(
        {**mannequins.DIRS, "sw": mannequins.DIRS["se"], "w": mannequins.DIRS["e"]},
        glb_name="ual.glb")}

    def test_r2_default_follows_generate_order_including_sw_w(self):
        # generate [w, sw, e, se, s]: the default is the preset members that
        # are generated, listed in generate order (not preset order)
        with tempfile.TemporaryDirectory() as t:
            text = base_project(generate="[w, sw, e, se, s]", mirror="{}",
                                animation_dirs="[w]")
            proj = project.load_project(write_project(t, text),
                                        profiles=self.SW_W_PROFILES)
            self.assertEqual(proj.characters[0].face_dirs, ["w", "sw", "e", "se", "s"])

    def test_r2_all_five_face_dirs_when_sw_w_are_generated(self):
        # generate [s, se, e, sw, w]: all five preset members appear; sw and w
        # are in the default only because they are generated
        with tempfile.TemporaryDirectory() as t:
            text = base_project(generate="[s, se, e, sw, w]", mirror="{}")
            proj = project.load_project(write_project(t, text),
                                        profiles=self.SW_W_PROFILES)
            self.assertEqual(proj.characters[0].face_dirs, ["s", "se", "e", "sw", "w"])

    def test_r2_explicit_face_dirs_kept_as_given(self):
        two = ("- {name: courier, still: stills/courier.png, describe: \"a courier\", pronoun: she, face_dirs: [e]}\n"
               "  - {name: guard, still: stills/guard.png, describe: \"a guard\", pronoun: he, face_dirs: []}")
        with tempfile.TemporaryDirectory() as t:
            text = base_project(generate="[s, se, e]", mirror="{}", naming="{character}_{animation}_{dir}",
                                characters=two)
            proj = project.load_project(write_project(t, text))
            self.assertEqual(proj.characters[0].face_dirs, ["e"])  # explicit non-empty, as given
            self.assertEqual(proj.characters[1].face_dirs, [])     # explicit [] is none, not the default

    def test_r4_face_dirs_unknown_direction(self):
        # explicit face_dirs is validated exactly as before: the error names
        # the index, the field and the offending value
        with tempfile.TemporaryDirectory() as t:
            text = base_project(generate="[s, se, e]", mirror="{}",
                                characters="- {name: courier, still: stills/courier.png, describe: \"a courier\", pronoun: she, face_dirs: [n]}")
            with self.assertRaises(project.ProjectError) as cm:
                project.load_project(write_project(t, text))
            self.assertEqual(
                str(cm.exception),
                "project.yaml: characters[0].face_dirs: 'n' not in directions.generate")

    def test_r1_the_face_line_comes_from_the_view_preset_not_project_py(self):
        # the default set is declared in views/isometric.yaml: the preset's
        # face_dirs covers exactly the five camera-facing directions of the
        # view's own turn prompts (the back-facing ne, n, nw are absent), and
        # project.py hard-codes no direction list
        preset = project.load_view("isometric")
        self.assertEqual(set(preset["face_dirs"]), {"s", "se", "e", "sw", "w"})
        self.assertIn("turn", preset)
        src = (Path(project.__file__).read_text().replace("\n", " "))
        self.assertNotIn("s\", \"se\", \"e", src)
        self.assertNotIn("'s', 'se', 'e'", src)
        readme = (Path(project.__file__).resolve().parents[2] / "README.md").read_text()
        self.assertIn("face_dirs: []", readme)  # README documents [] as none


class TestViewPreset(unittest.TestCase):
    def test_isometric_carries_the_guard_kit_turn_prompts_verbatim(self):
        # verbatim from the new-kit branch's newkit.TURN
        preset = project.load_view("isometric")
        self.assertEqual(preset["turn"]["s"],
                         "turned to face straight down toward the bottom of the image, toward the camera, "
                         "so we see {pos} front from above")
        self.assertEqual(preset["turn"]["e"],
                         "turned to face directly toward the right side of the image, seen in side profile from above")
        self.assertEqual(preset["turn"]["ne"],
                         "turned away from the camera, facing toward the upper right of the image, so we see "
                         "{pos} back and the back of {pos} head from above at three-quarters")
        self.assertEqual(preset["turn"]["n"],
                         "turned fully away from the camera, facing straight up toward the top of the image, "
                         "so we see only {pos} back and the back of {pos} head from above")
        self.assertTrue(preset["turn"]["se"])
        self.assertIn("head up and", preset["face"])
        self.assertIn("45 degrees", preset["camera"])

    def test_load_view_rejects_an_unknown_view(self):
        with tempfile.TemporaryDirectory() as t:
            with self.assertRaises(project.ProjectError) as cm:
                project.load_view("noman", views_dir=Path(t))
            self.assertIn("no preset views/noman.yaml", str(cm.exception))


class TestAC4CanvasKeyRemoved(unittest.TestCase):
    """death-repro FR 7 (dec-821): the canvas is automatic."""

    def test_r2_a_canvas_key_is_a_project_error(self):
        second = ('\n  - {name: jump, canvas: 1024, source: anims/ual.glb#Jog_Fwd_Loop, '
                  'frames: 22, clip_frames: 45, prompt: "{who} jumping"}\n')
        with tempfile.TemporaryDirectory() as t:
            with self.assertRaises(project.ProjectError) as cm:
                project.load_project(write_project(t, base_project() + second.lstrip("\n")))
            self.assertEqual(
                str(cm.exception),
                "project.yaml: animations[1].canvas: canvas is now automatic; "
                "use drive.margin or drive_scale")

    def test_drive_margin_and_aspect_default_and_validate(self):
        with tempfile.TemporaryDirectory() as t:
            p = project.load_project(write_project(t, base_project()))
            self.assertEqual((p.drive.margin, p.drive.aspect), (1.3, (16, 9)))
        for block, key in (("{margin: 0}", "drive.margin"), ("{aspect: [16]}", "drive.aspect"),
                           ("{aspect: [16, 0]}", "drive.aspect")):
            with self.subTest(block=block), tempfile.TemporaryDirectory() as t:
                text = base_project().replace("\nanimations:", f"\ndrive: {block}\nanimations:")
                with self.assertRaises(project.ProjectError) as cm:
                    project.load_project(write_project(t, text))
                self.assertIn(f"project.yaml: {key}:", str(cm.exception))


class TestAnchorDirGate(unittest.TestCase):
    """FR 4 amendment 10 with FR 14's anchor requirement: the profile gate also
    needs `dirs["s"]`, the anchor direction the go master is keyed off (FR 6),
    whether or not `s` is generated. A profile without it fails the load with
    the existing missing-dirs ProjectError instead of crashing the anchor later
    with KeyError 's'."""

    def profiles_without(self, *skip):
        """The stub GLB's profile with its `dirs` missing the given directions."""
        dirs = {d: v for d, v in mannequins.DIRS.items() if d not in skip}
        return {hashlib.sha256(ASSET).hexdigest(): mannequins.profile(dirs,
                                                                     glb_name="ual.glb")}

    def test_r1_anchor_direction_s_missing_is_a_project_error(self):
        # AC2-r1: generate [se, e] — `s` is not generated, so the old gate let
        # this through; the profile must still carry the anchor direction
        with tempfile.TemporaryDirectory() as t:
            with self.assertRaises(project.ProjectError) as cm:
                project.load_project(write_project(t, base_project(generate="[se, e]",
                                                                   mirror="{}")),
                                     profiles=self.profiles_without("s"))
            msg = str(cm.exception)
            # the existing missing-dirs message shape: the GLB, the direction,
            # the sha256 prefix and the mannequin-profile command
            self.assertIn("has no dirs entry for s;", msg)
            self.assertIn("ual.glb", msg)
            self.assertIn(hashlib.sha256(ASSET).hexdigest()[:16], msg)
            self.assertIn("sprite-pipeline mannequin-profile", msg)
            self.assertRegex(msg, r"sprite-pipeline mannequin-profile \S*ual\.glb$")

    def test_r2_profile_carrying_s_se_e_loads(self):
        # AC2-r2: the same project whose profile carries s loads clean
        with tempfile.TemporaryDirectory() as t:
            proj = project.load_project(write_project(t, base_project(generate="[se, e]",
                                                                       mirror="{}")),
                                        profiles=self.profiles_without())
            self.assertEqual(proj.generate, ["se", "e"])

    def test_r3_missing_generated_direction_message_unchanged(self):
        # AC2-r3: the pre-existing case keeps naming the generated direction
        # only — s is present here, so it must not appear in the list
        with tempfile.TemporaryDirectory() as t:
            with self.assertRaises(project.ProjectError) as cm:
                project.load_project(write_project(t, base_project(generate="[s, se]",
                                                                   mirror="{}")),
                                     profiles=self.profiles_without("se"))
            msg = str(cm.exception)
            self.assertIn("has no dirs entry for se;", msg)
            # only the generated direction is listed: s is present, and a
            # second entry would read "for se, s;" / "for s, se;"
            self.assertNotIn(", s;", msg)
            self.assertNotIn("for s;", msg)
            self.assertIn("sprite-pipeline mannequin-profile", msg)


if __name__ == "__main__":
    unittest.main()
