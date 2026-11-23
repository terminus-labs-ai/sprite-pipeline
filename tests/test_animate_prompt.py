"""The planned animate command carries the character prompt.

`comfy animate` requires --prompt (the sp:prompt node); the stage planner
composes it from the character's describe line, the project style, the view's
camera line, the view's face line (face directions only), the background line
and the character's per-pair override. No network, GPU, Blender or ComfyUI.
"""
import tempfile
import unittest
from pathlib import Path

import yaml

from test_workflow_paths import cfg, stubbed_example

from sprite_pipeline import comfy, project, stages, stills


def plan(proj, d="s", anim="run"):
    ctx = stages.StageCtx(proj, cfg(Path("/nonexistent-home")))
    return ctx.plan_pair("courier", proj.animation(anim), d, "animate")[0][0]


def prompt_of(argv):
    ns = comfy.build_parser().parse_args(argv[argv.index("animate"):])
    return ns.prompt


class TestAnimateArgv(unittest.TestCase):
    def test_planned_argv_parses_with_the_comfy_animate_parser(self):
        with tempfile.TemporaryDirectory() as t:
            proj = project.load_project(stubbed_example(t)[0])
            ns = comfy.build_parser().parse_args(plan(proj)[3:])
            self.assertTrue(ns.prompt.strip())
            self.assertEqual(ns.pose_prompt, "a woman jogging in place, head up")  # courier: she (FR 2)


class TestAnimatePromptComposition(unittest.TestCase):
    def setUp(self):
        self._t = tempfile.TemporaryDirectory()
        self.addCleanup(self._t.cleanup)
        self.proj = project.load_project(stubbed_example(self._t.name)[0])
        self.char = self.proj.characters[0]
        self.char.face_dirs = ["s"]
        self.view = project.load_view(self.proj.view)

    def test_look_style_and_camera_present(self):
        p = prompt_of(plan(self.proj, "s"))
        self.assertIn(self.char.describe, p)
        self.assertIn(self.proj.style, p)
        self.assertIn("isometric camera looking down at her", p)
        self.assertIn(self.view["background"], p)
        self.assertNotIn("{", p)

    def test_face_line_only_for_a_face_direction(self):
        self.assertIn("She keeps her head up and her face clearly visible",
                      prompt_of(plan(self.proj, "s")))
        self.assertNotIn("face clearly visible", prompt_of(plan(self.proj, "n")))

    def test_they_pronoun_uses_plural_verbs(self):
        self.char.pronoun = "they"
        p = prompt_of(plan(self.proj, "s"))
        self.assertIn("They keep their head up", p)
        self.assertIn("in the direction they move.", p)

    def test_override_extra_appended_for_the_pair_only(self):
        self.char.overrides = {"run-s": {"seed": 3, "extra": "Holding a lantern."}}
        self.assertTrue(prompt_of(plan(self.proj, "s")).endswith("Holding a lantern."))
        self.assertNotIn("lantern", prompt_of(plan(self.proj, "n")))
        self.assertNotIn("lantern", prompt_of(plan(self.proj, "s", "punch")))


class TestFacingLines(unittest.TestCase):
    def setUp(self):
        self._t = tempfile.TemporaryDirectory()
        self.addCleanup(self._t.cleanup)
        self.proj = project.load_project(stubbed_example(self._t.name)[0])
        self.char = self.proj.characters[0]
        self.char.face_dirs = ["s"]
        self.view = project.load_view(self.proj.view)

    def test_isometric_preset_has_a_facing_line_for_all_eight_directions(self):
        self.assertEqual(set(self.view["facing"]), project.KNOWN_DIRS)

    def test_each_generated_direction_gets_its_own_facing_line(self):
        for d in self.proj.generate:
            line = self.view["facing"][d].format(**stills.pronoun_forms("she"))
            self.assertIn(line, prompt_of(plan(self.proj, d)), d)

    def test_back_facing_line_hides_the_face(self):
        self.assertIn("face is never visible", prompt_of(plan(self.proj, "n")))

    def test_describe_map_gives_each_direction_its_own_look(self):
        self.char.describe = {d: f"LOOK-{d}-END" for d in self.proj.generate}
        for d in self.proj.generate:
            p = prompt_of(plan(self.proj, d))
            self.assertTrue(p.startswith(f"LOOK-{d}-END."), d)
            for other in self.proj.generate:
                if other != d:
                    self.assertNotIn(f"LOOK-{other}-END", p)


# the isometric view's lines for a `she` character, copied from
# views/isometric.yaml so the expected prompt is stated independently
SE_FACING = ("Her body, head and gaze all point toward the lower right of the image, "
             "toward the camera, so we see her front from above the whole time.")
CAMERA = ("Seen from a high fixed isometric camera looking down at her at about 45 degrees, "
          "small in the frame with empty space around her, full body always inside the frame.")
BACKGROUND = ("Background description: the background is one perfectly flat uniform "
              "chroma-key green, no floor, no shadow, no shapes.")


class TestFacingLineSwitch(unittest.TestCase):
    """AC 8 (FR 11): `animations[].facing_line: false` drops the facing line."""

    def load(self, edit):
        t = tempfile.TemporaryDirectory()
        self.addCleanup(t.cleanup)
        p = stubbed_example(t.name)[0]
        d = yaml.safe_load(p.read_text())
        edit(d)
        p.write_text(yaml.safe_dump(d))
        proj = project.load_project(p)
        proj.characters[0].face_dirs = ["s"]
        return proj

    def test_AC8_r1_facing_line_false_omits_the_facing_line(self):
        def edit(d):
            next(a for a in d["animations"] if a["name"] == "death")["facing_line"] = False
        proj = self.load(edit)
        p = prompt_of(plan(proj, "se", "death"))
        self.assertNotIn(SE_FACING, p)
        self.assertNotIn("lower right", p)
        self.assertIn(CAMERA, p)
        self.assertIn(SE_FACING, prompt_of(plan(proj, "se", "run")))

    def test_AC8_r2_no_key_keeps_todays_prompt(self):
        proj = self.load(lambda d: None)
        char = proj.characters[0]
        self.assertEqual(prompt_of(plan(proj, "se", "run")),
                         f"{char.describe}. {proj.style}. {CAMERA} {SE_FACING} {BACKGROUND}")


if __name__ == "__main__":
    unittest.main()
