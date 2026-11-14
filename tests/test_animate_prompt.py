"""The planned animate command carries the character prompt.

`comfy animate` requires --prompt (the sp:prompt node); the stage planner
composes it from the character's describe line, the project style, the view's
camera line, the view's face line (face directions only), the background line
and the character's per-pair override. No network, GPU, Blender or ComfyUI.
"""
import tempfile
import unittest
from pathlib import Path

from test_workflow_paths import cfg, stubbed_example

from sprite_pipeline import comfy, project, stages


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


if __name__ == "__main__":
    unittest.main()
