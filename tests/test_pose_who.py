"""sprite-pipeline-19: the animate step's pose prompt fills `{who}` from the
character (FR 2): its `who` when given, else a default from its pronoun."""

import tempfile
import unittest
from pathlib import Path

import yaml

from sprite_pipeline import project, stages
from test_workflow_paths import cfg, stubbed_example


def pose_prompt(char_fields):
    """The planned --pose-prompt of run/s for the example courier with
    char_fields merged into its character entry."""
    with tempfile.TemporaryDirectory() as t:
        p, _ = stubbed_example(t)
        d = yaml.safe_load(p.read_text())
        d["characters"][0].update(char_fields)
        p.write_text(yaml.safe_dump(d))
        proj = project.load_project(p)
        ctx = stages.StageCtx(proj, cfg(Path("/nonexistent-home")))
        argv = ctx.plan_pair("courier", proj.animation("run"), "s", "animate")[0][0]
        return argv[argv.index("--pose-prompt") + 1]


class TestPoseWho(unittest.TestCase):

    def test_default_from_pronoun(self):
        self.assertEqual(pose_prompt({"pronoun": "she"}), "a woman jogging in place, head up")
        self.assertEqual(pose_prompt({"pronoun": "he"}), "a man jogging in place, head up")
        self.assertEqual(pose_prompt({"pronoun": "they"}), "a person jogging in place, head up")

    def test_explicit_who_wins(self):
        self.assertEqual(pose_prompt({"pronoun": "she", "who": "a tall android"}),
                         "a tall android jogging in place, head up")


if __name__ == "__main__":
    unittest.main()
