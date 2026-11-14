"""sprite-pipeline-22: a character override `<animation>-<dir>: {seed: N}`
sets the animate step's seed for that pair only (FR 2)."""

import tempfile
import unittest
from pathlib import Path

import yaml

from sprite_pipeline import project, stages
from test_workflow_paths import cfg, stubbed_example


def animate_seeds(overrides, pairs):
    """The planned animate --seed for each (animation, dir) in pairs, for the
    example courier with the given overrides."""
    with tempfile.TemporaryDirectory() as t:
        p, _ = stubbed_example(t)
        d = yaml.safe_load(p.read_text())
        d["characters"][0]["overrides"] = overrides
        p.write_text(yaml.safe_dump(d))
        proj = project.load_project(p)
        ctx = stages.StageCtx(proj, cfg(Path("/nonexistent-home")))
        out = {}
        for anim, dr in pairs:
            argv = ctx.plan_pair("courier", proj.animation(anim), dr, "animate")[0][0]
            out[(anim, dr)] = argv[argv.index("--seed") + 1]
        return out, str(proj.seed)


class TestOverrideSeed(unittest.TestCase):

    def test_override_seed_applies_to_its_pair_only(self):
        seeds, base = animate_seeds({"run-s": {"seed": 11}},
                                    [("run", "s"), ("run", "se"), ("punch", "s")])
        self.assertEqual(seeds[("run", "s")], "11")
        self.assertEqual(seeds[("run", "se")], base)
        self.assertEqual(seeds[("punch", "s")], base)

    def test_extra_only_override_keeps_the_project_seed(self):
        seeds, base = animate_seeds({"run-s": {"extra": "short hair"}}, [("run", "s")])
        self.assertEqual(seeds[("run", "s")], base)


if __name__ == "__main__":
    unittest.main()
