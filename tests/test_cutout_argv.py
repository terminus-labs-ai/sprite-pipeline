"""Regression test: plan_cutout emits sprite-gen cutout with the correct argv.

At main 4e3b883 plan_cutout used --source/--out-dir which sprite-gen 4.3.0
does not recognise (it takes input as a positional and --out <file>).

sprite-gen cutout --help:
    usage: sprite-gen cutout [-h] [--out OUT]
                             [--key {auto,white,magenta,green,measured}] ...
                             input

Acceptance: argv[1] == "cutout", the input is positional (no --source),
--out points to <out_dir>/<stem>.keyed.png, --key measured is present.
"""
import unittest
from pathlib import Path

from sprite_pipeline import stages


class TestCutoutArgv(unittest.TestCase):
    """plan_cutout must emit a command that matches sprite-gen 4.3.0 cutout's parser."""

    def test_cutout_positional_input_not_source_flag(self):
        still = Path("/tmp/se.png")
        out_dir = Path("/tmp/master")
        cmds = stages.plan_cutout(still, out_dir)
        self.assertEqual(len(cmds), 1)
        argv = list(cmds[0][0])  # copy so we can inspect/modify
        # argv[0] is the sprite-gen entry point (we don't assert its path)
        exe = argv.pop(0)
        self.assertEqual(argv[0], "cutout")
        # input must be positional (no --source flag)
        self.assertNotIn("--source", argv)
        self.assertNotIn("--out-dir", argv)
        # the still path is the first positional after "cutout"
        self.assertEqual(argv[1], str(still))
        # --out <file> must be present (not --out-dir)
        out_idx = argv.index("--out")
        expected_out = str(out_dir / "se.keyed.png")
        self.assertEqual(argv[out_idx + 1], expected_out)
        # --key measured must be present
        key_idx = argv.index("--key")
        self.assertEqual(argv[key_idx + 1], "measured")

    def test_cutout_stem_becomes_keyed_png_for_nested_paths(self):
        still = Path("/some/path/to/image.png")
        out_dir = Path("/output/master")
        cmds = stages.plan_cutout(still, out_dir)
        argv = list(cmds[0][0])
        exe = argv.pop(0)
        self.assertEqual(argv[1], str(still))
        self.assertEqual(argv[argv.index("--out") + 1],
                         str(out_dir / "image.keyed.png"))

    def test_cutout_exe_is_sprite_gen(self):
        still = Path("/tmp/se.png")
        out_dir = Path("/tmp/master")
        cmds = stages.plan_cutout(still, out_dir)
        argv = cmds[0][0]
        self.assertEqual(argv[0], str(stages.sprite_gen_exe()))


if __name__ == "__main__":
    unittest.main()
