"""Amendment 12 hygiene (sprite-pipeline-54): the `s`-stop condition and the
advisory size report each have one definition in cli.py. No behaviour
change: the example project's `go --dry-run` and both stops' printed output
are byte-identical to the base.

- H1: "the project waits at the `s` stop" (a direction besides `s` is
  generated and `s` is unapproved) is `cli._s_stop_waits`, called from all
  six predicate sites, each keeping its own extra conjunct outside it.
- H2: the `[<char>/<d>] ref size <ratio>x[ CHECK]` line and the sheet
  label are `cli._print_ref_size`, used by the `s` stop and the full
  review stop; the same inputs print the same lines and carry the same
  labels.

No network, GPU, Blender or ComfyUI: the run is faked at subprocess.run.
"""
import ast
import io
import subprocess
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

import mannequins

from test_review import GEN_DIRS, ReviewFixture
from test_workflow_paths import stubbed_example

from sprite_pipeline import project
from sprite_pipeline import cli as cli_mod

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src" / "sprite_pipeline"


def run_cli(argv, fake=None):
    """Invoke cli.main capturing stdout and stderr; an argparse SystemExit
    becomes its exit code and its message joins the captured output."""
    if fake is not None:
        subprocess.run = fake
    buf, err = io.StringIO(), io.StringIO()
    try:
        with redirect_stdout(buf), redirect_stderr(err):
            code = cli_mod.main(argv)
    except SystemExit as e:
        code = e.code if isinstance(e.code, int) else 1
        buf.write(str(e) + "\n")
    buf.write(err.getvalue())
    return code, buf.getvalue()


# The base dry-run output of the example project, computed once at import
# time on a stubbed copy: the byte-identity reference of H1-r1.
def _base_dry_run():
    with tempfile.TemporaryDirectory() as t:
        p, base = stubbed_example(t)
        code, out = run_cli(["go", str(p), "--dry-run"])
        assert code == 0, out
        return out.replace(str(base), "$TMP")


BASE_DRY_RUN = _base_dry_run()


def direction_still(path, ratio, d):
    """A raw edit on the 832x1024 edit canvas: one standing figure
    `ratio` x direction `d`'s target (the stub mannequin's `dirs[d].body_h`
    at frame_size 704 scaled by 832 / 704; `ne` sits below the 0.85 CHECK
    band)."""
    target = mannequins.DIRS[d]["body_h"] * 832 / 704
    im = mannequins.pose_frame(832, 1024, body_h=round(ratio * target),
                               body_w=200, feet_from_bottom=10)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    im.save(path)


class RatioFixture(ReviewFixture):
    """The example project whose faked edit writes each raw edit at a fixed
    ratio of its direction's target: 0.97 (inside the band) for `s`, `se`,
    `e` and `n`, 0.84 (below the band, so the label and line carry ` CHECK`)
    for `ne`. The expected ratio and label of every direction are stated
    here, independently of production code."""

    def _make_still(self, argv):
        dst = Path(argv[argv.index("--copy-to") + 1])
        d = dst.name.split(".")[0]
        direction_still(dst, 0.84 if d == "ne" else 0.97, d)


# ---------------------------------------------------------------------- H1-r1

class TestSStopPredicate(unittest.TestCase):
    """H1: the one predicate `cli._s_stop_waits` is called from all six
    sites, each keeping its own extra conjunct outside it; the example
    project's `go --dry-run` prints exactly the base output."""

    def test_r1_all_six_sites_call_the_one_predicate(self):
        tree = ast.parse((SRC / "cli.py").read_text())
        sites = []

        class Visitor(ast.NodeVisitor):
            def __init__(self):
                self.stack = []

            def visit_FunctionDef(self, fn):
                self.stack.append(fn.name)
                self.generic_visit(fn)
                self.stack.pop()

            def visit_Call(self, call):
                if (getattr(call.func, "id", None) == "_s_stop_waits"
                        and self.stack and self.stack[-1] != "_s_stop_waits"):
                    sites.append((self.stack[-1], call.lineno))
                # the six sites pass only constants or local names: the
                # arguments need no further descent

        Visitor().visit(tree)
        # the predicate's sites (amendment 12): the `s`-wait of the
        # stills-redo condition (`_redoes_stills`) and of `approve`, the
        # non-`s` edit hold in `run_stills`, and the four of the `go` loop —
        # the s stop's dry-run verdict, the skip-over-the-stop check, the
        # stills-redo disjunct and the s stop's real-run gate
        self.assertEqual(len(sites), 7,
                         f"the predicate sites call the helper: {sorted(sites)}")
        self.assertEqual({name for name, _ in sites},
                         {"_redoes_stills", "cmd_approve", "run_stills", "_go"})
        self.assertEqual([n for n, _ in sorted(sites) if n == "_go"],
                         ["_go"] * 4)
        # the flattened condition is gone from cli.py
        src = (SRC / "cli.py").read_text()
        self.assertNotIn('any(d != "s" for d in', src)

    def test_r1_dry_run_identical_to_base(self):
        with tempfile.TemporaryDirectory() as t:
            p, base = stubbed_example(t)
            code, out = run_cli(["go", str(p), "--dry-run"])
            self.assertEqual(code, 0, out)
            self.assertEqual(out, BASE_DRY_RUN.replace("$TMP", str(base)))


# ---------------------------------------------------------------------- H2-r1

class TestSizeReportHelper(RatioFixture):
    """H2: the one helper `cli._print_ref_size` serves both stops; same
    inputs print the same lines and carry the same labels as the base."""

    def test_r1_both_stops_print_the_base_lines_and_labels(self):
        code, s_out = self.run_cli(["go", str(self.project)], fake=self.fake)
        self.assert_stopped_with_approve_command(code, s_out)
        code, out = self.run_cli(self.approve_cmd())
        self.assertEqual(code, 0, out)
        code, f_out = self.run_cli(["go", str(self.project)], fake=self.fake)
        self.assert_stopped_with_approve_command(code, f_out)
        # the advisory lines: the s stop prints `s` only; the full stop
        # prints every existing direction, s first, in the generate order
        s_lines = [l for l in s_out.splitlines() if "ref size" in l]
        f_lines = [l for l in f_out.splitlines() if "ref size" in l]
        self.assertEqual(s_lines, ["[courier/s] ref size 0.97x"])
        self.assertEqual(f_lines, ["[courier/s] ref size 0.97x",
                                   "[courier/se] ref size 0.97x",
                                   "[courier/e] ref size 0.97x",
                                   "[courier/ne] ref size 0.84x CHECK",
                                   "[courier/n] ref size 0.97x"])
        # the sheet labels: the s sheet holds `s 0.97` among its three
        # panels, the refs sheet labels every tile `se 0.97` and so on
        from PIL import Image, ImageDraw

        TILE_W = round(832 * 340 / 1024)
        STRIP_H = 24

        def strip(text):
            s = Image.new("RGB", (TILE_W, STRIP_H), (40, 42, 52))
            ImageDraw.Draw(s).text((8, 5), text, fill=(230, 230, 230))
            return s

        s_sheet = Image.open(self.out_char / "s-sheet.png")
        self.assertEqual(s_sheet.size, (TILE_W * 3, 340 + STRIP_H))
        self.assertEqual(s_sheet.crop((0, 340, TILE_W, 364)), strip("anchored still"))
        self.assertEqual(s_sheet.crop((TILE_W, 340, 2 * TILE_W, 364)), strip("s guide"))
        self.assertEqual(s_sheet.crop((2 * TILE_W, 340, 3 * TILE_W, 364)), strip("s 0.97"))
        f_sheet = Image.open(self.out_char / "refs-sheet.png")
        self.assertEqual(f_sheet.size, (TILE_W * 5, 340 + STRIP_H))
        expected = {"s": "s 0.97", "se": "se 0.97", "e": "e 0.97",
                    "ne": "ne 0.84 CHECK", "n": "n 0.97"}
        for i, d in enumerate(GEN_DIRS):
            x = i * TILE_W
            self.assertEqual(f_sheet.crop((x, 340, x + TILE_W, 364)),
                             strip(expected[d]), d)


if __name__ == "__main__":
    unittest.main()
