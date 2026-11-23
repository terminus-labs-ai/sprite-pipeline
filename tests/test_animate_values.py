"""sprite-pipeline-70: animate workflow values from YAML (spec FR 12-14, AC 9, 10).

Each test plans the real animate argv with `stages`, parses it with the real
`comfy animate` parser and builds the graph `comfy animate` would post
(`comfy.animate_values` + `comfy.animate_graph`), so the stages producer and
the comfy consumer are exercised together. No network, GPU or ComfyUI.
"""
import json
import tempfile
import unittest
from pathlib import Path

import yaml

from sprite_pipeline import comfy, project, stages
from test_workflow_paths import cfg, stubbed_example

SHIPPED = Path(__file__).resolve().parent.parent / "src" / "sprite_pipeline" / "workflows" / "animate.json"

# the shipped animate graph with an sp: title on every FR 12/13 slot node
TITLES = {"1": "sp:unet", "2": "sp:lora,sp:lora_strength", "3": "sp:shift",
          "18": "sp:length,sp:pose_strength,sp:width,sp:height,sp:pose_start_percent,"
                "sp:pose_end_percent,sp:reference_image_strength",
          "19": "sp:sampler", "20": "sp:scheduler,sp:steps", "21": "sp:seed,sp:cfg"}


def titled_graph(drop=()):
    """The shipped graph with TITLES applied, minus the titles in `drop`."""
    g = json.loads(SHIPPED.read_text())
    for nid, title in TITLES.items():
        parts = [t for t in title.split(",") if t not in drop]
        g[nid].setdefault("_meta", {})["title"] = ",".join(parts)
    g["8"]["inputs"]["text"] = "workflow negative"
    return g


def build(edit, graph, pairs=(("run", "s"),)):
    """Load the example project after `edit(yaml_dict)`, with every
    animation using a workflow file holding `graph`. Returns, per (anim, dir)
    pair, the graph `comfy animate` would post for the planned argv."""
    with tempfile.TemporaryDirectory() as t:
        p, base = stubbed_example(t)
        wf = base / "wf.json"
        wf.write_text(json.dumps(graph))
        d = yaml.safe_load(p.read_text())
        for a in d["animations"]:
            a["workflow"] = str(wf)
        edit(d)
        p.write_text(yaml.safe_dump(d))
        proj = project.load_project(p)
        ctx = stages.StageCtx(proj, cfg(Path("/nonexistent-home")))
        out = {}
        for anim, dr in pairs:
            argv = ctx.plan_pair("courier", proj.animation(anim), dr, "animate")[-1][0]
            ns = comfy.build_parser().parse_args(argv[argv.index("animate"):])
            out[(anim, dr)] = comfy.animate_graph(wf, comfy.animate_values(ns), ns.drop_lora)
        return out


def one(edit, graph=None, anim="run", d="s"):
    return build(edit, graph or titled_graph(), ((anim, d),))[(anim, d)]


def set_project(**kw):
    """Set `kw` in the project `animate:` block; a top-level `seed` or
    `pose_strength` the example carries is removed when the block sets it."""
    def edit(d):
        for k in kw:
            d.pop(k, None)
        d.setdefault("animate", {}).update(kw)
    return edit


def anim_block(name, **kw):
    def edit(d):
        next(a for a in d["animations"] if a["name"] == name)["animate"] = kw
    return edit


class TestAC9WorkflowValues(unittest.TestCase):

    def test_AC9_r1_project_block_fills_steps_and_cfg(self):
        g = one(set_project(steps=10, cfg=1.5))
        self.assertEqual(g["20"]["inputs"]["steps"], 10)
        self.assertEqual(g["21"]["inputs"]["cfg"], 1.5)

    def test_AC9_r2_animation_block_wins_over_project(self):
        def edit(d):
            set_project(cfg=1.5)(d)
            anim_block("death", cfg=2.0)(d)
        out = build(edit, titled_graph(), (("death", "se"), ("run", "s")))
        self.assertEqual(out[("death", "se")]["21"]["inputs"]["cfg"], 2.0)
        self.assertEqual(out[("run", "s")]["21"]["inputs"]["cfg"], 1.5)

    def test_AC9_r3_unset_value_leaves_input_and_needs_no_slot(self):
        g = titled_graph(drop=("sp:shift",))
        g["3"]["inputs"]["shift"] = 8
        self.assertEqual(one(lambda d: None, g)["3"]["inputs"]["shift"], 8)

    def test_AC9_r4_set_value_without_slot_is_workflow_error(self):
        with self.assertRaises(comfy.WorkflowError) as cm:
            one(set_project(sampler="lcm"), titled_graph(drop=("sp:sampler",)))
        self.assertIn("wf.json", str(cm.exception))
        self.assertIn("sp:sampler", str(cm.exception))

    def test_AC9_r5_every_key_reaches_its_node_input(self):
        expected = {  # key: (value, node, input)
            "unet": ("u.safetensors", "1", "unet_name"),
            "lora": ("l.safetensors", "2", "lora_name"),
            "lora_strength": (0.4, "2", "strength_model"),
            "sampler": ("euler", "19", "sampler_name"),
            "scheduler": ("beta", "20", "scheduler"),
            "steps": (12, "20", "steps"),
            "shift": (3.5, "3", "shift"),
            "cfg": (2.5, "21", "cfg"),
            "seed": (42, "21", "noise_seed"),
            "pose_strength": (0.85, "18", "pose_strength"),
            "pose_start_percent": (0.1, "18", "pose_start_percent"),
            "pose_end_percent": (0.9, "18", "pose_end_percent"),
            "reference_image_strength": (0.7, "18", "reference_image_strength"),
            "negative": ("no zoom", "8", "text"),
            "pose_prompt": ("a figure falling", "9", "text"),
        }
        for key, (value, node, inp) in expected.items():
            with self.subTest(key=key):
                g = one(set_project(**{key: value}))
                self.assertEqual(g[node]["inputs"][inp], value)

    def test_AC9_r6_top_level_seed_and_pose_strength_fill_slots(self):
        def edit(d):
            d["seed"], d["pose_strength"] = 7, 0.6
        g = one(edit)
        self.assertEqual(g["21"]["inputs"]["noise_seed"], 7)
        self.assertEqual(g["18"]["inputs"]["pose_strength"], 0.6)

    def test_AC9_r7_both_seed_forms_is_project_error(self):
        def edit(d):
            d["seed"] = 3
            d["animate"] = {"seed": 7}
        with self.assertRaises(project.ProjectError) as cm:
            one(edit)
        self.assertIn("project.yaml", str(cm.exception))
        self.assertIn("seed", str(cm.exception))
        self.assertIn("animate.seed", str(cm.exception))

    def test_AC9_r8_override_seed_wins_over_animate_seed(self):
        def edit(d):
            set_project(seed=7)(d)
            d["characters"][0]["overrides"] = {"death-se": {"seed": 9}}
        out = build(edit, titled_graph(), (("death", "se"), ("death", "s")))
        self.assertEqual(out[("death", "se")]["21"]["inputs"]["noise_seed"], 9)
        self.assertEqual(out[("death", "s")]["21"]["inputs"]["noise_seed"], 7)

    def test_unknown_animate_key_is_project_error_naming_the_path(self):
        with self.assertRaises(project.ProjectError) as cm:
            one(anim_block("run", stepz=3))
        self.assertIn("project.yaml: animations[0].animate.stepz: unknown key", str(cm.exception))


class TestAC10NoLora(unittest.TestCase):

    def test_AC10_r1_lora_none_removes_node_and_rewires(self):
        g = one(set_project(lora="none", lora_strength=0.6))
        self.assertNotIn("2", g)
        self.assertEqual(g["3"]["inputs"]["model"], ["1", 0])
        refs = [v for n in g.values() for v in n["inputs"].values()
                if isinstance(v, list) and len(v) == 2 and isinstance(v[0], str)]
        self.assertTrue(all(r[0] in g for r in refs), refs)

    def test_AC10_r2_lora_none_without_slot_leaves_graph(self):
        graph = titled_graph(drop=("sp:lora", "sp:lora_strength"))
        g = one(set_project(lora="none"), graph)
        self.assertIn("2", g)
        self.assertEqual(g["3"]["inputs"]["model"], ["2", 0])
        self.assertEqual(g["2"], graph["2"])

    def test_AC10_r3_lora_file_kept_and_filled(self):
        g = one(set_project(lora="foo.safetensors", lora_strength=0.6))
        self.assertEqual(g["2"]["inputs"]["lora_name"], "foo.safetensors")
        self.assertEqual(g["2"]["inputs"]["strength_model"], 0.6)
        self.assertEqual(g["3"]["inputs"]["model"], ["2", 0])


class TestDropLoraUnit(unittest.TestCase):

    def test_every_consumer_of_the_lora_is_rewired(self):
        g = {"1": {"inputs": {}}, "2": {"inputs": {"model": ["1", 0]}, "_meta": {"title": "sp:lora"}},
             "3": {"inputs": {"model": ["2", 0]}}, "4": {"inputs": {"model": ["2", 0], "x": ["3", 0]}}}
        out = comfy.drop_lora(g)
        self.assertEqual(out, {"1": {"inputs": {}}, "3": {"inputs": {"model": ["1", 0]}},
                               "4": {"inputs": {"model": ["1", 0], "x": ["3", 0]}}})


if __name__ == "__main__":
    unittest.main()
