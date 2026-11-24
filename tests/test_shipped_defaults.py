"""sprite-pipeline-71: shipped animate workflow and example YAML defaults
(spec FR 15, 16; AC 11)."""
import re
import tempfile
import unittest
from pathlib import Path

from sprite_pipeline import comfy, project
from test_workflow_paths import stubbed_example

ROOT = Path(__file__).resolve().parent.parent
SHIPPED = ROOT / "src" / "sprite_pipeline" / "workflows" / "animate.json"
EXAMPLE = ROOT / "examples" / "isometric" / "project.yaml"
NEGATIVE = "camera movement, zoom in, zoom out, pan, tilt, tracking shot, cropped limbs"
SLOTS = ("unet", "sampler", "scheduler", "steps", "shift", "cfg", "seed", "pose_strength",
         "pose_start_percent", "pose_end_percent", "reference_image_strength",
         "negative", "pose_prompt")


def by_slot(graph, slot):
    return [graph[n] for n in comfy.slot_nodes(graph).get(slot, [])]


class ShippedWorkflow(unittest.TestCase):
    def setUp(self):
        self.g = comfy.load_workflow(SHIPPED)

    def test_r1_every_slot_titled_and_no_lora(self):
        for slot in SLOTS:
            self.assertTrue(by_slot(self.g, slot), f"no node titled sp:{slot}")
        self.assertFalse([n for n in self.g.values() if n["class_type"].startswith("LoraLoader")])
        self.assertNotIn("lora", comfy.slot_nodes(self.g))
        # nothing reads a removed node
        for node in self.g.values():
            for v in node["inputs"].values():
                if isinstance(v, list) and len(v) == 2 and isinstance(v[0], str):
                    self.assertIn(v[0], self.g)
        self.assertEqual(self.g["3"]["inputs"]["model"], ["1", 0])

    def test_r2_values(self):
        expect = {"unet": ("unet_name", "wan_animate_2_distill_int8_convrot.safetensors"),
                  "sampler": ("sampler_name", "lcm"), "scheduler": ("scheduler", "simple"),
                  "steps": ("steps", 10), "shift": ("shift", 5), "cfg": ("cfg", 1.5),
                  "pose_strength": ("pose_strength", 0.85),
                  "reference_image_strength": ("reference_image_strength", 1.0),
                  "pose_start_percent": ("pose_start_percent", 0.0),
                  "pose_end_percent": ("pose_end_percent", 1.0),
                  "seed": ("noise_seed", 7)}
        for slot, (inp, val) in expect.items():
            for node in by_slot(self.g, slot):
                self.assertEqual(node["inputs"][inp], val, slot)
        neg = by_slot(self.g, "negative")[0]["inputs"]["text"]
        self.assertTrue(neg.endswith(NEGATIVE), neg)
        self.assertEqual(neg, project.ANIMATE_DEFAULTS["negative"])


class ExampleYaml(unittest.TestCase):
    def test_r3_loads_with_defaults(self):
        with tempfile.TemporaryDirectory() as t:
            p = project.load_project(stubbed_example(t)[0])
        self.assertEqual(list(p.drive.background), [30, 202, 24])
        self.assertEqual(p.drive.render_size, 1152)
        self.assertEqual(p.drive.scale, 0.605)
        self.assertEqual(p.drive.margin, 1.3)
        self.assertEqual(list(p.drive.aspect), [16, 9])
        self.assertEqual(p.reference.height_ratio, 1.07)
        self.assertEqual(p.seed, 7)
        self.assertEqual(p.pose_strength, 0.85)
        self.assertEqual(p.animate, {
            "unet": "wan_animate_2_distill_int8_convrot.safetensors", "lora": "none",
            "lora_strength": 1.0, "sampler": "lcm", "scheduler": "simple", "steps": 10,
            "shift": 5, "cfg": 1.5, "pose_start_percent": 0.0, "pose_end_percent": 1.0,
            "reference_image_strength": 1.0, "negative": NEGATIVE})
        death = p.animation("death")
        self.assertEqual(death.clip_frames, 65)
        self.assertIs(death.facing_line, False)

    def test_r4_every_key_listed_with_comment(self):
        text = EXAMPLE.read_text()
        keys = ("drive", "background", "render_size", "scale", "margin", "aspect",
                "reference", "height_ratio", "animate", "unet", "lora", "lora_strength",
                "sampler", "scheduler", "steps", "shift", "cfg", "pose_strength",
                "pose_start_percent", "pose_end_percent", "reference_image_strength",
                "negative", "height_scale", "clip_frames", "facing_line", "drive_scale",
                "pixel_scale")
        lines = text.splitlines()
        for k in keys:
            self.assertRegex(text, rf"\b{k}\b", k)
            # the key's own line carries a comment, or a comment names it
            self.assertTrue(any("#" in l and re.search(rf"\b{k}\b", l) for l in lines),
                            f"{k} has no comment")
        self.assertNotRegex(text, r"(?m)^(seed|pose_strength):")


if __name__ == "__main__":
    unittest.main()
