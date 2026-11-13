"""Tests for sprite_pipeline.build: manifest loading/validation, cache keying, dry-run
command planning, skip-if-exists, .tres writer, mirror mapping. No network, no GPU,
no real clips.
"""
import dataclasses
import re
import sys
import tempfile
import unittest
from pathlib import Path

from sprite_pipeline import build

GLB_REL = "ual/Animation Library[Standard]/Godot/AnimationLibrary_Godot_Standard.glb"
ANIM_SRC = Path("/anim-src")
GLB = str(ANIM_SRC / GLB_REL)
MOTIONS = build.TOOLS / "motions.yaml"
FIXTURES = Path(__file__).resolve().parent / "fixtures"
KIT_FILE = FIXTURES / "vexa-street.yaml"


def load_ctx(tmp, move_names=("run",), dirs=("se",), force=False):
    """Real manifests, temp paths, so stage_done() sees nothing."""
    tmp = Path(tmp)
    paths = build.Paths(cache=tmp / "motion-cache", comfy_input=tmp / "input",
                        blender="/bin/blender", sprite_gen="/bin/sprite-gen")
    lib = build.resolve_glbs(build.load_motion_lib(MOTIONS), ANIM_SRC)
    kit = build.Kit.from_dict(build.load_yaml(KIT_FILE))
    kit = dataclasses.replace(kit, output_root=tmp / "out")
    ctx = build.Ctx(lib, kit, list(move_names), list(dirs), paths, force=force)
    return ctx, lib, kit, paths


class TestManifests(unittest.TestCase):
    def test_mini_yaml_parses_real_manifests(self):
        m = build.mini_yaml_loads(MOTIONS.read_text())
        self.assertEqual(m["elev"], 45)
        self.assertEqual(m["default_glb"], GLB_REL)
        run = m["moves"]["run"]
        self.assertEqual(run["action"], "Jog_Fwd_Loop")
        self.assertEqual((run["repeats"], run["pre"], run["post"]), (2, 1, 0))
        self.assertEqual(run["clip_length"], 45)
        self.assertEqual(run["pose_prompt"], "{who} jogging in place, head up")
        self.assertEqual(run["head_pitch"], 25)
        self.assertEqual(run["directions"], ["s", "se", "e", "ne", "n"])
        punch = m["moves"]["punch"]
        self.assertEqual(punch["pose_prompt"], "{who} throwing a fast cross punch, head up")
        self.assertEqual(punch["clip_length"], 33)
        self.assertEqual(m["moves"]["pistol_shoot"]["clip_length"], 21)

        k = build.mini_yaml_loads(KIT_FILE.read_text())
        self.assertEqual(k["name"], "vexa-street")
        self.assertEqual(k["seed"], 7)
        self.assertEqual(k["frame_size"], 704)
        self.assertEqual(k["pixel_scale"], 0.2023)
        self.assertEqual(k["face_dirs"], ["s", "se", "e"])
        self.assertEqual(k["refs"]["se"], "vexa-ref-se.png")
        self.assertTrue(k["look"].startswith("Vexa, an adult woman hacker:"))
        self.assertIn("full body always inside the frame.", k["look"])
        self.assertTrue(k["face"].startswith(" She keeps her head up"))
        self.assertIn("chroma-key green", k["background"])

    def test_mini_yaml_agrees_with_pyyaml(self):
        try:
            import yaml
        except ModuleNotFoundError:
            self.skipTest("PyYAML not importable in this interpreter")
        for f in (MOTIONS, KIT_FILE):
            self.assertEqual(yaml.safe_load(f.read_text()), build.mini_yaml_loads(f.read_text()), f)

    def test_loaded_clip_lengths_are_4n1(self):
        lib = build.load_motion_lib(MOTIONS)
        for name, mv in lib.moves.items():
            self.assertEqual(mv.clip_length % 4, 1, f"{name} clip_length {mv.clip_length}")

    def test_validation_rejects_bad_clip_length(self):
        ctx, lib, kit, _ = load_ctx(tempfile.mkdtemp())
        bad = dataclasses.replace(lib, moves=dict(lib.moves, **{"run": dataclasses.replace(lib.moves["run"], clip_length=44)}))
        with self.assertRaises(ValueError) as cm:
            build.validate(bad, kit, ["run"], ["se"])
        self.assertIn("4n+1", str(cm.exception))

    def test_validation_rejects_unknown_direction(self):
        ctx, lib, kit, _ = load_ctx(tempfile.mkdtemp())
        with self.assertRaises(ValueError) as cm:
            build.validate(lib, kit, ["run"], ["sw"])
        self.assertIn("unknown generation direction", str(cm.exception))
        bad = dataclasses.replace(lib, moves=dict(lib.moves, **{"run": dataclasses.replace(lib.moves["run"], directions=["x"])}))
        with self.assertRaises(ValueError) as cm:
            build.validate(bad, kit, ["run"], ["s"])
        self.assertIn("unknown directions", str(cm.exception))

    def test_validation_rejects_unknown_move_and_bad_state(self):
        ctx, lib, kit, _ = load_ctx(tempfile.mkdtemp())
        with self.assertRaises(ValueError) as cm:
            build.validate(lib, kit, ["yolo"], ["s"])
        self.assertIn("unknown move", str(cm.exception))
        bad = dataclasses.replace(lib, moves=dict(lib.moves, **{"run": dataclasses.replace(lib.moves["run"], state="flap")}))
        with self.assertRaises(ValueError) as cm:
            build.validate(bad, kit, ["run"], ["se"])
        self.assertIn("state", str(cm.exception))

    def test_prompt_assembly_face_dirs(self):
        ctx, lib, kit, _ = load_ctx(tempfile.mkdtemp())
        self.assertEqual(kit.prompt_for("se"), kit.look + kit.face + kit.background)
        self.assertEqual(kit.prompt_for("ne"), kit.look + kit.background)


class TestConfig(unittest.TestCase):
    def test_defaults_under_home_and_overrides(self):
        import os
        from sprite_pipeline import config
        with tempfile.TemporaryDirectory() as tmp:
            old = os.environ.get("SPRITE_PIPELINE_HOME")
            os.environ["SPRITE_PIPELINE_HOME"] = tmp
            try:
                c = config.load()
                self.assertEqual(c["motion_cache"], str(Path(tmp) / "motion-cache"))
                self.assertEqual(c["blender"], "blender")
                (Path(tmp) / "config.toml").write_text('blender = "/opt/blender"\nanim_src = "/srv/anim"\n')
                c = config.load()
                self.assertEqual((c["blender"], c["anim_src"]), ("/opt/blender", "/srv/anim"))
                (Path(tmp) / "config.toml").write_text('bogus = 1\n')
                with self.assertRaises(SystemExit):
                    config.load()
            finally:
                if old is None:
                    os.environ.pop("SPRITE_PIPELINE_HOME")
                else:
                    os.environ["SPRITE_PIPELINE_HOME"] = old

    def test_relative_glb_resolves_against_anim_src_absolute_kept(self):
        lib = build.load_motion_lib(MOTIONS)
        self.assertEqual(build.resolve_glbs(lib, ANIM_SRC).moves["run"].glb, GLB)
        absolute = dataclasses.replace(lib, moves={"run": dataclasses.replace(lib.moves["run"], glb="/x/a.glb")})
        self.assertEqual(build.resolve_glbs(absolute, ANIM_SRC).moves["run"].glb, "/x/a.glb")


class TestDriveCacheKey(unittest.TestCase):
    def test_key_changes_with_params(self):
        m = build.load_motion_lib(MOTIONS).moves["run"]
        base = build.drive_hash(m, 45, 704)
        self.assertEqual(len(base), 16)
        self.assertNotEqual(base, build.drive_hash(dataclasses.replace(m, repeats=3), 45, 704))
        self.assertNotEqual(base, build.drive_hash(dataclasses.replace(m, pre=2), 45, 704))
        self.assertNotEqual(base, build.drive_hash(dataclasses.replace(m, head_pitch=10.0), 45, 704))
        self.assertNotEqual(base, build.drive_hash(dataclasses.replace(m, action="Walk_Loop"), 45, 704))
        self.assertNotEqual(base, build.drive_hash(m, 45, 896))
        self.assertEqual(base, build.drive_hash(m, 45, 704))  # stable


class TestDryRunCommands(unittest.TestCase):
    """The planned command list for run/se, including the face fix (head pitch 25, pose strength 0.75)."""

    def test_drive_commands(self):
        ctx, lib, kit, p = load_ctx(tempfile.mkdtemp(), ("run",), ("se",))
        m = lib.moves["run"]
        h = ctx.hashes["run"]
        frames = str(p.cache / f"run-se-{h}" / "frames")
        cmds = ctx.plan_pair(m, "se", "drive")
        self.assertEqual(cmds[0][0],
                         [p.blender, "-b", "--python", str(p.tools / "render_drive.py"), "--",
                          GLB, "Jog_Fwd_Loop", frames, "45", "45", "2", "1", "0", "25"])
        self.assertEqual(cmds[1][0],
                         ["ffmpeg", "-v", "error", "-y", "-framerate", "30",
                          "-i", f"{frames}/f%04d.png", "-c:v", "libx264", "-crf", "12",
                          "-pix_fmt", "yuv420p", str(p.cache / f"run-se-{h}.mp4")])
        self.assertEqual(cmds[2][0],
                         ["cp", "-f", str(p.cache / f"run-se-{h}.mp4"),
                          str(p.comfy_input / f"drive-run-se-{h}.mp4")])
        self.assertEqual(ctx.env["RW"], "704")
        self.assertEqual(ctx.env["RH"], "704")

    def test_animate_command(self):
        ctx, lib, kit, p = load_ctx(tempfile.mkdtemp(), ("run",), ("se",))
        h = ctx.hashes["run"]
        argv = ctx.plan_pair(lib.moves["run"], "se", "animate")[0][0]
        self.assertEqual(argv,
                         [sys.executable, "-m", "sprite_pipeline.comfy_run", "animate",
                          "--ref", "vexa-ref-se.png", "--video", f"drive-run-se-{h}.mp4",
                          "--w", "704", "--h", "704", "--length", "45", "--seed", "7",
                          "--prefix", "vexa-street/dir8-run-se",
                          "--copy-to", str(kit.output_root / "clips" / "run-se.mp4"),
                          "--pose-prompt", "a woman jogging in place, head up",
                          "--prompt", kit.look + kit.face + kit.background,
                          "--pose-strength", "0.75"])

    def test_overrides_change_seed_and_prompt_for_one_pair(self):
        ctx, lib, kit, p = load_ctx(tempfile.mkdtemp(), ("run", "punch"), ("se", "n"))
        kit.overrides = {"punch-n": {"seed": 9, "extra": "No weapon."}}
        a = ctx.plan_pair(lib.moves["punch"], "n", "animate")[0][0]
        self.assertEqual(a[a.index("--seed") + 1], "9")
        self.assertEqual(a[a.index("--prompt") + 1], kit.look + " No weapon." + kit.background)
        b = ctx.plan_pair(lib.moves["run"], "n", "animate")[0][0]
        self.assertEqual(b[b.index("--seed") + 1], "7")
        self.assertNotIn("No weapon.", b[b.index("--prompt") + 1])

    def test_guard_kit_loads_and_validates(self):
        g = build.Kit.from_dict(build.load_yaml(FIXTURES / "guard.yaml"))
        lib = build.load_motion_lib(MOTIONS)
        build.validate(lib, g, ["run", "punch"], ["s", "se", "e", "ne", "n"])
        self.assertEqual(g.pose_prompt(lib.moves["run"]), "a man jogging in place, head up")
        self.assertEqual(g.override("punch", "n")["seed"], 9)

    def test_key_and_loop_commands(self):
        ctx, lib, kit, p = load_ctx(tempfile.mkdtemp(), ("run", "punch"), ("se",))
        key_argv = ctx.plan_pair(lib.moves["run"], "se", "key")[0][0]
        self.assertEqual(key_argv,
                         [p.sprite_gen, "video-frames", "--clip",
                          str(kit.output_root / "clips" / "run-se.mp4"),
                          "--out-dir", str(kit.output_root / "work" / "frames-run-se"),
                          "--spill", "small", "--allow-subject-edge-contact"])
        loop_argv = ctx.plan_pair(lib.moves["run"], "se", "loop")[0][0]
        self.assertEqual(loop_argv,
                         [p.sprite_gen, "video-loop",
                          "--frames-dir", str(kit.output_root / "work" / "frames-run-se" / "keyed"),
                          "--out-dir", str(kit.output_root / "loops" / "run-se"),
                          "--fps", "30", "--state", "run", "--cycle", "fixed",
                          "--start", "1", "--length", "22", "--seam-pop-max", "off",
                          "--name", "se-run"])
        punch_argv = ctx.plan_pair(lib.moves["punch"], "se", "loop")[0][0]
        self.assertEqual(punch_argv,
                         [p.sprite_gen, "video-loop",
                          "--frames-dir", str(kit.output_root / "work" / "frames-punch-se" / "keyed"),
                          "--out-dir", str(kit.output_root / "loops" / "punch-se"),
                          "--fps", "30", "--state", "attack", "--cycle", "once",
                          "--start", "0", "--length", "33", "--name", "se-punch"])

    def test_pixel_plan_dir_major_sets(self):
        ctx, lib, kit, p = load_ctx(tempfile.mkdtemp(), ("run", "punch"), ("s", "se"))
        argv = ctx.plan_kit_stage("pixel")[0][0]
        i = argv.index("--ref")
        self.assertEqual(argv[:3], [sys.executable, "-m", "sprite_pipeline.pixelize"])
        self.assertEqual(argv[i:i + 10],
                         ["--ref", kit.palette, "--scale", "0.2023", "--colors", "40",
                          "--accent-boost", "3", "--fps", "30"])
        sets = [x for j, x in enumerate(argv) if argv[j - 1] == "--set"]
        self.assertEqual(sets, [
            f"s-run={kit.output_root}/loops/run-s/cycle",
            f"s-punch={kit.output_root}/loops/punch-s/cycle",
            f"se-run={kit.output_root}/loops/run-se/cycle",
            f"se-punch={kit.output_root}/loops/punch-se/cycle",
        ])


class TestSkipIfExists(unittest.TestCase):
    def test_stage_done_predicates_and_force(self):
        tmp = tempfile.mkdtemp()
        ctx, lib, kit, p = load_ctx(tmp, ("run",), ("se",))
        m = lib.moves["run"]
        for stage in build.PAIR_STAGES:
            self.assertFalse(ctx.stage_done(stage, m, "se"))
        h = ctx.hashes["run"]
        p.cache.mkdir(parents=True, exist_ok=True)
        p.comfy_input.mkdir(parents=True, exist_ok=True)
        (p.cache / f"run-se-{h}.mp4").write_bytes(b"x")
        (p.comfy_input / f"drive-run-se-{h}.mp4").write_bytes(b"x")
        self.assertTrue(ctx.stage_done("drive", m, "se"))
        (kit.output_root / "clips").mkdir(parents=True)
        (kit.output_root / "clips" / "run-se.mp4").write_bytes(b"x")
        self.assertTrue(ctx.stage_done("animate", m, "se"))
        keyed = kit.output_root / "work" / "frames-run-se" / "keyed"
        keyed.mkdir(parents=True)
        (keyed / "000.png").write_bytes(b"x")
        self.assertTrue(ctx.stage_done("key", m, "se"))
        cyc = kit.output_root / "loops" / "run-se" / "cycle"
        cyc.mkdir(parents=True)
        (cyc / "000.png").write_bytes(b"x")
        self.assertTrue(ctx.stage_done("loop", m, "se"))
        # stage_done itself is force-agnostic; main() gates it on not force
        forced = build.Ctx(lib, kit, ["run"], ["se"], p, force=True)
        self.assertTrue(forced.stage_done("loop", m, "se"))


class TestTresWriter(unittest.TestCase):
    def test_write_and_parse_back(self):
        with tempfile.TemporaryDirectory() as tmp:
            f = Path(tmp) / "sf.tres"
            build.write_tres(f, "atlas.png",
                             [("run_s", True, [(0, 0, 10, 10), (10, 0, 10, 10)]),
                              ("punch_se", False, [(0, 10, 8, 8)])])
            text = f.read_text()
            self.assertIn('[gd_resource type="SpriteFrames" load_steps=5 format=3]', text.splitlines()[0])  # self + 1 ext + 3 subs
            self.assertIn('[ext_resource type="Texture2D" path="atlas.png" id="1_atlas"]', text)
            subs = {m[0]: tuple(m[1:]) for m in re.findall(
                r'\[sub_resource type="AtlasTexture" id="([^"]+)"\]\natlas = ExtResource\("1_atlas"\)\nregion = Rect2\((\d+), (\d+), (\d+), (\d+)\)',
                text)}
            anims = re.findall(
                r'\{\n"frames": \[(.*?)\],\n"loop": (true|false),\n"name": &"([^"]+)",\n"speed": ([0-9.]+)\n\}',
                text, re.S)
            self.assertEqual([a[2] for a in anims], ["run_s", "punch_se"])
            self.assertEqual([a[1] == "true" for a in anims], [True, False])
            self.assertEqual([a[3] for a in anims], ["30.0", "30.0"])
            self.assertEqual(re.findall(r'SubResource\("([^"]+)"\)', anims[0][0]),
                             ["AtlasTexture_run_s_f0", "AtlasTexture_run_s_f1"])
            self.assertEqual(subs["AtlasTexture_run_s_f0"], ("0", "0", "10", "10"))
            self.assertEqual(subs["AtlasTexture_run_s_f1"], ("10", "0", "10", "10"))
            self.assertEqual(subs["AtlasTexture_punch_se_f0"], ("0", "10", "8", "8"))
            self.assertIn('[resource]', text)
            self.assertIn('\nanimations = [{', text)  # Godot 4: untyped array of dicts
            self.assertEqual(text.count('"duration": 1.0'), 3)  # one frame dict per frame


class TestMirrorMapping(unittest.TestCase):
    def test_mirror_table_and_order(self):
        self.assertEqual(build.MIRROR_OF, {"sw": "se", "w": "e", "nw": "ne"})
        self.assertEqual(build.EIGHT_DIRS, ["n", "ne", "e", "se", "s", "sw", "w", "nw"])
        for d in ("s", "se", "e", "ne", "n"):
            self.assertNotIn(d, build.MIRROR_OF)

    def test_mirror_flips_horizontally(self):
        from PIL import Image
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            row = [(255, 0, 0, 255), (0, 255, 0, 255), (0, 0, 255, 255), (255, 255, 0, 255)]
            src = tmp / "se-run_000.png"
            im = Image.new("RGBA", (4, 1))
            im.putdata(row)
            im.save(src)
            flipped = Image.open(src).convert("RGBA").transpose(Image.FLIP_LEFT_RIGHT)
            self.assertEqual(list(flipped.getdata()), list(reversed(row)))


class TestExport(unittest.TestCase):
    def test_export_packs_available_dirs_and_flags_missing(self):
        from PIL import Image
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            ctx, lib, kit, p = load_ctx(tmp, ("run",), ("s", "se"))
            out = ctx.sprite_out()
            out.mkdir(parents=True)
            row = [(255, 0, 0, 255), (0, 255, 0, 255), (0, 0, 255, 255), (255, 255, 0, 255)]
            for d in ("s", "se"):
                for i in range(2):
                    im = Image.new("RGBA", (4, 1))
                    im.putdata(row if i == 0 else list(reversed(row)))
                    im.save(out / f"{d}-run_{i:03d}.png")
            ok, missing = build.do_export(ctx, [], dry=False)
            self.assertFalse(ok)
            # sw mirrors se, so it is present; the other four real dirs were never built
            self.assertEqual(missing, ["run/n", "run/ne", "run/e", "run/w", "run/nw"])
            self.assertTrue((out / "atlas.png").is_file())
            self.assertTrue((out / "contact-sheet.png").is_file())
            text = (out / "sprite_frames.tres").read_text()
            self.assertIn('"name": &"run_s"', text)
            self.assertIn('"name": &"run_se"', text)
            self.assertNotIn('"name": &"run_n"', text)
            self.assertIn("loop\": true", text)  # run is a fixed loop
            # mirrored sw would flip; s has no mirror, check se mirror is written
            self.assertTrue((out / "sw-run_000.png").is_file())
            sw0 = list(Image.open(out / "sw-run_000.png").convert("RGBA").getdata())
            se0 = list(Image.open(out / "se-run_000.png").convert("RGBA").getdata())
            self.assertEqual(sw0, list(reversed(se0)))


    def test_export_ignores_pixelize_sheet_strips(self):
        """pixelize.py writes <dir>-<move>_sheet.png next to the frames; export must not pack them."""
        from PIL import Image
        with tempfile.TemporaryDirectory() as tmp:
            ctx, lib, kit, p = load_ctx(Path(tmp), ("run",), ("s", "se", "e", "ne", "n"))
            out = ctx.sprite_out(); out.mkdir(parents=True)
            for d in ("s", "se", "e", "ne", "n"):
                for i in range(2):
                    Image.new("RGBA", (4, 4), (255, 0, 0, 255)).save(out / f"{d}-run_{i:03d}.png")
                Image.new("RGBA", (12, 4)).save(out / f"{d}-run_sheet.png")
            log = []
            ok, missing = build.do_export(ctx, log, dry=False)
            self.assertTrue(ok, log)
            self.assertEqual(missing, [])
            self.assertEqual(Image.open(out / "atlas.png").size, (8, 32))

if __name__ == "__main__":
    unittest.main()
