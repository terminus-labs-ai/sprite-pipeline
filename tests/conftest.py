"""Test bootstrap: the mannequin-profile gate (FR 4) of `project.load_project`.

Every suite fixture writes stub source GLB bytes (a 2x2 PNG, a raw image
dump) whose hash no packaged profile (FR 14) knows, and the fixtures create
them in all kinds of places — `setUp`, the test body, a module-level helper
that some suites reach through their own copy of `tempfile`. Rather than
teach every one of them about the gate, this bootstrap registers each stub
GLB found under the system temp root (where every test project lives) into
the test registry (tests/mannequins.py), around every test and around every
test-class setup. `project.anchor_profiles` — the default profile map of
`load_project` and of `go`'s anchoring — is wrapped here to merge that
registry over the packaged and host profiles; production code knows nothing
of the registry. The content rule (FR 4) is unchanged for
every other GLB: a project outside the temp tree is still gated on its sha256.

A test that wants the gate to *reject* (AC15-r4) uses a source GLB outside
the temp tree, or the explicit profile map of `load_project(..., profiles=…)`.
See tests/mannequins.py for the registry and the figure-image helpers.
"""
import functools
import shutil
import tempfile
import unittest
from pathlib import Path

import mannequins

from sprite_pipeline import project

_anchor_profiles = project.anchor_profiles


def _anchor_profiles_with_stubs():
    """The packaged + host profiles with the registered stub GLBs merged in."""
    profiles = _anchor_profiles()
    profiles.update(mannequins.profiles())
    return profiles


project.anchor_profiles = _anchor_profiles_with_stubs

GLB_SUFFIXES = (".glb", ".gltf")
# every TemporaryDirectory in the suite (and pytest's tmp_path tree) lives here
TEST_TMP_ROOT = Path(tempfile.gettempdir())
_SEEN = set()


def register_glb(glb):
    """Register the canonical stub mannequin profile for one GLB file."""
    try:
        payload = glb.read_bytes()
    except OSError:
        return
    mannequins.register(payload, glb_name=glb.name,
                        resolution=mannequins.RESOLUTION)


def register_glbs_under(root):
    """Register every source GLB under `root` whose path has not been seen
    yet (idempotent per path)."""
    root = Path(root)
    if not root.is_dir():
        return
    for glb in sorted(root.rglob("*")):
        if glb.suffix in GLB_SUFFIXES and glb.is_file() and str(glb) not in _SEEN:
            _SEEN.add(str(glb))
            register_glb(glb)


def _register_written(dst):
    """The copytree hook: register a source GLB the moment it is written, so a
    fixture that builds its project inline passes the gate on its first load."""
    if Path(dst).suffix in GLB_SUFFIXES:
        register_glb(Path(dst))
    return dst


def _copytree_with_registration(src, dst, *args, **kwargs):
    """`shutil.copytree` that registers every source GLB it copies."""
    names = shutil._original_copytree(src, dst, *args, **kwargs)
    src = Path(src)
    for copied in Path(dst).rglob("*"):
        if copied.suffix in GLB_SUFFIXES and copied.is_file() and str(copied) not in _SEEN:
            _SEEN.add(str(copied))
            register_glb(copied)
    return names


shutil._original_copytree = shutil.copytree
shutil.copytree = _copytree_with_registration


def _wrap(cls, name):
    """Run the scan before `cls`'s `name` (a setUp or a setUpClass)."""
    original = getattr(cls, name)

    def wrapper(*args, **kwargs):
        register_glbs_under(TEST_TMP_ROOT)
        return original(*args, **kwargs)
    setattr(cls, name, wrapper)


def pytest_collection_finish(session):
    """Wrap setUp and setUpClass of every collected TestCase class: the scan
    lands before the fixture that writes the project, however the fixture
    names its temp directory."""
    for item in session.items:
        cls = getattr(item, "cls", None)
        if not (cls and issubclass(cls, unittest.TestCase)):
            continue
        for name in ("setUp", "setUpClass"):
            fn = getattr(cls, name, None)
            if fn is not None and not getattr(fn, "_sp_glb_scan", False):
                _wrap(cls, name)
                getattr(cls, name)._sp_glb_scan = True


def pytest_runtest_setup(item):
    register_glbs_under(TEST_TMP_ROOT)


def pytest_runtest_call(item):
    """The scan again right before the test body runs: a fixture that writes
    its stub GLB in setUp (or in the body) lands in the registry before the
    first `load_project` call, whatever names its temp directory."""
    register_glbs_under(TEST_TMP_ROOT)
