"""Test bootstrap: the mannequin-profile gate (FR 4) of `project.load_project`.

Suite fixtures write stub source GLB bytes (a 2x2 PNG, a raw image dump)
whose hash no packaged profile (FR 14) knows. Each fixture registers the
stub bytes it writes in the test registry (tests/mannequins.py).
`project.anchor_profiles` — the default profile map of `load_project` — is
wrapped here to merge that registry over the packaged and host profiles;
production code knows nothing of the registry. Nothing is discovered from the filesystem: a GLB passes the
gate only if its exact bytes were registered, so the content rule (FR 4) is
unchanged for every other GLB.

A test that wants the gate to *reject* (AC15-r4) writes bytes no fixture
registers, or passes an explicit map through `load_project(..., profiles=…)`.
"""
import mannequins

from sprite_pipeline import project

_anchor_profiles = project.anchor_profiles


def _anchor_profiles_with_stubs():
    """The packaged + host profiles with the registered stub GLBs merged in."""
    profiles = _anchor_profiles()
    profiles.update(mannequins.profiles())
    return profiles


project.anchor_profiles = _anchor_profiles_with_stubs
