"""The compiled module loads and reports the crate version."""

import screentrack


def test_module_exposes_its_version() -> None:
    assert screentrack.__version__ == "0.1.0"


def test_module_reports_its_build_profile() -> None:
    assert isinstance(screentrack.__debug_build__, bool)
