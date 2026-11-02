"""Hand-written hard cases from scripts/09_hard_cases.py, run as tests.

Cases the guard handles today must keep passing. Known failures are strict xfails: they document
what the flat-slot design cannot do, and a test fails as soon as one starts passing, so this list
has to be updated when the guard improves.
"""
import importlib.util
from pathlib import Path

import pytest

from amana.guard import check

_spec = importlib.util.spec_from_file_location(
    "hard_cases", Path(__file__).resolve().parents[1] / "scripts" / "09_hard_cases.py")
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

KNOWN_FAILURES = {
    "2 drugs: doses swapped",
    "2 drugs: timing swapped",
    "warning flipped 2",
    "wrong prohibition, same count",
    "condition dropped (only if pain)",
    "max dose to min dose",
    "garbled LASA partner: prednisone",
    "esomeprazole-like spelling for omeprazole",
    "SAFE: morning and evening = twice",
    "SAFE: every 12 hours words",
    "SAFE: half tablet",
    "SAFE: word order, dose phrase",
    "SAFE: for two weeks word",
    "SAFE: do not forget prohibition style",
}


def _params():
    for name, should_flag, src, hyp in _mod.CASES:
        marks = [pytest.mark.xfail(strict=True, reason="known guard limit")] if name in KNOWN_FAILURES else []
        yield pytest.param(should_flag, src, hyp, id=name, marks=marks)


@pytest.mark.parametrize("should_flag,src,hyp", list(_params()))
def test_hard_case(should_flag, src, hyp):
    assert bool(check(src, hyp).flags) == should_flag


def test_known_failures_are_real_cases():
    assert KNOWN_FAILURES <= {c[0] for c in _mod.CASES}
