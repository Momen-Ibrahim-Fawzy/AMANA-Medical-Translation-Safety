"""AMANA-Guard 2: detection, non-detection and robustness. The cases here are development data."""
import importlib.util
import random
from pathlib import Path

import pytest

from amana import guard2 as G2
from amana import lexicon as L
from amana.rx import ERROR_TYPES, Style, applicable_errors, inject_error, render_ar, render_en, sample_rx

_spec = importlib.util.spec_from_file_location(
    "hard_cases", Path(__file__).resolve().parents[1] / "scripts" / "09_hard_cases.py")
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)


def kinds(src, hyp):
    r = G2.check(src, hyp)
    return {f.kind for f in r.flags}, {f.kind for f in r.review}


def test_source_code_has_no_control_characters():
    # a heredoc once turned a regex \b into a backspace, silently disabling warning extraction
    text = Path(G2.__file__).read_text(encoding="utf-8")
    assert not any(ord(c) < 32 and c not in "\n\r\t" for c in text)


@pytest.mark.parametrize("name,should_flag,src,hyp", _mod.CASES, ids=[c[0] for c in _mod.CASES])
def test_dev_hard_cases(name, should_flag, src, hyp):
    assert (not G2.check(src, hyp).ok) == should_flag


def test_renderer_output_is_accepted_and_errors_are_caught():
    # a regression test, not validation: the generator and the guard share a vocabulary
    rng = random.Random(3)
    for _ in range(200):
        rx = sample_rx(rng, rng.choice(L.DRUGS))
        src = render_en(rx)
        assert G2.check(src, render_ar(rx)).ok
        assert G2.check(src, render_ar(rx, Style.random_benign(rng, rx))).ok


@pytest.mark.parametrize("kind", ERROR_TYPES)
def test_each_error_type_does_not_pass(kind):
    rng = random.Random(1)
    total = 0
    for _ in range(300):
        rx = sample_rx(rng, rng.choice(L.DRUGS))
        if kind in applicable_errors(rx):
            bad, _ = inject_error(rng, rx, kind)
            total += 1
            assert not G2.check(render_en(rx), render_ar(bad)).ok
    assert total > 0


def test_doses_stay_with_their_drug():
    src = "Take metformin 500 mg twice daily and atorvastatin 20 mg once daily."
    assert G2.check(src, "خذ أتورفاستاتين 20 ملغ مرة يوميا وميتفورمين 500 ملغ مرتين يوميا.").ok
    assert "strength" in kinds(src, "خذ أتورفاستاتين 500 ملغ مرة يوميا وميتفورمين 20 ملغ مرتين يوميا.")[0]


def test_lethal_unit_errors_are_flags_not_review():
    src = "Take 1 tablet of levothyroxine 25 mcg by mouth once daily."
    flags, _ = kinds(src, "خذ قرصا واحدا من ليفوثيروكسين 25 مليغراما عن طريق الفم مرة واحدة يوميا.")
    assert "unit" in flags
    src = "Take 1 tablet of methotrexate 2.5 mg by mouth once weekly."
    flags, _ = kinds(src, "خذ قرصا واحدا من ميثوتريكسات 2.5 ملغ عن طريق الفم مرة واحدة يوميا.")
    assert "frequency" in flags


def test_what_cannot_be_read_is_review_not_ok():
    src = "Take 2 capsules of omeprazole 20 mg by mouth twice daily."
    r = G2.check(src, "خذ كبسولات من أوميبرازول 20 ملغ عن طريق الفم مرتين يوميا.")
    assert r.status == "needs_review"


def test_drug_name_inside_a_word_is_not_a_warning():
    src = "Take 1 tablet of levothyroxine 50 mcg by mouth once daily."
    assert G2.check(src, "خذ 1 قرص من ليفوتيروكسين 50 مكغ عن طريق الفم مرة يوميا.").ok


def test_topic_word_in_ordinary_text_is_not_a_warning():
    assert not G2.parse("Use an alcohol-based hand sanitizer.", G2.EN).warnings


@pytest.mark.parametrize("hyp", ["", "   ", "؟؟؟", "abc", "١٢٣", "\n\n", "لا", "." * 500, "و" * 2000])
def test_no_crash_on_noisy_input(hyp):
    G2.check("Take 1 tablet of warfarin 3 mg by mouth once daily.", hyp)
    G2.check(hyp, "خذ قرصا واحدا.")


def test_random_noise_never_crashes():
    rng = random.Random(0)
    alphabet = "ابتثجحخدذرزسشصضطظعغفقكلمنهوي ٠١٢٣٤٥٦٧٨٩ .،؟ mg12"
    for _ in range(300):
        G2.check("Do not take 2 tablets of warfarin 3 mg.", "".join(rng.choice(alphabet) for _ in range(rng.randint(0, 80))))


# ---------------------------------------------------------------- what the rules cannot verify

def test_unfamiliar_drug_replaced_by_another_is_flagged():
    src = "Take 1 tablet of apixaban 5 mg by mouth twice daily."
    assert "drug" in kinds(src, "خذ قرصا واحدا من ريفاروكسابان 5 ملغ عن طريق الفم مرتين يوميا.")[0]


def test_unfamiliar_drug_transliterated_or_left_in_latin_passes():
    src = "Take 1 tablet of apixaban 5 mg by mouth twice daily."
    assert G2.check(src, "خذ قرصا واحدا من أبيكسابان 5 ملغ عن طريق الفم مرتين يوميا.").ok
    assert G2.check(src, "خذ قرصا واحدا من apixaban 5 ملغ عن طريق الفم مرتين يوميا.").ok


def test_unverifiable_drug_goes_to_review_not_pass():
    report = G2.check("Take 1 tablet of apixaban 5 mg by mouth twice daily.",
                      "خذ قرصا واحدا من الدواء 5 ملغ عن طريق الفم مرتين يوميا.")
    assert report.status == "needs_review"


def test_unrecognized_warning_is_never_a_silent_pass():
    report = G2.check("Take 1 tablet of warfarin 3 mg once daily. Tell your doctor if you notice unusual bleeding.",
                      "خذ قرصا واحدا من وارفارين 3 ملغ مرة واحدة يوميا.")
    assert report.status != "ok" and report.unverified and report.coverage < 1


def test_swapped_taper_is_flagged():
    src = "Take 40 mg daily for 5 days, then 20 mg daily for 5 days."
    assert "sequence" in kinds(src, "خذ 20 ملغ يوميا لمدة 5 أيام، ثم 40 ملغ يوميا لمدة 5 أيام.")[0]
    assert "sequence" not in kinds(src, "خذ 40 ملغ يوميا لمدة 5 أيام، ثم 20 ملغ يوميا لمدة 5 أيام.")[0]


def test_regex_sources_have_no_doubled_backslashes():
    # a patch once wrote r"\\\\bif\\\\b", which matches nothing and fails silently
    assert chr(92) * 2 not in Path(G2.__file__).read_text(encoding="utf-8")


def test_report_serializes_for_batch_use():
    d = G2.check("Take 1 tablet of warfarin 3 mg once daily.", "خذ قرصا واحدا من وارفارين 3 ملغ مرة واحدة يوميا.").to_dict()
    assert d["status"] == "ok" and d["coverage"] == 1 and d["flags"] == []
