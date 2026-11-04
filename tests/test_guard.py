"""The baseline checker, AMANA-Guard (amana/guard.py). Guard 2 has its own tests in test_guard2.py."""
import random

import pytest

from amana import lexicon as L
from amana.guard import check
from amana.rx import ERROR_TYPES, Style, applicable_errors, inject_error, render_ar, render_en, sample_rx


def test_guard_accepts_reference_and_paraphrases():
    rng = random.Random(0)
    for _ in range(300):
        rx = sample_rx(rng, rng.choice(L.DRUGS))
        src = render_en(rx)
        assert check(src, render_ar(rx)).ok
        assert check(src, render_ar(rx, Style.random_benign(rng, rx))).ok


@pytest.mark.parametrize("kind", ERROR_TYPES)
def test_guard_flags_each_error_type(kind):
    rng = random.Random(1)
    hits = total = 0
    for _ in range(400):
        rx = sample_rx(rng, rng.choice(L.DRUGS))
        if kind not in applicable_errors(rx):
            continue
        bad, _ = inject_error(rng, rx, kind)
        total += 1
        hits += not check(render_en(rx), render_ar(bad)).ok
    assert total > 0 and hits == total


def test_guard_reads_free_form_arabic():
    src = "Take 2 tablets of paracetamol 500 mg by mouth every 6 hours for 3 days."
    # number words, Egyptian unit spelling, 'حبتين', hamza-free spelling, definite drug name
    assert check(src, "خذ حبتين من الباراسيتامول ٥٠٠ مجم بالفم كل ست ساعات لمدة ثلاثة ايام").ok
    report = check(src, "خذ حبتين من الباراسيتامول 50 مجم بالفم كل ست ساعات لمدة ثلاثة ايام")
    assert [f.kind for f in report.flags] == ["strength"]


def test_guard_laterality_and_lasa():
    src = "Instill 1 drop of timolol into the left eye twice daily."
    assert [f.kind for f in check(src, "ضع قطرة واحدة من تيمولول في العين اليمنى مرتين يومياً.").flags] == ["laterality"]
    src = "Take 1 tablet of hydralazine 25 mg by mouth twice daily."
    assert [f.kind for f in check(src, "تناول قرصاً واحداً من هيدروكسيزين 25 ملغ عن طريق الفم مرتين يومياً.").flags] == ["drug"]


def test_guard_flags_extra_quantity_with_another_unit():
    src = "Take 1 tablet of warfarin 25 mcg by mouth once daily."
    report = check(src, "خذ قرصا واحدا من وارفارين 25 ميكروغرام و25 ملغ عن طريق الفم مرة واحدة يوميا.")
    assert [f.kind for f in report.flags] == ["dose"]


def test_guard_on_real_mt_outputs():
    # cases from the manual review of NLLB-200 and OPUS-MT outputs
    src = "Take 1 tablet of atorvastatin 40 mg by mouth once daily. Do not crush or chew the tablets."
    assert check(src, "خذ قرص واحد من atorvastatin 40 mg عن طريق الفم مرة واحدة يوميًا. لا تسحق أو تمضغ الأقراص.").ok
    assert [f.kind for f in check(src, "خذ قرص واحد من atorvastatin 40 mg عن طريق الفم مرة واحدة يوميًا.").flags] == ["omission"]
    src = "Take 2 tablets of glipizide 5 mg by mouth twice daily on an empty stomach."
    assert check(src, "خذ قرصين من الغليبزيد 5 ملغم من الفم مرتين يومياً على بطن فارغ.").ok
    src = "Take 1 tablet of levothyroxine 25 mcg by mouth once daily."
    report = check(src, "خذ قرصاً واحداً من ليفوثيروكسين 25 مليغراماً عن طريق الفم مرة واحدة يومياً.")
    assert [f.kind for f in report.flags] == ["unit"]
    src = "Take 1 tablet of prednisolone 5 mg by mouth once daily for 2 weeks."
    assert [f.kind for f in check(src, "تناول قرصاً واحداً من بريدنيزون 5 ملغ عن طريق الفم مرة واحدة يومياً لمدة أسبوعين.").flags] == ["drug"]
    # transliterated or split drug names are accepted; a different drug is not
    src = "Instill 1 drop of chloramphenicol into both eyes twice daily."
    assert check(src, "ضع قطرة واحدة من الكلور فينيكول في كلا العينين مرتين يوميًا.").ok
    src = "Take 1 capsule of esomeprazole 40 mg by mouth once daily."
    assert check(src, "خذ 1 كبسولة من إسوميبرازول 40 ملغ عن طريق الفم مرة واحدة يوميا.").ok
    assert [f.kind for f in check(src, "خذ كبسولة واحدة من أوميبرازول 40 ملغ عن طريق الفم مرة واحدة يوميا.").flags] == ["drug"]
    # judged acceptable by a native-speaker review: 'لوح' for tablet, 'السمبرازول' for esomeprazole
    src = "Take 1 tablet of glimepiride 1 mg by mouth once daily with food."
    assert check(src, "خذ لوحاً واحداً من الجليمبيريد 1 ملغم عن طريق الفم مرة واحدة يومياً مع الطعام.").ok
    src = "Take 1 capsule of esomeprazole 40 mg by mouth once daily for 14 days."
    assert check(src, "خذ كبسولة واحدة من السمبرازول 40 ملغم عن طريق الفم مرة واحدة يومياً لمدة 14 يوماً.").ok
