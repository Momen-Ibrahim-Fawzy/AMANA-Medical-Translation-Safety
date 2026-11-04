"""An outside reviewer's probes (data/review_probes.jsonl), kept as regression tests after the fixes they led to.

They were scored once before any fix (results/review_probes_first_run.json). Two safe sentences have nothing the
rules can compare, so they are reported as not checked: strict xfails that fail if they start passing silently.
"""
import json
from pathlib import Path

import pytest

from amana.guard2 import check

ROOT = Path(__file__).resolve().parents[1]
ROWS = [json.loads(x) for x in (ROOT / "data" / "review_probes.jsonl").read_text(encoding="utf-8").splitlines()]
NOT_CHECKED = {"safe-shake", "safe-tell-doctor"}


def _params():
    for r in ROWS:
        marks = [pytest.mark.xfail(strict=True, reason="no rule covers it; reported as not checked")] \
            if r["id"] in NOT_CHECKED else []
        yield pytest.param(r, id=r["id"], marks=marks)


@pytest.mark.parametrize("row", list(_params()))
def test_probe(row):
    assert check(row["src_en"], row["hyp_ar"]).ok == row["should_pass"]


def test_not_checked_sentences_say_so():
    for r in ROWS:
        if r["id"] in NOT_CHECKED:
            report = check(r["src_en"], r["hyp_ar"])
            assert report.coverage == 0 and [f.kind for f in report.review] == ["unverified"]


@pytest.mark.parametrize("src,hyp", [
    # false alarms found on development data while fixing the probes
    ("A reduced loading dose of 0.5 mcg/kg given over 10 minutes is recommended.",
     "يوصى بجرعة تحميل مخفضة قدرها 0.5 ميكروجرام/كلجم تعطى على مدار 10 دقائق."),
    ("Lansoprazole……………….15 mg for one gastro-resistant hard capsule.", "لانسوبرازول................15 ملغ لكل كبسولة."),
    ("Apply hydrocortisone 1% cream twice daily.", "ضع كريم هيدروكورتيزون 1% (هيدروكورتيزون 1%) مرتين يومياً."),
    ("Avoid getting Clobetasol Propionate Shampoo, 0.05% in your eyes.", "تجنب وصول شامبو Clobetasol Propionate، 0. 05% إلى عينيك."),
    ("Take the second pantoprazole tablet 1 hour before your evening meal.", "تناول قرص البانتوبرازول الثاني قبل ساعة من العشاء."),
    ("If you have dialysis two or three times a week, take it after dialysis.",
     "إذا كنت تقوم بغسيل الكلى مرتين أو ثلاث مرات في الأسبوع، تناوله بعد الغسيل."),
])
def test_no_false_flag(src, hyp):
    assert not check(src, hyp).flags


def test_breastfeeding_rendered_as_touching_is_caught():
    assert check("Do not breastfeed while receiving mitoxantrone.", "لا تلمس أثناء تناول الميتوكسانترون.").flags


def test_general_text_reports_its_path():
    assert check("Shake the bottle well before use.", "رج الزجاجة جيداً قبل الاستخدام.").general
    assert not check("Take 1 tablet of warfarin 5 mg once daily.", "تناول قرصاً واحداً من وارفارين 5 ملغ مرة واحدة يومياً.").general
