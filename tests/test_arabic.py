"""Arabic agreement, digits and number words, and the instruction generator."""
import pytest

from amana import lexicon as L
from amana.arabic import count_phrase, normalize, number_words_to_digits, to_indic, to_western
from amana.rx import Rx, render_ar, render_en

TAB = L.FORMS["tablet"].noun
DAY = L.DURATION_UNITS["day"][2]


@pytest.mark.parametrize("n,expected", [(1, "قرصاً واحداً"), (2, "قرصين"), (3, "3 أقراص"),
                                        (10, "10 أقراص"), (11, "11 قرصاً"), (100, "100 قرص")])
def test_number_noun_agreement(n, expected):
    assert count_phrase(n, TAB) == expected


def test_duration_genitive():
    assert count_phrase(1, DAY, genitive=True) == "يوم واحد"
    assert count_phrase(14, DAY, genitive=True) == "14 يوماً"


def test_digits_roundtrip():
    assert to_indic("0.25 ملغ") == "٠٫٢٥ ملغ"
    assert to_western("٠٫٢٥") == "0.25"
    assert to_western("39,768 حالة") == "39768 حالة"


def test_number_words():
    assert number_words_to_digits(normalize("ثلاث مرات لمدة خمسة عشر يوما")) == "3 مرات لمده 15 يوما"


def _rx(**kw):
    base = dict(drug=L.DRUG_BY_EN["metformin"], count=2, site="oral", freq="twice_daily",
                strength=500, unit="mg", timing="after_meals", duration=("day", 30), warning="alcohol")
    base.update(kw)
    return Rx(**base)


def test_render_reference():
    rx = _rx()
    assert render_en(rx) == ("Take 2 tablets of metformin 500 mg by mouth twice daily after meals for 30 days. "
                             "Do not drink alcohol while taking this medicine.")
    assert render_ar(rx) == ("تناول قرصين من ميتفورمين 500 ملغ عن طريق الفم مرتين يومياً بعد الأكل لمدة 30 يوماً. "
                             "لا تشرب الكحول أثناء تناول هذا الدواء.")
