"""Structured medication instructions rendered to English and Arabic.

A critical error is a change to ONE clinical slot on the Arabic side only (the source stays correct).
A benign variant is a change of Arabic *style* only (synonyms, digits, word order, spelling).
Both are produced from the same structure, so every label is exact by construction.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, replace

from . import lexicon as L
from .arabic import Noun, count_phrase, fmt_num, strip_hamza, to_indic

ERROR_TYPES = ("dose", "strength", "unit", "frequency", "duration", "timing", "route",
               "laterality", "negation", "drug", "omission")

ML = Noun("1 مل", "2 مل", "مل", "مل")
TBSP = Noun("ملعقة كبيرة واحدة", "ملعقتين كبيرتين", "ملاعق كبيرة", "ملعقة كبيرة")


@dataclass(frozen=True)
class Rx:
    drug: L.Drug
    count: float
    site: str
    freq: str
    strength: float | None = None
    unit: str | None = None  # strength unit
    dose_unit: str = "default"  # 'default' | 'ml' | 'tbsp'
    timing: str | None = None
    duration: tuple[str, int] | None = None
    prn: bool = False
    warning: str | None = None
    # error-only fields
    negate_main: bool = False
    flip_warning: bool = False
    drop_warning: bool = False

    @property
    def form(self) -> L.Form:
        return L.FORMS[self.drug.form]

    def slots(self) -> dict:
        """Canonical clinical content (what must survive translation)."""
        return {
            "drug": self.drug.en, "count": self.count, "dose_unit": self.dose_unit,
            "strength": self.strength, "unit": self.unit, "site": self.site, "freq": self.freq,
            "timing": self.timing, "duration": self.duration, "prn": self.prn,
            "warning": None if self.drop_warning else self.warning,
            "warning_polarity": not self.flip_warning, "main_polarity": not self.negate_main,
        }


@dataclass
class Style:
    """Benign Arabic style knobs. Style() is the reference style."""

    indic_digits: bool = False
    unit_variant: int = 0
    daily_alt: bool = False
    verb_alt: bool = False
    timing_alt: bool = False
    site_alt: bool = False
    duration_first: bool = False
    hamza_free: bool = False
    definite_drug: bool = False

    @staticmethod
    def random_benign(rng: random.Random, rx: Rx, k: int | None = None) -> "Style":
        knobs = ["indic_digits", "daily_alt", "verb_alt", "hamza_free", "definite_drug"]
        if rx.strength is not None:
            knobs.append("unit_variant")
        if rx.timing:
            knobs.append("timing_alt")
        if L.SITE[rx.site][1] != L.SITE[rx.site][2] and rx.drug.form not in ("inhaler",):
            knobs.append("site_alt")
        if rx.duration:
            knobs.append("duration_first")
        k = k or rng.randint(1, 3)
        style = Style()
        for knob in rng.sample(knobs, min(k, len(knobs))):
            if knob == "unit_variant":
                style.unit_variant = rng.randint(1, len(L.UNITS_AR[rx.unit]) - 1)
            else:
                setattr(style, knob, True)
        return style


# ----------------------------------------------------------------------------------- rendering

def _dose_en(rx: Rx) -> str:
    if rx.drug.form == "cream":
        return "a thin layer"
    if rx.dose_unit == "tbsp":
        return f"{fmt_num(rx.count)} tablespoon{'s' if rx.count != 1 else ''}"
    if rx.drug.form == "syrup" or rx.dose_unit == "ml":
        return f"{fmt_num(rx.count)} ml"
    sg, pl = rx.form.noun_en
    return f"{fmt_num(rx.count)} {sg if rx.count == 1 else pl}"


def _dose_ar(rx: Rx) -> str:
    if rx.drug.form == "cream":
        return "طبقة رقيقة"
    if rx.dose_unit == "tbsp":
        return count_phrase(rx.count, TBSP)
    if rx.dose_unit == "ml":
        return f"{fmt_num(rx.count)} مل"
    if rx.drug.form == "syrup":
        return f"{fmt_num(rx.count)} مل"
    return count_phrase(rx.count, rx.form.noun)


def render_en(rx: Rx) -> str:
    drug = rx.drug.en + (" cream" if rx.drug.form == "cream" else "")
    parts = [rx.form.verb_en, _dose_en(rx), "of", drug]
    if rx.strength is not None:
        parts.append(f"{fmt_num(rx.strength)} {rx.unit}")
    if rx.drug.form != "inhaler":
        parts.append(L.SITE[rx.site][0])
    parts.append(L.FREQ[rx.freq][0])
    if rx.prn:
        parts.append("as needed")
    if rx.timing:
        parts.append(L.TIMING[rx.timing][0])
    if rx.duration:
        unit, n = rx.duration
        sg, pl, _ = L.DURATION_UNITS[unit]
        parts.append(f"for {n} {sg if n == 1 else pl}")
    text = " ".join(parts) + "."
    if rx.warning:
        text += " " + _warning(rx.warning).en
    return text


def _warning(key: str) -> L.Warning_:
    return next(w for w in L.WARNINGS if w.key == key)


def render_ar(rx: Rx, style: Style | None = None) -> str:
    s = style or Style()
    form = rx.form
    verb = form.verbs_ar[1 if s.verb_alt else 0]
    if rx.negate_main:
        verb = form.neg_ar[verb]
    drug_ar = rx.drug.ar
    if s.definite_drug and " " not in drug_ar:
        drug_ar = "ال" + drug_ar
    if rx.drug.form == "cream":
        drug_ar = "كريم " + drug_ar
    parts = [verb, _dose_ar(rx), "من", drug_ar]
    if rx.strength is not None:
        unit_ar = L.UNITS_AR[rx.unit][s.unit_variant % len(L.UNITS_AR[rx.unit])]
        parts.append(f"{fmt_num(rx.strength)} {unit_ar}")
    if rx.drug.form != "inhaler":
        parts.append(L.SITE[rx.site][2 if s.site_alt else 1])
    parts.append(L.FREQ[rx.freq][2 if s.daily_alt else 1])
    if rx.prn:
        parts.append("عند الحاجة")
    if rx.timing:
        table = L.TIMING if rx.timing in L.TIMING else L.TIMING_AR_EXTRA
        parts.append(table[rx.timing][2 if s.timing_alt else 1])
    main = " ".join(parts)
    if rx.duration:
        unit, n = rx.duration
        dur = "لمدة " + count_phrase(n, L.DURATION_UNITS[unit][2], genitive=True)
        main = f"{dur}، {main}" if s.duration_first else f"{main} {dur}"
    text = main + "."
    if rx.warning and not rx.drop_warning:
        w = _warning(rx.warning)
        text += " " + (w.ar_flipped if rx.flip_warning else w.ar)
    if s.indic_digits:
        text = to_indic(text)
    if s.hamza_free:
        text = strip_hamza(text)
    return text


# ----------------------------------------------------------------------------------- sampling

FORM_FREQS = {
    "tablet": ("once_daily", "twice_daily", "three_daily", "q8h", "q12h", "bedtime"),
    "capsule": ("once_daily", "twice_daily", "three_daily", "q8h", "q12h"),
    "syrup": ("q6h", "q8h", "three_daily", "twice_daily"),
    "eye_drop": ("once_daily", "twice_daily", "three_daily", "four_daily", "bedtime"),
    "ear_drop": ("twice_daily", "three_daily", "four_daily"),
    "inhaler": ("q4h", "q6h", "twice_daily"),
    "insulin": ("once_daily", "bedtime", "three_daily"),
    "cream": ("once_daily", "twice_daily", "three_daily"),
}
SITES = {"eye_drop": ("eye_left", "eye_right", "eye_both"), "ear_drop": ("ear_left", "ear_right", "ear_both")}
PRN_DRUGS = {"paracetamol", "ibuprofen", "tramadol", "paracetamol syrup", "ibuprofen suspension",
             "salbutamol", "hydroxyzine", "cetirizine"}
# chronic once-daily drugs: one unit, once a day (keeps regimens clinically plausible)
ONCE_DAILY = {"atorvastatin", "amlodipine", "losartan", "warfarin", "aspirin", "clopidogrel",
              "levothyroxine", "cetirizine", "bisoprolol", "lisinopril", "digoxin", "citalopram",
              "fluoxetine", "glimepiride", "latanoprost", "insulin glargine", "trazodone", "esomeprazole"}
TWICE_MAX = {"omeprazole", "metformin", "furosemide", "clonidine", "clonazepam", "glipizide",
             "gabapentin", "doxycycline", "celecoxib", "prednisolone", "prednisone", "budesonide"}


def sample_rx(rng: random.Random, drug: L.Drug) -> Rx:
    form = L.FORMS[drug.form]
    site = rng.choice(SITES.get(drug.form, (form.site,)))
    freq = rng.choice(FORM_FREQS[drug.form])
    if drug.en == "methotrexate":
        freq = "weekly"
    elif drug.en in ONCE_DAILY:
        freq = rng.choice(("once_daily", "bedtime") if drug.en in ("atorvastatin", "trazodone", "latanoprost",
                                                                    "insulin glargine") else ("once_daily",))
    elif drug.en in TWICE_MAX:
        freq = rng.choice(("once_daily", "twice_daily"))
    timing = None
    if drug.form in ("tablet", "capsule", "syrup") and rng.random() < 0.55:
        timing = rng.choice(tuple(L.TIMING))
    if drug.en == "insulin aspart":
        timing, freq = "before_meals", "three_daily"
    if freq == "bedtime" and timing == "morning":
        timing = None
    duration = rng.choice(L.DURATIONS) if rng.random() < 0.7 else None
    warnings = [w.key for w in L.WARNINGS if drug.form in w.forms]
    warning = rng.choice(warnings) if warnings and rng.random() < 0.5 else None
    prn = drug.en in PRN_DRUGS and freq in ("q4h", "q6h", "q8h") and rng.random() < 0.6
    count = rng.choice(form.counts)
    if drug.en in ONCE_DAILY and drug.form in ("tablet", "capsule"):
        count = 1
    if drug.en == "levothyroxine":
        timing = "empty_stomach"
    return Rx(drug=drug, count=count, site=site, freq=freq,
              strength=rng.choice(drug.strengths) if drug.strengths else None,
              unit=drug.unit if drug.strengths else None, timing=timing, duration=duration,
              prn=prn, warning=warning)


# ----------------------------------------------------------------------------------- errors

def applicable_errors(rx: Rx) -> list[str]:
    errs = ["frequency", "drug"]
    if rx.drug.form != "cream":
        errs.append("dose")
    if rx.strength is not None:
        errs += ["strength", "unit"]
    elif rx.drug.form in ("syrup", "insulin"):
        errs.append("unit")
    if rx.duration:
        errs.append("duration")
    if rx.timing:
        errs.append("timing")
    if rx.site in L.ROUTE_ERRORS:
        errs.append("route")
    if rx.site in L.LATERALITY_ERRORS:
        errs.append("laterality")
    errs.append("negation")
    if rx.warning and _warning(rx.warning).protective:
        errs.append("omission")
    return errs


def _new_count(rng: random.Random, rx: Rx) -> float:
    if rx.drug.form in ("tablet", "capsule", "eye_drop", "inhaler"):
        return rng.choice([c for c in (1, 2, 3, 4) if c != rx.count])
    return rx.count * 10 if rng.random() < 0.6 else max(rx.count / 2, 1)  # decimal-point slips


def inject_error(rng: random.Random, rx: Rx, kind: str) -> tuple[Rx, str]:
    """Return (mutated rx, short human-readable description of the error)."""
    if kind == "dose":
        n = _new_count(rng, rx)
        return replace(rx, count=n), f"dose {fmt_num(rx.count)} -> {fmt_num(n)}"
    if kind == "strength":
        s = rx.strength * 10 if rng.random() < 0.5 else rx.strength / 10
        return replace(rx, strength=s), f"strength {fmt_num(rx.strength)} -> {fmt_num(s)} {rx.unit}"
    if kind == "unit":
        if rx.strength is not None:
            new = {"mg": "g", "mcg": "mg"}[rx.unit]
            return replace(rx, unit=new), f"unit {rx.unit} -> {new}"
        if rx.drug.form == "syrup":
            return replace(rx, dose_unit="tbsp"), "unit ml -> tablespoon"
        return replace(rx, dose_unit="ml"), "unit insulin units -> ml"
    if kind == "frequency":
        new = rng.choice(L.FREQ_ERRORS[rx.freq])
        return replace(rx, freq=new), f"frequency {rx.freq} -> {new}"
    if kind == "duration":
        unit, n = rx.duration
        if rng.random() < 0.5:
            new_unit = {"day": "week", "week": "month", "month": "day"}[unit]
            return replace(rx, duration=(new_unit, n)), f"duration {unit} -> {new_unit}"
        new_n = n * 10 if n < 10 else n // 2 if n > 2 else n + 5
        return replace(rx, duration=(unit, new_n)), f"duration {n} -> {new_n} {unit}"
    if kind == "timing":
        new = L.TIMING_ERRORS[rx.timing]
        return replace(rx, timing=new), f"timing {rx.timing} -> {new}"
    if kind == "route":
        new = rng.choice(L.ROUTE_ERRORS[rx.site])
        return replace(rx, site=new), f"route {rx.site} -> {new}"
    if kind == "laterality":
        new = L.LATERALITY_ERRORS[rx.site]
        return replace(rx, site=new), f"laterality {rx.site} -> {new}"
    if kind == "negation":
        if rx.warning and rng.random() < 0.6:
            return replace(rx, flip_warning=True), f"warning '{rx.warning}' polarity flipped"
        return replace(rx, negate_main=True), "main instruction negated"
    if kind == "drug":
        partner = L.LASA.get(rx.drug.en)
        if partner and L.DRUG_BY_EN[partner].form == rx.drug.form:
            new = L.DRUG_BY_EN[partner]
            tag = "LASA"
        else:
            new = rng.choice([d for d in L.DRUGS if d.form == rx.drug.form and d.en != rx.drug.en])
            tag = "same-form"
        return replace(rx, drug=new), f"drug {rx.drug.en} -> {new.en} ({tag})"
    if kind == "omission":
        return replace(rx, drop_warning=True), f"warning '{rx.warning}' omitted"
    raise ValueError(kind)
