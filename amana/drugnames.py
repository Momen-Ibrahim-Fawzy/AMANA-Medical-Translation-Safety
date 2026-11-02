"""Open drug vocabulary: English and Arabic names from Wikidata, and a transliteration comparison.

The comparison reduces an English name and an Arabic word to a shared consonant spelling, so a drug can
be looked for in a translation even when its Arabic name is not in any table.
"""
from __future__ import annotations

import csv
import re
from difflib import SequenceMatcher
from functools import lru_cache
from pathlib import Path

from .arabic import normalize

TABLE = Path(__file__).parent / "data" / "drug_names.tsv"
# chosen on the development names only (see scripts/15_drug_names_eval.py). A wrong match is a silent
# pass and a missed match only asks for review, so the strict end of the sweep is used.
MATCH_THRESHOLD = 0.9
# Wikidata lists these as drugs, but in instructions they are ordinary words
_COMMON = {"water", "oxygen", "iron", "zinc", "sugar", "salt", "alcohol", "ethanol", "caffeine", "calcium", "sodium",
           "potassium", "magnesium", "nicotine", "air", "gold", "silver", "copper", "sulfur", "iodine", "honey",
           "charcoal", "lactose", "glucose", "sucrose", "urea", "nitrogen", "carbon", "hydrogen", "helium", "chlorine"}

_AR_CLASS = str.maketrans({"ب": "B", "پ": "B", "ت": "T", "ط": "T", "ث": "T", "ة": "T", "د": "D", "ذ": "D", "ض": "D",
                           "ج": "G", "غ": "G", "ق": "G", "ك": "K", "خ": "K", "ف": "F", "ڤ": "F", "س": "S", "ص": "S",
                           "ز": "S", "ش": "S", "ظ": "S", "ل": "L", "م": "M", "ن": "N", "ر": "R"})


def skeleton_en(name: str) -> str:
    t = name.lower()
    for a, b in (("ph", "f"), ("th", "t"), ("sch", "s"), ("sh", "s"), ("ch", "k"), ("ck", "k"), ("gh", "g"), ("qu", "k")):
        t = t.replace(a, b)
    t = re.sub(r"c(?=[eiy])", "s", t)
    t = t.translate(str.maketrans({"c": "k", "q": "k", "x": "1", "z": "s", "v": "f", "p": "b", "j": "g"}))
    t = t.replace("1", "ks")
    t = re.sub(r"[^bdfgklmnrst]", "", t)
    return re.sub(r"(.)\1+", r"\1", t).upper()


def skeleton_ar(word: str) -> str:
    t = normalize(word).translate(_AR_CLASS)
    t = re.sub(r"[^BTDGKFSLMNR]", "", t)
    return re.sub(r"(.)\1+", r"\1", t)


def similarity(en: str, ar: str) -> float:
    a, b = skeleton_en(en), skeleton_ar(ar)
    return SequenceMatcher(None, a, b).ratio() if a and b else 0.0


def _clitic_forms(word: str) -> set:
    out = {word}
    for prefix in ("وال", "بال", "فال", "لل", "ال", "و", "ب", "ل", "ف"):
        if word.startswith(prefix) and len(word) - len(prefix) >= 4:
            out.add(word[len(prefix):])
    return out


@lru_cache(maxsize=1)
def _load() -> tuple[dict, dict]:
    """English name -> normalized Arabic label, and the reverse. Common words are left out."""
    en_to_ar, ar_to_en = {}, {}
    if not TABLE.exists():
        return en_to_ar, ar_to_en
    with TABLE.open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            if row["en"] in _COMMON or len(row["en"]) < 5:
                continue
            ar = normalize(row["ar"])
            en_to_ar[row["en"]] = ar
            ar_to_en.setdefault(ar, row["en"])
    return en_to_ar, ar_to_en


def known_english() -> dict:
    return _load()[0]


def find_in_arabic(en: str, text: str) -> tuple[float, str | None]:
    """Best transliteration score of an English name against the words of an Arabic text."""
    best, word = 0.0, None
    label = known_english().get(en)
    t = normalize(text)
    if label and label in t:
        return 1.0, label
    for w in re.findall(r"[ء-ي]{4,}", t):
        for form in _clitic_forms(w):
            score = similarity(en, form)
            if score > best:
                best, word = score, form
    return best, word


def other_drugs_in_arabic(en: str, text: str) -> set:
    """Other drugs named in an Arabic text, by exact label."""
    ar_to_en = _load()[1]
    found = set()
    for w in re.findall(r"[ء-ي]{4,}", normalize(text)):
        for form in _clitic_forms(w):
            other = ar_to_en.get(form)
            if other and other != en:
                found.add(other)
    return found
