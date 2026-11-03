"""Arabic text utilities: digits, normalisation, number agreement and number-word parsing."""
from __future__ import annotations

import re
from dataclasses import dataclass

WEST_TO_INDIC = str.maketrans("0123456789", "٠١٢٣٤٥٦٧٨٩")
INDIC_TO_WEST = str.maketrans("٠١٢٣٤٥٦٧٨٩۰۱۲۳۴۵۶۷۸۹", "01234567890123456789")

DIACRITICS = re.compile(r"[ً-ْٰـ]")  # harakat, dagger alif, tatweel
# letters from other Arabic-script alphabets, common in real leaflets (ھذا, تاریخ, ڤالسارتان)
VARIANT_LETTERS = str.maketrans({"ھ": "ه", "ہ": "ه", "ە": "ه", "ی": "ي", "ک": "ك", "ڤ": "ف", "پ": "ب", "گ": "ج"})


def to_indic(text: str) -> str:
    """Western digits -> Arabic-Indic digits, '.' between digits -> Arabic decimal separator."""
    text = re.sub(r"(?<=\d)\.(?=\d)", "٫", text)
    return text.translate(WEST_TO_INDIC)


def to_western(text: str) -> str:
    text = text.translate(INDIC_TO_WEST)
    text = re.sub(r"(?<=\d)[٬,](?=\d{3}(?!\d))", "", text)  # thousands separators
    return re.sub(r"(?<=\d)[٫,](?=\d)", ".", text)


def strip_hamza(text: str) -> str:
    """Informal spelling: hamzated alifs written as bare alif (very common in real text)."""
    return re.sub("[أإآ]", "ا", text)


def normalize(text: str) -> str:
    """Canonical form used for matching only (never for rendering)."""
    text = to_western(text)
    text = DIACRITICS.sub("", text).translate(VARIANT_LETTERS)
    text = strip_hamza(text)
    text = text.replace("ى", "ي").replace("ة", "ه")
    return re.sub(r"\s+", " ", text).strip()


def fmt_num(x: float) -> str:
    return str(int(x)) if float(x).is_integer() else f"{x:g}"


@dataclass(frozen=True)
class Noun:
    """Forms an Arabic counted noun needs (accusative context, e.g. object of an imperative).

    one:    'قرصاً واحداً'   (1)
    dual:   'قرصين'          (2)
    plural: 'أقراص'          (3-10, written after the digit)
    acc:    'قرصاً'          (11+, singular accusative after the digit)
    """

    one: str
    dual: str
    plural: str
    acc: str
    # genitive variants (after a preposition such as لمدة) differ only for 1 and 11+
    one_gen: str | None = None
    acc_gen: str | None = None


def count_phrase(n: float, noun: Noun, genitive: bool = False) -> str:
    """Correct Arabic number-noun agreement with digits for n >= 3 (pharmacy-label style)."""
    if not float(n).is_integer():
        return f"{fmt_num(n)} {noun.plural}" if n > 1 else f"{fmt_num(n)} {noun.acc_gen or noun.acc}"
    n = int(n)
    if n == 1:
        return (noun.one_gen if genitive and noun.one_gen else noun.one)
    if n == 2:
        return noun.dual
    if 3 <= n <= 10:
        return f"{n} {noun.plural}"
    # 11..99 take the singular accusative; 100, 200, ... take the singular genitive
    if n % 100 == 0:
        return f"{n} {noun.acc_gen or noun.acc}"
    return f"{n} {noun.acc}"


# --- number words, for reading free MT output -------------------------------------------------

_UNITS = {
    "واحد": 1, "واحده": 1, "واحدا": 1, "واحدة": 1, "اثنان": 2, "اثنين": 2, "اثنتان": 2, "اثنتين": 2,
    "ثلاث": 3, "ثلاثه": 3, "اربع": 4, "اربعه": 4, "خمس": 5, "خمسه": 5, "ست": 6, "سته": 6,
    "سبع": 7, "سبعه": 7, "ثمان": 8, "ثماني": 8, "ثمانيه": 8, "تسع": 9, "تسعه": 9,
    "عشر": 10, "عشره": 10, "عشرين": 20, "عشرون": 20, "ثلاثين": 30, "ثلاثون": 30,
    "اربعين": 40, "اربعون": 40, "خمسين": 50, "خمسون": 50, "ستين": 60, "مائه": 100, "مئه": 100,
    "نصف": 0.5, "ربع": 0.25,
}
_ORDER_WORDS = {"مره": 1, "مرتين": 2, "مرتان": 2}

# known dual nouns -> the number 2 is implicit in the noun
DUAL_STEMS = (
    "قرصين", "قرصان", "حبتين", "حبتان", "كبسولتين", "قطرتين", "نقطتين", "بختين", "ملعقتين",
    "وحدتين", "يومين", "يومان", "اسبوعين", "اسبوعان", "شهرين", "شهران", "ساعتين", "ساعتان", "مرتين", "مرتان", "جرعتين",
    "عامين", "عامان", "سنتين", "سنتان",
)


def number_words_to_digits(text: str) -> str:
    """Rewrite simple Arabic number words as digits (input must already be normalize()d).

    Only handles what medication instructions use: 1-10, tens, compound 'خمسه عشر', 'نصف'.
    """
    tokens = text.split(" ")
    out: list[str] = []
    i = 0
    while i < len(tokens):
        tok = tokens[i]
        bare = tok[2:] if tok.startswith("ال") and tok[2:] in _UNITS else tok
        bare = bare[1:] if bare.startswith("و") and bare[1:] in _UNITS and bare not in _UNITS else bare
        if bare in _UNITS:
            val = _UNITS[bare]
            # compound teens: خمسه عشر -> 15
            if i + 1 < len(tokens) and tokens[i + 1] in ("عشر", "عشره") and val < 10:
                out.append(fmt_num(val + 10))
                i += 2
                continue
            out.append(fmt_num(val))
        else:
            out.append(tok)
        i += 1
    return " ".join(out)
