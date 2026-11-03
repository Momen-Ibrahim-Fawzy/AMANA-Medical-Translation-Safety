"""AMANA-Guard: explainable clinical-slot checker for English -> Arabic medical translation.

Extracts the clinically critical content of the English source and of the Arabic translation
into the same canonical slots (quantities, frequency, duration, timing, site, prohibitions, drugs)
and reports every disagreement with its type. It is a reviewer's tool: every flag says what changed.
"""
from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field
from difflib import SequenceMatcher

from . import lexicon as L
from .arabic import DUAL_STEMS, fmt_num, normalize, number_words_to_digits

NUM = r"(\d+(?:\.\d+)?)"

# ------------------------------------------------------------------------------ English side

EN_UNITS = {
    "mg": r"mg|milligrams?", "mcg": r"mcg|micrograms?|µg", "g": r"g|grams?", "ml": r"ml|mls|millilit(?:er|re)s?",
    "unit": r"units?|iu", "tablet": r"tablets?|tabs?|pills?", "capsule": r"capsules?|caps?",
    "drop": r"drops?", "puff": r"puffs?|inhalations?", "tbsp": r"tablespoons?|tbsp", "tsp": r"teaspoons?|tsp",
}
EN_QTY = re.compile(NUM + r"\s*(" + "|".join(f"(?:{v})" for v in EN_UNITS.values()) + r")\b", re.I)
EN_WORDNUM = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "half": 0.5, "a": 1}


def _en_unit(tok: str) -> str:
    for k, pat in EN_UNITS.items():
        if re.fullmatch(pat, tok, re.I):
            return k
    return tok


def _en_freq(t: str):
    if re.search(r"\bonce (?:a )?week(?:ly)?\b|\bweekly\b|\bevery week\b", t):
        return "weekly"
    m = re.search(r"every\s+" + NUM + r"\s*hours?", t)
    if m:
        return round(24 / float(m.group(1)), 2)
    for pat, v in ((r"\bonce\b", 1), (r"\btwice\b", 2), (r"\bthree times\b", 3), (r"\bfour times\b", 4),
                   (r"\b3 times\b", 3), (r"\b4 times\b", 4), (r"\b2 times\b", 2)):
        if re.search(pat, t):
            return v
    if re.search(r"at bedtime|every morning|every evening|\bdaily\b", t):
        return 1
    return None


DUR_DAYS = {"day": 1, "week": 7, "month": 30}


def _en_duration(t: str):
    m = re.search(r"for\s+" + NUM + r"\s*(day|week|month)s?", t)
    if m:
        return float(m.group(1)) * DUR_DAYS[m.group(2)]
    m = re.search(r"for\s+(one|two|three|a)\s+(day|week|month)s?", t)
    if m:
        return EN_WORDNUM[m.group(1)] * DUR_DAYS[m.group(2)]
    return None


EN_TIMING = {"before_meals": r"before (?:meals|food|eating)", "after_meals": r"after (?:meals|food|eating)",
             "with_food": r"with (?:food|meals)", "empty_stomach": r"empty stomach",
             "morning": r"in the morning|every morning", "evening": r"in the evening|every evening",
             "bedtime": r"bedtime"}
EN_SITE = {"oral": r"by mouth|orally", "sublingual": r"under the tongue", "rectal": r"rectal",
           "eye": r"\beyes?\b", "ear": r"\bears?\b", "subcut": r"under the skin|subcutaneous",
           "iv": r"\bvein\b|intravenous|\biv\b", "im": r"muscle|intramuscular|\bim\b", "inhaled": r"inhal",
           "topical": r"affected area|skin\b(?! *\.)"}
EN_SIDE = {"left": r"\bleft\b", "right": r"\bright\b", "both": r"\bboth\b"}
# imperative prohibitions only ('Do not drive'), not descriptive negation ('it does not contain')
EN_PROHIBIT = re.compile(r"(?:^|[.;:!?]\s*|\b(?:and|or|but)\s+)(?:do not|don't|never|avoid|you must not|you should not)\b", re.I)
EN_NEG = re.compile(r"\bnot\b|n't\b|\bno\b|\bnever\b|\bnone\b|\bnor\b|\bneither\b|\bwithout\b|\bcannot\b", re.I)

# ------------------------------------------------------------------------------ Arabic side
# all patterns run on normalize()d text: no hamza, ة->ه, ى->ي, western digits

AR_UNITS = {
    "mg": r"(?i:mg|milligrams?)|ملغ|مجم|ملغم|مليغراما|ملليغراما|مليغرامات|مليغرام|ملليغرام|ملي غرام|ملجم|مغ|ملليجرام|مليجرام|ميلليغرام|ميليغرام",
    "mcg": r"(?i:mcg|µg|micrograms?)|ميكروغراما|ميكروغرامات|ميكروغرام|ميكروجراما|ميكروغ|مكروغرام|ميكروجرام|مكغ|ميكرو غرام|مكجم",
    "g": r"(?i:grams?|g)|غراما|غرام|غرامات|جرام|جم|غم|غ",
    "ml": r"(?i:ml|millilitres?|milliliters?)|مل|ملل|ملليلتر|مليلتر|ميليلتر|مللتر|ملليلترات|مللي لتر",
    "unit": r"وحده|وحدات",
    "tablet": r"قرص|اقراص|قرصا|حبه|حبات|حبوب|لوحا|لوح|الواح",
    "capsule": r"كبسوله|كبسولات",
    "drop": r"قطره|قطرات|نقطه|نقاط",
    "puff": r"بخه|بخات|نفخه|نفخات|بخاخ",
    "tbsp": r"ملعقه كبيره|ملاعق كبيره|ملعقتين كبيرتين",
    "tsp": r"ملعقه صغيره|ملاعق صغيره",
}
_AR_UNIT_ALT = "|".join(f"(?:{v})" for k, v in sorted(AR_UNITS.items(), key=lambda kv: -len(kv[1])))
AR_QTY = re.compile(NUM + r"\s*(" + _AR_UNIT_ALT + r")(?![ء-يa-zA-Z])")
AR_DUALS = {"قرصين": "tablet", "قرصان": "tablet", "لوحين": "tablet", "لوحان": "tablet", "حبتين": "tablet", "حبتان": "tablet",
            "كبسولتين": "capsule", "كبسولتان": "capsule", "قطرتين": "drop", "قطرتان": "drop",
            "نقطتين": "drop", "نقطتان": "drop", "بختين": "puff", "بختان": "puff", "نفختين": "puff",
            "نفختان": "puff", "وحدتين": "unit", "وحدتان": "unit", "ملعقتين كبيرتين": "tbsp", "ملعقتين صغيرتين": "tsp"}


AR_SINGULAR = {"tablet": r"قرص|قرصا|حبه|لوح|لوحا", "capsule": r"كبسوله", "drop": r"قطره|نقطه",
               "puff": r"بخه|نفخه", "unit": r"وحده", "tbsp": r"ملعقه كبيره"}


def _ar_unit(tok: str) -> str:
    for k, pat in AR_UNITS.items():
        if re.fullmatch(pat, tok):
            return k
    return tok


def _ar_freq(t: str):
    if re.search(r"اسبوعيا|في الاسبوع|كل اسبوع|بالاسبوع", t):
        return "weekly"
    m = re.search(r"كل\s+" + NUM + r"\s*(?:ساعات|ساعه)", t)
    if m:
        return round(24 / float(m.group(1)), 2)
    if re.search(r"كل ساعتين", t):
        return 12.0
    if re.search(r"مرتين|مرتان", t):
        return 2
    m = re.search(NUM + r"\s*مرات", t)
    if m:
        return float(m.group(1)) if not float(m.group(1)).is_integer() else int(float(m.group(1)))
    if re.search(r"\bمره\b|يوميا|كل يوم|عند النوم|قبل النوم", t):
        return 1
    return None


AR_DUR_UNITS = {"day": r"يوم|ايام|يوما", "week": r"اسبوع|اسابيع|اسبوعا", "month": r"شهر|اشهر|شهور|شهرا"}


def _ar_duration(t: str):
    for unit, pat in AR_DUR_UNITS.items():
        m = re.search(r"(?:لمده|مده|خلال|طوال)\s+" + NUM + r"\s*(?:" + pat + r")(?![ء-ي])", t)
        if m:
            return float(m.group(1)) * DUR_DAYS[unit]
    for word, days in (("يومين", 2), ("اسبوعين", 14), ("شهرين", 60)):
        if re.search(r"(?:لمده|مده|خلال|طوال)\s+" + word, t):
            return days
    for unit, pat in AR_DUR_UNITS.items():
        if re.search(r"(?:لمده|مده|خلال|طوال)\s+(?:" + pat + r")(?:\s+1)?(?![ء-ي])", t):
            return DUR_DAYS[unit]
    return None


AR_TIMING = {"before_meals": r"قبل (?:الاكل|الطعام|الوجبات|وجبات|الوجبه|تناول الطعام)",
             "after_meals": r"بعد (?:الاكل|الطعام|الوجبات|وجبات|الوجبه|تناول الطعام)",
             "with_food": r"مع (?:الطعام|الاكل|الوجبات|الوجبه)",
             "empty_stomach": r"معده فارغه|بطن فارغ|معده خاويه|الريق", "morning": r"الصباح|صباحا",
             "evening": r"المساء|مساء", "bedtime": r"النوم"}
AR_SITE = {"oral": r"الفم|بالفم|فمويا|عن طريق الفم", "sublingual": r"تحت اللسان",
           "rectal": r"الشرج|المستقيم|شرجيا", "eye": r"العين|للعين|العينين|عينيك|عينك",
           "ear": r"الاذن|للاذن|الاذنين|اذنك|اذنيك", "subcut": r"تحت الجلد",
           "iv": r"الوريد|وريدي", "im": r"العضل", "inhaled": r"استنشاق|استنشق",
           "topical": r"المنطقه المصابه|المكان المصاب|الجلد المصاب"}
AR_SIDE = {"left": r"اليسري|اليسار|الايسر", "right": r"اليمني|اليمين|الايمن", "both": r"كلتا|كلا|العينين|الاذنين"}
AR_PROHIBIT = re.compile(r"(?:^|\s|\.|،)(?:لا\s+ت(?!زال)|تجنب|يجب عدم|يمنع|ممنوع|اياك)")
AR_NEG = re.compile(r"(?<![ء-ي])(?:لا(?! ?زال)|ولا|فلا|لم|ولم|فلم|لن|ولن|ليس|ليست|ليسوا|لست|بدون|بلا|ابدا)(?![ء-ي])")


# ------------------------------------------------------------------------------ extraction

@dataclass
class Slots:
    qty: Counter = field(default_factory=Counter)  # (value, unit) -> count
    freq: object = None
    duration: float | None = None
    timing: set = field(default_factory=set)
    site: set = field(default_factory=set)
    side: set = field(default_factory=set)
    prohibitions: int = 0
    sentences: int = 0
    drugs: set = field(default_factory=set)
    numbers: Counter = field(default_factory=Counter)
    negated: bool = False
    text: str = ""  # normalized Arabic, for the lenient drug-name check


def base_name(d: L.Drug) -> str:
    """'paracetamol syrup' and 'paracetamol' are the same active ingredient."""
    return d.en.replace(" syrup", "").replace(" suspension", "")


def _strip_drug_names(t: str, names) -> str:
    for n in names:
        t = t.replace(n, " ")
    return t


def extract_en(text: str) -> Slots:
    t = re.sub(r"(?<=\d),(?=\d{3}(?!\d))", "", text.lower())  # 39,768 -> 39768
    s = Slots()
    s.drugs = {base_name(d) for d in L.DRUGS if re.search(r"\b" + re.escape(base_name(d)) + r"\b", t)}
    for m in EN_QTY.finditer(t):
        s.qty[(float(m.group(1)), _en_unit(m.group(2)))] += 1
    s.freq = _en_freq(t)
    s.duration = _en_duration(t)
    s.timing = {k for k, p in EN_TIMING.items() if re.search(p, t)}
    s.site = {k for k, p in EN_SITE.items() if re.search(p, t)}
    s.side = {k for k, p in EN_SIDE.items() if re.search(p, t)} if s.site & {"eye", "ear"} else set()
    s.prohibitions = len(EN_PROHIBIT.findall(t))
    s.negated = bool(EN_NEG.search(t))
    s.sentences = len([x for x in re.split(r"[.!?]\s*", t) if x.strip()])
    s.numbers = Counter(float(x) for x in re.findall(NUM, t))
    return s


_AR_DRUGS = None


def _ar_drug_table():
    global _AR_DRUGS
    if _AR_DRUGS is None:
        table = {normalize(d.ar.replace("شراب ", "").replace("معلق ", "")): base_name(d) for d in L.DRUGS}
        _AR_DRUGS = dict(sorted(table.items(), key=lambda kv: -len(kv[0])))
    return _AR_DRUGS


def _edit1(a: str, b: str) -> bool:
    """True if a and b differ by at most one edit (for MT spelling variants of drug names)."""
    if abs(len(a) - len(b)) > 1:
        return False
    if len(a) == len(b):
        return sum(x != y for x, y in zip(a, b)) <= 1
    if len(a) > len(b):
        a, b = b, a
    return any(b[:i] + b[i + 1:] == a for i in range(len(b)))


def _skeleton(w: str) -> str:
    """Consonant skeleton of a transliterated name: long vowels dropped, g/j/q spellings merged
    (غليبيزيد, جليبيزيد and قليبيزيد all -> جلبزد)."""
    return re.sub(r"[اويه]", "", w.translate(str.maketrans("غق", "جج")))


def _find_ar_drugs(t: str) -> set:
    found = {key for key in set(_ar_drug_table().values()) if re.search(rf"\b{re.escape(key)}\b", t, re.I)}
    words = re.findall(r"[ء-ي]+", t)
    for name, key in _ar_drug_table().items():
        if " " in name and name in t:
            found.add(key)
    single = {n: k for n, k in _ar_drug_table().items() if " " not in n}
    multi_tail = {n.split()[-1]: k for n, k in _ar_drug_table().items() if " " in n}  # 'حمض الفيوسيديك'
    for w in words:
        cands = {w, w[2:] if w.startswith("ال") else w, w[1:] if w[:1] in "بول" else w,
                 w[3:] if w[:3] in ("بال", "وال", "لل") else w}
        exact = [single[c] for c in cands if c in single]
        if exact:
            found.update(exact)
            continue
        # spelling variants: one edit on the letters or on the consonant skeleton. A variant that
        # is equally close to two drugs (omeprazole / esomeprazole) is not credited to either.
        close = set()
        for name, key in {**single, **{n[2:] if n.startswith("ال") else n: k for n, k in multi_tail.items()}}.items():
            if len(name) < 6:
                continue
            for c in cands:
                if len(c) >= 6 and (_edit1(c, name) or (len(_skeleton(name)) >= 5 and _edit1(_skeleton(c), _skeleton(name)))):
                    close.add(key)
        if len(close) == 1:
            found |= close
    return found


def _resembles(t: str, key: str, threshold: float = 0.8) -> bool:
    """Does any run of words in t look like the Arabic name of drug `key`?"""
    names = [n for n, k in _ar_drug_table().items() if k == key]
    words = [w[2:] if w.startswith("ال") and len(w) > 4 else w for w in re.findall(r"[ء-ي]+", t)]
    for name in names:
        name = " ".join(w[2:] if w.startswith("ال") and len(w) > 4 else w for w in name.split())
        k = len(name.split())
        for n in (k, k + 1):  # one extra word: a name split in two ('الكلور فينيكول')
            for i in range(len(words) - n + 1):
                window = " ".join(words[i:i + n])
                w, nm = window.replace(" ", ""), name.replace(" ", "")
                if SequenceMatcher(None, w, nm).ratio() >= threshold:
                    return True
                if len(_skeleton(nm)) >= 5 and SequenceMatcher(None, _skeleton(w), _skeleton(nm)).ratio() >= threshold:
                    return True
    return False


def extract_ar(text: str) -> Slots:
    t = number_words_to_digits(normalize(text))
    s = Slots()
    s.drugs = _find_ar_drugs(t)
    s.text = t
    names = [n for n in _ar_drug_table()]
    t_nodrug = _strip_drug_names(t, names)
    for m in AR_QTY.finditer(t_nodrug):
        s.qty[(float(m.group(1)), _ar_unit(m.group(2)))] += 1
    for word, unit in AR_DUALS.items():
        for _ in re.finditer(r"(?<![ء-ي])" + word + r"(?![ء-ي])", t_nodrug):
            s.qty[(2.0, unit)] += 1
    # singular counted nouns mean one: 'قرصا واحدا' (-> 'قرصا 1') or a bare 'ضع قطره في العين'
    for unit, pat in AR_SINGULAR.items():
        for _ in re.finditer(r"(?<!\d)(?<!\d )(?<![ء-ي])(?:" + pat + r")(?![ء-ي])", t_nodrug):
            s.qty[(1.0, unit)] += 1
    s.freq = _ar_freq(t_nodrug)
    s.duration = _ar_duration(t_nodrug)
    s.timing = {k for k, p in AR_TIMING.items() if re.search(p, t_nodrug)}
    s.site = {k for k, p in AR_SITE.items() if re.search(p, t_nodrug)}
    if any(u == "puff" for _, u in s.qty):  # 'خذ بختين' = two inhalations
        s.site.add("inhaled")
    s.side = {k for k, p in AR_SIDE.items() if re.search(p, t_nodrug)} if s.site & {"eye", "ear"} else set()
    if "both" in s.side and len(s.side) > 1 and not re.search("كلتا|كلا", t_nodrug):
        s.side.discard("both")
    s.prohibitions = len(AR_PROHIBIT.findall(" " + t_nodrug))
    s.negated = bool(AR_NEG.search(t_nodrug))
    s.sentences = len([x for x in re.split(r"[.!?؟]\s*", t_nodrug) if x.strip()])
    s.numbers = Counter(float(x) for x in re.findall(NUM, t_nodrug))
    s.numbers[2.0] += sum(len(re.findall(r"(?<![ء-ي])" + w + r"(?![ء-ي])", t_nodrug))
                          for w in DUAL_STEMS)
    return s


# ------------------------------------------------------------------------------ comparison

@dataclass
class Flag:
    kind: str
    detail: str

    def __str__(self):
        return f"[{self.kind}] {self.detail}"


@dataclass
class Report:
    flags: list
    src: Slots
    hyp: Slots

    @property
    def ok(self) -> bool:
        return not self.flags

    @property
    def score(self) -> float:
        """1.0 = no clinical disagreement; each flag costs 0.5 (floored at 0)."""
        return max(0.0, 1.0 - 0.5 * len(self.flags))


_SIDE_SET = lambda s: s.side - ({"both"} if len(s.side) > 1 else set())  # noqa: E731
_SITE_IGNORE = {"topical"}  # too loosely worded to compare safely
_EQUIV_UNIT = {"tablet": {"tablet", "capsule"}, "capsule": {"tablet", "capsule"}}


def _q_close(a, b):
    (va, ua), (vb, ub) = a, b
    return abs(va - vb) < 1e-6 and (ua == ub or ub in _EQUIV_UNIT.get(ua, ()))


def compare(src: Slots, hyp: Slots) -> list:
    flags: list[Flag] = []
    # quantities: every source quantity must appear with the same value and unit
    hyp_left = list(hyp.qty.elements())
    for q in src.qty.elements():
        hit = next((h for h in hyp_left if _q_close(q, h)), None)
        if hit:
            hyp_left.remove(hit)
            continue
        same_val = next((h for h in hyp_left if abs(h[0] - q[0]) < 1e-6), None)
        same_unit = next((h for h in hyp_left if h[1] == q[1] or h[1] in _EQUIV_UNIT.get(q[1], ())), None)
        if same_val:
            hyp_left.remove(same_val)
            flags.append(Flag("unit", f"{fmt_num(q[0])} {q[1]} rendered as {fmt_num(same_val[0])} {same_val[1]}"))
        elif same_unit:
            hyp_left.remove(same_unit)
            kind = "strength" if q[1] in ("mg", "mcg", "g") else "dose"
            flags.append(Flag(kind, f"{fmt_num(q[0])} {q[1]} rendered as {fmt_num(same_unit[0])} {same_unit[1]}"))
        else:
            flags.append(Flag("dose", f"{fmt_num(q[0])} {q[1]} missing in translation"))
    for h in hyp_left:
        if not any(_q_close(q, h) for q in src.qty):
            flags.append(Flag("dose", f"unexpected quantity {fmt_num(h[0])} {h[1]} in translation"))
    if src.freq is not None and src.freq != hyp.freq:
        flags.append(Flag("frequency", f"{src.freq}/day in source vs {hyp.freq} in translation"))
    if src.duration is not None and src.duration != hyp.duration:
        flags.append(Flag("duration", f"{fmt_num(src.duration)} days in source vs "
                                      f"{fmt_num(hyp.duration) if hyp.duration else 'none'} in translation"))
    t_src = src.timing - {"bedtime"}
    t_hyp = hyp.timing - {"bedtime"}
    if t_src != t_hyp and (t_src or t_hyp - {"morning"}):
        flags.append(Flag("timing", f"{sorted(t_src)} vs {sorted(t_hyp)}"))
    s_src, s_hyp = src.site - _SITE_IGNORE, hyp.site - _SITE_IGNORE
    if s_src and not s_src <= s_hyp or (s_hyp - s_src) & {"rectal", "iv", "im", "sublingual", "eye", "ear"}:
        flags.append(Flag("route", f"{sorted(s_src)} vs {sorted(s_hyp)}"))
    elif src.side and _SIDE_SET(src) != _SIDE_SET(hyp):
        flags.append(Flag("laterality", f"{sorted(src.side)} vs {sorted(hyp.side)}"))
    if src.prohibitions != hyp.prohibitions:
        kind = "omission" if hyp.sentences < src.sentences else "negation"
        flags.append(Flag(kind, f"{src.prohibitions} prohibition(s) in source vs {hyp.prohibitions} in translation"))
    # domain-general checks (free text outside medication instructions)
    if not flags:
        missing = [n for n in src.numbers if n != 1 and src.numbers[n] > hyp.numbers.get(n, 0)]
        if missing:
            flags.append(Flag("number", f"{', '.join(fmt_num(n) for n in missing)} missing or changed in translation"))
    if not any(f.kind in ("negation", "omission") for f in flags) and (src.negated != hyp.negated):
        flags.append(Flag("negation", f"source {'is' if src.negated else 'is not'} negated, translation "
                                      f"{'is' if hyp.negated else 'is not'}"))
    # a different known drug is always an error (LASA swaps); the source drug only has to be
    # recognizable, since MT transliterates names freely (إسوميبرازول, الكلور فينيكول)
    missing = {d for d in src.drugs - hyp.drugs if not _resembles(hyp.text, d)}
    if src.drugs and hyp.drugs - src.drugs:
        flags.append(Flag("drug", f"{sorted(src.drugs)} vs {sorted(hyp.drugs)}"))
    elif missing:
        flags.append(Flag("drug", f"{sorted(missing)} not found in translation"))
    return flags


def check(src_en: str, hyp_ar: str) -> Report:
    s, h = extract_en(src_en), extract_ar(hyp_ar)
    return Report(compare(s, h), s, h)
