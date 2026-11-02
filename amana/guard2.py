"""AMANA-Guard 2: drug-centered checker for English -> Arabic medication instructions.

guard.py compares one flat set of slots per text. Here each drug gets its own record, so a dose or
a timing cannot be credited to the wrong drug. Warnings are read as (topic, polarity) pairs instead
of being counted. Each finding is either a flag (both sides were understood and they disagree) or a
review item (the checker could not establish agreement). A translation passes only if it has
neither. Text with no medication content goes to the guard.py checks. guard.py is kept unchanged as
the baseline.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from functools import lru_cache

from . import guard as G
from .arabic import fmt_num, normalize, number_words_to_digits
from .guard import Flag

# ------------------------------------------------------------------------------ drug names

# spellings seen in real MT output and accepted in review
ALIASES = {
    "سمبرازول": "esomeprazole", "اسوميبرازول": "esomeprazole", "كلور فينيكول": "chloramphenicol",
    "حمض الفوزيدي": "fusidic acid", "حمض الفوسيديك": "fusidic acid", "ليوثيروكسين": "levothyroxine",
    "يوثيروكسين": "levothyroxine", "ليفوتيروكسين": "levothyroxine", "غليبزيد": "glipizide",
    "غليبيزيد": "glipizide", "جليبزيد": "glipizide", "جليمبيريد": "glimepiride",
    "غليميبيريد": "glimepiride", "غليمبيريد": "glimepiride",
}
_FUZZY_MIN, _RIVAL_GAP, _FUZZY_MIN_LEN = 0.8, 0.03, 6


@dataclass
class Mention:
    start: int
    end: int
    key: str
    certain: bool = True
    rival: str | None = None  # another drug the spelling is about as close to


@lru_cache(maxsize=1)
def _names() -> tuple[dict, dict]:
    table = {**G._ar_drug_table(), **ALIASES}
    return ({n: k for n, k in table.items() if " " not in n}, {n: k for n, k in table.items() if " " in n})


def _sim(c: str, name: str) -> float:
    best = SequenceMatcher(None, c, name).ratio()
    sk = G._skeleton(name)
    if len(sk) >= 5:
        best = max(best, SequenceMatcher(None, G._skeleton(c), sk).ratio())
    return best


def _clitic_forms(w: str) -> set:
    return {w, w[2:] if w.startswith("ال") else w, w[1:] if w[:1] in "بولف" else w,
            w[3:] if w[:3] in ("بال", "وال", "لل", "فال") else w}


@lru_cache(maxsize=None)
def _classify(w: str) -> tuple[str, bool, str | None] | None:
    single, multi = _names()
    tails = {}
    for n, k in multi.items():
        tail = n.split()[-1]
        tails[tail[2:] if tail.startswith("ال") else tail] = k
    forms = _clitic_forms(w)
    for c in forms:
        if c in single:
            return single[c], True, None
    scores: dict[str, float] = {}
    for name, key in {**single, **tails}.items():
        if len(name) < _FUZZY_MIN_LEN:
            continue
        for c in forms:
            if len(c) >= _FUZZY_MIN_LEN and not (c.startswith('ال') and c[2:] in forms):
                scores[key] = max(scores.get(key, 0.0), _sim(c, name))
    ranked = sorted(scores.items(), key=lambda kv: -kv[1])
    if not ranked or ranked[0][1] < _FUZZY_MIN:
        return None
    rival = ranked[1][0] if len(ranked) > 1 and ranked[0][1] - ranked[1][1] < _RIVAL_GAP else None
    return ranked[0][0], False, rival


def _ar_mentions(t: str) -> list[Mention]:
    multi = _names()[1]
    found, taken = [], []
    for name, key in sorted(multi.items(), key=lambda kv: -len(kv[0])):
        for m in re.finditer(r"(?:ال)?" + re.escape(name), t):
            found.append(Mention(m.start(), m.end(), key))
            taken.append((m.start(), m.end()))
    for key in set(G._ar_drug_table().values()):
        for m in re.finditer(rf"\b{re.escape(key)}\b", t, re.I):
            found.append(Mention(m.start(), m.end(), key))
            taken.append((m.start(), m.end()))
    for m in re.finditer(r"[ء-ي]+", t):
        if any(a <= m.start() < b for a, b in taken):
            continue
        hit = _classify(m.group())
        if hit:
            found.append(Mention(m.start(), m.end(), hit[0], hit[1], hit[2]))
    return sorted(found, key=lambda x: x.start)


def _en_mentions(t: str) -> list[Mention]:
    found, taken = [], []
    keys = sorted(set(G._ar_drug_table().values()), key=len, reverse=True)
    for key in keys:
        for m in re.finditer(rf"\b{re.escape(key)}\b", t):
            if not any(a <= m.start() < b for a, b in taken):
                found.append(Mention(m.start(), m.end(), key))
                taken.append((m.start(), m.end()))
    return sorted(found, key=lambda x: x.start)


# ------------------------------------------------------------------------------ warnings

TOPICS = {  # topic: (English pattern, Arabic pattern on normalize()d text)
    "alcohol": (r"\balcohol\b", r"كحول"),
    "drive": (r"\bdrive\b|\bdriving\b", r"(?<![ء-ي])(?:تقد|تقود|قياده|القياده)(?![ء-ي])|قد السياره"),
    "crush": (r"\bcrush\b|\bchew\b", r"تسحق|اسحق|تمضغ|امضغ|سحق|مضغ"),
    "contact_lens": (r"contact lens", r"عدسات"),
    "swallow": (r"\bswallow\b", r"تبلع|ابلع|تبتلع|ابتلع"),
    "skip_meals": (r"\bskip\b", r"(?<![ء-ي])(?:تفوت|فوت)(?![ء-ي])"),
    "stop_suddenly": (r"\bsuddenly\b", r"فجاه"),
    "rash": (r"\brash\b", r"طفح"),
    "feel_better": (r"feel better", r"تحسن"),
    "rinse": (r"\brinse\b", r"شطف|اشطف|تشطف"),
}
EN_IMPERATIVE = re.compile(r"\s*(?:do not|don't|never|avoid|stop|keep|continue|rinse|see)\b")
EN_PRO = re.compile(r"\b(?:do not|don't|never|avoid|must not|should not)\b")
AR_PRO = re.compile(r"(?:^|\s|،)(?:لا\s+[تي](?!زال)|تجنب|ابتعد عن|يجب عدم|يمنع|ممنوع|اياك)")
EN_STOP, EN_GO = r"\bstop\b", r"\bkeep\b|\bcontinue\b"
AR_STOP, AR_GO = r"توقف|اوقف|ايقاف", r"استمر|واصل|تابع"


def _sign(topic: str, sentence: str, prohibited: bool, stop: str, go: str) -> int:
    base = 1
    if topic == "rash":  # the instruction is to stop
        base = -1 if re.search(go, sentence) and not re.search(stop, sentence) else 1
    elif topic == "feel_better":  # the instruction is to continue
        base = -1 if re.search(stop, sentence) and not re.search(go, sentence) else 1
    return -base if prohibited else base


# ------------------------------------------------------------------------------ conditions, limits

EN_COND = r"as needed|when needed|if needed|only if|only when|if you have|when you have"
AR_COND = r"كما هو مطلوب|عند الحاجه|عند اللزوم|حسب الحاجه|عند الضروره|فقط|في حال|عند الشعور|اذا كان لديك|اذا كنت تعاني"
EN_MAX, EN_MIN = r"more than|exceed|maximum|at most|up to", r"less than|fewer than|at least|minimum"
AR_MAX, AR_MIN = r"اكثر من|تتجاوز|يتجاوز|تزيد|اقصي", r"اقل من|ادني"
EN_WITHOUT = r"without (?:food|meals|eating)"
AR_WITHOUT = r"بدون (?:ال)?(?:طعام|اكل)|دون (?:ال)?(?:طعام|اكل)|من غير (?:ال)?(?:طعام|اكل)"
EN_VERB = r"\b(?:take|instill|apply|inhale|inject|use|give)\b"
AR_VERB = (r"(?<![ء-ي])[وف]?(?:خذ|تاخذ|تناول|ضع|تضع|وضع|قطر|ادهن|احقن|حقن|اعط|اعطاء|استنشق|استخدم|استخدام|"
           r"طبق|تطبيق|ادخل|ادخال|قم)(?![ء-ي])")
_WORDNUM = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "a": 1, "an": 1}
_COUNTED = r"tablets?|capsules?|drops?|puffs?|tablespoons?|teaspoons?|units?"


def _en_pre(text: str) -> str:
    t = re.sub(r"(?<=\d),(?=\d{3}(?!\d))", "", text.lower())
    t = re.sub(rf"\bhalf (?:of )?(?:a |an |one )?({_COUNTED})\b", r"0.5 \1", t)
    t = re.sub(rf"\b(one|two|three|four|five|six|seven|eight|a|an) ({_COUNTED})\b",
               lambda m: f"{_WORDNUM[m.group(1)]} {m.group(2)}", t)
    return re.sub(r"\btwo times\b", "2 times", t)


def _ar_pre(text: str) -> str:
    t = number_words_to_digits(normalize(text))
    t = re.sub(r"اثنتي 10|اثني 10|اثنتا 10|اثنا 10", "12", t)
    t = re.sub(r"(?<![ء-ي])ملغما(?![ء-ي])", "ملغم", t)
    return re.sub(r"(قبل|بعد) تناول (?=(?:ال)?وجب)", r"\1 ", t)


def _en_freq(t: str):
    if re.search(r"every other day", t):
        return 0.5
    if re.search(r"morning and (?:in the )?evening", t):
        return 2
    return G._en_freq(t)


def _ar_freq(t: str):
    if re.search(r"كل يومين|يوم بعد يوم", t):
        return 0.5
    if re.search(r"صباحا و(?:في )?مساء|الصباح و(?:في )?المساء|مساء و(?:في )?صباحا|المساء و(?:في )?الصباح", t):
        return 2
    if re.search(r"كل (?:صباح|مساء|ليله)", t):
        return 1
    return G._ar_freq(t)


# ------------------------------------------------------------------------------ document model

@dataclass
class Instruction:
    drug: str | None
    slots: G.Slots
    conditions: bool = False
    limits: set = field(default_factory=set)
    prohibited: int = 0  # prohibitions in this clause that no topic, limit or food rule explains


@dataclass
class Document:
    instructions: dict
    warnings: dict
    mentions: list
    unknown_prohibitions: int = 0
    has_verb: bool = False
    sites: set = field(default_factory=set)
    text: str = ""
    numbers: set = field(default_factory=set)


@dataclass
class _Lang:
    mentions: object
    pre: object
    extract: object
    freq: object
    pro: re.Pattern
    topic_idx: int
    stop: str
    go: str
    cond: str
    lim_max: str
    lim_min: str
    without: str
    with_food: str
    verb: str
    boundary: str
    split: str


EN = _Lang(_en_mentions, _en_pre, G.extract_en, _en_freq, EN_PRO, 0, EN_STOP, EN_GO, EN_COND, EN_MAX, EN_MIN, EN_WITHOUT,
           G.EN_TIMING["with_food"], EN_VERB, r",|;|\band\b|\bor\b|\bthen\b", r"\.(?!\d)|[;!?\n]")
AR = _Lang(_ar_mentions, _ar_pre, G.extract_ar, _ar_freq, AR_PRO, 1, AR_STOP, AR_GO, AR_COND, AR_MAX, AR_MIN, AR_WITHOUT,
           G.AR_TIMING["with_food"], AR_VERB, r"،|,|؛|(?:^|\s)و|\sثم\s|\sاو\s", r"\.(?!\d)|[؟!؛\n]")


def _clauses(sentence: str, ms: list[Mention], lang: _Lang) -> list[tuple[str | None, str]]:
    """Cut a sentence between mentions of different drugs, at the coordinator nearest the second."""
    ms = [m for i, m in enumerate(ms) if i == 0 or m.key != ms[i - 1].key]
    if len(ms) < 2:
        return [(ms[0].key if ms else None, sentence)]
    out, start = [], 0
    for a, b in zip(ms, ms[1:]):
        cuts = [x.start() for x in re.finditer(lang.boundary, sentence[a.end:b.start + 1])]
        cut = a.end + cuts[-1] if cuts else b.start
        out.append((a.key, sentence[start:cut]))
        start = cut
    out.append((ms[-1].key, sentence[start:]))
    return out


def parse(text: str, lang: _Lang) -> Document:
    t = lang.pre(text)
    doc = Document({}, {}, lang.mentions(t), text=t)
    doc.has_verb = bool(re.search(lang.verb, t))
    buffers: dict = {}
    extras: dict = {}
    current, pending = None, []
    for sentence in re.split(lang.split, t):
        if not sentence.strip():
            continue
        ms = lang.mentions(sentence)
        prohibited = bool(lang.pro.search(sentence))
        topics = [k for k, pats in TOPICS.items() if re.search(pats[lang.topic_idx], sentence)]
        if lang is EN and not EN_IMPERATIVE.match(sentence):
            topics = []  # 'alcohol-based hand sanitizer' is not a warning about alcohol
        for k in topics:
            doc.warnings[k] = _sign(k, sentence, prohibited, lang.stop, lang.go)
        extract = lang.extract(sentence)
        if topics and not ms and not extract.qty:
            continue
        has_food = re.search(lang.with_food, sentence) or re.search(lang.without, sentence)
        has_limit = re.search(lang.lim_max, sentence) or re.search(lang.lim_min, sentence)
        unknown = prohibited and not topics and not has_food and not has_limit
        doc.unknown_prohibitions += unknown
        for key, clause in _clauses(sentence, ms, lang):
            key = key or current
            if key is None:
                pending.append(clause)
            else:
                buffers.setdefault(key, []).append(clause)
                current = key
            if key is not None or not ms:
                ex = extras.setdefault(key, {"prohibited": 0, "food": set(), "limits": set()})
                ex["prohibited"] += unknown
                ex["limits"] |= ({"max"} if re.search(lang.lim_max, clause) else set()) | \
                                ({"min"} if re.search(lang.lim_min, clause) else set())
                without = re.search(lang.without, clause)
                withf = re.search(lang.with_food, clause)
                if without or withf:
                    ex["food"].add(("empty_stomach" if without else "with_food") if not prohibited
                                   else ("with_food" if without else "empty_stomach"))
    if pending:
        first = next(iter(buffers), None)
        buffers[first] = pending + buffers.get(first, [])
    for key, parts in (buffers or {None: []}).items():
        body = ". ".join(parts)
        s = lang.extract(body)
        s.freq = lang.freq(body)
        s.timing = set(s.timing)
        ex = extras.get(key, {}) if key in extras else extras.get(None, {})
        if ex.get("food"):
            s.timing = (s.timing - {"with_food", "empty_stomach"}) | ex["food"]
        doc.instructions[key] = Instruction(key, s, bool(re.search(lang.cond, body)), ex.get("limits", set()),
                                            ex.get("prohibited", 0))
        doc.sites |= s.site
    doc.numbers = set(lang.extract(t).numbers)
    return doc


def _is_medication(doc: Document) -> bool:
    return bool(doc.mentions or doc.warnings or any(i.slots.qty or i.slots.freq for i in doc.instructions.values())
                or re.search(r"\b(?:medicine|medication|tablets?|capsules?)\b", doc.text))


# ------------------------------------------------------------------------------ comparison

@dataclass
class Report2:
    flags: list
    review: list
    src: object = None
    hyp: object = None

    @property
    def ok(self) -> bool:
        return not self.flags and not self.review

    @property
    def status(self) -> str:
        return "flag" if self.flags else "needs_review" if self.review else "ok"

    @property
    def score(self) -> float:
        return max(0.0, 1.0 - 0.5 * len(self.flags) - 0.25 * len(self.review))


def _quantities(src: G.Slots, hyp: G.Slots, tag: str, flags: list, review: list):
    left = list(hyp.qty.elements())
    for q in src.qty.elements():
        hit = next((h for h in left if G._q_close(q, h)), None)
        if hit:
            left.remove(hit)
            continue
        same_val = next((h for h in left if abs(h[0] - q[0]) < 1e-6), None)
        same_unit = next((h for h in left if h[1] == q[1] or h[1] in G._EQUIV_UNIT.get(q[1], ())), None)
        if same_val:
            left.remove(same_val)
            flags.append(Flag("unit", f"{tag}{fmt_num(q[0])} {q[1]} rendered as {fmt_num(same_val[0])} {same_val[1]}"))
        elif same_unit:
            left.remove(same_unit)
            kind = "strength" if q[1] in ("mg", "mcg", "g") else "dose"
            flags.append(Flag(kind, f"{tag}{fmt_num(q[0])} {q[1]} rendered as {fmt_num(same_unit[0])} {same_unit[1]}"))
        else:
            review.append(Flag("dose", f"{tag}{fmt_num(q[0])} {q[1]} not found in translation"))
    for h in left:
        if not any(G._q_close(q, h) for q in src.qty):
            flags.append(Flag("dose", f"{tag}unexpected quantity {fmt_num(h[0])} {h[1]} in translation"))


def _same_freq(a, b) -> bool:
    if isinstance(a, str) or isinstance(b, str):
        return a == b
    return abs(float(a) - float(b)) < 1e-6


def _compare_instruction(a: Instruction, b: Instruction, hyp_doc: Document, flags: list, review: list):
    tag = f"{a.drug}: " if a.drug else ""
    src, hyp = a.slots, b.slots
    _quantities(src, hyp, tag, flags, review)
    if src.freq is not None:
        if hyp.freq is None:
            review.append(Flag("frequency", f"{tag}{src.freq}/day not found in translation"))
        elif not _same_freq(src.freq, hyp.freq):
            flags.append(Flag("frequency", f"{tag}{src.freq}/day in source vs {hyp.freq} in translation"))
    if src.duration is not None:
        if hyp.duration is None:
            review.append(Flag("duration", f"{tag}{fmt_num(src.duration)} days not found in translation"))
        elif src.duration != hyp.duration:
            flags.append(Flag("duration", f"{tag}{fmt_num(src.duration)} days in source vs {fmt_num(hyp.duration)}"))
    elif hyp.duration is not None:
        flags.append(Flag("duration", f"{tag}no duration in source, {fmt_num(hyp.duration)} days in translation"))
    t_src, t_hyp = src.timing - {"bedtime"}, hyp.timing - {"bedtime"}
    if not t_src and {"morning", "evening"} <= t_hyp and src.freq is not None and _same_freq(src.freq, 2):
        t_hyp -= {"morning", "evening"}  # 'morning and evening' is how twice daily is said
    if t_src != t_hyp:
        if t_src and not t_hyp:
            review.append(Flag("timing", f"{tag}{sorted(t_src)} not found in translation"))
        elif t_src or t_hyp - {"morning"}:
            flags.append(Flag("timing", f"{tag}{sorted(t_src)} vs {sorted(t_hyp)}"))
    s_src, s_hyp = src.site - G._SITE_IGNORE, hyp.site - G._SITE_IGNORE
    danger = {"rectal", "iv", "im", "sublingual", "eye", "ear"}
    if (s_hyp - s_src) & danger or (s_src and s_hyp and not s_src <= s_hyp):
        flags.append(Flag("route", f"{tag}{sorted(s_src)} vs {sorted(s_hyp)}"))
    elif s_src and not s_src <= (hyp_doc.sites - G._SITE_IGNORE):
        review.append(Flag("route", f"{tag}{sorted(s_src)} not found in translation"))
    elif src.side and s_src <= s_hyp and G._SIDE_SET(src) != G._SIDE_SET(hyp):
        (flags if hyp.side else review).append(Flag("laterality", f"{tag}{sorted(src.side)} vs {sorted(hyp.side)}"))
    if a.conditions != b.conditions:
        side = "translation adds a condition" if b.conditions else "condition in source is missing"
        flags.append(Flag("condition", f"{tag}{side}"))
    if a.limits != b.limits:
        flags.append(Flag("limit", f"{tag}{sorted(a.limits) or 'no limit'} vs {sorted(b.limits) or 'no limit'}"))
    if b.prohibited > a.prohibited:
        flags.append(Flag("negation", f"{tag}translation prohibits something the source does not"))


def _compare_warnings(src: Document, hyp: Document, flags: list, review: list):
    for k, v in src.warnings.items():
        if k not in hyp.warnings:
            if hyp.unknown_prohibitions > src.unknown_prohibitions:
                review.append(Flag("omission", f"warning '{k}' not recognized in translation"))
            else:
                flags.append(Flag("omission", f"warning '{k}' missing in translation"))
        elif hyp.warnings[k] != v:
            flags.append(Flag("negation", f"warning '{k}' reversed in translation"))
    for k in hyp.warnings:
        if k not in src.warnings:
            flags.append(Flag("warning", f"translation has a warning about '{k}' that the source does not"))
    if src.unknown_prohibitions > hyp.unknown_prohibitions and not set(src.warnings) - set(hyp.warnings):
        flags.append(Flag("omission", f"{src.unknown_prohibitions} prohibition(s) in source vs "
                                      f"{hyp.unknown_prohibitions} in translation"))


def _leftover_numbers(src_text: str, doc: Document) -> set:
    t = re.sub(r"\bfor\s+[\d.]+\s*\w+|\bevery\s+[\d.]+\s*\w+|\b[\d.]+\s+times\b", " ", src_text)
    nums = {float(x) for x in re.findall(G.NUM, t)}
    used = {q[0] for i in doc.instructions.values() for q in i.slots.qty}
    return {n for n in nums - used if n != 1}


def compare(src: Document, hyp: Document) -> tuple[list, list]:
    flags: list[Flag] = []
    review: list[Flag] = []
    src_keys = [k for k in src.instructions if k]
    # resolve hypothesis mentions against the drugs the source names
    credited = {}
    for m in hyp.mentions:
        if m.certain:
            credited.setdefault(m.key, "certain")
        elif m.rival and (m.key in src_keys or m.rival in src_keys):
            own = m.key if m.key in src_keys else m.rival
            other = m.rival if own == m.key else m.key
            review.append(Flag("drug", f"spelling is about as close to {other} as to {own}"))
            credited.setdefault(own, "near")
        else:
            credited.setdefault(m.key, "near")
    if src_keys:
        for k in credited:
            if k not in src_keys:
                flags.append(Flag("drug", f"{sorted(src_keys)} vs {k} in translation"))
        for k in src_keys:
            if k not in credited:
                flags.append(Flag("drug", f"{k} not found in translation"))
    # pair instructions by drug; what is left is paired in order, so a swapped drug still has its
    # dose and timing compared
    hyp_left = dict(hyp.instructions)
    pairs = []
    for k, ins in src.instructions.items():
        if k in hyp_left:
            pairs.append((ins, hyp_left.pop(k)))
    src_left = [ins for ins in src.instructions.values() if not any(ins is a for a, _ in pairs)]
    for ins in src_left:
        if hyp_left:
            pairs.append((ins, hyp_left.pop(next(iter(hyp_left)))))
    for a, b in pairs:
        _compare_instruction(a, b, hyp, flags, review)
    _compare_warnings(src, hyp, flags, review)
    if src.has_verb and not hyp.has_verb:
        review.append(Flag("action", "no dosing verb recognized in translation"))
    missing = _leftover_numbers(src.text, src) - hyp.numbers
    if missing and not flags:
        review.append(Flag("number", f"{', '.join(fmt_num(n) for n in sorted(missing))} not found in translation"))
    return flags, review


def check(src_en: str, hyp_ar: str) -> Report2:
    s, h = parse(src_en, EN), parse(hyp_ar, AR)
    if not _is_medication(s):
        baseline = G.check(src_en, hyp_ar)
        return Report2(baseline.flags, [], s, h)
    flags, review = compare(s, h)
    return Report2(flags, review, s, h)
