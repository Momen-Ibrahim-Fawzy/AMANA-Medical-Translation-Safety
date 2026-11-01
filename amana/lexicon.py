"""Bilingual clinical lexicon for medication instructions (English source, Arabic target).

Arabic drug names follow the usual transliterations on Saudi/Egyptian package inserts.
Look-alike/sound-alike (LASA) pairs follow the ISMP confused-drug-names list.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .arabic import Noun


@dataclass(frozen=True)
class Drug:
    en: str
    ar: str
    form: str
    strengths: tuple[float, ...] = ()
    unit: str = "mg"  # strength unit


DRUGS: tuple[Drug, ...] = (
    # tablets
    Drug("paracetamol", "باراسيتامول", "tablet", (500,)),
    Drug("ibuprofen", "إيبوبروفين", "tablet", (200, 400)),
    Drug("metformin", "ميتفورمين", "tablet", (500, 850, 1000)),
    Drug("atorvastatin", "أتورفاستاتين", "tablet", (10, 20, 40)),
    Drug("amlodipine", "أملوديبين", "tablet", (5, 10)),
    Drug("losartan", "لوسارتان", "tablet", (50, 100)),
    Drug("warfarin", "وارفارين", "tablet", (1, 3, 5)),
    Drug("aspirin", "أسبرين", "tablet", (81, 100)),
    Drug("clopidogrel", "كلوبيدوجريل", "tablet", (75,)),
    Drug("furosemide", "فوروسيميد", "tablet", (20, 40)),
    Drug("levothyroxine", "ليفوثيروكسين", "tablet", (25, 50, 100), "mcg"),
    Drug("prednisolone", "بريدنيزولون", "tablet", (5, 20)),
    Drug("prednisone", "بريدنيزون", "tablet", (5, 20)),
    Drug("cetirizine", "سيتريزين", "tablet", (10,)),
    Drug("methotrexate", "ميثوتريكسات", "tablet", (2.5,)),
    Drug("tramadol", "ترامادول", "tablet", (50,)),
    Drug("trazodone", "ترازودون", "tablet", (50, 100)),
    Drug("hydralazine", "هيدرالازين", "tablet", (25,)),
    Drug("hydroxyzine", "هيدروكسيزين", "tablet", (25,)),
    Drug("metronidazole", "ميترونيدازول", "tablet", (250, 500)),
    Drug("bisoprolol", "بيسوبرولول", "tablet", (2.5, 5)),
    Drug("lisinopril", "ليسينوبريل", "tablet", (10, 20)),
    Drug("digoxin", "ديجوكسين", "tablet", (0.125, 0.25)),
    Drug("clonidine", "كلونيدين", "tablet", (0.1,)),
    Drug("clonazepam", "كلونازيبام", "tablet", (0.5,)),
    Drug("glimepiride", "جليميبيريد", "tablet", (1, 2)),
    Drug("glipizide", "جليبيزيد", "tablet", (5,)),
    # capsules
    Drug("amoxicillin", "أموكسيسيلين", "capsule", (250, 500)),
    Drug("omeprazole", "أوميبرازول", "capsule", (20, 40)),
    Drug("esomeprazole", "إيزوميبرازول", "capsule", (20, 40)),
    Drug("gabapentin", "جابابنتين", "capsule", (300,)),
    Drug("doxycycline", "دوكسيسيكلين", "capsule", (100,)),
    Drug("fluoxetine", "فلوكسيتين", "capsule", (20,)),
    Drug("celecoxib", "سيليكوكسيب", "capsule", (100, 200)),
    Drug("citalopram", "سيتالوبرام", "tablet", (20,)),
    # liquids (dose given in ml)
    Drug("paracetamol syrup", "شراب باراسيتامول", "syrup"),
    Drug("ibuprofen suspension", "معلق إيبوبروفين", "syrup"),
    Drug("amoxicillin suspension", "معلق أموكسيسيلين", "syrup"),
    Drug("lactulose syrup", "شراب لاكتولوز", "syrup"),
    # eye drops
    Drug("chloramphenicol", "كلورامفينيكول", "eye_drop"),
    Drug("timolol", "تيمولول", "eye_drop"),
    Drug("moxifloxacin", "موكسيفلوكساسين", "eye_drop"),
    Drug("latanoprost", "لاتانوبروست", "eye_drop"),
    # ear drops
    Drug("ciprofloxacin", "سيبروفلوكساسين", "ear_drop"),
    Drug("ofloxacin", "أوفلوكساسين", "ear_drop"),
    # inhalers
    Drug("salbutamol", "سالبوتامول", "inhaler", (100,), "mcg"),
    Drug("budesonide", "بوديزونيد", "inhaler", (200,), "mcg"),
    # insulin
    Drug("insulin glargine", "إنسولين جلارجين", "insulin"),
    Drug("insulin aspart", "إنسولين أسبارت", "insulin"),
    # creams
    Drug("hydrocortisone", "هيدروكورتيزون", "cream"),
    Drug("clotrimazole", "كلوتريمازول", "cream"),
    Drug("mupirocin", "موبيروسين", "cream"),
    Drug("fusidic acid", "حمض الفيوسيديك", "cream"),
)

DRUG_BY_EN = {d.en: d for d in DRUGS}

# ISMP look-alike / sound-alike pairs that exist in this lexicon (symmetric)
LASA_PAIRS = (
    ("hydralazine", "hydroxyzine"),
    ("tramadol", "trazodone"),
    ("clonidine", "clonazepam"),
    ("metformin", "metronidazole"),
    ("prednisolone", "prednisone"),
    ("celecoxib", "citalopram"),
    ("glimepiride", "glipizide"),
    ("omeprazole", "esomeprazole"),
    ("insulin glargine", "insulin aspart"),
    ("ciprofloxacin", "ofloxacin"),
)
LASA = {a: b for a, b in LASA_PAIRS} | {b: a for a, b in LASA_PAIRS}


@dataclass(frozen=True)
class Form:
    key: str
    verb_en: str
    noun_en: tuple[str, str]  # singular, plural
    verbs_ar: tuple[str, ...]  # first = reference style, others = benign alternatives
    neg_ar: dict = field(default_factory=dict)  # verb -> negated imperative
    noun: Noun | None = None
    counts: tuple[float, ...] = (1,)
    site: str = "oral"


FORMS = {
    "tablet": Form("tablet", "Take", ("tablet", "tablets"), ("تناول", "خذ"),
                   {"تناول": "لا تتناول", "خذ": "لا تأخذ"},
                   Noun("قرصاً واحداً", "قرصين", "أقراص", "قرصاً", "قرص واحد", "قرص"), (1, 2)),
    "capsule": Form("capsule", "Take", ("capsule", "capsules"), ("تناول", "خذ"),
                    {"تناول": "لا تتناول", "خذ": "لا تأخذ"},
                    Noun("كبسولة واحدة", "كبسولتين", "كبسولات", "كبسولة"), (1, 2)),
    "syrup": Form("syrup", "Take", ("ml", "ml"), ("تناول", "خذ"),
                  {"تناول": "لا تتناول", "خذ": "لا تأخذ"},
                  Noun("1 مل", "2 مل", "مل", "مل"), (2.5, 5, 10, 15)),
    "eye_drop": Form("eye_drop", "Instill", ("drop", "drops"), ("ضع", "قطّر"),
                     {"ضع": "لا تضع", "قطّر": "لا تقطّر"},
                     Noun("قطرة واحدة", "قطرتين", "قطرات", "قطرة"), (1, 2), "eye"),
    "ear_drop": Form("ear_drop", "Instill", ("drop", "drops"), ("ضع", "قطّر"),
                     {"ضع": "لا تضع", "قطّر": "لا تقطّر"},
                     Noun("قطرة واحدة", "قطرتين", "قطرات", "قطرة"), (2, 3, 4), "ear"),
    "inhaler": Form("inhaler", "Inhale", ("puff", "puffs"), ("استنشق", "خذ"),
                    {"استنشق": "لا تستنشق", "خذ": "لا تأخذ"},
                    Noun("بخة واحدة", "بختين", "بخات", "بخة"), (1, 2), "inhaled"),
    "insulin": Form("insulin", "Inject", ("unit", "units"), ("احقن", "أعط"),
                    {"احقن": "لا تحقن", "أعط": "لا تعط"},
                    Noun("وحدة واحدة", "وحدتين", "وحدات", "وحدة"), (6, 8, 10, 12, 16, 20, 24, 30), "subcut"),
    "cream": Form("cream", "Apply", ("", ""), ("ادهن", "ضع"),
                  {"ادهن": "لا تدهن", "ضع": "لا تضع"}, None, (1,), "topical"),
}

UNITS_AR = {"mg": ("ملغ", "مجم", "ملغم", "مليغرام"), "mcg": ("ميكروغرام", "مكغ", "ميكروجرام"),
            "g": ("غرام", "جم", "غ"), "ml": ("مل", "ملليلتر")}

# frequency: key -> (english, arabic reference, arabic benign alternative, doses/day)
FREQ = {
    "once_daily": ("once daily", "مرة واحدة يومياً", "مرة واحدة في اليوم", 1),
    "twice_daily": ("twice daily", "مرتين يومياً", "مرتين في اليوم", 2),
    "three_daily": ("three times daily", "ثلاث مرات يومياً", "ثلاث مرات في اليوم", 3),
    "four_daily": ("four times daily", "أربع مرات يومياً", "أربع مرات في اليوم", 4),
    "q4h": ("every 4 hours", "كل 4 ساعات", "مرة كل 4 ساعات", 6),
    "q6h": ("every 6 hours", "كل 6 ساعات", "مرة كل 6 ساعات", 4),
    "q8h": ("every 8 hours", "كل 8 ساعات", "مرة كل 8 ساعات", 3),
    "q12h": ("every 12 hours", "كل 12 ساعة", "مرة كل 12 ساعة", 2),
    "weekly": ("once weekly", "مرة واحدة أسبوعياً", "مرة واحدة في الأسبوع", 1 / 7),
    "bedtime": ("once daily at bedtime", "مرة واحدة يومياً عند النوم", "مرة واحدة في اليوم عند النوم", 1),
}
# clinically dangerous substitutions an MT system could plausibly make
FREQ_ERRORS = {
    "once_daily": ("twice_daily", "weekly", "four_daily"),
    "twice_daily": ("four_daily", "once_daily", "q4h"),
    "three_daily": ("once_daily", "q4h"),
    "four_daily": ("twice_daily", "q4h"),
    "q4h": ("q12h", "four_daily"),
    "q6h": ("q12h", "q4h"),
    "q8h": ("q4h", "once_daily"),
    "q12h": ("q4h", "once_daily"),
    "weekly": ("once_daily", "twice_daily"),
    "bedtime": ("twice_daily", "four_daily"),
}

TIMING = {
    "before_meals": ("before meals", "قبل الأكل", "قبل الطعام"),
    "after_meals": ("after meals", "بعد الأكل", "بعد الطعام"),
    "with_food": ("with food", "مع الطعام", "مع الأكل"),
    "empty_stomach": ("on an empty stomach", "على معدة فارغة", "على الريق"),
    "morning": ("in the morning", "في الصباح", "صباحاً"),
}
TIMING_ERRORS = {"before_meals": "after_meals", "after_meals": "before_meals",
                 "with_food": "empty_stomach", "empty_stomach": "with_food", "morning": "evening"}
TIMING_AR_EXTRA = {"evening": ("in the evening", "في المساء", "مساءً")}

# site/route: key -> (english, arabic reference, arabic benign alternative)
SITE = {
    "oral": ("by mouth", "عن طريق الفم", "بالفم"),
    "sublingual": ("under the tongue", "تحت اللسان", "تحت اللسان"),
    "rectal": ("rectally", "عن طريق الشرج", "شرجياً"),
    "eye_left": ("into the left eye", "في العين اليسرى", "في العين اليسرى"),
    "eye_right": ("into the right eye", "في العين اليمنى", "في العين اليمنى"),
    "eye_both": ("into both eyes", "في كلتا العينين", "في العينين"),
    "ear_left": ("into the left ear", "في الأذن اليسرى", "في الأذن اليسرى"),
    "ear_right": ("into the right ear", "في الأذن اليمنى", "في الأذن اليمنى"),
    "ear_both": ("into both ears", "في كلتا الأذنين", "في الأذنين"),
    "inhaled": ("by inhalation", "عن طريق الاستنشاق", "استنشاقاً"),
    "subcut": ("under the skin", "تحت الجلد", "تحت الجلد"),
    "iv": ("into a vein", "في الوريد", "في الوريد"),
    "im": ("into a muscle", "في العضل", "في العضل"),
    "topical": ("to the affected area", "على المنطقة المصابة", "على المكان المصاب"),
}
LATERALITY_ERRORS = {"eye_left": "eye_right", "eye_right": "eye_left", "eye_both": "eye_left",
                     "ear_left": "ear_right", "ear_right": "ear_left", "ear_both": "ear_right"}
ROUTE_ERRORS = {"oral": ("rectal", "sublingual"), "eye_left": ("ear_left",), "eye_right": ("ear_right",),
                "eye_both": ("ear_both",), "ear_left": ("eye_left",), "ear_right": ("eye_right",),
                "ear_both": ("eye_both",), "subcut": ("iv", "im")}

# duration units: key -> (english sg, english pl, Arabic noun in genitive context after لمدة)
DURATION_UNITS = {
    "day": ("day", "days", Noun("يوم واحد", "يومين", "أيام", "يوماً")),
    "week": ("week", "weeks", Noun("أسبوع واحد", "أسبوعين", "أسابيع", "أسبوعاً")),
    "month": ("month", "months", Noun("شهر واحد", "شهرين", "أشهر", "شهراً")),
}
DURATIONS = (("day", 3), ("day", 5), ("day", 7), ("day", 10), ("day", 14), ("week", 2),
             ("week", 6), ("month", 1), ("month", 3))


@dataclass(frozen=True)
class Warning_:
    key: str
    en: str
    ar: str  # reference
    ar_flipped: str  # polarity-flipped version (critical error)
    protective: bool  # True = a prohibition whose omission is dangerous
    forms: tuple[str, ...] = ("tablet", "capsule", "syrup")


WARNINGS = (
    Warning_("alcohol", "Do not drink alcohol while taking this medicine.",
             "لا تشرب الكحول أثناء تناول هذا الدواء.", "اشرب الكحول أثناء تناول هذا الدواء.", True),
    Warning_("stop", "Do not stop taking this medicine suddenly.",
             "لا تتوقف عن تناول هذا الدواء فجأة.", "توقف عن تناول هذا الدواء فجأة.", True),
    Warning_("drive", "Do not drive if you feel drowsy.",
             "لا تقد السيارة إذا شعرت بالنعاس.", "قد السيارة إذا شعرت بالنعاس.", True),
    Warning_("crush", "Do not crush or chew the tablets.",
             "لا تسحق الأقراص أو تمضغها.", "اسحق الأقراص أو امضغها.", True, ("tablet",)),
    Warning_("continue", "Keep taking it even if you feel better.",
             "استمر في تناوله حتى لو شعرت بتحسن.", "لا تستمر في تناوله إذا شعرت بتحسن.", False),
    Warning_("rash", "Stop using it and see a doctor if you develop a rash.",
             "توقف عن استخدامه وراجع الطبيب إذا ظهر لديك طفح جلدي.",
             "لا تتوقف عن استخدامه إذا ظهر لديك طفح جلدي.", False,
             ("tablet", "capsule", "syrup", "cream", "eye_drop", "ear_drop")),
    Warning_("contact_lens", "Do not wear contact lenses during treatment.",
             "لا ترتد العدسات اللاصقة أثناء العلاج.", "ارتد العدسات اللاصقة أثناء العلاج.", True, ("eye_drop",)),
    Warning_("swallow", "Do not swallow.", "لا تبلعه.", "ابلعه.", True, ("cream", "ear_drop")),
    Warning_("hypo", "Do not skip meals after the injection.",
             "لا تفوّت وجبات الطعام بعد الحقن.", "فوّت وجبات الطعام بعد الحقن.", True, ("insulin",)),
    Warning_("rinse", "Rinse your mouth with water after each use.",
             "اشطف فمك بالماء بعد كل استخدام.", "لا تشطف فمك بالماء بعد الاستخدام.", False, ("inhaler",)),
)
