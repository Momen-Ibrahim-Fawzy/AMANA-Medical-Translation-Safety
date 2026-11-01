"""Stress test of AMANA-Guard on Arabic written by hand, not by the benchmark renderer.

The cases deliberately target known weak spots (two drugs in one instruction, changed conditions,
garbled look-alike names, free paraphrase), so the pass rate is not an estimate of real-world
performance. Also reports 95% Wilson intervals and how often a garbled wrong drug name is accepted.

    python scripts/09_hard_cases.py
"""
from __future__ import annotations

import itertools
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from amana.guard import _ar_drug_table, check  # noqa: E402

# should_flag=True means the Arabic is clinically wrong
CASES = [
 ("count/freq swap", True, "Take 1 tablet of warfarin 5 mg by mouth twice daily.", "خذ قرصين من الوارفارين 5 ملغ عن طريق الفم مرة واحدة يوميا."),
 ("2 drugs: doses swapped", True, "Take 5 ml of paracetamol syrup and 10 ml of ibuprofen suspension by mouth.", "خذ 10 مل من شراب الباراسيتامول و5 مل من معلق الايبوبروفين عن طريق الفم."),
 ("2 drugs: timing swapped", True, "Take metformin 500 mg after meals and omeprazole 20 mg before meals.", "خذ ميتفورمين 500 ملغ قبل الوجبات وأوميبرازول 20 ملغ بعد الوجبات."),
 ("2 drugs: frequency swapped", True, "Take metformin 500 mg twice daily and omeprazole 20 mg once daily.", "خذ ميتفورمين 500 ملغ مرة واحدة يوميا وأوميبرازول 20 ملغ مرتين يوميا."),
 ("negation scope: with vs without food", True, "Do not take this medicine with food.", "لا تتناول هذا الدواء بدون طعام."),
 ("warning flipped, no negation word", True, "Stop using it and see a doctor if you develop a rash.", "استمر في استخدامه ولا داعي لزيارة الطبيب إذا ظهر طفح جلدي."),
 ("warning flipped 2", True, "Stop using it and see a doctor if you develop a rash.", "استمر في استخدامه إذا ظهر طفح جلدي."),
 ("wrong prohibition, same count", True, "Take 1 tablet of warfarin 5 mg by mouth once daily. Do not crush or chew the tablets.", "خذ قرصا واحدا من الوارفارين 5 ملغ عن طريق الفم مرة واحدة يوميا. لا تشرب الكحول."),
 ("condition dropped (only if pain)", True, "Take 1 tablet of ibuprofen 400 mg by mouth only if you have pain.", "خذ قرصا واحدا من الايبوبروفين 400 ملغ عن طريق الفم."),
 ("max dose changed", True, "Do not take more than 4 tablets in 24 hours.", "لا تتناول أكثر من 8 أقراص خلال 24 ساعة."),
 ("max dose to min dose", True, "Do not take more than 4 tablets in 24 hours.", "لا تتناول أقل من 4 أقراص خلال 24 ساعة."),
 ("garbled LASA partner: hydroxyzine", True, "Take 1 tablet of hydralazine 25 mg by mouth twice daily.", "خذ قرصا واحدا من هيدروكسيزن 25 ملغ عن طريق الفم مرتين يوميا."),
 ("garbled LASA partner: prednisone", True, "Take 1 tablet of prednisolone 5 mg by mouth once daily.", "خذ قرصا واحدا من بريدنيزن 5 ملغ عن طريق الفم مرة واحدة يوميا."),
 ("esomeprazole-like spelling for omeprazole", True, "Take 1 capsule of omeprazole 20 mg by mouth once daily.", "خذ كبسولة واحدة من السمبرازول 20 ملغ عن طريق الفم مرة واحدة يوميا."),
 ("SAFE: morning and evening = twice", False, "Take 1 tablet of metformin 500 mg by mouth twice daily.", "تناول قرصا واحدا من الميتفورمين 500 ملغ عن طريق الفم صباحا ومساء."),
 ("SAFE: every 12 hours words", False, "Take 1 tablet of metformin 500 mg by mouth every 12 hours.", "تناول قرصا واحدا من الميتفورمين 500 ملغ عن طريق الفم كل اثنتي عشرة ساعة."),
 ("SAFE: half tablet", False, "Take half a tablet of warfarin 5 mg by mouth once daily.", "تناول نصف قرص من الوارفارين 5 ملغ عن طريق الفم مرة واحدة يوميا."),
 ("SAFE: word order, dose phrase", False, "Take 1 tablet of warfarin 5 mg by mouth once daily in the evening.", "كل مساء، تناول عن طريق الفم قرصا واحدا من الوارفارين بجرعة 5 ملغ."),
 ("SAFE: for two weeks word", False, "Take 1 capsule of omeprazole 20 mg by mouth once daily for 14 days.", "تناول كبسولة واحدة من أوميبرازول 20 ملغ مرة واحدة يوميا عن طريق الفم لمدة أسبوعين."),
 ("SAFE: do not forget prohibition style", False, "Avoid alcohol while taking this medicine.", "ابتعد عن الكحول أثناء تناول هذا الدواء."),
]


def wilson(k: int, n: int, z: float = 1.96) -> list[float]:
    p, d = k / n, 1 + z * z / n
    c, h = (p + z * z / (2 * n)) / d, z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [round(100 * (c - h), 1), round(100 * (c + h), 1)]


def garbled_wrong_drug_accepted() -> list[list[str]]:
    """Source names drug A, translation has drug B with one interior letter dropped."""
    table = _ar_drug_table()
    name_of = {k: n for n, k in table.items()}
    accepted = []
    for a, b in itertools.permutations(sorted(set(table.values())), 2):
        n = name_of[b]
        for g in (n[:i] + n[i + 1:] for i in range(2, len(n) - 1)):
            if " " not in g and not any(f.kind == "drug" for f in check(f"Take {a}.", f"خذ {g}.").flags):
                accepted.append([a, b])
                break
    return accepted


def main():
    rows = []
    for name, should_flag, s, h in CASES:
        flags = [str(f) for f in check(s, h).flags]
        rows.append({"case": name, "should_flag": should_flag, "flags": flags,
                     "passed": bool(flags) == should_flag})
        print("PASS" if rows[-1]["passed"] else "FAIL", "|", name, "|", flags)
    bad = [r for r in rows if r["should_flag"]]
    safe = [r for r in rows if not r["should_flag"]]
    mt = json.loads((ROOT / "results" / "mt_audit.json").read_text(encoding="utf-8"))["summary"]
    tico = json.loads((ROOT / "results" / "guard_summary.json").read_text(encoding="utf-8"))["tico19_real_text"]
    pairs = garbled_wrong_drug_accepted()
    n_drugs = len(set(_ar_drug_table().values()))
    out = {
        "errors_caught": [sum(r["passed"] for r in bad), len(bad)],
        "safe_accepted": [sum(r["passed"] for r in safe), len(safe)],
        "garbled_wrong_drug_accepted": {"pairs": pairs, "n": len(pairs), "of": n_drugs * (n_drugs - 1)},
        "ci95_flag_rate": {s: {k: wilson(round(v["critical_flag_rate"] * v["n"]), v["n"]) for k, v in d.items()}
                           for s, d in mt.items()},
        "ci95_tico": {"recall": wilson(round(tico["recall"] * tico["n_bad"]), tico["n_bad"]),
                      "false_alarm": wilson(round(tico["false_alarm"] * tico["n_safe"]), tico["n_safe"])},
        "cases": rows,
    }
    (ROOT / "results" / "hard_cases.json").write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print({k: out[k] for k in ("errors_caught", "safe_accepted", "ci95_tico")}, "garbled accepted:", len(pairs))


if __name__ == "__main__":
    main()
