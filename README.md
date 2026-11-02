# AMANA (أمانة): clinical meaning preservation in English → Arabic medication translation

*Amana* means faithfulness to what you were entrusted with.

A translated medication instruction can be fluent, get a strong MT score, and still tell the patient
to take **25 mg instead of 25 mcg**, or to put drops in the **right** eye instead of the left. AMANA
is a research framework for measuring this. It is **not a clinically validated safety system**.

The most useful result here is a gap, not a high score. The checkers look excellent on the benchmark
they were built with, and much weaker as the test moves away from it:

| evaluation track | what is new to the checker | AMANA-Guard 2 | AMANA-QE |
|---|---|---|---|
| Held-out drugs (AMANA-Bench) | the drug only; same templates | recall 100%, false alarms 0% *(upper bound)* | AUROC 0.999 |
| Unseen dosage forms | eye drops, inhalers | not applicable (rules) | AUROC 0.85–0.86; false alarms 70–100% at 0.5 |
| Unseen error types | unit, laterality, negation | not applicable (rules) | AUROC 0.67–0.71; recall 31–36% at 0.5 |
| Real text, controlled errors (AMANA-TICO) | the wording; errors are still injected | recall 94.7%, false alarms 13.8% *(development data)* | AUROC 0.80; false alarms 70% at 0.5 |
| Hand-written hold-out (n = 40) | the phrasing | 20/20 errors, 14/20 safe *(same author, not blind)* | not run |
| Naturally occurring errors, independently annotated | everything | **not done** | **not done** |

AMANA-QE was not tested on the hand-written cases. Its unseen-type and unseen-form rows use two seeds each.

## Key findings

**1. When a safe translation is worded differently from the reference, standard metrics often prefer
a one-word clinical error.** The "reworded" test pairs a safe rewording with the reference plus one
clinical error. The "same wording" test applies the error to the safe rewording itself. There are
1,620 pairs on 13 held-out drugs, covering 11 error types. Intervals are 95% bootstrap intervals over
pairs, and pairs built from similar instructions are not fully independent.

| metric | safe preferred: reworded (95% CI) | reworded, 10× strength errors only | safe preferred: same wording (95% CI) |
|---|---:|---:|---:|
| BLEU | 23.3% (21.3–25.3) | 9% | 94.2% (93.4–95.0) |
| TER | 26.2% (24.3–28.0) | 7% | 93.3% (92.4–94.2) |
| BERTScore | 30.7% (28.6–33.0) | 3% | 95.6% (94.5–96.5) |
| chrF++ | 32.9% (30.5–35.1) | 2% | 98.6% (97.9–99.1) |
| chrF | 35.7% (33.3–38.0) | 2% | 98.6% (98.0–99.1) |
| COMET-22 | 46.7% (44.3–49.0) | **1%** | 94.6% (93.5–95.7) |
| LaBSE, reference | 82.9% (81.1–84.7) | 99% | 99.7% (99.4–99.9) |
| LaBSE, source only | 81.6% (79.8–83.4) | 98% | 97.2% (96.4–98.0) |

Ties count as half. They matter only for BLEU (5.4%) and TER (16.5%). So these metrics can see the
error: with the wording held equal, they catch it at least 93% of the time. The failure is one of
weighting, since a harmless rewording can cost more score than a tenfold dose change. The reworded
test is built to expose this, because the erroneous sentence is one word from the reference. It does
not estimate how metrics behave on natural MT output. Margins, ties and per-type results are in
`results/metric_stats.json`.

![A harmless rewording can cost more score than a clinical error](figures/metric_blindness.png)

The heatmap breaks the reworded test down by error type.

![Results by error type](figures/blindness_by_error_type.png)

**2. Open MT systems make these errors, and a high chrF does not rule them out.** Each system translated
300 instructions for held-out drugs. These inputs are generated, not real prescriptions, so this is an
exploratory audit, not an estimate of how often these models err on clinical text or a safety ranking.

| | OPUS-MT en-ar | NLLB-200-600M | M2M100-418M |
|---|---:|---:|---:|
| chrF / COMET-22 | **74.4** / 0.862 | 67.5 / **0.878** | 66.2 / 0.839 |
| not passed by AMANA-Guard 2 (95% CI) | 25.0% (20.4–30.2) | **6.7%** (4.4–10.1) | 43.0% (37.5–48.7) |
| of which flagged / needs review only | 9.7% / 15.3% | 6.7% / 0% | 43.0% / 0% |
| "mcg" doses written as "mg" (1000×) | **18 / 18** | 0 / 18 | 2 / 18 |
| "rash" warning → "البراز" ("stool") | 0 / 46 | 0 / 46 | **40 / 46** |
| safety warnings silently dropped | 2 / 109 | 19 / 109 | **57 / 109** |
| glimepiride (diabetes) → "حبوب منع الحمل" (contraceptive pills) | 0 / 26 | 0 / 26 | **5 / 26** |
| "Instill" (eye drops) → "لا تزال" ("still") | **44 / 44** | 0 / 44 | 0 / 44 |

The pattern rows are exact counts of each pattern in `results/mt_error_patterns.json`. The rates in the
second row are checker output, and no independent reviewer has adjudicated them. The first checker put
M2M100 at 21.3%. Guard 2 reports 43.0% mainly because it now checks what each warning says, not only
whether a warning is present. Most of that increase comes from warnings it can no longer find, such as
the "rash" sentence above.

**3. Reranking helps only when a clean candidate exists, which here means mostly NLLB.** The checkers
choose among each system's own 5-best beam candidates, with no retraining.

| not passed by Guard 2 | 1-best | QE rerank (does not use the guard) | Guard 2 + QE rerank | all 5 candidates fail |
|---|---:|---:|---:|---:|
| NLLB-200-600M | 6.7% | 3.7% | **2.3%** | 7 / 300 |
| OPUS-MT en-ar | 25.0% | 27.3% | 24.7% | 74 / 300 |
| M2M100-418M | 43.0% | 42.7% | 42.0% | 126 / 300 |

The Guard 2 + QE column is partly circular, because Guard 2 both chooses and judges. It shows that
candidates with a detected problem were avoided, not that the chosen output is safe. The QE column is
the fairer cross-check, and it makes OPUS-MT slightly worse. The reverse cross-check, the share AMANA-QE
scores below 0.5, moves little after Guard 2 + QE: NLLB 13.0% → 9.7%, OPUS-MT 27.7% → 26.3%, M2M100 21.7%
→ 20.3%. Changes in chrF stay within ±1.2 and in COMET-22 within ±0.003.

![MT audit](figures/mt_audit.png)

## Try it

The checkers use the Python standard library only.

```console
$ python -m amana "Take warfarin 3 mg once daily and aspirin 75 mg once daily." "خذ وارفارين 75 ملغ مرة واحدة يومياً وأسبرين 3 ملغ مرة واحدة يومياً."
flag
  flag   [strength] warfarin: 3 mg rendered as 75 mg
  flag   [strength] aspirin: 75 mg rendered as 3 mg

$ python -m amana "Instill 1 drop of timolol into the left eye twice daily." "لا تزال قطرة واحدة من تيمولول في العين اليسرى مرتين يوميا."
needs_review
  review [action] no dosing verb recognized in translation
```

The exit code is 0 for passes, 1 for flagged and 2 for needs review. `flag` means a detected mismatch.
`needs_review` means the checker could not establish the meaning. It is not evidence of an error, and
passing is not evidence of safety: the checker cannot see what it has no rule for.

```python
from amana.guard2 import check

report = check(source_en, translation_ar)
report.status, report.flags, report.review
```

## What is in the repository

| component | what it is |
|---|---|
| **AMANA-Bench** | 25,258 contrastive rows from 3,180 generated English instructions: 56 drugs, 8 dosage forms, 11 critical error types including ISMP look-alike/sound-alike pairs. Split by drug. |
| **AMANA-TICO** | 703 real professional EN→AR sentences from TICO-19 with one number or negation error injected. The text is real and the errors are constructed, and it covers only these 2 of the 11 error types. |
| **AMANA-Guard** (`amana/guard.py`) | The first rule-based checker, kept unchanged as the baseline. It compares flat slots. |
| **AMANA-Guard 2** (`amana/guard2.py`) | Parses each side into per-drug instructions and warnings with a topic and polarity, then compares them. It separates flags from cases it needs a person to review. It is still rules and regular expressions, with a closed list of 10 warning topics. |
| **AMANA-QE** | LaBSE cross-encoder fine-tuned on training drugs. It returns a **score for ranking, not a calibrated probability**. |
| **Adjudication packet** (`adjudication/packet.html`) | 60 pairs of MT candidates for a pharmacist or clinician to judge, blinded to system and checker. Not yet completed by anyone. |

## How far to trust each result

**The guard's 100% on AMANA-Bench is an upper bound.** It was developed against those templates. Every
instruction is generated from the same grammar it parses.

**The hand-written hold-out is weak evidence, and it is now spent.** `data/hard_cases_holdout.jsonl`
holds 20 errors and 20 safe paraphrases, frozen by SHA-256 before Guard 2 was implemented and never used
to tune it. The same author wrote the cases and the checker, knowing its aims, so this is not a blind
test, and 40 cases give wide intervals.

| run (file in `results/`) | errors caught (95% CI) | safe accepted (95% CI) |
|---|---:|---:|
| AMANA-Guard (baseline) | 13 / 20 | 10 / 20 |
| Guard 2, first run (`guard2_holdout_first_run.json`) | 19 / 20 (76.4–99.1) | 14 / 20 (48.1–85.5) |
| Guard 2, final code (`guard2_holdout.json`) | 20 / 20 (83.9–100) | 14 / 20 (48.1–85.5) |

Three code changes separate the runs: two literal control characters in regular expressions, and a change
to fuzzy drug matching prompted by a false match on development data. None was made by tuning on a hold-out
case, but all came after seeing the first score, and together they changed one result (h10). The first run
is the cleaner estimate. The 6 false alarms
are safe paraphrases that remain flagged: another verb for a prohibition, "must not", a paraphrased
maximum, "thrice" in words, a number word in place of a dual, and fasting wording. They were left
untouched on purpose. Any new claim needs fresh cases written by someone else.

**The TICO-19 and MT rates are development data for the guard.** False alarms on TICO-19 references and
reading MT outputs both led to guard changes. TICO-19 remains unseen for AMANA-QE. Guard 2's TICO-19
results (94.7%, 13.8%) are no better than the baseline's (94.9%, 13.5%), since this track has no medication
structure for it to use.

**AMANA-QE learned the generator to a large degree.** The drop from 0.999 to 0.67–0.71 on error types left
out of training is the clearest evidence. Calibration also fails away from the benchmark: expected
calibration error is 0.009 on held-out drugs, 0.26–0.36 on the unseen tracks, and 0.46 on TICO-19. At the
0.5 threshold, false alarms swing from 0% to 100% across tracks, and between 70% and 100% just from the
seed on unseen forms. It should not be used as a pass/fail gate.

**Human review so far is limited.** The author, a native Arabic speaker but not a clinician, judged 12
flags from the first guard and 20 benchmark instructions. All 20 instructions were judged correct, and
4 of the 12 flags were false alarms, which were then fixed. Guard 2's flags have not been judged by a
person outside development. No clinician has reviewed anything here.

## Known failures

- **Closed vocabulary.** Guard 2 knows 56 drugs plus aliases and 10 warning topics. A warning on any other
  topic is invisible unless other checks fire, and a drug outside the list cannot be verified.
- **Garbled drug names.** A garbled name of a different drug is accepted for 2 of 2,450 ordered drug pairs
  (baseline: 15).
- **No scope reasoning beyond simple patterns.** It handles per-drug doses, simple conditions and limits.
  Long-distance pronouns, exceptions and nested conditions were not tested in depth.
- **No severity.** All errors count equally. Severity tiers need clinical input.
- **Standard Arabic only.** Dialectal patient-facing Arabic is not covered.
- **The dev hard cases are now fitted.** `tests/test_guard2.py` passes all 20 cases because they were used
  to build Guard 2. `tests/test_hard_cases.py` keeps them as 14 strict expected failures against the
  baseline, so an unexplained change in either fails the suite.
- **Real data gaps.** AMANA-TICO has 0 of the 104 CMU doctor-patient sentences in TICO-19's test split:
  they contain no digits and no English negation, so the injector has nothing to change.

## Reproduce

Python 3.10 or newer. `requirements.txt` lists what the experiments need, and `requirements.lock.txt` is
the exact environment used (Python 3.11, CUDA 12.4 PyTorch build, which needs PyTorch's own package index).

```bash
pip install -r requirements.txt
python scripts/run_all.py --list                # show every step
python scripts/run_all.py --stages report       # no GPU: rebuilds summaries and figures
python scripts/run_all.py                       # everything: needs a GPU and model downloads
python scripts/run_all.py --from-step 8         # resume after a failure, using the numbers from --list
pytest tests
python scripts/10_guard2_eval.py --holdout      # re-scores the spent hold-out; do not tune on it
```

| stage | what it does | GPU | results in the repository |
|---|---|---|---|
| `data` | builds AMANA-Bench and AMANA-TICO, downloading TICO-19 | no | AMANA-TICO only; AMANA-Bench is rebuilt from a fixed seed and a test checks its hash |
| `audit` | scores metrics on the 1,620 contrastive pairs, with intervals | yes | yes (`metric_audit.json`, `metric_stats.json`) |
| `qe` | trains AMANA-QE and 4 generalization runs, about 70–85 minutes per generalization run on an 8 GB GPU | yes | yes, except checkpoints |
| `mt` | translates with 3 systems, scores candidates | yes | yes (`mt_outputs.jsonl`, `mt_audit.json`) |
| `report` | rebuilds AMANA-Bench, then guard evaluations, error patterns, figures, manifest | no | yes (`MANIFEST.json` has file hashes) |

Not run: NLLB-200-1.3B and CometKiwi-22, which could not be loaded. GitHub Actions runs the tests on Python 3.10–3.12.

See [docs/METHOD.md](docs/METHOD.md) for the method, [docs/REVIEWER_QA.md](docs/REVIEWER_QA.md) for the
hard questions, and [data/README.md](data/README.md) for the dataset card.

## Not a medical device

This is a research tool. It must not be used to approve translations for patients, and the regimens in
the benchmark are **not prescribing advice**.

## Related work

ACES (Amrhein et al., WMT 2022) for contrastive challenge sets. Mehandru et al. (2023) on MT metrics
missing harmful errors in clinical translation. Saadany et al. (2024) on critical errors in MT evaluation.
TICO-19 (Anastasopoulos et al., 2020). The ISMP List of Confused Drug Names.

## Author

**Momen Ibrahim Fawzy**

Code: MIT. Data: CC BY 4.0 (TICO-19 is CC0).
