# Method

## 1. The problem

Medication instructions are often written in English by prescribers and pharmacists and then
translated for Arabic-speaking patients. A translation can be fluent and close to a reference,
and still tell the patient to take 200 mg instead of 20 mg, to put drops in the *right* eye instead
of the left, or to drink alcohol while on the medicine.

MT systems are ranked and selected with automatic metrics. If a metric scores a dangerous one-word
error above a harmless rewording, it cannot be trusted as a safety gate. AMANA measures this
directly for English → Arabic, and offers two checkers built for the job.

## 2. AMANA-Bench: a contrastive challenge set

**Structured generation.** Each instruction is a structured record: drug, dosage form, count,
strength and unit, route or site, frequency, timing, duration, as-needed flag, warning. It is
rendered to English and to Arabic. The Arabic renderer follows Modern Standard Arabic
number-noun agreement:

| count | form | example |
|---|---|---|
| 1 | noun + واحد | قرصاً واحداً |
| 2 | dual | قرصين |
| 3–10 | digit + plural | 3 أقراص |
| 11+ | digit + singular accusative | 14 يوماً |

**Critical errors (11 types)** are edits to one slot, applied to the Arabic side only:

| type | example | why it matters |
|---|---|---|
| dose | 2 → 4 tablets, 5 → 50 ml | over- or under-dose |
| strength | 20 → 200 mg (×10 / ÷10) | decimal-point slips, a classic medication error |
| unit | mg → g, mcg → mg, units → ml, ml → tablespoons | 1000× errors; insulin units read as ml |
| frequency | daily → weekly, once → four times | methotrexate daily vs weekly is fatal |
| duration | days → weeks, 7 → 70 days | |
| timing | before ↔ after meals, empty stomach ↔ with food | absorption, hypoglycaemia |
| route | eye ↔ ear, under the skin → into a vein, oral → rectal | |
| laterality | left ↔ right eye or ear | wrong-site treatment |
| negation | "do not drink alcohol" → "drink alcohol"; main instruction negated | meaning reversed |
| omission | a protective warning dropped | silent loss of safety information |
| drug | look-alike/sound-alike swap (hydralazine ↔ hydroxyzine, ISMP) or same-form swap | wrong medicine |

**Benign variants** change only style: Arabic-Indic digits, unit spellings (ملغ / مجم / ملغم),
"يومياً" ↔ "في اليوم", verb synonyms (تناول / خذ), food words, word order (duration first),
hamza-free spelling and definite drug names.

**Two contrastive settings.**

- *Minimal*, as in ACES: a safe paraphrase against the reference with a one-word error. This is
  the hardest case for surface-overlap metrics, because the error is closer to the reference than
  the paraphrase is.
- *Matched*: a safe paraphrase against the same paraphrase with the error. Surface distance from
  the reference is equal, so the comparison isolates whether the metric sees the clinical change.

**Splits are by drug.** Look-alike/sound-alike partners stay together, and 13 drugs are held out
for test. The metric audit and every reported number use the test drugs only.

## 3. Metrics audited

- BLEU, chrF, chrF++, TER (sacrebleu)
- BERTScore (mBERT)
- LaBSE cosine to the reference, and LaBSE source-translation cosine (reference-free)
- COMET-22 (Unbabel/wmt22-comet-da)
- CometKiwi-22 (Unbabel/wmt22-cometkiwi-da, reference-free; a gated model under CC BY-NC-SA 4.0, used for scoring only)
- MetricX-24 (google/metricx-24-hybrid-large-v2p6 and -xl-v2p6), with and without the reference, scored by a
  re-implementation of its reference code. The XL checkpoint was scored on two Kaggle T4 GPUs with newer libraries;
  xCOMET-XL and the XXL checkpoint were not run

Reported per metric:

- contrastive accuracy (ties count as 0.5)
- AUROC for separating safe from critical translations
- the ratio of the median score drop for a critical edit to the drop for a benign edit

## 4. AMANA-Guard (explainable, rule-based, reference-free)

The guard extracts the same canonical slots from the English source and the Arabic translation and
reports each disagreement with its type, for example `[strength] 20 mg rendered as 200 mg`. It
reads free-form Arabic, not only the template, including:

- number words (ثلاث مرات, خمسة عشر), duals (حبتين, مرتين) and Arabic-Indic digits with ٫ ٬
- Egyptian and Gulf unit spellings (مجم, ملجم), and hamza-free spelling
- drug names with clitics (بالباراسيتامول), transliteration variants (غليبزيد / جليبيزيد, matched on a
  consonant skeleton) and drug names or units left in Latin script by the MT system. A variant
  that is equally close to two drugs (omeprazole / esomeprazole) is not credited to either.
- imperative prohibitions (لا ت…) versus descriptive negation (لم / لن / ليس)

Outside medication instructions it falls back to two domain-general checks: number preservation
and negation polarity.

## 4b. AMANA-Guard 2 (relation-aware rules)

`amana/guard.py` stays unchanged as the baseline. `amana/guard2.py` parses each side into a document of
per-drug instructions (quantities, frequency, timing, route, duration, condition, limit) and warnings
stored as (topic, polarity) over a closed list of 11 topics, and compares them. It reports two levels:

- **flag**: a detected mismatch, such as `[strength] warfarin: 3 mg rendered as 75 mg`
- **needs review**: the checker could not establish the meaning, such as no recognized dosing verb

A translation passes only with neither. Text that is not a medication instruction falls back to the
baseline's number and negation checks, plus known drug names and brands in capitals. A sentence there with no number
and no negation counts as not checked; before 0.5 it was counted as covered.

**Evaluation tracks.** (1) Held-out drugs on AMANA-Bench, an upper bound. (2) A linguistic hold-out,
`data/hard_cases_holdout.jsonl`: 20 errors and 20 safe paraphrases, frozen by SHA-256 before implementation
and checked by `scripts/10_guard2_eval.py --holdout`. It was written by the same author and is not blind.
It was scored on the first code and again after three later changes (two control-character bugs, one fuzzy matching change from development data). Both runs are reported, and it is now spent. (3) Real text
with controlled errors, AMANA-TICO, which is development data for the guard.

## 4c. What Guard 2 does when it cannot verify

A probe outside the benchmark showed that "no mismatch" often meant "no rule". The checker now adds:

- **Open drug names.** It looks for names from Wikidata, and for words placed or spelled like drug names, in the source.
  Each must be found in the translation in Latin script, by exact label, or by transliteration. A different known
  drug is a flag. Anything else is a review item.
- **Coverage.** A source sentence with nothing to compare is a review item, and each report gives the share checked.
- **Sequences.** Quantities on either side of "then" or "ثم" are compared in order.
- **Whole-text checks (0.5).** These cover what the per-drug slots miss:
    - percentages;
    - a per-kilogram or per-square-metre basis;
    - the time unit of an interval or a limit (hours against minutes);
    - before or after a named meal;
    - night against morning;
    - doses tied to morning and evening;
    - the liquid to take it with;
    - two kinds of added instruction (change the dose, stop other medicines);
    - a translation with more sentences than the source.

  Preprocessing also reads:
    - ".5", `ug` and `µg`;
    - "one and a half";
    - Arabic compound and hundreds number words (خمسة وعشرين, خمسمائة);
    - counts per week or per N weeks.

  All of these came from an outside reviewer's 41 probes (`data/review_probes.jsonl`). Those probes were scored once
  before any change (`results/review_probes_first_run.json`).

The name comparison reduces both sides to shared consonant classes. `scripts/15_drug_names_eval.py` chose the threshold
on 4 in 5 Wikidata names. A false match is a silent pass and a missed match only asks for review, so it takes the
strict end of the sweep (0.9). The sealed fifth was scored once afterwards.

## 5. AMANA-QE (learned, reference-free)

A LaBSE cross-encoder fine-tuned to classify (source, translation) pairs as safe or critical.

- **Training data:** train-split drugs only. Safe pairs are the reference, the paraphrase,
  Arabic-Indic and hamza-free variants. Critical pairs are minimal or matched errors.
- **Evaluation:**
  - held-out drugs
  - AMANA-TICO, real professional translations of COVID-19 and medical text with injected number
    and negation errors. This text is out of domain and never seen in training.
  - unseen error types (unit, laterality, negation) and unseen dosage forms (eye drops, inhalers), left
    out of training with `--exclude-types` and `--exclude-forms`, five seeds each (13–17)

The score is used for ranking and is not a calibrated probability. Expected calibration error is reported
for each track.

## 6. Real MT audit and safety reranking

300 generated AMANA-Bench instructions for test drugs (not real prescriptions) are translated with NLLB-200 (600M distilled), M2M100-418M, OPUS-MT
en-ar and one open LLM, Qwen2.5-7B-Instruct. The LLM runs with 4-bit (nf4) weights to fit an 8 GB GPU, one fixed
prompt (in `06_mt_audit.py`) and the same beam search, and its output is cut at the first line break, because a few beams
run on into an invented next prompt. `06_mt_audit.py` also lists NLLB-200-1.3B, which can be added with
`translate --systems NLLB-200-1.3B`, though it was not run. For each, the system keeps its 5-best beam
candidates. Five strategies are compared:

| strategy | how the output is chosen |
|---|---|
| 1-best | the system's normal output |
| Guard rerank | first candidate with no flags |
| QE rerank | highest AMANA-QE score |
| Guard+QE rerank | among baseline-clean candidates, the highest QE |
| Guard2+QE rerank | among candidates Guard 2 passes, the highest QE; otherwise the 1-best |

Each strategy is reported with both checkers' rates, AMANA-QE's unsafe share, chrF and COMET-22 against the
reference. QE rerank judged by Guard 2 and Guard 2 + QE judged by AMANA-QE are the cross-checks, because a
checker judging its own choice is partly circular.

**Manual review.** A sample of flagged 1-best outputs was read by hand before the numbers were
reported. The
review found guard false alarms (Latin-script drug names, transliteration variants, the
nominative dual قطرتان, "على بطن فارغ", and, from the author's native-speaker check of 12 flags (4 were false alarms), "لوح" for
tablet and "السمبرازول" for esomeprazole); these were fixed and added as regression tests in
`tests/test_guard.py`. It also confirmed real errors: safety warnings silently dropped
("Do not crush or chew the tablets"), "25 mcg" rendered as "25 مليغراماً" (a 1000× overdose),
and "Instill" rendered as "لا تزال" ("still"). The author also judged 20 sampled benchmark
instructions, with both Arabic versions, as correct. The author is not a clinician. One later reviewer judged the
60-pair packet, which is not truly blind (see the README). Because the guard was refined on these outputs, the
flag rates are not a held-out result.

## 6b. Real inputs

`scripts/17_peach.py` uses the patient information leaflets in PEACH (Al-Sabbagh, 2024). It keeps sentences of 5 to 35 words
that give an instruction or a quantity, each English sentence once. Leaflets are split by hash, and a sentence in both halves
goes to sealed. It reports the checker's output on the published Arabic, then injects one number, negation or unit error into
each sentence and counts only sentences that passed when clean, so text already flagged cannot inflate the catch rate.
Intervals resample leaflets. The corpus is not redistributed.

`scripts/16_real_inputs.py` takes sentences from the patient sections of US FDA labels through openFDA, keeping those of
5–35 words that give an instruction or a quantity, with at most 3 per label. It samples 300 sentences for each of a
development and a sealed set, split by label hash. Three MT systems translate them. Reading development output led to
fixes for false alarms, and the sealed set was then scored.

There are no references and no error labels, so the output is the checker's status and coverage, and one measure that uses
no parser: whether each digit sequence in the source appears in the translation.

## 7. Limitations

- The benchmark is template-based and real prescriptions are more varied. The MT audit also uses
  generated inputs, and AMANA-TICO has real text but constructed number and negation errors only.
  No test here uses naturally occurring clinical errors with independent annotation.
- The guard was developed against the template vocabulary. Its numbers on AMANA-Bench are an upper
  bound. False alarms on TICO-19 references also led to guard fixes, so TICO-19 is development data
  for the guard, not an independent test. (It is unseen for AMANA-QE.)
- The baseline guard compares flat slots. `scripts/09_hard_cases.py` documents the resulting failures.
  Guard 2 keeps those links but is still rule-based, with a closed vocabulary of drugs and warning topics.
  It passes the 20 development hard cases because it was built on them.
- The linguistic hold-out has the same author as the checker. The next test needs another author.
- In the MT audit the guard is both reranker and judge for "Guard rerank". Rates for that strategy
  are therefore paired with AMANA-QE as a second judge (whose threshold is uncalibrated on real
  text), and with a manually reviewed sample.
- AMANA-QE is trained on template text only. It ranks real TICO-19 translations reasonably
  (AUROC 0.80) but its 0.5 threshold does not transfer: most safe real sentences fall below it.
  It is used for ranking candidates, not as a pass/fail gate. On error types left out of training, its
  AUROC is 0.66–0.73 over five seeds (mean 0.69), so it has learned the generator to a large degree. Recall at 0.5
  ranges from 7% to 36% across those seeds, so any single-seed threshold result is unreliable.
- Bootstrap intervals resample pairs, and pairs from similar instructions are not fully independent.
- There are no severity tiers, and no clinician has reviewed any result.
- PEACH leaflets were published, not adjudicated, and may differ between language versions by design. The flag reading is
  non-clinical, and injected errors cover 3 of the 11 error types.
- FDA labels are US English for US patients and include storage and ingredient text, so the sample is broader than dosing
  instructions. The MT outputs in that track have no adjudicated errors.
- Arabic only, in Modern Standard Arabic. Dialectal patient-facing Arabic is future work.
- This is a research tool. It is not a medical device and not prescribing guidance.
