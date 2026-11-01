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
- CometKiwi-22 (reference-free) could not be loaded and is not in the reported run

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

## 5. AMANA-QE (learned, reference-free)

A LaBSE cross-encoder fine-tuned to classify (source, translation) pairs as safe or critical.

- **Training data:** train-split drugs only. Safe pairs are the reference, the paraphrase,
  Arabic-Indic and hamza-free variants. Critical pairs are minimal or matched errors.
- **Evaluation:**
  - held-out drugs
  - AMANA-TICO, real professional translations of COVID-19 and medical text with injected number
    and negation errors. This text is out of domain and never seen in training.

## 6. Real MT audit and safety reranking

300 generated AMANA-Bench instructions for test drugs (not real prescriptions) are translated with NLLB-200 (600M distilled), M2M100-418M and OPUS-MT
en-ar. `06_mt_audit.py` also lists NLLB-200-1.3B, which can be added with
`translate --systems NLLB-200-1.3B`. For each, the system keeps its 5-best beam candidates. Four
strategies are compared:

| strategy | how the output is chosen |
|---|---|
| 1-best | the system's normal output |
| Guard rerank | first candidate with no flags |
| QE rerank | highest AMANA-QE score |
| Guard+QE rerank | among guard-clean candidates, the highest QE |

Each strategy is reported with its flagged-critical rate, chrF and COMET-22 against the reference.

**Manual review.** A sample of flagged 1-best outputs was read by hand before the numbers were
reported. The
review found guard false alarms (Latin-script drug names, transliteration variants, the
nominative dual قطرتان, "على بطن فارغ", and, from the author's native-speaker check of 12 flags (4 were false alarms), "لوح" for
tablet and "السمبرازول" for esomeprazole); these were fixed and added as regression tests in
`tests/test_amana.py`. It also confirmed real errors: safety warnings silently dropped
("Do not crush or chew the tablets"), "25 mcg" rendered as "25 مليغراماً" (a 1000× overdose),
and "Instill" rendered as "لا تزال" ("still"). The author also judged 20 sampled benchmark
instructions, with both Arabic versions, as correct. The author is not a clinician, and no
independent or blinded review has been done. Because the guard was refined on these outputs, the
flag rates are not a held-out result.

## 7. Limitations

- The benchmark is template-based and real prescriptions are more varied. The MT audit also uses
  generated inputs, and AMANA-TICO has real text but constructed number and negation errors only.
  No test here uses naturally occurring clinical errors with independent annotation.
- The guard was developed against the template vocabulary. Its numbers on AMANA-Bench are an upper
  bound. False alarms on TICO-19 references also led to guard fixes, so TICO-19 is development data
  for the guard, not an independent test. (It is unseen for AMANA-QE.)
- The guard compares flat slots. It loses the links between a drug and its dose, timing or condition,
  and does not read warning content. `scripts/09_hard_cases.py` documents these failures.
- In the MT audit the guard is both reranker and judge for "Guard rerank". Rates for that strategy
  are therefore paired with AMANA-QE as a second judge (whose threshold is uncalibrated on real
  text), and with a manually reviewed sample.
- AMANA-QE is trained on template text only. It ranks real TICO-19 translations reasonably
  (AUROC 0.80) but its 0.5 threshold does not transfer: most safe real sentences fall below it.
  It is used for ranking candidates, not as a pass/fail gate. Its held-out split separates drugs,
  not sentence templates or error generators.
- Arabic only, in Modern Standard Arabic. Dialectal patient-facing Arabic is future work.
- This is a research tool. It is not a medical device and not prescribing guidance.
