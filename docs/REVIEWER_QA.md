# Questions a reviewer should ask

Short answers to the hard questions, with the evidence or the gap for each.

## Is the benchmark testing clinical safety, or its own generator?

Mostly its own generator. AMANA-Bench is a controlled instrument. It can show that a metric prefers a
one-word error, because the labels are exact by construction. It cannot show that a checker is safe, and
its 100% is reported as an upper bound. The evidence for generator learning is direct: AMANA-QE falls
from AUROC 0.999 to 0.66–0.73 (five seeds) on error types left out of training. The labels also leak through the Arabic alone: a
classifier that never sees the English reaches AUROC 0.711 on held-out drugs, and 0.94 for unit, route and negation
errors, because correct instructions never use some words that errors use.

## Why trust the metric finding, if the test is built to make metrics fail?

The reworded setting is deliberately adversarial, so 23–47% is not a natural error rate. It shows a
weighting problem under stated conditions. That is why the same-wording setting sits beside it: there,
the same metrics reach 93–99%. Both come with bootstrap intervals, ties and margins in
`results/metric_stats.json`. COMET-22's 46.7% (44.3–49.0) is below chance by a small margin, so the fair
reading is "about a coin flip", not "COMET-22 prefers errors".

Much of the raw result is a preprocessing artefact. With standard Arabic normalisation, BLEU's reworded score rises
from 23.3% to 40.1%, chrF's from 35.7% to 51.6%, BERTScore's from 30.7% to 45.4% and COMET-22's from 46.7% to 59.2%. On
rewordings that change only spelling, every surface metric reaches 100%. On genuine word changes they still prefer the
error 58–70% of the time. The README intervals now resample the 13 drugs, not the pairs, which widens BLEU's from
21.3–25.3% to 19.9–27.2%. COMET-22's raw 46.7% (42.9–50.5) is then not distinguishable from chance. The very small
p-values in `metric_stats.json` assume independent pairs and should be read as "< 0.001".

## Does AMANA-Guard 2 understand instructions?

No. It parses them with rules into per-drug records, which fixes the swapped-dose failure of a flat slot
comparison. It has no general semantics. It understands 11 warning topics, and it returns `needs_review` when it
cannot establish meaning, including for any sentence that gave it nothing to compare.

## Does `ok` mean the translation is right?

No. It means no mismatch was found in what the rules compared. Twice an earlier version said `ok` for things it had
not checked. First, it had no rule for them. Then, in 0.4, sentences with no dosing instruction went to a fallback
comparison but were still counted as fully covered. An outside review found the second problem, along with 20 dangerous
translations that passed. Both are fixed. A wrong word that no rule looks at can still pass within a sentence that has
other checkable content. For a sentence with no dosing instruction, a pass means only that its numbers, negations and
drug names matched.

## How does it do on real text?

On sealed FDA label sentences, each system passes 30–33%, and more than half go to review. Only 51% of sentences are
fully covered by the rules, and 31% have no dosing instruction, so only a narrower check runs on them. No error labels
exist for this text, so these are not accuracy figures. They show how much of the work still falls to a person.

On real human leaflets (PEACH, sealed half), it passes 47.5% and flags 14.4%. In a sample of 60 flags, read before the 0.5
fixes, 10 pointed at a real difference, so most flags on sound text are false alarms. No one has read the passed rows,
so misses are unknown. It stops passing 95–100% of injected number, negation and unit errors, about the same as the
first checker. That reading was not clinical and not independently checked.

## Were the outside probes a fair test?

They were the first cases not written by the author: 30 dangerous and 11 faithful translations. They were frozen when
received and scored once on unchanged code. Guard 2 caught 10 of 30 and accepted 8 of 11. The reviewer had read the
code, so the probes target its weak points: a fair adversarial test, but not a blind one. All 30 are now caught. That
was done by fixing exactly what they exposed, so they are regression tests now, not evidence.

## How independent is the hold-out?

Not very. The cases were frozen by hash before implementation and not tuned on, but the same author wrote
them and knew what Guard 2 was meant to fix. Both runs are reported: 19/20 errors on the first run, 20/20
after three code changes made after seeing that score, and 14/20 safe in both. Later fixes from real leaflets moved safe
cases to 15/20, which is not new evidence. The set is now spent. The next test has to be written by someone
else, without seeing the code.

## Why not lead with hold-out recall?

n = 20 gives a 95% interval of 83.9–100% for 20/20, under the authorship caveat above. A 30% false alarm
rate on the safe cases is at least as important, because a checker that rejects good paraphrases will be
ignored or switched off.

## Is reranking evidence of improved safety?

Only weakly. When the guard both chooses and judges, the drop is partly circular. Cross-judging is less
flattering: QE-only reranking moves Guard 2's rate from 24.7% to 27.0% on OPUS-MT, and Guard 2 + QE moves
QE's unsafe share only from 21.7% to 20.3% on M2M100. Reranking also cannot fix systematic errors. All 5
candidates fail for 74 of 300 OPUS-MT and 126 of 300 M2M100 instructions. One human rater judged 60 changed picks:
better in 5, worse in 4, the same in 51 (sign test p = 1.0). That is no evidence either way, and the packet could be
unblinded from public results.

## Isn't this one fragile model checking another?

The checker that decides is not a model. AMANA-Guard 2 is deterministic rules: the same input always gives the same
result, and every flag names the slot that differs. It cannot hallucinate, but it can miss what its rules do not cover,
which is why it reports `needs_review` instead of `ok` when it has nothing to compare. AMANA-QE is a learned model and is
used only to rank candidates, never to pass a translation. Neither one makes MT safe to use unreviewed. Their job is to
send the risky part of the output to a person.

## How can a sealed set be sealed if it is public?

"Sealed" here means the set was split off by hash before the code was written against it, then scored once with the
code frozen. The checker has no trained weights, so publishing the set cannot leak into a model. It can leak into the
author's next rule change, which is why each first score is kept in a `*_first_run.json` file and later scores are
labelled as no longer clean (see `results/README.md`). Once published, a sealed set is spent, and the next one has to
be new. AMANA-QE is trained only on training drugs; its held-out drugs and TICO-19 never enter training.

Until 0.5, every routine `report` run re-scored the sealed sets. They now run only as a separate `sealed` stage, and
each scoring is appended to `results/sealed_runs.jsonl` with the code revision. That makes re-scoring visible, but it
cannot prevent it. The history also cannot prove the order: the hold-out first appears in the same commit as Guard 2.
A future sealed set should have its hash published externally, with a timestamp, before it is scored.

## Why are there two checkers, `guard.py` and `guard2.py`?

`guard.py` is the first design, a flat comparison of slots, kept as the baseline that Guard 2 is measured against.
Guard 2 replaced its design, not just its rules: it parses each instruction into per-drug records and can abstain.
Deleting the baseline would remove the comparison. Each has its own tests (`test_guard.py`, `test_guard2.py`).

## Does this cover what patients and clinicians actually say?

No. It covers written medication instructions: generated ones, FDA patient labels and published leaflets. It does not
cover conversation, interpreting, colloquial or dialect speech, or handwritten prescriptions. Those need different data
and probably a different method.

## Are the 43.0% and 24.7% real error rates?

No. They are checker output on generated inputs. Some flags will be false alarms, some real errors are
invisible to the checker, and no independent person has judged them. The exact pattern counts, such as
"mcg" written as "mg" in 18 of 18 cases, are firmer because each is a specific, inspectable string
pattern. They are still not independent: all 18 are levothyroxine, and the 46 "rash" rows share one sentence.

## What is the AMANA-QE score?

A ranking score. Its output is not a calibrated probability: expected calibration error is 0.009 on held-out
drugs but 0.46 on TICO-19, and false alarms at 0.5 vary between 0% and 100% across tracks. In the unseen-form
track, seed choice alone moved them from 70% to 100% (four of five seeds give 100%). Across five seeds, AUROC is
stable (0.85–0.86 on unseen forms, 0.66–0.73 on unseen types), but recall at 0.5 on unseen types ranges from 7% to 36%.

## Is there clinical validation?

None. The author is a native Arabic speaker and not a clinician. One reviewer, whose clinical qualification was not
recorded, judged the 60 pairs in `adjudication/packet.html`. The 1-best of each pair can be found in public results,
so the packet is not blind for anyone who reads the repository. "Critical" in the benchmark is also the author's
construct: any one-slot change counts, with no severity. Severity tiers need clinical input.

## What evidence would change the conclusions?

1. Two pharmacists rating a random sample of benchmark rows and the FDA MT outputs on an agreed severity scale, with
   their agreement reported, from a packet built and sent outside this repository.
2. A fresh test set written by someone who has not seen the rules, with its hash published before scoring.
3. Real translated medication instructions with naturally occurring, independently annotated errors. PEACH now supplies
   real text, but nobody independent has annotated its differences, and no one has read the rows that passed.
4. Current MT systems, including commercial ones, and larger metrics such as xCOMET-XL and MetricX-24 XXL. MetricX-24
   large and XL are included: the larger model is better, but neither changes the conclusions.

## Why no MLflow, DVC, a config framework or a web service?

They do not address any current weakness. Results are versioned files with hashes in
`results/MANIFEST.json`, and one script runs every stage. Experiment tracking would add little to a handful
of runs, and a web service would invite clinical use that the evidence does not support. This can change if
the project grows past that point.
