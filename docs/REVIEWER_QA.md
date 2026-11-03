# Questions a reviewer should ask

Short answers to the hard questions, with the evidence or the gap for each.

## Is the benchmark testing clinical safety, or its own generator?

Mostly its own generator. AMANA-Bench is a controlled instrument. It can show that a metric prefers a
one-word error, because the labels are exact by construction. It cannot show that a checker is safe, and
its 100% is reported as an upper bound. The evidence for generator learning is direct: AMANA-QE falls
from AUROC 0.999 to 0.67–0.71 on error types left out of training.

## Why trust the metric finding, if the test is built to make metrics fail?

The reworded setting is deliberately adversarial, so 23–47% is not a natural error rate. It shows a
weighting problem under stated conditions. That is why the same-wording setting sits beside it: there,
the same metrics reach 93–99%. Both come with bootstrap intervals, ties and margins in
`results/metric_stats.json`. COMET-22's 46.7% (44.3–49.0) is below chance by a small margin, so the fair
reading is "about a coin flip", not "COMET-22 prefers errors".

## Does AMANA-Guard 2 understand instructions?

No. It parses them with rules into per-drug records, which fixes the swapped-dose failure of a flat slot
comparison. It has no general semantics. It understands 10 warning topics, and it returns `needs_review` when it
cannot establish meaning, including for any sentence that gave it nothing to compare.

## Does `ok` mean the translation is right?

No. It means every source sentence had something to compare and no mismatch was found. An earlier version also said
`ok` when it simply had no rule, which was the most serious flaw found in this project. That is fixed, but a wrong
word that no rule looks at can still pass within a sentence that has other checkable content.

## How does it do on real text?

On sealed FDA label sentences, each system passes 35–40%, and about half go to review. Only 56% of sentences are fully
covered by the rules. No error labels exist for this text, so these are not accuracy figures. They show how much of the
work still falls to a person.

On real human leaflets (PEACH, sealed half), it passes 57.4% and flags 14.3%. In a sample of 60 flags, 10 pointed at a real
difference, so most flags on sound text are false alarms. It stops passing 95–100% of injected number, negation and unit
errors, about the same as the first checker. That reading was not clinical and not independently checked.

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
flattering: QE-only reranking moves Guard 2's rate from 25.0% to 27.3% on OPUS-MT, and Guard 2 + QE moves
QE's unsafe share only from 21.7% to 20.3% on M2M100. Reranking also cannot fix systematic errors. All 5
candidates fail for 74 of 300 OPUS-MT and 126 of 300 M2M100 instructions.

## Are the 43.0% and 25.0% real error rates?

No. They are checker output on generated inputs. Some flags will be false alarms, some real errors are
invisible to the checker, and no independent person has judged them. The exact pattern counts, such as
"mcg" written as "mg" in 18 of 18 cases, are firmer because each is a specific, inspectable string
pattern.

## What is the AMANA-QE score?

A ranking score. Its output is not a calibrated probability: expected calibration error is 0.009 on held-out
drugs but 0.46 on TICO-19, and false alarms at 0.5 vary between 0% and 100% across tracks. In the unseen-form
track, seed choice alone moved them from 70% to 100%. Two seeds per track also cannot measure variance well.

## Is there clinical validation?

None. The author is a native Arabic speaker and not a clinician. `adjudication/packet.html` is ready for a
pharmacist or clinician to judge 60 blinded pairs, and no one has completed it yet. Severity tiers also need
clinical input.

## What evidence would change the conclusions?

1. A pharmacist's adjudication of the blinded packet.
2. A fresh test set written by another person.
3. Real translated medication instructions with naturally occurring, independently annotated errors. PEACH now supplies
   real text, but nobody independent has annotated its differences. A pharmacist judging the flagged leaflet rows would help.

## Why no MLflow, DVC, a config framework or a web service?

They do not address any current weakness. Results are versioned files with hashes in
`results/MANIFEST.json`, and one script runs every stage. Experiment tracking would add little to a handful
of runs, and a web service would invite clinical use that the evidence does not support. This can change if
the project grows past that point.
