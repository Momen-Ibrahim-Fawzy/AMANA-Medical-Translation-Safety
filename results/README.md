# Results

Every number in the main README comes from a file here. Each file is written by one script, and its SHA-256 is in
`MANIFEST.json`, so a changed result shows up as a changed hash. The files are versioned on purpose: they are the
evidence, and many of them can only be regenerated with a GPU. Together they are about 11 MB.

## Files kept from a first run

A sealed set is scored once, after the code is frozen. If the code changes later, the script writes a new result, but
that result is no longer a clean estimate, because whoever changed the code had seen the sealed scores. The
`*_first_run.json` files keep the one clean score so it cannot be overwritten. Nothing reads them back.

| file | first scored on | what changed after |
|---|---|---|
| `guard2_holdout_first_run.json` | the hand-written hold-out | three fixes prompted by other data, then the leaflet fixes |
| `fda_sealed_first_run.json` | the sealed half of the FDA label sentences | an "even if" fix, the leaflet fixes, the 0.5 fixes |
| `peach_sealed_first_run.json` | the sealed half of the PEACH leaflets | the 0.5 fixes |
| `review_probes_first_run.json` | an outside reviewer's 41 probes, scored on the code as received | the 0.5 fixes they led to |

## Every file

| file | written by | what it holds |
|---|---|---|
| `audit_set.jsonl` | `03_metric_audit.py` | the contrastive pairs the metrics are scored on |
| `metric_audit.json` | `03_metric_audit.py` | how often each metric prefers the safe translation |
| `metric_stats.json` | `11_metric_stats.py` | bootstrap intervals for the metric audit |
| `guard_summary.json` | `04_guard_eval.py` | the baseline checker on AMANA-Bench and AMANA-TICO |
| `amana_qe_summary.json`, `amana_qe_predictions.json` | `05_train_qe.py` | AMANA-QE scores on held-out drugs and TICO-19 |
| `qe_generalization_{forms,types}_seed{13..17}.json` | `05_train_qe.py --tag` | AMANA-QE trained without some dosage forms or error types |
| `mt_outputs.jsonl`, `mt_picks.jsonl`, `mt_flagged_examples.jsonl`, `mt_audit.json` | `06_mt_audit.py` | three MT systems' 5-best outputs, which candidate each strategy picked, and the audit |
| `mt_error_patterns.json` | `08_error_patterns.py` | exact counts of recurring MT errors, with the number of distinct drugs and warning sentences behind each |
| `hard_cases.json` | `09_hard_cases.py` | the baseline checker on the hand-written cases |
| `guard2_dev.json`, `guard2_holdout.json` | `10_guard2_eval.py` | Guard 2 on development data and the hold-out |
| `adjudication.json` | `12_adjudication_packet.py score` | totals from one rater's review of reranking; the packet can be unblinded from `mt_outputs.jsonl` |
| `review_probes.json` | `10_guard2_eval.py --probes` | both checkers on the outside probes, current code |
| `benchmark_controls.json` | `18_benchmark_controls.py` | a source-blind classifier, and surface metrics on normalised Arabic |
| `drug_names_eval.json` | `15_drug_names_eval.py` | open drug name matching, development and sealed |
| `fda_mt_outputs.jsonl`, `fda_dev.json`, `fda_sealed.json` | `16_real_inputs.py` | MT of FDA patient label sentences and Guard 2 on them |
| `peach_dev.json`, `peach_sealed.json` | `17_peach.py` | Guard 2 on human-translated leaflets; counts and row references only, no corpus text |
| `peach_sealed_flags_read.json` | by hand | a sample of 60 sealed flags, read after the sealed run |
| `sealed_runs.jsonl` | `scripts/sealed_log.py`, called by every sealed scoring | one line per scoring of a sealed set: time, code revision, result hash; starts at 0.5 |
| `MANIFEST.json` | `13_manifest.py` | hashes of data and results, code revision, package versions |
