# Can Laya replace an LLM judge for flagging failed agent runs?

**Short answer: not on this data.** We compared four ways to flag customer-service agent runs
that failed: a TF-IDF + logistic regression baseline, Laya zero-shot, Laya fine-tuned with its
official trainer, and Claude Haiku 4.5 as an LLM judge. Every model was evaluated on tasks it
never saw. The TF-IDF baseline on the last 1,024 tokens of each run was the best model.
Fine-tuning moved Laya from chance to useful, but across 3 seeds it averaged below TF-IDF, and
only one seed reached TF-IDF's level. Haiku as a judge beat zero-shot Laya but lost clearly to
TF-IDF.

![AUROC with 95% CI per model](results/auroc_by_model.png)

| on the same 497 test runs | AUROC (95% CI) | recall at 10% review | ECE | latency p50 | cost per 1,000 runs |
|---|---|---|---|---|---|
| **TF-IDF + LogReg, last 1,024 tokens** | **0.840** (0.738-0.925) | **0.45** | 0.055 | 1.6 ms (laptop CPU) | ~2 s of CPU |
| Laya fine-tuned, mean of 3 seeds | 0.798 (0.707-0.886) | 0.39 | 0.056 | ~92-103 ms (T4) | ~100 s of T4 GPU |
| Claude Haiku 4.5 judge | 0.702 (0.577-0.815) | 0.24 | 0.134 | 1,275 ms (API) | $4.07 |
| Laya zero-shot | 0.529 (0.461-0.630) | 0.09 | 0.352 | 94 ms (T4) | ~94 s of T4 GPU |

Paired bootstraps resample whole scenarios, so both models are scored on the same runs:
- TF-IDF beats Haiku by **+0.139 AUROC** (95% CI +0.047 to +0.222).
- Against the fine-tuned Laya mean, TF-IDF leads by +0.042 (CI −0.013 to +0.096). That's a tie
  or better on these runs. On the full test split, 2 of the 3 seeds are significantly worse than
  TF-IDF.

## Question and target

**Question:** can Laya replace an LLM judge for flagging failed agent runs?

**Target:** the run *failed* (reward 0) versus *succeeded* (reward 1). The reward comes from
tau2-bench's own check of the final database state, not from any model reading the transcript,
so the labels don't depend on the agent's wording. This was decided before any modelling.

"False success" (failed runs whose final message claims success) is a narrower target, studied in
arXiv 2606.09863. It was out of scope here; we only counted those runs (see Data).

## Data

**Source:** the public [τ²-bench](https://github.com/sierra-research/tau2-bench) leaderboard
trajectories (bucket `sierra-tau-bench-public/submissions/`). The repo's `data/tau2/results/final`
release was inventoried too, but not used. `tau2/data_check.py` downloads and inventories both;
the raw data is **not** redistributed here (see License).

**Corpus:** fixed after a data inventory and before any modelling.
- Leaderboard submissions only, domains airline / retail / telecom, default settings.
- **7,503 runs** from 7 agent submissions (Claude Opus 4.5, Claude Sonnet 4.5, GPT-5.2,
  Gemini 3 Pro, Gemini 3 Flash, GLM-5, Qwen3.5-397B), with **15.5% failed**.

**Excluded, with reasons:**

| excluded | why |
|---|---|
| `banking_knowledge` | Re-graded in v1.0.1, but the trajectory files still carry the old grades; runs up to 760k tokens. |
| `gpt-5-2-none` | Its files disagree with its published scores, unexplained. |
| Gemini 3 Flash airline | Run on an older version of 22 airline tasks. |
| 81 empty runs | No messages, ended in an infrastructure error, yet labelled as successes. |

**Split: task-disjoint**, 60/10/30, stratified by domain.
- Runs are grouped by **customer scenario** (a hash of the task's user scenario), not by task ID.
  Telecom's 114 task IDs share only 15 scenarios, differing only in hidden device faults; grouping
  by ID would put the same customer conversation in both train and test.
- Result: 179 scenario groups.

| split | scenarios | runs | failed |
|---|---|---|---|
| train | 107 | 4,604 | 16.2% |
| calibration | 18 | 730 | 12.1% |
| test | 54 | 2,169 | 15.3% |

**Length:** runs have a median of 2.7k-4.0k Laya tokens by domain. 88-97% are longer than Laya's
1,024-token window.

**Input:** every model except "TF-IDF full run" sees the **last 1,024 tokens** of the plain-text
transcript (messages, tool calls, tool results). Laya's window also has to hold the question, so
it sees the last ~981 tokens. laya keeps the *start* of a long text, so the tail is cut before it
reaches laya; otherwise the end, where outcomes show, would be dropped.

**Failed runs whose closing message claims success** (regex, counted only):

| domain | share of failed runs |
|---|---|
| airline | 42% |
| retail | 40% |
| telecom | 1% |

## Step 0: harness check

Before trusting any Laya number, our wrapper (`common/laya_agent.py`) had to reproduce Laya's
published result: laya-typed-decisions on the `LocalLLaMA/typed-decisions` test split (2,000
decisions), published at 0.766.

| run | accuracy | agreement |
|---|---|---|
| CPU, fp32 | **0.7660** | every per-type and per-workflow figure matches the model card to 3 decimals |
| Kaggle T4, fp16 | **0.7665** | 1,999 of 2,000 decisions agree with the CPU run |

On 81 tau2 runs scored on both CPU and GPU, scores differ by at most 0.0013.

## The four models

1. **TF-IDF + logistic regression.**
   - Unigram + bigram TF-IDF (30k features) with L2 logistic regression, untuned, as in
     arXiv 2606.09863.
   - Trained on the train split. Two variants: full run, and last 1,024 tokens.
2. **Laya zero-shot.**
   - `convaiinnovations/laya-typed-decisions`, laya 0.3.21, one yes/no question.
   - The wording was chosen on calibration only, from 3 candidates. The winner, W3, reads
     *"The agent left the customer's problem unresolved or handled it incorrectly."*
   - All three wordings scored ≈0.70 calibration log loss, against 0.693 for always answering 50%.
   - The temperature was refitted on calibration.
3. **Laya fine-tuned.**
   - Laya's **official** RLCD trainer (`NandhaKishorM/laya@4aa6761`,
     `notebooks/laya_finetune_typed_decisions_2xT4_kaggle.ipynb`), run on Kaggle 2×T4. The loss,
     optimiser, schedules and noise are unchanged.
   - The patches are marked `# PATCH (tau2)` in `kaggle/train_ddp_laya.py`:
     - paths, epochs, learning rates and seed come from command-line arguments;
     - the temperature holdout is our calibration split instead of a random 10% of train;
     - the seed replaces the fixed 42.
   - Same input as zero-shot (W3, the same tail), with hard 0/1 labels.
   - Training sequences were checked to match laya's inference encoding exactly.
4. **Claude Haiku 4.5 as judge.**
   - `claude-haiku-4-5`, sent the full transcript, blind (no task specification or policy).
   - Asked for one structured field, `p_fail` (0-100). Normal API calls.
   - Run on a stratified 497-run test sample, by domain and outcome, keeping the 15.3% failure
     rate. Raw responses are stored; all 497 parsed, with no errors.

## Pre-registered decision rules

Each rule was fixed before the test split was scored:
- **Laya wording:** lowest calibration log loss at each wording's own refit temperature.
  Candidates were limited to 3, with the original question as a reference only.
- **Fine-tuning setting:** 3 settings (S1 = official defaults: 4 epochs, learning rates
  2.5e-5 / 1e-4; S2 = 2 epochs; S3 = 4 epochs at half the learning rates). Each was trained once
  with seed 42. The temperature was fitted on calibration by the official code, and the lowest
  calibration log loss won.
- **Seeds:** the chosen setting was retrained with seeds 43 and 44. All 3 seeds are reported,
  with mean and range. **No seed is picked**, the headline uses the mean, and the paired
  bootstrap runs per seed.
- **Test:** each final model scored the test split **once**.

**Setting selection on calibration** (730 runs, 88 failed):

| setting | log loss | AUROC | fitted temperature |
|---|---|---|---|
| **S1 (chosen)** | **0.292** | 0.841 | 2.20 |
| S2 | 0.313 | 0.793 | 1.25 |
| S3 | 0.305 | 0.832 | 2.19 |

## Results on the full test split (2,169 runs)

| model | AUROC (95% CI) | recall at 10% review | ECE | Brier | latency p50 / p95 |
|---|---|---|---|---|---|
| TF-IDF, full run | 0.815 (0.711-0.878) | 0.40 | 0.040 | 0.106 | 3.1 / 5.1 ms |
| **TF-IDF, last 1,024 tokens** | **0.840** (0.742-0.899) | **0.43** | 0.048 | 0.093 | 1.6 / 2.2 ms |
| Laya zero-shot (refit T) | 0.498 (0.440-0.561) | 0.10 | 0.353 | 0.254 | 93.5 / 95.6 ms |
| Laya fine-tuned, seed 42 | 0.751 (0.660-0.821) | 0.31 | 0.034 | 0.116 | 92 / 95 ms |
| Laya fine-tuned, seed 43 | 0.827 (0.734-0.888) | 0.43 | 0.037 | 0.097 | 103 / 108 ms |
| Laya fine-tuned, seed 44 | 0.747 (0.663-0.808) | 0.38 | 0.124 | 0.136 | 103 / 108 ms |
| **Laya fine-tuned, mean (range)** | **0.775** (0.747-0.827) | 0.37 (0.31-0.43) | 0.065 (0.034-0.124) | 0.116 | |

- The best possible recall within a 10% review budget is 0.66 (budget ÷ failure rate).
- 95% CIs come from 1,000 bootstrap resamples of whole scenarios.
- Latency: TF-IDF on a 4-core laptop CPU, Laya on a Kaggle T4, Haiku as API round trips over a
  home connection.

**Paired bootstrap, AUROC(fine-tuned) − AUROC(TF-IDF last 1,024)**, resampling scenarios:

| seed | full test | 497 sample |
|---|---|---|
| 42 | −0.089 (−0.148 to −0.035) | −0.067 (−0.138 to −0.004) |
| 43 | −0.013 (−0.057 to +0.028) | +0.017 (−0.032 to +0.071) |
| 44 | −0.093 (−0.142 to −0.043) | −0.077 (−0.141 to −0.010) |

**AUROC by domain** (497 sample):

| model | airline | retail | telecom |
|---|---|---|---|
| TF-IDF, last 1,024 tokens | 0.707 | 0.800 | 0.837 |
| Haiku judge | 0.497 | 0.699 | 0.924 |
| Laya zero-shot | 0.519 | 0.533 | 0.668 |

## Caveats

- **One benchmark, one kind of task.** Three customer-service domains from one benchmark, 7 agent
  submissions. Nothing here says how the models do on other agents, tools or domains.
- **The calibration split is small:** 18 scenarios, 88 failures. It chose S1 at 0.841 calibration
  AUROC, but the same seed-42 model scored 0.751 on test. Selection on so few scenarios is noisy.
- **Telecom has only 15 scenarios** (5 in test), so its per-domain numbers have very wide
  intervals. The pooled AUROC also partly reflects differences in failure rate between domains
  (telecom 5.6%, retail 23.7%).
- **Hard labels.** Laya's official recipe was built for soft teacher distributions; here it was
  trained on 0/1 rewards with no class reweighting, as in the original.
- **Seed 44 was unstable.** Its training loss rose to 1.01 in the last epoch, and its fitted
  temperature hit laya's 5.0 cap. Its ranking (0.747) matches seed 42, but its calibration is
  worse (ECE 0.124). It stays in the mean, as decided before the runs.
- **Truncation.** Laya and the TF-IDF tail variant see only the last ~1k tokens, while 88-97% of
  runs are longer. Windowing (`predict_long`) was not tried.
- **Haiku setup.** A blind judge with one prompt, no wording search and no task specification.
  Scored on the 497-run sample only, not the full test split.
- **Untuned baseline.** TF-IDF hyperparameters were not tuned; logistic regression uses default
  regularisation.
- **Different hardware per model.** Latency and cost compare different setups: laptop CPU,
  Kaggle T4, and an API over a home connection.

## Reproduce

Run everything from the repo root, using the repo's `.venv`. Laya steps on CPU should run under
`nice -n 19`: at ~8 s per call, they take hours.

```
python tau2/data_check.py download     # τ²-bench trajectories (~5.9 GB; resumable; pinned file list in data/meta/)
python tau2/data_check.py normalize    # -> data/runs.jsonl
python tau2/data_check.py report       # inventory -> data_report.json
python tau2/experiment.py corpus       # -> data/corpus.jsonl (the corpus decisions above)
python tau2/experiment.py split        # task-disjoint split (committed as data/split.json)
python tau2/experiment.py tfidf
python tau2/kaggle/build_package.py    # Kaggle dataset; then: kaggle datasets version -p tau2/kaggle/upload -m ...
#   Laya zero-shot on Kaggle (kaggle/laya_tau2.ipynb) -> python tau2/experiment.py import-kaggle
#   Laya fine-tuning on Kaggle: build_package.py kernel --full --phase select | --phase seeds --setting S1
python tau2/haiku_judge.py estimate && python tau2/haiku_judge.py run   # needs an Anthropic API key in .env
python tau2/experiment.py report && python tau2/experiment.py report-matched && python tau2/experiment.py report-finetune
python tau2/make_chart.py
```

Step 0 is `python laya_check/typed_decisions.py`. The format check, which rendered runs in
typed-decisions' structured shape (AUROC 0.353, no help), is `python tau2/format_check.py`.

## License and attribution

- **τ²-bench** (Sierra Research) is MIT-licensed; that covers the repo's released results. The
  public leaderboard bucket states no separate license. Neither is redistributed here: only run
  IDs, labels, predictions and metadata are committed, and `tau2/data_check.py` downloads the
  trajectories. Please cite τ²-bench if you use them.
- **Laya** (Convai Innovations, `NandhaKishorM/laya`) is Apache-2.0. `kaggle/train_ddp_laya.py`
  is a modified copy of its official training script; the changes are marked in the file.
- **LocalLLaMA/typed-decisions** (used for Step 0) is downloaded, not redistributed.
- Reference paper: arXiv 2606.09863, *From Confident Closing to Silent Failure: Characterizing
  False Success in LLM Agents*.
