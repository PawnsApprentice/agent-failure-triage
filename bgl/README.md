# Laya on-call triage demo

On-call alert triage on real BlueGene/L (BGL) logs. The comparison covers Laya (two checkpoints),
Claude Haiku 4.5, a local Qwen2.5-1.5B, and a TF-IDF + logistic regression baseline. The
project was built for a LinkedIn post. This file holds the methodology, the final numbers, the
corrections made along the way, and the caveats that should go with any number quoted from here.

## Final results

### Template-level eval set (canonical, n=100: 50 page + 50 ignore, one row per template)

| model | TP | FP | TN | FN | accuracy | 95% CI | brier | ECE | p50 latency |
|---|---|---|---|---|---|---|---|---|---|
| laya-base (shipped T) | 26 | 41 | 9 | 24 | 0.350 | [0.264, 0.447] | 0.260 | 0.182 | 533ms |
| laya-base (refit T) | 26 | 41 | 9 | 24 | 0.350 | [0.264, 0.447] | 0.253 | 0.163 | 531ms |
| laya-typed-decisions | 49 | 49 | 1 | 1 | 0.500 | [0.404, 0.596] | 0.256 | 0.057 | 511ms |
| claude-haiku-4-5 | 43 | 14 | 36 | 7 | 0.790 | [0.700, 0.858] | 0.161 | 0.128 | 1057ms |
| qwen2.5-1.5b | 49 | 48 | 2 | 1 | 0.510 | [0.413, 0.606] | 0.377 | 0.407 | 1843ms |
| **tfidf + logreg** | 35 | 0 | 50 | 15 | **0.850** | [0.767, 0.907] | 0.114 | 0.181 | 1ms |

### Row-level eval set (original sample, n=200: 100 page + 100 ignore, templates repeat)

| model | TP | FP | TN | FN | accuracy | 95% CI | brier | ECE | p50 latency |
|---|---|---|---|---|---|---|---|---|---|
| laya-base (shipped T) | 20 | 45 | 55 | 80 | 0.375 | [0.311, 0.444] | 0.254 | 0.199 | 562ms |
| laya-base (refit T) | 20 | 45 | 55 | 80 | 0.375 | [0.311, 0.444] | 0.251 | 0.140 | 563ms |
| laya-typed-decisions | 99 | 92 | 8 | 1 | 0.535 | [0.466, 0.603] | 0.246 | 0.050 | 528ms |
| claude-haiku-4-5 | 98 | 16 | 84 | 2 | 0.910 | [0.862, 0.942] | 0.079 | 0.091 | 1034ms |
| qwen2.5-1.5b | 99 | 98 | 2 | 1 | 0.505 | [0.436, 0.574] | 0.446 | 0.458 | 2044ms |
| tfidf + logreg (**leaky**) | 89 | 0 | 100 | 11 | 0.945 | [0.904, 0.969] | 0.037 | 0.072 | 1ms |
| tfidf + logreg (leak-checked retrain) | - | - | - | - | 0.895 | [0.845, 0.930] | 0.101 | 0.157 | - |

The **leaky** TF-IDF row is the original model. 80% of its training rows share a masked
template with a row in this eval set (see "Leakage"), so do not quote it. The leak-checked row
comes from `python bgl/triage.py checks` and is the number to use for TF-IDF on this set.

### What the numbers say

- **TF-IDF and Haiku are the only models with real signal.** Their 95% intervals overlap on
  both eval sets, so neither one clearly beats the other at these sample sizes. TF-IDF never
  pages a non-alert (0 false pages on both sets) but misses 15 of 50 real alerts on the
  template-level set. Haiku catches more real alerts and has some false pages.
- **Laya (both checkpoints) and Qwen are at or below chance.** Qwen pages almost everything.
  With the page question selected by the protocol below, laya-typed also pages almost
  everything. laya-base sits below chance on both sets.
- **laya-typed's low ECE (0.05-0.06) is not good calibration.** Almost all of its predictions
  land near 0.55, and on a balanced eval set the observed page rate is also about 0.5. A model
  that always answers "slightly more likely page" gets a low ECE and carries no information.
  Use brier (about 0.25 = coin flip) and accuracy for Laya, not ECE.
- **Latency** covers the scored page/ignore decision only. Laya's severity/team pass is timed
  separately and appears only in the live view. The hardware is a 4-core laptop CPU with no
  usable GPU (see Caveats).

## Reproduce

Run from the repo root with the shared `.venv` (`python bgl/triage.py --help` lists the commands):

```
python bgl/triage.py sample    # build the original row-level 200-alert sample (superseded, see below)
python bgl/triage.py pilot     # 20-alert smoke test with Laya diagnostics
python bgl/triage.py run       # all models on whatever alerts.jsonl is -> results.jsonl
python bgl/triage.py rebuild   # template-level eval/calibration/train split -> alerts.jsonl, results.jsonl
python bgl/triage.py rehaiku   # re-run Haiku only on both eval sets (direct p_page + page boolean)
python bgl/triage.py relaya    # pick Laya's page wording on the calibration set, re-run Laya on both eval sets
python bgl/triage.py checks    # TF-IDF leakage check + Wilson CIs against results.jsonl
python bgl/triage.py plot      # results.jsonl -> calibration.png
python bgl/triage.py serve     # live view at http://127.0.0.1:8420 (replays results.jsonl)
```

Shared code lives in `common/` at the repo root: metrics (accuracy, brier, ECE, Wilson CI,
reliability bins, temperature fit), the Laya wrapper (`common/laya_agent.py`) and plotting.
The Qwen GGUF (`models/`) and the API key (`.env`) also stay at the repo root.

Files (all under `bgl/`):
- `data/BGL.zip`: the Loghub BGL release; `logs/`: run logs.
- `alerts.jsonl` / `results.jsonl`: canonical template-level eval set and its results.
- `alerts.jsonl.rowlevel200.bak`: the original row-level 200-alert eval set.
- `results_rowlevel200.jsonl`: results on that set with the current Haiku and Laya runs.
  Qwen is unchanged from the original run. The TF-IDF rows are the leaky original model.
- `results.jsonl.rowlevel200.bak`: the untouched original row-level results, kept for provenance.
- `laya_wording.json`: Laya page-question candidates, calibration scores, and the selected wording.
- Haiku and Laya rows store the model's raw output in `raw`: Haiku's response text, and Laya's
  raw answers plus the temperature used. Qwen and TF-IDF rows predate raw storage and have
  `raw: null`. The code now records it (Qwen's first-token logprobs, TF-IDF's class order and
  probabilities), and both mappings were checked live (see "Sanity checks performed").

## Dataset

Loghub's BGL (BlueGene/L) logs, full Zenodo release, streamed from the zip. Download it first
(~57 MB):

```
mkdir -p bgl/data && curl -L -C - -o bgl/data/BGL.zip "https://zenodo.org/records/8196385/files/BGL.zip?download=1"
```

Loghub's datasets are free for research or academic work; cite Loghub
(https://github.com/logpai/loghub) if you use them. The public repo does not include `BGL.zip`,
or the derived files that contain raw log text: `alerts.jsonl*`, `results.jsonl*` and
`results_rowlevel200.jsonl`. Re-run the commands above to regenerate them.

Line format:
`<label> <epoch> <date> <node> <timestamp> <node> RAS <COMPONENT> <LEVEL> <message>`. A label of
`-` means non-alert; anything else is an alert category. Models receive only the free-text
`message`. Label, timestamps, node IDs, component and level are stripped, so the input never
directly leaks the label. Example:

```
raw:      - 1118353989 2005.06.09 R20-M0-N7-C:J14-U11 2005-06-09-14.53.09.758330 R20-M0-N7-C:J14-U11 RAS KERNEL INFO generating core.3240
stripped: generating core.3240            (what every model sees)
masked:   generating core.<NUM>           (template key; also TF-IDF's input on the template-level set)
```

## Leakage and the template-level eval set

The original 200-row sample was deduplicated by exact message text and capped per BGL alert
category. That was not enough, because BGL reuses a small set of message templates with
different node, job and IP IDs filled in.

- 147 of the 200 rows share a masked template with another row, so they are not 200
  independent test cases.
- 4,019 of the original TF-IDF's 5,000 training rows (80%) share a masked template with an eval
  row, even though they differ in exact text. Retraining with masked text, and excluding training
  rows by masked template, drops TF-IDF on the 200-row set from 0.945 to 0.895.

To fix this, the eval set was rebuilt at template level:

- **Masking:** MAC-style colon-hex runs, `0x...` hex and bare digit runs are replaced with
  placeholders, then the data is deduplicated to one row per masked template.
- **Template counts:** the **entire** BGL dataset contains only **97 alert templates** (and
  11,166 non-alert templates). That is a property of BGL, not of the sampling.
- **Split** (deterministic; all pairwise template overlaps were verified to be 0, and the
  recomputed eval set matches `alerts.jsonl` exactly):
  - Eval: 50 + 50 templates. The requested cap of 100 + 100 would have left no alert templates
    to calibrate or train on.
  - Calibration: 15 + 15 templates.
  - TF-IDF training: the remaining 32 alert templates plus 2,000 non-alert templates, one masked
    row each, with `class_weight="balanced"`.
- **Overlap with the row-level set:** 2 of the 30 calibration templates also appear in the
  row-level 200-row eval set. The row-level Laya numbers therefore have a small calibration
  overlap that the template-level numbers do not.

## Haiku scoring

**Correction.** The first version asked Haiku for a `page` boolean plus a 0-100 "confidence that
the decision is correct", and computed `p_page = confidence if page else 1 - confidence`. Haiku
sometimes answered `page: false` with a confidence below 50. That formula then pushed `p_page`
above 0.5 and scored a correct "ignore" as a false page. On the template-level set this produced
45 false pages and an accuracy of 0.48. An explanation built on that number ("Haiku over-pages on
register dumps") is **retracted**.

**Current version.** Haiku returns `page` (boolean) and `p_page` ("probability 0-100 that this
alert should page on-call"), plus severity and team.
- The decision is scored from the boolean, and `p_page` is used directly for brier, ECE and
  the calibration curve.
- On both eval sets, 0 rows have a boolean that contradicts `p_page`.
- With this version Haiku has 14 false pages and 0.79 accuracy on the template-level set.
- Haiku's `p_page` is a stated number, not a token probability. The API does not expose
  logprobs, so its calibration curve measures a different kind of quantity than Laya's or Qwen's.

## Laya page-question wording

**Why the wording changed.** laya's `noul` question type scores a single statement. Its default
options are "yes, the statement holds" and "no, the statement does not hold". The original
question had two clauses: "Should this alert page an on-call engineer right now (true), or can
it be safely ignored (false)?" That leaves "the statement" ambiguous. On hand-written control
alerts, Laya's answers moved with the question's wording more than with the alert's content.
That is a problem with the prompt and the model, not an inversion in the code:
- laya always orders the options [false, true] and returns P(true).
- The live checks below confirm every model's p_page mapping.

**Protocol, fixed before any eval-set result was seen:**
- Three single-statement candidates share the criteria `true: "page an on-call engineer now"`
  and `false: "safe to ignore, no page needed"`.
- Each candidate is scored by negative log likelihood (NLL) on the 30-row template-disjoint
  calibration set, at that candidate's own refit temperature.
- The lowest NLL wins, separately for each checkpoint. Each checkpoint then runs once on both
  eval sets, with no further tuning.
- The original two-clause question is scored for reference only and was not eligible.

| wording | laya-base NLL (shipped T / refit T, T) | laya-typed NLL (shipped T / refit T, T) |
|---|---|---|
| W1 "This log line indicates a failure that requires an on-call engineer to act now." | 0.721 / 0.701, 5.000 | 0.753 / **0.707**, 5.000 **(chosen)** |
| W2 "This alert should page an on-call engineer immediately." | 0.726 / 0.701, 5.000 | 0.778 / 0.715, 5.000 |
| W3 "This system log message reports a problem serious enough to wake up an on-call engineer at night." | 0.694 / **0.693**, 4.820 **(chosen)** | 0.754 / 0.707, 5.000 |
| reference only: original two-clause question | 1.861 / 1.066, 5.000 | **0.644 / 0.644**, 1.917 |

Read this table carefully:
- **Every new wording scores close to ln 2 = 0.693, the NLL of always answering 0.5.** The refit
  temperature also hits or nearly hits laya's maximum of 5.0, which flattens predictions toward
  0.5. Neither checkpoint gets a usable signal on BGL from any of these wordings.
- **For laya-typed, the original question scored better on calibration (0.644) than all three
  candidates.** Under the protocol it could not be selected. With the original question,
  laya-typed scored 0.660 on the template-level eval set and 0.445 on the row-level set. That
  run is superseded, but it shows that Laya's result depends heavily on the question wording.
  Re-selecting the wording after seeing eval results would be tuning to the eval set, so it was
  not done.

Temperatures used: laya-base shipped 1.983, refit 4.820 (W3). laya-typed shipped 1.983, refit
5.000 (W1). Both checkpoints ship the same 1.983 for this question type, which matches a known
issue where laya-typed inherited the base checkpoint's temperatures. Temperature scaling never
changes which side of 0.5 a prediction falls on, which is why the shipped-T and refit-T rows
have identical confusion matrices.

## Sanity checks performed

- **Split cleanliness:** train/eval, train/calibration and calibration/eval template overlaps
  are all 0. The split was recomputed from scratch and matched `alerts.jsonl` exactly.
- **No inverted outputs.** Each model was checked live on a true-page alert:
  - Laya: `noul` 0.71 = P(page).
  - Qwen: logprob YES -0.027 vs NO -3.96, so p_page 0.98.
  - TF-IDF: `classes_=[0, 1]`, so p_page is read from the class-1 column.
  - Haiku: the raw response is stored and its boolean agrees with `p_page` on every row.
- **Haiku API failures:** none. Any API or parse error would stop the run, and every set has a
  Haiku row for every alert.

## Caveats

- **Only 97 alert templates exist in all of BGL.** Every template-level split is thin (50 eval,
  15 calibration, 32 train for alerts). Intervals are wide; treat differences inside them as
  noise.
- **95% CIs are Wilson score intervals**, which assume independent rows. That assumption holds
  for the template-level set (one row per template) but not for the row-level set, where 147 of
  200 rows share a template with another row. Row-level intervals are therefore too narrow.
- **Severity and team have no ground truth** in BGL. They appear in the live view only and are
  never scored.
- **Hardware:** 4 CPU cores and no usable GPU (a GeForce 940MX is too old for current PyTorch).
  Laya and Qwen latencies reflect this laptop, not production hardware. Running local models at
  full thread count starved the desktop input stack during development, so long runs here use
  `nice -n 19`.
- **Each model sees a different prompt shape**, by design:
  - Laya: a single-statement yes/no question.
  - Qwen: YES/NO, read from the first-token logprobs.
  - Haiku: structured JSON output.
  - TF-IDF: no prompt (it takes the text directly).

  Laya's numbers depend heavily on the wording (see above); the other models were not
  wording-tuned.
