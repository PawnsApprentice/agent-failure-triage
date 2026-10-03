# agent-failure-triage

**On 497 held-out τ²-bench agent runs scored by every model, a TF-IDF classifier (1.9 ms per
run) reached within-domain AUROC 0.80, against 0.72 for Claude Haiku 4.5 as a judge and 0.73 to
0.82 for fine-tuned Laya across 3 seeds (mean 0.76). The differences among these three are not
statistically clear. Zero-shot Laya was near chance (0.56).**

Within-domain AUROC compares failed and successful runs from the same domain only. Pooled
across domains, the numbers look more decisive (TF-IDF 0.84, Haiku 0.70), but a lookup table of
domain and agent failure rates, which never reads a transcript, already scores 0.69 pooled.

![Within-domain AUROC with 95% CI per model, on the 497 runs every model scored](tau2/results/auroc_by_model.png)

- **[`tau2/`](tau2/): the main experiment.** Can Laya replace an LLM judge for flagging failed
  agent runs? Labels come from τ²-bench's database check, the split is task-disjoint, and all
  four models are evaluated with pre-registered decision rules. Methods, full tables, the three
  fine-tuning seeds and the caveats are in [`tau2/README.md`](tau2/README.md).
- **[`bgl/`](bgl/): an earlier, abandoned attempt** on BlueGene/L system logs. BGL's labels were
  system alert tags rather than triage decisions, and the logs were out of domain for Laya, so the
  experiment couldn't answer the question.

Shared code (metrics, the Laya wrapper, plotting) is in `common/`. `laya_check/` reproduces Laya's
published accuracy, as a check of our harness before any experiment.

Code is MIT-licensed (see `LICENSE`), except `tau2/kaggle/train_ddp_laya.py`. That file is a
modified copy of Laya's official trainer and stays under Apache-2.0. Third-party data is not
redistributed; the scripts download it. See the license sections in each experiment's README.
