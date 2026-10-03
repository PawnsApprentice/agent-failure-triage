# agent-failure-triage

**On 2,169 held-out τ²-bench agent runs, TF-IDF (AUROC 0.84, 1.6 ms per run) matched or beat
fine-tuned Laya (0.78, mean over 3 seeds). On a 497-run subset scored by all models, it beat an
LLM judge (Claude Haiku 4.5: 0.70 vs TF-IDF's 0.84).**

![AUROC with 95% CI per model, on the 497 runs every model scored](tau2/results/auroc_by_model.png)

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
