"""One phone-readable chart for sharing: AUROC with 95% CI for every model, on the same 497
tau2-bench test runs (the Haiku judge's stratified sample), annotated with latency and cost.
The plotted AUROC is within-domain (failed/succeeded pairs from the same domain only): pooled
AUROC also rewards knowing that domains fail at different rates, which a blind judge can't.

usage: python tau2/make_chart.py   -> tau2/results/auroc_by_model.png (1080x1350, 4:5 portrait)
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
import experiment as ex  # also puts the repo root on sys.path

from common.metrics import auroc, cluster_bootstrap_ci

OUT_PATH = ex.RESULTS / "auroc_by_model.png"
SURFACE, INK, INK_2, MUTED, ACCENT = "#fcfcfb", "#0b0b0b", "#52514e", "#a3a29c", "#2a78d6"  # dataviz reference palette


def model_rows() -> tuple[list[dict], int]:
    sample_ids = set(json.loads((ex.RESULTS / "haiku_sample.json").read_text())["run_ids"])
    rows = [r for r in ex.split_rows("test") if r["run_id"] in sample_ids]
    matched = json.loads((ex.RESULTS / "report_matched.json").read_text())["models"]
    ft_rep = json.loads((ex.RESULTS / "report_finetune.json").read_text())["scopes"]["Haiku 497 sample"]
    seeds = [k for k in ft_rep if k.startswith("laya-ft seed")]

    # 95% CI of the fine-tuned mean (average AUROC of the three seeds), resampling whole scenarios
    sel = json.loads((ex.FT_OUT / "finetune_selection.json").read_text())
    ft = [{p["run_id"]: p["p_fail"] for p in ex.load_jsonl(ex.FT_OUT / f"test_{sel['chosen']}_seed{s.split()[-1]}.jsonl")}
          for s in seeds]
    clusters = defaultdict(list)
    for r in rows:
        clusters[r["group"]].append(([f[r["run_id"]] for f in ft], r["failed"], r["domain"]))

    def mean_auroc(items):
        return sum(ex.within_domain_auroc([(ps[i], y, d) for ps, y, d in items]) for i in range(len(ft))) / len(ft)

    items = [x for c in clusters.values() for x in c]
    lo, hi = cluster_bootstrap_ci(list(clusters.values()), mean_auroc, seed=ex.SEED)

    def entry(name, m, note):
        return {"name": name, "auroc": m["auroc_within_domain"]["value"], "ci": m["auroc_within_domain"]["ci95"],
                "pooled": m["auroc"]["value"], "note": note}

    out = [
        entry("TF-IDF + logistic regression", matched["tfidf-tail1024"],
              f"{matched['tfidf-tail1024']['latency_ms_p50']['value']:.1f} ms per run on a laptop CPU · ~$0"),
        {"name": "Laya, fine-tuned (mean of 3 seeds)", "auroc": mean_auroc(items), "ci": [lo, hi],
         "seeds": [ft_rep[s]["auroc_within_domain"]["value"] for s in seeds],
         "pooled": ft_rep["laya-ft mean of 3 seeds"]["auroc"]["mean"],
         "note": "~100 ms per run on a T4 GPU · self-hosted · ○ = each seed"},
        entry("Claude Haiku 4.5 as judge", matched["haiku-4.5 judge"], f"1.3 s per run · {matched['haiku-4.5 judge']['cost_per_1000'].split()[0]} per 1,000 runs"),
        entry("Laya, zero-shot", matched["laya-typed (refit T)"], "~94 ms per run on a T4 GPU · self-hosted"),
        entry("No transcript: failure rate per domain + agent", matched["rate-domain+agent"],
              "baseline · a lookup table from the training runs") | {"baseline": True},
    ]
    return out, len(rows)


def draw(models: list[dict], n_runs: int) -> None:
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.family": "DejaVu Sans", "text.color": INK})
    fig = plt.figure(figsize=(10.8, 13.5), dpi=100, facecolor=SURFACE)
    ax = fig.add_axes([0.07, 0.175, 0.86, 0.52], facecolor=SURFACE)

    fig.text(0.07, 0.945, "Spotting failed AI-agent runs", fontsize=40, fontweight="bold", va="top")
    # Wording backed by chart_claims.json (paired bootstraps, within-domain AUROC, these runs): every
    # pairwise difference among the three, including each fine-tuned seed vs Haiku, has a 95% CI
    # that includes zero.
    fig.text(0.07, 0.885, "A simple text classifier, an LLM judge and a\nfine-tuned small model: the differences are\n"
             "within the margin of error",
             fontsize=25, color=INK_2, va="top", linespacing=1.3)
    fig.text(0.07, 0.765, f"Within-domain AUROC with 95% CI · {n_runs} held-out\nτ²-bench runs (tasks unseen in training)",
             fontsize=21, color=INK_2, va="top", linespacing=1.35)

    xmin, xmax = 0.4, 1.0
    behind = {"facecolor": SURFACE, "edgecolor": "none", "pad": 1.5}  # dashed chance line passes behind text
    ys = list(range(len(models)))[::-1]
    for y, m in zip(ys, models):
        color, ink = (MUTED, INK_2) if m.get("baseline") else (ACCENT, INK)  # the baseline is a reference, not a model
        ax.text(xmin, y + 0.42, m["name"], fontsize=24, fontweight="bold", color=ink, va="bottom", bbox=behind, zorder=5)
        ax.text(xmin, y + 0.17, m["note"], fontsize=19, color=INK_2, va="bottom", bbox=behind, zorder=5)
        ax.hlines(y, m["ci"][0], m["ci"][1], color=color, linewidth=4, capstyle="round", zorder=2)
        for s in m.get("seeds", []):
            ax.scatter(s, y, s=230, facecolor=SURFACE, edgecolor=color, linewidth=2.5, zorder=3)
        ax.scatter(m["auroc"], y, s=520, color=color, edgecolor=SURFACE, linewidth=3, zorder=4)
        ax.text(m["ci"][1] + 0.015, y, f"{m['auroc']:.2f}", fontsize=24, fontweight="bold", color=ink, va="center")

    ax.axvline(0.5, color=MUTED, linewidth=2, linestyle=(0, (4, 4)), zorder=1)
    ax.text(0.507, -0.45, "chance", fontsize=18, color=INK_2, va="center")
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(-0.6, len(models) - 0.3)
    ax.set_yticks([])
    ax.set_xticks([0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    ax.tick_params(axis="x", labelsize=19, colors=INK_2, length=0, pad=10)
    ax.grid(axis="x", color="#e6e5e1", linewidth=1, zorder=0)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(MUTED)
    ax.set_xlabel("AUROC  (1.0 = perfect ranking of failed runs)", fontsize=19, color=INK_2, labelpad=14)

    fig.text(0.07, 0.088, "Label: run failed per the benchmark's database check. Task-disjoint split.\n"
                          "Within-domain: only failed/succeeded pairs from the same domain count.\n"
                          "Laya fine-tuned with its official trainer; no seed picked.",
             fontsize=16, color=INK_2, va="top", linespacing=1.4)
    fig.savefig(OUT_PATH, facecolor=SURFACE)
    print(f"wrote {OUT_PATH}")
    for m in models:
        print(f"  {m['name']:46s} within-domain AUROC {m['auroc']:.3f}  CI [{m['ci'][0]:.3f}, {m['ci'][1]:.3f}]"
              f"  (pooled {m['pooled']:.3f})  {m['note']}")


def subtitle_evidence() -> dict:
    """Paired bootstraps behind the subtitle's wording, on the same 497 runs (scenario clusters; 1,000
    resamples with the project seed ex.SEED, like every other bootstrap in the repo)."""
    sample_ids = set(json.loads((ex.RESULTS / "haiku_sample.json").read_text())["run_ids"])
    rows = [r for r in ex.split_rows("test") if r["run_id"] in sample_ids]
    sel = json.loads((ex.FT_OUT / "finetune_selection.json").read_text())
    seeds = [sel["info"]["selection_seed"]] + json.loads((ex.FT_OUT / "finetune_seeds.json").read_text())["seeds"]
    tf = {p["run_id"]: p["p_fail"] for p in ex.load_jsonl(ex.RESULTS / "pred_tfidf-tail1024.jsonl")}
    hk = {p["run_id"]: p["p_fail"] for p in ex.load_jsonl(ex.RESULTS / "pred_haiku.jsonl")}
    ft = [{p["run_id"]: p["p_fail"] for p in ex.load_jsonl(ex.FT_OUT / f"test_{sel['chosen']}_seed{s}.jsonl")}
          for s in seeds]
    clusters = defaultdict(list)
    for r in rows:
        k = r["run_id"]
        clusters[r["group"]].append((tf[k], hk[k], [f[k] for f in ft], r["failed"], r["domain"]))

    metrics = {"auroc": lambda items: auroc([p for p, *_ in items], [y for _, y, _ in items]),
               "auroc_within_domain": ex.within_domain_auroc}

    def a(xs, scores, metric):
        return metrics[metric]([(p, x[3], x[4]) for p, x in zip(scores, xs)])

    diffs = {}
    for metric in metrics:
        diffs[f"tfidf_minus_haiku ({metric})"] = lambda xs, m=metric: (
            a(xs, [x[0] for x in xs], m) - a(xs, [x[1] for x in xs], m))
        diffs[f"tfidf_minus_laya_ft_mean ({metric})"] = lambda xs, m=metric: (
            a(xs, [x[0] for x in xs], m) - sum(a(xs, [x[2][i] for x in xs], m) for i in range(len(ft))) / len(ft))
        diffs[f"laya_ft_mean_minus_haiku ({metric})"] = lambda xs, m=metric: (
            sum(a(xs, [x[2][i] for x in xs], m) for i in range(len(ft))) / len(ft) - a(xs, [x[1] for x in xs], m))
    for i, seed in enumerate(seeds):  # no seed is picked, so no single seed may be clearly apart from Haiku either
        diffs[f"laya_ft_seed{seed}_minus_haiku (auroc_within_domain)"] = lambda xs, i=i: (
            a(xs, [x[2][i] for x in xs], "auroc_within_domain") - a(xs, [x[1] for x in xs], "auroc_within_domain"))
    items = [x for c in clusters.values() for x in c]
    out = {}
    for name, f in diffs.items():
        lo, hi = cluster_bootstrap_ci(list(clusters.values()), f, seed=ex.SEED)
        out[name] = {"diff": f(items), "ci95": [lo, hi]}
        print(f"  {name:48s} {f(items):+.3f}  95% CI [{lo:+.3f}, {hi:+.3f}]")
    (ex.RESULTS / "chart_claims.json").write_text(json.dumps(out, indent=2) + "\n")
    return out


if __name__ == "__main__":
    draw(*model_rows())
    subtitle_evidence()
