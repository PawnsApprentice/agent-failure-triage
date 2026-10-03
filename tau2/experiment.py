"""Step 2 of the tau2 experiment: can Laya replace an LLM judge for flagging failed agent runs?
Target: run failed (reward 0) vs succeeded (reward 1); labels come from database state.

usage:
    python tau2/experiment.py corpus        # filter runs.jsonl -> data/corpus.jsonl (+ last-1,024-token tails)
    python tau2/experiment.py split         # task-disjoint 60/10/30 split by customer scenario -> data/split.json
    python tau2/experiment.py tfidf         # TF-IDF + logistic regression, full run and last 1,024 tokens,
                                            # each with and without the user simulator's control tokens
    python tau2/experiment.py baselines     # transcript-free baselines: train failure rate per domain (+ agent)
    python tau2/experiment.py laya-select   # Laya wording chosen on the calibration split only
    python tau2/experiment.py laya-test     # Laya-typed zero-shot on the test split with the chosen wording
    python tau2/experiment.py import-kaggle # use the Kaggle GPU run's outputs (tau2/kaggle/output/) instead
    python tau2/experiment.py report        # metrics with task-cluster bootstrap CIs -> results/report.json
    python tau2/experiment.py report-matched # every model incl. the Haiku judge on the same 500-run sample
    python tau2/experiment.py report-finetune # fine-tuned Laya (3 seeds) vs TF-IDF, from tau2/kaggle/output_ft/

Laya steps are resumable (predictions are appended per run) and should run under nice -n 19.
"""
from __future__ import annotations

import json
import math
import random
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from common import laya_agent
from common.metrics import (
    auroc,
    binary_nll,
    brier,
    cluster_bootstrap_ci,
    ece,
    fit_temperature,
    logit,
    percentile,
    recall_at_budget,
    stratified_auroc,
)

HERE = Path(__file__).parent
DATA = HERE / "data"
RUNS_PATH = DATA / "runs.jsonl"
CORPUS_PATH = DATA / "corpus.jsonl"
SPLIT_PATH = DATA / "split.json"
RESULTS = HERE / "results"
WORDING_PATH = RESULTS / "laya_wording.json"
REPORT_PATH = RESULTS / "report.json"

SEED = 20261002
TAIL_TOKENS = 1024
REVIEW_BUDGET = 0.10
LAYA_CHECKPOINT = "laya-typed"
DOMAINS = ("airline", "retail", "telecom")

# laya's noul type scores one statement; candidates share criteria so only the statement varies.
LAYA_CRITERIA = {"true": "the agent did not complete the customer's task",
                 "false": "the agent completed the customer's task"}
LAYA_WORDINGS = {
    "W1": "This agent run failed to complete the customer's task.",
    "W2": "By the end of this conversation, the customer's request was not fully and correctly carried out.",
    "W3": "The agent left the customer's problem unresolved or handled it incorrectly.",
}


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))


def tokenizer():
    return laya_agent.get_agent(LAYA_CHECKPOINT).tok


def tail_text(tok, text: str, max_tokens: int) -> tuple[str, int]:
    """Last `max_tokens` laya tokens of `text`, decoded back to text; trimmed further if
    re-tokenizing the decoded string comes out longer, so the result always fits."""
    ids = tok(text, add_special_tokens=False, verbose=False)["input_ids"]
    k = min(len(ids), max_tokens)
    while True:
        s = tok.decode(ids[len(ids) - k:], clean_up_tokenization_spaces=False)
        n = len(tok(s, add_special_tokens=False, verbose=False)["input_ids"])
        if n <= max_tokens:
            return s, n
        k -= 8


# ---------------------------------------------------------------------------
# Corpus and split
# ---------------------------------------------------------------------------


def in_corpus(r: dict) -> bool:
    """Decided after the Step 1 report: leaderboard bucket, airline/retail/telecom, default variant;
    excludes gpt-5-2-none (published scores disagree, unexplained) and gemini-3-flash airline
    (run on an older version of 22 airline tasks). Also excludes empty runs (no messages,
    termination infrastructure_error): they never happened, yet carry reward 1."""
    return (r["source"] == "bucket" and r["domain"] in DOMAINS and r["variant"] == "default"
            and r["n_messages"] > 0
            and not r["submission"].startswith("gpt-5-2-none")
            and not (r["submission"].startswith("gemini-3-flash") and r["domain"] == "airline"))


def build_corpus() -> None:
    tok = tokenizer()
    rows = []
    for r in load_jsonl(RUNS_PATH):
        if not in_corpus(r):
            continue
        r["run_id"] = f"{r['submission']}|{r['domain']}|{r['task_id']}|{r['trial']}"
        r["failed"] = r["reward"] == 0
        r["group"] = f"{r['domain']}|{r['task_hash']}"  # customer scenario: the leakage-safe task key
        r["tail_1024"], r["tail_tokens"] = tail_text(tok, r["text"], TAIL_TOKENS)
        rows.append(r)
    assert len({r["run_id"] for r in rows}) == len(rows), "run ids must be unique"
    write_jsonl(CORPUS_PATH, rows)
    print(f"wrote {len(rows)} runs to {CORPUS_PATH}", file=sys.stderr)


def make_split() -> None:
    """Task-disjoint 60/10/30 by customer scenario (domain + user_scenario hash), stratified by
    domain. Telecom tasks that share a scenario differ only in hidden injected device faults, so
    grouping by scenario keeps the same customer conversation out of both train and test."""
    corpus = load_jsonl(CORPUS_PATH)
    rng = random.Random(SEED)
    assignment = {}
    for dom in DOMAINS:
        groups = sorted({r["group"] for r in corpus if r["domain"] == dom})
        rng.shuffle(groups)
        n = len(groups)
        n_test, n_cal = math.floor(0.3 * n + 0.5), max(1, math.floor(0.1 * n + 0.5))
        for i, g in enumerate(groups):
            assignment[g] = "test" if i < n_test else "calibration" if i < n_test + n_cal else "train"
    SPLIT_PATH.write_text(json.dumps({"seed": SEED, "group_key": "domain|user_scenario hash",
                                      "groups": assignment}, indent=1) + "\n")
    print_split(corpus, assignment)


def print_split(corpus: list[dict], assignment: dict[str, str]) -> None:
    print(f"\n{'split':12s}{'domain':10s}{'groups':>8s}{'runs':>7s}{'failed':>8s}{'fail%':>8s}")
    for split in ("train", "calibration", "test"):
        for dom in DOMAINS + ("ALL",):
            rs = [r for r in corpus if assignment[r["group"]] == split and dom in (r["domain"], "ALL")]
            groups = {r["group"] for r in rs}
            fails = sum(r["failed"] for r in rs)
            print(f"{split:12s}{dom:10s}{len(groups):8d}{len(rs):7d}{fails:8d}{fails / len(rs):8.1%}")


def split_rows(split: str) -> list[dict]:
    assignment = json.loads(SPLIT_PATH.read_text())["groups"]
    return [r for r in load_jsonl(CORPUS_PATH) if assignment[r["group"]] == split]


# ---------------------------------------------------------------------------
# TF-IDF + logistic regression
# ---------------------------------------------------------------------------

# name -> (corpus field, strip the user simulator's control tokens?)
TFIDF_VIEWS = {"tfidf-full": ("text", False), "tfidf-tail1024": ("tail_1024", False),
               "tfidf-full-notok": ("text", True), "tfidf-tail1024-notok": ("tail_1024", True)}
# tau2's simulated customer ends a conversation by emitting one of these. They are benchmark
# plumbing that real traffic doesn't have, and the ending they mark correlates with the outcome.
CONTROL_TOKENS = re.compile(r"###(?:STOP|TRANSFER|OUT-OF-SCOPE)###")


def tfidf_input(r: dict, view: str) -> str:
    field, strip = TFIDF_VIEWS[view]
    return CONTROL_TOKENS.sub("", r[field]) if strip else r[field]


def run_tfidf() -> None:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression

    train, test = split_rows("train"), split_rows("test")
    for name in TFIDF_VIEWS:
        # Unigram+bigram TF-IDF (30k features) with L2 logistic regression, as in arXiv 2606.09863.
        vec = TfidfVectorizer(ngram_range=(1, 2), max_features=30_000, sublinear_tf=True, min_df=2)
        clf = LogisticRegression(max_iter=2000, C=1.0)
        clf.fit(vec.fit_transform([tfidf_input(r, name) for r in train]), [int(r["failed"]) for r in train])
        col = list(clf.classes_).index(1)
        preds = []
        for r in test:
            t0 = time.perf_counter()
            p = float(clf.predict_proba(vec.transform([tfidf_input(r, name)]))[0, col])
            preds.append({"run_id": r["run_id"], "p_fail": p, "latency_ms": (time.perf_counter() - t0) * 1000})
        write_jsonl(RESULTS / f"pred_{name}.jsonl", preds)
        print(f"{name}: trained on {len(train)} runs, scored {len(test)}", file=sys.stderr)


# ---------------------------------------------------------------------------
# Transcript-free baselines
# ---------------------------------------------------------------------------

RATE_BASELINES = {"rate-domain": lambda r: r["domain"],
                  "rate-domain+agent": lambda r: (r["domain"], r["submission"])}


def run_baselines() -> None:
    """Score each test run with the training split's failure rate for its domain, or for its
    domain and agent (submission). They never read the transcript, so they show how much of a
    pooled AUROC comes from base rates alone."""
    train, test = split_rows("train"), split_rows("test")
    for name, key in RATE_BASELINES.items():
        counts = defaultdict(lambda: [0, 0])
        for r in train:
            counts[key(r)][0] += r["failed"]
            counts[key(r)][1] += 1
        rate = {k: f / n for k, (f, n) in counts.items()}
        preds = []
        for r in test:
            t0 = time.perf_counter()
            p = rate[key(r)]
            preds.append({"run_id": r["run_id"], "p_fail": p, "latency_ms": (time.perf_counter() - t0) * 1000})
        write_jsonl(RESULTS / f"pred_{name}.jsonl", preds)
        print(f"{name}: {len(rate)} rates from {len(train)} train runs, scored {len(test)}", file=sys.stderr)


# ---------------------------------------------------------------------------
# Laya-typed zero-shot
# ---------------------------------------------------------------------------


def laya_question(wid: str) -> dict:
    return {"type": "noul", "instructions": LAYA_WORDINGS[wid], "criteria": LAYA_CRITERIA}


def laya_state_room(question: dict) -> int:
    """Tokens left for the state once laya has placed the question and options in its 1,024-token
    window (mirrors laya.common.build_sequence). laya keeps the *start* of a string state, so the
    tail must be pre-cut to this length or its end - where outcomes show - would be dropped."""
    from laya.common import build_sequence

    agent = laya_agent.get_agent(LAYA_CHECKPOINT)
    max_len, head = agent.cfg.get("max_len", 512), agent.cfg.get("head_max_len", 192)
    ids, _ = build_sequence(agent.tok, "", agent._to_internal(question), max_len, head)
    return max_len - len(ids)


def run_laya(rows: list[dict], wid: str, out_path: Path, checkpoint: str = LAYA_CHECKPOINT) -> None:
    question = laya_question(wid)
    room = laya_state_room(question)
    tok = tokenizer()
    done = {r["run_id"] for r in load_jsonl(out_path)} if out_path.exists() else set()
    todo = [r for r in rows if r["run_id"] not in done]
    print(f"Laya {wid}: state room {room} tokens; {len(done)} done, {len(todo)} to go -> {out_path.name}",
          file=sys.stderr)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("a") as f:
        for i, r in enumerate(todo, 1):
            state, n_tok = tail_text(tok, r["tail_1024"], room)
            out = laya_agent.predict(checkpoint, state, {"failed": question})
            f.write(json.dumps({"run_id": r["run_id"], "p_fail": float(out["answers"]["failed"]["noul"]),
                                "latency_ms": out["latency_ms"], "state_tokens": n_tok}) + "\n")
            f.flush()
            if i % 100 == 0 or i == len(todo):
                print(f"  {i}/{len(todo)}", file=sys.stderr)


def select_wording(cal: list[dict], out_dir: Path) -> dict:
    """Same protocol as BGL: lowest calibration-split NLL at each wording's own refit temperature,
    fixed before any test-split result is seen. Writes out_dir/laya_wording.json."""
    labels = {r["run_id"]: float(r["failed"]) for r in cal}
    t_shipped = laya_agent.noul_temperature(LAYA_CHECKPOINT)
    scores = {}
    for wid in LAYA_WORDINGS:
        path = out_dir / f"laya_calibration_{wid}.jsonl"
        run_laya(cal, wid, path)
        preds = load_jsonl(path)
        z_raw = [logit(p["p_fail"]) * t_shipped for p in preds]
        y = [labels[p["run_id"]] for p in preds]
        t_refit = fit_temperature(z_raw, y)
        scores[wid] = {"nll_shipped": binary_nll(z_raw, y, t_shipped), "nll_refit": binary_nll(z_raw, y, t_refit),
                       "t_shipped": t_shipped, "t_refit": t_refit,
                       "auroc_calibration": auroc([p["p_fail"] for p in preds], [bool(v) for v in y]), "n": len(y)}
        print(f"  {wid}: nll shipped={scores[wid]['nll_shipped']:.4f} refit={scores[wid]['nll_refit']:.4f} "
              f"(T={t_refit:.3f})  calibration AUROC={scores[wid]['auroc_calibration']:.3f}", file=sys.stderr)
    chosen = min(scores, key=lambda w: scores[w]["nll_refit"])
    selection = {"selection_rule": "lowest calibration-split NLL at each wording's own refit temperature",
                 "criteria": LAYA_CRITERIA, "wordings": LAYA_WORDINGS, "scores": scores, "chosen": chosen}
    (out_dir / "laya_wording.json").write_text(json.dumps(selection, indent=2) + "\n")
    print(f"chose {chosen}: {LAYA_WORDINGS[chosen]!r}", file=sys.stderr)
    return selection


def score_test(test: list[dict], out_dir: Path) -> Path:
    chosen = json.loads((out_dir / "laya_wording.json").read_text())["chosen"]
    path = out_dir / f"pred_laya-typed_{chosen}.jsonl"
    run_laya(test, chosen, path)
    return path


KAGGLE_OUT = HERE / "kaggle" / "output"


def import_kaggle() -> None:
    """Bring the Kaggle GPU run's outputs (unzipped into tau2/kaggle/output/) into results/, so
    `report` uses them. The partial local CPU predictions are moved aside, not deleted."""
    for name in ("laya_wording.json", "laya_predictions.csv", "run_info.json", "step0_gpu.json", "parity.json"):
        if not (KAGGLE_OUT / name).exists():
            raise SystemExit(f"missing {KAGGLE_OUT / name} - unzip Kaggle's output into {KAGGLE_OUT}")
    for old in RESULTS.glob("laya_calibration_W*.jsonl"):
        (RESULTS / "cpu_partial").mkdir(exist_ok=True)
        old.rename(RESULTS / "cpu_partial" / old.name)
    import csv

    files = defaultdict(list)
    with (KAGGLE_OUT / "laya_predictions.csv").open() as f:
        for row in csv.DictReader(f):
            name = (f"laya_calibration_{row['wording']}.jsonl" if row["split"] == "calibration"
                    else f"pred_laya-typed_{row['wording']}.jsonl")
            files[name].append({"run_id": row["run_id"], "p_fail": float(row["p_fail"]),
                                "latency_ms": float(row["latency_ms"]), "state_tokens": int(row["state_tokens"])})
    for name, rows in files.items():
        write_jsonl(RESULTS / name, rows)
    WORDING_PATH.write_text((KAGGLE_OUT / "laya_wording.json").read_text())
    for name in ("run_info.json", "step0_gpu.json", "parity.json"):
        (RESULTS / f"kaggle_{name}").write_text((KAGGLE_OUT / name).read_text())
    info = json.loads((KAGGLE_OUT / "run_info.json").read_text())
    print(f"imported {sum(len(v) for v in files.values())} Laya predictions from {info['gpu'] or info['device']}; "
          f"chosen wording {json.loads(WORDING_PATH.read_text())['chosen']}")


def laya_select() -> None:
    select_wording(split_rows("calibration"), RESULTS)


def laya_test() -> None:
    score_test(split_rows("test"), RESULTS)


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


def _refit(p: float, t_shipped: float, t_refit: float) -> float:
    """Re-scale a shipped-temperature noul probability to another temperature (exact for 2 options)."""
    return 1 / (1 + math.exp(-logit(p) * t_shipped / t_refit))


def model_predictions() -> dict[str, dict[str, dict]]:
    models = {name: {p["run_id"]: p for p in load_jsonl(RESULTS / f"pred_{name}.jsonl")}
              for name in [*RATE_BASELINES, *TFIDF_VIEWS]}
    w = json.loads(WORDING_PATH.read_text())
    s = w["scores"][w["chosen"]]
    shipped = {p["run_id"]: p for p in load_jsonl(RESULTS / f"pred_laya-typed_{w['chosen']}.jsonl")}
    models["laya-typed (shipped T)"] = shipped
    models["laya-typed (refit T)"] = {k: dict(v, p_fail=_refit(v["p_fail"], s["t_shipped"], s["t_refit"]))
                                     for k, v in shipped.items()}
    return models


def within_domain_auroc(items: list[tuple]) -> float:
    """items: (p_fail, failed, domain). AUROC over failed/succeeded pairs from the same domain."""
    return stratified_auroc([p for p, *_ in items], [y for _, y, _ in items], [d for *_, d in items])


# items: (p_fail, failed, domain). "auroc" is pooled over domains, so it also rewards knowing that
# domains fail at different rates; "auroc_within_domain" doesn't.
METRICS = {
    "auroc": lambda items: auroc([p for p, *_ in items], [y for _, y, _ in items]),
    "auroc_within_domain": within_domain_auroc,
    "recall@10%": lambda items: recall_at_budget([p for p, *_ in items], [y for _, y, _ in items], REVIEW_BUDGET),
    "ece": lambda items: ece([p for p, *_ in items], [float(y) for _, y, _ in items]),
    "brier": lambda items: brier([p for p, *_ in items], [float(y) for _, y, _ in items]),
}


def evaluate(test: list[dict], preds: dict[str, dict]) -> dict:
    by_group = defaultdict(list)
    for r in test:
        by_group[r["group"]].append((preds[r["run_id"]]["p_fail"], r["failed"], r["domain"]))
    items = [x for g in by_group.values() for x in g]
    out = {"n_runs": len(items), "n_groups": len(by_group), "failure_rate": sum(y for _, y, _ in items) / len(items)}
    out["recall@10%_ceiling"] = min(1.0, REVIEW_BUDGET / out["failure_rate"])
    for name, stat in METRICS.items():
        lo, hi = cluster_bootstrap_ci(list(by_group.values()), stat, seed=SEED)
        out[name] = {"value": stat(items), "ci95": [lo, hi]}
    lat = [preds[r["run_id"]]["latency_ms"] for r in test]
    rng = random.Random(SEED)
    boots = sorted(percentile([lat[rng.randrange(len(lat))] for _ in lat], 0.5) for _ in range(1000))
    out["latency_ms_p50"] = {"value": percentile(lat, 0.5), "ci95": [boots[25], boots[974]]}
    out["latency_ms_p95"] = percentile(lat, 0.95)
    return out


HAIKU_PRICE_IN, HAIKU_PRICE_OUT = 1.00, 5.00  # $/MTok, claude-haiku-4-5 (claude-api skill, cached 2026-06-24)


def report_matched() -> None:
    """Every model on exactly the same runs: the Haiku judge's stratified test sample."""
    sample_ids = set(json.loads((RESULTS / "haiku_sample.json").read_text())["run_ids"])
    test = [r for r in split_rows("test") if r["run_id"] in sample_ids]
    haiku = {p["run_id"]: p for p in load_jsonl(RESULTS / "pred_haiku.jsonl")}
    unparsed = [k for k, p in haiku.items() if p["p_fail"] is None]
    missing = [r["run_id"] for r in test if r["run_id"] not in haiku]
    if missing or unparsed:
        raise SystemExit(f"Haiku: {len(missing)} sample runs not yet scored, {len(unparsed)} unparseable responses")
    models = model_predictions() | {"haiku-4.5 judge": haiku}
    rep = {"n_runs": len(test), "models": {}}
    for name, preds in models.items():
        m = evaluate(test, preds)
        lat = [preds[r["run_id"]]["latency_ms"] for r in test]
        if name.startswith("haiku"):
            usage = [preds[r["run_id"]]["raw"] for r in test]
            per_run = sum(u["input_tokens"] * HAIKU_PRICE_IN + u["output_tokens"] * HAIKU_PRICE_OUT
                          for u in usage) / 1e6 / len(usage)
            m["cost_per_1000"] = f"${1000 * per_run:.2f} API"
        elif name.startswith("rate-"):
            m["cost_per_1000"] = "~0 (table lookup)"
        else:
            hw = "T4 GPU" if name.startswith("laya") else "laptop CPU"
            m["cost_per_1000"] = f"{sum(lat) / len(lat):.0f} s {hw} (self-hosted)"
        m["by_domain_auroc"] = {d: evaluate([r for r in test if r["domain"] == d], preds)["auroc"]["value"] for d in DOMAINS}
        rep["models"][name] = m
    (RESULTS / "report_matched.json").write_text(json.dumps(rep, indent=2) + "\n")

    any_m = next(iter(rep["models"].values()))
    print(f"\n=== same {rep['n_runs']} test runs for every model: {any_m['n_groups']} scenario groups, "
          f"failure rate {any_m['failure_rate']:.1%}, recall@10% ceiling {any_m['recall@10%_ceiling']:.2f} ===")
    print(f"{'model':24s}{'AUROC pooled':>22s}{'AUROC within domain':>22s}{'recall@10%':>22s}{'ECE':>22s}"
          f"{'lat p50 ms':>12s}{'p95 ms':>10s}  cost per 1,000 runs")
    for name, m in rep["models"].items():
        print(f"{name:24s}{_fmt(m, 'auroc'):>22s}{_fmt(m, 'auroc_within_domain'):>22s}{_fmt(m, 'recall@10%'):>22s}"
              f"{_fmt(m, 'ece'):>22s}"
              f"{m['latency_ms_p50']['value']:12.1f}{m['latency_ms_p95']:10.1f}  {m['cost_per_1000']}")
    print("\nAUROC by domain:")
    for name, m in rep["models"].items():
        print(f"  {name:24s}" + "  ".join(f"{d} {v:.3f}" for d, v in m["by_domain_auroc"].items()))


FT_OUT = HERE / "kaggle" / "output_ft"


def report_finetune() -> None:
    """Fine-tuned laya-typed (3 seeds of the setting chosen on calibration) vs TF-IDF on the last
    1,024 tokens, on the full test split and on the 497-run Haiku sample. No seed is picked: each
    seed is reported, plus mean and range; the paired bootstrap is run per seed."""
    from common.metrics import paired_cluster_bootstrap

    sel = json.loads((FT_OUT / "finetune_selection.json").read_text())
    extra = json.loads((FT_OUT / "finetune_seeds.json").read_text())
    chosen = sel["chosen"]
    if extra["setting"] != chosen:
        raise SystemExit(f"seed runs used {extra['setting']}, but calibration chose {chosen}")
    seeds = [sel["info"]["selection_seed"]] + extra["seeds"]
    sel["info"]["fitted_temperatures"] |= extra["info"]["fitted_temperatures"]
    ft = {s: {p["run_id"]: p for p in load_jsonl(FT_OUT / f"test_{chosen}_seed{s}.jsonl")} for s in seeds}
    tfidf = {p["run_id"]: p for p in load_jsonl(RESULTS / "pred_tfidf-tail1024.jsonl")}
    sample_ids = set(json.loads((RESULTS / "haiku_sample.json").read_text())["run_ids"])
    test_all = split_rows("test")
    scopes = {"full test": test_all, "Haiku 497 sample": [r for r in test_all if r["run_id"] in sample_ids]}
    rep = {"selection": sel, "scopes": {}}

    print("\n=== setting selection on calibration (seed 42; lowest log loss at the fitted temperature) ===")
    for s, m in sel["scores"].items():
        epochs, lr_e, lr_h = sel["info"]["settings"][s]
        print(f"  {s}{' *' if s == chosen else '  '} {epochs} epochs, lr {lr_e:g}/{lr_h:g}: "
              f"NLL {m['nll']:.4f}  AUROC {m['auroc']:.3f}  (n={m['n']})")
    print(f"  fitted noul temperatures: { {k: round(v[2], 3) for k, v in sel['info']['fitted_temperatures'].items()} }")

    paired_stats = {"auroc": lambda xs: auroc([p for p, *_ in xs], [y for _, y, _ in xs]),
                    "auroc_within_domain": within_domain_auroc}

    for scope, rows in scopes.items():
        missing = [r["run_id"] for r in rows for s in seeds if r["run_id"] not in ft[s]]
        if missing:
            raise SystemExit(f"{scope}: {len(missing)} missing fine-tuned predictions")
        out = {"tfidf-tail1024": evaluate(rows, tfidf)}
        for s in seeds:
            out[f"laya-ft seed {s}"] = evaluate(rows, ft[s])
            clusters = defaultdict(list)
            for r in rows:
                clusters[r["group"]].append((ft[s][r["run_id"]]["p_fail"], tfidf[r["run_id"]]["p_fail"], r["failed"],
                                             r["domain"]))
            out[f"laya-ft seed {s}"]["paired_vs_tfidf"] = {
                k: paired_cluster_bootstrap(list(clusters.values()), f, seed=SEED) for k, f in paired_stats.items()}
        per_seed = [out[f"laya-ft seed {s}"] for s in seeds]
        out["laya-ft mean of 3 seeds"] = {
            k: {"mean": sum(m[k]["value"] for m in per_seed) / len(per_seed),
                "range": [min(m[k]["value"] for m in per_seed), max(m[k]["value"] for m in per_seed)]}
            for k in ("auroc", "auroc_within_domain", "recall@10%", "ece", "brier")}
        rep["scopes"][scope] = out

        t = out["tfidf-tail1024"]
        print(f"\n=== {scope}: {t['n_runs']} runs, {t['n_groups']} scenario groups, failure rate "
              f"{t['failure_rate']:.1%}, recall@10% ceiling {t['recall@10%_ceiling']:.2f} ===")
        print(f"{'model':24s}{'AUROC pooled':>22s}{'AUROC within domain':>22s}{'recall@10%':>22s}{'ECE':>22s}"
              f"{'lat p50 ms':>12s}{'p95 ms':>9s}")
        for name in ["tfidf-tail1024"] + [f"laya-ft seed {s}" for s in seeds]:
            m = out[name]
            print(f"{name:24s}{_fmt(m, 'auroc'):>22s}{_fmt(m, 'auroc_within_domain'):>22s}{_fmt(m, 'recall@10%'):>22s}"
                  f"{_fmt(m, 'ece'):>22s}"
                  f"{m['latency_ms_p50']['value']:12.1f}{m['latency_ms_p95']:9.1f}")
        mm = out["laya-ft mean of 3 seeds"]
        print(f"{'laya-ft mean [range]':24s}" + "".join(
            f"{mm[k]['mean']:>8.3f} [{mm[k]['range'][0]:.3f}, {mm[k]['range'][1]:.3f}]"
            for k in ("auroc", "auroc_within_domain", "recall@10%", "ece")))
        for k in paired_stats:
            print(f"paired bootstrap, {k}(laya-ft) - {k}(tfidf-tail1024), resampling scenario groups:")
            for s in seeds:
                p = out[f"laya-ft seed {s}"]["paired_vs_tfidf"][k]
                print(f"  seed {s}: {p['diff']:+.3f}  95% CI [{p['ci95'][0]:+.3f}, {p['ci95'][1]:+.3f}]  "
                      f"P(laya-ft not better) {p['p_a_not_better']:.3f}")
    (RESULTS / "report_finetune.json").write_text(json.dumps(rep, indent=2) + "\n")


def report() -> None:
    test = split_rows("test")
    models = model_predictions()
    missing = {m: len(test) - sum(r["run_id"] in p for r in test) for m, p in models.items()}
    if any(missing.values()):
        raise SystemExit(f"missing test predictions: {missing}")
    rep = {"overall": {}, "by_domain": {}, "laya_wording": json.loads(WORDING_PATH.read_text())}
    for m, preds in models.items():
        rep["overall"][m] = evaluate(test, preds)
        rep["by_domain"][m] = {d: evaluate([r for r in test if r["domain"] == d], preds) for d in DOMAINS}
    laya_tok = [p["state_tokens"] for p in models["laya-typed (shipped T)"].values()]
    rep["laya_state_tokens"] = {"median": percentile(laya_tok, 0.5), "min": min(laya_tok), "max": max(laya_tok)}
    for name in ("run_info", "step0_gpu", "parity"):
        path = RESULTS / f"kaggle_{name}.json"
        if path.exists():
            rep[f"kaggle_{name}"] = json.loads(path.read_text())
    REPORT_PATH.write_text(json.dumps(rep, indent=2) + "\n")
    print_report(rep)


def _fmt(m: dict, key: str, digits: int = 3) -> str:
    v, (lo, hi) = m[key]["value"], m[key]["ci95"]
    return f"{v:.{digits}f} [{lo:.{digits}f}, {hi:.{digits}f}]"


def print_tables(rep: dict) -> None:
    def table(title: str, block: dict[str, dict]) -> None:
        any_m = next(iter(block.values()))
        print(f"\n=== {title}: {any_m['n_runs']} runs, {any_m['n_groups']} scenario groups, "
              f"failure rate {any_m['failure_rate']:.1%}, recall@10% ceiling {any_m['recall@10%_ceiling']:.2f} ===")
        print(f"{'model':24s}{'AUROC pooled':>22s}{'AUROC within domain':>22s}{'recall@10%':>22s}{'ECE':>22s}"
              f"{'Brier':>22s}{'latency p50 ms':>24s}")
        for name, m in block.items():
            lat = m["latency_ms_p50"]
            print(f"{name:24s}{_fmt(m, 'auroc'):>22s}{_fmt(m, 'auroc_within_domain'):>22s}{_fmt(m, 'recall@10%'):>22s}"
                  f"{_fmt(m, 'ece'):>22s}"
                  f"{_fmt(m, 'brier'):>22s}{lat['value']:>10.1f} [{lat['ci95'][0]:.1f}, {lat['ci95'][1]:.1f}]")

    table("test split, all domains", rep["overall"])
    for d in DOMAINS:
        table(f"test split, {d}", {m: rep["by_domain"][m][d] for m in rep["by_domain"]})


def print_report(rep: dict) -> None:
    print_tables(rep)
    w = rep["laya_wording"]
    print(f"\nLaya wording (chosen on calibration split: {w['chosen']}):")
    for wid, s in w["scores"].items():
        print(f"  {wid}{' *' if wid == w['chosen'] else '  '} nll shipped-T {s['nll_shipped']:.4f}  "
              f"refit-T {s['nll_refit']:.4f} (T {s['t_refit']:.3f})  cal AUROC {s['auroc_calibration']:.3f}  "
              f"{w['wordings'][wid]!r}")
    t = rep["laya_state_tokens"]
    print(f"Laya sees the last {t['median']} tokens of each run (min {t['min']}, max {t['max']}); "
          f"the rest of its 1,024-token window is the question.")
    if "kaggle_run_info" in rep:
        i, s0, par = rep["kaggle_run_info"], rep["kaggle_step0_gpu"], rep["kaggle_parity"]
        print(f"Laya ran on {i['gpu']} ({i['dtype']}, laya {i['laya']}); TF-IDF latency is this laptop's CPU. "
              f"GPU Step 0: {s0['accuracy']:.4f} vs 0.766, CPU decision agreement {s0['decision_agreement_with_cpu']:.4f}; "
              f"tau2 CPU/GPU max |diff| {par['max_abs_diff']:.4f} on {par['n']} runs.")


if __name__ == "__main__":
    commands = {"corpus": build_corpus, "split": make_split, "tfidf": run_tfidf, "baselines": run_baselines,
                "laya-select": laya_select, "laya-test": laya_test, "import-kaggle": import_kaggle, "report": report,
                "report-matched": report_matched, "report-finetune": report_finetune}
    cmd = sys.argv[1] if len(sys.argv) > 1 else "--help"
    if cmd in commands:
        commands[cmd]()
    else:
        print(__doc__.strip())
