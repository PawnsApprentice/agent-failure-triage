"""On-call alert triage on real BGL logs: Laya vs Claude Haiku vs Qwen vs TF-IDF.
Methodology, results and caveats are in bgl/README.md.

usage: python bgl/triage.py <command>

commands:
    sample    build the original row-level 200-alert sample -> alerts.jsonl (superseded)
    pilot     20-alert smoke test with Laya diagnostics
    run       all models on alerts.jsonl -> results.jsonl
    rebuild   template-level eval/calibration/train split -> alerts.jsonl, results.jsonl
    rehaiku   re-run Haiku only on both eval sets
    relaya    pick Laya's page wording on the calibration set, re-run Laya on both eval sets
    checks    TF-IDF leakage check + Wilson CIs against results.jsonl
    plot      results.jsonl -> calibration.png
    serve     live view at http://127.0.0.1:8420 (replays results.jsonl)
"""
from __future__ import annotations

import json
import math
import os
import random
import re
import sys
import time
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from common import anthropic_client, laya_agent  # noqa: E402
from common.metrics import (  # noqa: E402
    accuracy_counts, binary_brier, binary_nll, brier, confusion_counts, decide, ece, fit_temperature,
    logit, percentile, wilson_ci,
)
from common.plotting import reliability_and_latency  # noqa: E402

DATA_DIR = Path(__file__).parent / "data"
BGL_ZIP = DATA_DIR / "BGL.zip"
ALERTS_PATH = Path(__file__).parent / "alerts.jsonl"
MODELS_DIR = REPO_ROOT / "models"
QWEN_GGUF = MODELS_DIR / "qwen2.5-1.5b-instruct-q4_k_m.gguf"

QWEN_SPEED_LIMIT_S = 5.0  # per plan: drop Qwen from the comparison if it's slower than this per alert

# Haiku is pinned to the model the user asked for, with a one-time fallback to the current
# alias if that dated snapshot doesn't exist on this account.
HAIKU_MODEL_REQUESTED = "claude-haiku-4-5-20251001"
HAIKU_MODEL_FALLBACK = "claude-haiku-4-5"

SEVERITY_CRITERIA = {
    "low": "minor, informational, no user impact",
    "medium": "degraded but not fully down",
    "high": "critical, service down or data at risk",
}
TEAM_CRITERIA = {
    "infra": "hardware, kernel, node, or other low-level system failures",
    "app": "user application or job-level failures",
    "storage": "filesystem, storage, or I/O failures",
    "network": "network connectivity or communication failures",
    "security": "security incidents, unauthorized access, permission issues",
}

SEED = 20260929  # today's date, fixed for reproducibility
N_PER_CLASS = 100
MAX_PER_CATEGORY = 15  # cap so one alert template doesn't dominate the sample


def _open_bgl_lines():
    """Yield raw lines from the BGL log inside the Zenodo zip, without extracting it to disk."""
    with zipfile.ZipFile(BGL_ZIP) as zf:
        # the zip contains a single big log file; find it (name varies: BGL.log, BGL/BGL.log, ...)
        candidates = [n for n in zf.namelist() if n.lower().endswith(".log")]
        if not candidates:
            raise SystemExit(f"no .log file found in {BGL_ZIP}, contents: {zf.namelist()}")
        name = candidates[0]
        with zf.open(name) as f:
            for raw in f:
                yield raw.decode("utf-8", errors="replace").rstrip("\n")


def _parse_line(line: str):
    """Parse one raw BGL line into (label, component, level, message).

    Format: <label> <epoch> <date> <node> <fulltime> <noderepeat> RAS <component> <level> <content...>
    """
    parts = line.split(None, 9)
    if len(parts) < 10:
        return None
    label, _epoch, _date, _node, _fulltime, _noderepeat, _ras, component, level, content = parts
    return label, component, level, content


_WS_RE = re.compile(r"\s+")


def _strip_message(content: str) -> str:
    """Normalize whitespace only. The component/level/label are already excluded by construction
    since we only pass `content` (the free-text message) to the models."""
    return _WS_RE.sub(" ", content).strip()


def _scan_and_clean():
    """Parse BGL, dedupe by message, drop label-conflicting messages. Returns the pools every
    sample (the 200-row pilot/run set, and the separate calibration set) draws from."""
    if not BGL_ZIP.exists():
        raise SystemExit(f"missing {BGL_ZIP} - download it first (see plan)")

    # message -> set of labels seen ("-" for non-alert, anything else = alert category)
    # message -> (component, level) from first occurrence, message -> count per label-bucket
    msg_labels: dict[str, set[str]] = defaultdict(set)
    msg_meta: dict[str, tuple[str, str]] = {}
    msg_category: dict[str, str] = {}  # for alerts, the non "-" label (category), for cap logic
    msg_raw: dict[str, str] = {}  # first raw line seen for each stripped message, for the review checkpoint
    total_lines = 0

    print("scanning BGL log (this streams from the zip, no full extraction)...", file=sys.stderr)
    for line in _open_bgl_lines():
        total_lines += 1
        if total_lines % 500_000 == 0:
            print(f"  ...{total_lines:,} lines scanned", file=sys.stderr)
        parsed = _parse_line(line)
        if parsed is None:
            continue
        label, component, level, content = parsed
        msg = _strip_message(content)
        if not msg:
            continue
        msg_labels[msg].add(label)
        if msg not in msg_meta:
            msg_meta[msg] = (component, level)
            msg_raw[msg] = line
        if label != "-":
            msg_category.setdefault(msg, label)

    print(f"total lines scanned: {total_lines:,}", file=sys.stderr)
    print(f"unique stripped messages: {len(msg_labels):,}", file=sys.stderr)

    # drop messages that appear with both alert and non-alert labels anywhere in the full dataset
    conflicting = {m for m, labels in msg_labels.items() if "-" in labels and len(labels) > 1}
    print(f"dropping {len(conflicting):,} messages seen with both alert and non-alert labels", file=sys.stderr)

    clean = {m: labels for m, labels in msg_labels.items() if m not in conflicting}
    alert_msgs = [m for m, labels in clean.items() if labels != {"-"}]
    nonalert_msgs = [m for m, labels in clean.items() if labels == {"-"}]
    print(f"usable unique messages: {len(alert_msgs):,} alert, {len(nonalert_msgs):,} non-alert", file=sys.stderr)
    return alert_msgs, nonalert_msgs, msg_meta, msg_category, msg_raw


def _capped_sample(msgs: list[str], n: int, msg_category: dict, msg_meta: dict, rng: random.Random) -> list[str]:
    """Sample n messages, capping how many can share the same alert category (or component
    for non-alerts) so the sample isn't dominated by one repeated template."""
    msgs = list(msgs)
    rng.shuffle(msgs)
    picked: list[str] = []
    per_bucket: Counter[str] = Counter()
    for m in msgs:
        bucket = msg_category.get(m) or msg_meta[m][0]  # category for alerts, component for non-alerts
        if per_bucket[bucket] >= MAX_PER_CATEGORY:
            continue
        picked.append(m)
        per_bucket[bucket] += 1
        if len(picked) == n:
            break
    return picked


def _rows_from_messages(msgs: list[str], page: bool, msg_meta: dict, msg_category: dict, msg_raw: dict) -> list[dict]:
    rows = []
    for m in msgs:
        component, level = msg_meta[m]
        rows.append({
            "text": m,
            "page": page,
            "category": msg_category[m] if page else None,
            "component": component,
            "level": level,
            "_raw_example": msg_raw[m],
        })
    return rows


def sample_alerts():
    """Step 1: parse BGL, dedupe, drop label-conflicting messages, sample 200 balanced, write alerts.jsonl."""
    alert_msgs, nonalert_msgs, msg_meta, msg_category, msg_raw = _scan_and_clean()
    rng = random.Random(SEED)

    picked_alerts = _capped_sample(alert_msgs, N_PER_CLASS, msg_category, msg_meta, rng)
    picked_nonalerts = _capped_sample(nonalert_msgs, N_PER_CLASS, msg_category, msg_meta, rng)
    if len(picked_alerts) < N_PER_CLASS or len(picked_nonalerts) < N_PER_CLASS:
        print(
            f"WARNING: only found {len(picked_alerts)} alert / {len(picked_nonalerts)} non-alert "
            f"messages under the per-category cap of {MAX_PER_CATEGORY}",
            file=sys.stderr,
        )

    rows = (
        _rows_from_messages(picked_alerts, True, msg_meta, msg_category, msg_raw)
        + _rows_from_messages(picked_nonalerts, False, msg_meta, msg_category, msg_raw)
    )
    rng.shuffle(rows)
    for i, row in enumerate(rows):
        row["id"] = i

    ALERTS_PATH.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    print(f"wrote {len(rows)} alerts to {ALERTS_PATH}", file=sys.stderr)
    return rows


def build_calibration_set(n_per_class: int = 50) -> list[dict]:
    """A 100-row set (50 page / 50 ignore) disjoint from alerts.jsonl, for fitting temperatures
    without leaking into the pilot/full-run evaluation set."""
    return build_calibration_and_tfidf_sets(calib_n_per_class=n_per_class, tfidf_n=0)[0]


def build_calibration_and_tfidf_sets(calib_n_per_class: int = 50, tfidf_n: int = 5000):
    """One BGL scan, two disjoint extra sets: a small calibration split (temperature refit) and
    a large TF-IDF training split. Both exclude alerts.jsonl; the TF-IDF split also excludes the
    calibration rows, so all three (200-eval / 100-calib / 5000-tfidf-train) are non-overlapping.
    TF-IDF training wants raw volume, not template diversity, so it skips the per-category cap
    used for the small, human-reviewed sets.
    """
    existing = {r["text"] for r in load_alerts_file()}
    alert_msgs, nonalert_msgs, msg_meta, msg_category, msg_raw = _scan_and_clean()
    alert_msgs = [m for m in alert_msgs if m not in existing]
    nonalert_msgs = [m for m in nonalert_msgs if m not in existing]
    rng = random.Random(SEED + 1)  # different seed/pool than sample_alerts(), so it can't reselect the same rows

    picked_alerts = _capped_sample(alert_msgs, calib_n_per_class, msg_category, msg_meta, rng)
    picked_nonalerts = _capped_sample(nonalert_msgs, calib_n_per_class, msg_category, msg_meta, rng)
    calib_rows = (
        _rows_from_messages(picked_alerts, True, msg_meta, msg_category, msg_raw)
        + _rows_from_messages(picked_nonalerts, False, msg_meta, msg_category, msg_raw)
    )
    rng.shuffle(calib_rows)
    for i, row in enumerate(calib_rows):
        row["id"] = i
    print(f"calibration set: {len(calib_rows)} rows ({len(picked_alerts)} page / {len(picked_nonalerts)} ignore), "
          f"disjoint from {ALERTS_PATH.name}", file=sys.stderr)

    if tfidf_n <= 0:
        return calib_rows, []

    used = set(picked_alerts) | set(picked_nonalerts)
    tfidf_alert_pool = [m for m in alert_msgs if m not in used]
    tfidf_nonalert_pool = [m for m in nonalert_msgs if m not in used]
    rng2 = random.Random(SEED + 2)
    n_each = tfidf_n // 2
    picked_tfidf_alerts = rng2.sample(tfidf_alert_pool, min(n_each, len(tfidf_alert_pool)))
    picked_tfidf_nonalerts = rng2.sample(tfidf_nonalert_pool, min(n_each, len(tfidf_nonalert_pool)))
    tfidf_rows = (
        _rows_from_messages(picked_tfidf_alerts, True, msg_meta, msg_category, msg_raw)
        + _rows_from_messages(picked_tfidf_nonalerts, False, msg_meta, msg_category, msg_raw)
    )
    rng2.shuffle(tfidf_rows)
    for i, row in enumerate(tfidf_rows):
        row["id"] = i
    print(f"TF-IDF training set: {len(tfidf_rows)} rows ({len(picked_tfidf_alerts)} page / "
          f"{len(picked_tfidf_nonalerts)} ignore), disjoint from {ALERTS_PATH.name} and the calibration set",
          file=sys.stderr)
    return calib_rows, tfidf_rows


def print_sample_rows(rows, n_each=5):
    alerts = [r for r in rows if r["page"]][:n_each]
    nonalerts = [r for r in rows if not r["page"]][:n_each]
    for title, group in [("ALERT rows (page=True)", alerts), ("NON-ALERT rows (page=False)", nonalerts)]:
        print(f"\n=== {title} ===")
        for r in group:
            label = r["category"] or "-"
            print(f"  label={label} component={r['component']} level={r['level']}")
            print(f"    raw:     {r['_raw_example']}")
            print(f"    stripped (model input): {r['text']}")


def load_alerts_file() -> list[dict]:
    if not ALERTS_PATH.exists():
        raise SystemExit(f"missing {ALERTS_PATH} - run 'python triage.py sample' first")
    return [json.loads(line) for line in ALERTS_PATH.read_text().splitlines() if line]


def load_pilot_rows(n_each: int = 10) -> list[dict]:
    """20 alerts, 10 page / 10 ignore, deterministic subset of alerts.jsonl (by id order)."""
    rows = sorted(load_alerts_file(), key=lambda r: r["id"])
    alerts = [r for r in rows if r["page"]][:n_each]
    nonalerts = [r for r in rows if not r["page"]][:n_each]
    return alerts + nonalerts


def build_questions() -> dict:
    return {
        "page": {
            "type": "noul",
            "instructions": (
                "Should this alert page an on-call engineer right now (true), "
                "or can it be safely ignored (false)?"
            ),
        },
        "severity": {
            "type": "choice",
            "instructions": "How severe is this alert?",
            "criteria": SEVERITY_CRITERIA,
        },
        "team": {
            "type": "choice",
            "instructions": "Which team should own this alert?",
            "criteria": TEAM_CRITERIA,
        },
    }


# ---------------------------------------------------------------------------
# Laya
# ---------------------------------------------------------------------------

def laya_predict(checkpoint_key: str, text: str, questions: dict | None = None) -> dict:
    out = laya_agent.predict(checkpoint_key, text, questions if questions is not None else build_questions())
    answers = out["answers"]
    return {
        "model": checkpoint_key,
        "checkpoint": out["checkpoint"],
        "p_page": float(answers["page"]["noul"]) if "page" in answers else None,
        "severity": answers["severity"]["choice"] if "severity" in answers else None,
        "team": answers["team"]["choice"] if "team" in answers else None,
        "latency_ms": out["latency_ms"],
        "raw_answers": answers,
    }


SEV_TEAM_ONLY_QUESTIONS = {k: v for k, v in build_questions().items() if k != "page"}

# laya's `noul` type scores one statement ("yes, it holds" vs "no"), so each candidate is a single
# statement; the criteria are shared so only the statement varies between candidates.
LAYA_PAGE_CRITERIA = {"true": "page an on-call engineer now", "false": "safe to ignore, no page needed"}
LAYA_PAGE_WORDINGS = {
    "W1": "This log line indicates a failure that requires an on-call engineer to act now.",
    "W2": "This alert should page an on-call engineer immediately.",
    "W3": "This system log message reports a problem serious enough to wake up an on-call engineer at night.",
}
LAYA_WORDING_PATH = Path(__file__).parent / "laya_wording.json"


def page_wording_question(wording_id: str) -> dict:
    return {"type": "noul", "instructions": LAYA_PAGE_WORDINGS[wording_id], "criteria": LAYA_PAGE_CRITERIA}


def laya_page_question(checkpoint_key: str) -> dict:
    """The page question for this checkpoint: the wording selected on the calibration set
    (laya_wording.json, written by `relaya`) if one exists, else the original two-clause question."""
    if LAYA_WORDING_PATH.exists():
        return page_wording_question(json.loads(LAYA_WORDING_PATH.read_text())[checkpoint_key]["chosen"])
    return build_questions()["page"]


def laya_predict_page(checkpoint_key: str, text: str, page_question: dict | None = None) -> dict:
    """Fast path: ask only the page question. This is what's scored (accuracy/brier/ECE) and
    what's reported as Laya's latency, since page/ignore is the only decision being evaluated."""
    q = page_question if page_question is not None else laya_page_question(checkpoint_key)
    return laya_predict(checkpoint_key, text, questions={"page": q})


def laya_predict_sev_team(checkpoint_key: str, text: str) -> dict:
    """Severity + team only, timed separately from the scored page latency. Shown in the web
    view but not part of the page/ignore accuracy, brier, or latency numbers."""
    return laya_predict(checkpoint_key, text, questions=SEV_TEAM_ONLY_QUESTIONS)


def fit_page_temperature(checkpoint_key: str, calib_rows: list[dict], page_question: dict | None = None) -> dict:
    """Refit the 'noul:2' (page/ignore) temperature on a held-out calibration set.

    `answers["page"]["noul"]` is already temperature-scaled by the shipped `T_shipped`
    (see laya/agent.py:_decode_answers: `z = logits / t_scale`). We don't have the raw
    logits through the public API, but scaling is a pointwise rescale of the logit, so we
    can recover it: z_scaled = logit(p), z_raw = z_scaled * T_shipped. Then grid-search a
    new T' over the library's own allowed range [TEMP_MIN, TEMP_MAX] minimizing NLL of
    sigmoid(z_raw / T') against the true page/ignore labels.
    """
    t_shipped = laya_agent.noul_temperature(checkpoint_key)
    z_raw = [logit(laya_predict_page(checkpoint_key, r["text"], page_question)["p_page"]) * t_shipped
             for r in calib_rows]
    labels = [1.0 if r["page"] else 0.0 for r in calib_rows]
    t_refit = fit_temperature(z_raw, labels)
    return {
        "t_shipped": t_shipped,
        "t_refit": t_refit,
        "nll_shipped": binary_nll(z_raw, labels, t_shipped),
        "nll_refit": binary_nll(z_raw, labels, t_refit),
        "brier_shipped": binary_brier(z_raw, labels, t_shipped),
        "brier_refit": binary_brier(z_raw, labels, t_refit),
        "n": len(labels),
    }


def apply_page_temperature(checkpoint_key: str, t: float) -> None:
    laya_agent.set_noul_temperature(checkpoint_key, t)


# ---------------------------------------------------------------------------
# Claude Haiku (via the Anthropic API, structured output with stated confidence)
# ---------------------------------------------------------------------------


_haiku_model_resolved = None  # set once we learn whether the dated snapshot exists


def get_anthropic_client():
    return anthropic_client.get_client()


def _haiku_triage_schema() -> dict:
    return {
        "type": "json_schema",
        "schema": {
            "type": "object",
            "properties": {
                "page": {"type": "boolean", "description": "true = page an on-call engineer now, false = ignore"},
                "p_page": {
                    "type": "integer",
                    "description": "probability 0-100 that this alert should page on-call",
                },
                "severity": {"type": "string", "enum": list(SEVERITY_CRITERIA)},
                "team": {"type": "string", "enum": list(TEAM_CRITERIA)},
            },
            "required": ["page", "p_page", "severity", "team"],
            "additionalProperties": False,
        },
    }


def haiku_predict(text: str) -> dict:
    import anthropic

    global _haiku_model_resolved
    model = _haiku_model_resolved or HAIKU_MODEL_REQUESTED
    client = get_anthropic_client()
    prompt = (
        "You are triaging a raw on-call infrastructure alert message. Decide whether an on-call "
        "engineer should be paged right now (page), give the probability 0-100 that this alert "
        "should page on-call (p_page), and classify the alert's severity and owning team.\n\n"
        f"severity options: {', '.join(SEVERITY_CRITERIA)}\n"
        f"team options: {', '.join(TEAM_CRITERIA)}\n\n"
        f"Alert message:\n{text}"
    )
    t0 = time.perf_counter()
    try:
        response = client.messages.create(
            model=model,
            max_tokens=256,
            messages=[{"role": "user", "content": prompt}],
            output_config={"format": _haiku_triage_schema()},
        )
    except anthropic.NotFoundError:
        if _haiku_model_resolved is None and model != HAIKU_MODEL_FALLBACK:
            print(
                f"WARNING: model {model!r} not found on this account, falling back to {HAIKU_MODEL_FALLBACK!r}",
                file=sys.stderr,
            )
            _haiku_model_resolved = HAIKU_MODEL_FALLBACK
            return haiku_predict(text)
        raise
    latency_ms = (time.perf_counter() - t0) * 1000
    if _haiku_model_resolved is None:
        _haiku_model_resolved = model  # confirmed working, stop probing
    text_block = next(b.text for b in response.content if b.type == "text")
    parsed = json.loads(text_block)
    return {
        "model": "haiku",
        "checkpoint": model,
        "p_page": max(0, min(100, int(parsed["p_page"]))) / 100,
        "decision": bool(parsed["page"]),  # scored from the boolean, not from p_page
        "severity": parsed["severity"],
        "team": parsed["team"],
        "latency_ms": latency_ms,
        "raw": {"response_text": text_block, "stop_reason": response.stop_reason},
    }


# ---------------------------------------------------------------------------
# Qwen2.5-1.5B-Instruct (local GGUF via llama-cpp-python), only if fast enough
# ---------------------------------------------------------------------------

_qwen_llm = None


def get_qwen():
    global _qwen_llm
    if _qwen_llm is None:
        from llama_cpp import Llama

        if not QWEN_GGUF.exists():
            raise SystemExit(f"missing {QWEN_GGUF}")
        print(f"loading Qwen from {QWEN_GGUF.name}...", file=sys.stderr)
        _qwen_llm = Llama(
            model_path=str(QWEN_GGUF),
            n_ctx=1024,
            n_threads=os.cpu_count() or 4,
            verbose=False,
            logits_all=True,  # required for logprobs on the completion call
        )
    return _qwen_llm


def _qwen_prompt(text: str) -> str:
    return (
        "<|im_start|>system\nYou triage on-call infrastructure alerts. Answer with exactly one word.<|im_end|>\n"
        f"<|im_start|>user\nAlert: {text}\nShould an on-call engineer be paged right now? Answer YES or NO only.<|im_end|>\n"
        "<|im_start|>assistant\n"
    )


def qwen_predict(text: str) -> dict:
    llm = get_qwen()
    t0 = time.perf_counter()
    out = llm(_qwen_prompt(text), max_tokens=1, logprobs=10, temperature=0.0)
    latency_ms = (time.perf_counter() - t0) * 1000
    top_logprobs = out["choices"][0]["logprobs"]["top_logprobs"][0]

    def prob_of(prefix: str) -> float:
        best = max((lp for tok, lp in top_logprobs.items() if tok.strip().upper().startswith(prefix)), default=None)
        return math.exp(best) if best is not None else 0.0

    p_yes, p_no = prob_of("YES"), prob_of("NO")
    total = p_yes + p_no
    p_page = p_yes / total if total > 0 else 0.5
    return {
        "model": "qwen",
        "checkpoint": "Qwen2.5-1.5B-Instruct-Q4_K_M",
        "p_page": p_page,
        "severity": None,  # not asked - pilot only checks page/ignore for Qwen, per plan scope
        "team": None,
        "latency_ms": latency_ms,
        "raw": {"top_logprobs": top_logprobs, "p_yes": p_yes, "p_no": p_no},
    }


# ---------------------------------------------------------------------------
# TF-IDF + logistic regression (classic baseline, page/ignore only)
# ---------------------------------------------------------------------------

_tfidf_model = None  # (vectorizer, classifier), fit once per process


def train_tfidf_baseline(train_rows: list[dict], min_df: int = 2, class_weight: str | None = None):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression

    texts = [r["text"] for r in train_rows]
    labels = [1 if r["page"] else 0 for r in train_rows]
    vectorizer = TfidfVectorizer(max_features=20_000, ngram_range=(1, 2), min_df=min_df, sublinear_tf=True)
    X = vectorizer.fit_transform(texts)
    clf = LogisticRegression(max_iter=1000, C=1.0, class_weight=class_weight)
    clf.fit(X, labels)
    train_acc = clf.score(X, labels)
    print(f"TF-IDF+LogReg: fit on {len(train_rows)} rows, {len(vectorizer.vocabulary_)} features, "
          f"train accuracy={train_acc:.3f}", file=sys.stderr)
    return vectorizer, clf


def get_tfidf_model():
    if _tfidf_model is None:
        raise RuntimeError("TF-IDF model not trained yet - call train_tfidf_baseline() first")
    return _tfidf_model


def tfidf_predict(text: str) -> dict:
    vectorizer, clf = get_tfidf_model()
    t0 = time.perf_counter()
    X = vectorizer.transform([text])
    proba = clf.predict_proba(X)[0]
    latency_ms = (time.perf_counter() - t0) * 1000
    page_col = list(clf.classes_).index(1)
    return {
        "model": "tfidf",
        "checkpoint": "TF-IDF+LogisticRegression",
        "p_page": float(proba[page_col]),
        "severity": None,
        "team": None,
        "latency_ms": latency_ms,
        "raw": {"classes": [int(c) for c in clf.classes_], "proba": [float(v) for v in proba], "input": text},
    }


# ---------------------------------------------------------------------------
# Pilot: 20 alerts through every model, printed for review (checkpoint 2)
# ---------------------------------------------------------------------------


def _decision(row: dict) -> bool:
    """A BGL result row's scored decision: Haiku's stated boolean where present, else P(page) >= 0.5."""
    return decide(row["p_page"], row.get("decision"))


def _accuracy_counts(rows: list[dict]) -> tuple[int, int]:
    return accuracy_counts([_decision(r) for r in rows], [r["page"] for r in rows])


def diagnose_laya(checkpoint_key: str, rows: list[dict]) -> dict:
    """Checks 1 & 2 from the pilot-result review: exact question/raw output on true-page rows,
    and cold vs warm latency, page-only vs the full 3-question call. Must run before any other
    predict() call on this checkpoint in the process, or 'cold' isn't really cold."""
    print(f"\n=== diagnostic: {checkpoint_key} ===", file=sys.stderr)
    t0 = time.perf_counter()
    laya_agent.get_agent(checkpoint_key)  # load weights only, no forward pass yet
    load_s = time.perf_counter() - t0

    sample_text = rows[0]["text"]
    t0 = time.perf_counter()
    laya_predict_page(checkpoint_key, sample_text)  # first-ever forward pass for this checkpoint
    cold_ms = (time.perf_counter() - t0) * 1000

    # agent.predict() runs ALL questions passed to it in one forward pass (laya/agent.py
    # system_one docstring: "across state in a single, parallel forward pass" -- each question
    # is a row added to the same batched encode + _forward call, not a separate pass). So the
    # page-only vs full-3-question timing below should be close; a big gap would mean that
    # claim doesn't hold for this checkpoint/version.
    warm_page = [laya_predict_page(checkpoint_key, r["text"])["latency_ms"] for r in rows[:3]]
    warm_full = [laya_predict(checkpoint_key, r["text"])["latency_ms"] for r in rows[:3]]

    print(f"  load (weights only): {load_s:.2f}s")
    print(f"  cold call (page-only, first forward pass ever): {cold_ms:.0f}ms")
    print(f"  warm, page-only x3: {[f'{v:.0f}ms' for v in warm_page]}")
    print(f"  warm, full 3 questions x3: {[f'{v:.0f}ms' for v in warm_full]}")

    print(f"  question sent for 'page': {json.dumps(laya_page_question(checkpoint_key))}")
    for r in [r for r in rows if r["page"]][:3]:
        raw = laya_predict_page(checkpoint_key, r["text"])["raw_answers"]["page"]
        print(f"    true_page=True text={r['text'][:70]!r}")
        print(f"      raw answer: {raw}")

    return {
        "cold_ms": cold_ms,
        "warm_page_ms": sum(warm_page) / len(warm_page),
        "warm_full_ms": sum(warm_full) / len(warm_full),
    }


def pilot():
    rows = load_pilot_rows()
    n_page = sum(r["page"] for r in rows)
    print(f"pilot: {len(rows)} alerts ({n_page} page / {len(rows) - n_page} ignore)", file=sys.stderr)

    laya_diag: dict[str, dict] = {}
    for key in ["laya-base", "laya-typed"]:
        laya_diag[key] = diagnose_laya(key, rows)

    print("\nfitting page/ignore temperature on a separate calibration set...", file=sys.stderr)
    calib_rows = build_calibration_set()
    refit: dict[str, dict] = {}
    for key in ["laya-base", "laya-typed"]:
        refit[key] = fit_page_temperature(key, calib_rows)
        f = refit[key]
        print(f"  {key}: T shipped={f['t_shipped']:.3f} (nll={f['nll_shipped']:.3f} brier={f['brier_shipped']:.3f})  "
              f"-> T refit={f['t_refit']:.3f} (nll={f['nll_refit']:.3f} brier={f['brier_refit']:.3f})  n={f['n']}",
              file=sys.stderr)

    results: dict[str, list[tuple[dict, dict]]] = defaultdict(list)
    results_refit: dict[str, list[tuple[dict, dict]]] = defaultdict(list)

    for key in ["laya-base", "laya-typed"]:
        for r in rows:
            results[key].append((r, laya_predict(key, r["text"])))  # shipped temperature
        apply_page_temperature(key, refit[key]["t_refit"])
        for r in rows:
            results_refit[key].append((r, laya_predict(key, r["text"])))  # refit temperature
        apply_page_temperature(key, refit[key]["t_shipped"])  # restore - don't leak into other callers

    print("calling Haiku...", file=sys.stderr)
    for r in rows:
        results["haiku"].append((r, haiku_predict(r["text"])))

    print("timing Qwen on one alert...", file=sys.stderr)
    t0 = time.perf_counter()
    first_qwen = qwen_predict(rows[0]["text"])
    warm_dt = time.perf_counter() - t0
    # the first call includes model load; time a second, warm call for the real speed check
    t0 = time.perf_counter()
    qwen_predict(rows[1]["text"])
    dt = time.perf_counter() - t0
    print(f"Qwen: {dt:.2f}s/alert warm ({warm_dt:.2f}s including load)", file=sys.stderr)
    if dt < QWEN_SPEED_LIMIT_S:
        results["qwen"].append((rows[0], first_qwen))
        results["qwen"].append((rows[1], qwen_predict(rows[1]["text"])))
        for r in rows[2:]:
            results["qwen"].append((r, qwen_predict(r["text"])))
    else:
        print(f"Qwen dropped: {dt:.2f}s/alert >= {QWEN_SPEED_LIMIT_S}s limit", file=sys.stderr)

    def print_table(model: str, pairs: list[tuple[dict, dict]], label: str = ""):
        labels = [1.0 if r["page"] else 0.0 for r, _ in pairs]
        preds = [p["p_page"] for _, p in pairs]
        lat = [p["latency_ms"] for _, p in pairs]
        acc = sum(_decision(p) == bool(r["page"]) for r, p in pairs) / len(pairs)
        checkpoint = pairs[0][1]["checkpoint"]
        print(f"\n=== {model}{label} ({checkpoint}) ===")
        print(
            f"n={len(pairs)}  accuracy={acc:.2f}  brier={brier(preds, labels):.3f}  "
            f"p50={percentile(lat, 0.5):.0f}ms  p95={percentile(lat, 0.95):.0f}ms"
        )
        for r, p in pairs:
            mark = "OK" if _decision(p) == r["page"] else "XX"
            sev = p["severity"] or "-"
            team = p["team"] or "-"
            print(
                f"  [{mark}] true_page={r['page']!s:5} p_page={p['p_page']:.2f} "
                f"sev={sev:<6} team={team:<8} {r['text'][:60]}"
            )

    for key in ["laya-base", "laya-typed"]:
        d = laya_diag[key]
        f = refit[key]
        print(f"\n--- {key} latency: cold={d['cold_ms']:.0f}ms  warm(page-only)={d['warm_page_ms']:.0f}ms  "
              f"warm(3 questions)={d['warm_full_ms']:.0f}ms  |  temperature: shipped={f['t_shipped']:.3f} "
              f"-> refit={f['t_refit']:.3f} (fit on {f['n']} held-out rows, brier {f['brier_shipped']:.3f} "
              f"-> {f['brier_refit']:.3f}) ---")
        print_table(key, results[key], label=" [raw/shipped-T]")
        print_table(key, results_refit[key], label=" [refit-T]")

    for model in ["haiku", "qwen"]:
        if model in results:
            print_table(model, results[model])


# ---------------------------------------------------------------------------
# Full run: 200 alerts through every model, written to results.jsonl (step 3)
# ---------------------------------------------------------------------------

RESULTS_PATH = Path(__file__).parent / "results.jsonl"


def run():
    rows = load_alerts_file()
    n_page = sum(r["page"] for r in rows)
    print(f"run: {len(rows)} alerts ({n_page} page / {len(rows) - n_page} ignore)", file=sys.stderr)

    calib_rows, tfidf_rows = build_calibration_and_tfidf_sets(calib_n_per_class=50, tfidf_n=5000)

    print("training TF-IDF+LogReg baseline...", file=sys.stderr)
    global _tfidf_model
    _tfidf_model = train_tfidf_baseline(tfidf_rows)

    refit: dict[str, dict] = {}
    for key in ["laya-base", "laya-typed"]:
        laya_agent.get_agent(key)
        refit[key] = fit_page_temperature(key, calib_rows)
        f = refit[key]
        print(f"  {key}: T shipped={f['t_shipped']:.3f} (brier={f['brier_shipped']:.3f}) -> "
              f"T refit={f['t_refit']:.3f} (brier={f['brier_refit']:.3f})  n={f['n']}", file=sys.stderr)

    out_rows = predict_laya(rows, refit) + predict_haiku(rows) + predict_qwen(rows) + predict_tfidf(rows)
    write_results(RESULTS_PATH, out_rows)
    print_summary(out_rows, f"n={len(rows)} eval set")


# ---------------------------------------------------------------------------
# Shared: run every model over an eval set -> results rows (one per model x alert)
# ---------------------------------------------------------------------------


def _result_row(model: str, r: dict, pred: dict, sev_team: dict | None = None) -> dict:
    row = {
        "model": model,
        "checkpoint": pred["checkpoint"],
        "alert_id": r["id"],
        "text": r["text"],
        "page": r["page"],
        "p_page": pred["p_page"],
        "decision": pred.get("decision"),  # set only by models that state a boolean (Haiku)
        "severity": pred.get("severity"),
        "team": pred.get("team"),
        "severity_probs": None,
        "team_probs": None,
        "latency_ms": pred["latency_ms"],  # the scored page/ignore decision only
        "sev_team_latency_ms": None,  # timed separately, not part of any score
        "raw": pred.get("raw"),
    }
    if sev_team is not None:
        ans = sev_team["raw_answers"]
        row.update(severity=sev_team["severity"], team=sev_team["team"],
                   severity_probs=ans["severity"]["probabilities"], team_probs=ans["team"]["probabilities"],
                   sev_team_latency_ms=sev_team["latency_ms"])
        row["raw"] = {"page": row["raw"], "sev_team": ans}
    return row


def predict_laya(rows: list[dict], refit: dict[str, dict]) -> list[dict]:
    out = []
    for key in ["laya-base", "laya-typed"]:
        print(f"running {key} on {len(rows)} alerts...", file=sys.stderr)
        for i, r in enumerate(rows):
            sev_team = laya_predict_sev_team(key, r["text"])
            if key == "laya-base":
                page_raw = laya_predict_page(key, r["text"])
                page_raw["raw"] = {"answers": page_raw["raw_answers"], "page_temperature": refit[key]["t_shipped"]}
                out.append(_result_row("laya-base-raw", r, page_raw, sev_team))
            apply_page_temperature(key, refit[key]["t_refit"])
            page_refit = laya_predict_page(key, r["text"])
            apply_page_temperature(key, refit[key]["t_shipped"])
            page_refit["raw"] = {"answers": page_refit["raw_answers"], "page_temperature": refit[key]["t_refit"]}
            out.append(_result_row("laya-base-refit" if key == "laya-base" else "laya-typed", r, page_refit, sev_team))
            if (i + 1) % 50 == 0:
                print(f"  {key}: {i + 1}/{len(rows)}", file=sys.stderr)
    return out


def predict_haiku(rows: list[dict]) -> list[dict]:
    print(f"running Haiku on {len(rows)} alerts...", file=sys.stderr)
    out = []
    for i, r in enumerate(rows):
        out.append(_result_row("haiku", r, haiku_predict(r["text"])))
        if (i + 1) % 50 == 0:
            print(f"  haiku: {i + 1}/{len(rows)}", file=sys.stderr)
    return out


def predict_qwen(rows: list[dict]) -> list[dict]:
    print("timing Qwen...", file=sys.stderr)
    t0 = time.perf_counter()
    first = qwen_predict(rows[0]["text"])
    cold_dt = time.perf_counter() - t0
    t0 = time.perf_counter()
    second = qwen_predict(rows[1]["text"])
    dt = time.perf_counter() - t0
    print(f"Qwen: {dt:.2f}s/alert warm ({cold_dt:.2f}s including load)", file=sys.stderr)
    if dt >= QWEN_SPEED_LIMIT_S:
        print(f"Qwen dropped: {dt:.2f}s/alert >= {QWEN_SPEED_LIMIT_S}s limit", file=sys.stderr)
        return []
    out = [_result_row("qwen", rows[0], first), _result_row("qwen", rows[1], second)]
    for i, r in enumerate(rows[2:], start=2):
        out.append(_result_row("qwen", r, qwen_predict(r["text"])))
        if (i + 1) % 50 == 0:
            print(f"  qwen: {i + 1}/{len(rows)}", file=sys.stderr)
    return out


def predict_tfidf(rows: list[dict], transform=lambda text: text) -> list[dict]:
    print("running TF-IDF+LogReg...", file=sys.stderr)
    return [_result_row("tfidf", r, tfidf_predict(transform(r["text"]))) for r in rows]


def write_results(path: Path, out_rows: list[dict]) -> None:
    path.write_text("\n".join(json.dumps(row) for row in out_rows) + "\n")
    print(f"wrote {len(out_rows)} result rows to {path}", file=sys.stderr)


def print_summary(out_rows: list[dict], title: str) -> None:
    by_model: dict[str, list[dict]] = defaultdict(list)
    for row in out_rows:
        by_model[row["model"]].append(row)
    print(f"\n=== summary ({title}, page/ignore decision only) ===")
    for model in [m for m in PLOT_ORDER if m in by_model]:
        model_rows = by_model[model]
        preds = [r["p_page"] for r in model_rows]
        labels = [1.0 if r["page"] else 0.0 for r in model_rows]
        lat = [r["latency_ms"] for r in model_rows]
        k, n = _accuracy_counts(model_rows)
        lo, hi = wilson_ci(k, n)
        print(f"{model:16s} n={n:3d}  accuracy={k / n:.3f}  95% CI=[{lo:.3f},{hi:.3f}]  "
              f"brier={brier(preds, labels):.3f}  ece={ece(preds, labels):.3f}  "
              f"p50={percentile(lat, 0.5):.0f}ms  p95={percentile(lat, 0.95):.0f}ms")


# ---------------------------------------------------------------------------
# Plot: calibration (reliability) curves + latency, from results.jsonl
# ---------------------------------------------------------------------------

PLOT_LABELS = {
    "laya-base-raw": "Laya base (raw)",
    "laya-base-refit": "Laya base (refit)",
    "laya-typed": "Laya typed-decisions",
    "haiku": "Claude Haiku 4.5",
    "qwen": "Qwen2.5-1.5B",
    "tfidf": "TF-IDF + LogReg",
}
PLOT_ORDER = list(PLOT_LABELS)


def load_results() -> dict[str, list[dict]]:
    if not RESULTS_PATH.exists():
        raise SystemExit(f"missing {RESULTS_PATH} - run 'python triage.py run' first")
    by_model: dict[str, list[dict]] = defaultdict(list)
    for line in RESULTS_PATH.read_text().splitlines():
        if not line:
            continue
        row = json.loads(line)
        by_model[row["model"]].append(row)
    return by_model


def plot():
    by_model = load_results()
    n_eval = max((len(rows) for rows in by_model.values()), default=0)
    series = []
    for model in [m for m in PLOT_ORDER if m in by_model]:
        rows = by_model[model]
        k, n = _accuracy_counts(rows)
        series.append({
            "label": PLOT_LABELS[model],
            "preds": [r["p_page"] for r in rows],
            "labels": [1.0 if r["page"] else 0.0 for r in rows],
            "latencies": [r["latency_ms"] for r in rows],
            "accuracy": k / n,
        })
    reliability_and_latency(series, Path(__file__).parent / "calibration.png",
                            title=f"Calibration (reliability curve), n={n_eval}",
                            x_label="predicted P(page)", y_label="observed page rate",
                            latency_label="latency (ms, log scale) - page/ignore decision only")

    print_summary([row for rows in by_model.values() for row in rows], f"n={n_eval} eval set")


# ---------------------------------------------------------------------------
# Serve: live web view, replaying results.jsonl over SSE (step 3)
# ---------------------------------------------------------------------------

SERVE_HOST, SERVE_PORT = "127.0.0.1", 8420
SERVE_DELAY_S = 0.35  # pacing between alerts in the simulated "live" stream


def _load_results_by_alert():
    by_model = load_results()
    models = [m for m in PLOT_ORDER if m in by_model]
    by_alert: dict[int, dict] = {}
    for model in models:
        for r in by_model[model]:
            aid = r["alert_id"]
            entry = by_alert.setdefault(aid, {"alert_id": aid, "text": r["text"], "page": r["page"], "models": {}})
            entry["models"][model] = {
                "p_page": r["p_page"],
                "decision": r.get("decision"),
                "severity": r["severity"],
                "team": r["team"],
                "severity_probs": r["severity_probs"],
                "team_probs": r["team_probs"],
                "latency_ms": r["latency_ms"],
                "sev_team_latency_ms": r["sev_team_latency_ms"],
            }
    alert_ids = sorted(by_alert)
    return alert_ids, by_alert, models


SERVE_HTML = """<!doctype html>
<html>
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Laya on-call triage - live view</title>
<style>
  :root { color-scheme: light dark; }
  body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; margin: 0;
         background: #0f1115; color: #e6e8eb; }
  header { padding: 16px 20px; border-bottom: 1px solid #262a33; display: flex; align-items: center;
           gap: 16px; flex-wrap: wrap; position: sticky; top: 0; background: #0f1115; z-index: 2; }
  header h1 { font-size: 16px; margin: 0; font-weight: 600; color: #e6e8eb; }
  select, button { background: #1a1d24; color: #e6e8eb; border: 1px solid #353a46; border-radius: 6px;
                   padding: 6px 10px; font-size: 13px; }
  button { cursor: pointer; }
  button:hover { background: #242833; }
  .stats { display: flex; gap: 18px; font-size: 13px; color: #9aa2b1; margin-left: auto; flex-wrap: wrap; }
  .stats b { color: #e6e8eb; }
  main { max-width: 820px; margin: 0 auto; padding: 16px 20px 60px; }
  .card { background: #171a21; border: 1px solid #262a33; border-radius: 10px; padding: 14px 16px;
          margin-bottom: 10px; animation: in 0.25s ease-out; }
  @keyframes in { from { opacity: 0; transform: translateY(-6px); } to { opacity: 1; transform: none; } }
  .text { font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: 12.5px; color: #cdd3dd;
          margin-bottom: 10px; word-break: break-word; }
  .row { display: flex; align-items: center; gap: 8px; margin: 5px 0; font-size: 12px; color: #9aa2b1; }
  .row .label { width: 56px; flex-shrink: 0; }
  .bar-track { flex: 1; height: 14px; background: #262a33; border-radius: 7px; overflow: hidden;
               display: flex; }
  .bar-fill { height: 100%; background: linear-gradient(90deg, #4f8cff, #7aa8ff); border-radius: 7px 0 0 7px;
              transition: width 0.3s ease; }
  .bar-fill.page-true { background: linear-gradient(90deg, #e5484d, #ff7a7f); }
  .pct { width: 50px; text-align: right; font-variant-numeric: tabular-nums; }
  .chip { display: inline-block; padding: 2px 8px; border-radius: 999px; background: #262a33;
          font-size: 11px; color: #cdd3dd; margin-right: 4px; }
  .multibar { flex: 1; display: flex; gap: 2px; height: 14px; }
  .multibar .seg { background: #2d3340; border-radius: 3px; position: relative; display: flex;
                   align-items: center; justify-content: center; font-size: 9px; color: #0f1115;
                   overflow: hidden; }
  .multibar .seg.top { background: #4f8cff; color: #fff; }
  .verdict { font-size: 11px; font-weight: 600; margin-left: 4px; }
  .verdict.ok { color: #3ddc84; }
  .verdict.miss { color: #ff7a7f; }
  .lat { font-size: 11px; color: #828ba0; margin-top: 8px; }
  footer { text-align: center; color: #5c6372; font-size: 12px; padding: 20px; }
</style>
</head>
<body>
<header>
  <h1>Laya on-call triage &mdash; live view</h1>
  <select id="model"></select>
  <button id="restart">restart</button>
  <div class="stats">
    <span>n <b id="s-n">0</b></span>
    <span>accuracy <b id="s-acc">-</b></span>
    <span>avg latency <b id="s-lat">-</b></span>
  </div>
</header>
<main id="feed"></main>
<footer>replaying results.jsonl from `triage.py run` &middot; page/ignore probability bar is the scored decision &middot; severity/team timed separately</footer>
<script>
const MODELS = __MODELS_JSON__;
const sel = document.getElementById("model");
MODELS.forEach(m => { const o = document.createElement("option"); o.value = m.id; o.textContent = m.label; sel.appendChild(o); });

let es = null, n = 0, correct = 0, latSum = 0;

function pct(x) { return Math.round(x * 100) + "%"; }

function bars(probs, chosen) {
  if (!probs) return `<span class="chip">${chosen ?? "-"}</span>`;
  const entries = Object.entries(probs);
  const maxP = Math.max(...entries.map(e => e[1]));
  return `<div class="multibar">` + entries.map(([k, v]) => {
    const isTop = v === maxP;
    return `<div class="seg${isTop ? " top" : ""}" style="flex:${Math.max(v, 0.03)}" title="${k}: ${pct(v)}">${isTop ? k : ""}</div>`;
  }).join("") + `</div>`;
}

function decided(d) { return d.decision ?? (d.p_page >= 0.5); }

function card(d) {
  const el = document.createElement("div");
  el.className = "card";
  const predictedPage = decided(d);
  const ok = predictedPage === d.page;
  el.innerHTML = `
    <div class="text">${d.text.replace(/</g, "&lt;")}</div>
    <div class="row"><span class="label">page</span>
      <div class="bar-track"><div class="bar-fill${predictedPage ? " page-true" : ""}" style="width:${pct(d.p_page)}"></div></div>
      <span class="pct">${(d.p_page * 100).toFixed(1)}%</span>
      <span class="verdict ${ok ? "ok" : "miss"}">${ok ? "OK" : "XX"} true=${d.page}</span>
    </div>
    <div class="row"><span class="label">severity</span>${bars(d.severity_probs, d.severity)}</div>
    <div class="row"><span class="label">team</span>${bars(d.team_probs, d.team)}</div>
    <div class="lat">latency: page ${Math.round(d.latency_ms)}ms${d.sev_team_latency_ms != null ? ` &middot; severity+team ${Math.round(d.sev_team_latency_ms)}ms` : ""}</div>
  `;
  return el;
}

function start() {
  if (es) es.close();
  document.getElementById("feed").innerHTML = "";
  n = 0; correct = 0; latSum = 0;
  updateStats();
  es = new EventSource(`/stream?model=${encodeURIComponent(sel.value)}`);
  es.onmessage = (ev) => {
    const d = JSON.parse(ev.data);
    n++; latSum += d.latency_ms;
    if (decided(d) === d.page) correct++;
    updateStats();
    const feed = document.getElementById("feed");
    feed.insertBefore(card(d), feed.firstChild);
  };
  es.addEventListener("done", () => es.close());
}

function updateStats() {
  document.getElementById("s-n").textContent = n;
  document.getElementById("s-acc").textContent = n ? pct(correct / n) : "-";
  document.getElementById("s-lat").textContent = n ? Math.round(latSum / n) + "ms" : "-";
}

sel.addEventListener("change", start);
document.getElementById("restart").addEventListener("click", start);
start();
</script>
</body>
</html>
"""


def serve():
    import asyncio

    import uvicorn
    from fastapi import FastAPI
    from fastapi.responses import HTMLResponse, StreamingResponse

    alert_ids, by_alert, models = _load_results_by_alert()
    if not models:
        raise SystemExit(f"no models found in {RESULTS_PATH} - run 'python triage.py run' first")
    app = FastAPI()

    @app.get("/", response_class=HTMLResponse)
    def index():
        models_json = json.dumps([{"id": m, "label": PLOT_LABELS[m]} for m in models])
        return SERVE_HTML.replace("__MODELS_JSON__", models_json)

    @app.get("/stream")
    async def stream(model: str = models[0]):
        async def gen():
            for aid in alert_ids:
                row = by_alert[aid]
                m = row["models"].get(model)
                if m is None:
                    continue
                payload = {"alert_id": aid, "text": row["text"], "page": row["page"], **m}
                yield f"data: {json.dumps(payload)}\n\n"
                await asyncio.sleep(SERVE_DELAY_S)
            yield "event: done\ndata: {}\n\n"
        return StreamingResponse(gen(), media_type="text/event-stream")

    print(f"serving on http://{SERVE_HOST}:{SERVE_PORT}", file=sys.stderr)
    uvicorn.run(app, host=SERVE_HOST, port=SERVE_PORT)


# ---------------------------------------------------------------------------
# Checks: TF-IDF template-leakage retrain + 95% CIs on every model's accuracy
# ---------------------------------------------------------------------------

_MAC_RE = re.compile(r"\b(?:[0-9A-Fa-f]{2}:){3,}[0-9A-Fa-f]{2}\b")
_HEX0X_RE = re.compile(r"0[xX][0-9A-Fa-f]+")
_NUM_RE = re.compile(r"\d+")


def mask_template(text: str) -> str:
    """Collapse MAC/hex/numeric identifiers (node IDs, job IDs, IPs, timestamps-as-numbers) to
    placeholders, so messages that differ only in which node or job they're about collapse to
    the same template string. BGL repeats a small number of templates thousands of times with
    different IDs filled in, so exact-text dedup alone doesn't stop a bag-of-words model from
    having effectively seen a test template's wording during training."""
    text = _MAC_RE.sub("<HEX>", text)
    text = _HEX0X_RE.sub("<HEX>", text)
    text = _NUM_RE.sub("<NUM>", text)
    return text


def tfidf_leakage_check():
    """Retrain TF-IDF+LogReg on masked text, excluding any training candidate whose masked
    template matches a masked eval-row template (not just exact-text duplicates). Also reports
    how much of the ORIGINAL training set (from `run()`) shared a masked template with an eval
    row, to quantify how much of the 0.945 accuracy could have been template memorization."""
    eval_rows = load_alerts_file()
    eval_templates = {mask_template(r["text"]) for r in eval_rows}
    existing = {r["text"] for r in eval_rows}

    dup_templates = Counter(mask_template(r["text"]) for r in eval_rows)
    n_dup = sum(c for c in dup_templates.values() if c > 1)
    if n_dup:
        print(f"note: {n_dup}/{len(eval_rows)} eval rows share a masked template with another eval row "
              f"(same alert type, different node/job ID) - they aren't fully independent test cases",
              file=sys.stderr)

    print("\nrecomputing the ORIGINAL 5000-row TF-IDF training set to measure template overlap...",
          file=sys.stderr)
    _, original_train_rows = build_calibration_and_tfidf_sets(calib_n_per_class=50, tfidf_n=5000)
    leaked = [r for r in original_train_rows if mask_template(r["text"]) in eval_templates]
    print(f"original TF-IDF training set: {len(leaked)}/{len(original_train_rows)} rows share a masked "
          f"template with an eval-set row", file=sys.stderr)

    alert_msgs, nonalert_msgs, msg_meta, msg_category, msg_raw = _scan_and_clean()
    alert_msgs = [m for m in alert_msgs if m not in existing and mask_template(m) not in eval_templates]
    nonalert_msgs = [m for m in nonalert_msgs if m not in existing and mask_template(m) not in eval_templates]
    rng = random.Random(SEED + 3)
    n_each = 2500
    picked_alerts = rng.sample(alert_msgs, min(n_each, len(alert_msgs)))
    picked_nonalerts = rng.sample(nonalert_msgs, min(n_each, len(nonalert_msgs)))
    train_rows = (
        _rows_from_messages(picked_alerts, True, msg_meta, msg_category, msg_raw)
        + _rows_from_messages(picked_nonalerts, False, msg_meta, msg_category, msg_raw)
    )
    print(f"template-safe TF-IDF training set: {len(train_rows)} rows ({len(picked_alerts)} page / "
          f"{len(picked_nonalerts)} ignore), no shared masked template with any eval row", file=sys.stderr)

    masked_train = [{"text": mask_template(r["text"]), "page": r["page"]} for r in train_rows]
    vectorizer, clf = train_tfidf_baseline(masked_train)

    preds, labels = [], []
    for r in eval_rows:
        X = vectorizer.transform([mask_template(r["text"])])
        preds.append(float(clf.predict_proba(X)[0, 1]))
        labels.append(1.0 if r["page"] else 0.0)
    return preds, labels


def checks():
    by_model = load_results()
    n_eval = max((len(rows) for rows in by_model.values()), default=0)
    print(f"\n=== 95% CI on accuracy (Wilson score interval, n={n_eval}) ===")
    for model in PLOT_ORDER:
        if model not in by_model:
            continue
        k, n = _accuracy_counts(by_model[model])
        lo, hi = wilson_ci(k, n)
        print(f"{PLOT_LABELS[model]:24s} accuracy={k / n:.3f}  95% CI=[{lo:.3f}, {hi:.3f}]  (k={k}/{n})")

    print("\n=== TF-IDF leakage check (masked numeric/hex/MAC IDs, template-level train/eval exclusion) ===")
    preds, labels = tfidf_leakage_check()
    k = sum((p >= 0.5) == bool(y) for p, y in zip(preds, labels))
    n = len(labels)
    lo, hi = wilson_ci(k, n)
    print(f"{'TF-IDF (masked, leak-checked)':24s} accuracy={k / n:.3f}  95% CI=[{lo:.3f}, {hi:.3f}]  (k={k}/{n})  "
          f"brier={brier(preds, labels):.3f}  ece={ece(preds, labels):.3f}")


# ---------------------------------------------------------------------------
# Rebuild the eval set at template level: mask numbers/hex/MACs, one row per
# masked template, so train/calibration/eval are disjoint by template, not just by exact text.
# ---------------------------------------------------------------------------

TEMPLATE_MIN_PER_CLASS = 50   # eligibility floor (per the spec: below this, fall back to row-level + macro)
TEMPLATE_EVAL_CAP = 100       # the requested ceiling, reachable only if enough templates exist
TEMPLATE_EVAL_N = 50          # actual per-class eval size once a calibration/training reserve is set aside
TEMPLATE_CALIB_N = 15         # per class, template-disjoint from eval
NONALERT_TRAIN_TEMPLATE_CAP = 2000  # non-alert templates are abundant; cap training volume for speed


def _dedupe_to_templates(msgs: list[str]) -> dict[str, str]:
    """masked_template -> first exact message text seen for that template (deterministic order)."""
    reps: dict[str, str] = {}
    for m in msgs:
        reps.setdefault(mask_template(m), m)
    return reps


def _report_template_macro_accuracy():
    """Fallback when one class has <50 templates: keep the row-level 200-alert eval set, but also
    report template-macro accuracy (average per-template accuracy, each template weighted equally)
    next to row-level accuracy (each row weighted equally, so a repeated template dominates)."""
    by_model = load_results()
    print("\n=== template-level macro accuracy vs row-level accuracy (200-alert eval set) ===")
    for model in PLOT_ORDER:
        if model not in by_model:
            continue
        rows = by_model[model]
        by_template: dict[str, list[dict]] = defaultdict(list)
        for r in rows:
            by_template[mask_template(r["text"])].append(r)
        template_accs = [k / n for k, n in map(_accuracy_counts, by_template.values())]
        macro_acc = sum(template_accs) / len(template_accs)
        k, n = _accuracy_counts(rows)
        row_acc = k / n
        print(f"{model:16s} row-level accuracy={row_acc:.3f}  template-macro accuracy={macro_acc:.3f}  "
              f"({len(by_template)} distinct templates among {len(rows)} rows)")


def template_split():
    """Deterministic template-level eval/calibration/training split. Returns None when either
    class has fewer than TEMPLATE_MIN_PER_CLASS templates."""
    alert_msgs, nonalert_msgs, msg_meta, msg_category, msg_raw = _scan_and_clean()
    alert_templates = _dedupe_to_templates(alert_msgs)
    nonalert_templates = _dedupe_to_templates(nonalert_msgs)
    print(f"\nunique masked templates in the full dataset: {len(alert_templates)} alert, "
          f"{len(nonalert_templates)} non-alert", file=sys.stderr)

    if len(alert_templates) < TEMPLATE_MIN_PER_CLASS or len(nonalert_templates) < TEMPLATE_MIN_PER_CLASS:
        return None

    rng = random.Random(SEED + 10)
    alert_keys = list(alert_templates)
    nonalert_keys = list(nonalert_templates)
    rng.shuffle(alert_keys)
    rng.shuffle(nonalert_keys)

    # Eval is capped at 100 per the spec, but reaching that cap would leave nothing for a
    # template-disjoint calibration split (there are only 97 alert templates in all of BGL) -
    # so eval is sized at min(cap, target, templates left after reserving 2x the calibration
    # budget for train+calib). See README.md for why this matters.
    eval_n = min(TEMPLATE_EVAL_CAP, TEMPLATE_EVAL_N, len(alert_keys) - 2 * TEMPLATE_CALIB_N)
    if eval_n < 1:
        raise SystemExit(f"not enough alert templates ({len(alert_keys)}) to reserve a calibration split")

    eval_alert_keys = alert_keys[:eval_n]
    calib_alert_keys = alert_keys[eval_n:eval_n + TEMPLATE_CALIB_N]
    train_alert_keys = alert_keys[eval_n + TEMPLATE_CALIB_N:]

    eval_nonalert_keys = nonalert_keys[:eval_n]
    calib_nonalert_keys = nonalert_keys[eval_n:eval_n + TEMPLATE_CALIB_N]
    train_nonalert_keys = nonalert_keys[eval_n + TEMPLATE_CALIB_N:eval_n + TEMPLATE_CALIB_N + NONALERT_TRAIN_TEMPLATE_CAP]

    print(f"split: eval={eval_n}+{eval_n}  calibration={TEMPLATE_CALIB_N}+{TEMPLATE_CALIB_N}  "
          f"training={len(train_alert_keys)} alert + {len(train_nonalert_keys)} non-alert templates",
          file=sys.stderr)

    def build_rows(keys, templates_dict, page):
        msgs = [templates_dict[k] for k in keys]
        return _rows_from_messages(msgs, page, msg_meta, msg_category, msg_raw)

    eval_rows = (
        build_rows(eval_alert_keys, alert_templates, True)
        + build_rows(eval_nonalert_keys, nonalert_templates, False)
    )
    rng.shuffle(eval_rows)
    for i, row in enumerate(eval_rows):
        row["id"] = i

    calib_rows = (
        build_rows(calib_alert_keys, alert_templates, True)
        + build_rows(calib_nonalert_keys, nonalert_templates, False)
    )
    rng.shuffle(calib_rows)
    for i, row in enumerate(calib_rows):
        row["id"] = i

    # Template-level training: one MASKED row per remaining template, not one row per exact
    # message - after masking, every exact-text instance of a template is byte-identical, so
    # repeating them would just reweight the model toward whichever template logged more variants.
    train_rows = (
        [{"text": mask_template(alert_templates[k]), "page": True} for k in train_alert_keys]
        + [{"text": mask_template(nonalert_templates[k]), "page": False} for k in train_nonalert_keys]
    )
    print(f"TF-IDF template-level training set: {len(train_rows)} rows "
          f"({len(train_alert_keys)} page / {len(train_nonalert_keys)} ignore)", file=sys.stderr)
    return eval_rows, calib_rows, train_rows


def rebuild_template_eval():
    split = template_split()
    if split is None:
        print("fewer than 50 templates in one class - keeping the row-level 200-alert eval set and "
              "reporting template-macro accuracy instead of rebuilding", file=sys.stderr)
        _report_template_macro_accuracy()
        return
    eval_rows, calib_rows, train_rows = split

    for path in (ALERTS_PATH, RESULTS_PATH):
        backup = Path(str(path) + ".rowlevel200.bak")
        if path.exists() and not backup.exists():
            backup.write_text(path.read_text())
            print(f"backed up {path.name} -> {backup.name}", file=sys.stderr)

    ALERTS_PATH.write_text("\n".join(json.dumps(r) for r in eval_rows) + "\n")
    print(f"wrote {len(eval_rows)} template-level eval alerts to {ALERTS_PATH}", file=sys.stderr)

    global _tfidf_model
    _tfidf_model = train_tfidf_baseline(train_rows, min_df=1, class_weight="balanced")

    refit: dict[str, dict] = {}
    for key in ["laya-base", "laya-typed"]:
        laya_agent.get_agent(key)
        refit[key] = fit_page_temperature(key, calib_rows)
        f = refit[key]
        print(f"  {key}: T shipped={f['t_shipped']:.3f} (brier={f['brier_shipped']:.3f}) -> "
              f"T refit={f['t_refit']:.3f} (brier={f['brier_refit']:.3f})  n={f['n']}", file=sys.stderr)

    out_rows = (
        predict_laya(eval_rows, refit) + predict_haiku(eval_rows) + predict_qwen(eval_rows)
        + predict_tfidf(eval_rows, transform=mask_template)
    )
    write_results(RESULTS_PATH, out_rows)
    print_summary(out_rows, f"n={len(eval_rows)} template-level eval set")


# ---------------------------------------------------------------------------
# Re-run only Haiku (direct p_page + page boolean, raw response stored) on both eval sets
# ---------------------------------------------------------------------------

ROWLEVEL_ALERTS_BACKUP = Path(str(ALERTS_PATH) + ".rowlevel200.bak")
ROWLEVEL_RESULTS_BACKUP = Path(str(RESULTS_PATH) + ".rowlevel200.bak")
ROWLEVEL_RESULTS_PATH = Path(__file__).parent / "results_rowlevel200.jsonl"


def _load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def print_confusion(out_rows: list[dict], title: str) -> None:
    by_model: dict[str, list[dict]] = defaultdict(list)
    for row in out_rows:
        by_model[row["model"]].append(row)
    print(f"\n=== confusion matrix ({title}) ===")
    for model in [m for m in PLOT_ORDER if m in by_model]:
        rows = by_model[model]
        c = confusion_counts([_decision(r) for r in rows], [r["page"] for r in rows])
        print(f"{model:16s} TP={c['TP']:3d} FP={c['FP']:3d} TN={c['TN']:3d} FN={c['FN']:3d}  (n={len(rows)})")


def rehaiku():
    eval_sets = [
        ("template-level, n=100", ALERTS_PATH, RESULTS_PATH, RESULTS_PATH),
        # The original row-level results stay untouched in the .bak as provenance; the updated
        # copy (all other models unchanged, Haiku replaced) goes to its own file.
        ("row-level, n=200", ROWLEVEL_ALERTS_BACKUP, ROWLEVEL_RESULTS_BACKUP, ROWLEVEL_RESULTS_PATH),
    ]
    for title, alerts_path, src_results, dst_results in eval_sets:
        print(f"\n##### {title}: {alerts_path.name} #####", file=sys.stderr)
        alerts = _load_jsonl(alerts_path)
        other_models = [r for r in _load_jsonl(src_results) if r["model"] != "haiku"]
        haiku_rows = predict_haiku(alerts)
        rows = other_models + haiku_rows
        write_results(dst_results, rows)

        contradictions = [r for r in haiku_rows
                          if (r["decision"] and r["p_page"] < 0.5) or (not r["decision"] and r["p_page"] > 0.5)]
        ties = [r for r in haiku_rows if r["p_page"] == 0.5]
        print(f"\n=== Haiku page boolean vs p_page ({title}) ===")
        print(f"contradictions (page=true & p_page<50, or page=false & p_page>50): "
              f"{len(contradictions)}/{len(haiku_rows)}   p_page exactly 50: {len(ties)}")
        for r in contradictions[:5]:
            print(f"  raw={r['raw']['response_text']}  text={r['text'][:70]!r}")

        print_confusion(rows, title)
        print_summary(rows, title)


# ---------------------------------------------------------------------------
# Re-word Laya's page question: choose on the calibration set only, then run once on both eval sets
# ---------------------------------------------------------------------------


def relaya():
    split = template_split()
    if split is None:
        raise SystemExit("template split unavailable (fewer than 50 templates per class)")
    _, calib_rows, _ = split
    calib_templates = {mask_template(r["text"]) for r in calib_rows}
    rowlevel_templates = {mask_template(r["text"]) for r in _load_jsonl(ROWLEVEL_ALERTS_BACKUP)}
    print(f"calibration templates also present in the row-level 200 eval set: "
          f"{len(calib_templates & rowlevel_templates)}/{len(calib_templates)}", file=sys.stderr)

    # Selection rule, fixed before any eval-set result is seen: lowest calibration-set log loss
    # at each wording's own refit temperature (the configuration that is evaluated).
    selection: dict[str, dict] = {}
    for key in ["laya-base", "laya-typed"]:
        laya_agent.get_agent(key)
        candidates = {"legacy": build_questions()["page"]} | {w: page_wording_question(w) for w in LAYA_PAGE_WORDINGS}
        scores = {}
        for wid, question in candidates.items():
            f = fit_page_temperature(key, calib_rows, page_question=question)
            scores[wid] = {k: f[k] for k in ("t_shipped", "t_refit", "nll_shipped", "nll_refit",
                                             "brier_shipped", "brier_refit", "n")}
            print(f"  {key} {wid}: nll shipped-T={f['nll_shipped']:.4f}  nll refit-T={f['nll_refit']:.4f} "
                  f"(T={f['t_refit']:.3f})  brier refit-T={f['brier_refit']:.4f}", file=sys.stderr)
        chosen = min(LAYA_PAGE_WORDINGS, key=lambda w: scores[w]["nll_refit"])  # legacy is a reference, not a candidate
        selection[key] = {"chosen": chosen, "scores": scores}
        print(f"  {key}: chose {chosen}", file=sys.stderr)

    LAYA_WORDING_PATH.write_text(json.dumps({
        "selection_rule": "lowest calibration-set NLL at each wording's own refit temperature; "
                          "'legacy' (original two-clause question) is shown for reference only",
        "criteria": LAYA_PAGE_CRITERIA,
        "wordings": LAYA_PAGE_WORDINGS | {"legacy": build_questions()["page"]["instructions"]},
        **selection,
    }, indent=2) + "\n")
    print(f"wrote {LAYA_WORDING_PATH}", file=sys.stderr)

    refit = {key: {"t_shipped": s["scores"][s["chosen"]]["t_shipped"], "t_refit": s["scores"][s["chosen"]]["t_refit"]}
             for key, s in selection.items()}
    for title, alerts_path, results_path in [
        ("template-level, n=100", ALERTS_PATH, RESULTS_PATH),
        ("row-level, n=200", ROWLEVEL_ALERTS_BACKUP, ROWLEVEL_RESULTS_PATH),
    ]:
        print(f"\n##### {title} #####", file=sys.stderr)
        alerts = _load_jsonl(alerts_path)
        others = [r for r in _load_jsonl(results_path) if not r["model"].startswith("laya")]
        rows = others + predict_laya(alerts, refit)
        write_results(results_path, rows)
        print_confusion(rows, title)
        print_summary(rows, title)


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "--help"
    if cmd in ("-h", "--help"):
        print(__doc__.strip())
    elif cmd == "sample":
        rows = sample_alerts()
        print_sample_rows(rows)
    elif cmd == "pilot":
        pilot()
    elif cmd == "run":
        run()
    elif cmd == "plot":
        plot()
    elif cmd == "serve":
        serve()
    elif cmd == "checks":
        checks()
    elif cmd == "rebuild":
        rebuild_template_eval()
    elif cmd == "rehaiku":
        rehaiku()
    elif cmd == "relaya":
        relaya()
    else:
        raise SystemExit(f"unknown command {cmd!r}")
