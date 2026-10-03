"""Step 1 data check for the tau2 experiment (can Laya replace an LLM judge for flagging failed
agent runs?). Downloads the released tau2-bench trajectories and reports what is in them.
No models are built here.

usage:
    python tau2/data_check.py download   # resumable; safe to re-run after a dropped connection
    python tau2/data_check.py normalize  # one row per run -> tau2/data/runs.jsonl
    python tau2/data_check.py report     # inventory + token lengths -> tau2/data_report.json

Sources:
    repo    sierra-research/tau2-bench data/tau2/results/final/*.json, pinned commit
    bucket  public leaderboard bucket sierra-tau-bench-public/submissions/, the trajectory file
            each text submission's submission.json names per domain (voice excluded; legacy
            submissions have no trajectories)
"""
from __future__ import annotations

import hashlib
import json
import random
import re
import sys
import time
import urllib.error
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

HERE = Path(__file__).parent
META = HERE / "data" / "meta"
RAW = HERE / "data" / "raw"
RUNS_PATH = HERE / "data" / "runs.jsonl"
REPORT_PATH = HERE / "data_report.json"
LAYA_CONTEXT = 1024  # laya-typed-decisions max_len
BUCKET = "https://sierra-tau-bench-public.s3.us-west-2.amazonaws.com"
REPO_RAW = "https://raw.githubusercontent.com/sierra-research/tau2-bench"

CHUNK = 1 << 20
MAX_ATTEMPTS = 30


def _download(url: str, dest: Path, expected_size: int) -> str:
    """Download with HTTP Range resume into dest.part, retrying with backoff; skip if dest is
    already complete. A dropped connection costs only the bytes since the last chunk."""
    if dest.exists() and dest.stat().st_size == expected_size:
        return "skipped (complete)"
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    for attempt in range(1, MAX_ATTEMPTS + 1):
        have = part.stat().st_size if part.exists() else 0
        if have == expected_size:
            break
        if have > expected_size:
            part.unlink()
            have = 0
        req = urllib.request.Request(url, headers={"Range": f"bytes={have}-"} if have else {})
        try:
            with urllib.request.urlopen(req, timeout=60) as r, part.open("ab" if have else "wb") as f:
                if have and r.status != 206:  # server ignored Range: restart from zero
                    f.truncate(0)
                while chunk := r.read(CHUNK):
                    f.write(chunk)
        except (urllib.error.URLError, TimeoutError, ConnectionError, OSError) as e:
            wait = min(60, 2 * attempt)
            print(f"    {dest.name}: {type(e).__name__} at {have / 1e6:.1f}MB, retry {attempt} in {wait}s",
                  file=sys.stderr)
            time.sleep(wait)
    if not part.exists() or part.stat().st_size != expected_size:
        raise RuntimeError(f"{dest.name}: gave up after {MAX_ATTEMPTS} attempts")
    part.rename(dest)
    return "downloaded"


def download_targets() -> list[dict]:
    """Every file to fetch: (source, submission, domain, url, dest, size)."""
    targets = []
    sha = json.loads((META / "repo_commit.json").read_text())["sha"]
    for f in json.loads((META / "repo_final_files.json").read_text()):
        targets.append({"source": "repo", "submission": "tau2-bench-final", "domain": None,
                        "url": f"{REPO_RAW}/{sha}/data/tau2/results/final/{f['name']}",
                        "dest": RAW / "repo" / f["name"], "size": f["size"]})
    subs = json.loads((META / "submissions.json").read_text())
    listing = json.loads((META / "bucket_listing.json").read_text())
    for d, s in subs.items():
        if s["kind"] != "submissions" or not s["trajectories_available"]:
            continue
        sizes = {key.split("/trajectories/", 1)[1]: size for key, size in listing.get(d, [])}
        for domain, name in (s["trajectory_files"] or {}).items():
            targets.append({"source": "bucket", "submission": d, "domain": domain,
                            "url": f"{BUCKET}/submissions/{d}/trajectories/{name}",
                            "dest": RAW / "bucket" / d / name, "size": sizes.get(name)})
    return targets


PARALLEL_DOWNLOADS = 4  # throughput here is capped per connection (~350 KB/s), not by the link


def download() -> None:
    from concurrent.futures import ThreadPoolExecutor, as_completed

    targets = download_targets()
    missing = [t for t in targets if t["size"] is None]
    # banking_knowledge last: it is reported separately (re-graded in tau2-bench v1.0.1)
    todo = sorted((t for t in targets if t["size"] is not None), key=lambda t: t["domain"] == "banking_knowledge")
    print(f"{len(todo)} files ({sum(t['size'] for t in todo) / 1e9:.2f} GB) to fetch; "
          f"{len(missing)} named in submission.json but absent from the bucket", file=sys.stderr)
    for t in missing:
        print(f"  UNAVAILABLE {t['submission']} {t['domain']}: {t['dest'].name}", file=sys.stderr)
    failed = []
    with ThreadPoolExecutor(PARALLEL_DOWNLOADS) as pool:
        futures = {pool.submit(_download, t["url"], t["dest"], t["size"]): t for t in todo}
        for i, fut in enumerate(as_completed(futures), 1):
            t = futures[fut]
            try:
                status = fut.result()
            except RuntimeError as e:
                failed.append(str(e))
                status = "FAILED"
            print(f"  [{i}/{len(todo)}] {status:18s} {t['size'] / 1e6:7.1f}MB  {t['dest'].relative_to(RAW)}",
                  file=sys.stderr)
    print(f"done; {len(failed)} failed" + ("" if not failed else ": re-run to resume"), file=sys.stderr)


# ---------------------------------------------------------------------------
# Normalize: one row per simulation, with the plain-text rendering later models will see
# ---------------------------------------------------------------------------

REPO_NAME_RE = re.compile(r"^(?P<agent>.+?)_(?P<domain>airline|retail|telecom-workflow|telecom)_(?P<variant>[^_]+)_")


def render(messages: list[dict]) -> str:
    """Plain-text trajectory in message order. Fixed here because later Laya/TF-IDF inputs use it:
    `role: content`, `tool_call name(args json)` for every call, `tool name: content` for results."""
    tool_names = {}
    lines = []
    for m in messages:
        role = m.get("role")
        if role == "tool":
            lines.append(f"tool {tool_names.get(m.get('id'), '?')}: {m.get('content') or ''}")
            continue
        if m.get("content"):
            lines.append(f"{role}: {m['content']}")
        for call in m.get("tool_calls") or []:
            tool_names[call.get("id")] = call.get("name")
            args = json.dumps(call.get("arguments"), ensure_ascii=False, separators=(",", ":"))
            lines.append(f"{role} tool_call {call.get('name')}({args})")
    return "\n".join(lines)


def final_agent_message(messages: list[dict]) -> str:
    for m in reversed(messages):
        if m.get("role") == "assistant" and m.get("content"):
            return m["content"]
    return ""


def _task_hash(task: dict) -> str:
    return hashlib.sha1(json.dumps(task.get("user_scenario", task), sort_keys=True).encode()).hexdigest()[:12]


def normalize() -> None:
    targets = [t for t in download_targets() if t["dest"].exists()]
    n_rows = 0
    with RUNS_PATH.open("w") as out:
        for t in targets:
            d = json.loads(t["dest"].read_text())
            if not isinstance(d, dict) or "simulations" not in d:
                raise SystemExit(f"{t['dest']}: not the monolithic results format (keys: {list(d)[:5]})")
            info = d.get("info", {})
            if t["source"] == "repo":
                m = REPO_NAME_RE.match(t["dest"].name)
                domain, variant = m["domain"], m["variant"]
            else:
                domain, variant = t["domain"], "default"
            task_hash = {str(task["id"]): _task_hash(task) for task in d.get("tasks", [])}
            for s in d["simulations"]:
                msgs = s.get("messages") or []
                row = {
                    "source": t["source"], "submission": t["submission"], "file": t["dest"].name,
                    "agent_model": info.get("agent_info", {}).get("llm"),
                    "user_model": info.get("user_info", {}).get("llm"),
                    "domain": domain, "variant": variant,
                    "task_id": str(s.get("task_id")), "task_hash": task_hash.get(str(s.get("task_id"))),
                    "trial": s.get("trial"),
                    "reward": (s.get("reward_info") or {}).get("reward"),
                    "termination_reason": s.get("termination_reason"),
                    "n_messages": len(msgs),
                    "final_agent_message": final_agent_message(msgs),
                    "text": render(msgs),
                }
                out.write(json.dumps(row, ensure_ascii=False) + "\n")
                n_rows += 1
            print(f"  {n_rows:6d} runs after {t['dest'].relative_to(RAW)}", file=sys.stderr)
            del d


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

# Modelled on arXiv 2606.09863's closing-message scheme; used only to size the
# "failed but claims success" subset, never as a training label.
CLAIMS_SUCCESS_RE = re.compile(
    r"\b(successfully|all set|is (?:now )?confirmed|has been (?:\w+ )?(?:processed|updated|cancell?ed|refunded|changed|"
    r"modified|booked|issued|completed|exchanged|returned|submitted|resolved|fixed|applied|placed|added|removed)|"
    r"(?:I(?:'ve| have)|we(?:'ve| have)) (?:now |successfully |also )?(?:processed|updated|cancell?ed|refunded|changed|"
    r"modified|booked|issued|completed|exchanged|returned|submitted|resolved|fixed|applied|placed|added|removed)|"
    r"is now (?:\w+ ){0,3}(?:cancell?ed|updated|active|resolved|working|fixed|restored|enabled))\b", re.IGNORECASE)
ADMITS_FAILURE_RE = re.compile(
    r"\b(I(?:'m| am) (?:unable|not able)|I can(?:not|'t)|unable to|not (?:possible|able)|cannot be|"
    r"(?:transfer|transferring) you|human (?:agent|representative)|I(?:'m| am) sorry,? but)\b", re.IGNORECASE)


def closing_class(msg: str) -> str:
    s, f = bool(CLAIMS_SUCCESS_RE.search(msg)), bool(ADMITS_FAILURE_RE.search(msg))
    return "claims_success" if s and not f else "admits_failure" if f and not s else "ambiguous"


def _dist(values: list[int]) -> dict:
    from common.metrics import percentile

    return {"n": len(values), "median": percentile(values, 0.5), "p90": percentile(values, 0.9),
            "max": max(values), "pct_over_1024": round(100 * sum(v > LAYA_CONTEXT for v in values) / len(values), 1)}


def token_lengths(texts: list[str], tok, batch: int = 64) -> list[int]:
    out = []
    for i in range(0, len(texts), batch):
        enc = tok(texts[i:i + batch], add_special_tokens=False, verbose=False)
        out += [len(ids) for ids in enc["input_ids"]]
    return out


def report() -> None:
    from common import laya_agent

    tok = laya_agent.get_agent("laya-typed").tok
    runs, texts = [], []
    with RUNS_PATH.open() as f:
        for line in f:
            r = json.loads(line)
            texts.append(r.pop("text"))
            runs.append(r)
    print(f"tokenizing {len(runs)} runs...", file=sys.stderr)
    for r, n in zip(runs, token_lengths(texts, tok)):
        r["tokens"] = n
    del texts

    def group(key):
        g = defaultdict(list)
        for r in runs:
            g[key(r)].append(r)
        return dict(sorted(g.items(), key=lambda kv: str(kv[0])))

    def failure_stats(rs):
        rewards = [r["reward"] for r in rs if r["reward"] is not None]
        fails = sum(r == 0 for r in rewards)
        return {"runs": len(rs), "failed": fails, "failure_rate": round(fails / len(rewards), 3) if rewards else None,
                "non_binary_reward": sum(r not in (0, 1) for r in rewards), "missing_reward": len(rs) - len(rewards)}

    rep = {
        "total_runs": len(runs),
        "by_source": {k: failure_stats(v) for k, v in group(lambda r: r["source"]).items()},
        "by_domain": {k: failure_stats(v) | {"unique_tasks": len({r["task_id"] for r in v}),
                                              "tokens": _dist([r["tokens"] for r in v])}
                      for k, v in group(lambda r: r["domain"]).items()},
        "by_domain_model": {f"{k[0]} | {k[1]} | {k[2]}": failure_stats(v)
                            for k, v in group(lambda r: (r["domain"], r["agent_model"], r["variant"])).items()},
        "tokens_overall": _dist([r["tokens"] for r in runs]),
        "termination_reasons": dict(Counter(r["termination_reason"] for r in runs).most_common()),
        "termination_by_outcome": {
            "failed": dict(Counter(r["termination_reason"] for r in runs if r["reward"] == 0).most_common()),
            "succeeded": dict(Counter(r["termination_reason"] for r in runs if r["reward"] == 1).most_common())},
    }

    # Same task id with different content across files would break task-disjoint splits.
    variants = defaultdict(set)
    for r in runs:
        variants[(r["domain"], r["task_id"])].add(r["task_hash"])
    rep["task_ids_with_differing_content"] = {
        dom: sum(len(h) > 1 for (d, _), h in variants.items() if d == dom) for dom in sorted({d for d, _ in variants})}

    failed = [r for r in runs if r["reward"] == 0]
    failed_by_domain = group(lambda r: r["domain"] if r["reward"] == 0 else None)
    failed_by_domain.pop(None, None)
    rep["closing_message_on_failed_runs"] = {
        dom: dict(Counter(closing_class(r["final_agent_message"]) for r in rs)) for dom, rs in failed_by_domain.items()}
    rng = random.Random(0)
    by_cls = defaultdict(list)
    for r in failed:
        by_cls[closing_class(r["final_agent_message"])].append(r)
    rep["closing_message_samples"] = {
        c: [f"[{r['domain']}] {r['final_agent_message'][:220]}" for r in rng.sample(rs, min(10, len(rs)))]
        for c, rs in sorted(by_cls.items())}

    # Cross-check with arXiv 2606.09863: 9,876 leaderboard runs, 8 families, airline/retail/telecom, 1,730 failures.
    paper = [r for r in runs if r["source"] == "bucket" and r["domain"] in ("airline", "retail", "telecom")]
    rep["paper_crosscheck"] = {"our_bucket_runs_airline_retail_telecom": len(paper),
                               "our_failures": sum(r["reward"] == 0 for r in paper),
                               "submissions": sorted({r["submission"] for r in paper}),
                               "paper_runs": 9876, "paper_failures": 1730}
    REPORT_PATH.write_text(json.dumps(rep, indent=2, ensure_ascii=False) + "\n")
    print(f"wrote {REPORT_PATH}", file=sys.stderr)
    print_report(rep)


def print_report(rep: dict) -> None:
    print(f"\n=== tau2 data check: {rep['total_runs']} runs ===")
    for src, s in rep["by_source"].items():
        print(f"source {src:7s} runs={s['runs']:6d} failed={s['failed']:5d} ({s['failure_rate']:.1%})")
    print(f"\n{'domain':18s}{'runs':>7s}{'failed':>8s}{'fail%':>7s}{'tasks':>7s}"
          f"{'tok med':>9s}{'p90':>7s}{'max':>8s}{'>1024':>7s}{'nonbin':>8s}")
    for dom, s in rep["by_domain"].items():
        t = s["tokens"]
        print(f"{dom:18s}{s['runs']:7d}{s['failed']:8d}{s['failure_rate']:7.1%}{s['unique_tasks']:7d}"
              f"{t['median']:9d}{t['p90']:7d}{t['max']:8d}{t['pct_over_1024']:6.1f}%{s['non_binary_reward']:8d}")
    t = rep["tokens_overall"]
    print(f"{'ALL':18s}{rep['total_runs']:7d}{'':22s}{t['median']:9d}{t['p90']:7d}{t['max']:8d}{t['pct_over_1024']:6.1f}%")
    print(f"\n{'domain | agent model | variant':78s}{'runs':>6s}{'failed':>8s}{'fail%':>7s}")
    for k, s in rep["by_domain_model"].items():
        print(f"{k[:78]:78s}{s['runs']:6d}{s['failed']:8d}{s['failure_rate']:7.1%}")
    print("\ntermination reasons, failed runs:   ", rep["termination_by_outcome"]["failed"])
    print("termination reasons, succeeded runs:", rep["termination_by_outcome"]["succeeded"])
    print("task ids whose content differs across files:", rep["task_ids_with_differing_content"])
    print("\nclosing message of failed runs (regex; sizing only, not a label):")
    for dom, c in rep["closing_message_on_failed_runs"].items():
        n = sum(c.values())
        print(f"  {dom:18s} claims_success={c.get('claims_success', 0):5d} ({c.get('claims_success', 0) / n:.0%})"
              f"  admits_failure={c.get('admits_failure', 0):5d}  ambiguous={c.get('ambiguous', 0):5d}  of {n}")
    p = rep["paper_crosscheck"]
    print(f"\npaper cross-check (arXiv 2606.09863: {p['paper_runs']} runs, {p['paper_failures']} failures): "
          f"ours {p['our_bucket_runs_airline_retail_telecom']} runs, {p['our_failures']} failures "
          f"from {len(p['submissions'])} bucket submissions")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "--help"
    if cmd == "download":
        download()
    elif cmd == "normalize":
        normalize()
    elif cmd == "report":
        report()
    elif cmd == "show":
        print_report(json.loads(REPORT_PATH.read_text()))
    else:
        print(__doc__.strip())
