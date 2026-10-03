"""Animated version of the result for sharing: "4 models hunt for failed agent runs".

Each lane is one model; its 497 squares are the Haiku-sample test runs, ranked by that model's
p_fail within each domain (most suspicious first, filled column by column). A cursor sweeps
every domain block at the same pace; a square turns orange when the run really failed. Counters
add up failures found, measured per-run latency and API cost. Only real predictions are used.
Fine-tuned Laya is three thin tracks, one per seed, each ranked by its own scores: averaging the
seeds' scores would be an ensemble, which is not a model in the experiment.

usage: python tau2/make_animation.py   -> tau2/results/hunt.mp4 (1080x1350, H.264) + hunt.gif
       python tau2/make_animation.py --stills t1 t2 ...   -> tau2/results/hunt_still_<t>.png only
"""
from __future__ import annotations

import json
import math
import random
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
import experiment as ex  # also puts the repo root on sys.path
import make_chart
from PIL import Image, ImageDraw, ImageFont

W, H, FPS, SECONDS = 1080, 1350, 30, 15
BG, INK, DIM, FAINT, LINE = "#000000", "#e8e6e1", "#8a8780", "#1c1c1c", "#3a3a3a"
ORANGE, ORANGE_HOT, OK = "#ff8a1f", "#ffd29a", "#4a4a4a"
FONT_DIR = Path("/usr/share/fonts/TTF")
REGULAR, BOLD = FONT_DIR / "JetBrainsMonoNerdFont-Regular.ttf", FONT_DIR / "JetBrainsMonoNerdFont-Bold.ttf"
HAIKU_COST_PER_RUN = 4.07 / 1000  # $ per run, from report_matched.json's measured token usage
ROWS, PITCH, SQ, BLOCK_GAP, LEFT = 8, 14, 12, 32, 60
THIN_ROWS, THIN_PITCH, THIN_SQ = 4, 7, 6  # the per-seed tracks of the fine-tuned lane
LANE_Y = (205, 452, 752, 1002)
REPO_URL = "github.com/PawnsApprentice/agent-failure-triage"

# timeline (seconds): intro, sweep to the 10% mark, hold, sweep to the end, hold, end card
T_START, T_10, T_10_END, T_FULL, T_CARD = 0.8, 3.8, 4.8, 11.2, 12.0


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(str(BOLD if bold else REGULAR), size)


def load_lanes() -> tuple[list[dict], list[dict]]:
    sample_ids = set(json.loads((ex.RESULTS / "haiku_sample.json").read_text())["run_ids"])
    rows = [r for r in ex.split_rows("test") if r["run_id"] in sample_ids]

    def track(label: str, path: Path, where: str, cost: float = 0.0) -> dict:
        """One ranking: runs sorted within each domain, most suspicious first. Ties (Haiku answers
        in whole percentages) are broken by a seeded shuffle, not by file order."""
        preds = {p["run_id"]: p for p in ex.load_jsonl(path)}
        rng = random.Random(ex.SEED)
        blocks = {}
        for d in ex.DOMAINS:
            keyed = [(-preds[r["run_id"]]["p_fail"], rng.random(), r) for r in rows if r["domain"] == d]
            blocks[d] = [(r["failed"], preds[r["run_id"]]["latency_ms"]) for *_, r in sorted(keyed, key=lambda x: x[:2])]
        return {"label": label, "blocks": blocks, "where": where, "cost": cost}

    # end-card values and 95% CIs: the same scenario-cluster bootstraps as the static chart
    chart = {m["name"]: m for m in make_chart.model_rows()[0]}
    lanes = [
        {"name": "TF-IDF + LogReg", "chart": chart["TF-IDF + logistic regression"],
         "tracks": [track("", ex.RESULTS / "pred_tfidf-tail1024.jsonl", "laptop CPU")]},
        {"name": "Laya fine-tuned, 3 seeds", "chart": chart["Laya, fine-tuned (mean of 3 seeds)"],
         "tracks": [track(f"seed {s}", ex.FT_OUT / f"test_S1_seed{s}.jsonl", "T4 GPU") for s in (42, 43, 44)]},
        {"name": "Claude Haiku 4.5 judge", "chart": chart["Claude Haiku 4.5 as judge"],
         "tracks": [track("", ex.RESULTS / "pred_haiku.jsonl", "API", HAIKU_COST_PER_RUN)]},
        {"name": "Laya zero-shot", "chart": chart["Laya, zero-shot"],
         "tracks": [track("", ex.RESULTS / "pred_laya-typed_W3.jsonl", "T4 GPU")]},
    ]
    return lanes, rows


def progress(t: float) -> float:
    """Fraction of each domain reviewed at time t: slow to the 10% mark, hold, then the rest."""

    def ease(x: float) -> float:  # smoothstep
        return x * x * (3 - 2 * x)

    if t < T_START:
        return 0.0
    if t < T_10:
        return 0.10 * ease((t - T_START) / (T_10 - T_START))
    if t < T_10_END:
        return 0.10
    if t < T_FULL:
        return 0.10 + 0.90 * ease((t - T_10_END) / (T_FULL - T_10_END))
    return 1.0


def reviewed(n: int, p: float) -> int:
    return min(n, math.ceil(p * n - 1e-9))


def fmt_seconds(s: float) -> str:
    if s < 10:
        return f"{s:5.2f} s"
    if s < 600:
        return f"{s:5.0f} s"
    return f"{int(s // 60)}m {int(s % 60):02d}s"


def block_x() -> dict[str, int]:
    """x of each domain block's left edge, sized for the full-height lanes (ROWS rows of PITCH)."""
    x, out = LEFT, {}
    for d, n in (("airline", 82), ("retail", 218), ("telecom", 197)):
        out[d] = x
        x += math.ceil(n / ROWS) * PITCH + BLOCK_GAP
    return out


def draw_track(g, trk: dict, t: float, top: int, rows: int, pitch: int, sq: int) -> dict:
    """Draw one ranking's three domain blocks at y=top; return its counters at time t."""
    p = progress(t)
    out = {"found": 0, "seen": 0, "compute_ms": 0.0, "by_domain": {}}
    for d, bx in block_x().items():
        items = trk["blocks"][d]
        n, k_seen = len(items), reviewed(len(items), p)
        f_seen = sum(f for f, _ in items[:k_seen])
        out["found"] += f_seen
        out["seen"] += k_seen
        out["compute_ms"] += sum(ms for _, ms in items[:k_seen])
        out["by_domain"][d] = (f_seen, sum(f for f, _ in items))
        k_recent = reviewed(n, progress(t - 0.25))  # squares revealed in the last 0.25 s flash
        for j, (failed, _) in enumerate(items):
            cx, cy = bx + (j // rows) * pitch, top + (j % rows) * pitch
            color = FAINT if j >= k_seen else OK if not failed else ORANGE_HOT if j >= k_recent else ORANGE
            g.rectangle([cx, cy, cx + sq - 1, cy + sq - 1], fill=color)
        # 10% review-budget mark: the stepped boundary after ceil(10% of n) runs
        half = (pitch - sq) // 2 + 1
        c, r = divmod(reviewed(n, 0.10), rows)
        xa, xb, bottom = bx + c * pitch - half, bx + (c + 1) * pitch - half, top + rows * pitch - half
        if r == 0:
            g.line([xa, top - 4, xa, bottom + 2], fill=ORANGE, width=2)
        else:
            yr = top + r * pitch - half
            g.line([(xb, top - 4), (xb, yr), (xa, yr), (xa, bottom + 2)], fill=ORANGE, width=2)
        if 0 < k_seen < n:  # cursor: under the column being reviewed
            xc = bx + ((k_seen - 1) // rows) * pitch
            g.rectangle([xc, bottom + 5, xc + sq - 1, bottom + 7], fill=INK)
    return out


def counters(trk: dict, c: dict, n_fail_total: int) -> str:
    return (f"found {int(c['found']):2d}/{n_fail_total}   compute {fmt_seconds(c['compute_ms'] / 1000):>8s} "
            f"({trk['where']})   cost ${trk['cost'] * c['seen']:.2f}")


def draw_sweep(lanes: list[dict], t: float, n_fail_total: int) -> Image.Image:
    img = Image.new("RGB", (W, H), BG)
    g = ImageDraw.Draw(img)
    g.text((LEFT, 52), "> 4 models hunt for failed agent runs", font=font(40, True), fill=ORANGE)
    g.text((LEFT, 112), f"497 held-out runs · {n_fail_total} real failures · ranked within each domain", font=font(22), fill=DIM)
    y = 150  # legend
    g.rectangle([LEFT, y + 6, LEFT + SQ, y + 6 + SQ], fill=ORANGE)
    g.text((LEFT + 22, y), "real failure", font=font(20), fill=DIM)
    g.rectangle([LEFT + 190, y + 6, LEFT + 190 + SQ, y + 6 + SQ], fill=OK)
    g.text((LEFT + 212, y), "reviewed, ok", font=font(20), fill=DIM)
    g.line([LEFT + 395, y + 2, LEFT + 395, y + 24], fill=ORANGE, width=2)
    g.text((LEFT + 407, y), "10% review budget", font=font(20), fill=DIM)
    if T_10 <= t < T_10_END + 0.4:
        g.text((LEFT + 660, y), "◂ 10% reviewed", font=font(20, True), fill=ORANGE)

    for lane, y0 in zip(lanes, LANE_Y):
        g.text((LEFT, y0), lane["name"], font=font(28, True), fill=INK)
        if len(lane["tracks"]) == 1:
            trk = lane["tracks"][0]
            c = draw_track(g, trk, t, y0 + 74, ROWS, PITCH, SQ)
            for d, bx in block_x().items():
                g.text((bx, y0 + 44), f"{d} {int(c['by_domain'][d][0])}/{c['by_domain'][d][1]}", font=font(18), fill=DIM)
            g.text((LEFT, y0 + 196), counters(trk, c, n_fail_total), font=font(21), fill=INK)
        else:  # one thin track per seed, each with its own ranking and counters
            for d, bx in block_x().items():
                g.text((bx, y0 + 40), d, font=font(18), fill=DIM)
            for k, trk in enumerate(lane["tracks"]):
                top = y0 + 70 + k * 68
                c = draw_track(g, trk, t, top, THIN_ROWS, THIN_PITCH, THIN_SQ)
                g.text((LEFT, top + 36), f"{trk['label']}  {counters(trk, c, n_fail_total)}", font=font(18), fill=INK)
    footer(g)
    return img


def footer(g: ImageDraw.ImageDraw) -> None:
    g.line([LEFT, H - 92, W - LEFT, H - 92], fill=LINE, width=1)
    g.text((LEFT, H - 78), "τ²-bench, 497 held-out runs. " + REPO_URL, font=font(19), fill=DIM)
    g.text((LEFT, H - 50), "compute = measured per-run latency; Haiku cost $4.07 per 1,000 runs", font=font(16), fill=DIM)


def draw_card(lanes: list[dict]) -> Image.Image:
    img = Image.new("RGB", (W, H), BG)
    g = ImageDraw.Draw(img)
    g.text((LEFT, 52), "> 4 models hunt for failed agent runs", font=font(40, True), fill=ORANGE)
    g.text((LEFT, 190), "within-domain AUROC, 95% CI", font=font(30), fill=DIM)
    x0, x1, lo_axis = LEFT, W - LEFT - 120, 0.5  # the axis runs from 0.5 (chance) to 1.0

    def x(v: float) -> float:
        return x0 + (x1 - x0) * (v - lo_axis) / (1.0 - lo_axis)

    y_first, step = 270, 158
    y_axis = y_first + 4 * step - 30
    for v in (0.5, 0.6, 0.7, 0.8, 0.9, 1.0):  # faint grid behind the bars
        g.line([x(v), y_first + 30, x(v), y_axis], fill=FAINT, width=1)
        g.text((x(v), y_axis + 8), f"{v:.1f}", font=font(18), fill=DIM, anchor="ma")
    for i, lane in enumerate(lanes):
        m, y, top3 = lane["chart"], y_first + i * step, i < 3
        g.text((LEFT, y), lane["name"].replace(", 3 seeds", " (mean of 3 seeds)"), font=font(28, True), fill=INK if top3 else DIM)
        yb = y + 52  # bar top; bar is 30 px tall
        g.rectangle([x0, yb, x(m["auroc"]), yb + 30], fill=ORANGE if top3 else OK)
        # 95% CI whisker (clipped at the axis start, with an arrow, if it reaches below 0.5)
        lo, hi, ym = m["ci"][0], m["ci"][1], yb + 15
        g.line([x(max(lo, lo_axis)), ym, x(hi), ym], fill=INK, width=3)
        g.line([x(hi), ym - 11, x(hi), ym + 11], fill=INK, width=3)
        if lo >= lo_axis:
            g.line([x(lo), ym - 11, x(lo), ym + 11], fill=INK, width=3)
        else:
            g.polygon([(x0, ym), (x0 + 12, ym - 9), (x0 + 12, ym + 9)], fill=INK)
        for sv in m.get("seeds", []):  # each seed's own AUROC: a tick below the bar
            g.line([x(sv), yb + 34, x(sv), yb + 48], fill=ORANGE_HOT, width=4)
        g.text((x1 + 22, yb - 8), f"{m['auroc']:.2f}", font=font(38, True), fill=ORANGE if top3 else DIM)
    g.text((x0, y_axis + 36), "axis starts at 0.5 (chance)   ├─┤ 95% CI   ╹ below bar: each seed", font=font(19), fill=DIM)
    g.text((LEFT, 1010), "statistically tied: top 3", font=font(46, True), fill=ORANGE)
    g.text((LEFT, 1078), "the 95% CIs overlap; every paired-bootstrap difference includes 0", font=font(21), fill=DIM)
    g.text((LEFT, 1110), "zero-shot's CI reaches below the axis (0.47)", font=font(21), fill=DIM)
    footer(g)
    return img


def frame(lanes, n_fail_total, t, card):
    if t < T_CARD:
        return draw_sweep(lanes, t, n_fail_total)
    sweep = draw_sweep(lanes, T_CARD - 1e-6, n_fail_total)
    fade = min(1.0, (t - T_CARD) / 0.4)
    return Image.blend(sweep, card, fade)


def main() -> None:
    lanes, rows = load_lanes()
    n_fail_total = sum(r["failed"] for r in rows)
    card = draw_card(lanes)
    if len(sys.argv) > 1 and sys.argv[1] == "--stills":
        for t in map(float, sys.argv[2:]):
            path = ex.RESULTS / f"hunt_still_{t:g}s.png"
            frame(lanes, n_fail_total, t, card).save(path)
            print(f"wrote {path}")
        return
    mp4, gif = ex.RESULTS / "hunt.mp4", ex.RESULTS / "hunt.gif"
    with tempfile.TemporaryDirectory() as tmp:
        for i in range(FPS * SECONDS):
            frame(lanes, n_fail_total, i / FPS, card).save(f"{tmp}/f{i:04d}.png")
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(FPS), "-i", f"{tmp}/f%04d.png",
                        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", "-preset", "slow",
                        "-movflags", "+faststart", "-an", str(mp4)], check=True)
        # README GIF: half size, 15 fps, one optimised palette
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(mp4), "-vf",
                        "fps=15,scale=540:-1:flags=lanczos,split[a][b];[a]palettegen=max_colors=64[p];[b][p]paletteuse=dither=none",
                        str(gif)], check=True)
    for lane in lanes:
        m = lane["chart"]
        print(f"  {lane['name']:26s} within-domain AUROC {m['auroc']:.3f}  CI [{m['ci'][0]:.3f}, {m['ci'][1]:.3f}]")
        for trk in lane["tracks"]:
            f10 = sum(sum(f for f, _ in b[:reviewed(len(b), 0.10)]) for b in trk["blocks"].values())
            total_s = sum(ms for b in trk["blocks"].values() for _, ms in b) / 1000
            print(f"    {trk['label'] or 'ranking':8s} found at 10%: {int(f10)}/{n_fail_total}  compute {total_s:.1f} s  "
                  f"cost ${trk['cost'] * len(rows):.2f}")
    print(f"wrote {mp4} ({mp4.stat().st_size / 1e6:.1f} MB), {gif} ({gif.stat().st_size / 1e6:.1f} MB)")
    if shutil.which("ffprobe"):
        print(subprocess.run(["ffprobe", "-v", "error", "-show_entries", "stream=codec_name,width,height,r_frame_rate:format=duration",
                              "-of", "compact", str(mp4)], capture_output=True, text=True, check=False).stdout.strip())


if __name__ == "__main__":
    main()
