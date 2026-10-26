"""Offline validation of the robustness score against historical records.

Read-only. Scores every record in master + records shards, prints the score
distribution, and — where OS data exists (submitted alphas) — compares
robustness_score vs is_fitness as predictors of os_sharpe via Spearman rank
correlation. Run this before tuning wq_alpha/robustness.py weights or wiring
the multiplier into rewards.

Usage:
    python tools/replay_robustness.py
    python tools/replay_robustness.py --min-sharpe 1.0
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from wq_alpha.robustness import compute_robustness, record_robustness_score  # noqa: E402

DEFAULT_GLOBS = [
    "artifacts/sim/master.jsonl",
    "records/brain/alphas_part*.jsonl",
    "records/brain2/alphas_part*.jsonl",
]


def iter_records(globs: list[str]):
    seen: set[str] = set()
    for pattern in globs:
        for path in sorted(glob.glob(pattern)):
            with open(path) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    alpha_id = rec.get("alpha_id")
                    if alpha_id:
                        if alpha_id in seen:
                            continue
                        seen.add(alpha_id)
                    yield rec


def spearman(xs: list[float], ys: list[float]) -> float:
    def ranks(values: list[float]) -> list[float]:
        order = sorted(range(len(values)), key=lambda i: values[i])
        r = [0.0] * len(values)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
                j += 1
            avg = (i + j) / 2 + 1
            for k in range(i, j + 1):
                r[order[k]] = avg
            i = j + 1
        return r

    rx, ry = ranks(xs), ranks(ys)
    n = len(xs)
    mx, my = sum(rx) / n, sum(ry) / n
    cov = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    vx = sum((a - mx) ** 2 for a in rx) ** 0.5
    vy = sum((b - my) ** 2 for b in ry) ** 0.5
    if vx == 0 or vy == 0:
        return 0.0
    return cov / (vx * vy)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--globs", nargs="*", default=DEFAULT_GLOBS)
    ap.add_argument("--min-sharpe", type=float, default=0.0,
                    help="only consider records with |is_sharpe| above this")
    args = ap.parse_args()

    n_total = n_yearly = n_scored = 0
    buckets = [0] * 10
    os_rows: list[tuple[float, float, float]] = []  # (robustness, is_fitness, os_sharpe)

    for rec in iter_records(args.globs):
        if rec.get("status") not in ("done", None) and not rec.get("is_sharpe"):
            continue
        sharpe = rec.get("is_sharpe")
        if sharpe is None or abs(sharpe) < args.min_sharpe:
            continue
        n_total += 1
        if rec.get("yearly_stats"):
            n_yearly += 1
        score = record_robustness_score(rec)
        if score is None:
            continue
        n_scored += 1
        buckets[min(9, int(score * 10))] += 1

        os_sharpe = rec.get("os_sharpe")
        fitness = rec.get("is_fitness")
        if os_sharpe is not None and fitness is not None:
            orient = -1.0 if (sharpe or 0) < 0 else 1.0
            os_rows.append((score, abs(fitness), orient * float(os_sharpe)))

    print(f"records considered:        {n_total}")
    print(f"  with yearly_stats:       {n_yearly}")
    print(f"  scored (>=3 full years): {n_scored}")
    print("\nscore distribution:")
    for i, count in enumerate(buckets):
        lo, hi = i / 10, (i + 1) / 10
        bar = "#" * int(60 * count / max(1, max(buckets)))
        print(f"  [{lo:.1f},{hi:.1f}) {count:6d} {bar}")

    if len(os_rows) >= 10:
        rob, fit, oss = zip(*os_rows)
        print(f"\nOS validation sample (submitted alphas with os_sharpe): n={len(os_rows)}")
        print(f"  spearman(robustness_score, os_sharpe) = {spearman(list(rob), list(oss)):+.3f}")
        print(f"  spearman(|is_fitness|,    os_sharpe) = {spearman(list(fit), list(oss)):+.3f}")
        print("  -> robustness should beat |is_fitness| as an OS predictor;")
        print("     if it does not, tune weights in wq_alpha/robustness.py before wiring in.")
    else:
        print(f"\nOS validation sample too small (n={len(os_rows)}); "
              "treat distribution sanity-check only.")


if __name__ == "__main__":
    main()
