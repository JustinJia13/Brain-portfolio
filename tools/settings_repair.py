"""Settings-repair pass: turn strong-but-not-submittable alphas into submittable
ones by a diagnosed simulation-setting fix (mostly decay for turnover).

Standalone and self-contained — does NOT touch the legacy autoworkshop repair
agent. For each alpha with good Sharpe (>= 1.25) that fails a submission gate
(usually turnover), one LLM call diagnoses the binding constraint and proposes
targeted settings_override variants; those are re-simulated (same expression,
new settings) and admitted only if they become submittable AND year-consistent
robust. Acceptance is by robustness, not in-sample Sharpe, so this is not a
parameter sweep.

Usage:
    python tools/settings_repair.py --dry-run                  # select + diagnose, no sims
    python tools/settings_repair.py --credentials brain2.txt   # re-sim the fixes
    python tools/settings_repair.py --sim-log artifacts/sim/master.jsonl --max-alphas 10
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core import llm as anthropic  # noqa: E402
from core.json_utils import parse_json_array  # noqa: E402
from core.brain.submit import DEFAULT_SETTINGS  # noqa: E402
from core.logs import merge_session_to_master
from core.prompts import load_prompt  # noqa: E402
from core.quality import is_submittable, quality_rank_key  # noqa: E402
from core.robustness import record_robustness_score  # noqa: E402
from alpha_enhancement.workshop.core import (  # noqa: E402
    DEFAULT_WORKSHOP_SETTINGS,
    WORKSHOP_CONCENTRATION_LIMIT,
    _resolved_concentration_value,
    failing_checks,
)
from alpha_enhancement.workshop.records import load_records  # noqa: E402

OUT_DIR = Path("artifacts/workshop")
SIM_LOG = OUT_DIR / "settings_repair_sim.jsonl"
SUBMITTABLE_OUT = OUT_DIR / "settings_repair_submittable.jsonl"
BASE_SETTINGS = {**DEFAULT_SETTINGS, **DEFAULT_WORKSHOP_SETTINGS}
ALLOWED_KEYS = frozenset({"decay", "neutralization", "truncation", "universe", "delay"})
SHARPE_MIN = 1.25
SHARPE_MAX = 4.0      # above this IS Sharpe is almost always a lookahead/numerical artifact
TURNOVER_MAX = 0.999  # turnover pinned at 1.0 = degenerate daily flip, not fixable by decay
MAX_FIXES_PER_ALPHA = 4


def _is_plausible_near_miss(rec: dict) -> bool:
    """A real, settings-fixable near-miss — not a degenerate artifact.

    Positive-direction signal (negate lane handles negative), Sharpe in a sane
    band, and turnover not pinned at the pathological ceiling.
    """
    sharpe = rec.get("is_sharpe")
    if sharpe is None:
        return False
    sharpe = float(sharpe)
    if not (SHARPE_MIN <= sharpe <= SHARPE_MAX):  # signed: positive near-misses only
        return False
    fitness = rec.get("is_fitness")
    if fitness is None or abs(float(fitness)) > SHARPE_MAX + 1:  # absurd fitness = artifact
        return False
    turnover = rec.get("is_turnover")
    if turnover is not None and float(turnover) >= TURNOVER_MAX:
        return False
    # Over-concentrated (>= 40%) is a fake signal — a bet on a few names, not a
    # real cross-sectional alpha. The workshop never repairs these, and a
    # settings tweak can't make one legitimate; exclude. Moderate concentration
    # (10-40%) IS a fixable near-miss (neutralization / truncation) and stays.
    conc = _resolved_concentration_value(rec)
    if conc is not None and conc >= WORKSHOP_CONCENTRATION_LIMIT:
        return False
    return True


def _norm(expr: object) -> str:
    return " ".join(str(expr or "").split())


def _already_attempted() -> set[str]:
    """Source expressions a prior settings-repair run already tried (don't redo)."""
    done: set[str] = set()
    for r in load_records([str(SIM_LOG)]):
        src = _norm((r.get("params") or {}).get("source_expression"))
        if src:
            done.add(src)
    return done


def select_targets(records: list[dict], max_alphas: int) -> list[dict]:
    """Strong-Sharpe but not-submittable alphas worth a settings fix.

    Dedup / filtering applied before any LLM call:
      - drop expressions that already have a submittable record anywhere
        (a settings variant already solved it)
      - drop expressions a previous settings-repair run already attempted
      - keep one record per expression (best |sharpe|)
    """
    solved: set[str] = set()
    for r in records:
        if r.get("status") == "done" and is_submittable(r):
            solved.add(_norm(r.get("expression")))
    attempted = _already_attempted()

    by_expr: dict[str, dict] = {}
    for r in records:
        if r.get("status") != "done":
            continue
        if not _is_plausible_near_miss(r):
            continue
        if is_submittable(r):
            continue  # already good
        key = _norm(r.get("expression"))
        if not key or key in solved or key in attempted:
            continue
        old = by_expr.get(key)
        if old is None or float(r.get("is_sharpe") or 0) > float(old.get("is_sharpe") or 0):
            by_expr[key] = r
    ranked = sorted(by_expr.values(), key=lambda r: float(r.get("is_sharpe") or 0), reverse=True)
    return ranked[:max_alphas] if max_alphas else ranked


def diagnose_and_propose(client, prompt, record: dict, *, model: str) -> list[dict]:
    """One LLM call -> list of {settings_override, expression, diagnosis}."""
    checks = ", ".join(failing_checks(record)) or "(none — likely turnover/fitness)"
    settings = record.get("settings") or BASE_SETTINGS
    resp = client.messages.create(
        model=model,
        max_completion_tokens=None,  # reasoning model: never truncate the JSON
        system=prompt.system,
        messages=[{"role": "user", "content": prompt.render_user(
            expression=record.get("expression", ""),
            sharpe=record.get("is_sharpe"),
            fitness=record.get("is_fitness"),
            turnover=record.get("is_turnover"),
            concentration=record.get("max_concentration"),
            failing_checks=checks,
            settings=json.dumps({k: settings.get(k) for k in
                                 ("decay", "neutralization", "truncation", "universe", "delay")}),
        )}],
    )
    fixes: list[dict] = []
    for item in parse_json_array(resp.content[0].text, allow_object=True):
        if not isinstance(item, dict):
            continue
        expr = str(item.get("expression") or record.get("expression") or "").strip()
        raw = item.get("settings_override") or {}
        override = {k: raw[k] for k in raw if k in ALLOWED_KEYS} if isinstance(raw, dict) else {}
        if not expr or not override:
            continue  # a settings-repair fix must change a setting
        fixes.append({"expression": expr, "settings_override": override,
                      "diagnosis": str(item.get("diagnosis") or "")})
    return fixes[:MAX_FIXES_PER_ALPHA]


COMPETITION_ACCOUNT = "brain"


def _account_label(credentials: str) -> str:
    return Path(credentials).stem


def _admit_official(winners: list[dict], session, competition_id: str) -> list[dict]:
    """On the brain (competition) account: fetch the REAL score-addition.

    The fixes were just simulated on brain, so each alpha exists there — query
    the official before/after competition projection directly. These survivors
    are already on brain and officially scored, so they're submit-ready (no
    promote_to_brain step needed)."""
    from core.portfolio.admission import fetch_brain_performance_comparison

    survivors: list[dict] = []
    for r in winners:
        aid = r.get("alpha_id")
        perf = fetch_brain_performance_comparison(session, aid, competition_id) if aid else None
        score = (perf or {}).get("score") or {}
        before, after = score.get("before"), score.get("after")
        delta = (after - before) if (before is not None and after is not None) else None
        r["brain_score_delta"] = delta
        if delta is not None and delta > 0.0:
            survivors.append(r)
    return survivors


def _admit_local(winners: list[dict], session, portfolio: dict) -> list[dict]:
    """On any non-competition account: LOCAL score-addition proxy.

    Score each winner's account-independent PnL against the brain portfolio on
    disk (self-corr + delta_pool_sharpe); keep low-corr, positive-contribution
    alphas for the official brain check later."""
    from core.brain.api import fetch_pnl_series_with_retry
    from core.portfolio.local_admission import passes_local_gate, score_locally

    survivors: list[dict] = []
    for r in winners:
        aid = r.get("alpha_id")
        pnl = fetch_pnl_series_with_retry(session, aid, max_attempts=8) if aid else {}
        delay = int((r.get("settings") or {}).get("delay", 1))
        metrics = score_locally(pnl, delay, portfolio)
        r["local_admission"] = metrics
        if passes_local_gate(metrics):
            survivors.append(r)
    return survivors


def _queue_for_brain(survivors: list[dict], account: str) -> None:
    from alpha_discovery.theory_method.dedup import lane_lock
    from alpha_discovery.theory_method.runner import PROMOTE_QUEUE

    if not survivors:
        return
    PROMOTE_QUEUE.parent.mkdir(parents=True, exist_ok=True)
    with lane_lock(), PROMOTE_QUEUE.open("a") as f:
        for r in survivors:
            m = r.get("local_admission", {})
            f.write(json.dumps({
                "theory_id": (r.get("params") or {}).get("source_expression", "")[:40],
                "expression": r.get("expression"),
                "direction": "+",
                "source_account": account,
                "is_sharpe": r.get("is_sharpe"),
                "is_fitness": r.get("is_fitness"),
                "robustness_score": record_robustness_score(r),
                "local_max_self_correlation": m.get("max_self_correlation"),
                "local_delta_pool_sharpe": m.get("delta_pool_sharpe"),
                "settings_override": r.get("settings") or r.get("settings_override"),
                "source": "settings_repair",
            }) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sim-log", nargs="*",
                    help="sim logs to scan (default: recent theory sessions + master)")
    ap.add_argument("--max-alphas", type=int, default=0,
                    help="cap on target alphas PER ROUND (0 = no cap)")
    ap.add_argument("--rounds", type=int, default=1,
                    help="repair passes to run; each re-selects (picks up new discoveries, "
                         "skips already-attempted). <=0 runs until a round finds no targets.")
    ap.add_argument("--credentials", default="brain2.txt",
                    help="account to run repair sims on. On 'brain' the OFFICIAL Brain "
                         "score-addition is fetched; on any other account the local proxy is used.")
    ap.add_argument("--brain-records", default="records/brain",
                    help="brain portfolio records dir for the local score-addition gate")
    ap.add_argument("--competition-id", default="IQC2026S1")
    ap.add_argument("--batch-size", type=int, default=3)
    ap.add_argument("--model", default="gpt-5.5")
    ap.add_argument("--count-only", action="store_true",
                    help="report how many alphas are worth fixing (after dedup) and exit — no LLM, no sims")
    ap.add_argument("--dry-run", action="store_true", help="select + diagnose (LLM), no simulations")
    args = ap.parse_args()

    paths = [p for p in (args.sim_log or (
        sorted(glob.glob("artifacts/sim/sessions/theory_*.jsonl")) + ["artifacts/sim/master.jsonl"]
    )) if Path(p).exists()]
    account = _account_label(args.credentials)
    on_brain = account == COMPETITION_ACCOUNT

    # ── count-only: report and exit (no LLM, no sims) ─────────────────────
    if args.count_only:
        targets = select_targets(load_records(paths), args.max_alphas)
        if not targets:
            print("No strong-but-not-submittable alphas found (after dedup).")
            return
        from collections import Counter
        modes: Counter = Counter()
        for r in targets:
            for c in (failing_checks(r) or ["(turnover/fitness)"]):
                modes[c] += 1
        print(f"targets worth fixing (after dedup): {len(targets)}")
        print("failing-gate mix:", dict(modes.most_common()))
        print(f"estimated cost: ~{len(targets)} LLM calls, up to ~{len(targets)*MAX_FIXES_PER_ALPHA} sims")
        print("\ntop 10 by sharpe:")
        for r in targets[:10]:
            print(f"  sharpe={r.get('is_sharpe')} turn={r.get('is_turnover')} "
                  f"fit={r.get('is_fitness')}  {r.get('expression','')[:70]}")
        return

    client = anthropic.Anthropic()
    prompt = load_prompt("settings_repair")
    portfolio = {} if on_brain else None  # loaded lazily for the local gate

    from core.brain.api import sign_in
    from alpha_enhancement.workshop.simulation import submit_candidate_batch
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    session = None

    perpetual = args.rounds <= 0
    rnd = 0
    total_winners = total_survivors = 0
    while perpetual or rnd < args.rounds:
        rnd += 1
        print(f"\n===== settings-repair round {rnd}/{'∞' if perpetual else args.rounds} "
              f"(account={account}) =====")
        targets = select_targets(load_records(paths), args.max_alphas)
        if not targets:
            print("No more strong-but-not-submittable alphas (after dedup). Stopping.")
            break
        print(f"targets this round: {len(targets)} — processed ONE AT A TIME "
              f"(diagnose -> sim -> {'official score' if on_brain else 'local gate'} -> next)")

        any_simulated = False
        for idx, rec in enumerate(targets, 1):
            # 1) diagnose THIS alpha (one LLM call)
            fixes = diagnose_and_propose(client, prompt, rec, model=args.model)
            sh = rec.get("is_sharpe")
            print(f"\n[{idx}/{len(targets)}] sharpe={sh} turn={rec.get('is_turnover')} "
                  f"fit={rec.get('is_fitness')}: {rec.get('expression','')[:65]}")
            for i, fx in enumerate(fixes):
                print(f"    fix[{i}] {fx['settings_override']} — {fx['diagnosis'][:75]}")
            if args.dry_run:
                continue
            cands = [{
                "candidate_id": f"setrepair_r{rnd}_{idx}_{i}",
                "search_lane": "settings_repair",
                "expression": fx["expression"],
                "settings_override": fx["settings_override"],
                "fields": rec.get("fields"), "cluster_ids": rec.get("cluster_ids"),
                "dataset_id": rec.get("dataset_id"),
                "params": {"source_expression": rec.get("expression"),
                           "source_sharpe": sh, "diagnosis": fx["diagnosis"]},
            } for i, fx in enumerate(fixes)]
            if not cands:
                continue

            # 2) simulate THIS alpha's fixes, then score before moving on
            if session is None:
                session = sign_in(args.credentials)
            new_records = submit_candidate_batch(
                cands, session, SIM_LOG, dict(BASE_SETTINGS),
                args.credentials, args.batch_size, yearly_stats_attempts=4,
            ) or []
            any_simulated = True
            winners = sorted([r for r in new_records if is_submittable(r)],
                             key=quality_rank_key, reverse=True)
            if not winners:
                print(f"    -> {len(new_records)} simmed, none submittable")
                continue

            # 3) admit: official score on brain, local proxy elsewhere
            if on_brain:
                survivors = _admit_official(winners, session, args.competition_id)
                detail = "  ".join(f"Δscore={r.get('brain_score_delta')}" for r in survivors[:3])
            else:
                if not portfolio:
                    from core.portfolio.local_admission import load_brain_portfolio
                    portfolio = load_brain_portfolio(args.brain_records)
                survivors = _admit_local(winners, session, portfolio)
                _queue_for_brain(survivors, account)
                detail = "  ".join(
                    f"corr={(r.get('local_admission') or {}).get('max_self_correlation')}"
                    f"/poolΔ={(r.get('local_admission') or {}).get('delta_pool_sharpe')}"
                    for r in survivors[:3])
            with SUBMITTABLE_OUT.open("a") as f:
                for r in winners:
                    f.write(json.dumps(r) + "\n")
            total_winners += len(winners)
            total_survivors += len(survivors)
            print(f"    -> {len(winners)} submittable, {len(survivors)} pass score-add  {detail}")

        if any_simulated:
            merge_session_to_master(SIM_LOG)  # once per round, not per alpha
        if args.dry_run:
            print("\ndry-run: diagnosis only, no simulations.")
            break

    print(f"\n=== done: {total_winners} submittable, {total_survivors} passed score-add "
          f"over {rnd} round(s) -> {SUBMITTABLE_OUT}")
    if not on_brain and total_survivors:
        print("Next: python tools/promote_to_brain.py --credentials brain.txt  "
              "(official Brain score-addition on the queued survivors)")


if __name__ == "__main__":
    main()
