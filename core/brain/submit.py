"""Submit a JSONL candidate plan to WorldQuant Brain and append results to a simulation log.

Submission flow per batch:
  1. POST each expression to /simulations → get progress URL
  2. Poll each progress URL until Brain signals completion (Retry-After = 0)
  3. GET /alphas/{alpha_id} for full metrics and checks
  4. Append one JSON record per alpha to the simulation log

Usage:
    python -m core.brain.submit
        --plan artifacts/model16_batch.jsonl
        --sim-log artifacts/model16_sim.jsonl
        --batch-size 3
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from core.brain.api import fetch_yearly_stats_with_retry, is_auth_expired_response, sign_in
from core.logs import run_log_path
from core.robustness import attach_robustness, normalize_yearly_stats as _norm_yearly

BRAIN_BASE = "https://api.worldquantbrain.com"

DEFAULT_SETTINGS: dict[str, Any] = {
    "instrumentType": "EQUITY",
    "region": "USA",
    "universe": "TOP3000",
    "delay": 1,
    "decay": 0,
    "neutralization": "SUBINDUSTRY",
    "truncation": 0.01,
    "pasteurization": "ON",
    "unitHandling": "VERIFY",
    "nanHandling": "OFF",
    "language": "FASTEXPR",
    "visualization": False,
}


def _format_regime_value(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:g}"
    return str(value).lower()


def _same_setting(value: Any, default: Any) -> bool:
    try:
        return abs(float(value) - float(default)) <= 1e-12
    except (TypeError, ValueError):
        return str(value).upper() == str(default).upper()


def _regime_key(settings: dict) -> str:
    """Canonical regime string encoding settings axes that differentiate runs.

    Default decay/truncation are omitted for backward-compatible regime names.
    Non-default values are included so same-expression regime checks are not
    deduped away.
    """
    settings = settings or {}
    parts = [
        f"delay{settings.get('delay', DEFAULT_SETTINGS['delay'])}",
        str(settings.get("neutralization", DEFAULT_SETTINGS["neutralization"])).lower(),
        str(settings.get("universe", DEFAULT_SETTINGS["universe"])).lower(),
    ]
    for axis in ("decay", "truncation"):
        if axis not in settings:
            continue
        value = settings.get(axis)
        if not _same_setting(value, DEFAULT_SETTINGS[axis]):
            label = "trunc" if axis == "truncation" else axis
            parts.append(f"{label}{_format_regime_value(value)}")
    return "_".join(parts)


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def _load_plan(path: str) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def _completed_pairs(sim_log: str, default_settings: dict) -> set[tuple[str, str]]:
    """Return (expression, regime) pairs already logged in sim_log.

    For legacy records without a stored 'regime' field, derives it from the
    stored 'settings' field (or falls back to default_settings).
    """
    p = Path(sim_log)
    if not p.exists():
        return set()
    done: set[tuple[str, str]] = set()
    with p.open() as f:
        for line in f:
            try:
                r = json.loads(line)
                expr = r.get("expression")
                if not expr:
                    continue
                status = r.get("status")
                if status and status != "done":
                    continue
                settings = r.get("settings")
                if isinstance(settings, dict):
                    regime = _regime_key(settings)
                else:
                    regime = r.get("regime") or _regime_key(default_settings)
                done.add((expr, regime))
            except json.JSONDecodeError:
                pass
    return done


def _append_record(path: str, record: dict) -> None:
    with open(path, "a") as f:
        f.write(json.dumps(record) + "\n")


# ---------------------------------------------------------------------------
# Brain API calls
# ---------------------------------------------------------------------------

def _reauth(session, credentials: str) -> bool:
    """Re-authenticate an existing session."""
    from core.brain.api import authenticate_session

    return authenticate_session(session, credentials, logger=logging)


def submit_one(
    session,
    expression: str,
    settings: dict,
    *,
    max_retries: int = 5,
    sleep: int = 30,
    credentials: str = "brain.txt",
) -> str | None:
    """POST one expression; return the progress URL or None on failure."""
    payload = {"type": "REGULAR", "settings": settings, "regular": expression}
    for attempt in range(max_retries):
        try:
            r = session.post(f"{BRAIN_BASE}/simulations", json=payload)
            if "Location" in r.headers:
                return r.headers["Location"]
            if is_auth_expired_response(r):
                logging.warning("Session expired - waiting for shared re-authentication ...")
                if not _reauth(session, credentials):
                    time.sleep(sleep)
                continue
            logging.warning(
                "Submit attempt %d/%d — status=%s body=%s",
                attempt + 1, max_retries, r.status_code, r.text[:200],
            )
        except Exception as exc:
            logging.warning("Submit attempt %d/%d — exception: %s", attempt + 1, max_retries, exc)
        time.sleep(sleep)
    return None


POLL_REAUTH_AFTER_CONSECUTIVE_FAILURES = 8


def poll_one(session, progress_url: str, *, sleep: int = 30, credentials: str | None = None) -> dict:
    """Poll progress_url until Brain returns a result (Retry-After == 0).

    A dead session surfaces as the server aborting the connection (RemoteDisconnected)
    rather than a clean auth-expired response, so it raises here instead of returning —
    and re-auth on an auth-expired *response* never fires. After
    POLL_REAUTH_AFTER_CONSECUTIVE_FAILURES connection exceptions in a row, force a
    re-auth to rebuild the session instead of hammering a dead connection forever.
    """
    consecutive_failures = 0
    while True:
        try:
            r = session.get(progress_url)
            consecutive_failures = 0
            retry_after = float(r.headers.get("Retry-After", 0))
            if retry_after == 0:
                if credentials and is_auth_expired_response(r):
                    logging.warning("Session expired - waiting for shared re-authentication ...")
                    _reauth(session, credentials)
                    continue
                return r.json()
            time.sleep(max(retry_after, sleep))
        except Exception as exc:
            consecutive_failures += 1
            logging.warning("Poll error (%d): %s", consecutive_failures, exc)
            if credentials and consecutive_failures >= POLL_REAUTH_AFTER_CONSECUTIVE_FAILURES:
                logging.warning(
                    "Poll: %d consecutive connection failures — forcing re-authentication ...",
                    consecutive_failures,
                )
                try:
                    _reauth(session, credentials)
                except Exception as reauth_exc:
                    logging.warning("Poll: re-auth attempt failed: %s", reauth_exc)
                consecutive_failures = 0
            time.sleep(sleep)


def fetch_detail(session, alpha_id: str) -> dict:
    """GET full alpha metrics and checks."""
    try:
        return session.get(f"{BRAIN_BASE}/alphas/{alpha_id}").json()
    except Exception:
        return {}


# ---------------------------------------------------------------------------
# Record assembly
# ---------------------------------------------------------------------------

def build_record(
    candidate: dict,
    settings: dict,
    progress_url: str | None,
    progress: dict,
    alpha_id: str | None,
    status: str,
    detail: dict,
    yearly_stats: dict | None = None,
) -> dict:
    is_ = detail.get("is", {}) or {}
    os_ = detail.get("os", {}) or {}
    checks = is_.get("checks") or []
    by_name = {c["name"]: c.get("result") == "PASS" for c in checks}
    record = {
        "logged_at": datetime.now(timezone.utc).isoformat(),
        "candidate_id": candidate.get("candidate_id"),
        "template_id": candidate.get("template_id"),
        "operator_skeleton": candidate.get("operator_skeleton"),
        "abandon_group": candidate.get("abandon_group"),
        "is_probe": candidate.get("is_probe"),
        "theory_id": candidate.get("theory_id"),
        "target_delay": candidate.get("target_delay"),
        "settings_override": candidate.get("settings_override"),
        "experiment_stage": candidate.get("experiment_stage"),
        "template_family": candidate.get("template_family"),
        "field_roles": candidate.get("field_roles"),
        "novelty_score": candidate.get("novelty_score"),
        "dataset_id": candidate.get("dataset_id"),
        "category": candidate.get("category"),
        "search_lane": candidate.get("search_lane", "sweep"),
        "expression": candidate.get("expression"),
        "fields": candidate.get("fields"),
        "group": candidate.get("group"),
        "params": candidate.get("params"),
        "sweep_signal": candidate.get("sweep_signal"),
        "sweep_weight": candidate.get("sweep_weight"),
        "priority_score": candidate.get("priority_score"),
        "skip_qtable_update": bool(candidate.get("skip_qtable_update", False)),
        "skip_workshop": bool(candidate.get("skip_workshop", False)),
        "status": status,
        "progress_url": progress_url,
        "alpha_id": alpha_id,
        "settings": settings,
        "regime": _regime_key(settings),
        "is_sharpe": is_.get("sharpe"),
        "is_fitness": is_.get("fitness"),
        "is_turnover": is_.get("turnover"),
        "is_returns": is_.get("returns"),
        "is_drawdown": is_.get("drawdown"),
        "os_sharpe": os_.get("sharpe"),
        "grade": detail.get("grade"),
        "checks": checks,
        "is_concentrated_weight_ok": by_name.get("CONCENTRATED_WEIGHT", True),
        "is_sub_universe_sharpe_ok": by_name.get("LOW_SUB_UNIVERSE_SHARPE", True),
        "yearly_stats": yearly_stats or {},
        "oos_sharpe_2023": (_norm_yearly(yearly_stats).get(2023) or {}).get("sharpe"),
        "oos_fitness_2023": (_norm_yearly(yearly_stats).get(2023) or {}).get("fitness"),
        "raw_alpha_detail": detail,
    }
    return attach_robustness(record)


# ---------------------------------------------------------------------------
# Main submission loop
# ---------------------------------------------------------------------------

def run(
    plan_path: str,
    sim_log: str,
    *,
    session=None,
    batch_size: int = 3,
    credentials: str = "brain.txt",
    settings: dict | None = None,
    max_candidates: int = 0,
    abandon_after: int = 15,
    abandon_threshold: float = 0.75,
    probe_abandon_after: int = 60,
    shard: str = "1/1",
    yearly_stats_attempts: int = 1,
    yearly_stats_sleep: float = 30.0,
) -> list[dict]:
    """Submit all pending candidates from plan_path, appending results to sim_log.

    Returns list of new sim records logged in this run.
    """
    shard_idx, shard_total = (int(x) for x in shard.split("/"))
    assert 1 <= shard_idx <= shard_total, f"Invalid shard: {shard}"

    s = settings or dict(DEFAULT_SETTINGS)
    if session is None:
        session = sign_in(credentials)

    candidates = _load_plan(plan_path)
    done = _completed_pairs(sim_log, s)
    pending = [
        c for c in candidates
        if (c.get("expression", ""), _regime_key({**s, **c.get("settings_override", {})})) not in done
    ]
    if shard_total > 1:
        pending = [c for i, c in enumerate(pending) if i % shard_total == shard_idx - 1]
        print(f"Shard {shard_idx}/{shard_total}: {len(pending)} candidates")
    if max_candidates > 0:
        pending = pending[:max_candidates]

    n_batches = (len(pending) + batch_size - 1) // batch_size if pending else 0
    print(f"Plan: {len(candidates)} total | {len(done)} done | {len(pending)} pending")

    template_tracker: dict[str, list[float]] = {}
    abandoned_templates: set[str] = set()

    logged = 0
    new_records: list[dict] = []
    for bi, start in enumerate(range(0, len(pending), batch_size), 1):
        batch = [
            c for c in pending[start : start + batch_size]
            if _abandon_key(c) not in abandoned_templates
        ]
        if not batch:
            print(f"\n=== Batch {bi}/{n_batches} | Logged this run: {logged} (skipped — all abandoned) ===")
            continue
        print(f"\n=== Batch {bi}/{n_batches} | Logged this run: {logged} ===")

        in_flight: list[tuple[dict, str]] = []
        for c in batch:
            expr = c["expression"]
            # Merge per-candidate settings_override (e.g. decay/truncation from repair).
            candidate_settings = {**s, **c.get("settings_override", {})}
            url = submit_one(session, expr, candidate_settings, credentials=credentials)
            if url:
                in_flight.append((c, url, candidate_settings))
                print(f"  Submitted [{c.get('template_id', '?')}]: {expr[:72]}")
            else:
                _append_record(sim_log, build_record(c, candidate_settings, None, {}, None, "submit_failed", {}))
                logged += 1
                print(f"  FAILED: {expr[:72]}")

        for c, url, candidate_settings in in_flight:
            progress = poll_one(session, url, credentials=credentials)
            alpha_id = progress.get("alpha")
            if alpha_id:
                detail = fetch_detail(session, alpha_id)
                yearly = fetch_yearly_stats_with_retry(
                    session,
                    alpha_id,
                    max_attempts=yearly_stats_attempts,
                    sleep_seconds=yearly_stats_sleep,
                )
                if yearly:
                    print(f"  [yearly] {alpha_id} yearly-stats fetched")
                elif yearly_stats_attempts > 1:
                    print(f"  [yearly] {alpha_id} yearly-stats unavailable after {yearly_stats_attempts} attempts")
                status = "done"
            else:
                detail = {}
                yearly = {}
                status = "no_alpha_id"

            record = build_record(c, candidate_settings, url, progress, alpha_id, status, detail, yearly)
            _append_record(sim_log, record)
            new_records.append(record)
            logged += 1

            sharpe = record["is_sharpe"]
            fitness = record["is_fitness"]
            grade = record["grade"]
            sharpe_str = f"{sharpe:.3f}" if sharpe is not None else "N/A"
            fitness_str = f"{fitness:.3f}" if fitness is not None else "N/A"
            print(
                f"  Result: fitness={fitness_str}  sharpe={sharpe_str}  grade={grade}  "
                f"[{c.get('template_id', '?')}]  {c.get('expression', '')[:50]}"
            )

            tid = _abandon_key(c)
            effective_after = probe_abandon_after if c.get("is_probe") else abandon_after
            if tid and effective_after > 0:
                tracker_list = template_tracker.setdefault(tid, [])
                if sharpe is not None:
                    tracker_list.append(sharpe)
                if (
                    tid not in abandoned_templates
                    and len(tracker_list) >= effective_after
                    and max(tracker_list) < abandon_threshold
                ):
                    abandoned_templates.add(tid)
                    kind = "probe" if c.get("is_probe") else "hardcoded"
                    print(
                        f"  [ABANDON] template={tid}  kind={kind}  trials={len(tracker_list)}"
                        f"  max_sharpe={max(tracker_list):.3f}  threshold={abandon_threshold}"
                    )

    print(f"\nDone. Logged {logged} new records to {sim_log}.")
    return new_records


def _abandon_key(candidate: dict) -> str:
    return str(candidate.get("abandon_group") or candidate.get("template_id") or "")


def _submit_inline(
    session,
    candidates: list[dict],
    sim_log: str,
    settings: dict,
    credentials: str,
    batch_size: int,
    sleep: int = 30,
    yearly_stats_attempts: int = 1,
    yearly_stats_sleep: float = 30.0,
) -> list[dict]:
    """Submit a list of candidates inline (no abandon-tracking), return new records."""
    already_done = _completed_pairs(sim_log, settings)
    pending = [
        c for c in candidates
        if (
            c.get("expression", ""),
            _regime_key({**settings, **c.get("settings_override", {})}),
        ) not in already_done
    ]
    if not pending:
        return []

    new_records: list[dict] = []
    for start in range(0, len(pending), batch_size):
        batch = pending[start : start + batch_size]
        in_flight: list[tuple[dict, str, dict]] = []
        for c in batch:
            expr = c["expression"]
            candidate_settings = {**settings, **c.get("settings_override", {})}
            url = submit_one(session, expr, candidate_settings, credentials=credentials)
            if url:
                in_flight.append((c, url, candidate_settings))
                print(f"  [workshop] Submitted: {expr[:72]}")
            else:
                rec = build_record(c, candidate_settings, None, {}, None, "submit_failed", {})
                _append_record(sim_log, rec)
                new_records.append(rec)

        for c, url, candidate_settings in in_flight:
            progress = poll_one(session, url, credentials=credentials)
            alpha_id = progress.get("alpha")
            if alpha_id:
                detail = fetch_detail(session, alpha_id)
                yearly = fetch_yearly_stats_with_retry(
                    session,
                    alpha_id,
                    max_attempts=yearly_stats_attempts,
                    sleep_seconds=yearly_stats_sleep,
                )
                if yearly:
                    print(f"  [yearly] {alpha_id} yearly-stats fetched")
                elif yearly_stats_attempts > 1:
                    print(f"  [yearly] {alpha_id} yearly-stats unavailable after {yearly_stats_attempts} attempts")
                status = "done"
            else:
                detail = {}
                yearly = {}
                status = "no_alpha_id"
            rec = build_record(c, candidate_settings, url, progress, alpha_id, status, detail, yearly)
            _append_record(sim_log, rec)
            new_records.append(rec)
            sharpe = rec["is_sharpe"]
            fitness = rec["is_fitness"]
            sharpe_str = f"{sharpe:.3f}" if sharpe is not None else "N/A"
            fitness_str = f"{fitness:.3f}" if fitness is not None else "N/A"
            print(
                f"  [workshop] Result: sharpe={sharpe_str} fitness={fitness_str} "
                f"grade={rec['grade']}  {c.get('expression', '')[:50]}"
            )

    return new_records


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Submit a JSONL plan to WorldQuant Brain.")
    p.add_argument("--plan", required=True, help="Input JSONL plan file.")
    p.add_argument("--sim-log", required=True, help="Simulation log to append results to.")
    p.add_argument("--batch-size", type=int, default=3)
    p.add_argument("--credentials", default="brain.txt")
    p.add_argument("--settings-json", default="", help="Optional JSON file to override default settings.")
    p.add_argument("--max-candidates", type=int, default=0, help="Cap total candidates (0 = no cap).")
    p.add_argument(
        "--abandon-after",
        type=int,
        default=15,
        help="Abandon a non-probe template after N results if max Sharpe is below threshold. 0 disables.",
    )
    p.add_argument(
        "--probe-abandon-after",
        type=int,
        default=60,
        help="Abandon a probe template after N results if max Sharpe is below threshold. 0 disables.",
    )
    p.add_argument("--abandon-threshold", type=float, default=0.75)
    p.add_argument(
        "--yearly-stats-attempts",
        type=int,
        default=1,
        help="Retry yearly-stats fetch this many times after each completed simulation.",
    )
    p.add_argument(
        "--yearly-stats-sleep",
        type=float,
        default=30.0,
        help="Seconds to wait between yearly-stats retry attempts.",
    )
    p.add_argument("--log-file", default=None)
    p.add_argument(
        "--auto-workshop", action="store_true",
        help="After plan completes, auto-generate negate/repair/enhance candidates and submit them.",
    )
    p.add_argument(
        "--workshop-iterations", type=int, default=1,
        help="Number of auto-workshop rounds (default 1). Each round processes the previous round's results.",
    )
    p.add_argument(
        "--workshop-api-key", default=None,
        help="OpenAI API key for auto-workshop LLM calls (default: OPENAI_API_KEY env var).",
    )
    p.add_argument(
        "--workshop-neg-threshold", type=float, default=None, help=argparse.SUPPRESS,
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    logging.basicConfig(
        filename=args.log_file or str(run_log_path("submit")),
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    settings = None
    if args.settings_json:
        with open(args.settings_json) as f:
            settings = json.load(f)

    s = settings or dict(DEFAULT_SETTINGS)
    session = sign_in(args.credentials)

    # Load all existing records for workshop context
    all_records: list[dict] = []
    p_sim = Path(args.sim_log)
    if p_sim.exists():
        with p_sim.open() as f:
            for line in f:
                try:
                    all_records.append(json.loads(line))
                except json.JSONDecodeError:
                    pass

    new_records = run(
        args.plan, args.sim_log,
        session=session,
        batch_size=args.batch_size,
        credentials=args.credentials,
        settings=s,
        max_candidates=args.max_candidates,
        abandon_after=args.abandon_after,
        abandon_threshold=args.abandon_threshold,
        probe_abandon_after=args.probe_abandon_after,
        yearly_stats_attempts=args.yearly_stats_attempts,
        yearly_stats_sleep=args.yearly_stats_sleep,
    )
    all_records.extend(new_records)

    if args.auto_workshop and new_records:
        # Layering note: this is the only place the shared layer references the
        # enhancement engine. It is a lazy import inside main(), reached only via
        # the opt-in --auto-workshop flag, so the submit LIBRARY has no upward
        # dependency -- only this CLI convenience path does.
        from alpha_enhancement.workshop.core import generate_candidates, record_workshop_key
        known_exprs: set[str] = {
            record_workshop_key(r) for r in all_records if r.get("expression", "")
        }
        if args.workshop_neg_threshold is not None:
            print("  [workshop] --workshop-neg-threshold is deprecated and ignored; routing is now fixed.")

        for iteration in range(1, args.workshop_iterations + 1):
            print(f"\n{'='*50}")
            print(f"AUTO-WORKSHOP iteration {iteration}/{args.workshop_iterations}")
            print(f"{'='*50}")

            workshop_candidates = generate_candidates(
                new_records,
                all_records,
                api_key=args.workshop_api_key,
                known_exprs=known_exprs,
                verbose=True,
            )
            if not workshop_candidates:
                print("  No new candidates generated — stopping.")
                break

            print(f"  Submitting {len(workshop_candidates)} workshop candidates...")
            new_records = _submit_inline(
                session, workshop_candidates, args.sim_log, s,
                args.credentials, args.batch_size,
                yearly_stats_attempts=args.yearly_stats_attempts,
                yearly_stats_sleep=args.yearly_stats_sleep,
            )
            all_records.extend(new_records)
            if not new_records:
                break

    print(f"\nAll done. Sim log: {args.sim_log}")
    try:
        from core.logs import MASTER, merge_session_to_master

        n_new, n_updated = merge_session_to_master(args.sim_log)
        print(f"Merged to master: {n_new} new, {n_updated} updated  ({MASTER})")
    except Exception as exc:
        print(f"[warn] Could not merge {args.sim_log} to master: {exc}")


if __name__ == "__main__":
    main()
