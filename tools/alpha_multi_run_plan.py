"""Submit a multi-dataset plan to Brain and log results.

This is a thin wrapper around alpha_grammar's submission logic that:
1. Reads the JSONL plan from build_plan.py
2. Submits each candidate to Brain
3. Logs results to sim.jsonl (same format as alpha_grammar's runner)

Usage:
    python -m alpha_multi.run_plan \\
        --plan-path alpha_multi/artifacts/pv1-fundamental6_batch1.jsonl \\
        --simulation-log alpha_multi/artifacts/pv1-fundamental6_sim.jsonl \\
        --run-log alpha_multi/artifacts/pv1-fundamental6_runner.log \\
        --batch-size 2
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from wq_alpha.brain.api import sign_in, fetch_yearly_stats_with_retry
from wq_alpha.logs import MASTER, merge_session_to_master, run_log_path


DEFAULT_SIMULATION_SETTINGS = {
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


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Submit multi-dataset alpha plan to Brain.")
    p.add_argument("--credentials",
                   default=str(Path(__file__).parent.parent / "brain.txt"))
    p.add_argument("--plan-path", required=True,
                   help="JSONL plan from build_plan.py")
    p.add_argument("--simulation-log", default="",
                   help="Append-only simulation log (defaults to same dir as plan)")
    p.add_argument("--run-log", default="",
                   help="Text progress log")
    p.add_argument("--batch-size", type=int, default=2,
                   help="Concurrent submissions (default 2; Brain limit ~3)")
    p.add_argument("--max-candidates", type=int, default=0,
                   help="Optional hard cap on number of candidates")
    p.add_argument("--sleep-seconds", type=int, default=30)
    p.add_argument("--max-submit-retries", type=int, default=5)
    p.add_argument("--settings-json", default="",
                   help="Optional JSON file overriding default simulation settings")
    return p.parse_args()


def append_jsonl(path: str, record: dict[str, Any]) -> None:
    with open(path, "a") as f:
        f.write(json.dumps(record) + "\n")
        f.flush()


def load_completed_keys(simulation_log: str) -> tuple[set[str], set[str]]:
    candidate_ids: set[str] = set()
    expressions: set[str] = set()
    log_path = Path(simulation_log)
    if not log_path.exists():
        return candidate_ids, expressions
    with log_path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if rec.get("candidate_id"):
                candidate_ids.add(rec["candidate_id"])
            if rec.get("expression"):
                expressions.add(rec["expression"])
    return candidate_ids, expressions


def _reauth_if_needed(session, response, credentials_path: str) -> bool:
    """Re-authenticate if Brain returned 401 (session expired)."""
    from wq_alpha.brain.api import authenticate_session, is_auth_expired_response

    if is_auth_expired_response(response):
        logging.warning("Brain session expired - waiting for shared re-authentication ...")
        if not authenticate_session(session, credentials_path, logger=logging):
            logging.error("Re-authentication did not complete; request will retry after backoff.")
        return True
    return False


def submit_alpha(session, expression, settings, *, max_retries, sleep_seconds, credentials_path="") -> str | None:
    payload = {"type": "REGULAR", "settings": settings, "regular": expression}
    for attempt in range(max_retries):
        try:
            resp = session.post("https://api.worldquantbrain.com/simulations", json=payload)
            if "Location" in resp.headers:
                return resp.headers["Location"]
            logging.warning("Submit attempt %d/%d — status=%s", attempt + 1, max_retries, resp.status_code)
            if credentials_path and _reauth_if_needed(session, resp, credentials_path):
                continue
        except Exception as exc:
            logging.warning("Submit attempt %d/%d — %s", attempt + 1, max_retries, exc)
        time.sleep(sleep_seconds)
    return None


def wait_for_completion(session, url, *, sleep_seconds, credentials_path: str = "") -> dict[str, Any]:
    while True:
        try:
            resp = session.get(url)
            retry_after = float(resp.headers.get("Retry-After", 0))
            if retry_after == 0:
                if credentials_path and _reauth_if_needed(session, resp, credentials_path):
                    continue
                return resp.json()
            time.sleep(max(retry_after, sleep_seconds))
        except Exception as exc:
            logging.warning("wait_for_completion: %s", exc)
            time.sleep(sleep_seconds)


def get_alpha_detail(session, alpha_id: str) -> dict[str, Any]:
    try:
        return session.get(f"https://api.worldquantbrain.com/alphas/{alpha_id}").json()
    except Exception:
        return {}


def build_record(
    candidate, settings, progress_url, final_progress, alpha_id, status, alpha_detail,
    yearly_stats: dict | None = None,
) -> dict[str, Any]:
    alpha_detail = alpha_detail or {}
    is_stats = alpha_detail.get("is", {}) or {}
    os_stats = alpha_detail.get("os", {}) or {}
    checks_list = is_stats.get("checks") or []
    checks_by_name = {c["name"]: c.get("result", "FAIL") == "PASS" for c in checks_list}
    return {
        "logged_at": datetime.now(timezone.utc).isoformat(),
        "candidate_id": candidate.get("candidate_id"),
        "family_id": candidate.get("family_id"),
        "intention_id": candidate.get("intention_id"),
        "intention_label": candidate.get("intention_label"),
        "template_id": candidate.get("template_id"),
        "search_lane": candidate.get("search_lane"),
        "category": candidate.get("category"),
        "dataset_id": candidate.get("dataset_id"),
        "expression": candidate.get("expression"),
        "fields": candidate.get("fields"),
        "field_types": candidate.get("field_types"),
        "cluster_ids": candidate.get("cluster_ids"),
        "group": candidate.get("group"),
        "params": candidate.get("params"),
        "horizon_bucket": candidate.get("horizon_bucket"),
        "tree_depth": candidate.get("tree_depth"),
        "priority_score": candidate.get("priority_score"),
        "status": status,
        "progress_url": progress_url,
        "alpha_id": alpha_id,
        "settings": settings,
        "raw_progress_result": final_progress or {},
        "is_sharpe": is_stats.get("sharpe"),
        "is_fitness": is_stats.get("fitness"),
        "is_turnover": is_stats.get("turnover"),
        "is_returns": is_stats.get("returns"),
        "is_drawdown": is_stats.get("drawdown"),
        "is_margin": is_stats.get("margin"),
        "is_ic": is_stats.get("ic"),
        "is_concentrated_weight_ok": checks_by_name.get("CONCENTRATED_WEIGHT", True),
        "is_sub_universe_sharpe_ok": checks_by_name.get("LOW_SUB_UNIVERSE_SHARPE", True),
        "os_sharpe": os_stats.get("sharpe"),
        "os_fitness": os_stats.get("fitness"),
        "grade": alpha_detail.get("grade"),
        "checks": is_stats.get("checks"),
        "date_created": alpha_detail.get("dateCreated"),
        "yearly_stats": yearly_stats or {},
        "oos_sharpe_2023": (yearly_stats or {}).get(2023, {}).get("sharpe"),
        "raw_alpha_detail": alpha_detail,
        "candidate_spec": candidate,
    }


def run(
    plan_path: str,
    simulation_log: str,
    *,
    credentials: str = "brain.txt",
    batch_size: int = 2,
    max_candidates: int = 0,
    sleep_seconds: int = 30,
    max_submit_retries: int = 5,
    settings_override: dict | None = None,
) -> None:
    """Submit a plan and log results. Callable from other modules (e.g. enhance_loop)."""
    _run(
        plan_path=plan_path,
        simulation_log=simulation_log,
        run_log="",
        credentials=credentials,
        batch_size=batch_size,
        max_candidates=max_candidates,
        sleep_seconds=sleep_seconds,
        max_submit_retries=max_submit_retries,
        settings_json="",
        settings_override=settings_override,
    )


def main() -> None:
    args = parse_args()
    _run(
        plan_path=args.plan_path,
        simulation_log=args.simulation_log,
        run_log=args.run_log,
        credentials=args.credentials,
        batch_size=args.batch_size,
        max_candidates=args.max_candidates,
        sleep_seconds=args.sleep_seconds,
        max_submit_retries=args.max_submit_retries,
        settings_json=args.settings_json,
    )


def _run(
    plan_path: str,
    simulation_log: str,
    run_log: str,
    credentials: str,
    batch_size: int,
    max_candidates: int,
    sleep_seconds: int,
    max_submit_retries: int,
    settings_json: str,
    settings_override: dict | None = None,
) -> None:
    plan_path = str(plan_path)
    _plan_path = Path(plan_path)
    _run_id = _plan_path.stem.split("_batch")[0]
    sim_log = simulation_log or str(_plan_path.parent / f"{_run_id}_sim.jsonl")
    _run_log = run_log or str(run_log_path("alpha_multi"))

    logging.basicConfig(filename=_run_log, level=logging.INFO,
                        format="%(asctime)s - %(levelname)s - %(message)s")

    with _plan_path.open() as f:
        candidates = [json.loads(l) for l in f if l.strip()]

    done_ids, done_exprs = load_completed_keys(sim_log)
    remaining = [
        c for c in candidates
        if c.get("candidate_id") not in done_ids
        and c.get("expression") not in done_exprs
    ]
    if max_candidates > 0:
        remaining = remaining[:max_candidates]

    settings = DEFAULT_SIMULATION_SETTINGS.copy()
    if settings_override:
        settings.update(settings_override)
    elif settings_json:
        with open(settings_json) as f:
            settings = json.load(f)

    session = sign_in(credentials)

    print(f"Plan:      {plan_path}")
    print(f"Total:     {len(candidates)}  Remaining: {len(remaining)}")
    print(f"Sim log:   {sim_log}")

    total = len(remaining)
    logged = 0

    def _simulate_one(c: dict) -> dict:
        expr = c["expression"]
        print(f"[MULTI]   Submit: {expr[:80]}")
        url = submit_alpha(
            session, expr, settings,
            max_retries=max_submit_retries,
            sleep_seconds=sleep_seconds,
            credentials_path=credentials,
        )
        if not url:
            logging.warning("Submit failed: %s", expr)
            return build_record(c, settings, None, {}, None, "submit_failed", {})

        final = wait_for_completion(session, url, sleep_seconds=sleep_seconds,
                                    credentials_path=credentials)
        alpha_id = final.get("alpha")
        if alpha_id:
            detail = get_alpha_detail(session, alpha_id)
            yearly = fetch_yearly_stats_with_retry(session, alpha_id)
            status = "done"
            logging.info("Submitted: %s", expr)
        else:
            detail, yearly, status = {}, {}, "no_alpha_id"
        return build_record(c, settings, url, final, alpha_id, status, detail, yearly)

    with ThreadPoolExecutor(max_workers=batch_size) as pool:
        futures = {pool.submit(_simulate_one, c): c for c in remaining}
        for future in as_completed(futures):
            rec = future.result()
            append_jsonl(sim_log, rec)
            logged += 1

            cid = rec.get("candidate_id", "")
            sharpe_str = f"{rec['is_sharpe']:.3f}" if rec["is_sharpe"] is not None else "N/A"
            fitness_str = f"{rec['is_fitness']:.3f}" if rec["is_fitness"] is not None else "N/A"
            print(f"[MULTI]   Done [{logged}/{total}]: sharpe={sharpe_str} fitness={fitness_str} grade={rec['grade']}")
            logging.info("Logged: candidate=%s sharpe=%s fitness=%s", cid, sharpe_str, fitness_str)

    print(f"\nFinished. Logged {logged} records → {sim_log}")
    if Path(sim_log).exists():
        n_new, n_updated = merge_session_to_master(sim_log)
        print(f"Merged to master: {n_new} new, {n_updated} updated  ({MASTER})")

if __name__ == "__main__":
    main()
