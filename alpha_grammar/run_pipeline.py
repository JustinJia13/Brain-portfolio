"""Unified alpha_grammar pipeline runner.

Continuous generate → submit → Q-table-update loop:
  1. Load or fetch + cache field clusters from Brain
  2. Load or initialise QTable (or bootstrap from sim log)
  3. Loop indefinitely (or --rounds N times):
       a. Generate N candidates with a fresh random seed
       b. Submit all candidates to Brain
       c. Wait for simulation results
       d. Update Q-table immediately
       e. Save state; advance budget schedule
  4. Ctrl+C stops cleanly after the current round finishes

All outputs land inside alpha_grammar/artifacts/ by default.

Typical invocation (run from world_quant/):

    python -m alpha_grammar.run_pipeline \\
        --fields-csv all_fields.csv \\
        --exclude-dataset analyst4 \\
        --n-samples 500 \\
        --exploration-fraction 1.0 \\
        --simulation-log alpha_grammar/artifacts/all_sim.jsonl

Run exactly 3 rounds then stop:

    python -m alpha_grammar.run_pipeline ... --rounds 3

Force re-fetch of the field catalog:

    python -m alpha_grammar.run_pipeline ... --refetch-clusters
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from alpha_grammar.fields import build_data_profiles, cluster_profiles, filter_profiles
from alpha_grammar.models import DataProfile, FieldCluster
from wq_alpha.brain.api import (
    get_datafields, sign_in, fetch_yearly_stats_with_retry,
    fetch_pnl_series, fetch_pnl_series_with_retry, fetch_competition_alpha_ids,
)
from wq_alpha.field_catalogs import default_fields_csv
from wq_alpha.robustness import attach_robustness, normalize_yearly_stats

from alpha_grammar import GrammarConfig, GrammarPlanner
from alpha_grammar.feedback import SimResult, compute_quality_weight
from alpha_grammar.qtable import (
    QTable, SkeletonCounter, bootstrap_qtable,
    compute_q_reward, extract_primary_op, is_degenerate_sim_record,
)
from alpha_grammar.probe_prior import ProbePriorGrammarSampler
from alpha_grammar.novelty import TestedFingerprintRegistry, bootstrap_tested_fingerprints
from wq_alpha.logs import MASTER, run_log_path, session_log_path, merge_session_to_master


# ---------------------------------------------------------------------------
# Default simulation settings
# ---------------------------------------------------------------------------

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

_PIPELINE_DIR = Path(__file__).parent
_ARTIFACTS_DIR = _PIPELINE_DIR / "artifacts"
DEFAULT_PROBE_PRIOR = _PIPELINE_DIR.parent / "artifacts" / "probe" / "grammar_prior_from_probe.json"
SEARCH_SCOPE = {
    "region": "USA",
    "delay": 1,
    "universe": "TOP3000",
    "instrumentType": "EQUITY",
}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run the full alpha_grammar pipeline: plan → submit → log → repeat."
    )
    p.add_argument(
        "--credentials",
        default=str(Path(__file__).parent.parent / "brain.txt"),
        help="Path to Brain credentials JSON (default: ../brain.txt relative to this script)",
    )
    p.add_argument("--dataset-id", default="", help="Brain dataset id, e.g. fundamental6 (unused when --fields-csv set)")
    p.add_argument(
        "--category",
        default="",
        help="analyst | fundamental | news | option | price_volume | social_media (unused when --fields-csv set)",
    )
    p.add_argument(
        "--delay",
        type=int,
        choices=(0, 1),
        default=None,
        help=(
            "Brain simulation delay regime. Defaults to the settings JSON value "
            "when provided, otherwise 1."
        ),
    )
    p.add_argument(
        "--fields-csv",
        default="",
        help="Path to all_fields.csv. When set, bypasses Brain API cluster fetch and enables multi-dataset mode.",
    )
    p.add_argument(
        "--alpha-binary-prob",
        type=float,
        default=0.7,
        help="Probability of alpha_binary composite pattern at K>=9 (default 0.7).",
    )
    p.add_argument(
        "--grammar-sampler",
        choices=("uniform", "probe_prior", "raw"),
        default="uniform",
        help="Field/operator sampler for grammar discovery (default: uniform). "
             "'uniform' samples every usable field with equal probability (no cluster "
             "prior) — the cluster catalog covers only ~7%% of the live field universe, "
             "so the prior was inadequate. 'probe_prior' restores the cluster-weighted prior.",
    )
    p.add_argument(
        "--probe-prior",
        default=str(DEFAULT_PROBE_PRIOR),
        help="Probe prior JSON used when --grammar-sampler=probe_prior.",
    )
    p.add_argument(
        "--prior-prob-column",
        choices=("raw_probe_probability", "mixed_probability"),
        default="raw_probe_probability",
        help="Cluster probability column for probe-prior grammar sampling.",
    )
    p.add_argument("--preferred-field-prob", type=float, default=0.70)
    p.add_argument("--sleeper-field-prob", type=float, default=0.20)
    p.add_argument(
        "--operator-prior-prob",
        type=float,
        default=0.50,
        help="Probability of using probe-family operator/window bias for compatible grammar choices.",
    )
    p.add_argument("--rounds", type=int, required=True, help="Required. Number of generate→submit→update rounds to run (0 = run until Ctrl+C).")
    p.add_argument("--budget", type=int, required=True, help="Required. Max Δk complexity budget K (paper optimal=10; use 6 for simpler expressions).")
    p.add_argument("--allow-vector", action=argparse.BooleanOptionalAction, default=True, help="Include VECTOR-type fields (default: on). They are automatically wrapped in vec_avg/vec_sum at the leaf level. Pass --no-allow-vector to restrict to MATRIX.")
    p.add_argument("--fields", nargs="+", default=[], metavar="FIELD_ID",
                   help="Explicit field_ids to include (space-separated). Bypasses type/coverage filters for these fields.")
    p.add_argument("--fields-file", default="", metavar="PATH",
                   help="Path to a text file with one field_id per line. Combined with --fields.")
    p.add_argument("--n-samples", type=int, required=True, help="Required. Candidates generated per round.")
    p.add_argument(
        "--exploration-fraction",
        type=float,
        default=0.30,
        help="Explore fraction per batch (default 0.30)",
    )
    p.add_argument(
        "--min-coverage",
        type=float,
        default=0.0,
        help="Min coverage to include a field profile (default 0.0, no filter).",
    )
    p.add_argument(
        "--exclude-dataset",
        nargs="*",
        default=[],
        metavar="DATASET_ID",
        help="Dataset IDs to exclude when using --fields-csv (e.g. --exclude-dataset analyst4).",
    )
    p.add_argument(
        "--cluster-limit",
        type=int,
        default=1000000,
        help="Max clusters to retain (default: effectively unlimited). The old 400 cap "
             "dropped ~79%% of fields; uniform sampling needs the full field pool, so keep "
             "this high unless you deliberately want a small curated catalog.",
    )
    p.add_argument(
        "--balance-clusters",
        action="store_true",
        default=False,
        help="Balance cluster selection across economic_type buckets so vector/niche fields "
             "are not crowded out by high-alpha-count MATRIX fields.",
    )
    p.add_argument(
        "--batch-size",
        type=int,
        required=True,
        help="Required. Alphas to submit per Brain API batch before waiting.",
    )
    p.add_argument(
        "--max-candidates",
        type=int,
        required=True,
        help="Required. Hard cap on candidates submitted per batch (0 = no limit).",
    )
    p.add_argument("--sleep-seconds", type=int, default=30, help="Fallback/minimum sleep between simulation poll retries")
    p.add_argument(
        "--max-submit-retries",
        type=int,
        default=5,
        help="Max submit retries per candidate",
    )
    p.add_argument(
        "--settings-json",
        default="",
        help="Optional JSON file overriding default simulation settings",
    )
    p.add_argument(
        "--simulation-log",
        default="",
        help="Simulation log path (default: alpha_grammar/<dataset-id>_sim.jsonl)",
    )
    p.add_argument(
        "--run-log",
        default="",
        help="Runner text log path (default: alpha_grammar/<dataset-id>_runner.log)",
    )
    p.add_argument(
        "--rebuild-qtable",
        action="store_true",
        help="Discard any saved Q-table and rebuild from scratch using existing sim log",
    )
    p.add_argument(
        "--shadow-gate",
        action="store_true",
        help="Log-only: run the alpha_metric sim-slot gate alongside the Q-table and "
             "record its keep/kill decision per candidate to artifacts/gate_shadow.jsonl "
             "WITHOUT changing what gets simulated. Fail-safe (never breaks the run).",
    )
    p.add_argument(
        "--gate-mode",
        choices=["shadow", "reliable", "full", "auto"],
        default="shadow",
        help="alpha_metric gate authority (implies --shadow-gate logging). "
             "shadow=log only; reliable=LIVE dedup c_max + turnover hard-kill (adopt-now); "
             "full=quality surrogate authoritative; auto=read alpha_metric/cutover_state.json "
             "(self-healing rollback). Always fail-safe -> simulate on any error.",
    )
    p.add_argument(
        "--gate-beta",
        type=float,
        default=None,
        help="Quality-UCB width for the gate (default 2.5). Lower = more aggressive "
             "skipping / higher hit-rate among simulated but lower recall of good "
             "alphas; higher (e.g. 3.0) = safer. Only meaningful with --gate-mode full.",
    )
    p.add_argument(
        "--refetch-clusters",
        action="store_true",
        help="Force re-fetch of field catalog even if a cache exists",
    )
    p.add_argument(
        "--max-retries",
        type=int,
        default=20,
        help="Max retries for Brain API field fetch (default 20)",
    )
    p.add_argument(
        "--competition-id",
        default="IQC2026S1",
        help="Competition ID for portfolio self-correlation (default: IQC2026S1)",
    )
    p.add_argument(
        "--skip-portfolio-pnl",
        action="store_true",
        help="Skip loading portfolio PnL at startup (disables self-correlation computation)",
    )
    p.add_argument(
        "--fetch-sleep-seconds",
        type=float,
        default=60.0,
        help="Sleep between retries when fetching field catalog (default 60.0)",
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Field pinlist helper
# ---------------------------------------------------------------------------

def _resolve_include_fields(args: argparse.Namespace) -> set[str] | None:
    """Return explicit field_id set from --fields / --fields-file, or None if neither given."""
    ids: set[str] = set(args.fields or [])
    if args.fields_file:
        path = Path(args.fields_file)
        with path.open() as f:
            for line in f:
                fid = line.strip()
                if fid and not fid.startswith("#"):
                    ids.add(fid)
    return ids if ids else None


# ---------------------------------------------------------------------------
# Cluster helpers
# ---------------------------------------------------------------------------

def _cluster_cache_path(dataset_id: str) -> Path:
    return _ARTIFACTS_DIR / f"{dataset_id}_clusters.json"


def _numeric_allowed_types(allow_vector: bool) -> tuple[str, ...]:
    return ("MATRIX", "VECTOR") if allow_vector else ("MATRIX",)


def _filter_clusters_by_field_types(
    clusters: list[FieldCluster],
    allowed_types: tuple[str, ...],
) -> list[FieldCluster]:
    allowed = {field_type.upper() for field_type in allowed_types}
    filtered: list[FieldCluster] = []
    for cluster in clusters:
        members = tuple(
            member for member in cluster.members
            if member.field_type.upper() in allowed
        )
        if not members:
            continue
        champion = cluster.champion if cluster.champion in members else members[0]
        sleeper = cluster.sleeper if cluster.sleeper in members and cluster.sleeper != champion else None
        if sleeper is None:
            sleeper = next((member for member in members if member != champion), None)
        filtered.append(
            FieldCluster(
                cluster_id=cluster.cluster_id,
                category=cluster.category,
                dataset_id=cluster.dataset_id,
                label=cluster.label,
                champion=champion,
                sleeper=sleeper,
                members=members,
            )
        )
    return filtered


def _group_profiles_from_clusters(clusters: list[FieldCluster]) -> list[DataProfile]:
    seen: set[str] = set()
    profiles: list[DataProfile] = []
    for cluster in clusters:
        for member in cluster.members:
            if member.field_type.upper() != "GROUP" or member.field_id in seen:
                continue
            profiles.append(member)
            seen.add(member.field_id)
    return profiles


def fetch_and_cache_clusters(args: argparse.Namespace, session) -> list[FieldCluster]:
    search_scope = {**SEARCH_SCOPE, "delay": int(args.delay)}
    raw_fields = get_datafields(
        session,
        search_scope,
        dataset_id=args.dataset_id,
        max_retries=args.max_retries,
        default_sleep_seconds=args.fetch_sleep_seconds,
    )
    profiles = build_data_profiles(raw_fields, category=args.category, dataset_id=args.dataset_id)
    allowed_types = _numeric_allowed_types(args.allow_vector)
    profiles = filter_profiles(profiles, allowed_types=allowed_types, min_coverage=0.0,
                               include_fields=_resolve_include_fields(args))
    profiles = [p for p in profiles if p.field_type.upper() in allowed_types]
    clusters = cluster_profiles(profiles, max_clusters=args.cluster_limit,
                               balance_by_economic_type=args.balance_clusters)
    cache = _cluster_cache_path(args.dataset_id)
    _save_clusters(clusters, cache)
    print(f"Saved cluster cache: {cache}")
    return clusters


def _save_clusters(clusters: list[FieldCluster], path: Path) -> None:
    records = []
    for c in clusters:
        members = [
            {
                "field_id": m.field_id,
                "category": m.category,
                "dataset_id": m.dataset_id,
                "dataset_name": m.dataset_name,
                "subcategory_id": m.subcategory_id,
                "subcategory_name": m.subcategory_name,
                "field_type": m.field_type,
                "description": m.description,
                "coverage": m.coverage,
                "date_coverage": m.date_coverage,
                "user_count": m.user_count,
                "alpha_count": m.alpha_count,
                "economic_type": m.economic_type,
                "is_size_proxy": m.is_size_proxy,
            }
            for m in c.members
        ]
        records.append(
            {
                "cluster_id": c.cluster_id,
                "category": c.category,
                "dataset_id": c.dataset_id,
                "label": c.label,
                "champion_field_id": c.champion.field_id,
                "sleeper_field_id": c.sleeper.field_id if c.sleeper else None,
                "members": members,
            }
        )
    with path.open("w") as f:
        json.dump(records, f, indent=2)


def _load_clusters(path: Path) -> list[FieldCluster]:
    with path.open() as f:
        records = json.load(f)
    clusters = []
    for rec in records:
        members = tuple(
            DataProfile(
                field_id=m["field_id"],
                category=m["category"],
                dataset_id=m["dataset_id"],
                dataset_name=m.get("dataset_name", ""),
                subcategory_id=m.get("subcategory_id", ""),
                subcategory_name=m.get("subcategory_name", ""),
                field_type=m.get("field_type", ""),
                description=m.get("description", ""),
                coverage=m.get("coverage"),
                date_coverage=m.get("date_coverage"),
                user_count=m.get("user_count", 0),
                alpha_count=m.get("alpha_count", 0),
                economic_type=m.get("economic_type", "OTHER"),
                is_size_proxy=bool(m.get("is_size_proxy", False)),
            )
            for m in rec["members"]
        )
        by_id = {m.field_id: m for m in members}
        champion = by_id.get(rec["champion_field_id"], members[0] if members else None)
        sleeper_id = rec.get("sleeper_field_id")
        sleeper = by_id.get(sleeper_id) if sleeper_id else None
        if champion is None:
            continue
        clusters.append(
            FieldCluster(
                cluster_id=rec["cluster_id"],
                category=rec["category"],
                dataset_id=rec["dataset_id"],
                label=rec["label"],
                champion=champion,
                sleeper=sleeper,
                members=members,
            )
        )
    return clusters


# ---------------------------------------------------------------------------
# Simulation log helpers
# ---------------------------------------------------------------------------

def seed_feedback_from_log(store: "Any", sim_log_path: Path) -> int:
    """Archived — superseded by bootstrap_qtable().  Kept for compatibility."""
    if not sim_log_path.exists():
        return 0
    count = 0
    with sim_log_path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if rec.get("search_lane") != "grammar":
                continue
            if rec.get("status") != "done":
                continue
            if is_degenerate_sim_record(rec):
                continue
            is_sharpe = float(rec.get("is_sharpe") or 0.0)
            is_fitness = float(rec.get("is_fitness") or 0.0)
            raw_turnover = rec.get("is_turnover")
            is_turnover = float(raw_turnover) if raw_turnover is not None else None
            concentrated_weight_ok = bool(rec.get("is_concentrated_weight_ok", True))
            sub_universe_sharpe_ok = bool(rec.get("is_sub_universe_sharpe_ok", True))
            grade = rec.get("grade") or ""
            passed = grade.upper() == "PASS" or is_sharpe >= 1.25
            result = SimResult(
                candidate_id=rec.get("candidate_id", ""),
                expression=rec.get("expression", ""),
                cluster_ids=tuple(str(c) for c in (rec.get("cluster_ids") or [])),
                is_sharpe=is_sharpe,
                is_fitness=is_fitness,
                passed=passed,
                is_turnover=is_turnover,
                concentrated_weight_ok=concentrated_weight_ok,
                sub_universe_sharpe_ok=sub_universe_sharpe_ok,
            )
            store.update(result)
            count += 1
    store.end_of_batch()
    return count


def load_completed_keys(sim_log: str) -> tuple[set[str], set[str]]:
    """Return candidate_ids and expressions already in the simulation log."""
    candidate_ids: set[str] = set()
    expressions: set[str] = set()
    log_path = Path(sim_log)
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


def filter_remaining(
    candidates: list[dict[str, Any]],
    sim_log: str,
    max_candidates: int,
) -> list[dict[str, Any]]:
    done_ids, done_exprs = load_completed_keys(sim_log)
    remaining = [
        c for c in candidates
        if c.get("candidate_id") not in done_ids
        and c.get("expression") not in done_exprs
    ]
    if max_candidates > 0:
        remaining = remaining[:max_candidates]
    return remaining


def append_jsonl(path: str, record: dict[str, Any]) -> None:
    with open(path, "a") as f:
        f.write(json.dumps(record) + "\n")
        f.flush()


# ---------------------------------------------------------------------------
# Brain submission helpers
# ---------------------------------------------------------------------------

def _reauth_if_needed(session, response, credentials_path: str) -> bool:
    """Re-authenticate the session if Brain returned a 401.

    Returns True if re-auth was attempted (caller should retry the request),
    False if the response does not indicate an auth failure.
    Brain sessions expire after several hours; polling a long simulation run
    will eventually hit this.
    """
    from wq_alpha.brain.api import authenticate_session, is_auth_expired_response

    if is_auth_expired_response(response):
        logging.warning("Brain session expired - waiting for shared re-authentication ...")
        if not authenticate_session(session, credentials_path, logger=logging):
            logging.error("Re-authentication did not complete; request will retry after backoff.")
        return True
    return False


def submit_alpha(
    session,
    expression: str,
    settings: dict[str, Any],
    *,
    max_retries: int,
    sleep_seconds: int,
    credentials_path: str = "",
) -> str | None:
    payload = {"type": "REGULAR", "settings": settings, "regular": expression}
    for attempt in range(max_retries):
        try:
            response = session.post(
                "https://api.worldquantbrain.com/simulations", json=payload
            )
            if "Location" in response.headers:
                return response.headers["Location"]
            # Log why submit failed so future debugging is easier
            logging.warning(
                "Submit attempt %d/%d — no Location header: status=%s body=%s",
                attempt + 1, max_retries, response.status_code, response.text[:200],
            )
            if credentials_path and _reauth_if_needed(session, response, credentials_path):
                continue  # auth helper handles any needed backoff
        except Exception as exc:
            logging.warning("Submit attempt %d/%d — exception: %s", attempt + 1, max_retries, exc)
        time.sleep(sleep_seconds)
    return None


def wait_for_completion(
    session,
    progress_url: str,
    *,
    sleep_seconds: int,
    credentials_path: str = "",
) -> dict[str, Any]:
    while True:
        try:
            response = session.get(progress_url)
            retry_after = float(response.headers.get("Retry-After", 0))
            if retry_after == 0:
                if credentials_path and _reauth_if_needed(session, response, credentials_path):
                    # Auth-expired responses are not completion payloads.
                    continue
                body = response.json()
                return body
            time.sleep(max(retry_after, sleep_seconds))
        except Exception as exc:
            logging.warning("wait_for_completion error: %s", exc)
            time.sleep(sleep_seconds)


def get_alpha_detail(session, alpha_id: str) -> dict[str, Any]:
    try:
        return session.get(f"https://api.worldquantbrain.com/alphas/{alpha_id}").json()
    except Exception:
        return {}


# ---------------------------------------------------------------------------
# PnL-based self-correlation helpers
# ---------------------------------------------------------------------------

def _fetch_pnl_with_backoff(
    session,
    alpha_id: str,
    *,
    timeout: float = 20.0,
    max_attempts: int | None = None,
) -> dict[str, float]:
    """Fetch PnL retrying at 1s intervals until data is returned.

    Retries indefinitely (max_attempts=None) on both rate limits (None) and
    async delay (empty {}).  Pass max_attempts=N to cap for portfolio loading.
    """
    attempt = 0
    while max_attempts is None or attempt < max_attempts:
        attempt += 1
        result = fetch_pnl_series(session, alpha_id, timeout=timeout)
        if result:
            return result
        logging.warning("[PnL] not ready / rate limited for %s — waiting 1s (attempt %d)", alpha_id, attempt)
        print(f"[PnL] not ready / rate limited — waiting 1s (attempt {attempt}) ...")
        time.sleep(1)
    logging.warning("[PnL] exhausted retries for %s", alpha_id)
    return {}


def _portfolio_self_corr_info_template() -> dict[str, Any]:
    return {
        "max_self_correlation": None,
        "max_self_correlation_alpha_id": None,
        "max_self_correlation_alpha_sharpe": None,
        "max_self_correlation_alpha_fitness": None,
    }


def _extract_is_perf(alpha_detail: dict[str, Any]) -> tuple[float | None, float | None]:
    is_stats = alpha_detail.get("is") or {}
    raw_sharpe = is_stats.get("sharpe")
    raw_fitness = is_stats.get("fitness")
    sharpe = float(raw_sharpe) if raw_sharpe is not None else None
    fitness = float(raw_fitness) if raw_fitness is not None else None
    return sharpe, fitness


def load_portfolio_alpha_data(
    session,
    competition_id: str = "IQC2026S1",
    *,
    max_pnl_attempts: int | None = 10,
) -> dict[str, dict[str, Any]]:
    """Fetch PnL plus IS performance metadata for all competition alphas.

    Returns {alpha_id: {"pnl": {...}, "is_sharpe": float|None, "is_fitness": float|None}}.
    Alphas with no PnL are skipped.
    """
    alpha_ids = fetch_competition_alpha_ids(session, competition_id)
    print(f"[portfolio] Fetching PnL for {len(alpha_ids)} competition alphas ({competition_id}) ...")
    portfolio: dict[str, dict[str, Any]] = {}
    for aid in alpha_ids:
        pnl = _fetch_pnl_with_backoff(session, aid, max_attempts=max_pnl_attempts)
        if not pnl:
            continue
        detail = get_alpha_detail(session, aid)
        is_sharpe, is_fitness = _extract_is_perf(detail)
        settings = detail.get("settings") or {}
        portfolio[aid] = {
            "pnl": pnl,
            "is_sharpe": is_sharpe,
            "is_fitness": is_fitness,
            "delay": settings.get("delay"),
            "settings": settings,
            "expression": (detail.get("regular") or {}).get("code"),
            "status": detail.get("status"),
            "stage": detail.get("stage"),
        }
    print(f"[portfolio] Loaded PnL for {len(portfolio)}/{len(alpha_ids)} alphas.")
    return portfolio


def load_portfolio_pnls(
    session,
    competition_id: str = "IQC2026S1",
    *,
    max_pnl_attempts: int | None = 10,
) -> dict[str, dict[str, float]]:
    """Compatibility wrapper returning only competition PnL series."""
    alpha_data = load_portfolio_alpha_data(
        session,
        competition_id=competition_id,
        max_pnl_attempts=max_pnl_attempts,
    )
    return {
        aid: data["pnl"]
        for aid, data in alpha_data.items()
        if isinstance(data.get("pnl"), dict) and data.get("pnl")
    }


def _pnl_to_daily_returns(pnl: dict[str, float]) -> dict[str, float]:
    """Convert cumulative PnL (levels) to daily returns (first differences).

    The Brain /recordsets/pnl endpoint returns cumulative PnL in dollar terms.
    Correlation must be computed on daily changes, not levels, to avoid
    spurious correlation from the unit root.
    """
    keys = sorted(pnl)
    if len(keys) < 2:
        return {}
    return {keys[i]: pnl[keys[i]] - pnl[keys[i - 1]] for i in range(1, len(keys))}


def compute_max_self_corr(
    candidate_pnl: dict[str, float],
    portfolio_pnls: dict[str, dict[str, float]],
) -> float | None:
    """Compatibility wrapper returning only the signed max self-correlation."""
    portfolio_alpha_data = {
        aid: {"pnl": pnl, "is_sharpe": None, "is_fitness": None}
        for aid, pnl in portfolio_pnls.items()
    }
    return compute_max_self_corr_info(candidate_pnl, portfolio_alpha_data)["max_self_correlation"]


def compute_max_self_corr_info(
    candidate_pnl: dict[str, float],
    portfolio_alpha_data: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Return signed max self-correlation plus matched portfolio benchmark fields.

    Inputs are cumulative PnL series ({date: cumulative_pnl}); they are
    first-differenced to daily returns before correlation is computed.

    Returns the SIGNED maximum (not abs): a negatively-correlated alpha
    passes the self-correlation check because it adds diversification.
    Brain uses signed max for the self-correlation gate.

    Aligns on shared dates. If candidate_pnl is empty or no portfolio alpha
    shares enough dates (min 20) with the candidate, all returned fields are
    None.
    """
    info = _portfolio_self_corr_info_template()
    if not candidate_pnl or not portfolio_alpha_data:
        return info
    try:
        import numpy as np
    except ImportError:
        return info

    candidate_returns = _pnl_to_daily_returns(candidate_pnl)
    if not candidate_returns:
        return info

    candidate_dates = set(candidate_returns)
    max_corr: float | None = None
    matched_alpha_id: str | None = None
    matched_alpha_sharpe: float | None = None
    matched_alpha_fitness: float | None = None

    for alpha_id, alpha_data in portfolio_alpha_data.items():
        port_pnl = alpha_data.get("pnl") or {}
        port_returns = _pnl_to_daily_returns(port_pnl)
        shared = sorted(candidate_dates & set(port_returns))
        if len(shared) < 20:
            continue
        a = np.array([candidate_returns[d] for d in shared], dtype=float)
        b = np.array([port_returns[d] for d in shared], dtype=float)
        # Skip if either series is constant (std=0)
        if a.std() < 1e-12 or b.std() < 1e-12:
            continue
        corr = float(np.corrcoef(a, b)[0, 1])
        if max_corr is None or corr > max_corr:
            max_corr = corr
            matched_alpha_id = alpha_id
            raw_sharpe = alpha_data.get("is_sharpe")
            raw_fitness = alpha_data.get("is_fitness")
            matched_alpha_sharpe = float(raw_sharpe) if raw_sharpe is not None else None
            matched_alpha_fitness = float(raw_fitness) if raw_fitness is not None else None

    info["max_self_correlation"] = max_corr
    info["max_self_correlation_alpha_id"] = matched_alpha_id
    info["max_self_correlation_alpha_sharpe"] = matched_alpha_sharpe
    info["max_self_correlation_alpha_fitness"] = matched_alpha_fitness
    return info




def _parse_is_checks(is_checks: list[dict[str, Any]]) -> dict[str, bool]:
    """Return {check_name: passed} for each entry in the is.checks list."""
    return {c["name"]: c.get("result", "FAIL") == "PASS" for c in (is_checks or [])}


def _parse_check_values(is_checks: list[dict[str, Any]]) -> dict[str, float | None]:
    """Return {check_name: value} for checks that carry a numeric value field."""
    return {
        c["name"]: float(c["value"]) if c.get("value") is not None else None
        for c in (is_checks or [])
    }


def build_sim_record(
    candidate: dict[str, Any],
    settings: dict[str, Any],
    progress_url: str | None,
    final_progress: dict[str, Any],
    alpha_id: str | None,
    status: str,
    alpha_detail: dict[str, Any],
    yearly_stats: dict[int, dict] | None = None,
) -> dict[str, Any]:
    final_progress = final_progress or {}
    alpha_detail = alpha_detail or {}
    is_stats = alpha_detail.get("is") or {}
    os_stats = alpha_detail.get("os") or {}
    raw_checks = is_stats.get("checks") or []
    checks = _parse_is_checks(raw_checks)
    check_values = _parse_check_values(raw_checks)
    record = {
        "logged_at": datetime.now(timezone.utc).isoformat(),
        "candidate_id": candidate.get("candidate_id"),
        "family_id": candidate.get("family_id"),
        "intention_id": candidate.get("intention_id"),
        "intention_label": candidate.get("intention_label"),
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
        "template_id": candidate.get("template_id"),
        "tree_depth": candidate.get("tree_depth"),
        "priority_score": candidate.get("priority_score"),
        "status": status,
        "progress_url": progress_url,
        "alpha_id": alpha_id,
        "settings": settings,
        "raw_progress_result": final_progress,
        "is_sharpe": is_stats.get("sharpe"),
        "is_fitness": is_stats.get("fitness"),
        "is_turnover": is_stats.get("turnover"),
        "is_returns": is_stats.get("returns"),
        "is_drawdown": is_stats.get("drawdown"),
        "is_margin": is_stats.get("margin"),
        "is_ic": is_stats.get("ic"),
        "is_concentrated_weight_ok": checks.get("CONCENTRATED_WEIGHT", True),
        "is_sub_universe_sharpe_ok": checks.get("LOW_SUB_UNIVERSE_SHARPE", True),
        "max_self_correlation": is_stats.get("selfCorrelation"),
        "max_self_correlation_alpha_id": None,
        "max_self_correlation_alpha_sharpe": None,
        "max_self_correlation_alpha_fitness": None,
        "max_concentration": check_values.get("CONCENTRATED_WEIGHT"),
        "os_sharpe": os_stats.get("sharpe"),
        "os_fitness": os_stats.get("fitness"),
        "grade": alpha_detail.get("grade"),
        "checks": is_stats.get("checks"),
        "date_created": alpha_detail.get("dateCreated"),
        "yearly_stats": yearly_stats or {},
        "oos_sharpe_2023": (normalize_yearly_stats(yearly_stats).get(2023) or {}).get("sharpe"),
        "oos_fitness_2023": (normalize_yearly_stats(yearly_stats).get(2023) or {}).get("fitness"),
        "raw_alpha_detail": alpha_detail,
        "candidate_spec": candidate,
    }
    return attach_robustness(record)


# ---------------------------------------------------------------------------
# Per-batch submit loop
# ---------------------------------------------------------------------------

def run_batch_submission(
    session,
    candidates: list[dict[str, Any]],
    settings: dict[str, Any],
    sim_log: str,
    run_log: str,
    *,
    batch_size: int,
    max_candidates: int,
    sleep_seconds: int,
    max_submit_retries: int,
    credentials_path: str = "",
    portfolio_alpha_data: dict[str, dict[str, Any]] | None = None,
    label: str = "GRAMMAR",
) -> tuple[list[SimResult], list[dict]]:
    """Submit candidates using a rolling window of batch_size concurrent simulations.

    Returns (sim_results, raw_records) where raw_records are the full sim
    record dicts (as logged to sim.jsonl) for template extraction.

    label: prefix used in progress output lines (default "GRAMMAR", use "LLM" from LLM runner).
    """
    remaining = filter_remaining(candidates, sim_log, max_candidates)
    print(f"  Candidates in plan:  {len(candidates)}")
    print(f"  Remaining to submit: {len(remaining)}")

    if not remaining:
        return [], []

    total = len(remaining)

    def _simulate_one(candidate: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
        expression = candidate["expression"]
        cid_short = (candidate.get("candidate_id") or "")[:12]
        print(f"[{label}]   Submit: {cid_short} | {expression[:60]}")
        url = submit_alpha(
            session, expression, settings,
            max_retries=max_submit_retries,
            sleep_seconds=sleep_seconds,
            credentials_path=credentials_path,
        )
        if not url:
            logging.warning("Submit failed: %s", expression)
            return candidate, build_sim_record(candidate, settings, None, {}, None, "submit_failed", {})

        print(f"[{label}]   Wait:   {cid_short} | {expression[:60]}")
        final = wait_for_completion(session, url, sleep_seconds=sleep_seconds,
                                    credentials_path=credentials_path)
        alpha_id = final.get("alpha")
        if alpha_id:
            detail = get_alpha_detail(session, alpha_id)
            yearly = fetch_yearly_stats_with_retry(session, alpha_id)
            computed_self_corr_info = _portfolio_self_corr_info_template()
            if portfolio_alpha_data:
                candidate_pnl = fetch_pnl_series_with_retry(session, alpha_id, max_attempts=8)
                computed_self_corr_info = compute_max_self_corr_info(candidate_pnl, portfolio_alpha_data)
            status = "done"
        else:
            detail, yearly = {}, {}
            computed_self_corr_info = _portfolio_self_corr_info_template()
            status = "no_alpha_id"
            logging.warning("No alpha_id: %s | response=%s", candidate.get("candidate_id"), final)

        record = build_sim_record(candidate, settings, url, final, alpha_id, status, detail, yearly)
        if computed_self_corr_info["max_self_correlation"] is not None:
            record.update(computed_self_corr_info)
        return candidate, record

    sim_results: list[SimResult] = []
    raw_records: list[dict] = []
    logged_count = 0

    with ThreadPoolExecutor(max_workers=batch_size) as pool:
        futures = {pool.submit(_simulate_one, c): c for c in remaining}
        for future in as_completed(futures):
            candidate, record = future.result()
            append_jsonl(sim_log, record)
            raw_records.append(record)
            logged_count += 1

            alpha_id = record.get("alpha_id")
            sharpe = record["is_sharpe"]
            fitness = record["is_fitness"]
            grade = record.get("grade", "")
            sharpe_str = f"{sharpe:.3f}" if sharpe is not None else "N/A"
            fitness_str = f"{fitness:.3f}" if fitness is not None else "N/A"
            raw_self_corr = record.get("max_self_correlation")
            corr_str = f"{raw_self_corr:.3f}" if raw_self_corr is not None else "N/A"
            print(f"[{label}]   Done [{logged_count}/{total}]: sharpe={sharpe_str} fitness={fitness_str} grade={grade} self_corr={corr_str}")
            logging.info("Logged: candidate=%s | alpha=%s | sharpe=%s | fitness=%s",
                         candidate.get("candidate_id"), alpha_id, sharpe_str, fitness_str)

            passed = (grade or "").upper() == "PASS" or (sharpe is not None and sharpe >= 1.25)
            raw_turnover = record.get("is_turnover")
            is_turnover = float(raw_turnover) if raw_turnover is not None else None
            concentrated_weight_ok = bool(record.get("is_concentrated_weight_ok", True))
            sub_universe_sharpe_ok = bool(record.get("is_sub_universe_sharpe_ok", True))
            raw_max_conc = record.get("max_concentration")
            sim_results.append(
                SimResult(
                    candidate_id=candidate.get("candidate_id", ""),
                    expression=candidate["expression"],
                    cluster_ids=tuple(str(c) for c in (candidate.get("cluster_ids") or [])),
                    is_sharpe=float(sharpe or 0.0),
                    is_fitness=float(fitness or 0.0),
                    passed=passed,
                    is_turnover=is_turnover,
                    concentrated_weight_ok=concentrated_weight_ok,
                    sub_universe_sharpe_ok=sub_universe_sharpe_ok,
                    yearly_stats=record.get("yearly_stats") or None,
                    max_self_correlation=float(raw_self_corr) if raw_self_corr is not None else None,
                    max_self_correlation_alpha_id=str(record["max_self_correlation_alpha_id"]) if record.get("max_self_correlation_alpha_id") is not None else None,
                    max_self_correlation_alpha_sharpe=float(record["max_self_correlation_alpha_sharpe"]) if record.get("max_self_correlation_alpha_sharpe") is not None else None,
                    max_self_correlation_alpha_fitness=float(record["max_self_correlation_alpha_fitness"]) if record.get("max_self_correlation_alpha_fitness") is not None else None,
                    max_concentration=float(raw_max_conc) if raw_max_conc is not None else None,
                )
            )

    print(f"  Logged {logged_count} records to {sim_log}")
    return sim_results, raw_records


# ---------------------------------------------------------------------------
# CSV-based cluster loading (multi-dataset mode)
# ---------------------------------------------------------------------------

def load_clusters_from_csv(
    path: str,
    max_clusters: int = 400,
    exclude_datasets: list[str] | None = None,
    allow_vector: bool = False,
    include_fields: set[str] | None = None,
    balance_by_economic_type: bool = False,
) -> list[FieldCluster]:
    """Build FieldCluster list from all_fields.csv, bypassing Brain API.

    Enables multi-dataset mode: all datasets in the CSV are merged into one
    cluster pool so the grammar can generate cross-dataset expressions.

    Parameters
    ----------
    exclude_datasets:
        Dataset IDs to skip (e.g. ["analyst4"]).
    """
    import csv as _csv
    excluded = set(exclude_datasets or [])
    raw_fields: list[dict] = []
    skipped_datasets: set[str] = set()
    with open(path) as f:
        for row in _csv.DictReader(f):
            ds_id = row.get("dataset_id", "")
            if ds_id in excluded:
                skipped_datasets.add(ds_id)
                continue
            raw_fields.append({
                "id":          row["field_id"],
                "description": row.get("field_name", ""),
                "type":        row.get("type", "MATRIX"),
                "coverage":    float(row.get("coverage") or 0.0),
                "category":    row.get("category", ""),
                "subcategory": {"name": row.get("subcategory", "")},
                "alphaCount":  int(row.get("alpha_count") or 0),
                "userCount":   int(row.get("user_count") or 0),
                "dataset":     {"id": ds_id},
            })
    if skipped_datasets:
        print(f"  Excluded datasets: {sorted(skipped_datasets)}")
    profiles = build_data_profiles(raw_fields)
    allowed_types = _numeric_allowed_types(allow_vector)
    profiles = filter_profiles(profiles, allowed_types=allowed_types, min_coverage=0.0,
                               include_fields=include_fields)
    profiles = [p for p in profiles if p.field_type.upper() in allowed_types]
    return cluster_profiles(profiles, max_clusters=max_clusters,
                            balance_by_economic_type=balance_by_economic_type)


def load_group_profiles_from_csv(
    path: str,
    exclude_datasets: list[str] | None = None,
) -> list[DataProfile]:
    """Load GROUP-typed fields for use only as group arguments."""
    import csv as _csv

    excluded = set(exclude_datasets or [])
    raw_fields: list[dict[str, Any]] = []
    with open(path) as f:
        for row in _csv.DictReader(f):
            ds_id = row.get("dataset_id", "")
            if ds_id in excluded:
                continue
            if (row.get("type") or "").upper() != "GROUP":
                continue
            raw_fields.append({
                "id":          row["field_id"],
                "description": row.get("field_name", ""),
                "type":        row.get("type", "GROUP"),
                "coverage":    float(row.get("coverage") or 0.0),
                "category":    row.get("category", ""),
                "subcategory": {"name": row.get("subcategory", "")},
                "alphaCount":  int(row.get("alpha_count") or 0),
                "userCount":   int(row.get("user_count") or 0),
                "dataset":     {"id": ds_id},
            })
    profiles = build_data_profiles(raw_fields)
    profiles = filter_profiles(profiles, allowed_types=("GROUP",), min_coverage=0.0)
    return [p for p in profiles if p.field_type.upper() == "GROUP"]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    # Auto-resolve fields CSV from credentials when not explicitly passed
    if not args.fields_csv:
        resolved = default_fields_csv(args.credentials)
        if resolved.exists():
            args.fields_csv = str(resolved)

    _ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    tag = "all" if args.fields_csv else (args.dataset_id or "unknown")
    # Session log: timestamped, written this run only, merged to master at end
    _session_log = session_log_path("grammar")
    sim_log = args.simulation_log or str(_session_log)
    run_log = args.run_log or str(run_log_path("grammar"))
    cluster_cache = _ARTIFACTS_DIR / f"{tag}_clusters.json"
    # Q-table paths are stable (not per-session)

    logging.basicConfig(
        filename=run_log,
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    # ── 1. Load simulation settings ─────────────────────────────────────
    if args.settings_json:
        with open(args.settings_json) as f:
            settings = json.load(f)
    else:
        settings = dict(DEFAULT_SIMULATION_SETTINGS)
    if args.delay is not None:
        settings["delay"] = int(args.delay)
    else:
        args.delay = int(settings.get("delay", DEFAULT_SIMULATION_SETTINGS["delay"]))

    # ── 2. Sign in to Brain ──────────────────────────────────────────────
    session = sign_in(args.credentials)

    # ── 2b. Load portfolio context for self-correlation computation ──────
    portfolio_alpha_data: dict[str, dict[str, Any]] = {}
    if not args.skip_portfolio_pnl:
        portfolio_alpha_data = load_portfolio_alpha_data(session, competition_id=args.competition_id)

    # ── 3. Load or fetch clusters ────────────────────────────────────────
    group_profiles: list[DataProfile] = []
    if args.fields_csv:
        if not args.refetch_clusters and cluster_cache.exists() and not args.exclude_dataset:
            print(f"Loading all-dataset clusters from cache: {cluster_cache}")
            cached_clusters = _load_clusters(cluster_cache)
            group_profiles = _group_profiles_from_clusters(cached_clusters)
            clusters = _filter_clusters_by_field_types(
                cached_clusters,
                _numeric_allowed_types(args.allow_vector),
            )
            if not group_profiles:
                group_profiles = load_group_profiles_from_csv(args.fields_csv)
        elif not args.refetch_clusters and cluster_cache.exists() and args.exclude_dataset:
            print(f"Loading all-dataset clusters from cache: {cluster_cache}")
            excluded = set(args.exclude_dataset)
            cached_clusters = [c for c in _load_clusters(cluster_cache) if c.dataset_id not in excluded]
            group_profiles = _group_profiles_from_clusters(cached_clusters)
            clusters = _filter_clusters_by_field_types(
                cached_clusters,
                _numeric_allowed_types(args.allow_vector),
            )
            if not group_profiles:
                group_profiles = load_group_profiles_from_csv(args.fields_csv, args.exclude_dataset or [])
            print(f"  Excluded datasets: {sorted(excluded)}")
        else:
            print(f"Building multi-dataset clusters from {args.fields_csv} ...")
            group_profiles = load_group_profiles_from_csv(
                args.fields_csv,
                args.exclude_dataset or [],
            )
            clusters = load_clusters_from_csv(
                args.fields_csv,
                max_clusters=args.cluster_limit,
                exclude_datasets=args.exclude_dataset or [],
                allow_vector=args.allow_vector,
                include_fields=_resolve_include_fields(args),
                balance_by_economic_type=args.balance_clusters,
            )
            _save_clusters(clusters, cluster_cache)
            print(f"Saved cluster cache: {cluster_cache}")
    elif not args.refetch_clusters and cluster_cache.exists():
        print(f"Loading clusters from cache: {cluster_cache}")
        clusters = _filter_clusters_by_field_types(
            _load_clusters(cluster_cache),
            _numeric_allowed_types(args.allow_vector),
        )
    else:
        print(f"Fetching fields from Brain API for {args.dataset_id} ...")
        clusters = fetch_and_cache_clusters(args, session)
    print(f"Clusters available: {len(clusters)}")
    if group_profiles:
        print(f"Group field pool: {len(group_profiles)} GROUP fields available for group arguments")

    # Grammar samples uniformly by dataset → cluster → field.
    # No flat pool needed — clusters are passed directly to the grammar.
    total_members = sum(len(c.members) for c in clusters)
    print(f"Cluster pool: {len(clusters)} clusters, {total_members} total member fields")

    # ── 4. Initialise Q-table + skeleton counter ─────────────────────────
    qtable_path = _ARTIFACTS_DIR / f"{tag}_qtable.json"
    skeleton_path = _ARTIFACTS_DIR / f"{tag}_skeleton_counts.json"

    skeleton_counter = SkeletonCounter()

    if getattr(args, "rebuild_qtable", False) or not qtable_path.exists():
        qtable = QTable()
        n_bootstrapped = bootstrap_qtable(MASTER, qtable, skeleton_counter=skeleton_counter)
        if n_bootstrapped:
            qtable.save(str(qtable_path))
            skeleton_counter.save(str(skeleton_path))
            print(
                f"Bootstrapped Q-table from {MASTER}: "
                f"{n_bootstrapped} records, "
                f"{len(qtable.level1)} L1 entries, {len(qtable.level2)} L2 entries, "
                f"{len(skeleton_counter)} unique skeletons"
            )
            n_pruned1 = len(qtable.pruned_level1())
            n_pruned2 = len(qtable.pruned_level2())
            if n_pruned1 or n_pruned2:
                print(f"  Pruned on bootstrap: {n_pruned1} L1, {n_pruned2} L2 combinations")
        else:
            print("Starting with empty Q-table (no existing sim records found)")
    else:
        qtable = QTable.load(str(qtable_path))
        if skeleton_path.exists():
            skeleton_counter = SkeletonCounter.load(str(skeleton_path))
        n_pruned1 = len(qtable.pruned_level1())
        n_pruned2 = len(qtable.pruned_level2())
        print(
            f"Loaded Q-table: {len(qtable.level1)} L1 entries, {len(qtable.level2)} L2 entries  "
            f"({n_pruned1} L1 pruned, {n_pruned2} L2 pruned)"
        )
        print(f"Skeleton counter: {len(skeleton_counter)} unique skeletons tracked")

    fingerprint_path = _ARTIFACTS_DIR / f"{tag}_tested_fingerprints.json"
    if getattr(args, "rebuild_qtable", False) or not fingerprint_path.exists():
        fingerprint_registry = TestedFingerprintRegistry()
        fingerprint_sources: list[Path] = []
        seen_sources: set[str] = set()
        for raw_path in (MASTER, sim_log):
            source = Path(raw_path)
            key = str(source.resolve()) if source.exists() else str(source)
            if key in seen_sources:
                continue
            seen_sources.add(key)
            fingerprint_sources.append(source)
        n_fingerprints = bootstrap_tested_fingerprints(fingerprint_sources, fingerprint_registry)
        if n_fingerprints:
            fingerprint_registry.save(fingerprint_path)
            print(
                f"Bootstrapped tested fingerprints: {n_fingerprints} new, "
                f"{len(fingerprint_registry)} total across {len(fingerprint_sources)} log(s)"
            )
        else:
            print("Starting with empty tested fingerprint registry")
    else:
        fingerprint_registry = TestedFingerprintRegistry.load(fingerprint_path)
        print(f"Loaded tested fingerprints: {len(fingerprint_registry)} entries")

    # ── 5. Build planner ─────────────────────────────────────────────────
    import random as _random
    config = GrammarConfig(
        budget=args.budget,
        n_samples=args.n_samples,
        exploration_fraction=args.exploration_fraction,
        min_coverage=args.min_coverage,
        alpha_binary_prob=args.alpha_binary_prob,
    )
    planner = GrammarPlanner(
        config,
        qtable=qtable,
        skeleton_counter=skeleton_counter,
        group_profiles=group_profiles,
        fingerprint_registry=fingerprint_registry,
    )
    probe_sampler = None
    if args.grammar_sampler == "probe_prior":
        prior_path = Path(args.probe_prior)
        if prior_path.exists():
            probe_sampler = ProbePriorGrammarSampler.from_json(
                prior_path,
                probability_column=args.prior_prob_column,
                preferred_field_prob=args.preferred_field_prob,
                sleeper_prob=args.sleeper_field_prob,
                operator_prior_prob=args.operator_prior_prob,
            )
            if probe_sampler.row_count:
                print(
                    "Probe-prior grammar sampler enabled: "
                    f"{probe_sampler.row_count} prior rows, "
                    f"field split={args.preferred_field_prob:.0%}/"
                    f"{args.sleeper_field_prob:.0%}/"
                    f"{max(0.0, 1.0 - args.preferred_field_prob - args.sleeper_field_prob):.0%}, "
                    f"operator_prior={args.operator_prior_prob:.0%}"
                )
            else:
                print(f"[warn] Probe prior file had no usable rows; falling back to raw grammar: {prior_path}")
                probe_sampler = None
        else:
            print(f"[warn] Probe prior file missing; falling back to raw grammar: {prior_path}")

    # ── 6. Continuous generate → submit → update loop ────────────────────
    rounds_label = f"{args.rounds}" if args.rounds > 0 else "∞"
    print(f"\nRunning {rounds_label} round(s). Press Ctrl+C to stop cleanly.\n")

    round_num = 0
    try:
        while args.rounds == 0 or round_num < args.rounds:
            round_num += 1
            seed = _random.randint(1, 10_000_000)
            print(f"\n{'='*60}")
            print(f"ROUND {round_num}  (budget={args.budget}, seed={seed})")
            print(f"{'='*60}")

            print(f"Generating {args.n_samples} candidates ...")
            if probe_sampler is not None:
                candidates_specs = planner.build_plan(
                    clusters,
                    seed=seed,
                    field_sampler=probe_sampler,
                    choice_sampler=probe_sampler.choice,
                )
            else:
                # uniform / raw: flat-uniform over every usable field, no prior.
                from alpha_grammar.grammar import _uniform_all_fields
                candidates_specs = planner.build_plan(
                    clusters, seed=seed, field_sampler=_uniform_all_fields)
            candidates_dicts = [c.as_record() for c in candidates_specs]
            if probe_sampler is not None:
                for record in candidates_dicts:
                    params = record.get("params") if isinstance(record.get("params"), dict) else {}
                    params = dict(params)
                    params.update(
                        {
                            "grammar_sampling": "probe_prior",
                            "probe_prior_path": str(Path(args.probe_prior)),
                            "probe_prior_probability_column": args.prior_prob_column,
                            "operator_prior_probability": args.operator_prior_prob,
                            "preferred_field_probability": args.preferred_field_prob,
                            "sleeper_field_probability": args.sleeper_field_prob,
                        }
                    )
                    record["params"] = params

            # alpha_metric gate: log every candidate's decision alongside the Q-table;
            # in reliable/full/auto modes also DROP the candidates it kills (fail-safe).
            _gate_mode = getattr(args, "gate_mode", "shadow")
            if getattr(args, "shadow_gate", False) or _gate_mode != "shadow":
                try:
                    from alpha_metric.shadow import get_shadow_gate
                    if _gate_mode == "auto":
                        from alpha_metric.cutover import load_state
                        _gate_mode = load_state().get("mode", "reliable")
                    # pass the live q-table for dual-shadow paired logging (Task 4)
                    _summary = get_shadow_gate(
                        qtable=qtable, beta=getattr(args, "gate_beta", None)
                    ).log_round(candidates_dicts, round_num, mode=_gate_mode)
                    if _gate_mode != "shadow" and isinstance(_summary, dict) and "survivors" in _summary:
                        before = len(candidates_dicts)
                        candidates_dicts = _summary["survivors"]
                        print(f"[gate-mode={_gate_mode}] live-filtered "
                              f"{before}->{len(candidates_dicts)} candidates")
                except Exception as _shadow_exc:  # never break the live run
                    print(f"[shadow-gate] skipped ({_shadow_exc})")

            # Submit and collect results
            sim_results, raw_records = run_batch_submission(
                session,
                candidates_dicts,
                settings,
                sim_log,
                run_log,
                batch_size=args.batch_size,
                max_candidates=args.max_candidates,
                sleep_seconds=args.sleep_seconds,
                max_submit_retries=args.max_submit_retries,
                credentials_path=args.credentials,
                portfolio_alpha_data=portfolio_alpha_data or None,
            )

            # Update Q-table and skeleton counter from this round's completed results
            raw_by_candidate = {
                raw_rec.get("candidate_id"): raw_rec
                for raw_rec in raw_records
                if raw_rec.get("candidate_id")
            }
            for result in sim_results:
                raw_rec = raw_by_candidate.get(result.candidate_id)
                if raw_rec is None:
                    continue
                if raw_rec.get("status") != "done":
                    continue
                if is_degenerate_sim_record(raw_rec):
                    continue
                field_ids = raw_rec.get("fields") or []
                cluster_ids = raw_rec.get("cluster_ids") or []
                if not field_ids:
                    continue
                primary_field = str(field_ids[0])
                primary_cluster = str(cluster_ids[0]) if cluster_ids else ""
                primary_op = extract_primary_op(result.expression, primary_field)
                reward = compute_q_reward(result)
                params = raw_rec.get("params") if isinstance(raw_rec.get("params"), dict) else {}
                qtable.update(
                    primary_cluster,
                    primary_field,
                    primary_op,
                    reward,
                    skeleton=raw_rec.get("template_id") or "",
                    group=str(raw_rec.get("group") or ""),
                    window=params.get("window"),
                )
                template_id = raw_rec.get("template_id") or ""
                if template_id:
                    skeleton_counter.increment(template_id)

            for raw_rec in raw_records:
                fingerprint_registry.add_record(raw_rec)

            # Persist Q-table + skeleton + tested-fingerprint state
            qtable.save(str(qtable_path))
            skeleton_counter.save(str(skeleton_path))
            fingerprint_registry.save(fingerprint_path)

            # Round summary
            n_pass = sum(1 for r in sim_results if r.passed)
            n_total = len(sim_results)
            sharpes = [r.is_sharpe for r in sim_results if r.is_sharpe > 0]
            avg_sharpe = sum(sharpes) / len(sharpes) if sharpes else 0.0
            n_pruned1 = len(qtable.pruned_level1())
            n_pruned2 = len(qtable.pruned_level2())
            print(f"\nRound {round_num} summary:")
            print(f"  Simulated: {n_total}  |  Passed: {n_pass}  |  Avg IS Sharpe: {avg_sharpe:.3f}")
            print(f"  Q-table L1: {len(qtable.level1)} entries, {n_pruned1} pruned  |  L2: {len(qtable.level2)} entries, {n_pruned2} pruned")
            print(f"  Tested fingerprints: {len(fingerprint_registry)}")

    except KeyboardInterrupt:
        print(f"\n\nStopped after {round_num} round(s).")

    # Merge session log into master
    print(f"\nSimulation log: {sim_log}")
    if Path(sim_log).exists():
        n_new, n_updated = merge_session_to_master(sim_log)
        print(f"Merged to master: {n_new} new, {n_updated} updated  ({MASTER})")
    print(f"Runner log:     {run_log}")


if __name__ == "__main__":
    main()
