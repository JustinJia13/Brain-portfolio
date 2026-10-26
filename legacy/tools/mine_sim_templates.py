"""Mine all sim logs for passing alphas and extract parameterized templates.

Run once (or repeatedly — idempotent via signature dedup) to bootstrap
shared/discovered_templates.json from all historical sim results.

Also warm-starts grammar UCB priors by replaying all-pipeline sim history
through FeedbackStore when --warm-priors is set.

Usage:
    python tools/mine_sim_templates.py
    python tools/mine_sim_templates.py --warm-priors alpha_grammar/artifacts/all_feedback.json
    python tools/mine_sim_templates.py --dry-run
"""

from __future__ import annotations

import argparse
import glob
import json
import re
import sys
from pathlib import Path
from collections import defaultdict

ROOT = Path(__file__).parent.parent
SIM_GLOB = str(ROOT / "*" / "artifacts" / "*sim*.jsonl")
OUT_PATH = ROOT / "shared" / "discovered_templates.json"

SHARPE_MIN = 1.25
FITNESS_MIN = 1.0
NEAR_PASS_SHARPE = 1.3  # candidate template threshold

KNOWN_GROUPS = {"sector", "subindustry", "industry", "market"}

# Operators whose last positional arg is a group identifier
GROUP_OPERATORS = {
    "group_rank", "group_neutralize", "group_zscore", "group_scale",
    "group_backfill", "group_mean", "group_std", "group_sum",
    "group_count", "group_median",
}

# Operators whose second positional arg is a time window
WINDOW_ARG_OPERATORS = {
    "ts_mean", "ts_std", "ts_zscore", "ts_rank", "ts_sum",
    "ts_decay_linear", "ts_decay_exp", "ts_corr", "ts_regression",
    "ts_arg_max", "ts_arg_min", "ts_max", "ts_min",
    "ts_delta", "ts_skewness", "ts_kurtosis", "ts_moment",
    "last_diff_value", "kth_element",
}


# ---------------------------------------------------------------------------
# Expression generalization
# ---------------------------------------------------------------------------

def _replace_groups(expr: str) -> tuple[str, list[str]]:
    """Replace group identifiers (sector/subindustry/industry/market) with {group}."""
    groups_found = []
    def sub(m: re.Match) -> str:
        g = m.group()
        if g in KNOWN_GROUPS:
            groups_found.append(g)
            return "{group}"
        return g
    # Match identifiers that are group names (as standalone words, not substrings of field names)
    result = re.sub(r'\b(sector|subindustry|industry|market)\b', sub, expr)
    return result, groups_found


def _replace_windows(expr: str) -> tuple[str, list[int]]:
    """Replace likely window integers with {window}.

    Targets positional integer arguments (not named params like k=N, std=N).
    Integers in range [5, 500] that appear after a comma (positional args).
    """
    windows_found: list[int] = []

    def sub(m: re.Match) -> str:
        # m.group(1) = preceding separator, m.group(2) = the integer
        start = m.start(2)
        prefix = expr[:start]
        stripped = prefix.rstrip()
        if stripped and stripped[-1] == '=':
            return m.group()
        val = int(m.group(2))
        if 5 <= val <= 500:
            windows_found.append(val)
            return m.group(1) + "{window}"
        return m.group()

    # Match: separator (comma/open-paren + optional spaces) + integer + followed by ,/)/end
    result = re.sub(r'([,(]\s*)(\d+)(?=\s*[,)])', sub, expr)
    return result, windows_found


def generalise_expression(expr: str, fields: list[str]) -> tuple[str, list[int], list[str]]:
    """Return (pattern, observed_windows, observed_groups).

    Replaces:
      - field names → {field} (longest match first to avoid partial replacement)
      - group identifiers → {group}
      - window integers → {window}
    """
    pattern = expr

    # 1. Replace fields (longest first to avoid partial substring matches)
    sorted_fields = sorted(fields, key=len, reverse=True)
    for i, fid in enumerate(sorted_fields):
        placeholder = "{field}" if i == 0 else f"{{field{i + 1}}}"
        pattern = pattern.replace(fid, placeholder)

    # 2. Replace groups before windows so group names don't confuse window extraction
    pattern, groups_found = _replace_groups(pattern)

    # 3. Replace window integers
    pattern, windows_found = _replace_windows(pattern)

    return pattern, windows_found, groups_found


def compute_signature(pattern: str, dataset_id: str = "") -> str:
    """Normalised signature for dedup: dataset_id + stripped pattern."""
    normalised = re.sub(r"\s+", "", pattern.lower())
    return f"{dataset_id}:{normalised}" if dataset_id else normalised


# ---------------------------------------------------------------------------
# Sim log loading and filtering
# ---------------------------------------------------------------------------

def load_all_sims() -> list[dict]:
    records: list[dict] = []
    paths = sorted(glob.glob(SIM_GLOB))
    for path in paths:
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return records


def is_passing(rec: dict) -> bool:
    """True if alpha passes all submission-relevant Brain checks."""
    if rec.get("status") != "done":
        return False
    sharpe = rec.get("is_sharpe") or 0.0
    fitness = rec.get("is_fitness") or 0.0
    if sharpe < SHARPE_MIN or fitness < FITNESS_MIN:
        return False
    if not rec.get("is_concentrated_weight_ok", True):
        return False
    if not rec.get("is_sub_universe_sharpe_ok", True):
        return False
    return True


def is_near_pass(rec: dict) -> bool:
    """True if near-pass candidate: Sharpe >= 1.3, exactly 1 non-CW failing check."""
    if rec.get("status") not in ("done",):
        return False
    sharpe = rec.get("is_sharpe") or 0.0
    fitness = rec.get("is_fitness") or 0.0
    if sharpe < NEAR_PASS_SHARPE or fitness < 0.8:
        return False
    # Not CONCENTRATED_WEIGHT (fake signal — wrong structure, not a near-miss)
    if not rec.get("is_concentrated_weight_ok", True):
        return False
    # Exactly 1 non-SELF_CORRELATION failing check
    checks_raw = (rec.get("raw_alpha_detail") or {}).get("is", {}).get("checks") or []
    failing = [
        c["name"] for c in checks_raw
        if c.get("result") == "FAIL" and c.get("name") != "SELF_CORRELATION"
    ]
    return len(failing) == 1


def _dataset_category(dataset_id: str, rec_category: str | None = None) -> str:
    if rec_category:
        return rec_category
    prefixes = [
        ("model", "model"), ("fundamental", "fundamental"),
        ("analyst", "analyst"), ("news", "news"),
        ("socialmedia", "social"), ("pv", "price_volume"),
        ("options", "options"), ("trade", "trade"),
    ]
    for prefix, cat in prefixes:
        if dataset_id.startswith(prefix):
            return cat
    return "other"


def _make_window_range(windows: list[int]) -> list[int]:
    if not windows:
        return []
    result: set[int] = set()
    for w in windows:
        result.add(max(5, w // 2))
        result.add(w)
        result.add(min(500, w * 2))
    return sorted(result)


# ---------------------------------------------------------------------------
# Template extraction
# ---------------------------------------------------------------------------

def extract_template_entry(
    signature: str,
    pattern: str,
    dataset_id: str,
    category: str,
    fields: list[str],
    windows: list[int],
    best_sharpe: float,
    best_alpha_id: str,
    best_expr: str,
    source: str,
    status: str = "active",
) -> dict:
    priority = round(2.0 + 0.5 * min((best_sharpe - SHARPE_MIN) / 1.0, 2.0), 3)
    template_id = f"disc_{source}_{signature[:20]}"
    return {
        "template_id": template_id,
        "dataset_id": dataset_id,
        "category": category,
        "description": f"Discovered from {source} sim (best sharpe={best_sharpe:.2f})",
        "pattern": pattern,
        "fields": sorted(set(fields)),
        "windows": _make_window_range(windows),
        "groups": ["sector", "subindustry", "market"],
        "priority": priority,
        "status": status,
        "source": source,
        "signature": signature,
        "seed_alpha_id": best_alpha_id,
        "seed_sharpe": best_sharpe,
        "seed_expression": best_expr,
    }


def mine_templates(records: list[dict]) -> list[dict]:
    """Extract parameterized templates from sim records.

    Groups records by (signature, dataset_id) to merge field lists.
    Deduplicates by signature. Returns list of template entry dicts.
    """
    # key: (signature, dataset_id) → aggregated data
    groups: dict[tuple[str, str], dict] = defaultdict(lambda: {
        "fields": [], "windows": [], "best_sharpe": 0.0,
        "best_alpha_id": "", "best_expr": "", "source": "",
        "pattern": "", "category": "", "status": "active",
    })

    for rec in records:
        expr = rec.get("expression", "")
        fields = list(rec.get("fields") or [])
        dataset_id = rec.get("dataset_id") or ""
        category = _dataset_category(dataset_id, rec.get("category"))

        if not expr or not fields:
            continue
        # Skip multi-field composites (field_weights patterns) for now
        if len(fields) > 2:
            continue

        passing = is_passing(rec)
        near = is_near_pass(rec) if not passing else False
        if not passing and not near:
            continue

        pattern, windows_found, _ = generalise_expression(expr, fields)

        # Skip bare field patterns (no structure added)
        if pattern.strip() in ("{field}", "rank({field})", "-{field}"):
            continue

        sig = compute_signature(pattern, dataset_id)
        key = (sig, dataset_id)

        g = groups[key]
        g["pattern"] = pattern
        g["category"] = category
        g["source"] = rec.get("search_lane") or "unknown"
        g["fields"].extend(fields)
        g["windows"].extend(windows_found)

        sharpe = float(rec.get("is_sharpe") or 0.0)
        if sharpe > g["best_sharpe"]:
            g["best_sharpe"] = sharpe
            g["best_alpha_id"] = rec.get("alpha_id") or ""
            g["best_expr"] = expr

        # Prefer "active" over "candidate": if any record passes fully, mark active
        if passing and g["status"] != "active":
            g["status"] = "active"
        elif not passing and g["status"] != "active":
            g["status"] = "candidate"

    templates = []
    for (sig, dataset_id), g in groups.items():
        entry = extract_template_entry(
            signature=sig,
            pattern=g["pattern"],
            dataset_id=dataset_id,
            category=g["category"],
            fields=g["fields"],
            windows=g["windows"],
            best_sharpe=g["best_sharpe"],
            best_alpha_id=g["best_alpha_id"],
            best_expr=g["best_expr"],
            source=g["source"],
            status=g["status"],
        )
        templates.append(entry)

    return templates


# ---------------------------------------------------------------------------
# Discovered templates JSON merge (idempotent)
# ---------------------------------------------------------------------------

def load_existing(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path) as f:
        return json.load(f)


def merge_templates(existing: list[dict], new_templates: list[dict]) -> tuple[list[dict], int, int]:
    """Merge new_templates into existing, deduplicating by signature.

    Returns (merged_list, n_added, n_updated).
    """
    by_sig: dict[str, dict] = {t["signature"]: t for t in existing}
    added = updated = 0

    for t in new_templates:
        sig = t["signature"]
        if sig not in by_sig:
            by_sig[sig] = t
            added += 1
        else:
            # Update fields list and priority if new data improves it
            existing_entry = by_sig[sig]
            old_fields = set(existing_entry.get("fields") or [])
            new_fields = set(t.get("fields") or [])
            merged_fields = sorted(old_fields | new_fields)
            changed = merged_fields != sorted(old_fields)

            if t["seed_sharpe"] > existing_entry.get("seed_sharpe", 0):
                existing_entry.update({
                    "seed_sharpe": t["seed_sharpe"],
                    "seed_alpha_id": t["seed_alpha_id"],
                    "seed_expression": t["seed_expression"],
                    "priority": t["priority"],
                    "description": t["description"],
                })
                changed = True

            if changed:
                existing_entry["fields"] = merged_fields
                # Merge window ranges
                existing_windows = set(existing_entry.get("windows") or [])
                new_windows = set(t.get("windows") or [])
                existing_entry["windows"] = sorted(existing_windows | new_windows)
                updated += 1

    merged = sorted(by_sig.values(), key=lambda x: -x.get("priority", 0))
    return merged, added, updated


# ---------------------------------------------------------------------------
# Optional: warm-start grammar UCB priors
# ---------------------------------------------------------------------------

def warm_priors(sim_records: list[dict], feedback_path: Path) -> None:
    """Replay all sim results (any pipeline) through FeedbackStore and save.

    Skips records already in feedback.json (by comparing total_pulls to avoid
    double-counting when re-run).
    """
    sys.path.insert(0, str(ROOT))
    from alpha_grammar.feedback import FeedbackStore, SimResult

    store = FeedbackStore()
    if feedback_path.exists():
        store.load(str(feedback_path))
        print(f"  Loaded existing priors: {store.n_arms} arms, {store.total_pulls} pulls")
        existing_pulls = store.total_pulls
    else:
        existing_pulls = 0

    # Register clusters from all sim records first
    # (FeedbackStore.register_cluster needs coverage; we use alpha_count as proxy)
    cluster_coverage: dict[str, float] = {}
    for rec in sim_records:
        for cid in (rec.get("cluster_ids") or []):
            if cid not in cluster_coverage:
                cluster_coverage[cid] = 0.5  # neutral prior if no coverage data

    for cid, cov in cluster_coverage.items():
        store.register_cluster(cid, cov)

    # Replay all done records (any pipeline)
    replayed = 0
    for rec in sim_records:
        if rec.get("status") != "done":
            continue
        cluster_ids = tuple(str(c) for c in (rec.get("cluster_ids") or []))
        if not cluster_ids:
            continue

        sharpe = float(rec.get("is_sharpe") or 0.0)
        fitness = float(rec.get("is_fitness") or 0.0)
        raw_to = rec.get("is_turnover")
        is_turnover = float(raw_to) if raw_to is not None else None
        cw_ok = bool(rec.get("is_concentrated_weight_ok", True))
        sus_ok = bool(rec.get("is_sub_universe_sharpe_ok", True))
        grade = rec.get("grade") or ""
        passed = grade.upper() in ("PASS", "GOOD", "EXCELLENT", "SPECTACULAR") or sharpe >= 1.25

        result = SimResult(
            candidate_id=rec.get("candidate_id") or "",
            expression=rec.get("expression") or "",
            cluster_ids=cluster_ids,
            is_sharpe=sharpe,
            is_fitness=fitness,
            passed=passed,
            is_turnover=is_turnover,
            concentrated_weight_ok=cw_ok,
            sub_universe_sharpe_ok=sus_ok,
        )
        store.update(result)
        replayed += 1

    store.end_of_batch()
    feedback_path.parent.mkdir(parents=True, exist_ok=True)
    store.save(str(feedback_path))
    new_pulls = store.total_pulls - existing_pulls
    print(f"  Replayed {replayed} records → {new_pulls} new pulls")
    print(f"  Saved warm priors: {feedback_path} ({store.n_arms} arms, {store.total_pulls} pulls)")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true",
                        help="Print what would be written but don't write files")
    parser.add_argument("--warm-priors", metavar="FEEDBACK_JSON",
                        help="Warm-start FeedbackStore from all sim logs and save to this path "
                             "(e.g. alpha_grammar/artifacts/all_feedback.json)")
    parser.add_argument("--out", default=str(OUT_PATH),
                        help=f"Output path for discovered_templates.json (default: {OUT_PATH})")
    args = parser.parse_args()

    out_path = Path(args.out)

    print("Loading sim records from all pipelines...")
    records = load_all_sims()
    print(f"  Loaded {len(records)} total records")

    passing_count = sum(1 for r in records if is_passing(r))
    near_count = sum(1 for r in records if not is_passing(r) and is_near_pass(r))
    print(f"  Passing: {passing_count}  |  Near-pass (candidate): {near_count}")

    print("\nExtracting templates...")
    new_templates = mine_templates(records)
    print(f"  Extracted {len(new_templates)} unique structural patterns")
    active = sum(1 for t in new_templates if t["status"] == "active")
    candidate = sum(1 for t in new_templates if t["status"] == "candidate")
    print(f"  Active: {active}  |  Candidate (near-pass): {candidate}")

    print(f"\nMerging with existing {out_path}...")
    existing = load_existing(out_path)
    print(f"  Existing entries: {len(existing)}")
    merged, added, updated = merge_templates(existing, new_templates)
    print(f"  Added: {added}  |  Updated: {updated}  |  Total: {len(merged)}")

    if args.dry_run:
        print("\n[DRY RUN] Would write:")
        for t in merged[:10]:
            print(f"  [{t['status']}] {t['template_id']} | sharpe={t['seed_sharpe']:.2f} "
                  f"| fields={t['fields'][:2]} | pattern={t['pattern'][:60]}")
        if len(merged) > 10:
            print(f"  ... and {len(merged) - 10} more")
    else:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(merged, f, indent=2)
        print(f"\nWritten: {out_path}")

    if args.warm_priors:
        feedback_path = Path(args.warm_priors)
        print(f"\nWarm-starting UCB priors → {feedback_path} ...")
        warm_priors(records, feedback_path)

    print("\nDone.")


if __name__ == "__main__":
    main()
