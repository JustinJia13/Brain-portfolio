"""Verify PnL-based self-correlation computation against Brain's known values.

Strategy
--------
Brain only populates is.selfCorrelation for competition alphas (stage=OS).
For those alphas we can:
  1. Read Brain's reported selfCorrelation
  2. Fetch all competition PnL series
  3. For each competition alpha, compute our max |corr| against the *other*
     alphas in the same portfolio (excluding itself)
  4. Compare computed vs Brain-reported

Additionally, unit tests cover:
  - Empty / short series edge cases
  - Constant series (std=0 guard)
  - Perfect correlation / anti-correlation
  - Partial date overlap

Usage:
    python -m tools.test_self_corr_verification
    python -m tools.test_self_corr_verification --competition-id IQC2026S1 --n 10
"""

from __future__ import annotations

import argparse
import math
import sys
from typing import Any


# ---------------------------------------------------------------------------
# Unit tests (no network)
# ---------------------------------------------------------------------------

def _make_series(n: int, fn) -> dict[str, float]:
    """Build a cumulative PnL series by integrating fn (as daily returns)."""
    cumulative = 0.0
    result = {}
    for i in range(1, n + 1):
        cumulative += fn(i)
        result[f"2020-{i:05d}"] = cumulative
    return result


def _make_levels(n: int, fn) -> dict[str, float]:
    """Build a raw (levels) series — fn gives the value directly."""
    return {f"2020-{i:05d}": fn(i) for i in range(1, n + 1)}


def run_unit_tests() -> int:
    """Run offline unit tests. Returns number of failures."""
    import json
    from alpha_grammar.feedback import SimResult
    from alpha_grammar.qtable import compute_q_reward
    from alpha_grammar.run_pipeline import compute_max_self_corr, compute_max_self_corr_info
    from wq_alpha.workshop.history import WorkshopHistory
    from wq_alpha.workshop.core import (
        generate_candidates,
        record_workshop_key,
        route_record,
    )
    from wq_alpha.agents.repair import _build_round_feedback, _portfolio_feedback_line
    from wq_alpha.portfolio.admission import (
        AdmissionConfig,
        PROMOTE,
        REJECT,
        WATCH,
        score_record_admission,
    )
    from tools.backfill_self_correlation import _needs_backfill
    from tools.batch_workshop import select_candidates, _write_audit, _read_from_audit, _promote_to_templates

    failures = 0

    def check(name: str, got, expected, *, tol: float = 0.01):
        nonlocal failures
        if expected is None:
            ok = got is None
        else:
            ok = got is not None and abs(got - expected) <= tol
        status = "PASS" if ok else "FAIL"
        if not ok:
            failures += 1
        print(f"  [{status}] {name}: got={got}  expected={expected}")

    def check_true(name: str, condition: bool, detail: str = ""):
        nonlocal failures
        ok = bool(condition)
        if not ok:
            failures += 1
        suffix = f": {detail}" if detail else ""
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}{suffix}")

    def make_record(
        expr: str,
        *,
        fitness: float | None,
        sharpe: float = 1.0,
        checks: list[dict] | None = None,
        max_concentration: float | None = None,
        max_self_correlation: float | None = 0.20,
        max_self_correlation_alpha_sharpe: float | None = None,
        field: str = "test_field",
        status: str = "done",
        delay: int = 1,
    ) -> dict:
        if checks is None:
            checks = [
                {"name": "CONCENTRATED_WEIGHT", "result": "PASS"},
                {"name": "LOW_SUB_UNIVERSE_SHARPE", "result": "PASS"},
            ]
        settings = {
            "delay": delay,
            "neutralization": "SUBINDUSTRY",
            "universe": "TOP3000",
        }
        return {
            "status": status,
            "expression": expr,
            "is_fitness": fitness,
            "is_sharpe": sharpe,
            "checks": checks,
            "grade": "PASS",
            "max_concentration": max_concentration,
            "max_self_correlation": max_self_correlation,
            "max_self_correlation_alpha_sharpe": max_self_correlation_alpha_sharpe,
            "fields": [field],
            "dataset_id": "TEST",
            "settings": settings,
        }

    print("\n--- Unit tests ---")
    print("  (Inputs are cumulative PnL; function first-differences internally)\n")

    # 1. Empty candidate
    check("empty candidate", compute_max_self_corr({}, {"a": _make_series(50, math.sin)}), None)

    # 2. Empty portfolio
    check("empty portfolio", compute_max_self_corr(_make_series(50, math.sin), {}), None)

    # 3. Too few shared dates — after diff, n-1 shared needed; feed 10 days → 9 diffs < 20
    short_cum = {f"2020-{i:05d}": float(i) for i in range(1, 11)}
    check("< 20 shared dates after diff", compute_max_self_corr(short_cum, {"a": short_cum}), None)

    # 4. Identical daily returns → corr=1.0
    # Both series integrate the same sin(i) daily increments
    s = _make_series(100, math.sin)
    check("identical daily returns (corr=1.0)", compute_max_self_corr(s, {"a": s}), 1.0, tol=1e-6)

    # 5. Anti-correlated daily returns → signed max = -1.0 (diversifying, passes gate)
    s_neg = _make_series(100, lambda i: -math.sin(i))
    check("anti-correlated daily returns (signed max=-1.0)", compute_max_self_corr(s, {"a": s_neg}), -1.0, tol=1e-6)

    # 6. Constant cumulative series → zero daily returns → std=0 → skipped → None
    const_cum = {f"2020-{i:05d}": 42.0 for i in range(1, 100)}
    check("constant PnL → zero daily returns skipped", compute_max_self_corr(const_cum, {"a": s}), None)

    # 7. Orthogonal daily returns: sin vs cos increments → near-zero corr
    sin_s = _make_series(200, math.sin)
    cos_s = _make_series(200, math.cos)
    result7 = compute_max_self_corr(sin_s, {"a": cos_s})
    ok7 = result7 is not None and result7 < 0.1
    if not ok7:
        failures += 1
    print(f"  [{'PASS' if ok7 else 'FAIL'}] orthogonal daily returns (sin vs cos): "
          f"got={result7:.4f}  expected<0.1")

    # 8. Partial date overlap — cumulative series on different date ranges
    # After differencing, shared diff-dates = intersection of (2..50) and (27..75) = 27..50 (24 days)
    cand_cum = {}
    v = 0.0
    for i in range(1, 51):
        v += math.sin(i)
        cand_cum[f"2020-{i:05d}"] = v
    port_cum = {}
    v = 0.0
    for i in range(26, 76):
        v += math.sin(i)
        port_cum[f"2020-{i:05d}"] = v
    result8 = compute_max_self_corr(cand_cum, {"a": port_cum})
    check("partial overlap (≥20 shared diff-dates) corr≈1.0", result8, 1.0, tol=0.01)

    # 9. Multiple portfolio alphas — returns SIGNED max across all
    low_corr = _make_series(100, lambda i: math.cos(i * 0.5))
    high_corr = _make_series(100, math.sin)   # identical daily returns to candidate
    result9 = compute_max_self_corr(s, {"low": low_corr, "high": high_corr})
    check("signed max of two alphas (best≈1.0)", result9, 1.0, tol=0.01)

    info9 = compute_max_self_corr_info(
        s,
        {
            "low": {"pnl": low_corr, "is_sharpe": 0.8, "is_fitness": 0.7},
            "high": {"pnl": high_corr, "is_sharpe": 1.5, "is_fitness": 1.2},
        },
    )
    check("corr helper returns matched corr", info9["max_self_correlation"], 1.0, tol=0.01)
    ok9info = (
        info9["max_self_correlation_alpha_id"] == "high"
        and info9["max_self_correlation_alpha_sharpe"] == 1.5
        and info9["max_self_correlation_alpha_fitness"] == 1.2
    )
    if not ok9info:
        failures += 1
    print(
        f"  [{'PASS' if ok9info else 'FAIL'}] corr helper returns matched benchmark metadata: "
        f"id={info9['max_self_correlation_alpha_id']} sharpe={info9['max_self_correlation_alpha_sharpe']} "
        f"fitness={info9['max_self_correlation_alpha_fitness']}"
    )

    # 9b. Anti-correlated wins over low positive — signed max should be the positive one
    anti = _make_series(100, lambda i: -math.sin(i))  # corr = -1.0
    low_pos = _make_series(100, lambda i: math.cos(i * 0.5))  # small positive corr
    r9b = compute_max_self_corr(s, {"anti": anti, "low": low_pos})
    ok9b = r9b is not None and r9b > -1.0  # signed max > -1.0 (low_pos wins)
    if not ok9b:
        failures += 1
    print(f"  [{'PASS' if ok9b else 'FAIL'}] signed max picks highest (not abs): got={r9b:.4f}  expected>-1.0")

    # 10. Independent i.i.d. cumulative PnL — daily returns are iid → low corr
    import random
    rng = random.Random(42)
    cum1, cum2 = 0.0, 0.0
    noise1: dict[str, float] = {}
    noise2: dict[str, float] = {}
    for i in range(1, 501):
        k = f"2020-{i:05d}"
        cum1 += rng.gauss(0, 1)
        cum2 += rng.gauss(0, 1)
        noise1[k] = cum1
        noise2[k] = cum2
    result10 = compute_max_self_corr(noise1, {"a": noise2})
    ok10 = result10 is not None and result10 < 0.15
    if not ok10:
        failures += 1
    print(f"  [{'PASS' if ok10 else 'FAIL'}] independent i.i.d. daily returns: "
          f"got={result10:.4f}  expected<0.15")

    # 11. High self-corr exception keeps base reward when Sharpe beats benchmark by 10%
    reward_keep = compute_q_reward(
        SimResult(
            candidate_id="keep",
            expression="rank(x)",
            cluster_ids=(),
            is_sharpe=1.65,
            is_fitness=1.2,
            passed=False,
            max_self_correlation=0.75,
            max_self_correlation_alpha_sharpe=1.5,
        )
    )
    check("reward keeps base when high corr alpha beats benchmark by 10%", reward_keep, 1.2, tol=1e-6)

    # 12. High self-corr still zeros when Sharpe does not clear benchmark threshold
    reward_zero = compute_q_reward(
        SimResult(
            candidate_id="zero",
            expression="rank(x)",
            cluster_ids=(),
            is_sharpe=1.64,
            is_fitness=1.2,
            passed=False,
            max_self_correlation=0.75,
            max_self_correlation_alpha_sharpe=1.5,
        )
    )
    check("reward zeros when high corr alpha misses 1.10x Sharpe threshold", reward_zero, 0.0, tol=1e-6)

    # 13. Sub-threshold self-corr remains unchanged
    reward_low_corr = compute_q_reward(
        SimResult(
            candidate_id="lowcorr",
            expression="rank(x)",
            cluster_ids=(),
            is_sharpe=0.9,
            is_fitness=1.3,
            passed=False,
            max_self_correlation=0.69,
            max_self_correlation_alpha_sharpe=5.0,
        )
    )
    check("reward unchanged when corr < 0.70", reward_low_corr, 1.3, tol=1e-6)

    # 14. Concentration below 40% stays workshop-fixable (soft 0.75x penalty)
    reward_soft_conc = compute_q_reward(
        SimResult(
            candidate_id="softconc",
            expression="rank(x)",
            cluster_ids=(),
            is_sharpe=1.1,
            is_fitness=1.4,
            passed=False,
            concentrated_weight_ok=False,
            max_concentration=0.39,
        )
    )
    check("reward soft-penalizes concentration below 40%", reward_soft_conc, 1.05, tol=1e-6)

    # 15. Concentration at 40% or above still hard-zeros reward
    reward_hard_conc = compute_q_reward(
        SimResult(
            candidate_id="hardconc",
            expression="rank(x)",
            cluster_ids=(),
            is_sharpe=1.1,
            is_fitness=1.4,
            passed=False,
            concentrated_weight_ok=False,
            max_concentration=0.40,
        )
    )
    check("reward hard-zeros concentration at 40%", reward_hard_conc, 0.0, tol=1e-6)

    # 16. Missing benchmark Sharpe fails closed for high-corr alphas
    reward_fail_closed = compute_q_reward(
        SimResult(
            candidate_id="failclosed",
            expression="rank(x)",
            cluster_ids=(),
            is_sharpe=2.0,
            is_fitness=1.1,
            passed=False,
            max_self_correlation=0.80,
            max_self_correlation_alpha_sharpe=None,
        )
    )
    check("reward fails closed when benchmark Sharpe missing", reward_fail_closed, 0.0, tol=1e-6)

    # 17. Backfill selects rows that already have corr but are missing benchmark metadata
    backfill_candidate = {
        "status": "done",
        "alpha_id": "abc",
        "search_lane": "grammar",
        "is_sharpe": 1.2,
        "is_fitness": 1.1,
        "is_turnover": 0.05,
        "max_self_correlation": 0.75,
        "max_self_correlation_alpha_id": None,
        "max_self_correlation_alpha_sharpe": None,
        "max_self_correlation_alpha_fitness": None,
    }
    ok15 = _needs_backfill(backfill_candidate, None, False, 0)
    if not ok15:
        failures += 1
    print(f"  [{'PASS' if ok15 else 'FAIL'}] backfill re-selects corr rows missing benchmark metadata")

    # 18. Degenerate rows remain excluded from backfill
    degenerate_candidate = {
        "status": "done",
        "alpha_id": "deg",
        "search_lane": "grammar",
        "is_sharpe": 0.0,
        "is_fitness": None,
        "is_turnover": 0.0,
        "max_self_correlation": None,
    }
    ok16 = not _needs_backfill(degenerate_candidate, None, False, 0)
    if not ok16:
        failures += 1
    print(f"  [{'PASS' if ok16 else 'FAIL'}] degenerate rows are skipped by backfill")

    # 19. Negate routing is fitness-driven, but excludes extreme concentration
    negate_rec = make_record(
        "negate_expr",
        fitness=-1.2,
        sharpe=2.5,
        max_self_correlation=None,
        field="neg_field",
    )
    check_true("route negate when fitness <= -1.2", route_record(negate_rec) == "negate")

    negate_extreme_conc = make_record(
        "negate_extreme_conc",
        fitness=-2.0,
        sharpe=1.0,
        checks=[{"name": "CONCENTRATED_WEIGHT", "result": "FAIL", "value": 0.50}],
        max_concentration=None,
        max_self_correlation=None,
        field="neg_extreme_conc_field",
    )
    check_true("reject negate when concentration is at least 40%", route_record(negate_extreme_conc) is None)

    # 20. Repair allows LOW_SUB_UNIVERSE_SHARPE-only failures when corr is eligible
    repair_subuni = make_record(
        "repair_subuni",
        fitness=1.6,
        sharpe=1.2,
        checks=[{"name": "LOW_SUB_UNIVERSE_SHARPE", "result": "FAIL"}],
        field="repair_subuni_field",
    )
    check_true("route repair for LOW_SUB_UNIVERSE_SHARPE with fitness > 1.5", route_record(repair_subuni) == "repair")

    repair_positive_delta = make_record(
        "repair_positive_delta",
        fitness=1.2,
        sharpe=1.37,
        checks=[{"name": "LOW_SUB_UNIVERSE_SHARPE", "result": "FAIL"}],
        field="repair_positive_delta_field",
    )
    repair_positive_delta["admission_delta_pool_sharpe"] = 0.02
    check_true("route repair for repairable failure with positive delta", route_record(repair_positive_delta) == "repair")

    repair_negative_delta = make_record(
        "repair_negative_delta",
        fitness=1.6,
        sharpe=1.7,
        checks=[{"name": "LOW_SUB_UNIVERSE_SHARPE", "result": "FAIL"}],
        field="repair_negative_delta_field",
    )
    repair_negative_delta["admission_delta_pool_sharpe"] = -0.01
    check_true("reject repair for non-positive delta", route_record(repair_negative_delta) is None)

    repair_official_positive = make_record(
        "repair_official_positive",
        fitness=1.6,
        sharpe=1.7,
        checks=[{"name": "LOW_SUB_UNIVERSE_SHARPE", "result": "FAIL"}],
        field="repair_official_positive_field",
    )
    repair_official_positive["admission_delta_pool_sharpe"] = -0.01
    repair_official_positive["admission_brain_score_delta"] = 227.0
    check_true(
        "route repair when Brain score delta is positive despite proxy delta",
        route_record(repair_official_positive) == "repair",
    )

    # 21. Repair allows concentration failures only when max_concentration < 0.40
    repair_conc = make_record(
        "repair_conc",
        fitness=1.7,
        sharpe=1.1,
        checks=[{"name": "CONCENTRATED_WEIGHT", "result": "FAIL", "value": 0.39}],
        max_concentration=0.39,
        field="repair_conc_field",
    )
    check_true("route repair for concentration below 40%", route_record(repair_conc) == "repair")

    # 22. Repair rejects concentration at or above 40%
    repair_conc_bad = make_record(
        "repair_conc_bad",
        fitness=1.7,
        sharpe=1.1,
        checks=[{"name": "CONCENTRATED_WEIGHT", "result": "FAIL", "value": 0.40}],
        max_concentration=0.40,
        field="repair_conc_bad_field",
    )
    check_true("reject repair for concentration at 40%", route_record(repair_conc_bad) is None)

    # 23. Repair rejects unrepairable failing checks
    repair_unfixable = make_record(
        "repair_unfixable",
        fitness=1.8,
        sharpe=1.3,
        checks=[{"name": "LOW_SHARPE", "result": "FAIL"}],
        field="repair_unfixable_field",
    )
    check_true("reject repair for unrepairable checks", route_record(repair_unfixable) is None)

    # 24. Repair/enhance allow missing correlation for workshop processing
    repair_missing_corr = make_record(
        "repair_missing_corr",
        fitness=1.6,
        sharpe=1.2,
        checks=[{"name": "LOW_SUB_UNIVERSE_SHARPE", "result": "FAIL"}],
        max_self_correlation=None,
        field="repair_missing_corr_field",
    )
    check_true("route repair when correlation missing", route_record(repair_missing_corr) == "repair")
    check_true(
        "bypass corr routes repair when correlation missing",
        route_record(repair_missing_corr, bypass_self_corr=True) == "repair",
    )

    # 25. Enhance requires checks pass, fitness >= 1.0, positive/unknown delta, and correlation eligibility
    enhance_ok = make_record(
        "enhance_ok",
        fitness=1.8,
        sharpe=1.4,
        field="enhance_ok_field",
    )
    check_true("route enhance for passing alpha with fitness > 1.5", route_record(enhance_ok) == "enhance")

    enhance_secondary_unknown = make_record(
        "enhance_secondary_unknown",
        fitness=1.03,
        sharpe=1.29,
        field="enhance_secondary_unknown_field",
    )
    check_true("route provisional enhance before delta scoring", route_record(enhance_secondary_unknown) == "enhance")

    enhance_secondary_positive = make_record(
        "enhance_secondary_positive",
        fitness=1.03,
        sharpe=1.29,
        field="enhance_secondary_positive_field",
    )
    enhance_secondary_positive["admission_delta_pool_sharpe"] = 0.02
    check_true("route enhance for submittable positive-delta alpha", route_record(enhance_secondary_positive) == "enhance")

    enhance_secondary_negative = make_record(
        "enhance_secondary_negative",
        fitness=1.34,
        sharpe=1.70,
        field="enhance_secondary_negative_field",
    )
    enhance_secondary_negative["admission_delta_pool_sharpe"] = -0.01
    check_true("reject enhance for non-positive delta", route_record(enhance_secondary_negative) is None)

    enhance_official_positive = make_record(
        "enhance_official_positive",
        fitness=1.34,
        sharpe=1.70,
        field="enhance_official_positive_field",
    )
    enhance_official_positive["admission_delta_pool_sharpe"] = -0.01
    enhance_official_positive["admission_brain_score_delta"] = 197.0
    check_true(
        "route enhance when Brain score delta is positive despite proxy delta",
        route_record(enhance_official_positive) == "enhance",
    )

    # 26. Enhance rejects high-corr alphas that do not beat benchmark Sharpe by 10%
    enhance_high_corr_fail = make_record(
        "enhance_high_corr_fail",
        fitness=1.8,
        sharpe=1.64,
        max_self_correlation=0.75,
        max_self_correlation_alpha_sharpe=1.5,
        field="enhance_high_corr_fail_field",
    )
    check_true("reject enhance when high corr misses Sharpe exception", route_record(enhance_high_corr_fail) is None)

    # 27. Enhance accepts high-corr alphas that beat the benchmark by 10%
    enhance_high_corr_pass = make_record(
        "enhance_high_corr_pass",
        fitness=1.8,
        sharpe=1.65,
        max_self_correlation=0.75,
        max_self_correlation_alpha_sharpe=1.5,
        field="enhance_high_corr_pass_field",
    )
    check_true("accept enhance when high corr clears Sharpe exception", route_record(enhance_high_corr_pass) == "enhance")
    enhance_missing_corr = make_record(
        "enhance_missing_corr",
        fitness=1.8,
        sharpe=1.4,
        max_self_correlation=None,
        field="enhance_missing_corr_field",
    )
    check_true(
        "bypass corr routes enhance when correlation missing",
        route_record(enhance_missing_corr, bypass_self_corr=True) == "enhance",
    )

    # 28. Negate intentionally bypasses the correlation gate
    negate_high_corr = make_record(
        "negate_high_corr",
        fitness=-2.0,
        sharpe=0.1,
        max_self_correlation=0.90,
        max_self_correlation_alpha_sharpe=None,
        field="negate_high_corr_field",
    )
    check_true("negate bypasses self-correlation gate", route_record(negate_high_corr) == "negate")

    # 29. Batch selection uses the same route decisions as the shared router
    batch_records = [
        negate_rec,
        repair_subuni,
        enhance_ok,
        make_record("skip_expr", fitness=0.2, sharpe=0.5, field="skip_field"),
    ]
    selected = select_candidates(batch_records, set(), set(), verbose=False)
    selected_routes = {r["expression"]: r.get("_route") for r in selected}
    expected_routes = {
        r["expression"]: route_record(r)
        for r in batch_records
        if route_record(r) is not None
    }
    check_true("batch selection matches shared route decisions", selected_routes == expected_routes, detail=str(selected_routes))

    # 30. Field dedup keeps matching signals from different regimes separate
    same_field_delay0 = make_record(
        "shared_field_delay0",
        fitness=1.9,
        sharpe=1.3,
        field="shared_field",
        delay=0,
    )
    same_field_delay1 = make_record(
        "shared_field_delay1",
        fitness=1.8,
        sharpe=1.4,
        field="shared_field",
        delay=1,
    )
    selected_regimes = select_candidates([same_field_delay0, same_field_delay1], set(), set(), verbose=False)
    selected_keys = {record_workshop_key(r) for r in selected_regimes}
    check_true(
        "batch selection keeps same field in separate regimes",
        selected_keys == {record_workshop_key(same_field_delay0), record_workshop_key(same_field_delay1)},
        detail=str(selected_keys),
    )

    # 31. Negate generation is regime-aware and preserves source settings
    negate_delay0 = make_record("rank(x)", fitness=-2.0, sharpe=0.2, max_self_correlation=None, delay=0)
    negate_delay1 = make_record("rank(x)", fitness=-2.1, sharpe=0.3, max_self_correlation=None, delay=1)
    negate_candidates = generate_candidates(
        [negate_delay0, negate_delay1],
        [negate_delay0, negate_delay1],
        known_exprs={record_workshop_key(negate_delay0), record_workshop_key(negate_delay1)},
        verbose=False,
    )
    negate_delays = sorted(c.get("settings_override", {}).get("delay") for c in negate_candidates)
    check_true(
        "negate generation stays split by regime and preserves settings_override",
        len(negate_candidates) == 2 and negate_delays == [0, 1],
        detail=str(negate_delays),
    )

    # 32. Workshop history keeps delay-0 and delay-1 lineages separate
    history_seed_delay0 = make_record("seed_expr", fitness=1.7, sharpe=1.2, delay=0)
    history_seed_delay1 = make_record("seed_expr", fitness=1.8, sharpe=1.3, delay=1)
    history = WorkshopHistory([record_workshop_key(history_seed_delay0), record_workshop_key(history_seed_delay1)])
    child_delay0 = make_record("child_delay0", fitness=1.9, sharpe=1.4, delay=0)
    child_delay0["params"] = {"original_expression": "seed_expr"}
    child_delay1 = make_record("child_delay1", fitness=2.0, sharpe=1.5, delay=1)
    child_delay1["params"] = {"original_expression": "seed_expr"}
    history.update([child_delay0, child_delay1], iteration=0)
    prior_delay0 = history.context_for_enhance(record_workshop_key(history_seed_delay0))
    prior_delay1 = history.context_for_enhance(record_workshop_key(history_seed_delay1))
    ok_history = (
        [r.get("expression") for r in prior_delay0] == ["child_delay0"]
        and [r.get("expression") for r in prior_delay1] == ["child_delay1"]
    )
    check_true("workshop history separates regimes", ok_history)

    # 33. Audit output includes corr/regime and round-trips same expression across regimes
    from pathlib import Path
    from tempfile import TemporaryDirectory

    with TemporaryDirectory() as tmpdir:
        txt_path = Path(tmpdir) / "audit.txt"
        jsonl_path = Path(tmpdir) / "audit.jsonl"
        audit_record_delay0 = make_record(
            "audit_expr",
            fitness=1.9,
            sharpe=1.7,
            max_self_correlation=0.321,
            field="audit_field",
            delay=0,
        )
        audit_record_delay1 = make_record(
            "audit_expr",
            fitness=1.8,
            sharpe=1.7,
            max_self_correlation=0.654,
            field="audit_field",
            delay=1,
        )
        _write_audit([audit_record_delay0, audit_record_delay1], txt_path, jsonl_path)
        txt = txt_path.read_text()
        rows = [json.loads(line) for line in jsonl_path.read_text().splitlines() if line.strip()]
        reread = _read_from_audit(txt_path, jsonl_path)
        ok_audit = (
            "corr" in txt.splitlines()[0].lower()
            and "regime" in txt.splitlines()[0].lower()
            and "0.321" in txt
            and "delay0_subindustry_top3000" in txt
            and len(rows) == 2
            and {row.get("max_self_correlation") for row in rows} == {0.321, 0.654}
            and len(reread) == 2
            and {(row["expression"], row["regime"]) for row in reread}
            == {
                ("audit_expr", "delay0_subindustry_top3000"),
                ("audit_expr", "delay1_subindustry_top3000"),
            }
        )
        check_true("audit output includes corr/regime and round-trips", ok_audit)

    # 34. Admission promotes positive same-delay marginal contribution
    base_delay1 = _make_series(100, lambda i: 0.005 + (0.04 if i % 2 else -0.04))
    good_candidate = _make_series(100, lambda i: 0.03 + (0.02 if i % 3 == 0 else -0.01))
    portfolio_delay1 = {
        "base_d1": {
            "pnl": base_delay1,
            "delay": 1,
            "is_sharpe": 0.5,
            "is_fitness": 1.0,
        }
    }
    good_admission = score_record_admission(
        make_record("group_rank(good_field, sector)", fitness=1.6, sharpe=2.0, field="good_field"),
        good_candidate,
        portfolio_delay1,
        config=AdmissionConfig(min_fallback_delta_sharpe=0.01),
    )
    check_true(
        "admission promotes positive marginal pool Sharpe",
        good_admission["decision"] == PROMOTE and good_admission["delta_pool_sharpe"] > 0.01,
        detail=str(good_admission),
    )

    # 35. Admission rejects a candidate that lowers same-delay pool Sharpe
    bad_candidate = _make_series(100, lambda i: 0.001 + (0.05 if i % 2 else -0.05))
    bad_admission = score_record_admission(
        make_record("group_rank(bad_field, sector)", fitness=1.6, sharpe=2.0, field="bad_field"),
        bad_candidate,
        portfolio_delay1,
        config=AdmissionConfig(min_fallback_delta_sharpe=0.01),
    )
    check_true(
        "admission rejects negative marginal pool Sharpe",
        bad_admission["decision"] == REJECT
        and bad_admission["reason"] == "NON_POSITIVE_FALLBACK_DELTA_SHARPE",
        detail=str(bad_admission),
    )

    official_admission = score_record_admission(
        make_record("group_rank(official_field, sector)", fitness=1.6, sharpe=2.0, field="official_field"),
        bad_candidate,
        portfolio_delay1,
        config=AdmissionConfig(min_fallback_delta_sharpe=0.01),
        brain_performance={"score": {"before": 14355, "after": 14582}},
    )
    check_true(
        "admission promotes positive Brain score delta over proxy Sharpe delta",
        official_admission["decision"] == PROMOTE
        and official_admission["reason"] == "POSITIVE_BRAIN_SCORE_DELTA"
        and official_admission["brain_score_delta"] == 227.0,
        detail=str(official_admission),
    )

    # 36. Admission rejects high global self-correlation that misses 1.10x Sharpe exception
    high_corr_admission = score_record_admission(
        make_record("group_rank(highcorr_field, sector)", fitness=1.6, sharpe=1.0, field="highcorr_field"),
        base_delay1,
        {
            "submitted_clone": {
                "pnl": base_delay1,
                "delay": 1,
                "is_sharpe": 2.0,
                "is_fitness": 1.0,
            }
        },
        config=AdmissionConfig(min_fallback_delta_sharpe=0.01),
    )
    check_true(
        "admission rejects high global correlation",
        high_corr_admission["decision"] == REJECT
        and high_corr_admission["reason"] == "HIGH_GLOBAL_CORR",
        detail=str(high_corr_admission),
    )

    # 37. Admission watches missing candidate PnL instead of promoting
    missing_pnl_admission = score_record_admission(
        make_record("group_rank(missing_pnl_field, sector)", fitness=1.6, sharpe=2.0, field="missing_pnl_field"),
        {},
        portfolio_delay1,
        config=AdmissionConfig(min_fallback_delta_sharpe=0.01),
    )
    check_true(
        "admission marks missing PnL as WATCH",
        missing_pnl_admission["decision"] == WATCH
        and missing_pnl_admission["reason"] == "BRAIN_SCORE_UNAVAILABLE_AND_PENDING_PNL",
        detail=str(missing_pnl_admission),
    )

    # 38. Admission can reject from existing self-corr metadata even if PnL fetch is unavailable
    corr_fallback_admission = score_record_admission(
        make_record(
            "group_rank(corr_fallback_field, sector)",
            fitness=1.6,
            sharpe=1.0,
            field="corr_fallback_field",
            max_self_correlation=0.95,
            max_self_correlation_alpha_sharpe=2.0,
        ),
        {},
        {},
        config=AdmissionConfig(min_fallback_delta_sharpe=0.01),
    )
    check_true(
        "admission rejects high corr from record metadata without PnL",
        corr_fallback_admission["decision"] == REJECT
        and corr_fallback_admission["reason"] == "HIGH_GLOBAL_CORR",
        detail=str(corr_fallback_admission),
    )

    # 39. Delay-aware admission scores against same-delay pool but gates against global pool
    mixed_portfolio = {
        "delay0_base": {
            "pnl": base_delay1,
            "delay": 0,
            "is_sharpe": 0.5,
            "is_fitness": 1.0,
        },
        "delay1_clone": {
            "pnl": good_candidate,
            "delay": 1,
            "is_sharpe": 5.0,
            "is_fitness": 2.0,
        },
    }
    delay0_global_gate = score_record_admission(
        make_record("group_rank(delay0_field, sector)", fitness=1.6, sharpe=1.0, field="delay0_field", delay=0),
        good_candidate,
        mixed_portfolio,
        config=AdmissionConfig(min_fallback_delta_sharpe=0.01),
    )
    check_true(
        "admission uses same-delay pool and global corr gate",
        delay0_global_gate["same_delay_pool_size"] == 1
        and delay0_global_gate["max_corr_alpha_id"] == "delay1_clone"
        and delay0_global_gate["decision"] == REJECT,
        detail=str(delay0_global_gate),
    )

    # 40. Workshop promotion writes only PROMOTE records to template/submittable outputs
    with TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        promote_rec = make_record(
            "group_rank(ts_rank(promote_field, 60), sector)",
            fitness=1.6,
            sharpe=1.5,
            field="promote_field",
            delay=0,
        )
        promote_rec["alpha_id"] = "promote_alpha"
        reject_rec = make_record(
            "group_rank(reject_field, sector)",
            fitness=1.6,
            sharpe=1.5,
            field="reject_field",
        )
        reject_rec["alpha_id"] = "reject_alpha"

        def fake_admission_scorer(records: list[dict]) -> list[dict]:
            out = []
            for rec in records:
                enriched = dict(rec)
                if rec.get("alpha_id") == "promote_alpha":
                    enriched.update({
                        "admission_decision": PROMOTE,
                        "admission_reason": "POSITIVE_MARGINAL_CONTRIBUTION",
                        "admission_delay": rec.get("settings", {}).get("delay"),
                        "admission_delta_pool_sharpe": 0.25,
                        "admission_global_max_corr": 0.1,
                        "admission_residual_sharpe": 1.0,
                    })
                else:
                    enriched.update({
                        "admission_decision": REJECT,
                        "admission_reason": "NON_POSITIVE_DELTA",
                        "admission_delay": rec.get("settings", {}).get("delay"),
                        "admission_delta_pool_sharpe": -0.1,
                        "admission_global_max_corr": 0.2,
                        "admission_residual_sharpe": -0.5,
                    })
                out.append(enriched)
            return out

        _promote_to_templates(
            tmp / "missing_workshop_sim.jsonl",
            new_templates_path=tmp / "new_templates.jsonl",
            summary_path=tmp / "summary.txt",
            submittable_path=tmp / "submittable.jsonl",
            admission_path=tmp / "admission.jsonl",
            records=[promote_rec, reject_rec],
            admission_scorer=fake_admission_scorer,
        )
        promoted_rows = [json.loads(line) for line in (tmp / "submittable.jsonl").read_text().splitlines() if line.strip()]
        admission_rows = [json.loads(line) for line in (tmp / "admission.jsonl").read_text().splitlines() if line.strip()]
        new_template_rows = [json.loads(line) for line in (tmp / "new_templates.jsonl").read_text().splitlines() if line.strip()]
        ok_promotion_gate = (
            [row.get("alpha_id") for row in promoted_rows] == ["promote_alpha"]
            and len(admission_rows) == 2
            and [row.get("expression") for row in new_template_rows] == ["group_rank(ts_rank(promote_field, 60), sector)"]
            and new_template_rows[0].get("settings", {}).get("delay") == 0
            and promoted_rows[0].get("settings", {}).get("delay") == 0
        )
        check_true("workshop promotion gates templates by admission", ok_promotion_gate)

    # 41. LLM prompt helpers expose marginal portfolio direction
    dilutive_record = make_record(
        "group_rank(dilutive_field, sector)",
        fitness=1.6,
        sharpe=1.5,
        field="dilutive_field",
    )
    dilutive_record.update({
        "admission_decision": REJECT,
        "admission_reason": "NON_POSITIVE_DELTA",
        "admission_delta_pool_sharpe": -0.05,
        "admission_residual_sharpe": -0.2,
        "admission_global_max_corr": 0.82,
        "admission_same_delay_pool_size": 4,
    })
    prompt_line = _portfolio_feedback_line(dilutive_record)
    round_feedback = _build_round_feedback([dilutive_record], 3)
    check_true(
        "LLM prompt includes portfolio Sharpe direction",
        "decreases same-delay pool Sharpe" in prompt_line
        and "delta_pool_sharpe" in prompt_line
        and "dPool=-0.050" in round_feedback,
        detail=prompt_line,
    )

    n_tests = 55
    print(f"\nUnit tests: {n_tests - failures}/{n_tests} passed")
    return failures


# ---------------------------------------------------------------------------
# Integration test: compare computed vs Brain-reported for competition alphas
# ---------------------------------------------------------------------------

def _get_brain_self_corr(session, alpha_id: str) -> float | None:
    try:
        resp = session.get(f"https://api.worldquantbrain.com/alphas/{alpha_id}", timeout=15)
        if resp.status_code != 200:
            return None
        d = resp.json()
        return (d.get("is") or {}).get("selfCorrelation")
    except Exception:
        return None


def _pnl_stats(pnl: dict[str, float]) -> str:
    """Return a one-line summary of a PnL series for diagnostics."""
    if not pnl:
        return "empty"
    vals = list(pnl.values())
    import statistics
    mn = min(vals)
    mx = max(vals)
    mean = statistics.mean(vals)
    stdev = statistics.stdev(vals) if len(vals) > 1 else 0.0
    first5 = [f"{v:.5g}" for v in vals[:5]]
    return (f"n={len(vals)}  min={mn:.5g}  max={mx:.5g}  "
            f"mean={mean:.5g}  std={stdev:.5g}  first5={first5}")


def _diff_pnl(pnl: dict[str, float]) -> dict[str, float]:
    """First-difference a PnL series (levels → daily changes)."""
    keys = sorted(pnl)
    result = {}
    for i in range(1, len(keys)):
        result[keys[i]] = pnl[keys[i]] - pnl[keys[i - 1]]
    return result


def _corr_matrix(pnls: dict[str, dict[str, float]]) -> None:
    """Print pairwise |Pearson corr| matrix between all series in pnls."""
    from alpha_grammar.run_pipeline import compute_max_self_corr
    ids = list(pnls)
    print(f"\n  Pairwise |corr| matrix ({len(ids)} alphas):")
    header = "  " + " " * 14 + "  ".join(f"{aid[:8]:>8}" for aid in ids)
    print(header)
    for aid_a in ids:
        row_parts = []
        for aid_b in ids:
            if aid_a == aid_b:
                row_parts.append("    1.000")
            else:
                c = compute_max_self_corr(pnls[aid_a], {aid_b: pnls[aid_b]})
                row_parts.append(f"    {c:.3f}" if c is not None else "     None")
        print(f"  {aid_a[:8]:>8}  {'  '.join(p.strip() for p in row_parts)}")


def run_integration_test(
    session,
    competition_id: str,
    n: int,
    tol: float,
) -> int:
    """Fetch competition alphas, compute self-corr, compare with Brain. Returns failures."""
    from wq_alpha.brain.api import fetch_competition_alpha_ids, fetch_pnl_series
    from alpha_grammar.run_pipeline import compute_max_self_corr

    print(f"\n--- Integration test: {competition_id} (n={n}, tol={tol}) ---")

    all_ids = fetch_competition_alpha_ids(session, competition_id)
    if not all_ids:
        print("  No competition alphas found — skipping integration test.")
        return 0

    probe_ids = all_ids[:n]
    print(f"  Competition has {len(all_ids)} alphas; probing first {len(probe_ids)}")

    # Fetch all PnL series upfront
    print(f"\n  Fetching PnL for all {len(all_ids)} competition alphas ...")
    all_pnls_raw: dict[str, dict[str, float]] = {}
    for aid in all_ids:
        pnl = fetch_pnl_series(session, aid)
        if pnl:
            all_pnls_raw[aid] = pnl
    print(f"  Got PnL for {len(all_pnls_raw)}/{len(all_ids)} alphas")

    # --- Diagnostics: inspect PnL series ---
    print("\n  PnL series diagnostics (raw values):")
    for aid, pnl in all_pnls_raw.items():
        print(f"    {aid}: {_pnl_stats(pnl)}")

    # Check if values look like cumulative (large/growing) vs daily returns (small ~0.001)
    sample_vals = []
    for pnl in all_pnls_raw.values():
        sample_vals.extend(list(pnl.values())[:20])
    if sample_vals:
        import statistics
        abs_median = statistics.median(abs(v) for v in sample_vals)
        is_likely_cumulative = abs_median > 0.1
        print(f"\n  Median |value|={abs_median:.5g} → "
              f"{'looks CUMULATIVE (levels)' if is_likely_cumulative else 'looks like daily returns'}")

        if is_likely_cumulative:
            print("  → Will also test with first-differenced (daily-change) series")

    # Pairwise corr matrix (raw)
    if len(all_pnls_raw) >= 2:
        _corr_matrix(all_pnls_raw)

    if len(all_pnls_raw) < 2:
        print("  Not enough PnL data to compute pairwise correlation — skipping.")
        return 0

    # --- Main comparison loop ---
    # compute_max_self_corr first-differences internally (cumulative → daily returns)
    failures = 0
    results: list[dict[str, Any]] = []

    for aid in probe_ids:
        brain_corr = _get_brain_self_corr(session, aid)
        other_pnls = {k: v for k, v in all_pnls_raw.items() if k != aid}
        cand_pnl = all_pnls_raw.get(aid, {})
        computed = compute_max_self_corr(cand_pnl, other_pnls)
        results.append({"alpha_id": aid, "brain": brain_corr, "computed": computed})

    print(f"\n  {'alpha_id':<14}  {'brain':>8}  {'computed':>10}  {'delta':>8}  status")
    print("  " + "-" * 65)
    for r in results:
        aid = r["alpha_id"]
        brain = r["brain"]
        computed = r["computed"]

        brain_s = f"{brain:.4f}" if brain is not None else "    None"
        comp_s = f"{computed:.4f}" if computed is not None else "      None"

        if brain is None and computed is None:
            status = "SKIP (both None)"
            delta_s = "     N/A"
        elif brain is None:
            status = f"SKIP (brain=None)"
            delta_s = "     N/A"
        elif computed is None:
            status = "FAIL (no PnL)"
            delta_s = "     N/A"
            failures += 1
        else:
            delta = abs(computed - brain)
            delta_s = f"{delta:.4f}"
            status = "PASS" if delta <= tol else f"FAIL (>{tol})"
            if delta > tol:
                failures += 1

        print(f"  {aid:<14}  {brain_s:>8}  {comp_s:>10}  {delta_s:>8}  {status}")

    total_with_brain = sum(1 for r in results if r["brain"] is not None and r["computed"] is not None)
    passed = sum(
        1 for r in results
        if r["brain"] is not None and r["computed"] is not None
        and abs(r["computed"] - r["brain"]) <= tol
    )
    print(f"\n  Integration: {passed}/{total_with_brain} within tol={tol}")

    # Note: Brain computes selfCorrelation at submission time against the portfolio
    # that existed THEN. Discrepancies are expected for alphas submitted long ago
    # when the portfolio composition has since changed.
    print("  Note: discrepancies expected for alphas submitted against a different portfolio.")
    return failures


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Verify PnL-based self-correlation computation.")
    p.add_argument("--credentials", default="brain.txt")
    p.add_argument("--competition-id", default="IQC2026S1")
    p.add_argument("--n", type=int, default=10,
                   help="Number of competition alphas to probe (default 10)")
    p.add_argument("--tol", type=float, default=0.05,
                   help="Acceptable absolute error vs Brain's value (default 0.05)")
    p.add_argument("--unit-only", action="store_true",
                   help="Run only unit tests, skip network integration test")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    unit_failures = run_unit_tests()

    if args.unit_only:
        sys.exit(1 if unit_failures else 0)

    from wq_alpha.brain.api import sign_in
    session = sign_in(args.credentials)

    integ_failures = run_integration_test(
        session,
        competition_id=args.competition_id,
        n=args.n,
        tol=args.tol,
    )

    total = unit_failures + integ_failures
    print(f"\n{'='*60}")
    print(f"Total failures: {total}  (unit={unit_failures}, integration={integ_failures})")
    sys.exit(1 if total else 0)


if __name__ == "__main__":
    main()
