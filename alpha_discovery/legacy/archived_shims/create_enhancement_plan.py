"""ARCHIVED one-off script.

Kept as a historical record of a retired lane. It performs work at module
level (writes files / authenticates), so importing it would have side
effects; the guard below makes it inert unless run deliberately.
"""
if __name__ != "__main__":  # pragma: no cover - archived script
    raise SystemExit(0)

"""Generate enhancement_v4_plan.jsonl — 20 candidates spanning 9 orthogonal signal dimensions.

Groups:
  A — split timing fix (group_rank eliminates integer ties)
  B — composite temporal acceleration (delta block)
  C — composite + medium-term price momentum (120d / 252d)
  D — composite + volume-confirmed reversal
  E — composite + Sloan accruals (earnings quality)
  F — composite + options low-IV
  G — triple signal: reversal + composite + momentum
  H — composite + balance sheet quality (F6 leverage)
  I — slower pv1xa4 reversal windows (lower TO)
"""

import hashlib
import json
from pathlib import Path

# ---------------------------------------------------------------------------
# Composite building blocks (V3 drop-netdebt weights — best known at Sharpe 1.61)
# ---------------------------------------------------------------------------
COMP = (
    "2.0*anl4_totassets_flag + anl4_ffo_flag + 1.19*anl4_bvps_flag"
    " + 0.3*anl4_ptpr_flag + 0.4*anl4_capex_flag + 0.3*anl4_rd_exp_flag"
    " + 0.6*anl4_netprofit_flag"
)
RANK = (
    "rank(anl4_totassets_flag) + rank(anl4_bvps_flag)"
    " + rank(anl4_ffo_flag) + rank(anl4_netprofit_flag)"
)
COMP_FULL = f"normalize({COMP} + ({RANK}))"

# ---------------------------------------------------------------------------
# Candidate list: (template_id, dataset_id, expression)
# ---------------------------------------------------------------------------
CANDIDATES = [
    # --- A: split timing fix ---
    # Original fails CONCENTRATED_WEIGHT because ts_arg_max returns integers 0-59
    # and most stocks tie at 0. group_rank spreads uniformly within group.
    (
        "enh_split_v1", "pv1",
        "group_rank(ts_arg_max(split, 60), subindustry)",
    ),
    (
        "enh_split_v2", "pv1",
        "group_rank(ts_arg_max(split, 40), sector)",
    ),
    (
        "enh_split_v3", "pv1",
        "normalize(group_rank(ts_arg_max(split, 60), subindustry)"
        " + 0.1*rank(ts_sum(split, 252)))",
    ),

    # --- B: temporal acceleration (delta block) ---
    # All existing templates are level-only. Adding rank(ts_delta(top_fields, dw))
    # as a 3rd block adds directional momentum: stocks actively improving, not just high.
    (
        "enh_delta_dw10", "analyst4",
        f"normalize(({COMP}) + ({RANK})"
        " + rank(ts_delta(anl4_totassets_flag + anl4_bvps_flag, 10)))",
    ),
    (
        "enh_delta_dw20", "analyst4",
        f"normalize(({COMP}) + ({RANK})"
        " + rank(ts_delta(anl4_totassets_flag + anl4_bvps_flag, 20)))",
    ),
    (
        "enh_delta_dw40", "analyst4",
        f"normalize(({COMP}) + ({RANK})"
        " + rank(ts_delta(anl4_totassets_flag + anl4_bvps_flag, 40)))",
    ),

    # --- C: composite + medium-term price momentum ---
    # pv1xa4 already uses short-term reversal. Medium-term momentum (3-12 months)
    # is the opposite regime and orthogonal. Fundamental + momentum = high conviction.
    (
        "enh_momentum_120", "analyst4",
        f"normalize({COMP_FULL} + group_rank(ts_rank(returns, 120), sector))",
    ),
    (
        "enh_momentum_252", "analyst4",
        f"normalize({COMP_FULL} + group_rank(ts_rank(returns, 252), sector))",
    ),

    # --- D: composite + volume-confirmed reversal ---
    # pv1xa4 uses plain return reversal. Volume-confirmed: high-volume drawdowns
    # represent forced/panic selling -> stronger mean-reversion premium.
    (
        "enh_vol_rev_w10", "analyst4",
        f"normalize({COMP_FULL}"
        " + group_rank(-ts_mean(ts_zscore(returns, 10), 5)"
        " * (0.5 + 0.5*ts_rank(volume, 10)), sector))",
    ),
    (
        "enh_vol_rev_w20", "analyst4",
        f"normalize({COMP_FULL}"
        " + group_rank(-ts_mean(ts_zscore(returns, 20), 10)"
        " * (0.5 + 0.5*ts_rank(volume, 20)), sector))",
    ),

    # --- E: composite + Sloan accruals (earnings quality) ---
    # Analyst flags measure analyst *sentiment*. Sloan accruals measure cash earnings
    # quality (cashflow_op - income). Orthogonal: sentiment vs actual quality.
    (
        "enh_accruals_60", "analyst4",
        f"normalize({COMP_FULL}"
        " + group_rank(ts_zscore(ts_backfill(cashflow_op, 63)"
        " - ts_backfill(income, 63), 60), subindustry))",
    ),
    (
        "enh_accruals_120", "analyst4",
        f"normalize({COMP_FULL}"
        " + group_rank(ts_zscore(ts_backfill(cashflow_op, 63)"
        " - ts_backfill(income, 63), 120), sector))",
    ),

    # --- F: composite + options low-IV ---
    # Options never passed standalone, but as a filter: low-IV = options market
    # not pricing tail risk = analyst upgrades not yet priced in by derivatives.
    (
        "enh_lowiv_30", "analyst4",
        f"normalize({COMP_FULL}"
        " + group_rank(-ts_rank(implied_volatility_mean_30, 60), sector))",
    ),
    (
        "enh_lowiv_call30", "analyst4",
        f"normalize({COMP_FULL}"
        " + group_rank(-ts_rank(implied_volatility_call_30, 60), subindustry))",
    ),

    # --- G: triple signal — reversal + composite + momentum ---
    # Three genuinely orthogonal time scales:
    #   short-term reversal (HF, <1 week)
    #   fundamental revision (LF, ~1 month)
    #   price momentum (UF, 6-12 months)
    # Sharpe should be near-additive since correlations are near-zero.
    (
        "enh_triple_120", "analyst4",
        f"normalize(group_rank(-ts_mean(ts_zscore(returns, 5), 5), sector)"
        f" + {COMP_FULL}"
        " + group_rank(ts_rank(returns, 120), sector))",
    ),
    (
        "enh_triple_252", "analyst4",
        f"normalize(group_rank(-ts_mean(ts_zscore(returns, 10), 5), sector)"
        f" + {COMP_FULL}"
        " + group_rank(ts_rank(returns, 252), sector))",
    ),

    # --- H: composite + balance sheet quality (F6 leverage) ---
    # Analyst revision + low leverage = improving AND financially sound.
    # ts_backfill(63) handles quarterly reporting gaps in F6 fields.
    (
        "enh_f6_quality", "analyst4",
        f"normalize({COMP} + ({RANK})"
        " + 0.5*rank(ts_backfill(-liabilities/assets, 63)))",
    ),

    # --- I: slower pv1xa4 reversal windows ---
    # Current pv1xa4 uses window=5-20, dw=3-10 (TO ~50-60%).
    # Slower windows reduce TO, improving is_fitness = Sharpe/sqrt(TO).
    (
        "enh_slow_w20", "pv1",
        "normalize(group_rank(-ts_mean(ts_zscore(returns, 20), 10), sector)"
        " + normalize(2.0*anl4_totassets_flag + anl4_ffo_flag + 1.5*anl4_bvps_flag"
        " + 0.5*anl4_ptpr_flag + 0.5*anl4_netdebt_flag + 0.4*anl4_capex_flag"
        " + 0.3*anl4_rd_exp_flag))",
    ),
    (
        "enh_slow_w40", "pv1",
        "normalize(group_rank(-ts_mean(ts_zscore(returns, 40), 20), sector)"
        " + normalize(2.0*anl4_totassets_flag + anl4_ffo_flag + 1.5*anl4_bvps_flag"
        " + 0.5*anl4_ptpr_flag + 0.5*anl4_netdebt_flag + 0.4*anl4_capex_flag"
        " + 0.3*anl4_rd_exp_flag))",
    ),
    (
        "enh_slow_w60", "pv1",
        "normalize(group_rank(-ts_mean(ts_zscore(returns, 60), 20), sector)"
        " + normalize(2.0*anl4_totassets_flag + anl4_ffo_flag + 1.5*anl4_bvps_flag"
        " + 0.5*anl4_ptpr_flag + 0.5*anl4_netdebt_flag + 0.4*anl4_capex_flag"
        " + 0.3*anl4_rd_exp_flag))",
    ),
]

# ---------------------------------------------------------------------------
# Write JSONL
# ---------------------------------------------------------------------------
out = Path("alpha_search/artifacts/enhancement_v4_plan.jsonl")
out.parent.mkdir(parents=True, exist_ok=True)

with out.open("w") as f:
    for i, (tid, dsid, expr) in enumerate(CANDIDATES):
        cid = hashlib.sha1(expr.encode()).hexdigest()[:16]
        cat = "price_volume" if dsid == "pv1" else "analyst"
        rec = {
            "candidate_id": cid,
            "template_id": tid,
            "dataset_id": dsid,
            "category": cat,
            "search_lane": "enhancement",
            "expression": expr,
            "fields": [],
            "group": "",
            "params": {},
            "priority_score": round(4.20 - i * 0.01, 3),
        }
        f.write(json.dumps(rec) + "\n")

print(f"Wrote {len(CANDIDATES)} candidates to {out}")
print()
for i, (tid, dsid, expr) in enumerate(CANDIDATES):
    print(f"  {i+1:2d}. [{dsid}] {tid}")
    print(f"      {expr[:100]}{'...' if len(expr) > 100 else ''}")
