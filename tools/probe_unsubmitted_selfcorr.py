"""Probe whether Brain API can return self-correlation for unsubmitted alphas.

Checks every known endpoint and recordset column for any correlation-related
data on alphas that were simulated but never submitted to a competition or
portfolio.

Usage:
    python -m tools.probe_unsubmitted_selfcorr
    python -m tools.probe_unsubmitted_selfcorr --n 5 --lane grammar
    python -m tools.probe_unsubmitted_selfcorr --alpha-id XXXXXXXXX
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

BRAIN_BASE = "https://api.worldquantbrain.com"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=5,
                   help="Number of unsubmitted alphas to probe (default: 5)")
    p.add_argument("--credentials", default="brain.txt")
    p.add_argument("--sim-log", default="artifacts/sim/master.jsonl")
    p.add_argument("--alpha-id", default=None,
                   help="Probe a specific alpha ID instead of loading from sim-log")
    p.add_argument("--lane", default=None,
                   help="Filter sim-log by search_lane (e.g. 'grammar')")
    p.add_argument("--dump-detail", action="store_true",
                   help="Dump full JSON of /alphas/{id} for first alpha")
    p.add_argument("--poll", action="store_true",
                   help="Poll repeatedly until selfCorrelation appears or --max-minutes elapsed")
    p.add_argument("--poll-seconds", type=float, default=60.0,
                   help="Seconds between poll rounds (default: 60)")
    p.add_argument("--max-minutes", type=float, default=10.0,
                   help="Stop polling after this many minutes (default: 10)")
    return p.parse_args()


def _load_unsubmitted(sim_log: str, n: int, lane: str | None) -> list[dict]:
    """Return up to n unsubmitted (no selfCorrelation) done records from sim-log."""
    results: list[dict] = []
    p = Path(sim_log)
    if not p.exists():
        return results
    with p.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if rec.get("status") != "done":
                continue
            if not rec.get("alpha_id"):
                continue
            if lane and rec.get("search_lane") != lane:
                continue
            # Only want ones where selfCorrelation was never populated
            if rec.get("max_self_correlation") is not None:
                continue
            results.append(rec)
            if len(results) >= n:
                break
    return results


def _probe_detail(session, alpha_id: str) -> dict:
    """GET /alphas/{id} and extract all correlation-related fields."""
    r = session.get(f"{BRAIN_BASE}/alphas/{alpha_id}", timeout=15)
    if r.status_code != 200:
        return {"error": r.status_code}
    d = r.json()
    is_data = d.get("is") or {}
    checks = is_data.get("checks") or []
    sc_check = next((c for c in checks if c.get("name") == "SELF_CORRELATION"), None)
    return {
        "raw": d,
        "selfCorrelation": is_data.get("selfCorrelation"),
        "prodCorrelation": is_data.get("prodCorrelation"),
        "sc_check_result": sc_check.get("result") if sc_check else "missing",
        "sc_check_value": sc_check.get("value") if sc_check else None,
        "grade": d.get("grade"),
        "stage": d.get("stage"),
        "status": d.get("status"),
    }


def _probe_recordsets(session, alpha_id: str) -> dict:
    """Enumerate recordsets and flag any with correlation-related columns."""
    r = session.get(f"{BRAIN_BASE}/alphas/{alpha_id}/recordsets", timeout=15)
    if r.status_code != 200:
        return {"error": r.status_code}
    try:
        body = r.json()
    except Exception:
        return {"error": f"empty_body (status={r.status_code})"}
    rs_names = [x.get("name") for x in body.get("results", [])]
    results: dict[str, dict] = {"names": rs_names}
    for rs_name in rs_names:
        r2 = session.get(
            f"{BRAIN_BASE}/alphas/{alpha_id}/recordsets/{rs_name}", timeout=15
        )
        if r2.status_code != 200:
            results[rs_name] = {"error": r2.status_code}
            continue
        try:
            data = r2.json()
        except Exception:
            results[rs_name] = {"error": f"empty_body (status={r2.status_code})"}
            continue
        schema = data.get("schema") or {}
        cols = [p["name"] for p in schema.get("properties", [])]
        records = data.get("records") or []
        corr_cols = [c for c in cols if "corr" in c.lower() or "self" in c.lower()]
        results[rs_name] = {
            "columns": cols,
            "rows": len(records),
            "corr_cols": corr_cols,
            # If there are correlation columns, grab first row
            "sample": records[0] if records and corr_cols else None,
        }
    return results


def _probe_extra_endpoints(session, alpha_id: str) -> dict:
    """Try known and speculative sub-endpoints that might expose correlation."""
    candidates = [
        "correlation",
        "self-correlation",
        "competitions",
        "portfolio",
        "portfolios",
        "checks",
        "analytics",
    ]
    results: dict[str, dict] = {}
    for suffix in candidates:
        r = session.get(f"{BRAIN_BASE}/alphas/{alpha_id}/{suffix}", timeout=10)
        entry: dict = {"status": r.status_code}
        if r.status_code == 200:
            try:
                body = r.json()
                entry["body_preview"] = json.dumps(body)[:300]
            except Exception:
                entry["body_preview"] = r.text[:300]
        results[suffix] = entry
    return results


def _report_alpha(alpha_id: str, detail: dict, recordsets: dict, extras: dict) -> None:
    sc = detail.get("selfCorrelation")
    pc = detail.get("prodCorrelation")
    sc_result = detail.get("sc_check_result")
    sc_value = detail.get("sc_check_value")

    # Headline
    if sc is not None:
        print(f"  *** SELF_CORRELATION AVAILABLE: {sc:.4f}")
    else:
        print(f"  selfCorrelation  = {sc}  (PENDING/null)")
    if pc is not None:
        print(f"  *** PROD_CORRELATION AVAILABLE: {pc:.4f}")
    else:
        print(f"  prodCorrelation  = {pc}")
    print(f"  SC check         = {sc_result} (value={sc_value})")
    print(f"  grade/stage      = {detail.get('grade')} / {detail.get('stage')}")

    # Recordsets
    print(f"  recordsets       = {recordsets.get('names')}")
    for rs_name in (recordsets.get("names") or []):
        rs = recordsets.get(rs_name, {})
        if rs.get("corr_cols"):
            print(f"  ** CORR COLS in '{rs_name}': {rs['corr_cols']}")
            if rs.get("sample"):
                print(f"     sample row: {rs['sample']}")
        else:
            print(f"  '{rs_name}': {rs.get('rows')} rows, cols={rs.get('columns')}")

    # Extra endpoints
    for suffix, info in extras.items():
        status = info.get("status")
        if status == 200:
            print(f"  ** /{suffix}: 200 OK → {info.get('body_preview','')[:120]}")
        # Silent on non-200 unless unexpected
        elif status not in (403, 404, 405, 400, 422):
            print(f"  /{suffix}: {status}")


def main() -> None:
    args = parse_args()

    from core.brain.api import sign_in
    session = sign_in(args.credentials)

    if args.alpha_id:
        records = [{"alpha_id": args.alpha_id, "expression": "(specified)", "is_sharpe": "?"}]
    else:
        records = _load_unsubmitted(args.sim_log, args.n, args.lane)

    if not records:
        print(f"No unsubmitted alphas found in {args.sim_log}.")
        return

    lane_tag = f"  lane={args.lane}" if args.lane else ""
    print(f"Probing {len(records)} unsubmitted alpha(s){lane_tag} ...\n")

    alpha_ids = [(rec.get("alpha_id") or rec.get("id", ""), rec) for rec in records]

    if args.poll:
        _poll_loop(session, alpha_ids, args)
    else:
        _single_pass(session, alpha_ids, args)


def _has_correlation(detail: dict, recordsets: dict) -> bool:
    if detail.get("selfCorrelation") is not None or detail.get("prodCorrelation") is not None:
        return True
    for rs_name in (recordsets.get("names") or []):
        if recordsets.get(rs_name, {}).get("corr_cols"):
            return True
    return False


def _single_pass(session, alpha_ids: list, args) -> None:
    any_found = False
    for alpha_id, rec in alpha_ids:
        expr = str(rec.get("expression", ""))[:70]
        sharpe = rec.get("is_sharpe")
        print(f"=== {alpha_id}  sharpe={sharpe}  {expr} ===")

        detail = _probe_detail(session, alpha_id)
        if "error" in detail:
            print(f"  ERROR fetching detail: {detail['error']}")
            print()
            continue

        if args.dump_detail and alpha_id == alpha_ids[0][0]:
            print("  --- full detail JSON ---")
            print(json.dumps(detail.get("raw", {}), indent=2)[:3000])
            print("  --- end ---")

        recordsets = _probe_recordsets(session, alpha_id)
        extras = _probe_extra_endpoints(session, alpha_id)
        _report_alpha(alpha_id, detail, recordsets, extras)

        if _has_correlation(detail, recordsets):
            any_found = True
        print()

    print("=" * 60)
    if any_found:
        print("RESULT: Self-correlation data IS available for unsubmitted alphas.")
    else:
        print("RESULT: No self-correlation data found for any unsubmitted alpha.")
        print("        Brain only populates selfCorrelation after submission.")


def _poll_loop(session, alpha_ids: list, args) -> None:
    deadline = time.monotonic() + args.max_minutes * 60
    poll_interval = args.poll_seconds
    round_num = 0
    resolved: dict[str, float] = {}  # alpha_id -> selfCorrelation value

    while time.monotonic() < deadline:
        round_num += 1
        elapsed = (args.max_minutes * 60 - (deadline - time.monotonic())) / 60
        print(f"--- Poll round {round_num}  (t={elapsed:.1f}m / {args.max_minutes}m) ---")

        for alpha_id, rec in alpha_ids:
            if alpha_id in resolved:
                continue
            detail = _probe_detail(session, alpha_id)
            sc = detail.get("selfCorrelation")
            pc = detail.get("prodCorrelation")
            sc_result = detail.get("sc_check_result")
            sc_value = detail.get("sc_check_value")
            if sc is not None:
                resolved[alpha_id] = sc
                print(f"  *** RESOLVED {alpha_id}: selfCorrelation={sc:.4f}  prodCorr={pc}")
            else:
                print(f"  {alpha_id}: selfCorr=None  prodCorr={pc}  "
                      f"SC_check={sc_result}(val={sc_value})")

        if len(resolved) == len(alpha_ids):
            print("\nAll alphas resolved.")
            break

        remaining = deadline - time.monotonic()
        if remaining <= 0:
            break
        wait = min(poll_interval, remaining)
        print(f"  sleeping {wait:.0f}s ...\n")
        time.sleep(wait)

    print("=" * 60)
    if resolved:
        print(f"RESULT: selfCorrelation appeared for {len(resolved)}/{len(alpha_ids)} alphas:")
        for aid, val in resolved.items():
            print(f"  {aid}: {val:.4f}")
    else:
        print(f"RESULT: selfCorrelation stayed None for all {len(alpha_ids)} alphas "
              f"over {args.max_minutes:.0f} minutes.")


if __name__ == "__main__":
    main()
