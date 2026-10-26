"""Test self-correlation availability for Brain alphas.

FINDINGS:
- Regular API simulations: SELF_CORRELATION check stays PENDING forever.
  is.selfCorrelation is absent (None). No known endpoint resolves it.
- Submitted/competition alphas (stage=OS, status=ACTIVE): is.selfCorrelation
  is populated as a float (max absolute correlation over IS period).
  These also lack the SELF_CORRELATION check entry.

TRIGGER ENDPOINT (does NOT resolve for unsubmitted):
  PATCH /alphas/{id}/check  body=[{"id": alpha_id}]
  Returns 200 with alpha JSON but selfCorrelation still None.

CONCLUSION:
  Self-correlation is only computed when an alpha is added to a competition
  or portfolio (transitions to stage=OS). The website screenshot showing
  correlation for an unsubmitted alpha may reflect an alpha that was
  previously submitted, removed, but correlation was retained, OR a UI
  feature that triggers full portfolio-style checks.

Usage:
    python -m tools.test_self_correlation_timing [--n 5]
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path


MASTER = Path("artifacts/sim/master.jsonl")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=5)
    p.add_argument("--max-minutes", type=float, default=3.0)
    p.add_argument("--poll-seconds", type=float, default=30.0)
    p.add_argument("--credentials", default="brain.txt")
    p.add_argument("--lane", default="grammar")
    p.add_argument("--trigger-check", action="store_true",
                   help="PATCH /check to attempt to trigger correlation computation")
    return p.parse_args()


def _get_corr_info(session, alpha_id: str) -> dict:
    try:
        resp = session.get(f"https://api.worldquantbrain.com/alphas/{alpha_id}", timeout=15)
        if resp.status_code != 200:
            return {"error": f"HTTP_{resp.status_code}"}
        d = resp.json()
        is_data = d.get("is") or {}
        checks = is_data.get("checks") or []
        sc_check = next((c for c in checks if c.get("name") == "SELF_CORRELATION"), None)
        return {
            "stage": d.get("stage"),
            "status": d.get("status"),
            "selfCorrelation": is_data.get("selfCorrelation"),
            "prodCorrelation": is_data.get("prodCorrelation"),
            "check_result": sc_check.get("result") if sc_check else "missing",
            "check_value": sc_check.get("value") if sc_check else None,
        }
    except Exception as e:
        return {"error": str(e)}


def _trigger_check(session, alpha_id: str) -> None:
    """PATCH /alphas/{id}/check to request correlation computation."""
    try:
        r = session.patch(
            f"https://api.worldquantbrain.com/alphas/{alpha_id}/check",
            json=[{"id": alpha_id}],
            timeout=15,
        )
        print(f"    PATCH /check: {r.status_code}")
    except Exception as e:
        print(f"    PATCH /check error: {e}")


def main() -> None:
    args = parse_args()

    from wq_alpha.brain.api import sign_in
    session = sign_in(args.credentials)

    # --- Competition alphas (submitted, selfCorrelation should be populated) ---
    r = session.get("https://api.worldquantbrain.com/competitions/IQC2026S1/alphas",
                    params={"limit": args.n})
    comp_alphas = r.json().get("results", []) if r.status_code == 200 else []

    # --- Grammar (unsubmitted) alphas ---
    grammar_alphas: list[dict] = []
    if MASTER.exists():
        with MASTER.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if (rec.get("status") == "done"
                        and rec.get("search_lane") == args.lane
                        and (rec.get("is_sharpe") or 0) >= 1.25
                        and rec.get("alpha_id")):
                    grammar_alphas.append(rec)
                    if len(grammar_alphas) >= args.n:
                        break

    print("=== Competition alphas (stage=OS) ===")
    for a in comp_alphas[:args.n]:
        info = _get_corr_info(session, a["id"])
        print(f"  {a['id']}: stage={info.get('stage')}  "
              f"selfCorr={info.get('selfCorrelation')}  "
              f"prodCorr={info.get('prodCorrelation')}  "
              f"check={info.get('check_result')}")

    print()
    print(f"=== Grammar alphas (stage=IS, unsubmitted, sharpe>=1.25) ===")
    for rec in grammar_alphas:
        aid = rec["alpha_id"]
        if args.trigger_check:
            print(f"  Triggering check for {aid}...")
            _trigger_check(session, aid)
        info = _get_corr_info(session, aid)
        print(f"  {aid}  sharpe={rec.get('is_sharpe')}:  "
              f"stage={info.get('stage')}  "
              f"selfCorr={info.get('selfCorrelation')}  "
              f"check={info.get('check_result')}:{info.get('check_value')}")

    if args.trigger_check and grammar_alphas:
        print(f"\nPolling {len(grammar_alphas)} grammar alphas for up to {args.max_minutes} min...")
        max_polls = int(args.max_minutes * 60 / args.poll_seconds)
        resolved: set[str] = set()
        for poll in range(1, max_polls + 1):
            time.sleep(args.poll_seconds)
            print(f"  Poll {poll}/{max_polls} ({poll * args.poll_seconds:.0f}s):")
            for rec in grammar_alphas:
                aid = rec["alpha_id"]
                if aid in resolved:
                    continue
                info = _get_corr_info(session, aid)
                sc = info.get("selfCorrelation")
                print(f"    {aid}: selfCorr={sc}  check={info.get('check_result')}")
                if sc is not None:
                    resolved.add(aid)
            if len(resolved) == len(grammar_alphas):
                print("All resolved!")
                break
        if resolved:
            print(f"\n{len(resolved)}/{len(grammar_alphas)} grammar alphas resolved.")
        else:
            print("\nNone resolved. SELF_CORRELATION remains PENDING for unsubmitted alphas.")


if __name__ == "__main__":
    main()
