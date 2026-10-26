"""Probe Brain alpha detail endpoint to find per-year performance data.

Usage:
    python probe_alpha_detail.py --alpha-id QPlr1NPQ --credentials brain.txt
"""

import argparse
import json
from wq_alpha.brain.api import sign_in


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--alpha-id", required=True)
    p.add_argument("--credentials", default="brain.txt")
    p.add_argument("--dump", action="store_true", help="Dump full JSON")
    args = p.parse_args()

    session = sign_in(args.credentials)

    # 1. Basic alpha detail
    detail = session.get(
        f"https://api.worldquantbrain.com/alphas/{args.alpha_id}"
    ).json()

    if args.dump:
        print(json.dumps(detail, indent=2))
        return

    # 2. Print top-level keys
    print("=== Top-level keys ===")
    for k, v in detail.items():
        if isinstance(v, dict):
            print(f"  {k}: dict({list(v.keys())})")
        elif isinstance(v, list):
            print(f"  {k}: list[{len(v)}]")
        else:
            print(f"  {k}: {v!r}")

    # 3. Drill into 'is' (in-sample) section
    is_data = detail.get("is", {})
    print("\n=== 'is' keys ===")
    for k, v in is_data.items():
        if isinstance(v, dict):
            print(f"  {k}: dict({list(v.keys())})")
        elif isinstance(v, list):
            print(f"  {k}: list[{len(v)}] → {v[:2]}")
        else:
            print(f"  {k}: {v!r}")

    # 4. Try known sub-endpoints
    print("\n=== Trying sub-endpoints ===")
    for suffix in ["recordsets", "yearly", "annual", "performance", "breakdown"]:
        url = f"https://api.worldquantbrain.com/alphas/{args.alpha_id}/{suffix}"
        resp = session.get(url)
        raw = resp.text.strip()
        print(f"  GET .../{suffix} → {resp.status_code} ({len(raw)} bytes)")
        if raw:
            print(f"    raw[:200]: {raw[:200]}")

    # 5. Fetch each recordset's actual data
    rs_list = session.get(
        f"https://api.worldquantbrain.com/alphas/{args.alpha_id}/recordsets"
    ).json().get("results", [])
    print(f"\n=== Recordset names: {[r['name'] for r in rs_list]} ===")

    # Fetch yearly-stats in full
    url = f"https://api.worldquantbrain.com/alphas/{args.alpha_id}/recordsets/yearly-stats"
    data = session.get(url).json()
    cols = [p["name"] for p in data["schema"]["properties"]]
    print(f"\n=== yearly-stats columns: {cols} ===")
    print(f"{'year':<6}", "  ".join(f"{c:<10}" for c in cols[1:]))
    for row in data["records"]:
        print(f"{row[0]:<6}", "  ".join(f"{str(v):<10}" for v in row[1:]))


if __name__ == "__main__":
    main()
