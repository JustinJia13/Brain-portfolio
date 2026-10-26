"""Check active/pending Brain simulations."""
import json
from wq_alpha.brain.api import sign_in

BASE = "https://api.worldquantbrain.com"
session = sign_in("brain.txt")

# Try alphas endpoint with various status filters
for params in [
    {"status": "SIMULATING", "limit": 20},
    {"status": "PENDING", "limit": 20},
    {"stage": "IS", "limit": 20},
    {"limit": 10},  # most recent
]:
    resp = session.get(f"{BASE}/alphas", params=params)
    if resp.status_code != 200:
        print(f"GET /alphas {params} → {resp.status_code}")
        continue
    data = resp.json()
    count = data.get("count", 0)
    results = data.get("results", [])
    print(f"GET /alphas {params} → {count} total, {len(results)} returned")
    for a in results:
        print(f"  {a.get('id')}  grade={a.get('grade')}  stage={a.get('stage')}  status={a.get('status')}  created={a.get('dateCreated','')[:19]}")
    print()
