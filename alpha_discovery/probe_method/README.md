# alpha_discovery/probe_method

`alpha_discovery/probe_method` builds fast field-level signal probe plans. It is the bridge
between pure grammar exploration and disciplined seed discovery.

Default first pass:

```bash
python -m alpha_discovery/probe_method.build_plan \
  --clusters alpha_discovery/grammar_method/artifacts/all_clusters.json \
  --out artifacts/probe/probe_plan.jsonl \
  --pack core \
  --cluster-limit 400
```

Submit with the existing Brain submitter:

```bash
python -m core/brain.submit \
  --plan artifacts/probe/probe_plan.jsonl \
  --sim-log artifacts/probe/probe_sim.jsonl \
  --batch-size 3
```

After simulations finish, summarize probe hits:

```bash
python -m alpha_discovery/probe_method.analyze_results \
  --sim-log artifacts/probe/probe_sim.jsonl \
  --out-summary artifacts/probe/probe_summary.json \
  --out-seeds artifacts/probe/probe_seeds.jsonl
```

## Probe Expressions

The `core` pack iterates one representative field through:

- `rank(x)` and `rank(-x)`
- `group_rank(x, industry)` and `group_rank(-x, industry)`
- `group_zscore(x, sector)`
- `group_rank(ts_rank(x, 60), industry)`
- `group_rank(-ts_rank(x, 120), industry)`
- `group_zscore(ts_zscore(x, 120), sector)`
- `group_rank(-ts_delta(x, 5), industry)`
- `group_rank(-ts_delta(x, 20), industry)`
- `group_rank(-((x) - ts_mean(x, 60)), industry)`
- `group_rank(ts_mean(x, 20), industry)`

Use `--pack broad` after the first pass for clipped, scaled, stability, and
decayed variants. Use `--pack event` only for targeted event/staleness probing.

## Cluster Strategy

Default `--representatives champion` means one field per cluster. This is the
right first pass because it maximizes breadth per simulation. Use
`--representatives champion_sleeper` for clusters that light up, and
`--representatives all --max-fields-per-cluster N` only for focused follow-up.

## A-Derived Enhancement

After probes produce seed alphas, use deterministic A-derived variants before
or alongside LLM workshop. This separates signal enhancement from raw mining:

```bash
python -m alpha_discovery/probe_method.enhance_plan build \
  --input artifacts/probe/probe_seeds.jsonl artifacts/workshop/workshop_submittable.jsonl \
  --out artifacts/enhance/self_enhance_plan.jsonl \
  --pack all \
  --max-alphas 50
```

Submit and summarize:

```bash
python -m core/brain.submit \
  --plan artifacts/enhance/self_enhance_plan.jsonl \
  --sim-log artifacts/enhance/self_enhance_sim.jsonl \
  --batch-size 3

python -m alpha_discovery/probe_method.enhance_plan summary \
  --sim-log artifacts/enhance/self_enhance_sim.jsonl \
  --out-txt artifacts/enhance/self_enhance_summary.txt
```

The `robustness_check` variants include `rank(A)`, `sign(A)`, and
`power(rank(A), 2)` retention tests. The `self_enhance` variants include
industry/sector remaps, cap and IR neutralization, smoothing/decay/hump, and
self-strength gates.

For settings robustness, use the existing regime-check lane. It now supports an
optional delay grid:

```bash
python -m tools.regime_check build \
  --input artifacts/workshop/workshop_submittable.jsonl \
  --output artifacts/regime_check/plan.jsonl \
  --promote-only \
  --mode one-axis \
  --delays 0,1 \
  --universes TOP3000,TOP2000,TOP1000
```
