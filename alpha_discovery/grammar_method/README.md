# alpha_discovery/grammar_method — Grammar-Guided Alpha Discovery

Self-contained pipeline for discovering WorldQuant Brain alpha expressions using a
context-free grammar (CFG) with Q-table pruning over field-operator combinations.

## Table of Contents

1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Production Rules](#production-rules)
4. [Q-Table Pruning](#q-table-pruning)
5. [Tested Registry](#tested-registry)
6. [Coarse-to-Fine Budget Scheduling](#coarse-to-fine-budget-scheduling)
7. [File Structure](#file-structure)
8. [How to Run](#how-to-run)
9. [Configuration Reference](#configuration-reference)
10. [Outputs](#outputs)

## Overview

`alpha_discovery/grammar_method` generates syntactically valid Brain FASTEXPR expressions via stochastic
grammar sampling. Rather than a bandit that boosts promising clusters, it uses a Q-table
to eliminate dead `(cluster, op)` and `(field, op)` combinations — pruning the search
space as evidence accumulates.

**Design principle:** discovery only. Exploitation (refining good signals) is handled
by `tools/batch_workshop.py` and `alpha_search/pipeline.py`.

## Architecture

```
run_pipeline.py          ← unified entry point
    │
    ├─ startup ─────────── bootstrap Q-table from artifacts/sim/master.jsonl
    │                       load QTable + TestedRegistry (or rebuild from master)
    │
    ├─ build phase ──────── GrammarPlanner.build_plan(clusters)
    │       │                   BrainGrammar   (CFG sampler)
    │       │                   QTable         (prune dead cluster×op, field×op)
    │       │                   TestedRegistry (skip already-answered field×op pairs)
    │       └─ writes  →   <dataset>_batch<N>.jsonl
    │
    ├─ submit phase ──────── Brain API (submit + wait + log)
    │       │
    │       └─ writes  →   artifacts/sim/sessions/grammar_<ts>.jsonl (session log)
    │
    └─ update phase ──────── qtable.update() per result
                             tested.add() per result
                             qtable.save() + tested.save()
                             merge session log → artifacts/sim/master.jsonl
```

## Production Rules

Each rule consumes a **Δk budget**. Total budget K bounds expression depth.

**Alpha (top level):**

```
Alpha ::= group_op(Expr, group)          Δk=1
        | cs_op(Expr)                    Δk=1
        | group_backfill(Expr, group, d) Δk=2
        | group_mean(Expr, field, group) Δk=2
        | trade_when(Alpha, Expr, 0)     Δk=3
        | double_group_op(Expr, g, g)    Δk=2
        | alpha_binary(Alpha, op, Alpha) Δk=3  (budget ≥ 9 only)
```

**Expr (inner):**

```
Expr ::= field_id                        Δk=0
       | unary_op(Expr)                  Δk=1
       | binary_op(Expr, Expr)           Δk=2
       | if_else(Expr, Expr, Expr)       Δk=3
       | ts_rolling(Expr, w)             Δk=2
       | ts_delta_op(Expr, dw)           Δk=2
       | ts_paired(Expr, Expr, w)        Δk=3
       | group_op(Expr, group)           Δk=1  (inner; norm_depth ≥ 1)
       | cs_op(Expr)                     Δk=1  (inner; norm_depth ≥ 1)
```

**Budget K → reachable shapes:**

| K | Reachable shapes |
|---|-----------------|
| 3 | `group_op(ts_rolling(field, w))`, `group_op(ts_delta(field, dw))` |
| 4 | K=3 + unary field transforms + `ts_paired(A, B, w)` |
| 5 | K=4 + delta+rolling combos + binary field arithmetic |
| 6 | All shapes including nested transforms |

**Canonicalization:** commutative args (`ts_corr`, `+`, `*`) sorted alphabetically,
halving the effective search space for paired operators.

## Q-Table Pruning

The Q-table learns which `(cluster, op)` and `(field, op)` pairs are dead ends and
permanently removes them from candidate generation.

### Two levels

| Level | Key | Prunes when... |
|-------|-----|---------------|
| L1 | `(cluster_id, specific_op)` | op never works on any field in this cluster |
| L2 | `(field_id, specific_op)` | op never works on this specific field |

Both levels are updated from every simulation result with the same reward.

### Specific operators tracked

```
ts_autocorr, ts_corr, ts_zscore, ts_delta, ts_rank, ts_mean, ts_std,
ts_max, ts_min, ts_sum, group_neutralize, group_zscore, group_scale,
group_rank, trade_when, if_else, zscore, rank,
add, sub, divide, multiply
```

### Reward formula

```python
base = min(abs(is_fitness), 4.0)

# Concentration >= 40%: hard zero (fake signal, not fixable)
# Concentration 10-39.99%: 0.5x (borderline, workshop can repair)
if not concentrated_weight_ok:
    if max_concentration >= 0.40:
        return 0.0
    base *= 0.5

# Sub-universe Sharpe fail: soft penalty only
if not sub_universe_sharpe_ok:
    base *= 0.5

# Self-correlation < 0.70: no penalty
# Self-correlation ≥ 0.70: keep reward only if candidate beats the matched
# portfolio alpha by at least 10% IS Sharpe; otherwise hard zero.
if max_self_correlation is not None:
    if max_self_correlation >= 0.70:
        if max_self_correlation_alpha_sharpe is None or max_self_correlation_alpha_sharpe <= 0:
            return 0.0
        if is_sharpe < 1.10 * max_self_correlation_alpha_sharpe:
            return 0.0
```

Fitness is the primary signal — no hard zeros for low Sharpe or high turnover since
fitness already reflects signal quality.

### Prune condition

```
MIN_OBS = 8       # minimum observations before pruning is considered
DEAD_THRESHOLD = 0.05   # mean_q below this → pruned permanently
```

Once pruned, a combination is never re-activated (prevents flip-flopping).

### Bootstrap

On startup (or `--rebuild-qtable`), the Q-table is populated from
`artifacts/sim/master.jsonl` — full history across grammar, workshop, and search pipelines.
Only `status=="done"` records are included. To use the high-correlation Sharpe
exception on historical rows, enrich the log first with
`python -m tools.backfill_self_correlation`.

```python
from core.qtable import QTable, TestedRegistry, bootstrap_qtable
from tools.sim_log import MASTER

qt = QTable()
tested = TestedRegistry()
n = bootstrap_qtable(MASTER, qt, tested, populate_tested=True)
print(qt.summary(top_n=20, bottom_n=30))
```

### Workshop Routing

Workshop routing now uses a shared fitness-driven policy across the live
auto-workshop path and `tools/batch_workshop.py`. `repair` and `enhance`
fail closed when `max_self_correlation` metadata is missing, so backfill
correlation before auditing historical grammar results. Batch audit previews
also show the raw `max_self_correlation` value in a `corr` column.

## Tested Registry

Tracks `(field_id, specific_op)` pairs already submitted to Brain. Once a pair is
registered it is never re-submitted, regardless of Q-value.

- **Pruning** = combination consistently yields 0 reward → stop generating it
- **Tested** = we already have an answer for this combination → don't repeat the question

The registry prevents re-asking semantically identical questions in different expression shapes
(e.g. `rank(ts_mean(pe_ratio, 20))` vs `group_rank(ts_mean(pe_ratio, 20), sector)` both
test `pe_ratio × ts_mean`).

## Coarse-to-Fine Budget Scheduling

K=3 expressions establish field priors cheaply. Later rounds shift toward deeper K=5/6.

```
batch_1:   K=3 → 50%,  K=4 → 30%,  K=5 → 15%,  K=6 →  5%
batch_2:   K=3 → 30%,  K=4 → 30%,  K=5 → 25%,  K=6 → 15%
batch_3+:  K=3 → 10%,  K=4 → 25%,  K=5 → 35%,  K=6 → 30%
```

## File Structure

```
alpha_discovery/grammar_method/
├── __init__.py          Public API: GrammarPlanner, GrammarConfig, QTable, TestedRegistry
├── grammar.py           CFG production rules, BrainGrammar, GrammarConfig, canonicalize
├── diversity.py         tree_fingerprint, fingerprint_jaccard (diagnostic utilities only)
├── feedback.py          SimResult, compute_quality_weight
├── qtable.py            QTable, TestedRegistry, bootstrap_qtable, compute_q_reward
├── planner.py           GrammarPlanner (build_plan)
├── models.py            DataProfile, FieldCluster, ExpressionNode, CandidateSpec
├── fields.py            build_data_profiles, cluster_profiles, filter_profiles
├── wq_api.py            sign_in, get_datafields (Brain API helpers)
│
└── run_pipeline.py      ← Unified entry point (build + submit + update Q-table)

artifacts/
├── all_clusters.json    Cached field clusters (fetched once, --refetch-clusters to rebuild)
├── all_qtable.json      Q-table state (updated each round)
└── all_tested.json      Tested (field, op) registry (updated each round)

artifacts/sim/
├── master.jsonl                     Permanent deduped history across all pipelines
└── sessions/grammar_<ts>.jsonl      Per-run session log
```

## How to Run

All commands are run from `world_quant/` (the project root).

```bash
# Multi-dataset mode (recommended)
python -m alpha_discovery/grammar_method.run_pipeline \
    --fields-csv all_fields.csv \
    --exclude-dataset analyst4 \
    --n-samples 300 \
    --credentials brain2.txt

# Single dataset
python -m alpha_discovery/grammar_method.run_pipeline \
    --dataset-id fundamental6 \
    --category fundamental \
    --n-samples 200

# Rebuild Q-table from master history on next startup
python -m alpha_discovery/grammar_method.run_pipeline \
    --fields-csv all_fields.csv \
    --rebuild-qtable \
    --n-samples 300

# Inspect Q-table without running a pipeline round
python -c "
from core.qtable import QTable, TestedRegistry, bootstrap_qtable
from tools.sim_log import MASTER
qt = QTable(); tested = TestedRegistry()
n = bootstrap_qtable(MASTER, qt, tested)
print(f'Processed {n} records')
print(qt.summary(top_n=20, bottom_n=30))
"
```

## Configuration Reference

All knobs are in `GrammarConfig` (`grammar.py`).

| Parameter | Default | Description |
|-----------|---------|-------------|
| `budget` | `6` | Max Δk budget K for expression depth |
| `n_samples` | `500` | Candidates to generate per batch |
| `min_coverage` | `0.45` | Min champion.coverage to qualify a cluster |
| `windows` | `(20,40,60,120,200,252)` | Rolling lookback windows (days) |
| `delta_windows` | `(5,10,20)` | Momentum/lag windows (days) |
| `groups` | `("market","industry","subindustry")` | Cross-sectional neutralisation groups |

Q-table pruning constants (in `qtable.py`):

| Constant | Default | Description |
|----------|---------|-------------|
| `MIN_OBS` | `8` | Minimum observations before pruning considered |
| `DEAD_THRESHOLD` | `0.05` | Mean Q below this → prune |

**`min_coverage` tradeoff:**

| Setting | Effect |
|---------|--------|
| `0.45` (default) | Broad-market fields only. Low noise, fast convergence. |
| `0.0` | All fields including event-driven and sector-specific. Use for news, social media, option datasets. |

## Outputs

| File | Contents |
|------|----------|
| `artifacts/sim/sessions/grammar_<ts>.jsonl` | This run's simulation records |
| `artifacts/sim/master.jsonl` | Full deduped history, all pipelines |
| `artifacts/all_clusters.json` | Cached cluster definitions |
| `artifacts/all_qtable.json` | Q-table with running sums and observation counts |
| `artifacts/all_tested.json` | Tested (field, op) pairs registry |
| `artifacts/all_runner.log` | Submit/wait progress text log |

**Resuming:** `run_pipeline.py` skips candidates whose `candidate_id` or `expression`
already appears in the session log. Safe to interrupt and re-run.
