# Deduplication / sim-budget plan

**Status:** design only — not implemented. Supersedes the implicit "exact-fingerprint
dedup + coarse Q-table pruning" we run today.

## The problem, measured (not assumed)

Run over `artifacts/sim/master.jsonl` (84,717 records with usable expression+fields),
keying each sim to a `(operator-skeleton, field-set)` cell where the skeleton has
windows/decay constants normalized out:

| metric | value | reading |
|---|---|---|
| distinct cells | 75,842 | avg **1.12 sims/cell** |
| cells simmed >1× | 7.9% | we barely re-sim the same thing |
| cells, best sharpe < 1.25 | **92.3%** | almost everything we try is weak |
| cells with a submittable winner | 5.8% | the live region is tiny |
| **sims spent in weak cells** | **90.5%** | the budget bleed |

**Conclusion that redirects the whole effort:** the waste is *not* duplication.
Identity-dedup (what we have) already catches the literal repeats; only 7.9% of cells
recur. **90% of sim budget is spent re-exploring structure×field neighborhoods the
history already proved are dead.** So the lever is not a stricter blocklist — it is a
*predictive* prune that reads the 85k as a map of dead vs. live regions.

A second, separate problem (post-sim): among the ~5.8% live cells, the robust
submittables are dominated by **one idea** — `anl4_adjusted_netincome_ft` smoothed via
`kth_element`/`group_backfill`, repeated as window-sweep variants. Strong but mutually
correlated → they don't add competition score. Identity/structure dedup cannot see this;
it is a PnL-correlation problem and only shows up after simulation.

These are two different gates at two different stages. Keep them separate.

## Two-stage funnel

```
candidate ─▶ [Stage 1: pre-sim structural prune]  ─▶ sim ─▶ [Stage 2: post-sim PnL gate] ─▶ promote
             cheap, no sim, reads 85k history              needs PnL, reads portfolio
```

### Stage 1 — pre-sim neighbor-outcome prune (reclaims the 90%)

Not a blocklist. A k-NN over the simulated population that asks *"do my structural
neighbors tell me this region is dead?"* before spending a sim.

**Signature.** For each historical sim and each new candidate, compute:
- `skeleton` = expression with field ids → `F`, numeric constants (windows/decay/k) → `N`,
  whitespace stripped. (Reuse `derive_fields` + the regex from the measurement script.)
- `field_set` = `frozenset(derive_fields(expr))`.
- `cluster_set` = clusters those fields map to (via `all_clusters.json`).

**Index.** Build once from master (+ records/* submitted): map each cell signature to its
outcome summary `{n, best_sharpe, any_submittable, in_portfolio}`. Persist to
`artifacts/theory/sim_index.jsonl` (or a parquet/npz sidecar); refresh incrementally on
merge, same hook as the Q-table update.

**Decision for a new candidate** (cheapest check first):
1. Exact/near-exact cell hit (same skeleton + field_set) →
   - best_sharpe < 1.25 → **SKIP** (known dud, the 90% case).
   - already in portfolio → **SKIP** (redundant).
   - else → it's a window-sweep of a known winner: **keep at most one representative**.
2. No exact hit → look at k nearest neighbors (same skeleton, sibling field; or same
   field_set, neighbor skeleton):
   - all neighbors weak (best < 1.0) → **SKIP** (dead neighborhood).
   - neighbors sparse / none → **SIM** (genuinely unexplored — where new score lives).
3. Bias what survives toward **low-density** regions of the signature space (few neighbors)
   — that is the operational meaning of "more unique," and it tracks novelty far better
   than expression-string uniqueness.

**Relation to existing code.** This is a strict generalization of `GrammarCoverage`
(`wq_alpha/theory/coverage.py`), which already prunes dead `(field, op)` pairs off the
Q-table. Stage 1 takes the same idea from coarse `(field, op)` to fine
`(skeleton, field_set)` neighborhoods over the raw 85k. Prefer extending `coverage.py`
over a parallel module.

**Distance.** Start cheap and exact-ish: skeleton-equality + field-set Jaccard +
cluster overlap. Only escalate to an expression embedding + ANN if the cheap signature
proves too brittle (it collapses window-sweeps well, which is the main mass). Do **not**
build the embedding index first — measure whether the signature suffices.

### Stage 2 — post-sim PnL-correlation / score-add gate (the redundancy the structure can't see)

Unchanged in spirit from what we discussed; this is the only place correlation is real,
because it needs the candidate's simulated PnL.

- After a candidate sims as submittable+robust, pull its PnL (1 API call) and run
  `local_admission.score_locally`: `max_self_correlation` vs the held portfolio +
  `delta_pool_sharpe`.
- Gate: corr ≥ 0.70 → **block** (redundant — this is the `anl4_adjusted_netincome_ft`
  cluster's failure mode and theory_0268's −161 score-delta). 0.50–0.70 → **deprioritize**.
  corr ≤ −0.70 → **flag sign-flipped dupe**.
- For the brain (competition) pool, prefer the *actual* `brain_score_delta` when we re-sim
  there (via `promote_to_brain`), not the proxy.

## What explicitly NOT to do

- Do **not** tighten identity-dedup expecting throughput — the data says it buys ≤8%.
- Do **not** treat expression-string uniqueness as the diversity metric — it does not
  track PnL correlation (different structure on the same factor stays correlated; same
  structure on a different field can decorrelate). Stage 1 density + Stage 2 PnL are the
  real metrics.
- Do **not** build the embedding/ANN index before proving the cheap signature is
  insufficient.

## Build order (when we implement)

1. Extract the signature helper (`skeleton(expr, fields)`) into a shared module — it's
   already prototyped in the measurement script.
2. Build `sim_index` from master + records; wire incremental refresh into the merge hook.
3. Extend `coverage.py` with the neighbor-outcome query; call it in `enumerate_family`'s
   variant filter (it already takes `coverage=`).
4. Keep Stage 2 where it is (`local_admission` / `promote_to_brain`); just make sure every
   promoted candidate passes through it.
5. Re-run the 90%-waste measurement after Stage 1 is live — success = sims-in-weak-cells
   drops well below 90%, submittable-yield-per-sim rises.

## Open questions

- k and the "dead neighborhood" thresholds (best<1.0? <0.5?) need calibration against a
  held-out slice of master — pick them so Stage 1 would not have pruned the 5.8% winners.
- Does cluster-overlap distance add anything over field-set Jaccard, given fields already
  map 1:1 to clusters? Measure before keeping both.
