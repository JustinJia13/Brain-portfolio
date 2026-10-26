# CALIBRATION_REPORT.md

**Date:** 2026-06-19 · **Data case:** Case B-plus · **Ship mode:** **FULL GATE (β=2.5)**

The dated answer to "does the `alpha_metric` sim-slot gate work on the data we have
*now*." All tests use already-run sims (zero new sim budget) and a **time split**
(train older, test newer) wherever a label allows — this is what makes the verdict
honest and stresses transfer/decay. Regenerate with `alpha_metric/verify.py`.

## Data (Stage 0)
- Sim store `artifacts/sim/master.jsonl`: 85,221 rows; 82,136 with timestamp +
  `is_sharpe`. Span 2026-03-26 → 06-18, **batched** (Mar 8.6k / Apr 10.9k / May 41
  / Jun 64.4k).
- **Case B-plus:** no PnL vectors for the corpus; only the 31 submitted-pool alphas
  have daily PnL. ~3.7k self-correlation triples (`max_self_correlation` + matched
  id) partially validate Object A.
- **Sub-universe Sharpe: 303 labels, all in March 26–28** → cannot be time-split.
  Feasibility/coverage run on `is_sharpe`; the sub-Sharpe head is transfer-UNVERIFIED.

## Feasibility verdict (T-B1, the HARD GATE) — **PASS**
Median head `phi → is_sharpe`, trained on older sims, tested on newer:

| split | rank-IC | turnover-only baseline IC | decile lift (top-decile pass / base) |
|---|---|---|---|
| 70/30 time | **0.296** | −0.118 | **4.77×** (36.0% / 7.5%) |
| strict pre-June → June | **0.280** | −0.100 | **5.48×** (41.6% / 7.6%) |

Structure predicts *future* quality, beats baselines, and transfers across the
biggest regime shift in the data. → **Object B is justified; build it.**

## Object A — redundancy
- **psi (T-A1):** time-split Spearman(corr_model, realized self-corr) = **0.50**;
  psi improves MSE over identity; out-of-time **`sigma_corr = 0.239`**. High regime
  (the kill's regime): corr_model ≥0.5 → 0.58, ≥0.7 → 0.62, ≥0.85 → **0.66**.
  → trustworthy as a high-threshold duplicate kill, not a low-range predictor.
- **Canonicalization (T-A7):** identity preserved; 3,918 correct behavioural
  collapses in 15k audited, **0 false collapses**.
- **Channels (T-A3/4/5/6):** jointly validated via psi (assembled corr_model tracks
  realized at 0.50); per-channel isolation against realized corr is **blocked by
  Case B** (no PnL for controlled pairs). `kappa_g`=0.84 prior UNVERIFIED.

## Object B — surrogate
- **Out-of-time coverage (T-B5) — PASS:** sharpe/ret/turnover within mean|err| ≈
  0.035 of nominal across levels; mild upper-tail under-coverage (~0.04) → quality
  UCB slightly optimistic (widen if desired).
- **Fitness (T-B6):** reconstruction beats a direct head out-of-time (IC 0.317 vs
  0.299) → **keep reconstruction** (design validated).
- **Binding component (T-B7):** Sharpe (300) > fitness (267) > sub-Sharpe (166) ≫
  turnover (15); concentrated-weight (30) is **unmodeled** (low frequency).
- **Feature sufficiency (T-B2) — actionable:** φ is sufficient but **over-specified**
  — dropping path-n-grams **+0.06 IC**, op-hash +0.02 (hashed feats overfit
  transfer). Pruning lifts IC 0.29 → 0.35. *(Top follow-up.)*
- **Sub-Sharpe head:** trained on 303 March-only labels; transfer UNVERIFIED.

## Gate
- **Turnover reliable kill (T-G1) — PASS:** out-of-time IC 0.71; precision-at-cap
  **0.89** at the optimistic edge → justifies the no-ε hard kill.
- **End-to-end budget backtest (T-G3, TIME-split, n=6000, base rate good 7.6%) —
  PASS:**

| β | slots saved | good recall | quality-zone recall | enrichment |
|---|---|---|---|---|
| 1.5 | 83% | 43% | 53% | 3.1× |
| 2.0 | 71% | 54% | 67% | 2.2× |
| **2.5 (default)** | **45%** | **67%** | **85%** | **1.5×** |
| 3.0 | 20% | 74% | 93% | 1.1× |

  Reliable-kills-only floor: 5% saved @ 79% recall. **Robustness floor verified**
  (`good_killed_quality = 0` with B disabled). A *random*-split backtest had
  overstated recall (0.91 at β=2.0) — temporal leakage; the time split is honest and
  the default β was raised 2.0 → 2.5 accordingly.

## Operating decision
- **Ship FULL GATE at β=2.5** (save ~45% of slots, keep ~85% of good in-zone). Move
  to β=3.0 if missing good alphas is costlier than the slots.
- Reliable-kills-only is the automatic fallback (gate degrades to it if the
  surrogate is absent/untrained).
- This verdict can drift as the sim set / search distribution evolve — re-run
  `verify.py` periodically and **retrain respecting time order**.

## Open items
1. **Prune hashed φ features** (B2) → +0.06 IC, then retrain + re-run G3. *(DONE — see Pre-deployment below)*
2. Retrain the deployed surrogate respecting time order. *(DONE)*
3. Sub-Sharpe transfer is unverifiable until BRAIN sub-universe Sharpe is recorded
   on newer sims; consider re-enabling that capture. *(OPEN — Task 3)*
4. Concentrated-weight is an unmodeled (rare) submission failure mode. *(decided: route-to-simulate)*
5. Per-channel A validation needs PnL vectors (Case A) — currently joint-only.

---

# Pre-deployment validation — 2026-06-19 (champion-challenger)

## Task 1 — prune φ, retrain in time order  (DONE)
B2 ablation acted on: dropped the hashed groups (path-n-grams + op-hash); φ = scalars
+ field-category fractions (196→**36 dims**). Time-split B1 rank-IC **0.296→0.332**.
Post-prune out-of-time coverage within ~0.05 (mild Sharpe upper-tail under-cover).
**Post-prune G3 frontier (authoritative):**

| β | slots saved | good recall | quality-zone recall | enrichment |
|---|---|---|---|---|
| 2.0 | 70% | 60% | 76% | 2.4× |
| **2.5 (default)** | **46%** | **72%** | **0.91** | 1.6× |
| 3.0 | 23% | 75% | 0.95 | 1.2× |

Pruning *improved* quality-zone recall (0.85→0.91 at β=2.5). Deployed `surrogate.pkl`
+ `coverage_index.npz` retrained on pruned φ.

## Task 2 — champion (q-table) vs challenger (quality surrogate), out-of-time  (DONE)
Scoring set = simulated alphas only (n=23,335 test, base good 7.4%); both policies
built point-in-time on the older 70%.

- **The incumbent q-table quality pruner is effectively inert.** Verified on the
  *real production artifact* `all_qtable.json`: of ~22,600 buckets, **~9 prune** at
  the production threshold (0.05). Reconstruction agrees: **0% slots saved at θ=0.05**,
  rising only to ~9% even at an aggressive θ=0.5 (recall 0.99). This is exactly the
  `dedup.md` failure mode — the q-table never catches the dead regions.
- **The challenger offers a real frontier** the q-table cannot reach: e.g. **38%
  saved @ 92% recall**, 66% @ 77%. At matched ultra-high recall (~99%) the two are
  comparable (~8% saved); the challenger's value is *extending* the frontier.
- **Disagreement cells (decisive):** at the challenger's β=2.5, among champion≠
  challenger decisions the challenger correctly **skips 2,790 bad** alphas vs the
  champion's 336, and the two keep a similar handful of good ones. **McNemar χ² =
  1816 (p ≈ 0)** in the challenger's favour.
- **2e mechanism tests inconclusive** — too few qualifying test alphas in high-
  variance / sparse buckets (the q-table prunes so little there's nothing to compare);
  moot given the incumbent is inert.

**Dedup correction (Task 2a).** The q-learning bucket pruner has no per-candidate
dedup, but the surrounding system *does* have crude **exact-fingerprint dedup**
(`TestedFingerprintRegistry`) + skeleton-novelty down-weighting. So `c_max` is not a
first-ever dedup — it **upgrades exact-structure dedup to behavioural near-duplicate
dedup** (catches twins that differ textually but behave identically; T-A7 found 3,918
such collapses per 15k). Adopt `c_max` as an additive behavioural-dedup kill.

## Task 5 — cleanups  (DONE)
- **A6b `kappa_g`:** the 31 pool alphas have no neutralized/un-neutralized
  counterparts; measuring `corr(f, neut(f))` needs new sims. **Keep 0.84 prior,
  UNVERIFIED.**
- **Concentrated-weight (B7):** only 30/85,450 labels — too sparse to model.
  **Decision: route concentrated-weight-risk candidates to simulate** (accept as a
  known, rare gap; the gate defaults to simulate anyway).
- **B8 coverage branch:** on historical replay it rescues ~0 missed-good (the corpus
  is self-covering, so almost nothing is low-density); it *does* correctly route a
  genuinely-novel exotic candidate to simulate (n_eff=0). It's near-free. **Keep as a
  production novelty safety-net; re-evaluate on live data (Task 4).**

## Task 3 — sub-Sharpe temporal transfer  (DONE, for free)
The "all 303 sub-Sharpe labels in March" blocker was a master-extraction artifact:
**~78,629 sub-Sharpe values are on disk** in the raw detail files
(`LOW_SUB_UNIVERSE_SHARPE` check), spread Mar/Apr/Jun — no sims or auth needed.
Time-split on sub-Sharpe: rank-IC **0.291** (70/30, vs is_sharpe 0.33), coverage
mean|err| 0.033; strict pre-June→June IC 0.207. **PASS — sub-Sharpe transfers.** The
deployed sub-Sharpe head was retrained on ~78k labels (was 303); held-out coverage
near-nominal. `db_adapter` now backfills sub-Sharpe from the detail cache.

## Cutover decision (state of the three conditions)
1. **Post-prune frontier is on the table** — ✅ DONE (Task 1).
2. **Task 2 favours the challenger** — ✅ MET, decisively (incumbent inert; challenger
   frontier dominates; McNemar χ²=1816).
3. **Live shadow confirmation** — ⏳ NOT YET (Task 4 wired: `run_pipeline --shadow-gate`
   now logs both the gate and the q-table per candidate; needs live accumulation).

Task 3 (sub-Sharpe transfer) is now ✅ RESOLVED — the binding-est head is
transfer-validated and retrained on 78k labels. **Only condition 3 (live shadow)
remains** before the full quality-surrogate cutover.

**Recommendation:**
- **Adopt now (low-risk, no head-to-head needed):** the behavioural-dedup `c_max`
  kill + turnover hard-kill, on top of / replacing the inert q-table. Strict improvement.
- **Quality surrogate switch:** historical evidence strongly favours it (condition 2
  met). Gate the *full* cutover on **Task 4** (live shadow) per the plan, and on
  **Task 3** (sub-Sharpe transfer) before leaning on the sub-Sharpe head — until then
  rely on the fitness/main-Sharpe reconstruction (transfer-proven) as the quality gate.
- **Ship mode now:** **FULL GATE at β=2.5 in shadow**, q-table still authoritative for
  live keep/skip, until Task 4 confirms. Default β raised 2.0→2.5 (honest frontier).

---

# Shadow→cutover trigger — 2026-06-19 (condition 3 machinery)

Built `alpha_metric/cutover.py` (monitor + ship-mode state + rollback) and wired
gate modes into the pipeline. Conditions 1–2 already met offline; this is how
condition 3 (live) is decided and guarded.

**§1 safety decision:** keep `sigma_inflate=1.0` but **cut over at β=3.0** (G3:
0.95 in-zone recall, 23% saved) rather than 2.5 — err toward simulate while proving
live, since B5 showed ~0.04 upper-tail under-coverage. Both β are evaluable post-hoc
from the raw logs, so this is reversible.

**Three live conditions (monitor, with dynamic power = achieved CI half-width):**
- Cond1 quality-zone recall ≥ 0.88 (target h≤0.04 ⇒ ~197 in-zone-good at p≈0.91) — PRIMARY.
- Cond2 calibration coverage mean|err| ≤ 0.05 — **HOLD on fail** (drift = unsafe skips).
- Cond3 novelty rate < 10% — **HOLD/retrain on fail** (corpus no longer covers live).
Cutover requires all three PASS *and* powered; a rolling kill-switch auto-reverts
full→reliable on breach (recall<0.85 | cov_err>0.07 | novel>20%).

**Ship-mode state machine** (`cutover_state.json`, default `shadow`):
`shadow` (log only) → `reliable` (dedup c_max + turnover hard-kill LIVE = adopt-now)
→ `full` (quality surrogate authoritative). `--gate-mode auto` reads the state so the
rollback self-heals without a code change. All modes fail-safe → simulate on error.

**Monitor dry-run (synthetic, historical-as-live — validates plumbing only,
optimistic by construction):** Cond1 recall 0.993 (powered), Cond2 |err| 0.010
(powered), Cond3 novel 0.002 (powered) → "RECOMMEND CUTOVER". On genuine live data
this will be honest; do not read the synthetic numbers as the live verdict.

## Runnable commands (this repo)
```
PY=${PYTHON:-python3}

# (A) ADOPT-NOW — dedup c_max + turnover hard-kill go LIVE (not gated by cutover):
$PY -c "from alpha_metric.cutover import save_state; save_state('reliable',3.0,'adopt-now dedup+turnover')"
#     then run the live pipeline with the gate authority read from state:
$PY -m alpha_grammar.run_pipeline --fields-csv all_fields.csv --rounds 4 --n-samples 390 \
    --budget 10 --batch-size 3 --max-candidates 0 --refetch-clusters --delay 1 \
    --credentials brain2.txt --gate-mode auto

# (B) SHADOW ACCUMULATE (gather live labels; reliable mode also adopts-now + logs raw):
#     (same run command as above; logs land in artifacts/gate_shadow.jsonl)

# (C) CUTOVER MONITOR — 3-condition status + power + recommendation:
$PY -m alpha_metric.cutover --beta 3.0
$PY -m alpha_metric.cutover --beta 2.5      # compare operating points on the same logs

# (D) CUTOVER FLIP — make the quality surrogate authoritative (ONLY flips if all
#     conditions PASS & powered; otherwise prints HOLD/KEEP-ACCUMULATING):
$PY -m alpha_metric.cutover --beta 3.0 --apply
#     (live runs already on --gate-mode auto pick up 'full' on the next round)

# (E) ROLLBACK kill-switch — run each round / periodically (auto-reverts full->reliable):
$PY -m alpha_metric.cutover rollback
```
