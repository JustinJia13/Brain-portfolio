# ASSUMPTIONS.md — the register tested by `VERIFICATION_PLAN.md`

Each assumption the `alpha_discovery/sim_slot_gate` sim-slot gate rests on, with an id the plan
tests. **Status** ∈ {UNTESTED, PASS, PARTIAL, FAIL, UNVERIFIED(Case B)}.
Append results; never delete. Dates are absolute.

**Data case (Stage 0):** **Case B-plus.** No PnL vectors for the 85k corpus; only
the 31 submitted-pool alphas have daily PnL. Self-correlation labels exist
(`max_self_correlation` + matched id, ~3.5k) and partially validate Object A.
**Time-orderable:** yes (`logged_at`), span 2026-03-26 → 06-18, batched (Mar 8.6k /
Apr 10.9k / May 41 / Jun 64.4k). NOTE (corrected 2026-06-19): the master log
surfaced only 303 sub-universe-Sharpe values (March), but **~78.6k are on disk in
the raw detail files** (`LOW_SUB_UNIVERSE_SHARPE` check), spread Mar/Apr/Jun — so
sub-Sharpe IS time-splittable and Task 3 ran for free (no sims/auth).

---

## Object A — analytic redundancy

| id | assumption | critical | status |
|----|------------|----------|--------|
| A1 | structural `corr_model` maps monotonically to realised PnL-corr via `psi` | ★ | **PASS** 2026-06-19 (time-split Spearman 0.50; high-regime ≥0.85→0.66) |
| A2 | per-primitive behavioural similarity is the right unit of redundancy | | PARTIAL (jointly via A1) |
| A3 | the four channels combine **multiplicatively** with weak interactions | ★ | PARTIAL — jointly validated via A1 (assembled corr_model tracks realised); isolation blocked by Case B (no PnL for controlled pairs) |
| A4 | field similarity `<v_f,v_g>` (incl. **metadata-only unseen fields**) tracks realised field corr | ★ | PARTIAL — embedding sane (assets↔liab 0.90, vs returns −0.12); realised-corr isolation blocked by Case B |
| A5 | linear TS ops/windows → spectral `Omega_time` predicts realised corr | | PASS (spectral sanity pts) / PARTIAL (realised via A1) |
| A5b| nonlinear TS templates (`ts_zscore/std_dev/corr/...`) predict realised corr | | PARTIAL (jointly via A1) |
| A6 | CS-warp table (rank≈zscore≈…, winsorize≈0.98) matches realised | | PARTIAL (jointly via A1) |
| A6b| `kappa_g = √(1−R²_group)` tracks realised neutralisation attenuation | | UNVERIFIED (Case B) — prior 0.84 used |
| A7 | canonicalization preserves behaviour (no false collapse of distinct alphas) | ★ | **PASS** 2026-06-19 (3918 correct collapses/15k, 0 false in audit) |
| A8 | `sigma_corr` (psi residual spread) gives a usable redundancy interval | | PASS 2026-06-19 (out-of-time sigma_corr=0.239) |

## Object B — submission-component surrogate

| id | assumption | critical | status |
|----|------------|----------|--------|
| B1 | **structure `phi(X)` predicts quality on FUTURE sims at all** (HARD GATE) | ★ | **PASS** 2026-06-19 |
| B2 | the chosen `phi` features are sufficient / non-redundant | | **PARTIAL/actionable** 2026-06-19 — sufficient but OVER-specified: dropping path-n-grams +0.06 IC, op-hash +0.02 (hashed feats overfit transfer). **Prune → IC 0.29→0.35** |
| B3 | the `phi→quality` map transfers across time (no fatal decay) | ★ | **PASS** 2026-06-19 (is_sharpe 0.33; **sub-Sharpe 0.29** 70/30, 0.21 strict — Task 3) |
| B4 | it transfers across the search distribution (operator/field mix) | | **PASS** 2026-06-19 |
| B5 | quantile heads are **calibrated** out-of-time (coverage ≈ nominal) | ★ | **PASS** 2026-06-19 (mean\|err\|≈0.035; mild upper-tail under-cover) |
| B6 | fitness is better **reconstructed** than given a direct head | | **PASS** 2026-06-19 (recon IC 0.317 > direct 0.299 — keep reconstruction) |
| B7 | sub-universe Sharpe is the most-often binding submission component | | **PARTIAL/corrected** — Sharpe(300)>fitness(267)>sub-Sharpe(166)≫turn(15); concentrated-weight(30) → route-to-simulate. Sub-Sharpe head now trained on ~78k labels (was 303) & transfer-validated |
| B8 | the `n_eff` coverage branch reduces missed-good in novel regions for its cost | | **PARTIAL** 2026-06-19 — ~0 value on historical replay (corpus self-covering) but near-free + rescues genuinely-novel; KEEP as live safety-net, re-eval Task 4 |

## Gate

| id | assumption | critical | status |
|----|------------|----------|--------|
| G1 | turnover + redundancy are reliable enough for **no-eps hard kills** | ★ | **PASS** 2026-06-19 (turnover IC 0.71, precision@cap 0.89) |
| G2 | a safety margin inside the BRAIN bars keeps misses bounded | | PARTIAL — margin via β UCB (β=2.5 default); explicit bar-margin not separately swept |
| G3 | **end-to-end the gate saves meaningful slots at acceptable miss rate** | ★ | **PASS** 2026-06-19 — post-prune time-split: β=2.5 → 46% saved @ 0.91 qz-recall; floor verified. **vs incumbent q-table (inert, <1% pruned): challenger dominates, McNemar χ²=1816** |

---

## Graceful Degradation Map (ship mode implied by status)

- **B1 FAIL** → do **not** ship Object B. Ship **reliable-kills-only**: turnover
  hard-kill + redundancy `c_max` + ε trickle (+ coverage abstention). Legitimate
  working outcome.
- **B5 FAIL (uncalibrated)** → withdraw the confident-skip branch; keep coverage +
  ε only. Effectively reliable-kills-only on the quality axis.
- **A1 FAIL (no monotone psi)** → redundancy falls back to crude exact-structure /
  field-set dedup; lean on default-to-simulate.
- **A4 unseen FAIL** → confident redundancy restricted to seen fields; novel-field
  candidates routed to simulate.
- **G1 FAIL (turnover)** → demote turnover from hard-kill to soft (ε-rescued UCB).
- **B8 no value** → drop coverage branch + `d_red` density (simplification fork);
  A then needed only for `c_max`.
- **G3 bad frontier everywhere** → ship only highest-precision kills (turnover cap +
  exact dedup); simulate the rest.

**Current ship mode (2026-06-19): FULL GATE at β=2.5 — in SHADOW; q-table still
authoritative for live keep/skip until Task 4 confirms.** B1 passed the hard gate,
B5 is calibrated out-of-time, φ pruned + retrained (IC 0.33), and the champion-
challenger shows the incumbent q-table is **effectively inert (<1% pruned, verified
on the real artifact)** while the challenger dominates (McNemar χ²=1816). Reliable-
kills-only is the automatic fallback.

**Cutover conditions (PRE_DEPLOYMENT_VALIDATION.md):**
1. Post-prune frontier on the table — ✅ DONE.
2. Task 2 favours challenger — ✅ MET decisively.
3. Live shadow confirms (Task 4) — ⏳ **trigger BUILT** (`alpha_discovery/sim_slot_gate/cutover.py`:
   3-condition monitor w/ power; recall≥0.88, coverage|err|≤0.05, novel<10%; auto-
   rollback). Gate modes wired (`--gate-mode shadow|reliable|full|auto`). Dry-run on
   synthetic in-corpus data validates plumbing (optimistic by construction). Needs
   genuine LIVE accumulation. §1 safety: cut over at **β=3.0** (err toward simulate).

**Task 3 (sub-Sharpe transfer) — ✅ RESOLVED for free** (78k labels mined from detail
files; IC 0.29 out-of-time; head retrained on them). The sub-Sharpe gate is now
trustworthy. Only Task 4 (live shadow) remains before full cutover.

**Adopt now regardless (low-risk):** behavioural-dedup `c_max` (upgrades the
incumbent's crude exact-fingerprint dedup) + turnover hard-kill.
