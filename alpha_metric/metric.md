# `alpha_metric` — the sim-slot gate, for a quant teammate

*Audience: you, in six months, extending this. Plain language, motivated formulas.*

## 1. The job

A random-search generator emits candidate alphas as FastExpr strings. The scarce
resource is **BRAIN simulation slots** (full IS backtests) — there is no cheaper
partial-sim probe. Most candidates are not worth a slot: they would either **fail
submission** or **add nothing to the pool**. This package is a **sim-free
pre-screen** that decides, per candidate, **simulate or skip**.

The predictions it makes internally are a **throwaway filter**. The kept/killed
decision is the *only* deliverable. We never report the predictions as alpha
quality and never optimise an alpha toward them. When in doubt the gate
**simulates** — skipping a good alpha is the expensive error; spending a slot on a
bad one is cheap.

This replaces the old Q-table / skeleton-counter / tested-fingerprint pruning
(`alpha_grammar/qtable.py`), which bucketed by `(field, op)` and (per `dedup.md`)
let ~90% of sims land in already-dead regions.

## 2. The two objects, and exactly where each enters the gate

| | Object A — redundancy | Object B — submission surrogate |
|---|---|---|
| what | analytic distance `d_red(X,Y)` | fitted quantile regression |
| nature | closed-form, ~tens of calibrated scalars; **not** a fitted `(X,Y)→d` model | sklearn quantile GBM on structure `phi(X)` |
| enters gate as | (i) pool redundancy `c_max`; (ii) — | predicted submission components |
| coverage `n_eff` | (see §7 — moved to phi-space) | |

They are **independent**: A does not feed B, B does not use A's distance. There is
**no KNN regression of Sharpe** anywhere (§4).

## 3. Object A — the analytic redundancy distance

An alpha is parsed (`parse.py` — a hand-written recursive-descent FastExpr parser,
since the repo only had a tree *generator*; handles infix, kwargs, multi-statement
variable bindings, ternaries; 99.6% corpus coverage) and decomposed under top-level
± into **additive primitives** (`signature.py`). Each primitive carries four
behavioural channels:

- **field** (`fields.py`): an open-vocabulary embedding `v_f ∈ R^48` seeded from
  metadata (dataset / category / subcategory / units / hashed description tokens),
  unit-normalised; `rho_field = <v_f, v_g>`. Never builds the 10^4 × 10^4 matrix;
  set-to-set similarity is symmetric best-match. A never-seen field is usable
  immediately (it folds in the field-id's own tokens), so engineered `*_derivative`
  fields still embed sensibly.
- **temporal filter** (`filters.py` + `channels.py`): every TS operator is a filter
  with a magnitude response `|H(ω)|`; a chain's `|H| = Π |H_i|`. Similarity is the
  **ω-weighted power-spectrum overlap** `Omega_time = <p1,p2>/(||p1|| ||p2||)`,
  `p_i = |H_i|² √ω`. **Deviation from the spec, on purpose** (see §9).
- **CS warp** (`channels.omega_cs`): rank/zscore/normalize/scale are mutually
  monotone ≈1; winsorize ≈0.98 vs rank; bucket/sign quantise lower; group-neutralise
  attenuates by `kappa_g = √(1−R²_group)` (prior ≈0.84).
- **sign**: orientation `s_τ ∈ {±1}` (reverse/negate/inverse flip; multiply/divide
  combine).

`aggregate.corr_model` combines them as a **generalised cosine**:

```
rho_sig(τ,σ) = s_τ s_σ · field_sim · Omega_time · Omega_cs · κ_τ κ_σ
corr_model(X,Y) = Σ c^X_τ c^Y_σ rho_sig(τ,σ) / sqrt(G_X G_Y)
```

Then `d_red(X,Y) = 1 − |psi(corr_model)|`, where **psi** (`calibrate.py`, sklearn
isotonic) maps the structural estimate to realised PnL-correlation. `psi` is fit on
~3.7k realised self-correlation triples (`max_self_correlation` + matched pool alpha)
anchored by ~465 exact pairwise PnL-correlations among the 31 submitted alphas
(there are no PnL vectors for the 85k corpus — "Case B-plus").

It is a strong **estimator**, not exact: multiplicative / `ts_corr` terms are
first-order (child `|H|` composed, fields unioned); filters are magnitude-only.

## 4. Why not tree-edit distance, and why KNN-as-quality-predictor was deleted

Tree-edit distance measures *syntax*; two alphas with different trees can be
behaviourally identical (`rank` vs `zscore`, `ts_mean_5` vs `ts_mean_6`) and TED
misses it. `corr_model` measures *behaviour* directly.

The v1 KNN Sharpe predictor `Ŝ = Σ k(d) S_i / Σ k(d)` is **deleted**. It is the same
regression of Sharpe on structure as Object B but with a metric built to estimate
*correlation* not Sharpe, an isotropic kernel where the map is anisotropic, and
local-only variance. The GBM learns the target geometry and dominates. KNN survives
**only** as the coverage count `n_eff`, where there is no target and the
misalignment objection does not apply.

## 5. Object B — the submission-component surrogate

`features.py` turns a signature into `phi(X)`: operator multiset counts, hashed
root-to-leaf path n-grams (1/2/3), per-term filter summary (passband centroid /
bandwidth / DC gain / #TS stages), CS-warp & sign summaries, field-category
fractions, depth / node / nesting, support flags, window stats. Dim ≈ 196.

`surrogate.py` fits **quantile gradient-boosting heads** (sklearn
`HistGradientBoostingRegressor(loss="quantile")`) at levels 0.16/0.50/0.84 for:

```
subsharpe  <- sub-universe Sharpe    (the TIGHTEST must-pass gate)
sharpe     <- main Sharpe            (feeds fitness only)
ret        <- returns                (feeds fitness only)
turnover   <- turnover               (reliable kill; near-deterministic)
```

**Fitness is reconstructed, never regressed** (§7.3 of the spec). The BRAIN formula
was confirmed numerically on the corpus (mean abs err 0.0025 = rounding):

```
fitness = sharpe · sqrt(|ret| / max(turnover, 0.125))
```

The optimistic edge is pushed through this monotone formula:
`fitness_upper = q_hi(sharpe)·sqrt(|q_hi(ret)| / max(q_lo(turnover),0.125))`.
This beats a fitness head: fewer opaque targets, error stays interpretable, the
turnover head is reused, the model never relearns a square root.

## 6. The single most important property of B: calibration, not accuracy

The map `phi → Sharpe` is genuinely hard. That does **not** break B, *provided its
uncertainty is calibrated*. The gate trusts the **interval**, never the point
estimate:

- Where `phi` doesn't determine `S`, the interval is **wide** → the gate
  **simulates** (doesn't trust the model). Hardness → wider intervals → more sims —
  the correct, safe direction.
- So the acceptance test is **quantile coverage on held-out sims**
  (`P[y ≤ q_0.84] ≈ 0.84`), *not* R²/accuracy. **Result:** coverage is within ~0.02
  of nominal on all four heads, even the label-sparse sub-Sharpe head (200 train
  labels). As a bonus there is real signal too — Spearman(pred, actual) ≈ 0.51
  sharpe, 0.50 ret, 0.75 turnover.

## 7. The gate (`gate.py`)

`should_simulate` skips only on a **reliable** signal or a **confident** quality
miss; simulates when under-explored or when anything plausibly clears. Per-metric
UCB points in the exploration-preserving direction:

1. **Reliable kills** (no ε-rescue): turnover `q50 − β·σ > tau_to` (even optimistic
   too high); redundancy `c_max − β·σ_corr > tau_corr` (even optimistic, a
   duplicate).
2. **ε trickle** (`eps`): un-calcify B's blind spots.
3. **Coverage abstention**: `n_eff < n_min` → simulate (B is off-support).
4. **Quality conjunction**: skip only if `subsharpe UCB < tau_sub` **or**
   `fitness_upper < tau_fit` (the binding constraint emerges per-candidate; never a
   single combined scalar).

**Two distinct uncertainties, both required.** `σ(phi)` from B = how noisy the
prediction is at X. `n_eff` = how much data is near X. A GBM can be narrowly (over)
confident off-support; `n_eff` guards that independently of B's claimed `σ`.

**Coverage lives in phi-space, not d_red space — a deliberate change from the
spec.** `d_red = 1−|psi(corr_model)|` compresses every distance into ≈[0.32, 0.91]
(psi caps near 0.68, because high structural corr only predicts ~0.68 PnL-corr), so
even structural twins get negligible kernel weight and coverage would fire
everywhere. Coverage asks "does B have nearby training data," and **B operates on
phi**; a standardised-phi Gaussian density answers that correctly *and* is fully
vectorised (fast enough for the backtest's millions of evals). `c_max` still uses
corr_model+psi over the 31 pool members (cheap, exact).

## 8. The robustness floor

If `phi` had no Sharpe signal, B would collapse to the base rate with wide
intervals everywhere, and the gate would simulate nearly everything **except** the
two kills that don't pass through the hard map: **turnover** (near-deterministic in
structure) and **redundancy** `c_max` (analytic, no learning). So the system
degrades *gracefully to simulating* on the Sharpe axis (safe) while keeping real
savings on the two reliable axes. `backtest.robustness_floor` verifies this by
zeroing B's quality branch. *(In practice B is far from the floor — see §6.)*

## 9. Known approximations & the ceiling

- **Smooth-envelope filters (deliberate deviation from the spec's exact forms).**
  The spec lists exact responses (e.g. the Dirichlet kernel for `ts_mean`). Those
  carry sidelobe ripple whose *pattern* differs between window 5 and 6, which
  spuriously drops `Omega_time(ts_mean_5, ts_mean_6)` below the spec's own >0.98
  threshold. Behavioural similarity tracks the **passband**, not ripple — so we use
  each filter's smooth single-pole envelope (corners scale identically,
  `ω_c = 2π/window`). This meets every §9 spectral acceptance test with margin; the
  exact forms do not.
- **ω-weighted power overlap.** `|H|²` sharpens passband distinctions; the `√ω`
  weight suppresses the very-low-frequency / DC mode every low-pass shares (it is
  cross-sectionally neutralised in production and not where alphas differ).
- **First-order multiplicative / `ts_corr` terms; magnitude-only filters**
  (phase / lead-lag dropped — `ts_delay` is all-pass).
- **Redundancy is a high-threshold duplicate detector, not a low-range PnL-corr
  predictor** (the key calibration finding — §10). The only exact object would be
  the realised signal tensor, unavailable precisely because the budget is sim slots.
  If a partial-sim probe ever exists, that becomes a two-stage extension.

## 10. Validation summary (what was actually measured)

**Object A.** `corr_model` cleanly separates structural near-duplicates (same
skeleton, swapped field/window: mean **0.935**, 97% > 0.5) from unrelated pairs
(mean **0.128**, 9% > 0.5). Its *global* Spearman vs realised PnL-corr is only
**0.21** — but that is because most pairs are structurally dissimilar yet share a
**factor-driven PnL-corr floor** (~0.1–0.2) that structure cannot and should not
predict. In the **high regime the kill uses**, it *is* predictive: realised PnL-corr
is 0.37 at corr_model ≥ 0.3, **0.62 at ≥ 0.85**. So the redundancy term is trusted
as a **high-threshold duplicate kill** (`tau_corr` set accordingly) and as the
coverage signal — never as a precise low-range predictor.

**Object B.** Held-out quantile coverage ≈ nominal on all heads (the headline test),
turnover precision-at-cap 0.78, discriminative Spearman 0.50–0.75.

**Budget backtest** (`backtest.py` + `verify.py`, **TIME-split**: surrogate trained
on older sims, evaluated on newer — no leakage and it stresses transfer/decay; the
authoritative frontier. n=6000 test, base rate good = 7.6%). β is the *quality* UCB
width; `beta_kill=1.0` is fixed for the reliable kills (decoupled from β because a
single width couples the quality gate to the redundancy/turnover kills, whose
threshold then runs past psi's ~0.68 cap):

| β (quality) | slots saved | good recall | quality-zone recall | enrichment |
|---|---|---|---|---|
| 1.5 | 83% | 43% | 53% | 3.1× |
| 2.0 | 71% | 54% | 67% | 2.2× |
| **2.5 (default)** | **45%** | **67%** | **85%** | **1.5×** |
| 3.0 | 20% | 74% | 93% | 1.1× |

The "quality-zone recall" column isolates Object B: among candidates that reach the
quality conjunction (not reliably killed, not coverage-forced), the fraction of good
alphas kept. At the **β=2.5** default the gate saves ~45% of slots while keeping 85%
of good alphas in the zone; raise toward 3.0 to protect more. **An earlier
random-split backtest overstated recall (0.91 at β=2.0) — temporal leakage; the
time split is honest and the default was made more conservative accordingly.** The
remaining recall gap also includes **intended redundancy dedup** (good-but-duplicate
alphas BRAIN would reject for self-corr), not only quality false-negatives.

**Robustness floor — verified.** With B's quality branch disabled,
`good_killed_quality = 0`; only turnover + redundancy fire (reliable-kills-only mode
saves ~5% at 79% recall) and the Sharpe axis degrades to *simulate*. Fails safe.

*Re-tune with `verify.py` / `backtest.sweep` after retraining; pick β on this
frontier for your slot budget vs recall appetite. Retrain respecting time order.*

## 11. How to extend

- **Add an operator** → add its `|H|` to `filters._TS_BUILDERS` (or its warp family
  to `channels._WARP_FAMILY`), then re-run `calibrate.calibrate()` and retrain the
  surrogate.
- **Re-tune the gate** → sweep `GateParams` on `backtest.sweep` (the saved-vs-killed
  frontier); adjust `tau_*`, `beta`, `n_min`.
- **Refine field embeddings** in PnL space from single-field probe alphas
  (`FieldEmbedder.refine`, a documented v1 stub).
- **Slot-constrained variant** → if proposals exceed slots, replace the gate's
  boolean with a priority queue ranked by predicted pass-probability.

**Anti-patterns (do not):** regress fitness with its own head; gate on a single
combined scalar; KNN-predict Sharpe; validate B with R²; treat `corr_model` as exact
for multiplicative terms; prune on neighbour mean instead of UCB; drop the ε trickle.
