# Spectacular Alpha Templates
Target: Sharpe>1.8 Fitness>2.5. Current best: Sharpe=1.62 Fitness=1.99.

## BANNED PATTERN

  signed_power(kth_element(FIELD,W,k=K), ts_zscore(FIELD,W2))

Do not use. Do not fix. Do not apply to other fields.
kth_element selects extreme values; signed_power raises to dynamic exponent (ts_zscore)
computing x^N=+/-inf for outliers. group_neutralize cannot fix inf weights.
All weight collapses to one stock. Confirmed unfixable. m16 fields are fine, pattern breaks.

## Templates

### Template A -- Model16 Derivative Composite [HIGHEST PRIORITY]

m16 _derivative fields: cov=1.0, pre-computed ranking changes. Safe operators only.

Fields: analyst_revision_rank_derivative, cashflow_efficiency_rank_derivative,
composite_factor_score_derivative, earnings_certainty_rank_derivative,
growth_potential_rank_derivative, multi_factor_acceleration_score_derivative,
multi_factor_static_score_derivative, relative_valuation_rank_derivative

A1 rank composite:
  normalize(rank(analyst_revision_rank_derivative) + rank(relative_valuation_rank_derivative)
    + rank(cashflow_efficiency_rank_derivative) + rank(growth_potential_rank_derivative))

A2 divergence:
  group_rank(ts_corr(analyst_revision_rank_derivative,relative_valuation_rank_derivative,W),sector)
  W in {20,40,60}

A3 acceleration:
  group_rank(ts_mean(multi_factor_acceleration_score_derivative,20),sector)

A4 revision momentum:
  group_rank(ts_mean(ts_delta(analyst_revision_rank_derivative,5),20),sector)

### Template B -- Time-Since-Event [HIGH]

  group_neutralize(ts_arg_max(ts_backfill(split,252),60),subindustry)
  group_neutralize(ts_arg_max(densify(ts_backfill(split,60)),60),subindustry)
  group_neutralize(ts_arg_max(ts_backfill(dividend,252),252),market)

### Template C -- Flag Composite Expansion

Not in baseline: anl4_ptp_flag(17825 alphas), anl4_epsr_flag(10266), anl4_gric_flag(7503),
anl4_ebitda_flag(3529), anl4_epsa_flag(775, cov=100% -- only full-coverage flag)

### Template D -- Gated Composite

  trade_when(normalize(COMPOSITE), rank(normalize(COMPOSITE))>0.8, 0)
  Risk: LOW_TURNOVER if TO<0.01. Current TO~0.02.

### Template E -- Last Revision Sparse

  zscore(last_diff_value(FIELD,W))
  Near-miss: zscore(last_diff_value(fn_antidilutive_securities_excl_from_eps_a,20))
  Sharpe=1.14 Fitness=1.39 in fundamental2.

## Priority: A>B>C>D>E
