# Insider transaction flows cross-checked by option positioning

Status: ITERATE
Decision: ITERATE - signal is not dead, but evidence is below expansion threshold

## Metrics
- target_delay: 1
- n_done: 11
- best_sharpe: 1.16
- median_sharpe: 0.26
- best_fitness: 0.24
- n_standalone_pass: 0
- n_positive_delta: 0
- n_positive_residual: 0

## Next Rule
Stop when kill/promote is reached. Otherwise run at most the configured max_iterations.
