# News tone persistence filtered by option IV term structure

Status: KILLED
Decision: KILL - best Sharpe below 0.8 kill threshold

## Metrics
- target_delay: 1
- n_done: 20
- best_sharpe: 0.52
- median_sharpe: -0.14
- best_fitness: 0.12
- n_standalone_pass: 0
- n_positive_delta: 0
- n_positive_residual: 0

## Next Rule
Stop when kill/promote is reached. Otherwise run at most the configured max_iterations.
