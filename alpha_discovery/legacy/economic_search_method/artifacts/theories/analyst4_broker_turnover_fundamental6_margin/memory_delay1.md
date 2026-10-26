# Broker-coverage turnover as signal quality filter on margin trend

Status: KILLED
Decision: KILL - best Sharpe below 0.8 kill threshold

## Metrics
- target_delay: 1
- n_done: 3
- best_sharpe: 0.34
- median_sharpe: 0.3
- best_fitness: 0.1
- n_standalone_pass: 0
- n_positive_delta: 0
- n_positive_residual: 0

## Next Rule
Stop when kill/promote is reached. Otherwise run at most the configured max_iterations.
