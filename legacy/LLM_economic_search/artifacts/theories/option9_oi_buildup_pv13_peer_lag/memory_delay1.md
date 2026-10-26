# Option open interest buildup ahead of peer-relative move

Status: KILLED
Decision: KILL - second-stage produced no standalone pool-positive candidate

## Metrics
- target_delay: 1
- stage: iterate
- n_second_stage_done: 16
- n_standalone_pass: 0
- n_promote: 0
- best_sharpe: 1.12
- best_delta_pool_sharpe: None

## Next Rule
Stop when kill/promote is reached. Otherwise run at most the configured max_iterations.
