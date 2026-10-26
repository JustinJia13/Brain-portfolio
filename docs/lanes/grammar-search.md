# Grammar Search: CFG-Generated FASTEXPR Discovery

**A context-free grammar over FASTEXPR with a Δk depth budget, semantic constraints enforced at derivation time, and three stacked prune datastores guarding every simulation slot.**

[&larr; back to the pipeline overview](../../README.md)

The bet is that alpha discovery is a *search* problem, not a *generation* problem: a cheap stochastic sampler over a well-specified expression grammar covers structure space faster than any LLM can propose it, so the intelligence belongs in SELECTION — what you refuse to simulate — not in authoring. Hence the shape: an extremely productive generator (a CFG whose productions each cost Δk, bounded by a total budget K) bolted to an increasingly expensive stack of pre-sim filters, because the binding constraint is the Brain simulation slot, not candidate supply. Every gate writes its evidence to a persistent datastore so the next round starts from the accumulated 92k-sim history rather than from scratch.

## Pipeline

```mermaid
flowchart TD
    CSV[("all_fields.csv - 7642 catalog rows plus header")]
    CLJ[("all_clusters.json - 5451 clusters over 7490 usable fields")]
    G["BrainGrammar.sample - CFG over FASTEXPR, each production costs delta-k, total budget K"]
    SEM{"semantic feasibility checked AT DERIVATION TIME"}
    CAN["canonicalize - commutative operands sorted alphabetically"]
    Q{"two-level Q-table prune - L1 cluster x op, L2 field x op"}
    QT[("all_qtable.json - 8.0 MB, 309k lines, 31627 entries")]
    FPG{"cluster-scoped n-gram Jaccard above 0.80"}
    FP[("all_tested_fingerprints.json 39 MB plus all_skeleton_counts.json 9114 skeletons")]
    NOV["novelty rank - 0.55 skeleton plus 0.35 anti-Jaccard plus 0.10 cluster"]
    GATE{"alpha_discovery/sim_slot_gate sim-slot gate - LIVE branch is turnover LCB plus redundancy c_max only"}
    DROP[("DISCARDED - no simulation slot spent")]
    SUB["run_batch_submission - Brain submit, poll, detail plus yearly plus PnL fetch"]
    LOG[("session grammar_TIMESTAMP.jsonl merged to master.jsonl - 18506 of 91969 rows")]
    UPD["compute_q_reward times robustness multiplier"]
    WS["workshop repair or enhance, then finalist gate"]

    CSV -->|"type and coverage filter, VECTOR wrapped in vec_avg"| CLJ
    CLJ -->|"uniform over dataset then cluster then field"| G
    G -->|"partial derivation, delta-k debited from K"| SEM
    SEM -->|"REJECT and resample - division RHS not a size proxy, paired fields identical, inner normalizer depth exhausted"| G
    SEM -->|"complete valid tree"| CAN
    CAN -->|"canonical FASTEXPR string plus operator fingerprint"| Q
    QT -.->|"pruned-set snapshot, one-way, never un-pruned"| Q
    Q -->|"REJECT dead combo - but only 9 of 8987 L1 and 19 of 22640 L2 are pruned, measured INERT"| G
    Q -->|"survivor"| FPG
    FP -.->|"per-cluster fingerprint and skeleton history"| FPG
    FPG -->|"REJECT structural near-duplicate"| G
    FPG -->|"novel enough"| NOV
    NOV -->|"3x oversample trimmed to n_samples, 67 pct of pool dropped"| DROP
    NOV -->|"top n_samples by novelty score"| GATE
    GATE -->|"shadow frontier at beta 2.5 would save 46 pct of slots at 0.91 quality-zone recall, McNemar chi2 1816 vs Q-table"| DROP
    GATE -->|"survivors spend a real sim slot"| SUB
    SUB -->|"IS stats, submission checks, self-correlation"| LOG
    LOG -.->|"value_sum and n_obs per cluster-op and field-op"| QT
    LOG -.->|"tested fingerprints and template_id counts"| FP
    LOG -->|"reward from fitness, zeroed on concentration at or above 0.40 or self-corr at or above 0.70"| UPD
    UPD -.->|"persisted each round"| QT
    LOG -->|"16188 done, 281 fully submittable = 1.7 pct"| WS
```

## Stages

| Stage | Implementation | What it does |
|---|---|---|
| 1. Field catalog to clusters | `alpha_discovery/grammar_method/fields.py` | 7643 CSV rows filtered by type and coverage, clustered into 5451 cached clusters over 7490 usable fields |
| 2. CFG derivation under Δk budget | `alpha_discovery/grammar_method/grammar.py` | Alpha and Expr productions each debit Δk from budget K; K bounds reachable expression shapes |
| 3. Derivation-time semantic constraints | `alpha_discovery/grammar_method/grammar.py feasible_*_productions` | Division RHS must be a size proxy, paired fields distinct, inner-normalizer depth capped at 1. Not a post-filter |
| 4. Commutative canonicalization | `alpha_discovery/grammar_method/grammar.py canonicalize` | Sorts commutative operands so add of a and b equals add of b and a; halves the paired-operator space |
| 5. Two-level Q-table prune | `alpha_discovery/grammar_method/qtable.py` | One-way prune of cluster-op and field-op pairs. MIN_OBS 8 and 12, mean_q below 0.05. Measured near-inert |
| 6. Structural novelty veto and rank | `alpha_discovery/grammar_method/novelty.py plus planner.py` | Cluster-scoped n-gram Jaccard above 0.80 vetoed; 3x oversampled pool ranked by novelty, trimmed to n_samples |
| 7. Sim-slot gate and submit | `alpha_discovery/grammar_method/run_pipeline.py run_batch_submission` | alpha_discovery/sim_slot_gate surrogate kills 46 pct of slots, then survivors are threaded to Brain and polled to completion |
| 8. Reward, persist, merge | `alpha_discovery/grammar_method/run_pipeline.py main` | Fitness times robustness reward with concentration and self-correlation vetoes; Q-table, fingerprints, master.jsonl all updated per round |

## Running it

```bash
python -m alpha_discovery/grammar_method.run_pipeline --fields-csv all_fields.csv --credentials brain2.txt --delay 1 --exclude-dataset analyst4 --rounds 0 --budget 6 --n-samples 300 --batch-size 3 --max-candidates 300 --gate-mode reliable
```

## Outcome

Measured on artifacts/sim/master.jsonl (2026-06): 18,506 grammar-lane records of 91,969 total, 16,188 completed, 482 at IS Sharpe >= 1.25 (3.0%), and 281 passing every submission gate including concentrated-weight and sub-universe (1.7% hit rate). That is the repo's best generator — the decompose-expand variant scored 0/100 submittable (memory note llm-generator-underperforms), which is why PIPELINE.md keeps grammar running as the background evidence collector. Two honest negatives sit inside the lane. (1) dedup.md's measurement over 84,717 records: only 7.9% of (skeleton, field-set) cells ever recur, 92.3% of cells top out below Sharpe 1.25, and 90.5% of sim budget is spent in cells history already proved dead — so identity dedup was solving the wrong problem. (2) The lane's own Q-table pruner is effectively inert: I verified 9 of 8987 L1 and 19 of 22640 L2 entries pruned (<0.15%), matching CALIBRATION_REPORT.md line 112 and ASSUMPTIONS.md G3, where the alpha_discovery/sim_slot_gate surrogate challenger dominates it (46% slots saved at 0.91 quality-zone recall, McNemar chi2 = 1816). The real pre-sim cull today is the 3x novelty oversample-and-trim plus that gate, not the Q-table.

