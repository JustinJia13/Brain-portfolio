# Template Sweep Method — fix the forms, search the bindings

## The problem

The oldest approach in this repository, and the opposite bet from
[`../grammar_method/`](../grammar_method/README.md).

Grammar search *derives* expression shapes and treats the functional form as unknown. This
method assumes the opposite: that the useful functional forms are already known from the
literature and from practice — a cross-sectional rank of a time-series z-score, a group-neutral
reversal, a ratio against a size proxy — and that the only thing left to search is which fields
and windows to bind into them.

That makes the search space small enough to sweep exhaustively, which is its real attraction:
you get coverage guarantees instead of sampling noise. Its weakness is equally clear — it can
only ever find what the template author already imagined.

## How it discovers alpha

A fixed registry of parameterised templates is crossed with field bindings and window
parameters, and the resulting grid is swept and logged.

The part worth a closer look is `regime_check.py`. It evaluates the **same expression** under a
settings grid **declared in advance** — neutralization, decay, truncation, universe — rather
than searching settings for the best result. That distinction is the whole anti-overfit point:
a pre-declared grid measures robustness, while an adaptive search over the same grid
manufactures it. `../../PIPELINE.md` explicitly warns against using template variants as an
adaptive parameter search loop, and that warning is why this lane is retained in a narrow role.

`consolidate_logs.py` was a one-time migration that folded scattered per-lane sim logs into the
single `artifacts/sim/master.jsonl` ledger. It has already run.

## The code

| File | LOC | What it does |
|---|---|---|
| `alpha_discovery/template_sweep_method/regime_check.py` | 368 | Predeclared regime checks for passing alpha expressions |
| `alpha_discovery/template_sweep_method/pipeline.py` **(CLI)** | 258 | Unified CLI entry point for the alpha-search pipeline |
| `alpha_discovery/template_sweep_method/templates.py` | 242 | Proven alpha template library for targeted window/settings sweeps |
| `alpha_discovery/template_sweep_method/consolidate_logs.py` **(CLI)** | 240 | Consolidate all sim logs and runner logs into alpha_search/artifacts/ |
| `alpha_discovery/template_sweep_method/feedback.py` **(CLI)** | 156 | Template performance tracker |
| `alpha_discovery/template_sweep_method/sweep.py` **(CLI)** | 105 | Generate a JSONL candidate plan from the proven alpha template library |
| `alpha_discovery/template_sweep_method/catalog.py` | 63 | Field catalog loaded from all_fields.csv |
| `alpha_discovery/template_sweep_method/__init__.py` | 2 | Template search and sweep helpers |

## Running it

```bash
# the sweep itself
python -m alpha_discovery.template_sweep_method.pipeline --dry-run
python -m alpha_discovery.template_sweep_method.sweep --help

# the device actually worth using: a pre-declared settings grid for one expression
python tools/regime_check.py --help
```

`../../PIPELINE.md` keeps `pipeline.py` alive for deliberate compatibility experiments only.

## Results

No standalone sweep output survives in this repository: the lane's historical results were
folded into `artifacts/sim/master.jsonl` by its own `consolidate_logs.py` migration, and the
`sweep` lane tag accounts for 8,444 of the 91,969 records there — the second-largest lane after
grammar.

Its durable contribution is not a set of alphas but the **regime-check device**, which is still
used as a robustness probe on finalists, and the negative lesson recorded in `../../PIPELINE.md`:
template variants must not be run as an adaptive parameter search, because sweeping settings and
then reporting the best one is overfitting with extra steps.

