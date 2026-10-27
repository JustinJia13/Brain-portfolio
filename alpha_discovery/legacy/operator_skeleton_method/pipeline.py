"""Operator-first alpha discovery pipeline.

Flow:
  1. Load cluster pool (from alpha_discovery/grammar_method cluster cache)
  2. Sample cluster pairs per screening template (uniform random)
  3. Submit screening expressions to Brain, log results
  4. Rank by IS Sharpe × year consistency
  5. Promote top-K pairs to full template suite + submit
  6. Log promoted results in workshop-compatible format

Typical invocation (from world_quant/):

    python -m alpha_discovery.legacy.operator_skeleton_method.pipeline --n-pairs 200 --seed 42

Dry run (print expressions with cluster labels, no Brain submission):

    python -m alpha_discovery.legacy.operator_skeleton_method.pipeline --dry-run --n-pairs 20

Show cluster pool summary:

    python -m alpha_discovery.legacy.operator_skeleton_method.pipeline --show-clusters
"""

from __future__ import annotations

import argparse
from pathlib import Path

from core.logs import run_log_path

import numpy as np

from alpha_discovery.legacy.operator_skeleton_method.field_sampler import (
    load_clusters,
    summarize_clusters,
    get_pruned_fields,
    ClusterRecord,
)
from alpha_discovery.legacy.operator_skeleton_method.templates import (
    SCREENING_TEMPLATES,
    RelationshipTemplate,
)
from alpha_discovery.legacy.operator_skeleton_method.screener import submit_screening_batch, ScreeningResult
from alpha_discovery.legacy.operator_skeleton_method.promoter import (
    select_top_pairs,
    build_promotion_candidates,
    build_all_screening_candidates,
)

_DIR = Path(__file__).parent
_ARTIFACTS = _DIR / "artifacts"
_DEFAULT_QTABLE = Path(__file__).parent.parent / "alpha_discovery/grammar_method/artifacts/all_qtable.json"
_DEFAULT_CREDS = Path(__file__).parent.parent / "brain.txt"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Operator-first relationship scanner: templates × clusters → Brain screening."
    )
    p.add_argument("--credentials", default=str(_DEFAULT_CREDS))
    p.add_argument("--n-pairs", type=int, default=200,
                   help="Cluster pairs to sample per screening template (default 200)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--top-k", type=int, default=20,
                   help="Top-K pairs to promote after screening (default 20)")
    p.add_argument("--sharpe-threshold", type=float, default=1.25)
    p.add_argument("--batch-size", type=int, default=3)
    p.add_argument("--sleep-seconds", type=int, default=15)
    p.add_argument("--max-submit-retries", type=int, default=5)
    p.add_argument("--screening-log",
                   default=str(_ARTIFACTS / "screening_results.jsonl"))
    p.add_argument("--promotion-log",
                   default=str(_ARTIFACTS / "promoted_candidates.jsonl"))
    p.add_argument("--qtable-path", default=str(_DEFAULT_QTABLE))
    p.add_argument("--skip-promotion", action="store_true", default=False)
    p.add_argument("--dry-run", action="store_true", default=False,
                   help="Print candidate expressions with cluster labels, no submission")
    p.add_argument("--show-clusters", action="store_true", default=False,
                   help="Print cluster pool summary and exit")
    return p.parse_args()


def _load_pruned_map(
    templates: list[RelationshipTemplate],
    qtable_path: str,
) -> dict[str, frozenset[str]]:
    pruned_map: dict[str, frozenset[str]] = {}
    for tmpl in templates:
        pf = get_pruned_fields(tmpl, qtable_path)
        pruned_map[tmpl.template_id] = pf
    return pruned_map


def dry_run_print(
    clusters: list[ClusterRecord],
    templates: list[RelationshipTemplate],
    n_pairs: int,
    rng: np.random.Generator,
) -> None:
    candidates = build_all_screening_candidates(clusters, templates, n_pairs, rng)

    # Group by template_id for clean display
    by_template: dict[str, list[dict]] = {}
    for c in candidates:
        by_template.setdefault(c["template_id"], []).append(c)

    print(f"\nDRY RUN — {len(candidates)} candidate expressions "
          f"({n_pairs} pairs × {len(templates)} templates)\n")

    for tmpl_id, items in by_template.items():
        tmpl = next(t for t in templates if t.template_id == tmpl_id)
        # Deduplicate by (field_a, field_b) to show unique pairs (windows collapse to one row)
        seen_pairs: set[tuple[str, str]] = set()
        unique_items = []
        for item in items:
            key = (item["field_a"], item["field_b"])
            if key not in seen_pairs:
                seen_pairs.add(key)
                unique_items.append(item)

        print(f"  ── {tmpl_id}  ({tmpl.description})")
        for item in unique_items:
            la = item["label_a"][:40]
            lb = item["label_b"][:40]
            fa = item["field_a"]
            fb = item["field_b"]
            # Show one representative expression (first window if windowed)
            w = item["window"]
            expr = tmpl.render(fa, fb, w=w, c=item.get("field_c"))
            windows_str = f"  w={w}" if w is not None else ""
            print(f"     [{la}]  /  [{lb}]{windows_str}")
            print(f"       → {expr}")
        print()


def main() -> None:
    args = parse_args()
    _ARTIFACTS.mkdir(parents=True, exist_ok=True)

    # ── 1. Load cluster pool ─────────────────────────────────────────────────
    print("Loading cluster pool ...")
    clusters = load_clusters()
    print(f"  {len(clusters)} clusters loaded")

    if args.show_clusters:
        summarize_clusters(clusters)
        return

    templates = list(SCREENING_TEMPLATES)
    rng = np.random.default_rng(args.seed)

    # ── 2. Dry run ───────────────────────────────────────────────────────────
    if args.dry_run:
        dry_run_print(clusters, templates, args.n_pairs, rng)
        return

    # ── 3. Q-table exclusions ────────────────────────────────────────────────
    pruned_map = _load_pruned_map(templates, args.qtable_path)

    # ── 4. Build screening candidates ───────────────────────────────────────
    print(f"\nBuilding screening candidates ({args.n_pairs} pairs/template, seed={args.seed}) ...")
    rng = np.random.default_rng(args.seed)
    screening_candidates = build_all_screening_candidates(
        clusters, templates, args.n_pairs, rng, pruned_map
    )
    print(f"  {len(screening_candidates)} screening expressions generated")

    # ── 5. Submit ────────────────────────────────────────────────────────────
    print(f"\nSubmitting → {args.screening_log}")
    from core.brain.api import sign_in
    session = sign_in(args.credentials)

    screening_results = submit_screening_batch(
        session, screening_candidates, args.screening_log,
        batch_size=args.batch_size,
        sleep_seconds=args.sleep_seconds,
        max_submit_retries=args.max_submit_retries,
        credentials_path=args.credentials,
    )

    # ── 6. Summary ───────────────────────────────────────────────────────────
    done = [r for r in screening_results if r.status == "done"]
    passed = [r for r in done if r.passes_promotion(args.sharpe_threshold)]
    avg_sharpe = sum(r.is_sharpe for r in done) / len(done) if done else 0.0
    print(f"\nScreening summary:")
    print(f"  Done: {len(done)}  Passed (sharpe≥{args.sharpe_threshold}): {len(passed)}")
    print(f"  Avg IS Sharpe: {avg_sharpe:.3f}")

    if passed:
        print("\nTop results:")
        for r in sorted(passed, key=lambda x: x.screening_score, reverse=True)[:10]:
            print(f"  {r.is_sharpe:.3f}  [{r.label_a[:35]}] / [{r.label_b[:35]}]")
            print(f"         {r.expression[:80]}")

    if args.skip_promotion or not passed:
        return

    # ── 7. Promote ───────────────────────────────────────────────────────────
    top = select_top_pairs(screening_results, top_k=args.top_k,
                           sharpe_threshold=args.sharpe_threshold)
    promotion_candidates = build_promotion_candidates(top)
    promo_dicts = [c.as_candidate_dict() for c in promotion_candidates]
    print(f"\nPromoting {len(top)} pairs → {len(promo_dicts)} expressions → {args.promotion_log}")

    from alpha_discovery.grammar_method.run_pipeline import run_batch_submission, DEFAULT_SIMULATION_SETTINGS
    run_batch_submission(
        session, promo_dicts, DEFAULT_SIMULATION_SETTINGS,
        args.promotion_log,
        str(run_log_path("operator")),
        batch_size=args.batch_size,
        max_candidates=0,
        sleep_seconds=args.sleep_seconds,
        max_submit_retries=args.max_submit_retries,
        credentials_path=args.credentials,
        label="PROMO",
    )
    print(f"Done. Feed to workshop: python -m tools.batch_workshop --sim-log {args.promotion_log}")


if __name__ == "__main__":
    main()
