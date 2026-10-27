"""Generate a JSONL candidate plan from the proven alpha template library.

Usage:
    python -m alpha_discovery.template_sweep_method.sweep --output artifacts/sweep/alpha_batch.jsonl
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from alpha_discovery.template_sweep_method.templates import AlphaTemplate, get_alpha_templates


def _candidate_id(expression: str) -> str:
    """Stable 16-char hex ID from expression string."""
    return hashlib.sha1(expression.encode()).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Proven-alpha sweep (primary pipeline)
# ---------------------------------------------------------------------------

def build_alpha_candidates(templates: list[AlphaTemplate]) -> list[dict]:
    """Expand proven AlphaTemplate objects into candidate dicts.

    For each template generates (window/constant variants) × (settings variants).
    Deduplicates by (expression, regime_key). No field or group sweeping.
    """
    from core.brain.submit import DEFAULT_SETTINGS, _regime_key

    seen: set[tuple[str, str]] = set()
    candidates: list[dict] = []

    for tmpl in templates:
        for expr, params, settings_override in tmpl.variants():
            settings = {**DEFAULT_SETTINGS, **settings_override}
            regime = _regime_key(settings)
            key = (expr, regime)
            if key in seen:
                continue
            seen.add(key)
            candidates.append({
                "candidate_id": _candidate_id(expr + regime),
                "template_id": tmpl.alpha_id,
                "dataset_id": tmpl.dataset_id,
                "category": tmpl.category,
                "search_lane": "alpha_sweep",
                "expression": expr,
                "fields": [],
                "group": "",
                "params": params,
                "priority_score": tmpl.priority,
                "settings_override": settings_override,
                "is_probe": False,
            })

    candidates.sort(key=lambda c: c["priority_score"], reverse=True)
    return candidates


def build_alpha_plan(output_path: str = "artifacts/sweep/alpha_batch.jsonl") -> list[dict]:
    """Build a JSONL candidate plan from ALPHA_TEMPLATES and write to output_path."""
    import json

    templates = get_alpha_templates()
    if not templates:
        print("No alpha templates registered. Add entries to ALPHA_TEMPLATES in alpha_discovery/template_sweep_method/templates.py.")
        return []

    candidates = build_alpha_candidates(templates)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        for c in candidates:
            f.write(json.dumps(c) + "\n")

    print(f"Wrote {len(candidates)} candidates to {output_path}")
    _print_alpha_summary(candidates)
    return candidates


def _print_alpha_summary(candidates: list[dict]) -> None:
    from collections import Counter
    by_alpha = Counter(c["template_id"] for c in candidates)
    by_lane = Counter(c.get("search_lane", "?") for c in candidates)
    print(f"Search lanes: {dict(by_lane)}")
    print("By alpha template:")
    for alpha_id, n in sorted(by_alpha.items(), key=lambda x: -x[1]):
        print(f"  {alpha_id:50s}  {n:4d} variants")


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Generate JSONL candidate plan from proven alpha templates.")
    parser.add_argument("--output", default="artifacts/sweep/alpha_batch.jsonl", help="Output JSONL path.")
    args = parser.parse_args()

    build_alpha_plan(output_path=args.output)


if __name__ == "__main__":
    main()
