"""Theory card -> FASTEXPR family rendering.

The card never contains final expressions. It declares motif skeletons with
typed slots ({field} {w} {d} {group}); this module validates skeleton operators
against the grammar registry and expands slots with the grammar vocabulary,
then stratified-samples the cross product down to a bounded family in which
every declared field and every skeleton appears at least once. That uniform
coverage is what makes the family verdict a designed experiment instead of an
LLM-cherry-picked sample.
"""

from __future__ import annotations

import itertools
import random
import re
from typing import Any

from alpha_discovery.grammar_method.grammar import GrammarConfig
from core.expr.operators import BANNED_GENERATION_OPS, OPERATOR_REGISTRY

# field, field2..field4 (relational/interaction), plus window/delta/group slots.
SLOT_PATTERN = re.compile(r"\{(field\d*|w|d|group)\}")
IDENT_CALL_PATTERN = re.compile(r"([a-zA-Z_][a-zA-Z0-9_]*)\s*\(")
MAX_SECONDARY_FIELDS = 6       # cap candidates per secondary field slot to bound the product
MAX_VARIANTS_PER_SKELETON = 400

# Operators valid in skeletons beyond the generation registry (structural ops
# the grammar gates behind separate probability config).
EXTRA_SKELETON_OPS: frozenset[str] = frozenset({
    "if_else", "trade_when", "bucket", "normalize", "ts_step", "kth_element",
})

DEFAULT_FAMILY_MIN = 5
DEFAULT_FAMILY_MAX = 20


_DEGENERATE_BINARY_OPS = ("subtract", "divide", "ts_regression", "ts_corr", "ts_covariance")


def is_degenerate_expr(expr: str) -> bool:
    """True if a binary op has two structurally-identical operands.

    subtract(X, X)=0, divide(X, X)=1, ts_corr(X, X)=1 — dead signals that arise
    when a relational skeleton reuses the same field slot in both operands.
    """
    for op in _DEGENERATE_BINARY_OPS:
        i = expr.find(op + "(")
        while i >= 0:
            start = i + len(op) + 1
            depth, k, arg_start, args = 1, start, start, []
            while k < len(expr) and depth > 0:
                c = expr[k]
                if c == "(":
                    depth += 1
                elif c == ")":
                    depth -= 1
                elif c == "," and depth == 1:
                    args.append(expr[arg_start:k].strip())
                    arg_start = k + 1
                k += 1
            args.append(expr[arg_start:k - 1].strip())
            if len(args) >= 2 and args[0] == args[1]:
                return True
            i = expr.find(op + "(", i + 1)
    return False


def skeleton_ops(skeleton: str) -> set[str]:
    return set(IDENT_CALL_PATTERN.findall(skeleton))


def invalid_skeleton_ops(skeleton: str) -> set[str]:
    allowed = (set(OPERATOR_REGISTRY) - set(BANNED_GENERATION_OPS)) | EXTRA_SKELETON_OPS
    return skeleton_ops(skeleton) - allowed


def slotify_skeletons(card: dict, clusters: dict[str, dict]) -> dict:
    """Safe auto-repair: replace literal member-field names in skeletons with {field}.

    LLMs sometimes hardcode the cluster's field name despite instructions; that
    repair is unambiguous, so do it in code. Anything else (literal windows,
    ranges, alternatives) stays a validation error.
    """
    fields = set(card_fields(card, clusters))
    if not fields:
        return card
    skeletons: list[str] = []
    for s in card.get("motif_skeletons") or []:
        s = str(s)
        # map distinct hardcoded member-fields to distinct slots (field, field2, ...)
        present: list[str] = []
        for m in re.finditer(r"\b[a-z][a-z0-9_]{3,}\b", s):
            t = m.group()
            if t in fields and t not in present:
                present.append(t)
        for i, fld in enumerate(present):
            slot = "{field}" if i == 0 else f"{{field{i + 1}}}"
            s = re.sub(r"\b" + re.escape(fld) + r"\b", slot, s)
        skeletons.append(s)
    card["motif_skeletons"] = skeletons
    return card


def validate_card(card: dict, *, known_clusters: set[str]) -> list[str]:
    """Structural validation; returns a list of problems (empty = valid)."""
    problems: list[str] = []
    cluster_ids = card.get("cluster_ids") or []
    if not cluster_ids:
        problems.append("no cluster_ids")
    for cid in cluster_ids:
        if cid not in known_clusters:
            problems.append(f"unknown cluster_id: {cid}")
    band = card.get("horizon_band") or []
    if not (isinstance(band, (list, tuple)) and len(band) == 2 and band[0] <= band[1]):
        problems.append(f"bad horizon_band: {band}")
    skeletons = card.get("motif_skeletons") or []
    if not skeletons:
        problems.append("no motif_skeletons")
    for skeleton in skeletons:
        bad = invalid_skeleton_ops(str(skeleton))
        if bad:
            problems.append(f"unknown ops {sorted(bad)} in skeleton: {skeleton}")
        if not SLOT_PATTERN.search(str(skeleton)):
            problems.append(f"skeleton has no slots: {skeleton}")
    if str(card.get("direction") or "+") not in ("+", "-"):
        problems.append(f"bad direction: {card.get('direction')}")
    return problems


def _band_values(values: tuple[int, ...], band: tuple[int, int]) -> list[int]:
    lo, hi = band
    inside = [v for v in values if lo <= v <= hi]
    if inside:
        return inside
    # nothing inside the band: take the closest vocabulary value
    return [min(values, key=lambda v: min(abs(v - lo), abs(v - hi)))]


def card_fields(card: dict, clusters: dict[str, dict]) -> list[str]:
    """Member fields of the declared clusters: champion first, sleeper last."""
    ordered: list[str] = []
    sleepers: list[str] = []
    for cid in card.get("cluster_ids") or []:
        cluster = clusters.get(cid) or {}
        champion = cluster.get("champion_field_id")
        sleeper = cluster.get("sleeper_field_id")
        if champion and champion not in ordered:
            ordered.append(str(champion))
        for member in cluster.get("members") or []:
            fid = str(member.get("field_id") or "")
            if not fid or fid in ordered or fid in sleepers:
                continue
            if fid == sleeper:
                sleepers.append(fid)
            else:
                ordered.append(fid)
    return ordered + sleepers


def field_type_map(card: dict, clusters: dict[str, dict]) -> dict[str, str]:
    """{field_id: field_type} over the card's declared clusters."""
    types: dict[str, str] = {}
    for cid in card.get("cluster_ids") or []:
        for member in (clusters.get(cid) or {}).get("members") or []:
            fid = str(member.get("field_id") or "")
            if fid:
                types[fid] = str(member.get("field_type") or "").upper()
    return types


def field_token(field_id: str, field_types: dict[str, str]) -> str:
    """Render a field for substitution.

    VECTOR fields are multi-valued per stock per day and MUST be collapsed to a
    scalar with a vector aggregator before any TS/CS operator, or Brain errors
    the simulation (no alpha returned). Mirrors alpha_discovery.grammar_method._field_node.
    """
    if field_types.get(field_id) == "VECTOR":
        return f"vec_avg({field_id})"
    return field_id


def _render(skeleton: str, binding: dict[str, Any], field_types: dict[str, str]) -> str:
    def sub(match: re.Match) -> str:
        slot = match.group(1)
        value = binding[slot]
        if slot.startswith("field"):  # field, field2..field4 — VECTOR-wrap each
            return field_token(str(value), field_types)
        return str(value)
    return " ".join(SLOT_PATTERN.sub(sub, skeleton).split())


def enumerate_family(
    card: dict,
    clusters: dict[str, dict],
    *,
    config: GrammarConfig | None = None,
    family_min: int = DEFAULT_FAMILY_MIN,
    family_max: int = DEFAULT_FAMILY_MAX,
    known_expressions: set[str] | None = None,
    coverage: "Any | None" = None,
    seed: int = 7,
) -> list[dict]:
    """Render the card's expression family as candidate dicts.

    known_expressions: whitespace-normalized expressions already tested
    (master + registry); matching variants are still emitted but flagged
    cached=True so the runner can use logged results as zero-cost evidence.
    coverage: optional GrammarCoverage. When given, variants whose (field, op)
    the grammar already found DEAD are dropped, and the family is biased toward
    novel/strong (field, op) — so the lane stops re-covering ground already
    explored across the 76k-sim master.
    """
    cfg = config or GrammarConfig()
    rng = random.Random(seed)
    known = known_expressions or set()
    primary_cluster = (card.get("cluster_ids") or [None])[0]

    fields = card_fields(card, clusters)
    if not fields:
        return []
    field_types = field_type_map(card, clusters)
    band = tuple(card.get("horizon_band") or (cfg.windows[0], cfg.windows[-1]))
    windows = _band_values(cfg.windows, band)
    deltas = _band_values(cfg.delta_windows, band)
    groups = [g for g in cfg.groups if g != "none"] or ["industry"]
    skeletons = [str(s) for s in card.get("motif_skeletons") or []]

    # full grid per skeleton, over the slots that skeleton uses. Field slots
    # ({field}, {field2}..{field4}) each range over the cluster fields so
    # relational/interaction skeletons (ratios, ts_corr, conditionals) are
    # realised across field pairs.
    secondary = fields[:MAX_SECONDARY_FIELDS]
    variants: list[dict] = []
    seen: set[str] = set()
    for skeleton in skeletons:
        used = set(SLOT_PATTERN.findall(skeleton))
        field_slots = sorted(s for s in used if s.startswith("field"))
        axes: list[tuple[str, list]] = [
            (fs, fields if fs == "field" else secondary) for fs in field_slots
        ]
        if "w" in used:
            axes.append(("w", windows))
        if "d" in used:
            axes.append(("d", deltas))
        if "group" in used:
            axes.append(("group", groups))
        names = [n for n, _ in axes]
        n_gen = 0
        for combo in itertools.product(*(vals for _, vals in axes)):
            if n_gen >= MAX_VARIANTS_PER_SKELETON:
                break
            binding = dict(zip(names, combo))
            # distinct field slots must get distinct fields (no divide(x, x))
            fvals = [binding[fs] for fs in field_slots]
            if len(set(fvals)) < len(fvals):
                continue
            n_gen += 1
            expr = _render(skeleton, binding, field_types)
            if expr in seen or is_degenerate_expr(expr):
                continue  # drop subtract(X,X)/divide(X,X) — identically dead
            seen.add(expr)
            strength = 0.5
            if coverage is not None:
                field = binding.get("field")
                op = coverage.primary_op(expr, field)
                # Skip ground the grammar already found dead; keep cached
                # exact-matches (free evidence) regardless.
                if expr not in known and coverage.is_dead(field, op, primary_cluster):
                    continue
                strength = coverage.strength(field, op)
            variants.append({
                "skeleton": skeleton,
                "binding": binding,
                "expression": expr,
                "cached": expr in known,
                "strength": strength,
            })

    if not variants:
        return []

    # Stratified cap: every skeleton >=1, every field >=1, then fill by strength.
    # Bias the order toward novel/strong (field, op) so sims don't pile onto
    # already-explored structures; rng breaks ties.
    variants.sort(key=lambda v: (v.get("strength", 0.5), rng.random()), reverse=True)
    chosen: list[dict] = []
    chosen_keys: set[str] = set()

    def take(v: dict) -> None:
        if v["expression"] not in chosen_keys:
            chosen.append(v)
            chosen_keys.add(v["expression"])

    for skeleton in skeletons:
        for v in variants:
            if v["skeleton"] == skeleton:
                take(v)
                break
    for field in fields:
        if any(v["binding"].get("field") == field for v in chosen):
            continue
        for v in variants:
            if v["binding"].get("field") == field:
                take(v)
                break
        if len(chosen) >= family_max:
            break
    for v in variants:
        if len(chosen) >= family_max:
            break
        take(v)

    if len(chosen) < family_min:
        return []  # hypothesis space too small to be a designed experiment

    theory_id = card.get("theory_id") or ""
    out: list[dict] = []
    for i, v in enumerate(chosen):
        field_vals = [v["binding"][s] for s in sorted(v["binding"]) if s.startswith("field")]
        primary = field_vals[0] if field_vals else None
        out.append({
            "candidate_id": f"{theory_id}_m{i:02d}",
            "theory_id": theory_id,
            "abandon_group": theory_id,
            "search_lane": "theory",
            "expression": v["expression"],
            "fields": field_vals,
            "cluster_ids": list(card.get("cluster_ids") or []),
            "dataset_id": _binding_dataset(primary, card, clusters) if primary else None,
            "group": v["binding"].get("group"),
            "params": {
                "skeleton": v["skeleton"],
                "binding": {k: val for k, val in v["binding"].items()},
                "direction": card.get("direction", "+"),
            },
            "template_id": f"theory:{v['skeleton']}",
            "cached": v["cached"],
        })
    return out


def _binding_dataset(field: str, card: dict, clusters: dict[str, dict]) -> str | None:
    for cid in card.get("cluster_ids") or []:
        cluster = clusters.get(cid) or {}
        for member in cluster.get("members") or []:
            if str(member.get("field_id")) == field:
                return cluster.get("dataset_id")
    return None


__all__ = [
    "enumerate_family",
    "card_fields",
    "invalid_skeleton_ops",
    "validate_card",
]
