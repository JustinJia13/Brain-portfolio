"""Wiki generation: structured aggregation, narrative pages, prior export.

Structured layer (deterministic, zero LLM cost):
    wiki/structured/cluster_stats.jsonl   per-cluster evidence aggregates

Narrative layer (LLM, incremental — only pages whose input hash changed):
    wiki/clusters/<cluster_id>.md
    wiki/mechanisms.md (registry view, deterministic)
    wiki/index.md      (nav + summary, deterministic)

Prior export:
    artifacts/probe/grammar_prior_from_wiki.json in the exact cluster_priors
    schema consumed by ProbePriorGrammarSampler.from_json — this revives the
    grammar lane's probe-prior input.
"""

from __future__ import annotations

import hashlib
import json
import re
import statistics
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from wq_alpha.quality import PASS_FITNESS, PASS_SHARPE, PASS_TURNOVER, is_submittable, quality_rank_key
from wq_alpha.robustness import record_robustness_score

ROOT = Path(__file__).resolve().parents[2]
WIKI_DIR = ROOT / "wiki"
STRUCTURED_DIR = WIKI_DIR / "structured"
CLUSTER_STATS_PATH = STRUCTURED_DIR / "cluster_stats.jsonl"
STATE_PATH = WIKI_DIR / ".state.json"
PRIOR_EXPORT_PATH = ROOT / "artifacts" / "probe" / "grammar_prior_from_wiki.json"

MASTER_PATH = ROOT / "artifacts" / "sim" / "master.jsonl"
MECHANISMS_PATH = ROOT / "artifacts" / "theory" / "mechanisms.jsonl"

_OP_PATTERN = re.compile(r"([a-zA-Z_][a-zA-Z0-9_]*)\s*\(")


def _page_name(cluster_id: str) -> str:
    name = cluster_id.replace(":", "_").replace("/", "_")
    if len(name) > 100:
        digest = hashlib.sha256(cluster_id.encode()).hexdigest()[:10]
        name = f"{name[:88]}_{digest}"
    return name


def _field_to_cluster(clusters: dict[str, dict]) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for cid, cluster in clusters.items():
        for member in cluster.get("members") or []:
            fid = str(member.get("field_id") or "")
            if fid and fid not in mapping:
                mapping[fid] = cid
    return mapping


def _record_clusters(rec: dict, field_map: dict[str, str]) -> set[str]:
    found: set[str] = set()
    for cid in rec.get("cluster_ids") or []:
        found.add(str(cid))
    if not found:
        for fid in rec.get("fields") or []:
            cid = field_map.get(str(fid))
            if cid:
                found.add(cid)
    return found


# ---------------------------------------------------------------------------
# Structured layer
# ---------------------------------------------------------------------------

def build_cluster_stats(
    master_path: Path = MASTER_PATH,
    *,
    clusters: dict[str, dict] | None = None,
) -> dict[str, dict]:
    """Aggregate all master records per cluster -> cluster_stats rows."""
    from wq_alpha.theory.context import load_clusters

    clusters = clusters or load_clusters()
    field_map = _field_to_cluster(clusters)
    acc: dict[str, dict] = {}

    if not master_path.exists():
        return {}
    with master_path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if rec.get("status") != "done" or rec.get("is_sharpe") is None:
                continue
            expr = str(rec.get("expression") or "")
            cids = _record_clusters(rec, field_map)
            if not cids:
                continue
            sharpe = float(rec.get("is_sharpe") or 0.0)
            fitness = float(rec.get("is_fitness") or 0.0)
            passing = abs(sharpe) >= PASS_SHARPE and abs(fitness) >= PASS_FITNESS
            submittable = is_submittable(rec)
            rob = record_robustness_score(rec)
            ops = set(_OP_PATTERN.findall(expr))
            for cid in cids:
                row = acc.setdefault(cid, {
                    "cluster_id": cid,
                    "n_sims": 0, "n_passing": 0, "n_submittable": 0,
                    "sharpes": [], "robust_scores": [],
                    "op_counter": Counter(), "op_pass_counter": Counter(),
                    "pos": 0, "neg": 0,
                    "top": [],
                })
                row["n_sims"] += 1
                row["n_passing"] += int(passing)
                row["n_submittable"] += int(submittable)
                row["sharpes"].append(sharpe)
                if rob is not None:
                    row["robust_scores"].append(rob)
                row["op_counter"].update(ops)
                if submittable:
                    row["op_pass_counter"].update(ops)
                if sharpe > 0.5:
                    row["pos"] += 1
                elif sharpe < -0.5:
                    row["neg"] += 1
                entry = {
                    "expression": expr[:200],
                    "submittable": submittable,
                    "is_sharpe": sharpe,
                    "is_fitness": fitness,
                    "robustness_score": rob,
                }
                row["top"].append((quality_rank_key(rec), entry))
                if not submittable:
                    # candidates for failure exemplars: dead signals and
                    # spiky check-failers are both falsification evidence
                    row.setdefault("failures", []).append((abs(sharpe), fitness, entry))

    out: dict[str, dict] = {}
    for cid, row in acc.items():
        sharpes = row["sharpes"]
        top = sorted(row["top"], key=lambda t: t[0], reverse=True)[:10]
        # representative failures: 3 dead signals (lowest |sharpe|) plus 2
        # spiky check-failers (highest fitness among non-submittable)
        fails = row.get("failures") or []
        dead = [e for _, _, e in sorted(fails, key=lambda t: t[0])[:3]]
        spiky = [e for _, _, e in sorted(fails, key=lambda t: t[1], reverse=True)[:2]
                 if e not in dead]
        out[cid] = {
            "cluster_id": cid,
            "n_sims": row["n_sims"],
            "n_passing": row["n_passing"],
            "n_submittable": row["n_submittable"],
            "pass_rate": round(row["n_passing"] / row["n_sims"], 4),
            "submittable_rate": round(row["n_submittable"] / row["n_sims"], 4),
            "mean_abs_sharpe": round(statistics.mean(abs(s) for s in sharpes), 3),
            "max_abs_sharpe": round(max(abs(s) for s in sharpes), 3),
            "direction_balance": {"pos": row["pos"], "neg": row["neg"]},
            "n_with_robustness": len(row["robust_scores"]),
            "median_robustness": (
                round(statistics.median(row["robust_scores"]), 3) if row["robust_scores"] else None
            ),
            "top_ops": [op for op, _ in row["op_counter"].most_common(8)],
            "top_passing_ops": [op for op, _ in row["op_pass_counter"].most_common(5)],
            "top_expressions": [t[1] for t in top],
            "failure_examples": dead + spiky,
        }

    STRUCTURED_DIR.mkdir(parents=True, exist_ok=True)
    with CLUSTER_STATS_PATH.open("w") as f:
        for cid in sorted(out):
            f.write(json.dumps(out[cid]) + "\n")
    return out


def load_cluster_stats() -> dict[str, dict]:
    out: dict[str, dict] = {}
    if not CLUSTER_STATS_PATH.exists():
        return out
    with CLUSTER_STATS_PATH.open() as f:
        for line in f:
            line = line.strip()
            if line:
                row = json.loads(line)
                out[row["cluster_id"]] = row
    return out


def _load_mechanisms() -> list[dict]:
    cards: list[dict] = []
    if MECHANISMS_PATH.exists():
        with MECHANISMS_PATH.open() as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        cards.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
    return cards


# ---------------------------------------------------------------------------
# Narrative layer
# ---------------------------------------------------------------------------

def _input_hash(*parts: str) -> str:
    h = hashlib.sha256()
    for part in parts:
        h.update(part.encode())
    return h.hexdigest()[:16]


def _load_state() -> dict:
    if STATE_PATH.exists():
        try:
            return json.loads(STATE_PATH.read_text())
        except json.JSONDecodeError:
            return {}
    return {}


def _save_state(state: dict) -> None:
    WIKI_DIR.mkdir(parents=True, exist_ok=True)
    STATE_PATH.write_text(json.dumps(state, indent=1))


def _cluster_mechanisms(cid: str, mechanisms: list[dict]) -> list[dict]:
    return [m for m in mechanisms if cid in (m.get("cluster_ids") or [])]


def generate_cluster_page(
    cid: str,
    stats_row: dict,
    mechanisms: list[dict],
    *,
    clusters: dict[str, dict],
    llm_client=None,
    model: str = "gpt-5.5",
) -> str:
    """Narrative page; falls back to a deterministic page when no LLM client."""
    cluster = clusters.get(cid) or {}
    entries = _cluster_mechanisms(cid, mechanisms)
    if llm_client is not None:
        from wq_alpha.prompts import load_prompt

        prompt = load_prompt("wiki_narrative")
        response = llm_client.messages.create(
            model=model,
            max_completion_tokens=1500,
            system=prompt.system,
            messages=[{"role": "user", "content": prompt.render_user(
                cluster_info=json.dumps({
                    "cluster_id": cid,
                    "label": cluster.get("label"),
                    "dataset_id": cluster.get("dataset_id"),
                    "fields": [m.get("field_id") for m in cluster.get("members") or []],
                }),
                cluster_stats=json.dumps({k: v for k, v in stats_row.items() if k != "top_expressions"}),
                mechanism_entries=json.dumps([{
                    "theory_id": m.get("theory_id"),
                    "hypothesis": m.get("hypothesis"),
                    "verdict": m.get("verdict") or "UNTESTED",
                    "last_support_rate": m.get("last_support_rate"),
                    "falsification_reason": m.get("falsification_reason"),
                    "retro": m.get("retro", False),
                } for m in entries]),
                top_expressions=json.dumps(stats_row.get("top_expressions") or []),
            )}],
        )
        body = response.content[0].text.strip()
    else:
        body = _deterministic_page(cid, cluster, stats_row, entries)
    header = (
        f"# {cid}\n\n"
        f"_dataset: {cluster.get('dataset_id')} — {cluster.get('label')}; "
        f"generated {datetime.now(timezone.utc).date()}_\n\n"
    )
    return header + body + "\n"


def _deterministic_page(cid: str, cluster: dict, stats: dict, entries: list[dict]) -> str:
    validated = [m for m in entries if m.get("verdict") == "VALIDATED"]
    falsified = [m for m in entries if m.get("verdict") == "FALSIFIED"]
    other = [m for m in entries if m.get("verdict") not in ("VALIDATED", "FALSIFIED")]
    lines = [
        "## Overview",
        f"Fields: {', '.join(str(m.get('field_id')) for m in (cluster.get('members') or [])[:10])}",
        "",
        "## Evidence summary",
        f"- sims: {stats.get('n_sims')} | submittable: {stats.get('n_submittable')} "
        f"(rate {stats.get('submittable_rate')}) | threshold-passing: {stats.get('n_passing')}",
        f"- mean |sharpe|: {stats.get('mean_abs_sharpe')} | max: {stats.get('max_abs_sharpe')}",
        f"- direction balance: {stats.get('direction_balance')}",
        f"- median robustness: {stats.get('median_robustness')} "
        f"(n={stats.get('n_with_robustness')})",
        f"- top submittable ops: {', '.join(stats.get('top_passing_ops') or []) or 'none'}",
        "",
        "## What works",
    ]
    for m in validated:
        lines.append(f"- ({m.get('theory_id')}) {m.get('hypothesis')} — "
                     f"support {m.get('last_support_rate')}")
    if not validated:
        lines.append("- nothing validated yet")
    lines += ["", "## What failed"]
    for m in falsified:
        lines.append(f"- ({m.get('theory_id')}) {m.get('hypothesis')} — "
                     f"{m.get('falsification_reason') or 'family evidence'}")
    if not falsified:
        lines.append("- nothing falsified yet")
    lines += ["", "## Open questions"]
    for m in other:
        lines.append(f"- ({m.get('theory_id')}) {m.get('hypothesis')} [untested/inconclusive]")
    if not other:
        lines.append("- propose theories for this cluster")
    return "\n".join(lines)


def generate_wiki(
    *,
    use_llm: bool = False,
    model: str = "gpt-5.5",
    min_sims: int = 10,
    rebuild_stats: bool = True,
) -> dict:
    """Regenerate the wiki incrementally. Returns counters."""
    from wq_alpha.theory.context import load_clusters

    clusters = load_clusters()
    stats = build_cluster_stats(clusters=clusters) if rebuild_stats else load_cluster_stats()
    mechanisms = _load_mechanisms()
    state = _load_state()
    pages = state.setdefault("pages", {})

    llm_client = None
    if use_llm:
        from wq_alpha import llm as anthropic

        llm_client = anthropic.Anthropic()

    mode = "llm" if llm_client is not None else "det"
    eligible = [(cid, row) for cid, row in sorted(stats.items())
                if row.get("n_sims", 0) >= min_sims]
    written = skipped = 0
    (WIKI_DIR / "clusters").mkdir(parents=True, exist_ok=True)
    for idx, (cid, row) in enumerate(eligible, 1):
        entries = _cluster_mechanisms(cid, mechanisms)
        digest = _input_hash(json.dumps(row, sort_keys=True),
                             json.dumps(entries, sort_keys=True, default=str))
        page_path = WIKI_DIR / "clusters" / f"{_page_name(cid)}.md"
        prev = pages.get(cid)
        if isinstance(prev, str):  # legacy state format: digest only
            prev = {"digest": prev, "mode": "det"}
        if prev and prev.get("digest") == digest and page_path.exists():
            # up to date for these inputs; a deterministic run must never
            # downgrade an existing LLM page, and an LLM run upgrades
            # deterministic pages even when inputs are unchanged
            if mode == "det" or prev.get("mode") == "llm":
                skipped += 1
                continue
        page_path.write_text(generate_cluster_page(
            cid, row, mechanisms, clusters=clusters, llm_client=llm_client, model=model))
        pages[cid] = {"digest": digest, "mode": mode}
        written += 1
        _save_state(state)  # incremental: Ctrl+C keeps all finished pages
        if llm_client is not None:
            print(f"  [{idx}/{len(eligible)}] {cid[:70]}", flush=True)

    _write_mechanisms_page(mechanisms)
    _write_index(stats, mechanisms, min_sims=min_sims)
    _save_state(state)
    return {"clusters_with_stats": len(stats), "pages_written": written, "pages_skipped": skipped}


def _write_mechanisms_page(mechanisms: list[dict]) -> None:
    groups: dict[str, list[dict]] = {"VALIDATED": [], "FALSIFIED": [], "OTHER": []}
    for m in mechanisms:
        groups.get(m.get("verdict") if m.get("verdict") in ("VALIDATED", "FALSIFIED") else "OTHER",
                   groups["OTHER"]).append(m)
    lines = ["# Mechanism registry", ""]
    for title, key in (("Validated", "VALIDATED"), ("Falsified", "FALSIFIED"), ("Untested / inconclusive", "OTHER")):
        lines += [f"## {title}", ""]
        for m in groups[key]:
            retro = " [retro]" if m.get("retro") and not m.get("live_confirmed") else ""
            lines.append(
                f"- **{m.get('theory_id')}**{retro}: {m.get('hypothesis')} "
                f"(clusters: {', '.join(m.get('cluster_ids') or [])}; "
                f"Beta({m.get('beta_alpha')}, {m.get('beta_beta')}); "
                f"support {m.get('last_support_rate', 'n/a')})"
            )
            if key == "FALSIFIED" and m.get("falsification_reason"):
                lines.append(f"  - why: {m['falsification_reason']}")
        lines.append("")
    WIKI_DIR.mkdir(parents=True, exist_ok=True)
    (WIKI_DIR / "mechanisms.md").write_text("\n".join(lines))


def _write_index(stats: dict[str, dict], mechanisms: list[dict], *, min_sims: int) -> None:
    n_validated = sum(1 for m in mechanisms if m.get("verdict") == "VALIDATED")
    n_falsified = sum(1 for m in mechanisms if m.get("verdict") == "FALSIFIED")
    by_evidence = sorted(stats.values(), key=lambda r: r.get("n_sims", 0), reverse=True)
    lines = [
        "# Research wiki",
        "",
        f"_regenerated {datetime.now(timezone.utc).isoformat()}_",
        "",
        f"- mechanisms: {len(mechanisms)} ({n_validated} validated, {n_falsified} falsified)"
        " — see [mechanisms.md](mechanisms.md)",
        f"- clusters with evidence: {len(stats)} (pages for those with >= {min_sims} sims)",
        "",
        "## Clusters by evidence",
        "",
    ]
    for row in by_evidence[:60]:
        cid = row["cluster_id"]
        lines.append(
            f"- [{cid}](clusters/{_page_name(cid)}.md) — sims {row['n_sims']}, "
            f"submittable {row.get('n_submittable', 0)} "
            f"(rate {row.get('submittable_rate', 0.0)}), "
            f"median robustness {row['median_robustness']}"
        )
    (WIKI_DIR / "index.md").write_text("\n".join(lines) + "\n")


# ---------------------------------------------------------------------------
# Prior export (contract: ProbePriorGrammarSampler.from_json)
# ---------------------------------------------------------------------------

def export_priors(out_path: Path = PRIOR_EXPORT_PATH) -> int:
    """Emit cluster_priors rows from the structured layer for the grammar lane."""
    from wq_alpha.theory.context import load_clusters

    clusters = load_clusters()
    stats = load_cluster_stats()
    mechanisms = _load_mechanisms()
    rows: list[dict] = []
    for cid, row in stats.items():
        cluster = clusters.get(cid) or {}
        field = str(cluster.get("champion_field_id") or "")
        if not field:
            continue
        validated = [m for m in _cluster_mechanisms(cid, mechanisms)
                     if m.get("verdict") == "VALIDATED"]
        med_rob = row.get("median_robustness") or 0.0
        # submittable-first quality ordering: submittable rate is the primary
        # evidence signal, falling back to threshold pass rate for old records
        # that carry no checks data
        quality_rate = row.get("submittable_rate") or row.get("pass_rate") or 0.0
        prob = min(1.0, 0.5 * quality_rate + 0.3 * med_rob + 0.2 * min(1.0, len(validated)))
        if prob <= 0.0:
            continue
        balance = row.get("direction_balance") or {}
        direction = "+" if (balance.get("pos") or 0) >= (balance.get("neg") or 0) else "-"
        top_ops = row.get("top_passing_ops") or row.get("top_ops") or []
        rows.append({
            "cluster_id": cid,
            "field": field,
            "raw_probe_probability": round(prob, 4),
            "mixed_probability": round(prob, 4),
            "probe_family": top_ops[0] if top_ops else "",
            "probe_template": (row.get("top_expressions") or [{}])[0].get("expression", ""),
            "preferred_direction": direction,
            "prior_score": round(med_rob, 4),
            "probe_quality": round(quality_rate, 4),
        })
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source": "wiki",
        "cluster_priors": rows,
    }, indent=1))
    return len(rows)


__all__ = ["build_cluster_stats", "export_priors", "generate_wiki", "load_cluster_stats"]
