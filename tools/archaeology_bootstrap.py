"""Archaeology bootstrap: seed the mechanism registry from historical sims.

Per cluster (with enough evidence in wiki/structured/cluster_stats.jsonl):
one LLM call (prompts/archaeology_cluster) induces 0-3 retrospective theory
cards plus falsified gists. Retro cards flow through the same embedding dedup
as live proposals (this run doubles as its shakedown test) and are registered
with retro=true — they are PRIOR, not posterior: Beta pseudo-counts stay
capped until a live confirmation family validates them.

Usage:
    python tools/archaeology_bootstrap.py --limit 3      # pilot on 3 clusters
    python tools/archaeology_bootstrap.py                # full run (~150-300 calls)
    python tools/archaeology_bootstrap.py --min-sims 30  # higher evidence bar
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core import llm as anthropic  # noqa: E402
from core.json_utils import parse_json_object  # noqa: E402
from core.prompts import load_prompt  # noqa: E402
from alpha_discovery.theory_method.context import load_clusters  # noqa: E402
from alpha_discovery.theory_method.dedup import MechanismRegistry, card_embed_text, embed_texts  # noqa: E402
from alpha_discovery.theory_method.enumerate_ import slotify_skeletons, validate_card  # noqa: E402
from core.wiki.generate import load_cluster_stats  # noqa: E402

FALSIFIED_GISTS_PATH = Path("artifacts/theory/falsified_gists.jsonl")


def archaeology_one_cluster(
    client: "anthropic.Anthropic",
    prompt,
    cid: str,
    stats_row: dict,
    cluster: dict,
    *,
    model: str,
) -> dict:
    members = "\n".join(
        f"- {m.get('field_id')}: {str(m.get('description') or '')[:110]}"
        for m in (cluster.get("members") or [])[:12]
    )
    def _line(e: dict) -> str:
        return (f"- {'SUBMITTABLE' if e.get('submittable') else 'not-submittable'} "
                f"score={e.get('robustness_score')} sharpe={e.get('is_sharpe')} "
                f"fitness={e.get('is_fitness')} :: {e.get('expression')}")

    best = [_line(e) for e in (stats_row.get("top_expressions") or [])]
    failures = [_line(e) for e in (stats_row.get("failure_examples") or [])]
    exemplars = "\n".join(
        ["## best (ranked submittable > score > fitness)"] + best
        + (["## representative failures"] + failures if failures else [])
    )
    aggregate = json.dumps({k: v for k, v in stats_row.items() if k != "top_expressions"})
    response = client.messages.create(
        model=model,
        max_completion_tokens=2500,
        system=prompt.system,
        messages=[{"role": "user", "content": prompt.render_user(
            cluster_info=json.dumps({
                "cluster_id": cid,
                "label": cluster.get("label"),
                "dataset_id": cluster.get("dataset_id"),
            }),
            member_fields=members or "(unknown fields)",
            aggregate_stats=aggregate,
            exemplars=exemplars or "(no exemplars)",
            cluster_id=cid,
        )}],
    )
    return parse_json_object(response.content[0].text) or {}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--min-sims", type=int, default=20)
    ap.add_argument("--limit", type=int, default=0, help="process at most N clusters (0 = all)")
    ap.add_argument("--model", default="gpt-5.5")
    args = ap.parse_args()

    stats = load_cluster_stats()
    clusters = load_clusters()
    known_clusters = set(clusters)
    eligible = [
        (cid, row) for cid, row in sorted(
            stats.items(), key=lambda t: t[1].get("n_sims", 0), reverse=True)
        if row.get("n_sims", 0) >= args.min_sims and cid in clusters
    ]
    if args.limit:
        eligible = eligible[: args.limit]
    print(f"archaeology over {len(eligible)} clusters (min_sims={args.min_sims})")

    client = anthropic.Anthropic()
    prompt = load_prompt("archaeology_cluster")
    registry = MechanismRegistry()
    done_marker = registry.path.parent / "archaeology_done.json"
    done: set[str] = set()
    if done_marker.exists():
        done = set(json.loads(done_marker.read_text()))

    n_cards = n_rejected = n_gists = 0
    for i, (cid, row) in enumerate(eligible, 1):
        if cid in done:
            continue
        cluster = clusters[cid]
        try:
            parsed = archaeology_one_cluster(client, prompt, cid, row, cluster, model=args.model)
        except Exception as exc:  # keep the sweep going; cluster can be retried
            print(f"  [{i}/{len(eligible)}] {cid[:60]} ERROR: {exc}")
            continue

        cards = [c for c in (parsed.get("retro_cards") or []) if isinstance(c, dict)]
        gists = [str(g) for g in (parsed.get("falsified_gists") or [])]
        valid_cards = []
        for card in cards:
            card["retro"] = True
            card.setdefault("cluster_ids", [cid])
            card = slotify_skeletons(card, clusters)
            problems = validate_card(card, known_clusters=known_clusters)
            if problems:
                n_rejected += 1
                print(f"      [invalid card] {str(card.get('hypothesis'))[:70]} -> {problems}")
                continue
            valid_cards.append(card)

        if valid_cards:
            embeddings = embed_texts([card_embed_text(c) for c in valid_cards])
            for card, emb in zip(valid_cards, embeddings):
                ruling = registry.adjudicate(card, emb, model=args.model)
                if ruling.ruling == "REJECTED":
                    n_rejected += 1
                    continue
                registry.register(card, emb, ruling)
                n_cards += 1

        if gists:
            FALSIFIED_GISTS_PATH.parent.mkdir(parents=True, exist_ok=True)
            with FALSIFIED_GISTS_PATH.open("a") as f:
                for gist in gists:
                    f.write(json.dumps({"cluster_id": cid, "gist": gist}) + "\n")
            n_gists += len(gists)

        done.add(cid)
        done_marker.write_text(json.dumps(sorted(done)))
        registry.save()
        print(f"  [{i}/{len(eligible)}] {cid[:60]}: +{len(valid_cards)} cards, "
              f"{len(gists)} falsified gists")

    print(f"\nregistered retro cards: {n_cards} (rejected {n_rejected}); "
          f"falsified gists: {n_gists}")
    print("Next: regenerate wiki (tools/build_wiki.py) and confirm retro theories "
          "with live families (tools/theory_lane.py) before treating them as VALIDATED.")


if __name__ == "__main__":
    main()
