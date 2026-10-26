"""Theory-lane orchestration.

One run = propose K cards -> dedup against the mechanism registry ->
Thompson-style budget allocation -> enumerate designed families ->
simulate (master-cache hits are free evidence, early-stop after the first
wave) -> family verdicts -> registry/posterior update -> optional single
refine round -> theories.jsonl log -> merge session into master.
"""

from __future__ import annotations

import json
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from wq_alpha.logs import merge_session_to_master, session_log_path
from wq_alpha.robustness import record_robustness_score
from wq_alpha.theory import context as ctx
from wq_alpha.theory.dedup import MechanismRegistry, card_embed_text, embed_texts, lane_lock
from wq_alpha.theory.economist import TheoryEconomistAgent
from wq_alpha.theory.enumerate_ import enumerate_family
from wq_alpha.theory.family import (
    EARLY_STOP_FIRST,
    FamilyVerdict,
    early_stop_after_first,
    evaluate_family,
    load_expression_cache,
    update_beta,
)

THEORY_DIR = Path("artifacts/theory")
THEORIES_LOG = THEORY_DIR / "theories.jsonl"
# Validated, submittable, robust brain2 winners queue here for re-simulation on
# the brain (competition) account, where the self-correlation and score-addition
# gates apply. brain2 itself is a discovery pool ruled only by submittable +
# robustness + fitness — no portfolio gates (different account).
PROMOTE_QUEUE = THEORY_DIR / "promote_to_brain.jsonl"
COMPETITION_ACCOUNT = "brain"


def _account_label(credentials: str) -> str:
    return Path(credentials).stem  # brain2.txt -> brain2


def _load_coverage():
    """Grammar Q-table coverage for the enumerator; None if unavailable."""
    try:
        from wq_alpha.theory.coverage import GrammarCoverage
        return GrammarCoverage.load()
    except Exception as exc:
        print(f"  [coverage] unavailable ({exc}); enumerating without grammar dedup")
        return None

_NOVELTY = {"low": 1.0, "medium": 0.7, "high": 0.4}


def _novelty(card: dict) -> float:
    estimate = str(card.get("crowding_estimate") or "medium").split()[0].lower()
    for key, value in _NOVELTY.items():
        if estimate.startswith(key):
            return value
    return 0.7


def _thompson_rank(cards: list[dict], rng: random.Random) -> list[dict]:
    def sample(card: dict) -> float:
        a = float(card.get("beta_alpha") or 1.0)
        b = float(card.get("beta_beta") or 1.0)
        return rng.betavariate(a, b) * _novelty(card)

    return sorted(cards, key=sample, reverse=True)


def select_batch_clusters(
    clusters: dict[str, dict],
    metadata: dict[str, dict],
    *,
    dataset: str | None = None,
    n_clusters: int = 20,
    rng: random.Random | None = None,
) -> list[str]:
    """Candidate clusters for one proposal batch, biased toward low crowding."""
    rng = rng or random.Random()
    pool = [
        (cid, ctx.cluster_crowding(c, metadata)["userCount"])
        for cid, c in clusters.items()
        if dataset is None or c.get("dataset_id") == dataset
    ]
    if not pool:
        return []
    pool.sort(key=lambda t: t[1])
    # 70% from the least-crowded half, 30% sampled from the rest
    half = max(1, len(pool) // 2)
    low = [cid for cid, _ in pool[:half]]
    high = [cid for cid, _ in pool[half:]]
    n_low = min(len(low), max(1, int(n_clusters * 0.7)))
    n_high = min(len(high), n_clusters - n_low)
    return rng.sample(low, n_low) + (rng.sample(high, n_high) if n_high > 0 else [])


def run_theory_lane(
    *,
    dataset: str | None = None,
    n_theories: int = 5,
    family_max: int = 15,
    max_theories_to_simulate: int | None = None,
    refine: bool = False,
    dry_run: bool = False,
    credentials: str = "brain.txt",
    batch_size: int = 3,
    model: str = "gpt-5.5",
    seed: int | None = None,
    rounds: int = 1,
) -> list[FamilyVerdict]:
    """Run the theory lane for `rounds` propose→simulate cycles.

    rounds <= 0 means run perpetually until interrupted — the intended mode for
    the brain2 discovery account, whose headless auth makes the process freely
    restartable. For the brain (competition) account, pass a bounded `rounds`
    so the single face-scan session covers a fixed amount of work.

    Each round draws a fresh random cluster batch; the mechanism registry and
    wiki compound across rounds (later rounds avoid explored mechanisms). One
    sign-in is reused across rounds. Everything is persisted PER ROUND — a kill
    at any point loses at most the in-progress round.
    """
    import itertools

    from wq_alpha.brain.submit import DEFAULT_SETTINGS
    from wq_alpha.workshop.core import DEFAULT_WORKSHOP_SETTINGS

    rng = random.Random(seed)
    clusters = ctx.load_clusters()
    metadata = ctx.load_field_metadata()
    account = _account_label(credentials)
    registry = MechanismRegistry(account=account)
    agent = TheoryEconomistAgent(model=model)
    # submit_candidate_batch sends settings straight to the API, so the dict
    # must carry the full required field set (instrumentType, region, language,
    # pasteurization, unitHandling, nanHandling) — not just the workshop axes.
    sim_settings = {**DEFAULT_SETTINGS, **DEFAULT_WORKSHOP_SETTINGS}
    sim_log = session_log_path("theory")

    session = None
    all_verdicts: list[FamilyVerdict] = []
    perpetual = rounds <= 0
    counter = itertools.count() if perpetual else range(rounds)

    for round_idx in counter:
        label = f"{round_idx + 1}/{'∞' if perpetual else rounds}"
        print(f"\n===== theory round {label} (account={account}) =====")
        verdicts, promotions, session = _run_round(
            clusters=clusters, metadata=metadata, registry=registry, agent=agent,
            account=account, sim_settings=sim_settings, sim_log=sim_log, session=session,
            rng=rng, dataset=dataset, n_theories=n_theories, family_max=family_max,
            max_theories_to_simulate=max_theories_to_simulate, refine=refine,
            dry_run=dry_run, credentials=credentials, batch_size=batch_size, model=model,
        )
        all_verdicts.extend(verdicts)
        if dry_run:
            continue
        # Commit shared state (registry + master + wiki) atomically so a
        # concurrently-running account's commit cannot interleave or clobber.
        with lane_lock():
            registry.save()
            _finalize_round(promotions, sim_log)

    return all_verdicts


def _finalize_round(promotions: list[dict], sim_log) -> None:
    """Persist a round's output: promotion queue + master merge + wiki refresh.

    Runs every round so a perpetual brain2 run never accumulates unpersisted
    state. sim_log is written incrementally during the round; this merges the
    cumulative session into master and refreshes the (deterministic) wiki."""
    if promotions:
        THEORY_DIR.mkdir(parents=True, exist_ok=True)
        with PROMOTE_QUEUE.open("a") as f:
            for row in promotions:
                row["queued_at"] = datetime.now(timezone.utc).isoformat()
                f.write(json.dumps(row) + "\n")
        print(f"[promote] {len(promotions)} submittable winner(s) queued for brain "
              f"re-sim -> {PROMOTE_QUEUE} (run: python tools/promote_to_brain.py)")
    if sim_log.exists():
        merge_session_to_master(sim_log)
        try:
            from wq_alpha.wiki.generate import generate_wiki
            generate_wiki(use_llm=False)
        except Exception as exc:
            print(f"[wiki] refresh skipped: {exc}")


def _run_round(
    *,
    clusters: dict, metadata: dict, registry: MechanismRegistry,
    agent: TheoryEconomistAgent, account: str, sim_settings: dict, sim_log,
    session, rng: random.Random, dataset: str | None, n_theories: int,
    family_max: int, max_theories_to_simulate: int | None, refine: bool,
    dry_run: bool, credentials: str, batch_size: int, model: str,
) -> tuple[list[FamilyVerdict], list[dict], object]:
    """One propose→dedup→enumerate→simulate→verdict cycle. Returns
    (verdicts, promotions, session); signs in lazily and returns the session
    so the caller reuses it across rounds."""
    from wq_alpha.brain.api import sign_in
    from wq_alpha.workshop.simulation import submit_candidate_batch

    # Pick up any mechanisms the other account committed since last round, so
    # the proposer context and dedup avoid re-exploring them.
    with lane_lock():
        registry.reload()

    batch_cluster_ids = select_batch_clusters(
        clusters, metadata, dataset=dataset, rng=rng
    )
    if not batch_cluster_ids:
        print(f"No clusters found (dataset={dataset!r}).")
        return [], [], session

    coverage = _load_coverage()  # grammar Q-table learning (shared by propose + enumerate)
    saturated = coverage.saturated_summary(batch_cluster_ids) if coverage else "(none)"

    print(f"[propose] {len(batch_cluster_ids)} candidate clusters; asking for {n_theories} cards ...")
    result = agent.propose(
        cluster_catalog=ctx.render_cluster_catalog(batch_cluster_ids, clusters, metadata),
        cluster_index=ctx.render_cluster_index(clusters, metadata),
        operator_menu=ctx.render_operator_menu(),
        wiki_excerpts=ctx.wiki_excerpts(batch_cluster_ids),
        mechanism_index=registry.index_lines(),
        falsified_index=registry.falsified_lines(),
        n_theories=n_theories,
        known_clusters=set(clusters),
        clusters=clusters,
        saturated_structures=saturated,
    )
    for card, problems in result.rejected:
        print(f"  [reject] {str(card.get('hypothesis'))[:80]} -> {problems}")
    if not result.cards:
        if result.truncated:
            print(f"No valid cards proposed: LLM response was TRUNCATED "
                  f"({len(result.raw_response)} chars, stop_reason=max_tokens). "
                  f"Lower --n-theories or raise the economist token cap.")
        elif not result.raw_response.strip():
            print("No valid cards proposed: LLM returned an empty response.")
        else:
            print(f"No valid cards proposed: response did not parse as a JSON "
                  f"array of cards ({len(result.raw_response)} chars). Snippet:")
            print("  " + result.raw_response[:400].replace("\n", "\n  "))
        return [], [], session

    # ── dedup ───────────────────────────────────────────────────────────
    embeddings = embed_texts([card_embed_text(c) for c in result.cards])
    accepted: list[dict] = []
    for card, emb in zip(result.cards, embeddings):
        ruling = registry.adjudicate(card, emb, model=model)
        print(f"  [dedup] {ruling.ruling:8} cos={ruling.max_cosine:.2f} "
              f"{str(card.get('hypothesis'))[:80]}")
        if ruling.ruling == "REJECTED":
            continue
        canonical = registry.register(card, emb, ruling)
        if ruling.ruling != "SAME" or canonical.get("verdict") not in ("VALIDATED", "FALSIFIED"):
            accepted.append(canonical)

    if not accepted:
        print("All cards deduped away.")
        return [], [], session

    ranked = _thompson_rank(accepted, rng)
    if max_theories_to_simulate:
        ranked = ranked[:max_theories_to_simulate]

    # ── enumerate ────────────────────────────────────────────────────────
    cache = load_expression_cache()
    families: list[tuple[dict, list[dict]]] = []
    for card in ranked:
        members = enumerate_family(
            card, clusters, family_max=family_max,
            known_expressions=set(cache), coverage=coverage,
            seed=rng.randrange(1 << 30),
        )
        n_cached = sum(1 for m in members if m.get("cached"))
        print(f"  [family] {card['theory_id']}: {len(members)} members "
              f"({n_cached} cached) — {str(card.get('hypothesis'))[:70]}")
        if members:
            families.append((card, members))

    if dry_run:
        print("\ndry-run: stopping before simulation. Cards registered, no posteriors updated.")
        for card, members in families:
            print(f"\n{card['theory_id']} — {card.get('hypothesis')}")
            for m in members:
                tag = " [cached]" if m.get("cached") else ""
                print(f"   {m['expression'][:130]}{tag}")
        return [], [], session

    # ── simulate + verdicts ─────────────────────────────────────────────
    if session is None:
        session = sign_in(credentials)
    verdicts: list[FamilyVerdict] = []
    promotions: list[dict] = []

    for card, members in families:
        direction = str(card.get("direction") or "+")
        cached_records = [dict(cache[m["expression"]]) for m in members if m.get("cached")]
        to_sim = [m for m in members if not m.get("cached")]

        family_records = list(cached_records)
        first, rest = to_sim[:EARLY_STOP_FIRST], to_sim[EARLY_STOP_FIRST:]
        for wave_idx, wave in enumerate((first, rest)):
            if not wave:
                continue
            if wave_idx == 1 and early_stop_after_first(family_records, direction):
                print(f"  [early-stop] {card['theory_id']}: first wave 0 supporters, "
                      f"skipping {len(rest)} members")
                break
            new_records = submit_candidate_batch(
                wave, session, sim_log, dict(sim_settings),
                credentials, batch_size, yearly_stats_attempts=4,
            )
            family_records.extend(new_records or [])

        verdict = evaluate_family(card["theory_id"], family_records, direction=direction)
        update_beta(card, verdict)
        card["verdict"] = (
            verdict.verdict if verdict.verdict != "INCONCLUSIVE" else card.get("verdict")
        )
        card["last_support_rate"] = round(verdict.support_rate, 3)
        if verdict.verdict == "FALSIFIED" and not card.get("falsification_reason"):
            card["falsification_reason"] = (
                f"family evidence {verdict.n_support}/{verdict.n_done} supporters"
            )
        if card.get("retro") and verdict.verdict == "VALIDATED":
            card["live_confirmed"] = True
        verdicts.append(verdict)
        sub_promoted = verdict.submittable_promoted()
        print(f"  [verdict] {card['theory_id']}: {verdict.verdict} "
              f"({verdict.n_support}/{verdict.n_done}), "
              f"promoted={len(verdict.promoted)} (submittable={len(sub_promoted)})")
        # Queue submittable+robust winners for re-simulation on the competition
        # account. brain2 is a separate pool, so only its winners are promoted;
        # a run already on brain needs no promotion.
        if verdict.verdict == "VALIDATED" and account != COMPETITION_ACCOUNT:
            for rec in sub_promoted:
                promotions.append({
                    "theory_id": card["theory_id"],
                    "hypothesis": card.get("hypothesis"),
                    "direction": direction,
                    "expression": rec.get("expression"),
                    "source_account": account,
                    "is_sharpe": rec.get("is_sharpe"),
                    "is_fitness": rec.get("is_fitness"),
                    "robustness_score": record_robustness_score(rec),
                    "cluster_ids": card.get("cluster_ids"),
                })

        if refine and verdict.verdict == "INCONCLUSIVE":
            revised = agent.refine(card, family_records, verdict.verdict,
                                   known_clusters=set(clusters))
            if revised is not None:
                print(f"  [refine] {card['theory_id']}: one refined family")
                revised_members = enumerate_family(
                    revised, clusters, family_max=family_max,
                    known_expressions=set(cache), coverage=coverage,
                    seed=rng.randrange(1 << 30),
                )
                to_sim = [m for m in revised_members if not m.get("cached")]
                refined_records = [dict(cache[m["expression"]]) for m in revised_members if m.get("cached")]
                refined_records.extend(submit_candidate_batch(
                    to_sim, session, sim_log, dict(sim_settings),
                    credentials, batch_size, yearly_stats_attempts=4,
                ) or [])
                rev_verdict = evaluate_family(card["theory_id"], refined_records, direction=direction)
                update_beta(card, rev_verdict)
                card["verdict"] = (
                    rev_verdict.verdict if rev_verdict.verdict != "INCONCLUSIVE" else card.get("verdict")
                )
                verdicts.append(rev_verdict)

        _append_theory_log(card, verdict)

    return verdicts, promotions, session


def _append_theory_log(card: dict, verdict: FamilyVerdict) -> None:
    THEORY_DIR.mkdir(parents=True, exist_ok=True)
    entry = {
        "logged_at": datetime.now(timezone.utc).isoformat(),
        "theory_id": card.get("theory_id"),
        "hypothesis": card.get("hypothesis"),
        "why_it_exists": card.get("why_it_exists"),
        "cluster_ids": card.get("cluster_ids"),
        "direction": card.get("direction"),
        "horizon_band": card.get("horizon_band"),
        "crowding_estimate": card.get("crowding_estimate"),
        "retro": bool(card.get("retro")),
        "beta_alpha": card.get("beta_alpha"),
        "beta_beta": card.get("beta_beta"),
        **verdict.as_dict(),
    }
    with THEORIES_LOG.open("a") as f:
        f.write(json.dumps(entry) + "\n")


__all__ = ["run_theory_lane", "select_batch_clusters"]
