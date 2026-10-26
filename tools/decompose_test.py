"""Dry test: decompose proven robust alphas and expand them into decorrelated cards.

No simulation. For each seed it: (1) asks the repurposed economist to decompose the
idea and propose expansion cards, (2) renders the cards via the real enumerator,
(3) scores each rendered expression for NOVELTY against the 85k-sim history and for
whether it actually moved OFF the seed's field/dataset (the decorrelation proxy).

    python tools/decompose_test.py --n-seeds 5 --n-expansions 4
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from wq_alpha.field_catalogs import derive_fields
from wq_alpha.quality import is_submittable
from wq_alpha.robustness import compute_robustness, normalize_yearly_stats
from wq_alpha.theory import context as ctx
from wq_alpha.theory.economist import TheoryEconomistAgent
from wq_alpha.theory.enumerate_ import enumerate_family

MASTER = Path("artifacts/sim/master.jsonl")
OUT = Path("artifacts/theory/decompose_test.jsonl")
NUM = re.compile(r"-?\d+\.?\d*")


def skeleton(expr: str, flds: list[str]) -> str:
    s = expr
    for f in sorted(flds, key=len, reverse=True):
        s = s.replace(f, "F")
    return re.sub(r"\s+", "", NUM.sub("N", s))


def build_history_index() -> tuple[set, dict]:
    """{(skeleton, field_set)} seen, and {field_set: best_sharpe} for dead-zone hints."""
    cells: set = set()
    best_by_fields: dict = defaultdict(lambda: -9.0)
    for line in MASTER.open():
        try:
            r = json.loads(line)
        except Exception:
            continue
        expr = r.get("expression")
        if not expr:
            continue
        flds = r.get("fields") or derive_fields(expr)
        if not flds:
            continue
        fs = frozenset(flds)
        cells.add((skeleton(expr, flds), fs))
        s = r.get("is_sharpe") or -9.0
        if s > best_by_fields[fs]:
            best_by_fields[fs] = s
    return cells, best_by_fields


def pick_seeds(n: int) -> list[dict]:
    seeds: list[dict] = []
    for line in MASTER.open():
        try:
            r = json.loads(line)
        except Exception:
            continue
        if not is_submittable(r):
            continue
        ys = r.get("yearly_stats")
        if not ys:
            continue
        try:
            rs = compute_robustness(normalize_yearly_stats(ys))
        except Exception:
            continue
        if rs.score < 0.5:
            continue
        expr = r.get("expression")
        flds = r.get("fields") or derive_fields(expr)
        seeds.append({
            "rob": rs.score, "sharpe": r.get("is_sharpe"), "fitness": r.get("is_fitness"),
            "dataset": r.get("dataset_id"), "expr": expr, "fields": flds,
            "skel": skeleton(expr, flds),
        })
    # diversify: best (sharpe) per distinct (skeleton, primary-field) so we don't
    # seed 20 window-sweep copies of the same idea.
    by_sig: dict = {}
    for s in seeds:
        sig = (s["skel"], s["fields"][0] if s["fields"] else "")
        cur = by_sig.get(sig)
        if cur is None or (s["sharpe"] or 0) > (cur["sharpe"] or 0):
            by_sig[sig] = s
    distinct = sorted(by_sig.values(), key=lambda s: (s["sharpe"] or 0), reverse=True)
    # then spread across distinct primary fields / datasets
    picked: list[dict] = []
    seen_field = set()
    for s in distinct:
        pf = s["fields"][0] if s["fields"] else ""
        if pf in seen_field:
            continue
        seen_field.add(pf)
        picked.append(s)
        if len(picked) >= n:
            break
    return picked


def neglected_menu(clusters: dict, metadata: dict, seed_clusters: list[str], k: int = 16) -> list[str]:
    """Dataset-stratified low-userCount cluster menu (+ the seed's own clusters)."""
    rows = []
    for cid, c in clusters.items():
        uc = ctx.cluster_crowding(c, metadata)["userCount"]
        rows.append((uc, c.get("dataset_id"), cid))
    rows.sort(key=lambda r: (r[0], r[2]))
    out: list[str] = list(seed_clusters)
    per_ds: dict = defaultdict(int)
    for uc, ds, cid in rows:
        if cid in out:
            continue
        if per_ds[ds] >= 2:  # cap any dataset -> stratified spread
            continue
        per_ds[ds] += 1
        out.append(cid)
        if len(out) >= k + len(seed_clusters):
            break
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-seeds", type=int, default=5)
    ap.add_argument("--n-expansions", type=int, default=4)
    ap.add_argument("--family-max", type=int, default=8)
    ap.add_argument("--model", default="gpt-5.5")
    ap.add_argument("--simulate", action="store_true",
                    help="actually simulate the generated expressions on BRAIN")
    ap.add_argument("--credentials", default="brain2.txt",
                    help="brain2.txt = headless discovery pool; brain.txt = competition (face-scan)")
    ap.add_argument("--max-sims", type=int, default=24, help="cap total expressions simulated")
    ap.add_argument("--batch-size", type=int, default=3)
    ap.add_argument("--from-saved", action="store_true",
                    help="skip the LLM; re-enumerate the cards already in decompose_test.jsonl "
                         "(deterministic, reproduces the same expressions) and simulate them")
    args = ap.parse_args()

    if args.from_saved:
        run_from_saved(args)
        return

    print("loading cluster catalog + field metadata ...")
    clusters = ctx.load_clusters()
    metadata = ctx.load_field_metadata()
    known_clusters = set(clusters)
    operator_menu = ctx.render_operator_menu()
    cluster_index = ctx.render_cluster_index(clusters, metadata)

    # field -> cluster index
    field_to_cluster: dict = {}
    for cid, c in clusters.items():
        for m in c.get("members") or []:
            fid = str(m.get("field_id") or "")
            if fid and fid not in field_to_cluster:
                field_to_cluster[fid] = cid

    print("indexing 85k-sim history for novelty scoring ...")
    hist_cells, best_by_fields = build_history_index()
    print(f"  {len(hist_cells)} distinct (skeleton, field-set) cells")

    seeds = pick_seeds(args.n_seeds)
    print(f"selected {len(seeds)} structurally-distinct robust seeds\n")

    agent = TheoryEconomistAgent(model=args.model)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    out_f = OUT.open("w")

    totals = {"cards": 0, "rejected": 0, "exprs": 0, "novel": 0, "off_field": 0, "off_dataset": 0}
    sim_members: list[dict] = []  # collected for --simulate

    for i, seed in enumerate(seeds, 1):
        seed_clusters = []
        for f in seed["fields"]:
            cid = field_to_cluster.get(f)
            if cid and cid not in seed_clusters:
                seed_clusters.append(cid)
        seed_ds = {clusters[c].get("dataset_id") for c in seed_clusters}
        menu = neglected_menu(clusters, metadata, seed_clusters)
        cluster_catalog = ctx.render_cluster_catalog(menu, clusters, metadata)

        print(f"=== SEED {i}/{len(seeds)} | sharpe={seed['sharpe']} rob={seed['rob']:.2f} ds={seed['dataset']} ===")
        print(f"  {seed['expr'][:120]}")
        print(f"  seed clusters: {seed_clusters or '(none resolved)'}")

        try:
            decomp, result = agent.expand_from_seed(
                seed_expression=seed["expr"],
                seed_sharpe=seed["sharpe"], seed_fitness=seed["fitness"],
                seed_robustness=round(seed["rob"], 3), seed_dataset=seed["dataset"],
                seed_fields=", ".join(seed["fields"]),
                cluster_catalog=cluster_catalog, cluster_index=cluster_index,
                operator_menu=operator_menu, mechanism_index="(none for test)",
                falsified_index="(none for test)", n_expansions=args.n_expansions,
                known_clusters=known_clusters, clusters=clusters,
            )
        except Exception as e:
            print(f"  !! LLM/parse error: {e}\n")
            continue

        edge = (decomp.get("edge") or "")[:160]
        print(f"  EDGE: {edge}")
        print(f"  cards valid={len(result.cards)} rejected={len(result.rejected)}")
        for card, probs in result.rejected[:3]:
            print(f"     rejected: {probs}")

        for card in result.cards:
            totals["cards"] += 1
            card_clusters = card.get("cluster_ids") or []
            card_ds = {clusters.get(c, {}).get("dataset_id") for c in card_clusters}
            variants = enumerate_family(card, clusters, family_max=args.family_max)
            off_field = off_ds = novel = 0
            samples = []
            for v in variants:
                expr = v.get("expression") or ""
                if args.simulate and len(sim_members) < args.max_sims:
                    m = dict(v)
                    m["search_lane"] = "decompose"
                    m["theory_id"] = f"decompose_seed{i}_card{totals['cards']}"
                    m.setdefault("direction", card.get("direction") or "+")
                    sim_members.append(m)
                vf = v.get("fields") or derive_fields(expr)
                fs = frozenset(vf)
                cell = (skeleton(expr, vf), fs)
                is_novel = cell not in hist_cells
                moved_field = not (fs & frozenset(seed["fields"]))
                moved_ds = bool(card_ds - seed_ds)
                totals["exprs"] += 1
                if is_novel:
                    novel += 1; totals["novel"] += 1
                if moved_field:
                    off_field += 1; totals["off_field"] += 1
                if moved_ds:
                    off_ds += 1; totals["off_dataset"] += 1
                if len(samples) < 2:
                    samples.append((expr[:100], is_novel, moved_field))
            print(f"   + [{card.get('crowding_estimate','?')[:18]}] {','.join(card_clusters)[:50]}"
                  f" -> {len(variants)} exprs | novel={novel} off-field={off_field}")
            print(f"       decorrelation: {(card.get('decorrelation') or '')[:120]}")
            for ex, nv, mf in samples:
                print(f"         {'NEW ' if nv else 'seen'} {'OFF ' if mf else 'same'} {ex}")
            out_f.write(json.dumps({
                "seed_expr": seed["expr"], "seed_sharpe": seed["sharpe"],
                "decomposition": decomp, "card": card,
                "n_variants": len(variants), "novel": novel,
                "off_field": off_field, "off_dataset": off_ds,
                "sample_expressions": [v.get("expression") for v in variants[:5]],
            }) + "\n")
        print()

    out_f.close()
    print("===== TOTALS =====")
    print(f"expansion cards: {totals['cards']}  (rejected: {totals['rejected']})")
    ex = totals["exprs"] or 1
    print(f"rendered expressions: {totals['exprs']}")
    print(f"  novel (no history cell):       {totals['novel']} ({100*totals['novel']/ex:.1f}%)")
    print(f"  moved off seed's field:        {totals['off_field']} ({100*totals['off_field']/ex:.1f}%)")
    print(f"  moved off seed's dataset:      {totals['off_dataset']} ({100*totals['off_dataset']/ex:.1f}%)")
    print(f"\nwrote {OUT}")

    if args.simulate and sim_members:
        simulate_on_brain(sim_members, args.credentials, args.batch_size)


def run_from_saved(args) -> None:
    """Reuse the cards already generated in decompose_test.jsonl: re-enumerate them
    (deterministic) into the exact same expressions, dedup, and simulate."""
    clusters = ctx.load_clusters()
    if not OUT.exists():
        print(f"no saved cards at {OUT}; run a generation pass first.")
        return
    cards = []
    for line in OUT.open():
        try:
            row = json.loads(line)
        except Exception:
            continue
        if row.get("card"):
            cards.append(row["card"])
    print(f"loaded {len(cards)} saved expansion cards from {OUT}")

    seen: set[str] = set()
    members: list[dict] = []
    for ci, card in enumerate(cards):
        for v in enumerate_family(card, clusters, family_max=args.family_max):
            expr = v.get("expression") or ""
            if not expr or expr in seen:
                continue
            seen.add(expr)
            m = dict(v)
            m["search_lane"] = "decompose"
            m["theory_id"] = f"decompose_card{ci}"
            m.setdefault("direction", card.get("direction") or "+")
            members.append(m)
    if args.max_sims and len(members) > args.max_sims:
        members = members[: args.max_sims]
    print(f"re-enumerated {len(members)} distinct expressions")
    if not args.simulate:
        print("(dry — pass --simulate to send these to BRAIN)")
        for m in members[:10]:
            print("  ", (m.get("expression") or "")[:100])
        return
    simulate_on_brain(members, args.credentials, args.batch_size)


def simulate_on_brain(members: list[dict], credentials: str, batch_size: int) -> None:
    from wq_alpha.brain.api import sign_in
    from wq_alpha.brain.submit import DEFAULT_SETTINGS
    from wq_alpha.logs import session_log_path
    from wq_alpha.workshop.core import DEFAULT_WORKSHOP_SETTINGS
    from wq_alpha.workshop.simulation import submit_candidate_batch

    print(f"\n===== SIMULATING {len(members)} expressions on {credentials} =====")
    settings = {**DEFAULT_SETTINGS, **DEFAULT_WORKSHOP_SETTINGS}
    sim_log = session_log_path("decompose")
    session = sign_in(credentials)
    records: list[dict] = []
    for start in range(0, len(members), batch_size):
        wave = members[start:start + batch_size]
        new = submit_candidate_batch(
            wave, session, sim_log, dict(settings), credentials, batch_size,
            yearly_stats_attempts=4,
        )
        records.extend(new or [])
        print(f"  simulated {min(start + batch_size, len(members))}/{len(members)}")

    # Fetch the OFFICIAL competition score-addition (brain_score_delta) for every
    # submittable result. Only submittables are scored — no point asking Brain to
    # project a score for a candidate that can't be submitted.
    score_records = fetch_score_deltas(records, session)

    sub = robust = score_add = 0
    rows = []
    for r in score_records:
        ok = is_submittable(r)
        rob = None
        ys = r.get("yearly_stats")
        if ys:
            try:
                rob = compute_robustness(normalize_yearly_stats(ys)).score
            except Exception:
                rob = None
        delta = r.get("brain_score_delta")
        sc = _self_corr(r)
        if ok:
            sub += 1
            if rob is not None and rob >= 0.5:
                robust += 1
            if delta is not None and delta > 0:
                score_add += 1
        rows.append((ok, r.get("is_sharpe"), r.get("is_fitness"), r.get("is_turnover"),
                     rob, sc, delta, (r.get("expression") or "")[:80]))
    # rank: submittable, then score-adding, then sharpe
    rows.sort(key=lambda x: (x[0], (x[6] or -1e9), x[1] or -9), reverse=True)
    print(f"\n----- BRAIN RESULTS: {len(records)} simulated -----")
    print(f"submittable: {sub}  |  submittable+robust(>=0.5): {robust}  |  "
          f"SCORE-ADDING (delta>0): {score_add}")
    print(f"{'sub':>4} {'sharpe':>7} {'fit':>6} {'turn':>6} {'rob':>5} {'selfcor':>8} {'score_d':>9}  expression")
    for ok, sh, fit, turn, rob, sc, delta, expr in rows[:30]:
        rs = f"{rob:.2f}" if rob is not None else "  - "
        scs = f"{sc:.2f}" if sc is not None else "   - "
        ds = f"{delta:+.1f}" if delta is not None else "    - "
        print(f"{'Y' if ok else 'n':>4} {sh if sh is not None else '?':>7} "
              f"{fit if fit is not None else '?':>6} {turn if turn is not None else '?':>6} "
              f"{rs:>5} {scs:>8} {ds:>9}  {expr}")
    out = THEORY_DIR / "decompose_brain_results.jsonl"
    with out.open("w") as f:
        for r in score_records:
            f.write(json.dumps(r) + "\n")
    print(f"\nsim log: {sim_log}\nfull results: {out}")


def _self_corr(record: dict):
    """Pull the SELF_CORRELATION check value (None if pending/absent)."""
    for chk in record.get("checks") or []:
        if str(chk.get("name")) == "SELF_CORRELATION":
            v = chk.get("value")
            try:
                return float(v) if v is not None else None
            except (TypeError, ValueError):
                return None
    return record.get("max_self_correlation")


def fetch_score_deltas(records: list[dict], session, competition_id: str = "IQC2026S1") -> list[dict]:
    """Annotate each submittable record with the official brain_score_delta.

    Mirrors wq_alpha/workshop/scoring.py: poll the before-and-after-performance
    endpoint, derive delta = after - before. Non-submittable or alpha_id-less
    records pass through unchanged.
    """
    from wq_alpha.portfolio.admission import fetch_brain_performance_comparison

    out: list[dict] = []
    scored = 0
    for r in records:
        rec = dict(r)
        alpha_id = rec.get("alpha_id")
        if not (alpha_id and is_submittable(rec)):
            out.append(rec)
            continue
        try:
            perf = fetch_brain_performance_comparison(
                session, str(alpha_id), competition_id, max_attempts=5, sleep_seconds=2.0)
            score = (perf or {}).get("score") or {}
            before = score.get("before")
            after = score.get("after")
            before = float(before) if before is not None else None
            after = float(after) if after is not None else None
            rec["brain_score_before"] = before
            rec["brain_score_after"] = after
            rec["brain_score_delta"] = (
                after - before if before is not None and after is not None else None)
            scored += 1
            print(f"  scored {scored}: delta={rec['brain_score_delta']} ({(rec.get('expression') or '')[:60]})")
        except Exception as e:
            rec["brain_score_delta"] = None
            print(f"  score fetch failed for {alpha_id}: {e}")
        out.append(rec)
    return out


if __name__ == "__main__":
    main()
