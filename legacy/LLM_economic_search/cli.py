"""CLI for the plan-first LLM economic search workspace."""

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from collections import defaultdict
from pathlib import Path
from typing import Any

from .brain_submit import submit_plan as submit_economic_plan
from .candidate_generation import plan_name, write_combined_plan
from .continuous_runner import (
    ContinuousRunLock,
    append_active_alpha_ids,
    append_event,
    append_promotions_to_ledger,
    cycle_id_for_now,
    harvest_promotions,
    merge_alpha_ids,
    read_active_alpha_ids,
    read_run_state,
    utc_timestamp,
    write_active_alpha_ids,
    write_run_state,
)
from .critique import DeterministicNoveltyCritic, DeterministicResultInterpreter
from .expression_generation import LLMExpressionGenerator
from .field_catalogs import (
    DEFAULT_DELAY_FIELD_DIR,
    DEFAULT_METADATA_PATH,
    DELAY_DEFINITION,
    fetch_delay_records,
    field_catalog_summary,
    records_from_metadata,
    resolve_delay_field_csv,
    write_delay_catalog,
)
from .history_index import (
    DEFAULT_FIELD_CSV,
    DEFAULT_INDEX_PATH,
    DEFAULT_SUMMARY_PATH,
    HistoryIndex,
    HistoryIndexer,
)
from .llm_agents import NoveltyCriticAgent, TheoryProposerAgent, mock_theories
from .portfolio_context import (
    DEFAULT_COMPETITION_ID,
    DEFAULT_PORTFOLIO_CONTEXT_PATH,
    fetch_and_write_portfolio_context,
    load_portfolio_context,
)
from .portfolio_scoring import score_sim_log
from .schemas import TheorySpec, normalize_expression
from .theory_store import (
    DEFAULT_ARTIFACTS_DIR,
    DEFAULT_THEORY_DIR,
    TheoryStore,
    candidate_index_key,
    row_delay,
)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def _ensure_index(args: argparse.Namespace) -> HistoryIndex:
    index = HistoryIndex(index_path=Path(args.index_path), summary_path=Path(args.summary_path))
    if not Path(args.index_path).exists() or not Path(args.summary_path).exists() or not index.is_compatible():
        print("History index missing; building it first ...")
        HistoryIndexer(
            index_path=Path(args.index_path),
            summary_path=Path(args.summary_path),
            field_csv=Path(args.fields_csv),
        ).build()
    return index


def _delay_field_summary(args: argparse.Namespace) -> tuple[Path, dict[str, Any]]:
    field_path = resolve_delay_field_csv(
        target_delay=args.delay,
        fields_csv=Path(args.fields_csv),
        delay_fields_dir=Path(args.delay_fields_dir),
        universe=args.field_universe,
        credentials=getattr(args, "credentials", None),
    )
    return field_path, field_catalog_summary(field_path, target_delay=args.delay)


def _with_target_delay(theory: TheorySpec, target_delay: int) -> TheorySpec:
    data = theory.to_dict()
    data["eligible_delays"] = sorted(set([int(target_delay), *[int(d) for d in theory.eligible_delays]]))
    delay_rationale = dict(data.get("delay_rationale") or {})
    delay_rationale.setdefault(str(target_delay), "Generated for the requested target delay.")
    data["delay_rationale"] = delay_rationale
    return TheorySpec.from_dict(data)


def cmd_index(args: argparse.Namespace) -> int:
    summary = HistoryIndexer(
        index_path=Path(args.index_path),
        summary_path=Path(args.summary_path),
        field_csv=Path(args.fields_csv),
    ).build()
    print(f"Wrote SQLite index: {args.index_path}")
    print(f"Wrote summary JSON: {args.summary_path}")
    print(f"Indexed records:    {summary.get('total_records', 0)}")
    print(f"Standalone passers: {summary.get('standalone_pass_records', 0)}")
    return 0


def cmd_propose(args: argparse.Namespace) -> int:
    index = _ensure_index(args)
    store = TheoryStore(Path(args.theory_dir))
    existing = store.list_theories()
    portfolio_context = load_portfolio_context(Path(args.portfolio_context_path))
    try:
        field_path, field_summary = _delay_field_summary(args)
    except FileNotFoundError as exc:
        if not args.mock:
            raise
        field_path = Path("")
        field_summary = {"target_delay": args.delay, "warning": str(exc)}
    agent = TheoryProposerAgent()
    try:
        theories = agent.propose(
            history_summary=index.summary(),
            existing_theories=existing,
            portfolio_context=portfolio_context,
            target_delay=args.delay,
            field_catalog_summary=field_summary,
            sprint_focus=args.sprint_focus,
            n=args.n,
            mock=args.mock,
        )
        source = "mock" if args.mock else "llm"
    except ValueError as exc:
        print(f"WARNING: LLM proposer failed ({exc}); using deterministic fallback theories.")
        theories = mock_theories(sprint_focus=args.sprint_focus, n=args.n)
        source = "fallback"
    theories = [_with_target_delay(theory, args.delay) for theory in theories]
    out_dir = Path(args.artifacts_dir) / "proposals"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / plan_name(f"theory_proposals_delay{args.delay}").replace(".jsonl", ".json")
    out_path.write_text(json.dumps([t.to_dict() for t in theories], indent=2, sort_keys=True))

    for theory in theories:
        store.save_theory(
            theory,
            event_type="created",
            payload={
                "source": "propose",
                "proposer_source": source,
                "mock": args.mock,
                "target_delay": args.delay,
                "field_catalog": str(field_path),
                "delay_definition": DELAY_DEFINITION,
            },
        )
    print(f"Proposed theories: {len(theories)}")
    print(f"Wrote proposals:   {out_path}")
    print(f"Target delay:      {args.delay}")
    print(f"Field catalog:     {field_path}")
    print(f"Stored under:      {args.theory_dir}")
    return 0


def cmd_critique(args: argparse.Namespace) -> int:
    index = _ensure_index(args)
    store = TheoryStore(Path(args.theory_dir))
    portfolio_context = load_portfolio_context(Path(args.portfolio_context_path))
    if args.theory_id:
        theories = [store.load_theory(args.theory_id)]
    else:
        theories = store.list_theories(statuses={"PROPOSED", "REVISE", "ITERATE"})
    critic = DeterministicNoveltyCritic(index, portfolio_context=portfolio_context, target_delay=args.delay)
    deterministic: dict[str, dict[str, Any]] = {t.theory_id: critic.critique(t) for t in theories}

    llm_decisions: dict[str, dict[str, Any]] = {}
    if not getattr(args, "no_llm_critic", False) and theories:
        llm_decisions = NoveltyCriticAgent().critique(
            theories=theories,
            deterministic_reports=deterministic,
            history_summary=index.summary(),
            portfolio_context=portfolio_context,
            target_delay=args.delay,
        )

    counts = defaultdict(int)
    for theory in theories:
        det = deterministic[theory.theory_id]
        llm = llm_decisions.get(theory.theory_id, {}) if isinstance(llm_decisions, dict) else {}
        decision = str(llm.get("decision") or det["decision"]).upper()
        reason = str(llm.get("reason") or det["reason"])
        status = {"ACCEPT": "ACCEPTED", "REJECT": "REJECTED", "REVISE": "REVISE"}.get(decision, "REVISE")
        updated = theory.with_status(status)
        store.save_theory(
            updated,
            event_type="critiqued",
            payload={"decision": decision, "reason": reason, "deterministic": det, "llm": llm},
        )
        store.write_decision(
            theory.theory_id,
            {
                "decision": decision,
                "reason": reason,
                "risk_report": det.get("risk_report", {}),
                "target_delay": args.delay,
            },
        )
        counts[decision] += 1
        print(f"{decision:<7} {theory.theory_id}: {reason}")
    print(f"Critiqued theories: {len(theories)}  decisions={dict(counts)}")
    return 0


def cmd_fields(args: argparse.Namespace) -> int:
    delays = [0, 1] if args.delay == "both" else [int(args.delay)]
    wrote = []
    for delay in delays:
        if args.from_metadata:
            records = records_from_metadata(
                metadata_path=Path(args.from_metadata),
                delay=delay,
                universe=args.field_universe,
            )
            source = str(args.from_metadata)
        else:
            records = fetch_delay_records(
                credentials=args.credentials,
                delay=delay,
                universe=args.field_universe,
            )
            source = "Brain API"
        if not records:
            print(f"No fields found for delay={delay} universe={args.field_universe} from {source}")
            continue
        path = write_delay_catalog(
            delay=delay,
            records=records,
            output_dir=Path(args.delay_fields_dir),
            universe=args.field_universe,
            credentials=args.credentials,
        )
        wrote.append(path)
        print(f"Wrote delay={delay} field catalog: {path} ({len(records)} records)")
    return 0 if wrote else 1


def cmd_portfolio(args: argparse.Namespace) -> int:
    alpha_ids = _parse_alpha_ids(args.alpha_ids, args.alpha_id)
    context = fetch_and_write_portfolio_context(
        credentials=args.credentials,
        competition_id=args.competition_id,
        alpha_ids=alpha_ids,
        include_account_scan=args.include_account_scan,
        out_path=Path(args.portfolio_context_path),
        field_csv=Path(args.fields_csv),
        limit=args.limit,
        max_pages=args.max_pages,
        sleep_seconds=args.sleep,
    )
    print(f"Fetched submitted alphas: {context.get('n_submitted', 0)}")
    print(f"Wrote portfolio context:  {args.portfolio_context_path}")
    for exposure in (context.get("theme_exposures") or [])[:10]:
        print(f"  {exposure['count']:>3}  {exposure['theme']}")
    return 0


def _parse_alpha_ids(alpha_ids_text: str, repeated_ids: list[str]) -> list[str]:
    raw_parts: list[str] = []
    if alpha_ids_text:
        raw_parts.extend(alpha_ids_text.replace(",", " ").split())
    raw_parts.extend(repeated_ids or [])
    seen: set[str] = set()
    out: list[str] = []
    for part in raw_parts:
        alpha_id = part.strip()
        if not alpha_id or alpha_id in seen:
            continue
        seen.add(alpha_id)
        out.append(alpha_id)
    return out


def _default_sim_log(args: argparse.Namespace) -> Path:
    return Path(args.artifacts_dir) / f"economic_delay{args.delay}_{args.stage}_sim.jsonl"


def _default_advance_sim_log(args: argparse.Namespace) -> Path:
    return Path(args.artifacts_dir) / f"economic_delay{args.delay}_advance_sim.jsonl"


def _write_proposals(args: argparse.Namespace, theories: list[TheorySpec]) -> Path:
    out_dir = Path(args.artifacts_dir) / "proposals"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / plan_name(f"theory_proposals_delay{args.delay}").replace(".jsonl", ".json")
    out_path.write_text(json.dumps([t.to_dict() for t in theories], indent=2, sort_keys=True))
    return out_path


def _critique_theories(
    args: argparse.Namespace,
    *,
    index: HistoryIndex,
    store: TheoryStore,
    theories: list[TheorySpec],
    portfolio_context: dict[str, Any],
) -> list[TheorySpec]:
    critic = DeterministicNoveltyCritic(index, portfolio_context=portfolio_context, target_delay=args.delay)
    deterministic: dict[str, dict[str, Any]] = {t.theory_id: critic.critique(t) for t in theories}

    llm_decisions: dict[str, dict[str, Any]] = {}
    use_llm_critic = not getattr(args, "no_llm_critic", False) and not getattr(args, "mock", False)
    if use_llm_critic and theories:
        llm_decisions = NoveltyCriticAgent().critique(
            theories=theories,
            deterministic_reports=deterministic,
            history_summary=index.summary(),
            portfolio_context=portfolio_context,
            target_delay=args.delay,
        )

    accepted: list[TheorySpec] = []
    counts = defaultdict(int)
    for theory in theories:
        det = deterministic[theory.theory_id]
        llm = llm_decisions.get(theory.theory_id, {}) if isinstance(llm_decisions, dict) else {}
        decision = str(llm.get("decision") or det["decision"]).upper()
        reason = str(llm.get("reason") or det["reason"])
        status = {"ACCEPT": "ACCEPTED", "REJECT": "REJECTED", "REVISE": "REVISE"}.get(decision, "REVISE")
        updated = theory.with_status(status)
        store.save_theory(
            updated,
            event_type="critiqued",
            payload={"decision": decision, "reason": reason, "deterministic": det, "llm": llm},
        )
        store.write_decision(
            theory.theory_id,
            {
                "decision": decision,
                "reason": reason,
                "risk_report": det.get("risk_report", {}),
                "target_delay": args.delay,
            },
        )
        counts[decision] += 1
        if status == "ACCEPTED":
            accepted.append(updated)
        print(f"{decision:<7} {theory.theory_id}: {reason}")
    print(f"Critiqued sprint theories: {len(theories)}  decisions={dict(counts)}")
    return accepted


def _generate_for_theories(
    args: argparse.Namespace,
    *,
    index: HistoryIndex,
    store: TheoryStore,
    theories: list[TheorySpec],
    field_path: Path,
    field_summary: dict[str, Any],
    portfolio_context: dict[str, Any],
) -> tuple[list[Any], Path | None]:
    generator = LLMExpressionGenerator(
        field_csv=field_path,
        history_index=index,
        target_delay=args.delay,
        min_distinct_skeletons=getattr(args, "min_distinct_skeletons", 4),
        max_window_only_variants=getattr(args, "max_window_only_variants", 3),
    )
    all_candidates = []
    for theory in theories:
        candidates, report = generator.generate(
            theory=theory,
            store=store,
            stage=args.stage,
            generation_mode=getattr(args, "generation_mode", args.stage),
            max_candidates=(
                getattr(args, "max_candidates_per_theory", 0)
                or getattr(args, "max_candidates", 0)
                or theory.max_probe_candidates
            ),
            portfolio_context=portfolio_context,
            history_summary=index.summary(),
            field_catalog_summary=field_summary,
            mock=getattr(args, "mock", False),
            max_repairs=getattr(args, "max_generation_repairs", 1),
        )
        if not candidates:
            print(f"No valid LLM expressions generated for {theory.theory_id}; report={report}")
            continue
        path = store.write_candidates(theory.theory_id, args.stage, candidates, target_delay=args.delay)
        store.append_event(
            theory.theory_id,
            "candidates_written",
            {
                "stage": args.stage,
                "path": str(path),
                "n_candidates": len(candidates),
                "target_delay": args.delay,
                "generation_report": report,
            },
        )
        all_candidates.extend(candidates)
        print(
            f"{theory.theory_id}: wrote {len(candidates)} LLM candidates to {path} "
            f"(rejected={report.get('rejected_count', 0)}, skeletons={report.get('distinct_operator_skeletons', 0)})"
        )

    if not all_candidates:
        print("Generated candidates: 0")
        return [], None

    combined = (
        Path(args.combined_out)
        if args.combined_out
        else Path(args.artifacts_dir) / "candidate_plans" / plan_name(f"economic_plan_delay{args.delay}_{args.stage}")
    )
    write_combined_plan(combined, all_candidates)
    print(f"Wrote combined plan: {combined}")
    print(f"Generated candidates: {len(all_candidates)}")
    return all_candidates, combined


def cmd_generate(args: argparse.Namespace) -> int:
    index = _ensure_index(args)
    store = TheoryStore(Path(args.theory_dir))
    portfolio_context = load_portfolio_context(Path(args.portfolio_context_path))
    field_path, field_summary = _delay_field_summary(args)
    if args.theory_id:
        theories = [store.load_theory(args.theory_id)]
    else:
        theories = store.list_theories(statuses={"ACCEPTED", "EXPAND", "ITERATE"})
    all_candidates, combined = _generate_for_theories(
        args,
        index=index,
        store=store,
        theories=theories,
        field_path=field_path,
        field_summary=field_summary,
        portfolio_context=portfolio_context,
    )
    print(f"Generated candidates: {len(all_candidates)}")
    return 0


def cmd_ingest(args: argparse.Namespace) -> int:
    store = TheoryStore(Path(args.theory_dir))
    rows = _read_jsonl(Path(args.results_jsonl))
    candidate_index = store.candidate_index()
    grouped_by_candidate_path: dict[str, list[dict[str, Any]]] = defaultdict(list)
    grouped_by_theory: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for row in rows:
        expr = normalize_expression(str(row.get("expression") or ""))
        delay = row_delay(row)
        meta = {}
        if delay is not None:
            meta = candidate_index.get(candidate_index_key(expr, delay), {})
        if not meta:
            meta = candidate_index.get(expr, {})
        theory_id = str(row.get("theory_id") or meta.get("theory_id") or "")
        if not theory_id:
            continue
        enriched = dict(meta)
        enriched.update(row)
        enriched["theory_id"] = theory_id
        if row_delay(enriched) is None and delay is not None:
            enriched["target_delay"] = delay
        candidate_path = meta.get("_candidate_path")
        if candidate_path:
            grouped_by_candidate_path[str(candidate_path)].append(enriched)
        else:
            grouped_by_theory[theory_id].append(enriched)

    count = 0
    for candidate_path, result_rows in grouped_by_candidate_path.items():
        store.write_results_for_candidate_path(candidate_path, result_rows)
        count += len(result_rows)
    if grouped_by_theory:
        counts = store.write_ingested_results(grouped_by_theory)
        count += sum(counts.values())

    interpreter = DeterministicResultInterpreter()
    touched: set[tuple[str, int | None]] = set()
    for rows_for_path in grouped_by_candidate_path.values():
        touched.update(
            (str(r.get("theory_id")), row_delay(r))
            for r in rows_for_path
            if r.get("theory_id")
        )
    touched.update((theory_id, row_delay(r)) for theory_id, rows in grouped_by_theory.items() for r in rows)
    for theory_id, delay in sorted(touched):
        theory = store.load_theory(theory_id)
        decision = interpreter.interpret(theory, store.all_results(theory_id, target_delay=delay), target_delay=delay)
        store.write_decision(theory_id, decision, target_delay=delay)
        store.write_memory(theory_id, interpreter.memory_markdown(theory, decision), target_delay=delay)
        if delay is None and decision.get("updated_status") and decision["updated_status"] != theory.status:
            store.update_status(theory_id, str(decision["updated_status"]), str(decision.get("reason") or ""))
        elif delay is not None and decision.get("updated_status") == "PROMOTED" and theory.status != "PROMOTED":
            store.update_status(theory_id, "PROMOTED", str(decision.get("reason") or ""))
        delay_label = f" delay={delay}" if delay is not None else ""
        print(f"{theory_id}{delay_label}: {decision.get('decision')} - {decision.get('reason')}")

    print(f"Ingested matched results: {count}")
    return 0


def cmd_submit(args: argparse.Namespace) -> int:
    new_records = submit_economic_plan(
        args.plan,
        args.sim_log,
        batch_size=args.batch_size,
        credentials=args.credentials,
        max_candidates=args.max_candidates,
    )
    print(f"Submitted economic records: {len(new_records)}")
    _merge_sim_log_to_master(Path(args.sim_log))
    if args.ingest:
        ingest_args = argparse.Namespace(**vars(args))
        ingest_args.results_jsonl = args.sim_log
        return cmd_ingest(ingest_args)
    return 0


def _score_log_if_needed(args: argparse.Namespace, sim_log: Path) -> Path:
    if getattr(args, "no_admission_score", False):
        print("ADMISSION: skipped because --no-admission-score was set.")
        return sim_log
    alpha_ids = _parse_alpha_ids(getattr(args, "alpha_ids", ""), getattr(args, "alpha_id", []))
    print("ADMISSION: scoring candidates for Brain score contribution, with local fallback after retries")
    scored = score_sim_log(
        sim_log,
        credentials=getattr(args, "credentials", "brain.txt"),
        competition_id=getattr(args, "competition_id", DEFAULT_COMPETITION_ID),
        alpha_ids=alpha_ids,
        top_per_theory=getattr(args, "admission_top_per_theory", 3),
        min_sharpe=getattr(args, "admission_min_sharpe", 1.0),
        min_brain_score_delta=getattr(args, "admission_min_brain_score_delta", 0.0),
        min_fallback_delta_sharpe=getattr(args, "admission_min_fallback_delta_sharpe", 0.0),
        performance_attempts=getattr(args, "performance_attempts", 5),
        performance_wait_seconds=getattr(args, "performance_wait", 2.0),
        max_pnl_attempts=getattr(args, "max_pnl_attempts", 3),
        sleep_seconds=getattr(args, "sleep", 0.5),
    )
    print(f"ADMISSION: wrote scored log {scored}")
    return scored


def _merge_sim_log_to_master(sim_log: Path) -> None:
    if not sim_log.exists():
        return
    try:
        from wq_alpha.logs import MASTER, merge_session_to_master

        n_new, n_updated = merge_session_to_master(sim_log)
        print(f"Merged to master: {n_new} new, {n_updated} updated  ({MASTER})")
    except Exception as exc:
        print(f"[warn] Could not merge {sim_log} to master: {exc}")


def cmd_advance(args: argparse.Namespace) -> int:
    index = _ensure_index(args)
    store = TheoryStore(Path(args.theory_dir))
    portfolio_context = _advance_portfolio_context(args)
    field_path, field_summary = _delay_field_summary(args)

    groups: dict[str, list[TheorySpec]] = {"iterate": [], "expand": []}
    for theory in store.list_theories():
        if args.theory_id and theory.theory_id != args.theory_id:
            continue
        decision = store.read_decision(theory.theory_id, target_delay=args.delay)
        branch = str(decision.get("decision") or "").upper()
        if branch == "ITERATE":
            groups["iterate"].append(theory)
        elif branch == "EXPAND":
            groups["expand"].append(theory)

    print(
        "ADVANCE: selected "
        f"{len(groups['iterate'])} iterate and {len(groups['expand'])} expand theories for delay={args.delay}"
    )
    generated: list[Any] = []
    combined_paths: list[Path] = []
    advanced_branches: list[tuple[TheorySpec, str]] = []

    for stage, theories, max_candidates in (
        ("iterate", groups["iterate"], args.max_iterate_candidates),
        ("expand", groups["expand"], args.max_expand_candidates),
    ):
        if not theories:
            continue
        sub_args = argparse.Namespace(**vars(args))
        sub_args.stage = stage
        sub_args.generation_mode = stage
        sub_args.max_candidates_per_theory = max_candidates
        sub_args.combined_out = ""
        candidates, combined = _generate_for_theories(
            sub_args,
            index=index,
            store=store,
            theories=theories,
            field_path=field_path,
            field_summary=field_summary,
            portfolio_context=portfolio_context,
        )
        generated.extend(candidates)
        if combined:
            combined_paths.append(combined)
            advanced_branches.extend((theory, stage) for theory in theories)

    if not generated:
        print("ADVANCE: no second-stage candidates were generated.")
        return 0
    if not args.simulate:
        print("Simulation skipped. Re-run advance with --simulate to send second-stage plans to Brain.")
        return 0

    sim_log = Path(args.sim_log) if args.sim_log else _default_advance_sim_log(args)
    sim_log.parent.mkdir(parents=True, exist_ok=True)
    submitted = 0
    for combined in combined_paths:
        print(f"SIMULATE: submitting second-stage plan {combined}")
        new_records = submit_economic_plan(
            str(combined),
            str(sim_log),
            batch_size=args.batch_size,
            credentials=args.credentials,
            max_candidates=args.max_submit_candidates,
        )
        submitted += len(new_records)
    print(f"Simulation log: {sim_log}")
    print(f"New simulation records: {submitted}")

    if args.no_ingest:
        _merge_sim_log_to_master(sim_log)
        print("Ingest skipped because --no-ingest was set.")
        return 0

    ingest_log = _score_log_if_needed(args, sim_log)
    _merge_sim_log_to_master(ingest_log)
    print("INGEST: attaching second-stage simulation results to theory histories")
    ingest_args = argparse.Namespace(**vars(args))
    ingest_args.results_jsonl = str(ingest_log)
    rc = cmd_ingest(ingest_args)
    if not getattr(args, "no_admission_score", False):
        _finalize_second_stage(store, advanced_branches, target_delay=args.delay)
    else:
        print("ADVANCE: terminal branch decisions skipped because admission scoring was disabled.")
    return rc


def _advance_portfolio_context(args: argparse.Namespace) -> dict[str, Any]:
    if getattr(args, "skip_portfolio", False):
        context = load_portfolio_context(Path(args.portfolio_context_path))
        print(f"ADVANCE: using existing portfolio context ({context.get('n_submitted', 0)} submitted)")
        return context
    alpha_ids = _parse_alpha_ids(getattr(args, "alpha_ids", ""), getattr(args, "alpha_id", []))
    context = fetch_and_write_portfolio_context(
        credentials=args.credentials,
        competition_id=args.competition_id,
        alpha_ids=alpha_ids,
        include_account_scan=args.include_account_scan,
        out_path=Path(args.portfolio_context_path),
        field_csv=Path(args.fields_csv),
        limit=args.limit,
        max_pages=args.max_pages,
        sleep_seconds=args.sleep,
    )
    print(f"ADVANCE: refreshed portfolio context ({context.get('n_submitted', 0)} submitted)")
    return context


def _finalize_second_stage(
    store: TheoryStore,
    branches: list[tuple[TheorySpec, str]],
    *,
    target_delay: int,
) -> None:
    seen: set[tuple[str, str]] = set()
    interpreter = DeterministicResultInterpreter()
    for theory, stage in branches:
        key = (theory.theory_id, stage)
        if key in seen:
            continue
        seen.add(key)
        results = store.all_results(theory.theory_id, target_delay=target_delay)
        stage_done = [
            row
            for row in results
            if row.get("status") == "done" and str(row.get("experiment_stage") or "") == stage
        ]
        prior = store.read_decision(theory.theory_id, target_delay=target_delay)
        if not stage_done:
            decision = {
                "decision": prior.get("decision") or stage.upper(),
                "reason": "second-stage simulation incomplete",
                "updated_status": prior.get("updated_status") or prior.get("decision") or theory.status,
                "target_delay": target_delay,
                "metrics": {"target_delay": target_delay, "stage": stage, "n_second_stage_done": 0},
            }
        else:
            promoted = [
                row
                for row in stage_done
                if _standalone_passing(row) and str(row.get("admission_decision") or "").upper() == "PROMOTE"
            ]
            if promoted:
                decision = {
                    "decision": "PROMOTE",
                    "reason": "second-stage standalone passer has positive marginal portfolio contribution",
                    "updated_status": "PROMOTED",
                    "target_delay": target_delay,
                    "metrics": _second_stage_metrics(stage_done, stage=stage, target_delay=target_delay),
                }
                if theory.status != "PROMOTED":
                    store.update_status(theory.theory_id, "PROMOTED", decision["reason"])
            else:
                decision = {
                    "decision": "KILL",
                    "reason": "second-stage produced no standalone pool-positive candidate",
                    "updated_status": "KILLED",
                    "target_delay": target_delay,
                    "metrics": _second_stage_metrics(stage_done, stage=stage, target_delay=target_delay),
                }
        store.write_decision(theory.theory_id, decision, target_delay=target_delay)
        store.write_memory(theory.theory_id, interpreter.memory_markdown(theory, decision), target_delay=target_delay)
        print(f"ADVANCE final {theory.theory_id} delay={target_delay}: {decision['decision']} - {decision['reason']}")


def _second_stage_metrics(rows: list[dict[str, Any]], *, stage: str, target_delay: int) -> dict[str, Any]:
    sharpes = [_float(row.get("is_sharpe")) for row in rows if row.get("is_sharpe") is not None]
    deltas = []
    for row in rows:
        value = row.get("admission_brain_score_delta")
        if value is None:
            value = row.get("admission_delta_pool_sharpe")
        if value is not None:
            deltas.append(_float(value))
    return {
        "target_delay": target_delay,
        "stage": stage,
        "n_second_stage_done": len(rows),
        "n_standalone_pass": sum(1 for row in rows if _standalone_passing(row)),
        "n_promote": sum(1 for row in rows if str(row.get("admission_decision") or "").upper() == "PROMOTE"),
        "best_sharpe": max(sharpes) if sharpes else None,
        "best_contribution_delta": max(deltas) if deltas else None,
    }


def _standalone_passing(row: dict[str, Any]) -> bool:
    checks = row.get("checks") or []
    return bool(checks) and not any(
        check.get("result") == "FAIL"
        for check in checks
        if isinstance(check, dict) and check.get("name") != "SELF_CORRELATION"
    )


def _float(value: Any) -> float:
    try:
        return float(value or 0.0)
    except (TypeError, ValueError):
        return 0.0


def cmd_run_forever(args: argparse.Namespace) -> int:
    continuous_dir = Path(args.continuous_dir) if args.continuous_dir else Path(args.artifacts_dir) / "continuous"
    continuous_dir.mkdir(parents=True, exist_ok=True)
    state_path = Path(args.state_path) if args.state_path else continuous_dir / f"run_state_delay{args.delay}.json"
    events_path = Path(args.events_path) if args.events_path else continuous_dir / f"events_delay{args.delay}.jsonl"
    ledger_path = Path(args.ledger_path) if args.ledger_path else continuous_dir / f"promotions_delay{args.delay}.csv"
    active_path = (
        Path(args.active_alpha_ids_path)
        if args.active_alpha_ids_path
        else continuous_dir / f"active_alpha_ids_delay{args.delay}.txt"
    )
    lock_path = Path(args.lock_file) if args.lock_file else continuous_dir / f"run_forever_delay{args.delay}.lock"

    initial_ids = _parse_alpha_ids(args.alpha_ids, args.alpha_id)
    active_ids = merge_alpha_ids(initial_ids, read_active_alpha_ids(active_path))
    write_active_alpha_ids(active_path, active_ids)

    lock = ContinuousRunLock(lock_path)
    try:
        lock.__enter__()
    except RuntimeError as exc:
        print(f"RUNNER: {exc}")
        append_event(events_path, "runner_lock_failed", {"error": str(exc)})
        return 2
    try:
        return _run_forever_locked(
            args,
            state_path=state_path,
            events_path=events_path,
            ledger_path=ledger_path,
            active_path=active_path,
            continuous_dir=continuous_dir,
            active_ids=active_ids,
        )
    finally:
        lock.__exit__(None, None, None)


def _run_forever_locked(
    args: argparse.Namespace,
    *,
    state_path: Path,
    events_path: Path,
    ledger_path: Path,
    active_path: Path,
    continuous_dir: Path,
    active_ids: list[str],
) -> int:
    state = read_run_state(state_path)
    prior_cycle_number = int(state.get("cycle_number") or 0)
    session_promotions = 0
    attempted_cycles = 0
    append_event(
        events_path,
        "runner_started",
        {
            "delay": args.delay,
            "max_cycles": args.max_cycles,
            "active_alpha_count": len(active_ids),
            "ledger_path": str(ledger_path),
        },
    )

    while args.max_cycles <= 0 or attempted_cycles < args.max_cycles:
        attempted_cycles += 1
        cycle_number = prior_cycle_number + attempted_cycles
        cycle_id = cycle_id_for_now(target_delay=args.delay)
        timestamp = utc_timestamp()
        active_ids = merge_alpha_ids(read_active_alpha_ids(active_path), active_ids)
        write_active_alpha_ids(active_path, active_ids)

        sprint_sim_log = continuous_dir / f"{cycle_id}_probe_sim.jsonl"
        advance_sim_log = continuous_dir / f"{cycle_id}_advance_sim.jsonl"
        scored_logs = [
            _scored_log_for_runner(sprint_sim_log, no_admission_score=args.no_admission_score),
            _scored_log_for_runner(advance_sim_log, no_admission_score=args.no_admission_score),
        ]

        cycle_state = {
            "cycle_id": cycle_id,
            "cycle_number": cycle_number,
            "delay": args.delay,
            "started_at": timestamp,
            "active_alpha_count": len(active_ids),
            "sprint_sim_log": str(sprint_sim_log),
            "advance_sim_log": str(advance_sim_log),
            "status": "running",
        }
        state.update(cycle_state)
        write_run_state(state_path, state)
        append_event(events_path, "cycle_started", cycle_state)

        try:
            sprint_args = _runner_sprint_args(args, cycle_number, active_ids, sprint_sim_log)
            sprint_rc = cmd_sprint(sprint_args)
            if sprint_rc:
                append_event(events_path, "sprint_nonzero_exit", {"cycle_id": cycle_id, "return_code": sprint_rc})
                if args.fail_fast:
                    return int(sprint_rc)

            advance_args = _runner_advance_args(args, active_ids, advance_sim_log)
            advance_rc = cmd_advance(advance_args)
            if advance_rc:
                append_event(events_path, "advance_nonzero_exit", {"cycle_id": cycle_id, "return_code": advance_rc})
                if args.fail_fast:
                    return int(advance_rc)

            store = TheoryStore(Path(args.theory_dir))
            promotions = harvest_promotions(
                store=store,
                target_delay=args.delay,
                cycle_id=cycle_id,
                timestamp=utc_timestamp(),
                sim_logs=scored_logs,
            )
            appended = append_promotions_to_ledger(ledger_path, promotions)
            if appended:
                active_ids = append_active_alpha_ids(active_path, [str(row["alpha_id"]) for row in appended])
                session_promotions += len(appended)
                append_event(
                    events_path,
                    "promotions_harvested",
                    {
                        "cycle_id": cycle_id,
                        "n_promotions": len(appended),
                        "alpha_ids": [row["alpha_id"] for row in appended],
                        "ledger_path": str(ledger_path),
                        "active_alpha_count": len(active_ids),
                    },
                )

            state.update(
                {
                    "cycle_id": cycle_id,
                    "cycle_number": cycle_number,
                    "last_completed_at": utc_timestamp(),
                    "last_cycle_status": "completed",
                    "last_cycle_promotions": len(appended),
                    "session_promotions": session_promotions,
                    "active_alpha_count": len(active_ids),
                    "ledger_path": str(ledger_path),
                    "active_alpha_ids_path": str(active_path),
                    "consecutive_failures": 0,
                }
            )
            write_run_state(state_path, state)
            append_event(
                events_path,
                "cycle_completed",
                {
                    "cycle_id": cycle_id,
                    "cycle_number": cycle_number,
                    "n_promotions": len(appended),
                    "session_promotions": session_promotions,
                },
            )

            if args.stop_after_promotions > 0 and session_promotions >= args.stop_after_promotions:
                append_event(
                    events_path,
                    "runner_stopped",
                    {
                        "reason": "stop_after_promotions",
                        "session_promotions": session_promotions,
                        "cycle_number": cycle_number,
                    },
                )
                return 0
        except KeyboardInterrupt:
            state.update(
                {
                    "cycle_id": cycle_id,
                    "cycle_number": cycle_number,
                    "last_cycle_status": "interrupted",
                    "interrupted_at": utc_timestamp(),
                    "session_promotions": session_promotions,
                }
            )
            write_run_state(state_path, state)
            append_event(events_path, "runner_interrupted", {"cycle_id": cycle_id, "cycle_number": cycle_number})
            return 130
        except Exception as exc:
            failure_count = int(state.get("consecutive_failures") or 0) + 1
            error_payload = {
                "cycle_id": cycle_id,
                "cycle_number": cycle_number,
                "error_type": exc.__class__.__name__,
                "error": str(exc),
                "traceback": traceback.format_exc(),
                "consecutive_failures": failure_count,
            }
            state.update(
                {
                    "cycle_id": cycle_id,
                    "cycle_number": cycle_number,
                    "last_cycle_status": "error",
                    "last_error": error_payload,
                    "consecutive_failures": failure_count,
                    "session_promotions": session_promotions,
                    "updated_at": utc_timestamp(),
                }
            )
            write_run_state(state_path, state)
            append_event(events_path, "cycle_error", error_payload)
            print(f"RUNNER: cycle {cycle_id} failed: {exc}")
            if args.fail_fast:
                raise

        if args.max_cycles <= 0 or attempted_cycles < args.max_cycles:
            sleep_seconds = max(0.0, float(args.sleep_minutes) * 60.0)
            if sleep_seconds > 0:
                print(f"RUNNER: sleeping {args.sleep_minutes} minutes before next cycle")
                time.sleep(sleep_seconds)

    append_event(
        events_path,
        "runner_stopped",
        {
            "reason": "max_cycles",
            "attempted_cycles": attempted_cycles,
            "session_promotions": session_promotions,
        },
    )
    return 0


def _runner_sprint_args(
    args: argparse.Namespace,
    cycle_number: int,
    active_ids: list[str],
    sim_log: Path,
) -> argparse.Namespace:
    sub_args = argparse.Namespace(**vars(args))
    sub_args.stage = "probe"
    sub_args.alpha_ids = " ".join(active_ids)
    sub_args.alpha_id = []
    sub_args.skip_index = not _refresh_due(cycle_number, args.refresh_index_every)
    sub_args.skip_fields = not _refresh_due(cycle_number, args.refresh_fields_every)
    sub_args.skip_portfolio = not _refresh_due(cycle_number, args.refresh_portfolio_every)
    sub_args.sim_log = str(sim_log)
    sub_args.combined_out = ""
    return sub_args


def _runner_advance_args(
    args: argparse.Namespace,
    active_ids: list[str],
    sim_log: Path,
) -> argparse.Namespace:
    sub_args = argparse.Namespace(**vars(args))
    sub_args.alpha_ids = " ".join(active_ids)
    sub_args.alpha_id = []
    sub_args.theory_id = ""
    sub_args.skip_portfolio = True
    sub_args.sim_log = str(sim_log)
    return sub_args


def _refresh_due(cycle_number: int, every: int) -> bool:
    if cycle_number <= 1:
        return True
    if every <= 0:
        return False
    return cycle_number % every == 0


def _scored_log_for_runner(sim_log: Path, *, no_admission_score: bool) -> Path:
    if no_admission_score:
        return sim_log
    return sim_log.with_name(f"{sim_log.stem}_scored{sim_log.suffix}")


def cmd_summarize(args: argparse.Namespace) -> int:
    store = TheoryStore(Path(args.theory_dir))
    theories = store.list_theories()
    counts = defaultdict(int)
    for theory in theories:
        counts[theory.status] += 1
    print(f"Theories: {len(theories)}  statuses={dict(counts)}")
    for theory in theories:
        decision = store.read_decision(theory.theory_id)
        line = f"{theory.status:<10} {theory.theory_id:<52} {theory.title}"
        if decision.get("decision"):
            line += f" | {decision.get('decision')}: {decision.get('reason')}"
        for delay in (0, 1):
            branch = store.read_decision(theory.theory_id, target_delay=delay)
            if branch.get("decision"):
                line += f" | d{delay} {branch.get('decision')}: {branch.get('reason')}"
        print(line)
    return 0


def cmd_sprint(args: argparse.Namespace) -> int:
    """Run the normal delay-specific research loop from one entrypoint."""
    store = TheoryStore(Path(args.theory_dir))

    if args.skip_index:
        index = _ensure_index(args)
    else:
        print("STEP 1/6 index: rebuilding history index")
        summary = HistoryIndexer(
            index_path=Path(args.index_path),
            summary_path=Path(args.summary_path),
            field_csv=Path(args.fields_csv),
        ).build()
        print(f"Indexed records: {summary.get('total_records', 0)}")
        index = HistoryIndex(index_path=Path(args.index_path), summary_path=Path(args.summary_path))

    if args.skip_portfolio:
        portfolio_context = load_portfolio_context(Path(args.portfolio_context_path))
        print(f"STEP 2/6 portfolio: using existing context ({portfolio_context.get('n_submitted', 0)} submitted)")
    else:
        print("STEP 2/6 portfolio: refreshing submitted alpha context")
        alpha_ids = _parse_alpha_ids(args.alpha_ids, args.alpha_id)
        portfolio_context = fetch_and_write_portfolio_context(
            credentials=args.credentials,
            competition_id=args.competition_id,
            alpha_ids=alpha_ids,
            include_account_scan=args.include_account_scan,
            out_path=Path(args.portfolio_context_path),
            field_csv=Path(args.fields_csv),
            limit=args.limit,
            max_pages=args.max_pages,
            sleep_seconds=args.sleep,
        )
        print(f"Fetched submitted alphas: {portfolio_context.get('n_submitted', 0)}")

    if args.skip_fields:
        print(f"STEP 3/6 fields: using existing delay={args.delay} catalog")
    else:
        print(f"STEP 3/6 fields: refreshing delay={args.delay} catalog")
        if args.fields_from_metadata:
            records = records_from_metadata(
                metadata_path=Path(args.fields_from_metadata),
                delay=args.delay,
                universe=args.field_universe,
            )
            source = str(args.fields_from_metadata)
        else:
            records = fetch_delay_records(
                credentials=args.credentials,
                delay=args.delay,
                universe=args.field_universe,
            )
            source = "Brain API"
        if not records:
            raise RuntimeError(f"No fields found for delay={args.delay} universe={args.field_universe} from {source}")
        field_catalog = write_delay_catalog(
            delay=args.delay,
            records=records,
            output_dir=Path(args.delay_fields_dir),
            universe=args.field_universe,
            credentials=args.credentials,
        )
        print(f"Wrote field catalog: {field_catalog} ({len(records)} records)")

    field_path, field_summary = _delay_field_summary(args)

    print(f"STEP 4/6 propose: requesting {args.n} delay={args.delay} theories")
    agent = TheoryProposerAgent()
    try:
        theories = agent.propose(
            history_summary=index.summary(),
            existing_theories=store.list_theories(),
            portfolio_context=portfolio_context,
            target_delay=args.delay,
            field_catalog_summary=field_summary,
            sprint_focus=args.sprint_focus,
            n=args.n,
            mock=args.mock,
        )
        proposer_source = "mock" if args.mock else "llm"
    except ValueError as exc:
        print(f"WARNING: LLM proposer failed ({exc}); using deterministic fallback theories.")
        theories = mock_theories(sprint_focus=args.sprint_focus, n=args.n)
        proposer_source = "fallback"
    theories = [_with_target_delay(theory, args.delay) for theory in theories]
    proposal_path = _write_proposals(args, theories)
    for theory in theories:
        store.save_theory(
            theory,
            event_type="created",
            payload={
                "source": "sprint",
                "proposer_source": proposer_source,
                "mock": args.mock,
                "target_delay": args.delay,
                "field_catalog": str(field_path),
                "delay_definition": DELAY_DEFINITION,
            },
        )
    print(f"Proposed theories: {len(theories)}")
    print(f"Wrote proposals:   {proposal_path}")

    print("STEP 5/6 critique: filtering sprint theories")
    accepted = _critique_theories(
        args,
        index=index,
        store=store,
        theories=theories,
        portfolio_context=portfolio_context,
    )
    print(f"Accepted sprint theories: {len(accepted)}")

    print(f"STEP 6/6 generate: {args.max_candidates_per_theory} {args.stage} candidates per accepted theory")
    candidates, combined = _generate_for_theories(
        args,
        index=index,
        store=store,
        theories=accepted,
        field_path=field_path,
        field_summary=field_summary,
        portfolio_context=portfolio_context,
    )
    if not combined:
        print("No combined plan was written; stopping before simulation.")
        return 0

    if not args.simulate:
        print("Simulation skipped. Re-run with --simulate to send the generated plan to Brain.")
        return 0
    if proposer_source == "fallback" and not args.allow_fallback_simulation:
        print(
            "Simulation skipped because proposer_source=fallback. "
            "The combined plan was written for inspection, but fallback theories are not simulated "
            "unless --allow-fallback-simulation is set."
        )
        return 0

    sim_log = Path(args.sim_log) if args.sim_log else _default_sim_log(args)
    sim_log.parent.mkdir(parents=True, exist_ok=True)
    print(f"SIMULATE: submitting {len(candidates)} candidates via economic Brain submitter")
    new_records = submit_economic_plan(
        str(combined),
        str(sim_log),
        batch_size=args.batch_size,
        credentials=args.credentials,
        max_candidates=args.max_submit_candidates,
    )
    print(f"Simulation log: {sim_log}")
    print(f"New simulation records: {len(new_records)}")

    if args.no_ingest:
        _merge_sim_log_to_master(sim_log)
        print("Ingest skipped because --no-ingest was set.")
        return 0

    ingest_log = _score_log_if_needed(args, sim_log)
    _merge_sim_log_to_master(ingest_log)
    print("INGEST: attaching simulation results to theory histories")
    ingest_args = argparse.Namespace(**vars(args))
    ingest_args.results_jsonl = str(ingest_log)
    return cmd_ingest(ingest_args)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="LLM economic hypothesis search workspace")
    parser.add_argument("--artifacts-dir", default=str(DEFAULT_ARTIFACTS_DIR))
    parser.add_argument("--theory-dir", default=str(DEFAULT_THEORY_DIR))
    parser.add_argument("--index-path", default=str(DEFAULT_INDEX_PATH))
    parser.add_argument("--summary-path", default=str(DEFAULT_SUMMARY_PATH))
    parser.add_argument("--fields-csv", default=str(DEFAULT_FIELD_CSV))
    parser.add_argument("--delay-fields-dir", default=str(DEFAULT_DELAY_FIELD_DIR))
    parser.add_argument("--field-universe", default="TOP3000")
    parser.add_argument("--portfolio-context-path", default=str(DEFAULT_PORTFOLIO_CONTEXT_PATH))
    parser.add_argument(
        "--llm-debug-dir",
        default="",
        help="Write raw LLM request/response payloads to this directory for debugging",
    )
    parser.add_argument(
        "--anthropic-max-tokens",
        type=int,
        default=0,
        help="Override Anthropic max_tokens for JSON-producing LLM calls; default is 12000",
    )
    parser.add_argument(
        "--anthropic-model",
        default="",
        help="Override Anthropic model for all LLM agents; default is claude-opus-4-7",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("index", help="Build the history SQLite index and summary")
    p.set_defaults(func=cmd_index)

    p = sub.add_parser("fields", help="Build delay-specific field catalogs")
    p.add_argument("--delay", required=True, choices=["0", "1", "both"])
    p.add_argument("--credentials", default="brain.txt")
    p.add_argument(
        "--from-metadata",
        default="",
        help=f"Build from local metadata JSON instead of Brain API, e.g. {DEFAULT_METADATA_PATH}",
    )
    p.set_defaults(func=cmd_fields)

    p = sub.add_parser("portfolio", help="Fetch submitted account alphas into portfolio context")
    p.add_argument("--credentials", default="brain.txt")
    p.add_argument("--competition-id", default=DEFAULT_COMPETITION_ID)
    p.add_argument("--include-account-scan", action="store_true")
    p.add_argument("--alpha-id", action="append", default=[], help="Explicit submitted alpha id; can be repeated")
    p.add_argument("--alpha-ids", default="", help="Whitespace- or comma-separated explicit submitted alpha ids")
    p.add_argument("--limit", type=int, default=100)
    p.add_argument("--max-pages", type=int, default=0, help="0 means fetch all pages")
    p.add_argument("--sleep", type=float, default=0.1)
    p.set_defaults(func=cmd_portfolio)

    p = sub.add_parser("propose", help="Propose economic theories")
    p.add_argument("--delay", required=True, type=int, choices=[0, 1])
    p.add_argument("--mock", action="store_true", help="Use deterministic mock theories; no LLM call")
    p.add_argument("--n", type=int, default=6)
    p.add_argument("--sprint-focus", default="")
    p.set_defaults(func=cmd_propose)

    p = sub.add_parser("critique", help="Critique proposed theories for novelty")
    p.add_argument("--delay", required=True, type=int, choices=[0, 1])
    p.add_argument("--theory-id", default="")
    p.add_argument("--no-llm-critic", action="store_true", help="Use deterministic novelty critic only")
    p.add_argument("--use-llm", action="store_true", help=argparse.SUPPRESS)
    p.set_defaults(func=cmd_critique)

    p = sub.add_parser("generate", help="Generate candidate JSONL plans for accepted theories")
    p.add_argument("--delay", required=True, type=int, choices=[0, 1])
    p.add_argument("--theory-id", default="")
    p.add_argument("--stage", default="probe", choices=["probe", "iterate", "expand"])
    p.add_argument("--max-candidates", type=int, default=0)
    p.add_argument("--mock", action="store_true", help="Use mock LLM expression output; no LLM call")
    p.add_argument("--max-generation-repairs", type=int, default=1)
    p.add_argument("--min-distinct-skeletons", type=int, default=4)
    p.add_argument("--max-window-only-variants", type=int, default=3)
    p.add_argument("--combined-out", default="")
    p.set_defaults(func=cmd_generate)

    p = sub.add_parser("ingest", help="Attach simulation results back to theory histories")
    p.add_argument("--results-jsonl", required=True)
    p.set_defaults(func=cmd_ingest)

    p = sub.add_parser("summarize", help="Print theory status summary")
    p.set_defaults(func=cmd_summarize)

    p = sub.add_parser(
        "sprint",
        help="Run index -> portfolio -> fields -> propose -> critique -> generate, with optional simulation and ingest",
    )
    p.add_argument("--delay", required=True, type=int, choices=[0, 1])
    p.add_argument("--n", type=int, default=10, help="Number of hypotheses to ask the proposer for")
    p.add_argument("--stage", default="probe", choices=["probe", "expand"])
    p.add_argument("--max-candidates-per-theory", type=int, default=12)
    p.add_argument("--sprint-focus", default="")
    p.add_argument("--mock", action="store_true", help="Use deterministic mock theories; no LLM proposer call")
    p.add_argument("--no-llm-critic", action="store_true", help="Use deterministic novelty critic only")
    p.add_argument("--use-llm-critic", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--max-generation-repairs", type=int, default=1)
    p.add_argument("--min-distinct-skeletons", type=int, default=4)
    p.add_argument("--max-window-only-variants", type=int, default=3)
    p.add_argument("--skip-index", action="store_true", help="Reuse existing history index if compatible")
    p.add_argument("--skip-portfolio", action="store_true", help="Reuse existing submitted portfolio context")
    p.add_argument("--skip-fields", action="store_true", help="Reuse existing delay-specific field catalog")
    p.add_argument(
        "--fields-from-metadata",
        default="",
        help=f"Build field catalog from local metadata JSON instead of Brain API, e.g. {DEFAULT_METADATA_PATH}",
    )
    p.add_argument("--credentials", default="brain.txt")
    p.add_argument("--competition-id", default=DEFAULT_COMPETITION_ID)
    p.add_argument("--include-account-scan", action="store_true")
    p.add_argument("--alpha-id", action="append", default=[], help="Explicit submitted alpha id; can be repeated")
    p.add_argument("--alpha-ids", default="", help="Whitespace- or comma-separated explicit submitted alpha ids")
    p.add_argument("--limit", type=int, default=100)
    p.add_argument("--max-pages", type=int, default=0, help="0 means fetch all pages")
    p.add_argument("--sleep", type=float, default=0.1)
    p.add_argument("--combined-out", default="")
    p.add_argument("--simulate", action="store_true", help="Run the generated plan through the economic Brain submitter")
    p.add_argument(
        "--allow-fallback-simulation",
        action="store_true",
        help="Allow Brain simulation if the LLM proposer failed and sprint used deterministic fallback theories",
    )
    p.add_argument("--sim-log", default="", help="Simulation log path; defaults under artifacts for the target delay")
    p.add_argument("--batch-size", type=int, default=3)
    p.add_argument("--max-submit-candidates", type=int, default=0, help="Cap submitted candidates; 0 means no cap")
    p.add_argument("--no-admission-score", action="store_true", help="Skip same-delay pool scoring before ingest")
    p.add_argument("--admission-top-per-theory", type=int, default=3)
    p.add_argument("--admission-min-sharpe", type=float, default=1.0)
    p.add_argument("--admission-min-brain-score-delta", type=float, default=0.0)
    p.add_argument("--admission-min-fallback-delta-sharpe", type=float, default=0.0)
    p.add_argument("--performance-attempts", type=int, default=5)
    p.add_argument("--performance-wait", type=float, default=2.0)
    p.add_argument("--max-pnl-attempts", type=int, default=3)
    p.add_argument("--no-ingest", action="store_true", help="Do not ingest simulation results after --simulate")
    p.set_defaults(func=cmd_sprint)

    p = sub.add_parser("advance", help="Run one decision-aware iterate/expand step for eligible theory branches")
    p.add_argument("--delay", required=True, type=int, choices=[0, 1])
    p.add_argument("--theory-id", default="", help="Optionally advance only one theory id")
    p.add_argument("--mock", action="store_true", help="Use mock LLM expression output; no LLM call")
    p.add_argument("--max-iterate-candidates", type=int, default=16)
    p.add_argument("--max-expand-candidates", type=int, default=36)
    p.add_argument("--max-generation-repairs", type=int, default=1)
    p.add_argument("--min-distinct-skeletons", type=int, default=4)
    p.add_argument("--max-window-only-variants", type=int, default=3)
    p.add_argument("--skip-portfolio", action="store_true", help="Reuse existing submitted portfolio context")
    p.add_argument("--credentials", default="brain.txt")
    p.add_argument("--competition-id", default=DEFAULT_COMPETITION_ID)
    p.add_argument("--include-account-scan", action="store_true")
    p.add_argument("--alpha-id", action="append", default=[], help="Explicit submitted alpha id; can be repeated")
    p.add_argument("--alpha-ids", default="", help="Whitespace- or comma-separated explicit submitted alpha ids")
    p.add_argument("--limit", type=int, default=100)
    p.add_argument("--max-pages", type=int, default=0, help="0 means fetch all pages")
    p.add_argument("--sleep", type=float, default=0.1)
    p.add_argument("--simulate", action="store_true", help="Run second-stage plans through the economic Brain submitter")
    p.add_argument("--sim-log", default="", help="Simulation log path; defaults under artifacts for the target delay")
    p.add_argument("--batch-size", type=int, default=3)
    p.add_argument("--max-submit-candidates", type=int, default=0, help="Cap submitted candidates; 0 means no cap")
    p.add_argument("--no-ingest", action="store_true", help="Do not ingest simulation results after --simulate")
    p.add_argument("--no-admission-score", action="store_true", help="Skip same-delay pool scoring before ingest")
    p.add_argument("--admission-top-per-theory", type=int, default=3)
    p.add_argument("--admission-min-sharpe", type=float, default=1.0)
    p.add_argument("--admission-min-brain-score-delta", type=float, default=0.0)
    p.add_argument("--admission-min-fallback-delta-sharpe", type=float, default=0.0)
    p.add_argument("--performance-attempts", type=int, default=5)
    p.add_argument("--performance-wait", type=float, default=2.0)
    p.add_argument("--max-pnl-attempts", type=int, default=3)
    p.set_defaults(func=cmd_advance)

    p = sub.add_parser("run-forever", help="Continuously run delay-specific sprint and advance cycles")
    p.add_argument("--delay", required=True, type=int, choices=[0, 1])
    p.add_argument("--n", type=int, default=10, help="Number of hypotheses to ask the proposer for each cycle")
    p.add_argument("--max-candidates-per-theory", type=int, default=12)
    p.add_argument("--max-iterate-candidates", type=int, default=16)
    p.add_argument("--max-expand-candidates", type=int, default=36)
    p.add_argument("--sprint-focus", default="")
    p.add_argument("--mock", action="store_true", help="Use mock theories/expression output")
    p.add_argument("--no-llm-critic", action="store_true", help="Use deterministic novelty critic only")
    p.add_argument("--max-generation-repairs", type=int, default=1)
    p.add_argument("--min-distinct-skeletons", type=int, default=4)
    p.add_argument("--max-window-only-variants", type=int, default=3)
    p.add_argument(
        "--fields-from-metadata",
        default="",
        help=f"Build field catalog from local metadata JSON instead of Brain API, e.g. {DEFAULT_METADATA_PATH}",
    )
    p.add_argument("--credentials", default="brain.txt")
    p.add_argument("--competition-id", default=DEFAULT_COMPETITION_ID)
    p.add_argument("--include-account-scan", action="store_true")
    p.add_argument("--alpha-id", action="append", default=[], help="Explicit submitted alpha id; can be repeated")
    p.add_argument("--alpha-ids", default="", help="Whitespace- or comma-separated explicit submitted alpha ids")
    p.add_argument("--limit", type=int, default=100)
    p.add_argument("--max-pages", type=int, default=0, help="0 means fetch all pages")
    p.add_argument("--sleep", type=float, default=0.1)
    p.add_argument("--simulate", action="store_true", help="Run generated plans through the economic Brain submitter")
    p.add_argument(
        "--allow-fallback-simulation",
        action="store_true",
        help="Allow Brain simulation if the LLM proposer failed and sprint used deterministic fallback theories",
    )
    p.add_argument("--batch-size", type=int, default=3)
    p.add_argument("--max-submit-candidates", type=int, default=0, help="Cap submitted candidates; 0 means no cap")
    p.add_argument("--no-admission-score", action="store_true", help="Skip same-delay pool scoring before ingest")
    p.add_argument("--admission-top-per-theory", type=int, default=3)
    p.add_argument("--admission-min-sharpe", type=float, default=1.0)
    p.add_argument("--admission-min-brain-score-delta", type=float, default=0.0)
    p.add_argument("--admission-min-fallback-delta-sharpe", type=float, default=0.0)
    p.add_argument("--performance-attempts", type=int, default=5)
    p.add_argument("--performance-wait", type=float, default=2.0)
    p.add_argument("--max-pnl-attempts", type=int, default=3)
    p.add_argument("--no-ingest", action="store_true", help="Do not ingest simulation results after --simulate")
    p.add_argument("--sleep-minutes", type=float, default=10.0)
    p.add_argument("--max-cycles", type=int, default=0, help="0 means run indefinitely")
    p.add_argument("--stop-after-promotions", type=int, default=0, help="0 means do not stop because of promotions")
    p.add_argument("--refresh-index-every", type=int, default=5)
    p.add_argument("--refresh-fields-every", type=int, default=24)
    p.add_argument("--refresh-portfolio-every", type=int, default=1)
    p.add_argument("--continuous-dir", default="", help="Directory for run state, events, ledger, and active alpha ids")
    p.add_argument("--state-path", default="")
    p.add_argument("--events-path", default="")
    p.add_argument("--ledger-path", default="")
    p.add_argument("--active-alpha-ids-path", default="")
    p.add_argument("--lock-file", default="")
    p.add_argument("--fail-fast", action="store_true", help="Stop on the first cycle exception or nonzero stage exit")
    p.set_defaults(func=cmd_run_forever)

    p = sub.add_parser("submit", help="Submit an economic candidate plan without legacy template abandonment")
    p.add_argument("--plan", required=True)
    p.add_argument("--sim-log", required=True)
    p.add_argument("--batch-size", type=int, default=3)
    p.add_argument("--credentials", default="brain.txt")
    p.add_argument("--max-candidates", type=int, default=0)
    p.add_argument("--ingest", action="store_true", help="Ingest the sim log into theory histories after submission")
    p.set_defaults(func=cmd_submit)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if getattr(args, "llm_debug_dir", ""):
        import os

        os.environ["LLM_ECONOMIC_SEARCH_DEBUG_DIR"] = str(args.llm_debug_dir)
    if getattr(args, "anthropic_max_tokens", 0):
        import os

        os.environ["ANTHROPIC_MAX_TOKENS"] = str(args.anthropic_max_tokens)
    if getattr(args, "anthropic_model", ""):
        import os

        os.environ["ANTHROPIC_MODEL"] = str(args.anthropic_model)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
