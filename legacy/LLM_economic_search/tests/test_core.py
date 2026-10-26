from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import LLM_economic_search.cli as cli_module
from alpha_grammar.models import CandidateSpec
from alpha_llm.semantic_novelty import AlphaSemanticNoveltyGate, SemanticThresholds
from alpha_llm.similarity import SimilarityFilter
from LLM_economic_search.candidate_generation import CandidateExpander
from LLM_economic_search.cli import _finalize_second_stage, main as cli_main
from LLM_economic_search.continuous_runner import (
    ContinuousRunLock,
    append_active_alpha_ids,
    append_promotions_to_ledger,
    harvest_promotions,
    merge_alpha_ids,
    read_active_alpha_ids,
)
from LLM_economic_search.critique import DeterministicNoveltyCritic, DeterministicResultInterpreter
from LLM_economic_search.expression_generation import LLMExpressionGenerator
from LLM_economic_search.expression_validation import AcceptedExpression, ExpressionValidator, windowless_fingerprint
from LLM_economic_search.history_index import HistoryIndex, HistoryIndexer
from LLM_economic_search.llm_agents import _extract_json, mock_theories
from LLM_economic_search.portfolio_scoring import score_records_for_admission
from LLM_economic_search.portfolio_context import build_portfolio_context, compact_portfolio_context
from LLM_economic_search.schemas import TheorySpec
from LLM_economic_search.semantic_gate import SemanticNoveltyGate, retrieve_semantic_neighbors
from LLM_economic_search.theory_store import TheoryStore


class _FakeSemanticAgent:
    def __init__(self, scorecards: list[dict[str, object]]) -> None:
        self.scorecards = scorecards
        self.payloads: list[dict[str, object]] = []

    def score_candidates(self, *, payload: dict[str, object]) -> dict[str, object]:
        self.payloads.append(payload)
        return {"scorecards": self.scorecards}


class _FakeAlphaSemanticAgent:
    def __init__(self, scorecards: list[dict[str, object]] | None = None, exc: Exception | None = None) -> None:
        self.scorecards = scorecards or []
        self.exc = exc
        self.payloads: list[dict[str, object]] = []

    def score_candidates(self, *, payload: dict[str, object]) -> dict[str, object]:
        self.payloads.append(payload)
        if self.exc:
            raise self.exc
        return {"scorecards": self.scorecards}


class PackageReorgCompatibilityTests(unittest.TestCase):
    def test_alpha_search_facades_point_to_wq_alpha(self) -> None:
        import alpha_search.submit as old_submit
        import alpha_search.wq_api as old_api
        import wq_alpha.brain.api as new_api
        import wq_alpha.brain.submit as new_submit

        self.assertIs(old_api.sign_in, new_api.sign_in)
        self.assertIs(old_api.fetch_yearly_stats, new_api.fetch_yearly_stats)
        self.assertIs(old_submit.run, new_submit.run)
        self.assertIs(old_submit.DEFAULT_SETTINGS, new_submit.DEFAULT_SETTINGS)
        self.assertIs(getattr(old_submit, "_regime_key"), new_submit._regime_key)

    def test_alpha_multi_agent_facades_point_to_wq_alpha(self) -> None:
        if importlib.util.find_spec("anthropic") is None:
            self.skipTest("anthropic package is not installed")

        import alpha_multi.enhance_agent as old_enhance
        import alpha_multi.repair_agent as old_repair
        import wq_alpha.agents.enhance as new_enhance
        import wq_alpha.agents.repair as new_repair

        self.assertIs(old_repair.AlphaRepairAgent, new_repair.AlphaRepairAgent)
        self.assertIs(old_enhance.AlphaEnhanceAgent, new_enhance.AlphaEnhanceAgent)
        self.assertIs(old_enhance.OverseerAgent, new_enhance.OverseerAgent)

    def test_tools_facades_point_to_wq_alpha(self) -> None:
        import tools.portfolio_admission as old_admission
        import tools.sim_log as old_logs
        import wq_alpha.logs as new_logs
        import wq_alpha.portfolio.admission as new_admission

        self.assertIs(old_logs.merge_session_to_master, new_logs.merge_session_to_master)
        self.assertEqual(old_logs.MASTER, new_logs.MASTER)
        self.assertIs(old_admission.score_record_admission, new_admission.score_record_admission)

    def test_master_merge_preserves_regime_variants_and_compacts_raw_blobs(self) -> None:
        import wq_alpha.logs as sim_logs

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            master = tmp / "master.jsonl"
            details = tmp / "details"
            session = tmp / "session.jsonl"
            rows = [
                {
                    "expression": " rank(close) ",
                    "status": "done",
                    "is_sharpe": 1.0,
                    "settings": {
                        "delay": 0,
                        "neutralization": "MARKET",
                        "universe": "TOP3000",
                        "decay": 0,
                        "truncation": 0.01,
                    },
                    "alpha_id": "A0",
                    "checks": [],
                    "raw_alpha_detail": {"is": {"checks": []}},
                },
                {
                    "expression": "rank(close)",
                    "status": "done",
                    "is_sharpe": 1.5,
                    "settings": {
                        "delay": 1,
                        "neutralization": "MARKET",
                        "universe": "TOP3000",
                        "decay": 0,
                        "truncation": 0.01,
                    },
                    "alpha_id": "A1",
                    "checks": [],
                    "raw_progress_result": {"alpha": "A1"},
                },
            ]
            with session.open("w") as handle:
                for row in rows:
                    handle.write(json.dumps(row) + "\n")

            with patch.object(sim_logs, "MASTER", master), patch.object(sim_logs, "DETAILS_DIR", details):
                self.assertEqual(sim_logs.merge_session_to_master(session), (2, 0))

            merged = [json.loads(line) for line in master.read_text().splitlines()]
            self.assertEqual(len(merged), 2)
            self.assertEqual({row["settings"]["delay"] for row in merged}, {0, 1})
            self.assertTrue(all("raw_alpha_detail" not in row for row in merged))
            self.assertTrue(all("raw_progress_result" not in row for row in merged))
            self.assertTrue((details / "A0_raw_alpha_detail.json").exists())
            self.assertTrue((details / "A1_raw_progress_result.json").exists())

    def test_compatibility_cli_help_entrypoints(self) -> None:
        env = os.environ.copy()
        env["PYTHONDONTWRITEBYTECODE"] = "1"
        modules = [
            "alpha_search.pipeline",
            "alpha_search.submit",
            "alpha_multi.run_plan",
            "tools.batch_workshop",
            "LLM_economic_search.cli",
        ]
        for module in modules:
            with self.subTest(module=module):
                result = subprocess.run(
                    [sys.executable, "-m", module, "--help"],
                    cwd=Path(__file__).resolve().parents[2],
                    env=env,
                    text=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    timeout=20,
                    check=False,
                )
                self.assertEqual(result.returncode, 0, result.stderr)


def _scorecard(candidate_key: str, score: float, tags: list[str] | None = None) -> dict[str, object]:
    return {
        "candidate_key": candidate_key,
        "overall_similarity_score": score,
        "economic_mechanism_similarity": score,
        "field_role_similarity": score,
        "operator_structure_similarity": score,
        "trading_condition_similarity": score,
        "portfolio_redundancy_risk": 20,
        "confidence": 90,
        "reason": "mock scorecard",
        "nearest_matching_mechanism": "mock neighbor",
        "semantic_tags": tags or [],
    }


def _alpha_scorecard(candidate_key: str, score: float, tags: list[str] | None = None) -> dict[str, object]:
    return {
        "candidate_key": candidate_key,
        "overall_similarity_score": score,
        "economic_mechanism_similarity": score,
        "field_role_similarity": score,
        "operator_structure_similarity": score,
        "trading_condition_similarity": score,
        "portfolio_redundancy_risk": 20,
        "confidence": 90,
        "reason": "mock alpha scorecard",
        "nearest_matching_mechanism": "mock alpha neighbor",
        "semantic_tags": tags or [],
    }


def _alpha_candidate(
    expression: str,
    *,
    fields: tuple[str, ...] = ("alpha_field",),
    template_id: str = "rank|ts_mean",
    dataset_id: str = "analyst4",
    category: str = "analyst",
) -> CandidateSpec:
    return CandidateSpec(
        candidate_id=f"test_{abs(hash(expression))}",
        family_id="test_family",
        category=category,
        dataset_id=dataset_id,
        intention_id="test_intention",
        intention_label="Test intention",
        template_id=template_id,
        economic_source="test",
        search_lane="llm_grammar",
        expression=expression,
        fields=fields,
        field_types=tuple("MATRIX" for _ in fields),
        size_proxy_flags=tuple(False for _ in fields),
        cluster_ids=tuple(f"{dataset_id}:test:{field}" for field in fields),
        group="subindustry",
        params={},
        horizon_bucket="medium",
        settings_signature="delay1",
        semantic_signature="test",
        rationale_type="test",
        tree_depth=2,
        priority_score=1.0,
    )


def _accepted_expr(expression: str) -> AcceptedExpression:
    return AcceptedExpression(
        source={
            "expression": expression,
            "field_roles": {"signal": "news_sent"},
            "mechanism_translation": "news sentiment change",
            "expected_sign": "positive",
            "horizon_days": 20,
            "novelty_argument": "test",
            "risk_notes": [],
            "generation_source": "unit_test",
        },
        expression=expression,
        fields=["news_sent"],
        field_roles={"signal": "news_sent"},
        operator_skeleton=["rank", "ts_delta"],
        windowless_fingerprint=windowless_fingerprint(expression),
        role_binding_fingerprint="signal=news_sent",
        datasets=["news12"],
        categories=["news"],
        cluster_ids=["news12:news_sentiment"],
        novelty_score=1.0,
        cross_delay_retest=False,
    )


class AlphaLLMNoveltyTests(unittest.TestCase):
    def test_similarity_filter_blocks_normalized_external_expression(self) -> None:
        novelty = SimilarityFilter()
        novelty.bootstrap(
            [{"expression": " rank ( assets ) ", "status": "done"}],
            count_skeletons=False,
            source_name="master",
        )

        verdict = novelty.is_too_similar(
            _alpha_candidate("rank(assets)", fields=("assets",), template_id="rank", dataset_id="fundamental6")
        )

        self.assertTrue(verdict.skip)
        self.assertEqual(verdict.reason, "normalized_exact_match")

    def test_external_history_does_not_saturate_hard_skeleton_counter(self) -> None:
        records = [
            {
                "expression": f"rank(ts_mean(field_{i}, 20))",
                "fields": [f"field_{i}"],
                "template_id": "rank|ts_mean",
                "status": "done",
            }
            for i in range(25)
        ]
        external_only = SimilarityFilter(max_skeleton_count=20)
        external_only.bootstrap(records, count_skeletons=False, source_name="master")
        candidate = _alpha_candidate(
            "rank(ts_mean(new_field, 20))",
            fields=("new_field",),
            template_id="rank|ts_mean",
        )

        self.assertFalse(external_only.is_too_similar(candidate).skip)

        active_history = SimilarityFilter(max_skeleton_count=20)
        active_history.bootstrap(records, count_skeletons=True, source_name="llm_sim")

        verdict = active_history.is_too_similar(candidate)
        self.assertTrue(verdict.skip)
        self.assertIn("skeleton_count=25", verdict.reason)

    def test_master_style_rows_infer_fields_and_operator_neighbors(self) -> None:
        novelty = SimilarityFilter()
        novelty.bootstrap(
            [{"expression": "group_rank(ts_rank(assets, 60), market)", "status": "done"}],
            count_skeletons=False,
            source_name="master",
        )

        neighbors = novelty.nearest_neighbors(
            _alpha_candidate(
                "group_rank(ts_zscore(assets, 20), market)",
                fields=("assets",),
                template_id="group_rank|ts_zscore",
                dataset_id="fundamental6",
            ),
            min_score=0,
        )

        self.assertGreaterEqual(len(neighbors), 1)
        self.assertEqual(neighbors[0]["fields"], ["assets"])
        self.assertEqual(neighbors[0]["operator_skeleton"], ["group_rank", "ts_rank"])

    def test_alpha_semantic_gate_routes_scorecards_with_soft_quota(self) -> None:
        novelty = SimilarityFilter()
        novelty.bootstrap(
            [
                {
                    "expression": "rank(ts_mean(alpha_field, 20))",
                    "fields": ["alpha_field"],
                    "template_id": "rank|ts_mean",
                    "dataset_id": "analyst4",
                    "category": "analyst",
                    "status": "done",
                }
            ],
            count_skeletons=False,
            source_name="master",
        )
        candidates = [_alpha_candidate(f"rank(ts_mean(alpha_field, {window}))") for window in (5, 10, 15, 40, 60)]
        fake_agent = _FakeAlphaSemanticAgent(
            [
                _alpha_scorecard("candidate_0", 40),
                _alpha_scorecard("candidate_1", 90),
                _alpha_scorecard("candidate_2", 70),
                _alpha_scorecard("candidate_3", 75),
                _alpha_scorecard("candidate_4", 80),
            ]
        )
        with tempfile.TemporaryDirectory() as td:
            artifact = Path(td) / "semantic.jsonl"
            gate = AlphaSemanticNoveltyGate(
                thresholds=SemanticThresholds(soft_quota_fraction=0.15),
                agent=fake_agent,
                artifact_path=artifact,
            )

            result = gate.filter_candidates(
                candidates,
                similarity_filter=novelty,
                requested_count=10,
                batch_index=3,
            )

            decisions = {row["semantic_gate_decision"] for row in result.decisions}
            self.assertEqual(len(fake_agent.payloads), 1)
            self.assertEqual(len(result.accepted), 3)
            self.assertEqual(len(result.quarantined), 2)
            self.assertIn("ACCEPT_LOW_SIMILARITY", decisions)
            self.assertIn("QUARANTINE_HIGH_SIMILARITY", decisions)
            self.assertIn("ACCEPT_SOFT_QUOTA", decisions)
            self.assertIn("QUARANTINE_SOFT_QUOTA", decisions)
            self.assertEqual(len(artifact.read_text().strip().splitlines()), 5)

    def test_alpha_semantic_gate_failure_only_quarantines_high_deterministic_risk(self) -> None:
        novelty = SimilarityFilter()
        novelty.bootstrap(
            [
                {
                    "expression": "rank(ts_mean(alpha_field, 20))",
                    "fields": ["alpha_field"],
                    "template_id": "rank|ts_mean",
                    "dataset_id": "analyst4",
                    "category": "analyst",
                    "status": "done",
                    "admission_global_max_corr": 0.81,
                    "admission_delta_pool_sharpe": -0.1,
                },
                {
                    "expression": "rank(ts_mean(low_field, 20))",
                    "fields": ["low_field"],
                    "template_id": "rank|ts_mean",
                    "dataset_id": "news12",
                    "category": "news",
                    "status": "done",
                },
            ],
            count_skeletons=False,
            source_name="master",
        )
        gate = AlphaSemanticNoveltyGate(agent=_FakeAlphaSemanticAgent(exc=RuntimeError("critic down")))

        result = gate.filter_candidates(
            [
                _alpha_candidate("rank(ts_mean(alpha_field, 5))"),
                _alpha_candidate(
                    "rank(ts_mean(low_field, 5))",
                    fields=("low_field",),
                    dataset_id="news12",
                    category="news",
                ),
            ],
            similarity_filter=novelty,
            requested_count=5,
            batch_index=4,
        )

        decisions = {row["expression"]: row["semantic_gate_decision"] for row in result.decisions}
        self.assertEqual(len(result.accepted), 1)
        self.assertEqual(len(result.quarantined), 1)
        self.assertEqual(decisions["rank(ts_mean(alpha_field, 5))"], "QUARANTINE_SEMANTIC_AGENT_FAILED")
        self.assertEqual(decisions["rank(ts_mean(low_field, 5))"], "ACCEPT_SEMANTIC_AGENT_FAILED")


class EconomicSearchTests(unittest.TestCase):
    def test_history_index_builds_summary(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            fields = root / "fields.csv"
            fields.write_text(
                "dataset_id,field_id,field_name,type,coverage,category,subcategory,alpha_count,user_count\n"
                "analyst4,eps_rev,EPS Revision,MATRIX,0.9,Analyst,Est,10,1\n"
                "news12,news_sent,News Sentiment,MATRIX,0.8,News,Sent,3,1\n"
            )
            log = root / "sim.jsonl"
            rec = {
                "expression": "group_neutralize(rank(ts_zscore(eps_rev, 60)), subindustry)",
                "fields": ["eps_rev"],
                "dataset_id": "analyst4",
                "category": "Analyst",
                "status": "done",
                "template_id": "group_neutralize|rank|ts_zscore",
                "is_sharpe": 1.3,
                "is_fitness": 1.1,
                "checks": [{"name": "LOW_SHARPE", "result": "PASS"}],
            }
            log.write_text(json.dumps(rec) + "\n")
            indexer = HistoryIndexer(
                index_path=root / "idx.sqlite",
                summary_path=root / "summary.json",
                field_csv=fields,
                log_paths=[log],
            )
            summary = indexer.build()
            self.assertEqual(summary["total_records"], 1)
            self.assertEqual(summary["standalone_pass_records"], 1)
            index = HistoryIndex(index_path=root / "idx.sqlite", summary_path=root / "summary.json")
            self.assertTrue(index.expression_exists(rec["expression"]))
            self.assertTrue(index.field_skeleton_exists(["eps_rev"], ["group_neutralize", "rank", "ts_zscore"]))

    def test_store_and_candidate_generation(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            fields = root / "fields.csv"
            fields.write_text(
                "dataset_id,field_id,field_name,type,coverage,category,subcategory,alpha_count,user_count\n"
                "analyst4,eps_rev,EPS Revision,MATRIX,0.9,Analyst,Est,10,1\n"
                "news12,news_sent,News Sentiment,MATRIX,0.8,News,Sent,3,1\n"
                "pv1,returns,Returns,MATRIX,1.0,Price Volume,PV,100,1\n"
            )
            theory = TheorySpec.from_dict(
                {
                    "theory_id": "t1",
                    "title": "Analyst news confirmation",
                    "mechanism": "revision confirmed by news",
                    "data_roles": {"signal": "analyst4 eps revision", "condition": "news sentiment"},
                    "expected_sign": "positive",
                    "horizon_days": [20, 60],
                    "template_families": ["confirm_product", "gated_trade_when"],
                    "field_cluster_targets": ["analyst revision", "news sentiment"],
                    "novelty_claim": "cross dataset",
                    "kill_criteria": ["kill"],
                    "expand_criteria": ["expand"],
                    "max_probe_candidates": 4,
                }
            )
            store = TheoryStore(root / "theories")
            store.save_theory(theory, event_type="created")
            expander = CandidateExpander(field_csv=fields)
            candidates = expander.generate(theory)
            self.assertGreaterEqual(len(candidates), 1)
            path = store.write_candidates(theory.theory_id, "probe", candidates)
            self.assertTrue(path.exists())
            self.assertIn(theory.theory_id, store.candidate_index()[candidates[0].expression]["theory_id"])

    def test_cli_ingest_links_results_by_expression(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            fields = root / "fields.csv"
            fields.write_text(
                "dataset_id,field_id,field_name,type,coverage,category,subcategory,alpha_count,user_count\n"
                "analyst4,eps_rev,EPS Revision,MATRIX,0.9,Analyst,Est,10,1\n"
                "news12,news_sent,News Sentiment,MATRIX,0.8,News,Sent,3,1\n"
            )
            theory = TheorySpec.from_dict(
                {
                    "theory_id": "t_ingest",
                    "title": "Ingest test",
                    "mechanism": "revision confirmed by news",
                    "data_roles": {"signal": "analyst4 eps revision", "condition": "news sentiment"},
                    "expected_sign": "positive",
                    "horizon_days": [20],
                    "template_families": ["confirm_product"],
                    "field_cluster_targets": ["analyst revision", "news sentiment"],
                    "novelty_claim": "cross dataset",
                    "kill_criteria": ["kill"],
                    "expand_criteria": ["expand"],
                    "max_probe_candidates": 1,
                }
            )
            theory_dir = root / "artifacts" / "theories"
            store = TheoryStore(theory_dir)
            store.save_theory(theory)
            candidates = CandidateExpander(field_csv=fields).generate(theory)
            store.write_candidates(theory.theory_id, "probe", candidates)
            result = {
                "expression": candidates[0].expression,
                "status": "done",
                "is_sharpe": 1.4,
                "is_fitness": 1.2,
                "checks": [{"name": "LOW_SHARPE", "result": "PASS"}],
                "admission_delta_pool_sharpe": 0.02,
                "admission_residual_sharpe": 0.2,
            }
            sim = root / "sim.jsonl"
            sim.write_text(json.dumps(result) + "\n")
            rc = cli_main(
                [
                    "--artifacts-dir",
                    str(root / "artifacts"),
                    "--theory-dir",
                    str(theory_dir),
                    "ingest",
                    "--results-jsonl",
                    str(sim),
                ]
            )
            self.assertEqual(rc, 0)
            decision = store.read_decision(theory.theory_id)
            self.assertEqual(decision["decision"], "PROMOTE")

    def test_deterministic_critic_accepts_cross_family_mock(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            fields = root / "fields.csv"
            fields.write_text(
                "dataset_id,field_id,field_name,type,coverage,category,subcategory,alpha_count,user_count\n"
                "analyst4,eps_rev,EPS Revision,MATRIX,0.9,Analyst,Est,10,1\n"
            )
            log = root / "sim.jsonl"
            log.write_text(
                json.dumps(
                    {
                        "expression": "rank(eps_rev)",
                        "fields": ["eps_rev"],
                        "dataset_id": "analyst4",
                        "category": "Analyst",
                        "status": "done",
                    }
                )
                + "\n"
            )
            indexer = HistoryIndexer(
                index_path=root / "idx.sqlite",
                summary_path=root / "summary.json",
                field_csv=fields,
                log_paths=[log],
            )
            indexer.build()
            index = HistoryIndex(index_path=root / "idx.sqlite", summary_path=root / "summary.json")
            theory = mock_theories(n=1)[0]
            decision = DeterministicNoveltyCritic(index).critique(theory)
            self.assertEqual(decision["decision"], "ACCEPT")

    def test_mock_theories_can_supply_ten_unique_fallbacks(self) -> None:
        theories = mock_theories(sprint_focus="delay 0 fallback", n=10)
        self.assertEqual(len(theories), 10)
        self.assertEqual(len({t.theory_id for t in theories}), 10)

    def test_extract_json_prefers_non_empty_theory_array_over_empty_brackets(self) -> None:
        parsed = _extract_json(
            'I will not return [] for this request.\n'
            '[{"theory_id":"t1","title":"T","mechanism":"M","data_roles":{},'
            '"expected_sign":"positive","horizon_days":[20],'
            '"template_families":["confirm_product"],"field_cluster_targets":["x"],'
            '"novelty_claim":"n","kill_criteria":["k"],"expand_criteria":["e"],'
            '"max_probe_candidates":12,"max_iterations":2,"status":"PROPOSED"}]'
        )
        self.assertIsInstance(parsed, list)
        self.assertEqual(parsed[0]["theory_id"], "t1")

    def test_expression_validator_rejects_window_only_duplicates(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            fields = root / "fields.csv"
            fields.write_text(
                "dataset_id,field_id,field_name,type,coverage,category,subcategory,alpha_count,user_count,delay,universe\n"
                "analyst4,eps_rev,EPS Revision,MATRIX,0.9,Analyst,Est,10,1,0,TOP3000\n"
                "news12,news_sent,News Sentiment,MATRIX,0.8,News,Sent,3,1,0,TOP3000\n"
            )
            theory = TheorySpec.from_dict(
                {
                    "theory_id": "validator_window",
                    "title": "Window variants",
                    "mechanism": "revision confirmed by news",
                    "data_roles": {"signal": "analyst4 eps revision", "condition": "news12 sentiment"},
                    "expected_sign": "positive",
                    "horizon_days": [20],
                    "field_cluster_targets": ["analyst4 revision", "news12 sentiment"],
                    "novelty_claim": "test",
                    "kill_criteria": ["kill"],
                    "expand_criteria": ["expand"],
                }
            )
            rows = [
                {
                    "expression": f"group_neutralize(multiply(rank(ts_zscore(eps_rev, {w})), rank(ts_zscore(news_sent, 20))), subindustry)",
                    "field_roles": {"signal": "eps_rev", "condition": "news_sent"},
                    "mechanism_translation": "same structure",
                    "expected_sign": "positive",
                    "horizon_days": 20,
                    "novelty_argument": "test",
                    "risk_notes": [],
                    "generation_source": "mock",
                }
                for w in (10, 20, 40, 60)
            ]
            validation = ExpressionValidator(field_csv=fields, target_delay=0, min_distinct_skeletons=1).validate_batch(
                theory=theory,
                candidates=rows,
                requested_count=4,
            )
            self.assertEqual(len(validation.accepted), 3)
            self.assertTrue(any(r["reason"] == "too_many_window_or_constant_variants" for r in validation.rejected))

    def test_expression_validator_rejects_missing_required_dataset(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            fields = root / "fields.csv"
            fields.write_text(
                "dataset_id,field_id,field_name,type,coverage,category,subcategory,alpha_count,user_count,delay,universe\n"
                "pv13,rel_ret_all,Peer Return,MATRIX,0.9,Relationship,PV,10,1,0,TOP3000\n"
                "fundamental6,return_assets,Return Assets,MATRIX,0.8,Fundamental,Ratio,3,1,0,TOP3000\n"
            )
            theory = TheorySpec.from_dict(
                {
                    "theory_id": "validator_pv13",
                    "title": "PV13 anchored value",
                    "mechanism": "peer-relative dislocation anchored by fundamentals",
                    "data_roles": {"signal": "pv13 relationship peer return", "condition": "fundamental6 return assets"},
                    "expected_sign": "positive",
                    "horizon_days": [20],
                    "field_cluster_targets": ["pv13 relationship", "fundamental6 value"],
                    "novelty_claim": "test",
                    "kill_criteria": ["kill"],
                    "expand_criteria": ["expand"],
                }
            )
            validation = ExpressionValidator(field_csv=fields, target_delay=0).validate_batch(
                theory=theory,
                candidates=[
                    {
                        "expression": "group_neutralize(rank(ts_zscore(return_assets, 20)), subindustry)",
                        "field_roles": {"signal": "return_assets"},
                        "mechanism_translation": "missing pv13",
                        "expected_sign": "positive",
                        "horizon_days": 20,
                        "novelty_argument": "test",
                        "risk_notes": [],
                        "generation_source": "mock",
                    }
                ],
                requested_count=1,
            )
            self.assertEqual(validation.accepted, [])
            self.assertEqual(validation.rejected[0]["reason"], "missing_required_theory_dataset")

    def test_expression_validator_rejects_group_mean(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            fields = root / "fields.csv"
            fields.write_text(
                "dataset_id,field_id,field_name,type,coverage,category,subcategory,alpha_count,user_count,delay,universe\n"
                "fundamental6,return_assets,Return Assets,MATRIX,0.8,Fundamental,Ratio,3,1,0,TOP3000\n"
                "pv1,volume,Volume,MATRIX,1.0,Price Volume,PV,100,1,0,TOP3000\n"
            )
            theory = TheorySpec.from_dict(
                {
                    "theory_id": "validator_group_mean",
                    "title": "Group mean ban",
                    "mechanism": "test",
                    "data_roles": {"signal": "fundamental6 return assets"},
                    "expected_sign": "positive",
                    "horizon_days": [20],
                    "field_cluster_targets": ["fundamental6 return assets"],
                    "novelty_claim": "test",
                    "kill_criteria": ["kill"],
                    "expand_criteria": ["expand"],
                }
            )
            validation = ExpressionValidator(field_csv=fields, target_delay=0).validate_batch(
                theory=theory,
                candidates=[
                    {
                        "expression": "group_mean(rank(return_assets), volume, subindustry)",
                        "field_roles": {"signal": "return_assets", "weight": "volume"},
                        "mechanism_translation": "forbidden weighted group average",
                        "expected_sign": "positive",
                        "horizon_days": 20,
                        "novelty_argument": "test",
                        "risk_notes": [],
                        "generation_source": "mock",
                    }
                ],
                requested_count=1,
            )
            self.assertEqual(validation.accepted, [])
            self.assertEqual(validation.rejected[0]["reason"], "banned_operator")
            self.assertEqual(validation.rejected[0]["details"]["operators"], ["group_mean"])

    def test_grammar_defaults_do_not_sample_group_mean(self) -> None:
        from alpha_grammar.grammar import BrainGrammar, GrammarConfig
        from alpha_grammar.operators import OPERATOR_REGISTRY

        config = GrammarConfig()
        grammar = BrainGrammar(config)
        self.assertIn("group_mean", OPERATOR_REGISTRY)
        self.assertIn("is_nan", OPERATOR_REGISTRY)
        self.assertIn("ts_count_nans", OPERATOR_REGISTRY)
        self.assertFalse(OPERATOR_REGISTRY["group_mean"].default_enabled)
        self.assertFalse(OPERATOR_REGISTRY["is_nan"].default_enabled)
        self.assertFalse(OPERATOR_REGISTRY["ts_count_nans"].default_enabled)
        self.assertEqual(config.allowed_group_weighted_ops, ())
        self.assertNotIn("is_nan", config.allowed_unary_ops)
        self.assertNotIn("ts_count_nans", config.allowed_ts_rolling_ops)
        self.assertNotIn("group_weighted_op", grammar.feasible_alpha_productions(10, n_profiles=2))
        self.assertNotIn("expr_group_weighted_op", grammar.feasible_expr_productions(10, 8, 1, n_profiles=2))

        explicit = GrammarConfig(
            allowed_group_weighted_ops=("group_mean",),
            allowed_unary_ops=("is_nan", "abs"),
            allowed_ts_rolling_ops=("ts_count_nans", "ts_mean"),
        )
        self.assertEqual(explicit.allowed_group_weighted_ops, ())
        self.assertEqual(explicit.allowed_unary_ops, ("abs",))
        self.assertEqual(explicit.allowed_ts_rolling_ops, ("ts_mean",))

    def test_alpha_llm_config_patch_strips_group_mean(self) -> None:
        from alpha_llm.proposer import AlphaProposerAgent

        agent = object.__new__(AlphaProposerAgent)
        agent._target_batch_size = 40
        config = agent._apply_patch(
            {
                "allowed_group_weighted_ops": ["group_mean"],
                "allowed_group_ops": ["group_rank", "group_mean"],
                "allowed_unary_ops": ["is_nan", "abs"],
                "allowed_ts_rolling_ops": ["ts_count_nans", "ts_mean"],
            }
        )
        self.assertEqual(config.allowed_group_weighted_ops, ())
        self.assertEqual(config.allowed_group_ops, ("group_rank",))
        self.assertEqual(config.allowed_unary_ops, ("abs",))
        self.assertEqual(config.allowed_ts_rolling_ops, ("ts_mean",))

    def test_mock_llm_expression_generation_writes_submit_compatible_records(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            fields = root / "fields.csv"
            fields.write_text(
                "dataset_id,field_id,field_name,type,coverage,category,subcategory,alpha_count,user_count,delay,universe\n"
                "analyst4,eps_rev,EPS Revision,MATRIX,0.9,Analyst,Est,10,1,0,TOP3000\n"
                "news12,news_sent,News Sentiment,MATRIX,0.8,News,Sent,3,1,0,TOP3000\n"
                "pv1,returns,Returns,MATRIX,1.0,Price Volume,PV,100,1,0,TOP3000\n"
            )
            theory = TheorySpec.from_dict(
                {
                    "theory_id": "llm_mock",
                    "title": "LLM mock",
                    "mechanism": "revision confirmed by news",
                    "data_roles": {"signal": "analyst4 eps revision", "condition": "news12 sentiment"},
                    "expected_sign": "positive",
                    "horizon_days": [20],
                    "field_cluster_targets": ["analyst4 revision", "news12 sentiment"],
                    "novelty_claim": "test",
                    "kill_criteria": ["kill"],
                    "expand_criteria": ["expand"],
                }
            )
            store = TheoryStore(root / "theories")
            store.save_theory(theory)
            generator = LLMExpressionGenerator(field_csv=fields, history_index=None, target_delay=0)
            records, report = generator.generate(
                theory=theory,
                store=store,
                stage="probe",
                max_candidates=6,
                portfolio_context={},
                history_summary={},
                field_catalog_summary={"target_delay": 0},
                mock=True,
            )
            self.assertGreaterEqual(len(records), 4)
            row = records[0].to_dict()
            self.assertEqual(row["target_delay"], 0)
            self.assertEqual(row["settings_override"]["delay"], 0)
            self.assertEqual(row["template_family"], "llm_generated")
            self.assertIn("mechanism_translation", row["params"])
            self.assertTrue(report["artifact_paths"]["validation"])

    def test_cli_sprint_mock_reaches_candidate_plan_without_llm_or_brain(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            artifacts = root / "artifacts"
            delay_dir = root / "field_catalogs"
            delay_dir.mkdir()
            fields = delay_dir / "fields_delay0_TOP3000.csv"
            fields.write_text(
                "dataset_id,field_id,field_name,type,coverage,category,subcategory,alpha_count,user_count,delay,universe\n"
                "analyst4,eps_rev,EPS Revision,MATRIX,0.9,Analyst,Est,10,1,0,TOP3000\n"
                "news12,news_sent,News Sentiment,MATRIX,0.8,News,Sent,3,1,0,TOP3000\n"
                "pv1,returns,Returns,MATRIX,1.0,Price Volume,PV,100,1,0,TOP3000\n"
            )
            rc = cli_main(
                [
                    "--artifacts-dir",
                    str(artifacts),
                    "--theory-dir",
                    str(artifacts / "theories"),
                    "--index-path",
                    str(root / "idx.sqlite"),
                    "--summary-path",
                    str(root / "summary.json"),
                    "--fields-csv",
                    str(fields),
                    "--delay-fields-dir",
                    str(delay_dir),
                    "--portfolio-context-path",
                    str(root / "portfolio.json"),
                    "sprint",
                    "--delay",
                    "0",
                    "--n",
                    "1",
                    "--max-candidates-per-theory",
                    "4",
                    "--mock",
                    "--skip-index",
                    "--skip-portfolio",
                    "--skip-fields",
                ]
            )
            self.assertEqual(rc, 0)
            plans = list((artifacts / "candidate_plans").glob("economic_plan_delay0_*.jsonl"))
            self.assertEqual(len(plans), 1)
            rows = [json.loads(line) for line in plans[0].read_text().splitlines() if line.strip()]
            self.assertGreaterEqual(len(rows), 1)
            self.assertEqual(rows[0]["settings_override"]["delay"], 0)

    def test_portfolio_context_builds_submitted_theme_exposures(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            fields = root / "fields.csv"
            fields.write_text(
                "dataset_id,field_id,field_name,type,coverage,category,subcategory,alpha_count,user_count\n"
                "analyst4,eps_rev,EPS Revision,MATRIX,0.9,Analyst,Est,10,1\n"
                "news12,news_sent,News Sentiment,MATRIX,0.8,News,Sent,3,1\n"
            )
            context = build_portfolio_context(
                [
                    {
                        "id": "abc123",
                        "status": "SUBMITTED",
                        "dateSubmitted": "2026-04-01T00:00:00-04:00",
                        "regular": {"code": "group_neutralize(rank(ts_zscore(eps_rev, 60)), subindustry)"},
                        "settings": {
                            "delay": 1,
                            "decay": 5,
                            "neutralization": "SUBINDUSTRY",
                            "universe": "TOP3000",
                        },
                        "is": {"sharpe": 1.5, "fitness": 1.2},
                    }
                ],
                field_csv=fields,
            )
            self.assertEqual(context["n_submitted"], 1)
            self.assertEqual(context["submitted_alphas"][0]["fields"], ["eps_rev"])
            self.assertEqual(context["submitted_alphas"][0]["settings"]["decay"], 5)
            self.assertIn({"key": "delay=1", "count": 1}, context["delay_exposures"])
            self.assertIn({"key": "delay=1", "count": 1}, context["setting_exposures"])
            self.assertTrue(context["regime_exposures"])
            compact = compact_portfolio_context(context)
            self.assertEqual(compact["submitted_alphas"][0]["settings"]["universe"], "TOP3000")
            self.assertTrue(context["do_not_repeat"])

    def test_delay_generation_sets_settings_override_and_filters_fields(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            fields = root / "fields.csv"
            fields.write_text(
                "dataset_id,field_id,field_name,type,coverage,category,subcategory,alpha_count,user_count,delay,universe\n"
                "analyst4,d0_sig,D0 Signal,MATRIX,0.9,Analyst,Est,10,1,0,TOP3000\n"
                "news12,d0_cond,D0 News,MATRIX,0.8,News,Sent,3,1,0,TOP3000\n"
                "analyst4,d1_sig,D1 Signal,MATRIX,0.9,Analyst,Est,10,1,1,TOP3000\n"
                "news12,d1_cond,D1 News,MATRIX,0.8,News,Sent,3,1,1,TOP3000\n"
            )
            theory = TheorySpec.from_dict(
                {
                    "theory_id": "delay_test",
                    "title": "Delay test",
                    "mechanism": "signal confirmed by news",
                    "data_roles": {"signal": "d0 signal", "condition": "d0 news"},
                    "expected_sign": "positive",
                    "horizon_days": [20],
                    "template_families": ["confirm_product"],
                    "field_cluster_targets": ["signal", "news"],
                    "novelty_claim": "test",
                    "kill_criteria": ["kill"],
                    "expand_criteria": ["expand"],
                    "max_probe_candidates": 1,
                }
            )
            candidates = CandidateExpander(
                field_csv=fields,
                target_delay=0,
                require_delay_metadata=True,
            ).generate(theory, target_delay=0)
            self.assertEqual(len(candidates), 1)
            row = candidates[0].to_dict()
            self.assertEqual(row["target_delay"], 0)
            self.assertEqual(row["settings_override"]["delay"], 0)
            self.assertTrue(set(row["fields"]) <= {"d0_sig", "d0_cond"})

    def test_history_gates_same_delay_but_allows_cross_delay_retest(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            fields = root / "fields.csv"
            fields.write_text(
                "dataset_id,field_id,field_name,type,coverage,category,subcategory,alpha_count,user_count,delay,universe\n"
                "analyst4,eps_rev,EPS Revision,MATRIX,0.9,Analyst,Est,10,1,0,TOP3000\n"
                "news12,news_sent,News Sentiment,MATRIX,0.8,News,Sent,3,1,0,TOP3000\n"
                "analyst4,eps_rev,EPS Revision,MATRIX,0.9,Analyst,Est,10,1,1,TOP3000\n"
                "news12,news_sent,News Sentiment,MATRIX,0.8,News,Sent,3,1,1,TOP3000\n"
            )
            expr = "group_neutralize(multiply(rank(ts_zscore(eps_rev, 20)), rank(ts_zscore(news_sent, 20))), subindustry)"
            log = root / "sim.jsonl"
            log.write_text(
                json.dumps(
                    {
                        "expression": expr,
                        "fields": ["eps_rev", "news_sent"],
                        "settings": {"delay": 1, "neutralization": "SUBINDUSTRY", "universe": "TOP3000"},
                        "status": "done",
                    }
                )
                + "\n"
            )
            indexer = HistoryIndexer(
                index_path=root / "idx.sqlite",
                summary_path=root / "summary.json",
                field_csv=fields,
                log_paths=[log],
            )
            indexer.build()
            index = HistoryIndex(index_path=root / "idx.sqlite", summary_path=root / "summary.json")
            theory = TheorySpec.from_dict(
                {
                    "theory_id": "cross_delay",
                    "title": "Cross delay",
                    "mechanism": "revision confirmed by news",
                    "data_roles": {"signal": "eps revision", "condition": "news sentiment"},
                    "expected_sign": "positive",
                    "horizon_days": [20],
                    "template_families": ["confirm_product"],
                    "field_cluster_targets": ["eps revision", "news sentiment"],
                    "novelty_claim": "test",
                    "kill_criteria": ["kill"],
                    "expand_criteria": ["expand"],
                    "max_probe_candidates": 1,
                }
            )
            candidate = {
                "expression": expr,
                "field_roles": {"signal": "eps_rev", "condition": "news_sent"},
                "mechanism_translation": "revision confirmed by news",
                "expected_sign": "positive",
                "horizon_days": 20,
                "novelty_argument": "test",
                "risk_notes": [],
                "generation_source": "mock",
            }
            same_delay = ExpressionValidator(
                field_csv=fields,
                history_index=index,
                target_delay=1,
            ).validate_batch(theory=theory, candidates=[candidate], requested_count=1)
            other_delay = ExpressionValidator(
                field_csv=fields,
                history_index=index,
                target_delay=0,
            ).validate_batch(theory=theory, candidates=[candidate], requested_count=1)
            self.assertEqual(same_delay.accepted, [])
            self.assertEqual(same_delay.rejected[0]["reason"], "duplicate_expression_same_delay")
            self.assertEqual(len(other_delay.accepted), 1)
            self.assertTrue(other_delay.accepted[0].cross_delay_retest)

    def test_expression_validator_vetoes_mutated_history_duplicates(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            fields = root / "fields.csv"
            fields.write_text(
                "dataset_id,field_id,field_name,type,coverage,category,subcategory,alpha_count,user_count,delay,universe\n"
                "analyst4,eps_rev,EPS Revision,MATRIX,0.9,Analyst,Est,10,1,1,TOP3000\n"
                "group,industry,Industry Group,MATRIX,1.0,Group,Group,1,1,1,TOP3000\n"
            )
            expr = "group_rank(ts_zscore(eps_rev, 20), industry)"
            log = root / "sim.jsonl"
            log.write_text(
                json.dumps(
                    {
                        "expression": expr,
                        "fields": ["eps_rev"],
                        "settings": {"delay": 1},
                        "status": "done",
                    }
                )
                + "\n"
            )
            HistoryIndexer(
                index_path=root / "idx.sqlite",
                summary_path=root / "summary.json",
                field_csv=fields,
                log_paths=[log],
            ).build()
            index = HistoryIndex(index_path=root / "idx.sqlite", summary_path=root / "summary.json")
            theory = TheorySpec.from_dict(
                {
                    "theory_id": "mutation_veto",
                    "title": "Mutation veto",
                    "mechanism": "revision signal",
                    "data_roles": {"signal": "analyst4 eps revision"},
                    "expected_sign": "positive",
                    "horizon_days": [20],
                    "field_cluster_targets": ["analyst4 revision"],
                    "novelty_claim": "test",
                    "kill_criteria": ["kill"],
                    "expand_criteria": ["expand"],
                }
            )
            validator = ExpressionValidator(field_csv=fields, history_index=index, target_delay=1)

            cases = {
                "exact": (expr, "duplicate_expression_same_delay"),
                "whitespace": ("group_rank ( ts_zscore ( eps_rev , 20 ) , industry )", "duplicate_expression_same_delay"),
                "add_zero": (f"add({expr}, 0)", "trivial_neutral_wrapper_of_prior_expression"),
                "window": ("group_rank(ts_zscore(eps_rev, 21), industry)", "field_skeleton_seen_in_history"),
            }
            for name, (candidate_expr, expected_reason) in cases.items():
                with self.subTest(name=name):
                    validation = validator.validate_batch(
                        theory=theory,
                        candidates=[{"expression": candidate_expr, "field_roles": {"signal": "eps_rev"}}],
                        requested_count=1,
                    )
                    self.assertEqual(validation.accepted, [])
                    self.assertEqual(validation.rejected[0]["reason"], expected_reason)

            sign_flip = validator.validate_batch(
                theory=theory,
                candidates=[{"expression": f"reverse({expr})", "field_roles": {"signal": "eps_rev"}}],
                requested_count=1,
            )
            self.assertEqual(sign_flip.rejected, [])
            self.assertEqual(len(sign_flip.accepted), 1)
            self.assertIn("sign_flip_of_prior_expression", sign_flip.accepted[0].validation_notes)
            self.assertLessEqual(sign_flip.accepted[0].novelty_score, 0.25)

    def test_semantic_neighbor_retrieval_prioritizes_similar_history(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            fields = root / "fields.csv"
            fields.write_text(
                "dataset_id,field_id,field_name,type,coverage,category,subcategory,alpha_count,user_count,delay,universe\n"
                "news12,news_sent,News Sentiment,MATRIX,0.8,News,Sent,3,1,1,TOP3000\n"
                "pv1,returns,Returns,MATRIX,1.0,Price Volume,PV,100,1,1,TOP3000\n"
            )
            log = root / "sim.jsonl"
            log.write_text(
                "\n".join(
                    [
                        json.dumps(
                            {
                                "expression": "rank(ts_delta(news_sent, 5))",
                                "fields": ["news_sent"],
                                "dataset_id": "news12",
                                "category": "News",
                                "settings": {"delay": 1},
                                "admission_delta_pool_sharpe": -0.02,
                                "admission_global_max_corr": 0.71,
                            }
                        ),
                        json.dumps(
                            {
                                "expression": "rank(ts_delta(returns, 5))",
                                "fields": ["returns"],
                                "dataset_id": "pv1",
                                "category": "Price Volume",
                                "settings": {"delay": 1},
                            }
                        ),
                    ]
                )
                + "\n"
            )
            HistoryIndexer(
                index_path=root / "idx.sqlite",
                summary_path=root / "summary.json",
                field_csv=fields,
                log_paths=[log],
            ).build()
            index = HistoryIndex(index_path=root / "idx.sqlite", summary_path=root / "summary.json")
            candidate = _accepted_expr("rank(ts_delta(news_sent, 7))")
            neighbors = retrieve_semantic_neighbors(index, candidate, target_delay=1, limit=2)
            self.assertEqual(neighbors[0]["fields"], ["news_sent"])
            self.assertIn("pool_hurt_neighbor", neighbors[0]["score_components"])
            self.assertIn("high_corr_neighbor", neighbors[0]["score_components"])

    def test_semantic_gate_routes_scorecards_with_soft_quota(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            fields = root / "fields.csv"
            fields.write_text(
                "dataset_id,field_id,field_name,type,coverage,category,subcategory,alpha_count,user_count,delay,universe\n"
                "news12,news_sent,News Sentiment,MATRIX,0.8,News,Sent,3,1,1,TOP3000\n"
            )
            log = root / "sim.jsonl"
            log.write_text(
                json.dumps(
                    {
                        "expression": "rank(ts_delta(news_sent, 5))",
                        "fields": ["news_sent"],
                        "dataset_id": "news12",
                        "category": "News",
                        "settings": {"delay": 1},
                    }
                )
                + "\n"
            )
            HistoryIndexer(
                index_path=root / "idx.sqlite",
                summary_path=root / "summary.json",
                field_csv=fields,
                log_paths=[log],
            ).build()
            index = HistoryIndex(index_path=root / "idx.sqlite", summary_path=root / "summary.json")
            theory = TheorySpec.from_dict(
                {
                    "theory_id": "semantic_gate",
                    "title": "Semantic gate",
                    "mechanism": "news sentiment change",
                    "data_roles": {"signal": "news sentiment"},
                    "expected_sign": "positive",
                    "horizon_days": [20],
                    "field_cluster_targets": ["news12 sentiment"],
                    "novelty_claim": "test",
                    "kill_criteria": ["kill"],
                    "expand_criteria": ["expand"],
                }
            )

            def run(scorecards: list[dict[str, object]], n: int | None = None):
                candidates = [_accepted_expr(f"rank(ts_delta(news_sent, {10 + i}))") for i in range(n or len(scorecards))]
                gate = SemanticNoveltyGate(
                    history_index=index,
                    target_delay=1,
                    agent=_FakeSemanticAgent(scorecards),
                )
                return gate.apply(
                    theory=theory,
                    candidates=candidates,
                    requested_count=len(candidates),
                    portfolio_context={},
                    history_summary={},
                    mock=False,
                )

            low = run([_scorecard("candidate_0", 20)])
            self.assertEqual(len(low.accepted), 1)
            self.assertEqual(low.accepted[0].source["semantic_gate_decision"], "ACCEPT_LOW_SIMILARITY")

            high = run([_scorecard("candidate_0", 90)])
            self.assertEqual(len(high.accepted), 0)
            self.assertEqual(high.quarantined[0]["routing"]["decision"], "QUARANTINE_HIGH_SIMILARITY")

            malformed = run([{"candidate_key": "candidate_0"}])
            self.assertEqual(len(malformed.accepted), 0)
            self.assertEqual(malformed.quarantined[0]["routing"]["decision"], "QUARANTINE_HIGH_SIMILARITY")
            self.assertIn("missing required", malformed.quarantined[0]["scorecard"]["reason"])

            sign = run([_scorecard("candidate_0", 90, tags=["sign_repair"])])
            self.assertEqual(len(sign.accepted), 1)
            self.assertEqual(sign.accepted[0].source["semantic_gate_decision"], "ACCEPT_SIGN_REPAIR")
            self.assertLessEqual(sign.accepted[0].novelty_score, 0.25)

            cross = run([_scorecard("candidate_0", 90, tags=["cross_delay_retest"])])
            self.assertEqual(len(cross.accepted), 1)
            self.assertTrue(cross.accepted[0].cross_delay_retest)

            medium_cards = [_scorecard(f"candidate_{i}", 75) for i in range(5)]
            medium = run(medium_cards, n=5)
            self.assertEqual(len(medium.accepted), 2)
            self.assertEqual(len(medium.quarantined), 3)
            self.assertTrue(all(r.source["semantic_gate_allowed_by_quota"] for r in medium.accepted))

    def test_result_interpreter_keeps_delay_branches_separate(self) -> None:
        theory = TheorySpec.from_dict(
            {
                "theory_id": "branch_test",
                "title": "Branch test",
                "mechanism": "same theory different delays",
                "data_roles": {},
                "expected_sign": "positive",
                "horizon_days": [20],
                "template_families": ["confirm_product"],
                "field_cluster_targets": [],
                "novelty_claim": "test",
                "kill_criteria": ["kill"],
                "expand_criteria": ["expand"],
            }
        )
        results = [
            {
                "status": "done",
                "target_delay": 1,
                "is_sharpe": 0.5,
                "is_fitness": 0.2,
                "checks": [{"name": "LOW_SHARPE", "result": "FAIL"}],
            },
            {
                "status": "done",
                "target_delay": 0,
                "is_sharpe": 1.4,
                "is_fitness": 1.2,
                "checks": [{"name": "LOW_SHARPE", "result": "PASS"}],
                "admission_delta_pool_sharpe": 0.03,
                "admission_residual_sharpe": 0.2,
            },
        ]
        interpreter = DeterministicResultInterpreter()
        d1 = interpreter.interpret(theory, results, target_delay=1)
        d0 = interpreter.interpret(theory, results, target_delay=0)
        self.assertEqual(d1["decision"], "KILL")
        self.assertEqual(d0["decision"], "PROMOTE")

    def test_portfolio_scoring_uses_same_delay_pool_and_global_corr(self) -> None:
        dates = [f"2023-01-{day:02d}" for day in range(1, 31)]
        portfolio = {
            "d0_alpha": {
                "delay": 0,
                "settings": {"delay": 0},
                "pnl": {date: float(i) for i, date in enumerate(dates)},
                "is_sharpe": 1.5,
                "is_fitness": 1.2,
            },
            "d1_alpha": {
                "delay": 1,
                "settings": {"delay": 1},
                "pnl": {date: float(30 - i) for i, date in enumerate(dates)},
                "is_sharpe": 1.4,
                "is_fitness": 1.1,
            },
        }
        row = {
            "alpha_id": "candidate_d0",
            "candidate_id": "c1",
            "theory_id": "t_pool",
            "expression": "rank(x)",
            "status": "done",
            "settings": {"delay": 0},
            "is_sharpe": 1.3,
            "checks": [{"name": "LOW_SHARPE", "result": "PASS"}],
        }
        scored = score_records_for_admission(
            [row],
            portfolio,
            candidate_pnls={"candidate_d0": {date: float(i * 2) for i, date in enumerate(dates)}},
        )
        self.assertEqual(scored[0]["admission_same_delay_pool_size"], 1)
        self.assertEqual(scored[0]["admission_global_pool_size"], 2)
        self.assertTrue(scored[0]["admission_corr_gate_evaluated"])
        self.assertIn("admission_delta_pool_sharpe", scored[0])
        self.assertIn("admission_residual_sharpe", scored[0])
        self.assertIn("admission_decision", scored[0])

    def test_advance_mock_selects_iterate_and_expand_branches(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            artifacts = root / "artifacts"
            delay_dir = root / "field_catalogs"
            delay_dir.mkdir()
            fields = delay_dir / "fields_delay1_TOP3000.csv"
            fields.write_text(
                "dataset_id,field_id,field_name,type,coverage,category,subcategory,alpha_count,user_count,delay,universe\n"
                "analyst4,eps_rev,EPS Revision,MATRIX,0.9,Analyst,Est,10,1,1,TOP3000\n"
                "news12,news_sent,News Sentiment,MATRIX,0.8,News,Sent,3,1,1,TOP3000\n"
                "pv1,returns,Returns,MATRIX,1.0,Price Volume,PV,100,1,1,TOP3000\n"
            )
            idx = root / "idx.sqlite"
            summary = root / "summary.json"
            HistoryIndexer(index_path=idx, summary_path=summary, field_csv=fields, log_paths=[]).build()
            store = TheoryStore(artifacts / "theories")
            for theory_id, branch in (("t_iter", "ITERATE"), ("t_expand", "EXPAND"), ("t_kill", "KILL")):
                theory = TheorySpec.from_dict(
                    {
                        "theory_id": theory_id,
                        "title": theory_id,
                        "mechanism": "revision confirmed by news",
                        "data_roles": {"signal": "analyst4 eps revision", "condition": "news12 sentiment"},
                        "expected_sign": "positive",
                        "horizon_days": [20],
                        "field_cluster_targets": ["analyst4", "news12"],
                        "novelty_claim": "test",
                        "kill_criteria": ["kill"],
                        "expand_criteria": ["expand"],
                        "status": "ACCEPTED",
                    }
                )
                store.save_theory(theory)
                store.write_decision(theory_id, {"decision": branch, "reason": "test"}, target_delay=1)
            (root / "portfolio.json").write_text("{}")
            rc = cli_main(
                [
                    "--artifacts-dir",
                    str(artifacts),
                    "--theory-dir",
                    str(artifacts / "theories"),
                    "--index-path",
                    str(idx),
                    "--summary-path",
                    str(summary),
                    "--fields-csv",
                    str(fields),
                    "--delay-fields-dir",
                    str(delay_dir),
                    "--portfolio-context-path",
                    str(root / "portfolio.json"),
                    "advance",
                    "--delay",
                    "1",
                    "--mock",
                    "--skip-portfolio",
                    "--no-admission-score",
                ]
            )
            self.assertEqual(rc, 0)
            self.assertTrue(list((artifacts / "theories" / "t_iter").glob("candidates_iterate_*_delay1.jsonl")))
            self.assertTrue(list((artifacts / "theories" / "t_expand").glob("candidates_expand_*_delay1.jsonl")))
            self.assertFalse(list((artifacts / "theories" / "t_kill").glob("candidates_*_delay1.jsonl")))

    def test_second_stage_finalizer_promotes_or_kills_terminal_branch(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            fields = root / "fields.csv"
            fields.write_text(
                "dataset_id,field_id,field_name,type,coverage,category,subcategory,alpha_count,user_count\n"
                "analyst4,eps_rev,EPS Revision,MATRIX,0.9,Analyst,Est,10,1\n"
            )
            store = TheoryStore(root / "theories")
            branches = []
            for theory_id, admission in (("t_bad", "REJECT"), ("t_good", "PROMOTE")):
                theory = TheorySpec.from_dict(
                    {
                        "theory_id": theory_id,
                        "title": theory_id,
                        "mechanism": "test",
                        "data_roles": {"signal": "analyst4 eps revision"},
                        "expected_sign": "positive",
                        "horizon_days": [20],
                        "field_cluster_targets": ["analyst4"],
                        "novelty_claim": "test",
                        "kill_criteria": ["kill"],
                        "expand_criteria": ["expand"],
                        "status": "ACCEPTED",
                    }
                )
                store.save_theory(theory)
                store.write_ingested_results(
                    {
                        theory_id: [
                            {
                                "theory_id": theory_id,
                                "expression": "rank(eps_rev)",
                                "experiment_stage": "iterate",
                                "target_delay": 1,
                                "status": "done",
                                "is_sharpe": 1.5,
                                "checks": [{"name": "LOW_SHARPE", "result": "PASS"}],
                                "admission_decision": admission,
                                "admission_delta_pool_sharpe": 0.02 if admission == "PROMOTE" else -0.01,
                            }
                        ]
                    }
                )
                branches.append((theory, "iterate"))
            _finalize_second_stage(store, branches, target_delay=1)
            self.assertEqual(store.read_decision("t_bad", target_delay=1)["decision"], "KILL")
            self.assertEqual(store.read_decision("t_good", target_delay=1)["decision"], "PROMOTE")

    def test_continuous_harvest_and_ledger_are_idempotent(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            store = TheoryStore(root / "theories")
            theory = TheorySpec.from_dict(
                {
                    "theory_id": "t_promote",
                    "title": "Promoted theory",
                    "mechanism": "test",
                    "data_roles": {"signal": "news12 sentiment"},
                    "expected_sign": "positive",
                    "horizon_days": [20],
                    "field_cluster_targets": ["news12"],
                    "novelty_claim": "test",
                    "kill_criteria": ["kill"],
                    "expand_criteria": ["expand"],
                    "status": "ACCEPTED",
                }
            )
            store.save_theory(theory)
            store.write_ingested_results(
                {
                    theory.theory_id: [
                        {
                            "theory_id": theory.theory_id,
                            "candidate_id": "cand1",
                            "expression": "rank(news_sent)",
                            "experiment_stage": "iterate",
                            "target_delay": 0,
                            "alpha_id": "alpha_promoted",
                            "status": "done",
                            "is_sharpe": 1.4,
                            "is_fitness": 1.1,
                            "is_turnover": 0.2,
                            "checks": [{"name": "LOW_SHARPE", "result": "PASS"}],
                            "admission_decision": "PROMOTE",
                            "admission_delta_pool_sharpe": 0.03,
                            "admission_residual_sharpe": 0.4,
                            "admission_global_max_corr": 0.2,
                        }
                    ]
                }
            )
            promotions = harvest_promotions(
                store=store,
                target_delay=0,
                cycle_id="cycle1",
                timestamp="2026-04-16T00:00:00Z",
            )
            self.assertEqual(len(promotions), 1)
            self.assertEqual(promotions[0]["alpha_id"], "alpha_promoted")
            self.assertEqual(promotions[0]["ledger_status"], "admission_promoted")

            ledger = root / "promotions.csv"
            first = append_promotions_to_ledger(ledger, promotions, mirror_xlsx=False)
            second = append_promotions_to_ledger(ledger, promotions, mirror_xlsx=False)
            self.assertEqual(len(first), 1)
            self.assertEqual(second, [])
            lines = [line for line in ledger.read_text().splitlines() if line.strip()]
            self.assertEqual(len(lines), 2)
            self.assertIn("alpha_promoted", lines[1])

    def test_continuous_active_alpha_ids_preserve_originals_and_promotions(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            active = Path(td) / "active_alpha_ids_delay0.txt"
            merged = merge_alpha_ids(["base1", "base2"], ["base2", "promoted1"])
            self.assertEqual(merged, ["base1", "base2", "promoted1"])
            appended = append_active_alpha_ids(active, ["base1", "base2"])
            appended = append_active_alpha_ids(active, ["promoted1", "base2"])
            self.assertEqual(appended, ["base1", "base2", "promoted1"])
            self.assertEqual(read_active_alpha_ids(active), ["base1", "base2", "promoted1"])

    def test_continuous_lock_prevents_two_runners(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            lock = Path(td) / "runner.lock"
            with ContinuousRunLock(lock):
                with self.assertRaises(RuntimeError):
                    with ContinuousRunLock(lock):
                        pass
            with ContinuousRunLock(lock):
                self.assertTrue(lock.exists())
            self.assertFalse(lock.exists())

    def test_run_forever_respects_max_cycles_without_sleeping(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            artifacts = root / "artifacts"
            calls: list[tuple[str, str, bool, bool, bool]] = []

            def fake_sprint(args):
                calls.append(("sprint", args.alpha_ids, args.skip_index, args.skip_fields, args.skip_portfolio))
                return 0

            def fake_advance(args):
                calls.append(("advance", args.alpha_ids, False, False, args.skip_portfolio))
                return 0

            with patch.object(cli_module, "cmd_sprint", fake_sprint), patch.object(cli_module, "cmd_advance", fake_advance):
                rc = cli_module.main(
                    [
                        "--artifacts-dir",
                        str(artifacts),
                        "--theory-dir",
                        str(artifacts / "theories"),
                        "--index-path",
                        str(root / "idx.sqlite"),
                        "--summary-path",
                        str(root / "summary.json"),
                        "--portfolio-context-path",
                        str(root / "portfolio.json"),
                        "run-forever",
                        "--delay",
                        "0",
                        "--n",
                        "2",
                        "--max-cycles",
                        "2",
                        "--sleep-minutes",
                        "0",
                        "--mock",
                        "--simulate",
                        "--no-admission-score",
                        "--alpha-ids",
                        "base1 base2",
                    ]
                )
            self.assertEqual(rc, 0)
            self.assertEqual([call[0] for call in calls], ["sprint", "advance", "sprint", "advance"])
            self.assertTrue(all("base1" in call[1] and "base2" in call[1] for call in calls))
            state = json.loads((artifacts / "continuous" / "run_state_delay0.json").read_text())
            self.assertEqual(state["cycle_number"], 2)
            self.assertEqual(state["last_cycle_status"], "completed")

    def test_run_forever_harvests_mock_promotion_into_ledger_and_active_ids(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            artifacts = root / "artifacts"

            def fake_sprint(args):
                return 0

            def fake_advance(args):
                store = TheoryStore(Path(args.theory_dir))
                theory = TheorySpec.from_dict(
                    {
                        "theory_id": "continuous_promote",
                        "title": "Continuous promote",
                        "mechanism": "test",
                        "data_roles": {"signal": "news12 sentiment"},
                        "expected_sign": "positive",
                        "horizon_days": [20],
                        "field_cluster_targets": ["news12"],
                        "novelty_claim": "test",
                        "kill_criteria": ["kill"],
                        "expand_criteria": ["expand"],
                        "status": "ACCEPTED",
                    }
                )
                store.save_theory(theory)
                store.write_ingested_results(
                    {
                        theory.theory_id: [
                            {
                                "theory_id": theory.theory_id,
                                "candidate_id": "cand_run",
                                "expression": "rank(news_sent)",
                                "experiment_stage": "expand",
                                "target_delay": 0,
                                "alpha_id": "alpha_run_promote",
                                "status": "done",
                                "is_sharpe": 1.6,
                                "is_fitness": 1.2,
                                "checks": [{"name": "LOW_SHARPE", "result": "PASS"}],
                                "admission_decision": "PROMOTE",
                                "admission_delta_pool_sharpe": 0.05,
                                "admission_residual_sharpe": 0.3,
                            }
                        ]
                    }
                )
                return 0

            with patch.object(cli_module, "cmd_sprint", fake_sprint), patch.object(cli_module, "cmd_advance", fake_advance):
                rc = cli_module.main(
                    [
                        "--artifacts-dir",
                        str(artifacts),
                        "--theory-dir",
                        str(artifacts / "theories"),
                        "--index-path",
                        str(root / "idx.sqlite"),
                        "--summary-path",
                        str(root / "summary.json"),
                        "--portfolio-context-path",
                        str(root / "portfolio.json"),
                        "run-forever",
                        "--delay",
                        "0",
                        "--max-cycles",
                        "1",
                        "--sleep-minutes",
                        "0",
                        "--mock",
                        "--simulate",
                        "--no-admission-score",
                        "--alpha-ids",
                        "base_alpha",
                    ]
                )
            self.assertEqual(rc, 0)
            ledger = artifacts / "continuous" / "promotions_delay0.csv"
            self.assertIn("alpha_run_promote", ledger.read_text())
            active = read_active_alpha_ids(artifacts / "continuous" / "active_alpha_ids_delay0.txt")
            self.assertEqual(active, ["base_alpha", "alpha_run_promote"])


if __name__ == "__main__":
    unittest.main()
