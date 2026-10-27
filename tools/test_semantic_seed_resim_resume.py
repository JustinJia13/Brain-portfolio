import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from core.brain.submit import _completed_pairs
from alpha_enhancement.workshop import semantic
from alpha_enhancement.workshop.scoring import score_seed_records
from alpha_enhancement.workshop.seed_resim import resimulate_missing_score_seeds
from alpha_enhancement.workshop.simulation import submit_candidate_batch


def _args() -> SimpleNamespace:
    return SimpleNamespace(
        batch_size=3,
        bypass_correlation_check=False,
        competition_id="IQC2026S1",
        credentials="brain.txt",
        pnl_sleep=0.0,
        yearly_stats_attempts=1,
        yearly_stats_sleep=0.0,
    )


def _seed(expr: str, alpha_id: str) -> dict:
    return {
        "status": "done",
        "expression": expr,
        "alpha_id": alpha_id,
        "settings": dict(semantic.DEFAULT_SETTINGS),
        "is_sharpe": 1.8,
        "is_fitness": 1.4,
        "grade": "GOOD",
    }


def _resim_record(expr: str, alpha_id: str, *, yearly_stats: dict | None = None) -> dict:
    return {
        "status": "done",
        "expression": expr,
        "alpha_id": alpha_id,
        "settings": dict(semantic.DEFAULT_SETTINGS),
        "params": {
            "original_alpha_id": f"original-{alpha_id}",
            "original_expression": expr,
            "seed_resim_reason": "missing_brain_score_delta_on_active_account",
        },
        "search_lane": "semantic_seed_resim",
        "is_sharpe": 1.8,
        "is_fitness": 1.4,
        "grade": "GOOD",
        "yearly_stats": yearly_stats or {},
    }


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def _fake_prescore(records: list[dict], session, *, competition_id: str, pnl_sleep: float) -> list[dict]:
    out = []
    for record in records:
        scored = dict(record)
        scored["admission_brain_score_delta"] = 2.0
        out.append(scored)
    return out


class SemanticSeedResimResumeTests(unittest.TestCase):
    def test_completed_pairs_only_treats_successful_rows_as_done(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            sim_log = Path(tmp) / "sim.jsonl"
            _write_jsonl(
                sim_log,
                [
                    {"expression": "rank(close)", "status": "done", "settings": dict(semantic.DEFAULT_SETTINGS)},
                    {"expression": "rank(open)", "status": "submit_failed", "settings": dict(semantic.DEFAULT_SETTINGS)},
                    {"expression": "rank(volume)", "settings": dict(semantic.DEFAULT_SETTINGS)},
                ],
            )

            done = _completed_pairs(str(sim_log), dict(semantic.DEFAULT_SETTINGS))

        regime = semantic.record_regime_safe({"settings": dict(semantic.DEFAULT_SETTINGS)})
        self.assertIn(("rank(close)", regime), done)
        self.assertIn(("rank(volume)", regime), done)
        self.assertNotIn(("rank(open)", regime), done)

    def test_reuses_cached_seed_resim_without_resubmitting(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            seed_resim_log = Path(tmp) / "semantic_seed_resim_sim.jsonl"
            _write_jsonl(
                seed_resim_log,
                [_resim_record("rank(close)", "cached-alpha", yearly_stats={"2023": {"fitness": 1.1}})],
            )

            with patch.object(semantic, "submit_candidate_batch") as submit_batch:
                result = resimulate_missing_score_seeds(
                    [_seed("rank(close)", "source-alpha")],
                    _args(),
                    object(),
                    settings=dict(semantic.DEFAULT_SETTINGS),
                    seed_resim_log=seed_resim_log,
                    prescore_fn=_fake_prescore,
                    route_fn=lambda *args, **kwargs: "enhance",
                    submit_fn=submit_batch,
                )

            submit_batch.assert_not_called()
            self.assertEqual(result[0]["alpha_id"], "cached-alpha")
            self.assertEqual(result[0]["admission_brain_score_delta"], 2.0)
            self.assertEqual(result[0]["_resimulated_from_alpha_id"], "source-alpha")

    def test_submits_only_seed_resims_missing_from_cache(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            seed_resim_log = Path(tmp) / "semantic_seed_resim_sim.jsonl"
            _write_jsonl(
                seed_resim_log,
                [_resim_record("rank(close)", "cached-alpha", yearly_stats={"2023": {"fitness": 1.1}})],
            )

            def submit_side_effect(candidates, *args, **kwargs):
                self.assertEqual([c["expression"] for c in candidates], ["rank(open)"])
                return [_resim_record("rank(open)", "new-alpha", yearly_stats={"2023": {"fitness": 1.2}})]

            result = resimulate_missing_score_seeds(
                [_seed("rank(close)", "source-a"), _seed("rank(open)", "source-b")],
                _args(),
                object(),
                settings=dict(semantic.DEFAULT_SETTINGS),
                seed_resim_log=seed_resim_log,
                prescore_fn=_fake_prescore,
                route_fn=lambda *args, **kwargs: "enhance",
                submit_fn=submit_side_effect,
            )

            by_expr = {record["expression"]: record for record in result}
            self.assertEqual(by_expr["rank(close)"]["alpha_id"], "cached-alpha")
            self.assertEqual(by_expr["rank(open)"]["alpha_id"], "new-alpha")

    def test_score_seed_records_prescores_only_uncached_sources_before_resim(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            seed_resim_log = Path(tmp) / "semantic_seed_resim_sim.jsonl"
            _write_jsonl(
                seed_resim_log,
                [_resim_record("rank(close)", "cached-alpha", yearly_stats={"2023": {"fitness": 1.1}})],
            )
            prescore_calls: list[list[str]] = []

            def prescore_side_effect(records, session, *, competition_id: str, pnl_sleep: float):
                prescore_calls.append([record["expression"] for record in records])
                if len(prescore_calls) == 1:
                    return [dict(record) for record in records]
                return _fake_prescore(records, session, competition_id=competition_id, pnl_sleep=pnl_sleep)

            def submit_side_effect(candidates, *args, **kwargs):
                self.assertEqual([c["expression"] for c in candidates], ["rank(open)"])
                return [_resim_record("rank(open)", "new-alpha", yearly_stats={"2023": {"fitness": 1.2}})]

            result = score_seed_records(
                [_seed("rank(close)", "source-a"), _seed("rank(open)", "source-b")],
                _args(),
                object(),
                settings=dict(semantic.DEFAULT_SETTINGS),
                seed_resim_log=seed_resim_log,
                prescore_fn=prescore_side_effect,
                reroute_fn=lambda records, args: records,
                route_fn=lambda *args, **kwargs: "enhance",
                submit_fn=submit_side_effect,
            )

            self.assertEqual(prescore_calls[0], ["rank(open)"])
            self.assertEqual(set(prescore_calls[1]), {"rank(close)", "rank(open)"})
            by_expr = {record["expression"]: record for record in result}
            self.assertEqual(by_expr["rank(close)"]["alpha_id"], "cached-alpha")
            self.assertEqual(by_expr["rank(open)"]["alpha_id"], "new-alpha")

    def test_backfills_cached_seed_resim_yearly_stats(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            seed_resim_log = Path(tmp) / "semantic_seed_resim_sim.jsonl"
            _write_jsonl(seed_resim_log, [_resim_record("rank(close)", "cached-alpha")])

            with (
                patch.object(semantic, "submit_candidate_batch") as submit_batch,
                patch(
                    "core.brain.api.fetch_yearly_stats_with_retry",
                    return_value={2023: {"sharpe": 1.3, "fitness": 1.1}},
                ) as fetch_yearly,
            ):
                result = resimulate_missing_score_seeds(
                    [_seed("rank(close)", "source-alpha")],
                    _args(),
                    object(),
                    settings=dict(semantic.DEFAULT_SETTINGS),
                    seed_resim_log=seed_resim_log,
                    prescore_fn=_fake_prescore,
                    route_fn=lambda *args, **kwargs: "enhance",
                    submit_fn=submit_batch,
                )

            submit_batch.assert_not_called()
            fetch_yearly.assert_called_once()
            self.assertEqual(result[0]["yearly_stats"][2023]["fitness"], 1.1)
            self.assertEqual(result[0]["oos_fitness_2023"], 1.1)
            rows = [json.loads(line) for line in seed_resim_log.read_text().splitlines()]
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["yearly_stats"]["2023"]["fitness"], 1.1)

    def test_submit_candidate_batch_dedupes_existing_expression_regime(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            sim_log = Path(tmp) / "workshop_sim.jsonl"
            _write_jsonl(
                sim_log,
                [
                    {
                        "expression": "rank(close)",
                        "status": "done",
                        "alpha_id": "cached-alpha",
                        "settings": dict(semantic.DEFAULT_SETTINGS),
                        "is_sharpe": 1.2,
                        "is_fitness": 1.1,
                        "yearly_stats": {"2023": {"fitness": 1.1}},
                    }
                ],
            )

            def fake_build_record(candidate, settings, url, progress, alpha_id, status, detail, yearly):
                return {
                    "candidate_id": candidate.get("candidate_id"),
                    "expression": candidate.get("expression"),
                    "settings": settings,
                    "status": status,
                    "alpha_id": alpha_id,
                    "is_sharpe": 1.3,
                    "is_fitness": 1.1,
                    "grade": "GOOD",
                }

            candidates = [
                {"candidate_id": "cached", "expression": "rank(close)"},
                {"candidate_id": "new", "expression": "rank(open)"},
            ]
            with (
                patch("core.brain.submit.submit_one", return_value="progress-url") as submit_one,
                patch("core.brain.submit.poll_one", return_value={"alpha": "new-alpha"}),
                patch("core.brain.submit.fetch_detail", return_value={}),
                patch("core.brain.submit.build_record", side_effect=fake_build_record),
                patch("core.brain.api.fetch_yearly_stats_with_retry", return_value={}),
            ):
                records = submit_candidate_batch(
                    candidates,
                    object(),
                    sim_log,
                    dict(semantic.DEFAULT_SETTINGS),
                    "brain.txt",
                    1,
                )

            self.assertEqual([record["expression"] for record in records], ["rank(open)"])
            submit_one.assert_called_once()
            self.assertEqual(submit_one.call_args.args[1], "rank(open)")

    def test_submit_candidate_batch_backfills_cached_yearly_in_place(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            sim_log = Path(tmp) / "workshop_sim.jsonl"
            _write_jsonl(
                sim_log,
                [
                    {
                        "expression": "rank(close)",
                        "status": "done",
                        "alpha_id": "cached-alpha",
                        "settings": dict(semantic.DEFAULT_SETTINGS),
                        "is_sharpe": 1.2,
                        "is_fitness": 1.1,
                        "yearly_stats": {},
                    }
                ],
            )

            with (
                patch("core.brain.submit.submit_one") as submit_one,
                patch(
                    "core.brain.api.fetch_yearly_stats_with_retry",
                    return_value={2023: {"sharpe": 1.3, "fitness": 1.1}},
                ) as fetch_yearly,
            ):
                records = submit_candidate_batch(
                    [{"candidate_id": "cached", "expression": "rank(close)"}],
                    object(),
                    sim_log,
                    dict(semantic.DEFAULT_SETTINGS),
                    "brain.txt",
                    1,
                )

            self.assertEqual(records, [])
            submit_one.assert_not_called()
            fetch_yearly.assert_called_once()
            rows = [json.loads(line) for line in sim_log.read_text().splitlines()]
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["yearly_stats"]["2023"]["fitness"], 1.1)

    def test_submit_candidate_batch_resimulates_incomplete_cached_row_in_place(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            sim_log = Path(tmp) / "workshop_sim.jsonl"
            _write_jsonl(
                sim_log,
                [{"expression": "rank(close)", "status": "done", "settings": dict(semantic.DEFAULT_SETTINGS)}],
            )

            def fake_build_record(candidate, settings, url, progress, alpha_id, status, detail, yearly):
                return {
                    "candidate_id": candidate.get("candidate_id"),
                    "expression": candidate.get("expression"),
                    "settings": settings,
                    "status": status,
                    "alpha_id": alpha_id,
                    "is_sharpe": 1.3,
                    "is_fitness": 1.1,
                    "grade": "GOOD",
                    "yearly_stats": yearly or {},
                }

            with (
                patch("core.brain.submit.submit_one", return_value="progress-url") as submit_one,
                patch("core.brain.submit.poll_one", return_value={"alpha": "new-alpha"}),
                patch("core.brain.submit.fetch_detail", return_value={}),
                patch("core.brain.submit.build_record", side_effect=fake_build_record),
                patch(
                    "core.brain.api.fetch_yearly_stats_with_retry",
                    return_value={2023: {"fitness": 1.1}},
                ),
            ):
                records = submit_candidate_batch(
                    [{"candidate_id": "cached", "expression": "rank(close)"}],
                    object(),
                    sim_log,
                    dict(semantic.DEFAULT_SETTINGS),
                    "brain.txt",
                    1,
                )

            self.assertEqual([record["alpha_id"] for record in records], ["new-alpha"])
            submit_one.assert_called_once()
            rows = [json.loads(line) for line in sim_log.read_text().splitlines()]
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["alpha_id"], "new-alpha")


if __name__ == "__main__":
    unittest.main()
