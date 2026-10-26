"""Offline tests for pool-delta Q-table reward veto."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from core.qtable import QTable, apply_pool_delta_reward_veto, update_qtable_from_records


def _base_record(**overrides):
    record = {
        "status": "done",
        "candidate_id": "c1",
        "expression": "rank(alpha_field)",
        "fields": ["alpha_field"],
        "cluster_ids": ["alpha_cluster"],
        "template_id": "rank",
        "is_sharpe": 1.6,
        "is_fitness": 1.4,
        "is_turnover": 0.2,
        "is_concentrated_weight_ok": True,
        "is_sub_universe_sharpe_ok": True,
    }
    record.update(overrides)
    return record


class QTablePoolDeltaTests(unittest.TestCase):
    def test_negative_delta_zeroes_reward(self) -> None:
        reward = apply_pool_delta_reward_veto(
            _base_record(admission_delta_pool_sharpe=-0.01),
            1.4,
        )
        self.assertEqual(reward, 0.0)

    def test_zero_delta_zeroes_reward(self) -> None:
        reward = apply_pool_delta_reward_veto(
            _base_record(admission_delta_pool_sharpe=0.0),
            1.4,
        )
        self.assertEqual(reward, 0.0)

    def test_positive_delta_preserves_reward(self) -> None:
        reward = apply_pool_delta_reward_veto(
            _base_record(admission_delta_pool_sharpe=0.03),
            1.4,
        )
        self.assertEqual(reward, 1.4)

    def test_positive_brain_score_delta_overrides_negative_proxy(self) -> None:
        reward = apply_pool_delta_reward_veto(
            _base_record(admission_brain_score_delta=227.0, admission_delta_pool_sharpe=-0.01),
            1.4,
        )
        self.assertEqual(reward, 1.4)

    def test_negative_brain_score_delta_zeroes_reward(self) -> None:
        reward = apply_pool_delta_reward_veto(
            _base_record(admission_brain_score_delta=-1.0, admission_delta_pool_sharpe=0.03),
            1.4,
        )
        self.assertEqual(reward, 0.0)

    def test_missing_delta_preserves_reward(self) -> None:
        reward = apply_pool_delta_reward_veto(_base_record(), 1.4)
        self.assertEqual(reward, 1.4)

    def test_non_positive_delta_reject_reason_zeroes_reward(self) -> None:
        reward = apply_pool_delta_reward_veto(
            _base_record(admission_decision="REJECT", admission_reason="NON_POSITIVE_DELTA"),
            1.4,
        )
        self.assertEqual(reward, 0.0)

    def test_incremental_update_uses_vetoed_reward(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            qtable_path = Path(td) / "qtable.json"
            count = update_qtable_from_records(
                [_base_record(admission_delta_pool_sharpe=-0.01)],
                qtable_path=qtable_path,
            )
            self.assertEqual(count, 1)
            qtable = QTable.load(qtable_path)
            entry = qtable.level2[("alpha_field", "rank")]
            self.assertEqual(entry.n_obs, 1)
            self.assertEqual(entry.mean_q, 0.0)


if __name__ == "__main__":
    unittest.main()
