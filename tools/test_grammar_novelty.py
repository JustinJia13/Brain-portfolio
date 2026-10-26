import unittest
from collections import Counter

import numpy as np

from alpha_grammar.grammar import (
    BrainGrammar,
    GrammarConfig,
    _field_node,
    _stratified_field,
)
from alpha_grammar.models import DataProfile, FieldCluster
from alpha_grammar.novelty import TestedFingerprintRegistry
from alpha_grammar.operators import OPERATOR_REGISTRY
from alpha_grammar.planner import GrammarPlanner


def profile(field_id: str, dataset_id: str, field_type: str = "MATRIX") -> DataProfile:
    return DataProfile(
        field_id=field_id,
        category="test",
        dataset_id=dataset_id,
        field_type=field_type,
        coverage=1.0,
        description=field_id,
    )


def cluster(cluster_id: str, dataset_id: str, members: tuple[DataProfile, ...]) -> FieldCluster:
    return FieldCluster(
        cluster_id=cluster_id,
        category="test",
        dataset_id=dataset_id,
        label=cluster_id,
        champion=members[0],
        sleeper=members[1] if len(members) > 1 else None,
        members=members,
    )


class GrammarNoveltyTests(unittest.TestCase):
    def test_same_cluster_high_jaccard_rejected_by_threshold(self) -> None:
        registry = TestedFingerprintRegistry()
        registry.add("cluster_a", "rank(ts_mean(field_a, 20))", ["field_a"])
        similarity = registry.max_similarity(
            "cluster_a",
            "rank(ts_mean(field_b, 60))",
            ["field_b"],
        )
        self.assertGreater(similarity, 0.80)

    def test_same_cluster_lower_jaccard_allowed_by_threshold(self) -> None:
        registry = TestedFingerprintRegistry()
        registry.add("cluster_a", "rank(ts_mean(field_a, 20))", ["field_a"])
        similarity = registry.max_similarity(
            "cluster_a",
            "group_rank(ts_delta(field_b, 5), sector)",
            ["field_b"],
        )
        self.assertLessEqual(similarity, 0.80)

    def test_similar_expression_in_different_cluster_is_allowed(self) -> None:
        registry = TestedFingerprintRegistry()
        registry.add("cluster_a", "rank(ts_mean(field_a, 20))", ["field_a"])
        similarity = registry.max_similarity(
            "cluster_b",
            "rank(ts_mean(field_b, 60))",
            ["field_b"],
        )
        self.assertEqual(similarity, 0.0)

    def test_vector_wrapper_sampling_is_balanced(self) -> None:
        rng = np.random.default_rng(7)
        vector_profile = profile("vec_field", "ds_a", "VECTOR")
        counts = Counter(_field_node(vector_profile, rng).operator for _ in range(4000))
        avg_ratio = counts["vec_avg"] / (counts["vec_avg"] + counts["vec_sum"])
        self.assertGreater(avg_ratio, 0.45)
        self.assertLess(avg_ratio, 0.55)

    def test_dataset_sampling_is_uniform_before_cluster_and_field(self) -> None:
        rng = np.random.default_rng(11)
        clusters = [
            cluster("ds_a:c1", "ds_a", (profile("a1", "ds_a"), profile("a2", "ds_a"))),
            cluster("ds_a:c2", "ds_a", (profile("a3", "ds_a"),)),
            cluster("ds_b:c1", "ds_b", (profile("b1", "ds_b"),)),
            cluster("ds_c:c1", "ds_c", (profile("c1", "ds_c"), profile("c2", "ds_c"), profile("c3", "ds_c"))),
        ]
        counts = Counter(_stratified_field(rng, clusters).dataset_id for _ in range(6000))
        for dataset_id in ("ds_a", "ds_b", "ds_c"):
            ratio = counts[dataset_id] / 6000
            self.assertGreater(ratio, 0.29)
            self.assertLess(ratio, 0.38)

    def test_group_fields_rejected_in_numeric_sampling(self) -> None:
        rng = np.random.default_rng(13)
        group_cluster = cluster("group:c1", "ds_group", (profile("grp_field", "ds_group", "GROUP"),))
        with self.assertRaises(ValueError):
            _stratified_field(rng, [group_cluster])

    def test_group_fields_are_accepted_as_group_arguments(self) -> None:
        rng = np.random.default_rng(17)
        grammar = BrainGrammar(GrammarConfig(group_field_prob=1.0, bucket_group_prob=0.0))
        grammar._group_profiles = (profile("grp_field", "ds_group", "GROUP"),)
        numeric_cluster = cluster("ds_a:c1", "ds_a", (profile("x", "ds_a"),))
        group_arg = grammar._group_expr(rng, [numeric_cluster], "market")
        self.assertEqual(group_arg.render(), "grp_field")

    def test_nan_presence_operators_are_hard_banned(self) -> None:
        self.assertIn("is_nan", OPERATOR_REGISTRY)
        self.assertIn("ts_count_nans", OPERATOR_REGISTRY)
        self.assertFalse(OPERATOR_REGISTRY["is_nan"].default_enabled)
        self.assertFalse(OPERATOR_REGISTRY["ts_count_nans"].default_enabled)

        config = GrammarConfig(
            allowed_unary_ops=("is_nan", "abs"),
            allowed_ts_rolling_ops=("ts_count_nans", "ts_mean"),
        )
        self.assertEqual(config.allowed_unary_ops, ("abs",))
        self.assertEqual(config.allowed_ts_rolling_ops, ("ts_mean",))

    def test_generated_candidates_do_not_use_nan_presence_operators(self) -> None:
        clusters = [
            cluster("ds_a:c1", "ds_a", (profile("a1", "ds_a"), profile("a2", "ds_a"))),
            cluster("ds_b:c1", "ds_b", (profile("b1", "ds_b"), profile("b2", "ds_b"))),
        ]
        config = GrammarConfig(
            n_samples=120,
            budget=10,
            min_coverage=0.0,
            allowed_unary_ops=("is_nan", "abs", "sqrt"),
            allowed_ts_rolling_ops=("ts_count_nans", "ts_mean", "ts_rank"),
        )
        candidates = GrammarPlanner(config).build_plan(clusters, seed=23)
        self.assertGreater(len(candidates), 0)
        for candidate in candidates:
            self.assertNotIn("is_nan(", candidate.expression)
            self.assertNotIn("ts_count_nans(", candidate.expression)
            self.assertNotIn("is_nan", candidate.template_id.split("|"))
            self.assertNotIn("ts_count_nans", candidate.template_id.split("|"))

    def test_planner_uses_cli_budget_without_batch_schedule(self) -> None:
        clusters = [
            cluster("ds_a:c1", "ds_a", (profile("a1", "ds_a"), profile("a2", "ds_a"))),
            cluster("ds_b:c1", "ds_b", (profile("b1", "ds_b"), profile("b2", "ds_b"))),
        ]
        candidates = GrammarPlanner(GrammarConfig(n_samples=30, budget=6)).build_plan(clusters, seed=29)
        self.assertGreater(len(candidates), 0)
        self.assertTrue(all("|k6|" in candidate.family_id for candidate in candidates))


if __name__ == "__main__":
    unittest.main()
