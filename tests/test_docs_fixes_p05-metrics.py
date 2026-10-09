"""Pins for the degenerate-path contracts documented in metrics.py.

Added by the p05-metrics documentation packet. No behavior changed in that
packet -- every assertion here passed on the pre-change code (verified
numerically before the docstrings were written) -- so these tests exist to
keep the new docstring claims from silently becoming lies rather than to
cover a fix.
"""

from __future__ import annotations

import pytest

from facial_recognition_benchmark.metrics import (
    adjusted_rand_index,
    pairwise_f1,
    recognition_diagnostics,
    score_clustering,
    score_recognition,
)


class TestPairwiseF1DegeneratePaths:
    def test_no_agreeing_pairs_scores_zero(self):
        """Zero true positives with any disagreement is 0.0, not an error."""
        assert pairwise_f1([1, 2], ["a", "a"]) == 0.0  # false negatives only
        assert pairwise_f1([1, 1, 1, 1], ["a", "b", "c", "d"]) == 0.0  # fp only
        assert pairwise_f1([1, 1, 2, 2], ["a", "b", "a", "b"]) == 0.0  # fp + fn

    def test_partitions_with_no_pairs_to_score_are_one(self):
        """A single item, or all singletons on both sides, is a perfect 1.0."""
        assert pairwise_f1([1], ["a"]) == 1.0
        assert pairwise_f1([1, 2, 3], ["a", "b", "c"]) == 1.0


class TestAdjustedRandIndexDegeneratePaths:
    def test_zero_denominator_returns_one(self):
        """Constant partitions on both sides: 1.0, not 0.0 and not NaN."""
        assert adjusted_rand_index([7, 7, 7, 7], ["a", "a", "a", "a"]) == 1.0
        # The other zero-denominator shape: all singletons on both sides.
        assert adjusted_rand_index([1, 2, 3], ["a", "b", "c"]) == 1.0

    def test_fewer_than_two_items_returns_one(self):
        assert adjusted_rand_index([1], ["a"]) == 1.0

    def test_can_agree_less_than_chance(self):
        """The index is not clamped to [0, 1]: worse-than-chance is negative."""
        assert adjusted_rand_index(
            [1, 2, 1, 2, 1, 2], ["a", "a", "a", "b", "b", "b"]
        ) == pytest.approx(-1 / 9)


class TestScoreRecognitionContract:
    def test_known_identification_is_macro_averaged_per_identity(self):
        """alice 1/2 and bob 1/1 average to 0.75; a pooled 2/3 would lie."""
        outputs = [
            {
                "known": ["alice", "bob"],
                "unknown_before": [None],
                "post_enrollment": ["alice", None],
            },
            {
                "known": ["bob"],
                "unknown_before": ["bob"],
                "post_enrollment": [],
            },
        ]
        expected = [
            {
                "known": ["alice", "alice"],
                "unknown_before": [None],
                "post_enrollment": ["alice", "alice"],
            },
            {
                "known": ["bob"],
                "unknown_before": [None],
                "post_enrollment": [],
            },
        ]
        score = score_recognition(outputs, expected)
        assert score["known_identification"] == 0.75
        assert score["recognition_score"] == 0.625

    def test_malformed_inputs_return_five_zeroes_without_diagnostics(self):
        expected = [{"known": ["a"], "unknown_before": [None], "post_enrollment": ["a"]}]
        malformed = [
            ([], expected),  # no cases
            ([{"known": ["a", "a"], "unknown_before": [None], "post_enrollment": ["a"]}], expected),
            ([{"known": ["a"], "unknown_before": [None], "post_enrollment": ["a", "a"]}], expected),
            ([{"known": ["a"], "unknown_before": [], "post_enrollment": ["a"]}], expected),
            ([{"known": ["a"], "unknown_before": [None], "post_enrollment": []}], expected),
        ]
        none_in_expected = (
            expected,
            [{"known": [None], "unknown_before": [None], "post_enrollment": ["a"]}],
        )
        for outputs, truth in malformed + [none_in_expected]:
            score = score_recognition(outputs, truth)
            assert set(score.values()) == {0.0}
            assert "_diagnostics" not in score

    def test_diagnostics_key_rides_on_the_returned_mapping(self):
        """The plugin pops `_diagnostics`; direct callers must too."""
        case = [{"known": ["a"], "unknown_before": [None], "post_enrollment": [None, None]}]
        score = score_recognition(case, [
            {"known": ["a"], "unknown_before": [None], "post_enrollment": ["a", "a"]}
        ])
        assert isinstance(score["_diagnostics"], list)
        assert all(
            isinstance(value, float)
            for key, value in score.items()
            if key != "_diagnostics"
        )

    def test_known_only_submission_scores_half_not_one(self):
        """Empty unknown/post denominators guard to 0.0, so lifecycle is 0.0."""
        case = [{"known": ["a"], "unknown_before": [], "post_enrollment": []}]
        score = score_recognition(case, case)
        assert score["unknown_rejection_recall"] == 0.0
        assert score["post_enrollment_accuracy"] == 0.0
        assert score["unknown_lifecycle"] == 0.0
        assert score["recognition_score"] == 0.5


class TestScoreClusteringContract:
    def test_wrong_length_case_stays_in_the_denominator(self):
        """A wrong-length case scores 0.0 but is not dropped from the mean."""
        score = score_clustering(
            [[1, 1, 2, 2], [1, 2]], [["a", "a", "b", "b"], ["a", "a"]]
        )
        assert score["clustering_pairwise_f1"] == 0.5
        assert score["adjusted_rand_index"] == 0.5


class TestRecognitionDiagnosticsBranches:
    def test_even_abstain_confuse_split_reports_confusion_only(self):
        """The elif chain means an even split never mentions the abstentions."""
        notes = recognition_diagnostics(
            known_identification=0.5,
            unknown_rejection=0.5,
            post_correct=1,
            post_abstained=1,
            post_confused=1,
            post_total=3,
        )
        assert len(notes) == 1
        assert "someone else's name" in notes[0]
