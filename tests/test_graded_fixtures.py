"""Submissions that half work, and the scores they must land between.

Every reference fixture in this suite scores exactly 1.0, and the degenerate
ones score 0.0. Nothing established that the benchmark can tell two *working*
submissions apart, which is the property that decides whether it grades or
merely detects.

These fixtures are the failure modes the course actually produces. Each one
must land strictly inside (0, 1): a fixture pinned at either end is not
graded, and the assertions say so rather than checking a threshold that a
future change could drift past unnoticed.
"""

from __future__ import annotations

import pytest

from facial_recognition_benchmark.metrics import score_clustering, score_recognition


def recognition_case(known, unknown_before, post_enrollment, expected_post=None):
    """One scenario in the shape `score_recognition` reads.

    `expected` is what a correct submission would have answered, so a fixture
    is defined by how its `actual` differs from it.
    """

    actual = [{
        "known": list(known),
        "unknown_before": list(unknown_before),
        "post_enrollment": list(post_enrollment),
    }]
    expected = [{
        "known": ["ana", "ben", "cal"],
        "unknown_before": [None, None],
        "post_enrollment": list(expected_post or ["dee"] * len(post_enrollment)),
    }]
    return score_recognition(actual, expected)


class TestCutoffTooStrict:
    """The measured failure: known faces recognized, strangers correctly
    rejected, and the newly enrolled person rejected along with them.

    Taken from a real submission at known 0.83 / unknown 1.00 / post 0.00.
    In that submission a profile built from one photo sat further from a new
    photo than a profile built from several did, so one cutoff could not
    serve both. The graded scenarios enroll from three images, so that
    explanation belongs to that case rather than to every run with this
    shape.
    """

    @pytest.fixture
    def scores(self):
        return recognition_case(
            known=["ana", "ben", "cal"],
            unknown_before=[None, None],
            post_enrollment=[None, None],
        )

    def test_the_score_lands_strictly_between_the_extremes(self, scores):
        value = scores["recognition_score"]
        assert 0.0 < value < 1.0, "a fixture at either end is not graded"

    def test_the_two_halves_disagree_in_the_expected_direction(self, scores):
        assert scores["known_identification"] == 1.0
        assert scores["unknown_rejection_recall"] == 1.0
        assert scores["post_enrollment_accuracy"] == 0.0

    def test_the_diagnostic_names_what_to_go_and_check(self, scores):
        note = " ".join(scores["_diagnostics"]).lower()
        assert "cutoff" in note
        assert "stored a profile" in note


class TestCutoffTooGenerous:
    """The mirror image: everyone gets a name, including strangers. The newly
    enrolled person is recognized precisely because the cutoff lets anyone
    through, so a benchmark that only reported post-enrollment accuracy would
    rank this above the strict one.
    """

    @pytest.fixture
    def scores(self):
        return recognition_case(
            known=["ana", "ben", "cal"],
            unknown_before=["ana", "ben"],
            post_enrollment=["dee", "dee"],
        )

    def test_the_score_lands_strictly_between_the_extremes(self, scores):
        assert 0.0 < scores["recognition_score"] < 1.0

    def test_it_fails_the_opposite_half_from_the_strict_cutoff(self, scores):
        assert scores["unknown_rejection_recall"] == 0.0
        assert scores["post_enrollment_accuracy"] == 1.0

    def test_the_headline_number_cannot_separate_the_two_cutoff_errors(self):
        """Both land at 0.7500, and that is the composite working as designed
        rather than a defect.

        `unknown_lifecycle` is the mean of rejection and post-enrollment, so a
        submission perfect at one and zero at the other lands halfway
        whichever way round it is. Rejecting every stranger while learning
        nobody, and learning everybody while rejecting nobody, are equally far
        from a working recognizer, and the capstone asks for both halves.

        What must never happen is the two being indistinguishable on the page.
        The components separate them completely, and the diagnostic names
        which one a team has, so this asserts the headline collapses AND that
        the readable numbers do not.
        """

        strict = recognition_case(["ana", "ben", "cal"], [None, None], [None, None])
        generous = recognition_case(["ana", "ben", "cal"], ["ana", "ben"], ["dee", "dee"])

        assert strict["recognition_score"] == generous["recognition_score"]
        assert strict["unknown_lifecycle"] == generous["unknown_lifecycle"]

        # The components, which is where a team actually reads the difference.
        assert strict["unknown_rejection_recall"] != generous["unknown_rejection_recall"]
        assert strict["post_enrollment_accuracy"] != generous["post_enrollment_accuracy"]
        assert " ".join(strict["_diagnostics"]) != " ".join(generous["_diagnostics"])


class TestPartiallyWorkingDescriptors:
    """Two of three known people recognized, everything else correct. The
    ordinary shape of a submission that is most of the way there."""

    def test_the_score_reflects_the_partial_credit(self):
        scores = recognition_case(
            known=["ana", "ben", None],
            unknown_before=[None, None],
            post_enrollment=["dee", "dee"],
        )
        assert 0.0 < scores["known_identification"] < 1.0
        assert 0.0 < scores["recognition_score"] < 1.0

    def test_a_better_submission_scores_higher_than_a_worse_one(self):
        """The property the whole benchmark rests on, asserted directly."""

        worse = recognition_case(["ana", None, None], [None, None], ["dee", "dee"])
        better = recognition_case(["ana", "ben", None], [None, None], ["dee", "dee"])
        best = recognition_case(["ana", "ben", "cal"], [None, None], ["dee", "dee"])
        assert (
            worse["recognition_score"]
            < better["recognition_score"]
            < best["recognition_score"]
        )


class TestClusteringSplits:
    """Over-splitting and under-splitting are different mistakes, and the two
    reported metrics disagree about them on purpose."""

    #: Two people, three photos each.
    TRUTH = [["a", "a", "a", "b", "b", "b"]]

    def test_one_giant_cluster_looks_respectable_on_f1_and_near_zero_on_ari(self):
        """This is the documented reason both metrics are reported, and it is
        the misreading a team would otherwise make: an F1 of 0.5 from a
        clusterer that did nothing at all."""

        scores = score_clustering([[0, 0, 0, 0, 0, 0]], self.TRUTH)
        assert scores["clustering_pairwise_f1"] > 0.4
        assert scores["adjusted_rand_index"] == pytest.approx(0.0, abs=1e-9)

    def test_every_photo_its_own_cluster_fails_both(self):
        scores = score_clustering([[0, 1, 2, 3, 4, 5]], self.TRUTH)
        assert scores["clustering_pairwise_f1"] == pytest.approx(0.0, abs=1e-9)
        assert scores["adjusted_rand_index"] <= 0.0

    def test_one_photo_in_the_wrong_group_lands_between_the_extremes(self):
        scores = score_clustering([[0, 0, 1, 1, 1, 1]], self.TRUTH)
        assert 0.0 < scores["clustering_pairwise_f1"] < 1.0
        assert 0.0 < scores["adjusted_rand_index"] < 1.0

    def test_a_closer_clustering_scores_higher(self):
        one_wrong = score_clustering([[0, 0, 1, 1, 1, 1]], self.TRUTH)
        two_wrong = score_clustering([[0, 1, 1, 1, 1, 0]], self.TRUTH)
        assert two_wrong["clustering_pairwise_f1"] < one_wrong["clustering_pairwise_f1"]

    def test_labels_are_arbitrary(self):
        """[0,0,0,1,1,1] and [7,7,7,2,2,2] are the same answer."""

        assert score_clustering([[0, 0, 0, 1, 1, 1]], self.TRUTH) == score_clustering(
            [[7, 7, 7, 2, 2, 2]], self.TRUTH
        )
