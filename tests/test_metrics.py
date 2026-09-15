from __future__ import annotations

import pytest

from facial_recognition_benchmark.metrics import (
    adjusted_rand_index,
    pairwise_f1,
    score_clustering,
    score_recognition,
)

EXPECTED_RECOGNITION = [
    {
        "known": ["a", "a", "b"],
        "unknown_before": [None, None],
        "post_enrollment": ["new", "new"],
    }
]


def test_perfect_recognition_scores_every_behavior():
    score = score_recognition(EXPECTED_RECOGNITION, EXPECTED_RECOGNITION)
    assert score["known_identification"] == 1.0
    assert score["unknown_lifecycle"] == 1.0
    assert score["recognition_score"] == 1.0


def test_always_unknown_does_not_get_misleading_recognition_score():
    output = [{"known": [None] * 3, "unknown_before": [None] * 2, "post_enrollment": [None] * 2}]
    score = score_recognition(output, EXPECTED_RECOGNITION)
    assert score["unknown_rejection_recall"] == 1.0
    assert score["post_enrollment_accuracy"] == 0.0
    assert score["unknown_lifecycle"] == 0.5
    assert score["recognition_score"] == 0.25


def test_missing_or_malformed_recognition_is_zero_not_renormalized():
    score = score_recognition([{"known": []}], EXPECTED_RECOGNITION)
    assert set(score.values()) == {0.0}


def test_cluster_label_names_do_not_matter():
    expected = ["first", "first", "second", "second"]
    actual = [9, 9, 2, 2]
    assert pairwise_f1(actual, expected) == 1.0
    assert adjusted_rand_index(actual, expected) == 1.0


@pytest.mark.parametrize("actual", [[0, 1, 2, 3], [0, 0, 0, 0]])
def test_degenerate_clusterings_are_not_perfect(actual):
    score = score_clustering([actual], [[0, 0, 1, 1]])
    assert score["clustering_pairwise_f1"] < 1.0


def test_missing_clustering_case_scores_zero():
    assert score_clustering([], [[0, 0]])["clustering_pairwise_f1"] == 0.0


class TestRecognitionDiagnostics:
    """A zero has to say which knob produced it.

    The case that motivated this is real. A student submission scored known
    identification 0.83, unknown rejection 1.00, post-enrollment 0.00 -- which
    reads as a broken recognizer and is not one. Its pipeline ranked the newly
    enrolled person first at cosine distance 0.63, and its own 0.2846 cutoff
    then rejected them. Same-person distances for people enrolled from several
    photos sat at 0.05-0.13, so that cutoff is right for those and too strict
    for a profile built from one image.

    None of that is visible unless abstaining and misnaming are counted apart,
    which is what these tests hold in place.
    """

    def _case(self, post_answers, unknown_answers=(None, None)):
        actual = [{
            "known": ["a", "b"],
            "unknown_before": list(unknown_answers),
            "post_enrollment": list(post_answers),
        }]
        expected = [{
            "known": ["a", "b"],
            "unknown_before": [None, None],
            "post_enrollment": ["new"] * len(post_answers),
        }]
        return score_recognition(actual, expected)

    def test_total_abstention_points_at_enrollment_and_the_cutoff(self):
        """Both are worth looking at: returning no name does not say which.

        A submission returns None for whatever reason it has, and nothing
        here sees a distance, a threshold or whether the profile was stored.
        """

        scores = self._case([None, None])
        assert scores["post_enrollment_accuracy"] == 0.0
        notes = " ".join(scores["_diagnostics"]).lower()
        assert "stored a profile" in notes
        assert "cutoff" in notes
        assert "one photo" in notes

    def test_the_two_ways_of_failing_are_sent_to_different_places(self):
        """Why they are counted apart. Being named someone else and being
        named nobody are different things to go and look at, and neither
        outcome on its own says which part of their code produced it."""

        confused = " ".join(self._case(["someone_else", "someone_else"])["_diagnostics"])
        abstained = " ".join(self._case([None, None])["_diagnostics"])

        assert "someone else's name" in confused
        assert "which profile they matched" in confused
        assert confused != abstained
        # A wrong name does not establish that a threshold admitted it.
        assert "not the threshold" not in confused

    def test_a_working_lifecycle_produces_no_complaint(self):
        scores = self._case(["new", "new"])
        assert scores["post_enrollment_accuracy"] == 1.0
        assert scores["_diagnostics"] == []

    def test_a_cutoff_that_never_fires_is_called_out(self):
        """Everyone recognized and no stranger rejected: the other failure."""

        actual = [{
            "known": ["a", "b"],
            "unknown_before": ["a", "b"],
            "post_enrollment": ["new"],
        }]
        expected = [{
            "known": ["a", "b"],
            "unknown_before": [None, None],
            "post_enrollment": ["new"],
        }]
        notes = " ".join(score_recognition(actual, expected)["_diagnostics"])
        assert "never fires" in notes

    def test_diagnostics_do_not_leak_into_the_metric_mapping(self):
        """The plugin must strip the key; the runner floats every value."""

        from types import SimpleNamespace

        from facial_recognition_benchmark.plugins import RecognitionBenchmark

        # `score` derives its own expected labels from the scenario, so the
        # case has to be scenario-shaped rather than a bare mapping.
        scenario = SimpleNamespace(
            known=[SimpleNamespace(person_id="a", enrollment=[], queries=[object()])],
            unknown_queries=[object()],
            unknown_person_id="new",
            unknown_enrollment=[],
            post_enrollment_queries=[object()],
        )
        benchmark = RecognitionBenchmark()
        actual = [{"known": ["a"], "unknown_before": [None], "post_enrollment": [None]}]
        scores = benchmark.score(actual, [scenario])
        assert "_diagnostics" not in scores
        for value in scores.values():
            assert isinstance(float(value), float)
        assert benchmark.last_diagnostics


class TestAcceptanceRefusesOnlyDegenerateAnswers:
    """What the discovery acceptance test is allowed to refuse.

    Its job is to kill a wrong binding, not to grade. A team whose threshold
    splits one person into two clusters has a working pipeline and a tuning
    problem, and the benchmark exists to put a number on exactly that. It
    used to require the exact partition, which refused those teams and told
    them, in a wiring message, that nothing took their function's output.

    Measured on a 2026 repository: 4, 5, 6, 4 and 5 groups across the five
    thresholds the search tries, against a fixture with 3 people. Every one
    of those is a real answer to the question.
    """

    #: The rule, restated. `roles.py` imports cogbench, which this suite does
    #: not have on its path, so the shape is asserted here and
    #: `python/cogbench/tests` covers the wiring. Kept deliberately literal:
    #: if the rule in roles.py changes, this file should have to change too.
    def _verdict(self, labels, count=12):
        groups = {}
        for index, label in enumerate(labels):
            groups.setdefault(label, []).append(index)
        if len(groups) <= 1:
            return "refused"
        if len(groups) >= count:
            return "refused"
        return "accepted"

    def test_one_cluster_for_everyone_is_refused(self):
        assert self._verdict(["x"] * 12) == "refused"

    def test_one_cluster_each_is_refused(self):
        assert self._verdict(list(range(12))) == "refused"

    def test_a_nearly_right_grouping_is_accepted(self):
        """The case this exists for: one person split, the rest correct."""

        labels = ["a", "a", "a", "a2", "b", "b", "b", "b", "c", "c", "c", "c"]
        assert self._verdict(labels) == "accepted"

    def test_the_exact_answer_is_accepted(self):
        assert self._verdict(["a"] * 4 + ["b"] * 4 + ["c"] * 4) == "accepted"
