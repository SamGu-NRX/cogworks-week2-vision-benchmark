"""Metric-validity harness: exact references, brute-force enumeration, and degenerate arms.

The existing suite pins that the metrics behave sensibly on a handful of
showcase answers. It cannot detect several denominators regressions because no
fixture distinguishes per-identity macro averaging from per-image pooling, and
the clustering degenerate branches (zero-pair truth, empty cases, wrong-length
answers) are unpinned at the metric level. This file holds the arithmetic to
independently computed values so a change in a denominator or a zero-branch
convention has to be made deliberately.

Three kinds of reference live here, in decreasing order of independence:

1. Hand-computed constants. Every expected value in
   ``TestPairwiseF1ExactReferences`` and ``TestAdjustedRandExactReferences``
   was worked out pair by pair (the arithmetic is written into each
   docstring), then verified against a brute-force enumeration of all 52
   partitions of five elements before being pinned.
2. A deliberately different implementation. ``_reference_f1`` counts pairs by
   intersecting equivalence classes rather than by the double index loop in
   ``metrics.pairwise_f1``, and agrees with it on every partition pair of
   five elements. Where scikit-learn is installed, ``adjusted_rand_index`` is
   also checked against ``sklearn.metrics.adjusted_rand_score``; the check
   skips silently where it is not, because the repo's test extra does not
   carry it.
3. The shipped public manifests, scored through the real plugin entry
   points with images never loaded. The manifests carry the clustering
   truth labels and the recognition lifecycle lengths, so degenerate
   baselines (one giant cluster, all singletons, an all-``None``
   recognizer) can be scored against the real published gold on both tiers
   with no download.

Behavior that looks wrong but is deliberate is pinned with a comment
explaining what would have to change for the pin to move: an empty
``pairwise_f1`` call returning 1.0, a lifecycle with no stranger questions
scoring 0.0 rejection recall, and one-giant-cluster F1 landing at 3/7 on the
test tier while its ARI is exactly 0.
"""

from __future__ import annotations

import math
import random
from collections import Counter
from functools import lru_cache
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np
import pytest

from facial_recognition_benchmark.adapters import AdapterContractError
from facial_recognition_benchmark.datasets import (
    DatasetError,
    assert_disjoint,
    load_manifest,
)
from facial_recognition_benchmark.drivers import (
    ClusteringScenario,
    RecognitionIdentity,
    RecognitionScenario,
    query_phases,
    recognition_expected,
    run_clustering_scenario,
    run_recognition_scenario,
)
from facial_recognition_benchmark.metrics import (
    adjusted_rand_index,
    pairwise_f1,
    score_clustering,
    score_recognition,
)
from facial_recognition_benchmark.plugins import ClusteringBenchmark, RecognitionBenchmark

# ---------------------------------------------------------------------------
# Independent reference implementations
# ---------------------------------------------------------------------------


def _reference_f1(actual: Sequence[Any], expected: Sequence[Any]) -> float:
    """Pairwise F1 by intersecting equivalence classes, not by an index loop.

    ``metrics.pairwise_f1`` walks every index pair. This counts the same
    quantity from contingency-cell sizes: two answers of the same person that
    land in the same predicted cluster. The two agree algebraically
    (2PR/(P+R) == 2TP/(2TP+FP+FN)) but share no code path, so an indexing or
    off-by-one regression in either shows up as a disagreement in
    ``test_all_partition_pairs_match_the_class_intersection_reference``.
    """

    def class_sizes(labels: Sequence[Any]) -> List[int]:
        counts: Dict[Any, int] = {}
        for label in labels:
            counts[label] = counts.get(label, 0) + 1
        return list(counts.values())

    actual_sizes = class_sizes(actual)
    expected_sizes = class_sizes(expected)
    joint = Counter(zip(actual, expected))
    tp = sum(math.comb(size, 2) for size in joint.values())
    fp = sum(math.comb(size, 2) for size in actual_sizes) - tp
    fn = sum(math.comb(size, 2) for size in expected_sizes) - tp
    if tp == 0:
        # Same convention as the metric: with no true pair left, a perfect
        # all-singletons answer is 1.0 and anything with a false pair is 0.0.
        return 1.0 if fp == 0 and fn == 0 else 0.0
    return 2.0 * tp / (2.0 * tp + fp + fn)


def _all_partitions(n: int) -> Iterator[List[List[int]]]:
    """Every set partition of ``range(n)``, as lists of classes."""

    if n == 0:
        yield []
        return
    first, _rest = n - 1, list(range(n - 1))
    for partition in _all_partitions(n - 1):
        yield [[first]] + partition
        for position, group in enumerate(partition):
            yield [group + [first]] + partition[:position] + partition[position + 1 :]


def _partition_labels(partition: List[List[int]]) -> List[int]:
    labels: Dict[int, int] = {}
    for index, group in enumerate(partition):
        for member in group:
            labels[member] = index
    return [labels[i] for i in range(len(labels))]


def _partitions_of_five() -> List[List[int]]:
    return [_partition_labels(partition) for partition in _all_partitions(5)]


# ---------------------------------------------------------------------------
# Hand-computed exact references
# ---------------------------------------------------------------------------


class TestPairwiseF1ExactReferences:
    """Every constant here was counted by hand over the six or ten pairs.

    The arithmetic is written out so a reviewer can re-derive it without
    running anything. Verified against the brute-force enumeration in
    ``TestPairwiseF1BruteForce`` before being pinned.
    """

    def test_partial_overlap_is_two_thirds(self):
        # expected [0,0,1,1], actual [0,0,1,2]: pairs (0,1) same/same -> TP;
        # (2,3) same/diff -> FN; all four mixed pairs differ in both. So
        # TP=1, FP=0, FN=1, P=1, R=1/2, F1 = 2*(1*1/2)/(1+1/2) = 2/3.
        assert pairwise_f1([0, 0, 1, 2], [0, 0, 1, 1]) == pytest.approx(2.0 / 3.0)

    def test_a_partial_overlap_keeping_its_shape_under_renaming(self):
        # Same partition as the 2/3 case with every label renamed: the pair
        # structure -- and therefore the score -- cannot move.
        assert pairwise_f1([7, 7, 2, 9], [0, 0, 1, 1]) == pytest.approx(2.0 / 3.0)

    def test_a_crossed_partition_has_zero_f1_but_negative_ari(self):
        # expected [0,0,1,1], actual [0,1,0,1]: the two same-expected pairs
        # (0,1) and (2,3) are both split, and two mixed pairs agree on
        # "different". TP=0, FP=2, FN=2 -> F1 0. The ARI reference below
        # gives -1/2 for the same answers: worse than random.
        assert pairwise_f1([0, 1, 0, 1], [0, 0, 1, 1]) == 0.0
        assert adjusted_rand_index([0, 1, 0, 1], [0, 0, 1, 1]) == pytest.approx(-0.5)

    def test_half_the_pairs_right_is_one_half(self):
        # expected [0,0,0,1,1], actual [0,0,1,1,1]: expected-same pairs are
        # (0,1),(0,2),(1,2),(3,4). Of those, (0,1) and (3,4) survive
        # (TP=2); (0,2) and (1,2) split (FN=2); (2,3) and (2,4) join
        # different people (FP=2). P=R=1/2, F1=1/2.
        assert pairwise_f1([0, 0, 1, 1, 1], [0, 0, 0, 1, 1]) == pytest.approx(0.5)

    def test_all_singletons_against_paired_truth_is_zero(self):
        # No predicted pair can be right when there are none: TP=0, FN=2.
        assert pairwise_f1([0, 1, 2, 3], [0, 0, 1, 1]) == 0.0

    def test_all_singletons_on_all_singleton_truth_is_perfect(self):
        # TP=0 but so are FP and FN: the all-singletons convention makes an
        # answer that refuses every pairing perfect against a partition that
        # asserts none. Pinned so the convention cannot drift silently.
        assert pairwise_f1([0, 1, 2, 3], [0, 1, 2, 3]) == 1.0

    def test_one_giant_cluster_on_all_singleton_truth_is_zero(self):
        # Every mixed pair joins: TP=0, FP=6, FN=0 -> 0.0 by the same
        # zero-pair branch.
        assert pairwise_f1([0, 0, 0, 0], [0, 1, 2, 3]) == 0.0

    def test_a_fully_renamed_answer_is_perfect(self):
        assert pairwise_f1([1, 1, 0, 0], [0, 0, 1, 1]) == 1.0

    def test_an_empty_direct_call_is_a_recorded_convention(self):
        # metrics.pairwise_f1([], []) returns 1.0: with no images there are no
        # pairs to disagree about. `score_clustering` never lets this surface
        # -- an empty case scores 0.0 and an empty case list scores 0.0 -- so
        # the convention is unreachable through the plugins. If this pin ever
        # moves, check every zero-pair branch again: they share the guard.
        assert pairwise_f1([], []) == 1.0


class TestPairwiseF1BruteForce:
    """Exhaustive agreement over every partition pair of five elements.

    52 partitions give 2,704 partition pairs, enough that a systematically
    wrong pair count cannot agree with the reference by accident.
    """

    def test_all_partition_pairs_match_the_class_intersection_reference(self):
        for actual in _partitions_of_five():
            for expected in _partitions_of_five():
                assert pairwise_f1(actual, expected) == pytest.approx(
                    _reference_f1(actual, expected), abs=1e-12
                )

    def test_f1_is_symmetric_in_its_two_arguments(self):
        # Pairwise F1 only ever swaps P and R, so the argument order cannot
        # matter. A weighted or per-class F1 regression would break this.
        for actual in _partitions_of_five():
            for expected in _partitions_of_five():
                assert pairwise_f1(actual, expected) == pytest.approx(
                    pairwise_f1(expected, actual), abs=1e-12
                )

    def test_relabeling_never_moves_f1(self):
        # Rename actual's labels through an arbitrary injection; the pair
        # structure is untouched.
        for actual in _partitions_of_five():
            relabeled = [label + 100 for label in actual]
            for expected in _partitions_of_five()[:8]:
                assert pairwise_f1(relabeled, expected) == pytest.approx(
                    pairwise_f1(actual, expected), abs=1e-12
                )


class TestAdjustedRandExactReferences:
    """ARI constants from the contingency-table formula, counted by hand.

    ARI = (sum_ij C(n_ij,2) - E) / (max - E) where E is the expected pair
    count under the hypergeometric null and max is the average of both
    partitions' own pair counts.
    """

    def test_identical_partitions_score_one(self):
        labels = [0, 0, 1, 1, 2, 2, 2]
        assert adjusted_rand_index(labels, list(labels)) == 1.0

    def test_two_one_one_against_two_two_is_four_sevenths(self):
        # expected [0,0,1,1], actual [0,0,1,2]: one contingency cell of 2
        # (sum=1), expected pairs 1+1=2, actual pairs 1, total pairs 6.
        # E = 2*1/6 = 1/3, max = 3/2. ARI = (1 - 1/3)/(3/2 - 1/3) = 4/7.
        assert adjusted_rand_index([0, 0, 1, 2], [0, 0, 1, 1]) == pytest.approx(4.0 / 7.0)

    def test_one_giant_on_all_singletons_is_exactly_zero(self):
        # sum_ij = 0 (no cell holds a pair), E = 0 (expected has no pairs),
        # so ARI = 0 regardless of how many pairs the answer invents.
        assert adjusted_rand_index([0, 0, 0, 0], [0, 1, 2, 3]) == 0.0

    def test_both_all_singletons_is_one(self):
        # Denominator is 0 -- neither partition asserts a pair -- and the
        # answers agree, so 1.0. sklearn's convention matches.
        assert adjusted_rand_index([0, 1, 2, 3], [0, 1, 2, 3]) == 1.0

    def test_fewer_than_two_points_is_one(self):
        assert adjusted_rand_index([0], [5]) == 1.0

    def test_all_partition_pairs_match_sklearn_when_available(self):
        adjusted_rand_score = pytest.importorskip(
            "sklearn.metrics", reason="scikit-learn is not installed"
        ).adjusted_rand_score
        for actual in _partitions_of_five():
            for expected in _partitions_of_five():
                assert adjusted_rand_index(actual, expected) == pytest.approx(
                    float(adjusted_rand_score(actual, expected)), abs=1e-10
                )


# ---------------------------------------------------------------------------
# Recognition denominators and abstention semantics
# ---------------------------------------------------------------------------


class TestRecognitionDenominators:
    """What divides what in score_recognition, and what a zero means.

    Known identification averages per person (the docstring's promise), only
    the ``None`` sentinel counts as saying "I do not know", and any malformed
    case zeroes the whole run instead of shrinking a denominator. None of
    that was distinguishable from the alternatives in the existing fixtures.
    """

    @staticmethod
    def _expected() -> List[Dict[str, Any]]:
        # Person "a" answers one query, person "b" answers three: the
        # smallest fixture where macro and micro averaging differ.
        return [
            {
                "known": ["a", "b", "b", "b"],
                "unknown_before": [None, None],
                "post_enrollment": ["new", "new"],
            }
        ]

    def test_known_identification_is_macro_not_micro(self):
        # "a" is right 0 times out of 1, "b" is right 2 times out of 3.
        # Macro average (0 + 2/3) / 2 = 1/3. Per-image pooling would give
        # 2/4 = 1/2, and mixed pooling with duplicates 3/4 -- all different.
        actual = [
            {
                "known": [None, "b", "b", "c"],
                "unknown_before": [None, None],
                "post_enrollment": ["new", "new"],
            }
        ]
        scores = score_recognition(actual, self._expected())
        assert scores["known_identification"] == pytest.approx(1.0 / 3.0)

    def test_abstaining_on_a_known_person_counts_as_wrong(self):
        actual = [
            {
                "known": [None, "b", "b", "b"],
                "unknown_before": [None, None],
                "post_enrollment": ["new", "new"],
            }
        ]
        scores = score_recognition(actual, self._expected())
        assert scores["known_identification"] == pytest.approx(0.5)  # (0 + 1) / 2

    def test_only_none_counts_as_saying_unknown(self):
        # The contract's word for "I do not know" is None. A submission that
        # answers the string "unknown" is naming a person nobody enrolled:
        # zero rejection credit and a misnaming on the knowns. Pinned so the
        # leniency of the discovered path (which maps such strings for teams
        # that were given benchmark-chosen names) is not read into the
        # declared contract. See the lane-asymmetry note in the validity
        # report; changing this is a recognition-v3 consequence.
        actual = [
            {
                "known": ["a", "b", "b", "b"],
                "unknown_before": ["unknown", ""],
                "post_enrollment": ["new", "new"],
            }
        ]
        scores = score_recognition(actual, self._expected())
        assert scores["unknown_rejection_recall"] == 0.0
        assert scores["post_enrollment_accuracy"] == 1.0

    def test_identities_are_pooled_across_cases_before_averaging(self):
        # Identity "a" answers correctly in case one and wrongly in case two:
        # one pooled cell [right, wrong] -> 1/2. Identity "b" is right -> 1.
        # Macro over {a: 1/2, b: 1} = 3/4. Averaging per case first would
        # give (1 + 0) / 2 = 1/2, so this pins pooled-per-identity.
        expected = [
            {"known": ["a", "b"], "unknown_before": [None], "post_enrollment": []},
            {"known": ["a"], "unknown_before": [], "post_enrollment": []},
        ]
        actual = [
            {"known": ["a", "b"], "unknown_before": [None], "post_enrollment": []},
            {"known": ["zzz"], "unknown_before": [], "post_enrollment": []},
        ]
        scores = score_recognition(actual, expected)
        assert scores["known_identification"] == pytest.approx(0.75)

    def test_a_wrong_length_unknown_section_zeroes_the_whole_case_list(self):
        # Two answers for two expected unknowns is fine; three is malformed.
        # The metric refuses to score a partial lifecycle: every section
        # zeroes, including the sections that were fine. Contrast the
        # clustering behavior, which zeroes only the malformed case.
        actual = [
            {
                "known": ["a", "b", "b", "b"],
                "unknown_before": [None, None, None],
                "post_enrollment": ["new", "new"],
            }
        ]
        scores = score_recognition(actual, self._expected())
        assert set(scores) - {"_diagnostics"} == {
            "recognition_score",
            "known_identification",
            "unknown_rejection_recall",
            "post_enrollment_accuracy",
            "unknown_lifecycle",
        }
        assert set(scores.values()) == {0.0}

    def test_a_wrong_length_post_section_zeroes_the_whole_case_list(self):
        actual = [
            {
                "known": ["a", "b", "b", "b"],
                "unknown_before": [None, None],
                "post_enrollment": ["new"],
            }
        ]
        assert set(score_recognition(actual, self._expected()).values()) == {0.0}

    def test_a_lifecycle_with_no_stranger_questions_scores_zero_rejection(self):
        # With zero unknown queries there is nothing to reject, and the
        # convention is to score the recall 0.0 rather than to skip the
        # half. Pinned deliberately: it drags a perfect stranger-free
        # lifecycle to (1 + 0) / 2 = 0.5 on the unknown half. If this moves,
        # the hosted and local lanes must move together.
        expected = [{"known": ["a"], "unknown_before": [], "post_enrollment": []}]
        actual = [{"known": ["a"], "unknown_before": [], "post_enrollment": []}]
        scores = score_recognition(actual, expected)
        assert scores["unknown_rejection_recall"] == 0.0
        assert scores["unknown_lifecycle"] == 0.0
        assert scores["recognition_score"] == pytest.approx(0.5)

    def test_an_empty_case_list_scores_zero(self):
        scores = score_recognition([], self._expected())
        assert set(scores.values()) == {0.0}

    def test_diagnosers_name_abstention_and_misnaming_apart(self):
        # All-None on the post section is a cutoff story; a confident wrong
        # name is not. The strings are diagnostics, not scored values, but a
        # regression that swaps their causes sends teams to the wrong knob.
        abstaining = [
            {"known": ["a", "b"], "unknown_before": [None, None], "post_enrollment": [None, None]}
        ]
        expected = [
            {"known": ["a", "b"], "unknown_before": [None, None], "post_enrollment": ["new", "new"]}
        ]
        notes = " ".join(score_recognition(abstaining, expected)["_diagnostics"]).lower()
        assert "cutoff" in notes

        misnaming = [
            {"known": ["a", "b"], "unknown_before": [None, None], "post_enrollment": ["zed", "zed"]}
        ]
        notes = " ".join(score_recognition(misnaming, expected)["_diagnostics"])
        assert "someone else's name" in notes
        assert "not the threshold" in notes


# ---------------------------------------------------------------------------
# Clustering denominators
# ---------------------------------------------------------------------------


class TestClusteringDenominators:
    """score_clustering averages per case and isolates malformed cases.

    The recognition side zeroes the whole run on a malformed case; the
    clustering side zeroes only the malformed case and keeps the rest. Both
    are pinned here because the asymmetry is load-bearing: clustering cases
    are independent draws, while a recognition lifecycle's sections share
    one submission state.
    """

    def test_a_wrong_length_case_zeroes_only_itself(self):
        actual = [[0, 0, 1, 1], [9, 9, 9]]  # second case: 3 labels for 2 images
        expected = [[0, 0, 1, 1], [0, 0, 1, 1]]
        assert score_clustering(actual, expected)["clustering_pairwise_f1"] == pytest.approx(0.5)

    def test_cases_are_averaged_equally(self):
        actual = [[0, 0, 1, 1], [5, 5, 6, 6], [0, 1, 0, 1]]
        expected = [[0, 0, 1, 1], [0, 0, 1, 1], [0, 0, 1, 1]]
        # Perfect, perfect, zero -> 2/3 regardless of case sizes.
        assert score_clustering(actual, expected)["clustering_pairwise_f1"] == pytest.approx(
            2.0 / 3.0
        )

    def test_an_empty_truth_case_scores_zero(self):
        assert score_clustering([[0, 0]], [[]])["clustering_pairwise_f1"] == 0.0

    def test_renaming_across_cases_scores_one(self):
        actual = [[7, 7, 3, 3], ["z", "z", "z", "a"]]
        expected = [[0, 0, 1, 1], [9, 9, 9, 4]]
        score = score_clustering(actual, expected)
        assert score["clustering_pairwise_f1"] == 1.0
        assert score["adjusted_rand_index"] == 1.0


# ---------------------------------------------------------------------------
# Driver slot alignment and lifecycle witnesses
# ---------------------------------------------------------------------------


def _color_image(red: int, green: int, blue: int) -> np.ndarray:
    value = np.empty((3, 3, 3), dtype=np.uint8)
    value[:, :] = (red, green, blue)
    return value


class _ContentLabelAdapter:
    """Name people by their pixel color, then answer by pixel color.

    A working recognizer that cannot be fooled by batch order: every answer
    is keyed to the photograph's content, so the slot machinery is the only
    thing under test.
    """

    def __init__(self, model: Any) -> None:
        self._model = model
        self._colors: Dict[Tuple[int, int, int], str] = {}

    def enroll(self, person_id: str, images: Sequence[np.ndarray]) -> None:
        for image in images:
            self._colors[tuple(int(channel) for channel in image[0, 0])] = person_id

    def recognize(self, images: Sequence[np.ndarray]) -> List[Optional[str]]:
        return [
            self._colors.get(tuple(int(channel) for channel in image[0, 0])) for image in images
        ]


class _ForgetterAdapter(_ContentLabelAdapter):
    """Answer correctly until the stranger is enrolled, then forget everyone.

    This is the failure the two-batch lifecycle exists to catch: correct
    before the enrolment call, ``None`` after it. With all known queries in
    one batch it would score a full marks known identification; the
    per-person split makes it 0.5.
    """

    def __init__(self, model: Any) -> None:
        super().__init__(model)
        self._calls = 0

    def recognize(self, images: Sequence[np.ndarray]) -> List[Optional[str]]:
        self._calls += 1
        if self._calls > 1:
            return [None for _ in images]
        return super().recognize(images)


class _PositionalCheaterAdapter(_ContentLabelAdapter):
    """Ignore the photographs; answer the enrolled names in rotation.

    The docstring in ``query_phases`` describes this submission: it keeps the
    names it was given and answers by position. Before the phases were
    shuffled it scored full marks without opening a photograph. It must never
    reach 1.0 again, and because the local deal is seeded from the scenario's
    own names, its exact score is deterministic and pinnable.
    """

    def __init__(self, model: Any) -> None:
        super().__init__(model)
        self._names: List[str] = []

    def enroll(self, person_id: str, images: Sequence[np.ndarray]) -> None:
        super().enroll(person_id, images)
        self._names.append(person_id)

    def recognize(self, images: Sequence[np.ndarray]) -> List[Optional[str]]:
        return [self._names[position % len(self._names)] for position in range(len(images))]


def _lifecycle_scenario() -> RecognitionScenario:
    # Two held-out queries per person: the smallest case where the phase
    # split deals one of each person's queries before the enrolment and one
    # after (floor(2/2) = 1 before). With one query per person everything
    # would land after, and the forgetter below would score all-None.
    return RecognitionScenario(
        known=[
            RecognitionIdentity(
                "red",
                [_color_image(255, 0, 0)],
                [_color_image(255, 0, 0), _color_image(255, 0, 0)],
            ),
            RecognitionIdentity(
                "green",
                [_color_image(0, 255, 0)],
                [_color_image(0, 255, 0), _color_image(0, 255, 0)],
            ),
            RecognitionIdentity(
                "blue",
                [_color_image(0, 0, 255)],
                [_color_image(0, 0, 255), _color_image(0, 0, 255)],
            ),
        ],
        unknown_person_id="cyan",
        unknown_queries=[_color_image(0, 255, 255)],
        unknown_enrollment=[_color_image(0, 255, 255)],
        post_enrollment_queries=[_color_image(0, 255, 255)],
    )


class TestDriverSlotAlignment:
    """The order gold is built in and the order questions are asked in.

    ``canonical_query_images`` and ``recognition_expected`` must number slots
    identically or every answer lands on the wrong question. A content-keyed
    adapter that scores full marks through the real driver is the proof;
    the forgetter and the positional cheater are the two order-dependent
    failures the design claims to catch.
    """

    def test_a_content_keyed_adapter_scores_full_marks(self):
        scenario = _lifecycle_scenario()
        output = run_recognition_scenario(lambda model: _ContentLabelAdapter(model), None, scenario)
        scores = score_recognition([output], [recognition_expected(scenario)])
        for metric in (
            "recognition_score",
            "known_identification",
            "unknown_rejection_recall",
            "post_enrollment_accuracy",
        ):
            assert scores[metric] == 1.0, metric

    def test_the_forgetter_scores_exactly_half_on_known(self):
        scenario = _lifecycle_scenario()
        output = run_recognition_scenario(lambda model: _ForgetterAdapter(model), None, scenario)
        scores = score_recognition([output], [recognition_expected(scenario)])
        # One of each person's queries is dealt before the enrolment and one
        # after, so forgetting at the enrolment costs exactly half.
        assert scores["known_identification"] == pytest.approx(0.5)
        assert scores["unknown_rejection_recall"] == 1.0  # the stranger was asked before enrolment
        assert scores["post_enrollment_accuracy"] == 0.0
        assert scores["recognition_score"] == pytest.approx(0.5)

    def test_the_positional_cheater_cannot_score_full_marks(self):
        scenario = _lifecycle_scenario()
        output = run_recognition_scenario(
            lambda model: _PositionalCheaterAdapter(model), None, scenario
        )
        scores = score_recognition([output], [recognition_expected(scenario)])
        assert scores["known_identification"] < 1.0
        assert scores["recognition_score"] < 0.5
        # The local deal is seeded from the names, so the same scenario asks
        # the same questions every time.
        again = run_recognition_scenario(
            lambda model: _PositionalCheaterAdapter(model), None, scenario
        )
        assert again == output

    def test_query_phases_partition_every_slot_exactly_once(self):
        before, after = query_phases((2, 2, 2), unknown_count=2, post_count=2, seed=7)
        assert sorted(before + after) == list(range(10))
        assert not set(before) & set(after)

    def test_each_person_is_split_across_the_phases(self):
        # Canonical slots 0-1 are person one, 2-3 person two, 4-5 person
        # three: each contributes one slot to each phase. The stranger's
        # questions are all dealt before the enrolment; the post-enrolment
        # ones after.
        before, after = query_phases((2, 2, 2), unknown_count=2, post_count=2, seed=7)
        for start in (0, 2, 4):
            person_slots = set(range(start, start + 2))
            assert len(person_slots & set(before)) == 1
            assert len(person_slots & set(after)) == 1
        assert set(range(6, 8)) <= set(before)
        assert set(range(8, 10)) <= set(after)

    def test_an_odd_query_count_leaves_the_extra_after_the_enrolment(self):
        before, after = query_phases((3,), unknown_count=0, post_count=0, seed=7)
        assert len(before) == 1
        assert len(after) == 2

    def test_counts_are_keyword_only(self):
        # unknown_count and post_count are adjacent integers; positionally
        # they transpose silently and the stranger's photos change sides.
        with pytest.raises(TypeError):
            query_phases((2, 2, 2), 2, 2, seed=7)  # type: ignore[misc]

    def test_transposed_counts_move_the_strangers_side_of_the_enrolment(self):
        before, _after = query_phases((2,), unknown_count=2, post_count=3, seed=1)
        assert len(before) == 2 + 1  # one known half plus both strangers
        before, _after = query_phases((2,), unknown_count=3, post_count=2, seed=1)
        assert len(before) == 1 + 3  # the transposition, if it were possible

    def test_recognition_expected_refuses_a_hosted_payload(self):
        from facial_recognition_benchmark.drivers import ShuffledQueryBatches

        scenario = RecognitionScenario(
            known=[RecognitionIdentity("a", [], [])],
            unknown_person_id="new",
            unknown_queries=[],
            unknown_enrollment=[],
            post_enrollment_queries=[],
            shuffled_queries=ShuffledQueryBatches(before_enrollment=[], after_enrollment=[]),
        )
        with pytest.raises(ValueError, match="gold"):
            recognition_expected(scenario)

    def test_a_recognizer_returning_non_string_labels_is_a_contract_error(self):
        class _IntLabels(_ContentLabelAdapter):
            def recognize(self, images: Sequence[np.ndarray]) -> List[Optional[str]]:
                return [0 for _ in images]  # type: ignore[list-item]

        with pytest.raises(AdapterContractError, match="string or None"):
            run_recognition_scenario(lambda model: _IntLabels(model), None, _lifecycle_scenario())


class TestClusteringLabelContract:
    """What cluster() may return, at the driver boundary."""

    @staticmethod
    def _scenario() -> ClusteringScenario:
        return ClusteringScenario(
            images=[_color_image(255, 0, 0), _color_image(0, 255, 0)],
            expected_labels=["a", "b"],
            seed=3,
        )

    @staticmethod
    def _run(labels: Any) -> List[Any]:
        class _Stub:
            def cluster(self, images: Sequence[Any], *, seed: int) -> Any:
                return labels

        return run_clustering_scenario(
            lambda model: _Stub(), None, TestClusteringLabelContract._scenario()
        )

    def test_integer_and_string_labels_pass(self):
        assert self._run([0, 1]) == [0, 1]
        assert self._run(["x", "y"]) == ["x", "y"]

    def test_numpy_labels_are_coerced_to_python(self):
        assert self._run(np.array([0, 1])) == [0, 1]
        assert self._run(np.array(["x", "y"])) == ["x", "y"]

    def test_booleans_are_not_cluster_labels(self):
        with pytest.raises(AdapterContractError, match="string or integer"):
            self._run([True, False])
        with pytest.raises(AdapterContractError, match="string or integer"):
            self._run(np.array([True, False]))

    def test_floats_are_not_cluster_labels(self):
        with pytest.raises(AdapterContractError, match="string or integer"):
            self._run(np.array([0.5, 1.5]))

    def test_a_string_return_is_not_a_label_list(self):
        with pytest.raises(AdapterContractError, match="not a string"):
            self._run("ab")

    def test_a_wrong_length_return_is_rejected(self):
        with pytest.raises(AdapterContractError, match="3 labels for 2 images"):
            self._run([0, 1, 2])


# ---------------------------------------------------------------------------
# Degenerate baselines scored against the shipped public manifests
# ---------------------------------------------------------------------------


@lru_cache(maxsize=None)
def _manifest(tier: str) -> Dict[str, Any]:
    return load_manifest(tier)


def _manifest_clustering_case(tier: str) -> ClusteringScenario:
    scenario = _manifest(tier)["clustering_scenarios"][0]
    return ClusteringScenario(
        images=(),  # never touched by the scoring path
        expected_labels=list(scenario["labels"]),
        seed=int(scenario["seed"]),
        scored=True,
        scenario_key=str(scenario["scenario_id"]),
    )


def _manifest_recognition_scenario(tier: str) -> RecognitionScenario:
    scenario = _manifest(tier)["recognition_scenarios"][0]
    known = [
        RecognitionIdentity(
            person_id=str(identity["person_id"]),
            enrollment=[object()] * len(identity["enrollment"]),
            queries=[object()] * len(identity["queries"]),
        )
        for identity in scenario["known"]
    ]
    unknown = scenario["unknown"]
    return RecognitionScenario(
        known=known,
        unknown_person_id=str(unknown["person_id"]),
        unknown_queries=[object()] * len(unknown["queries"]),
        unknown_enrollment=[object()] * len(unknown["enrollment"]),
        post_enrollment_queries=[object()] * len(unknown["post_enrollment_queries"]),
    )


class TestManifestDisjointness:
    """The leakage guard, against the shipped manifests and against itself."""

    def test_the_shipped_public_manifests_are_disjoint(self):
        assert_disjoint([_manifest("test"), _manifest("evaluation")])

    def test_the_tiers_use_different_row_ranges(self):
        test_rows = [int(sample["row_index"]) for sample in _manifest("test")["samples"]]
        evaluation_rows = [
            int(sample["row_index"]) for sample in _manifest("evaluation")["samples"]
        ]
        assert max(test_rows) < min(evaluation_rows)

    def test_a_shared_row_across_manifests_is_refused(self):
        first = {
            "manifest_id": "one",
            "split": "train",
            "samples": [{"row_index": 5, "source_identity_sha256": "h1"}],
        }
        second = {
            "manifest_id": "two",
            "split": "train",
            "samples": [{"row_index": 5, "source_identity_sha256": "h2"}],
        }
        with pytest.raises(DatasetError, match="share dataset row"):
            assert_disjoint([first, second])

    def test_the_same_row_index_in_a_different_split_is_a_different_row(self):
        first = {
            "manifest_id": "one",
            "split": "train",
            "samples": [{"row_index": 5, "source_identity_sha256": "h1"}],
        }
        second = {
            "manifest_id": "two",
            "split": "valid",
            "samples": [{"row_index": 5, "source_identity_sha256": "h2"}],
        }
        assert_disjoint([first, second])

    def test_a_shared_identity_across_manifests_is_refused(self):
        first = {
            "manifest_id": "one",
            "split": "train",
            "samples": [{"row_index": 1, "source_identity_sha256": "same"}],
        }
        second = {
            "manifest_id": "two",
            "split": "train",
            "samples": [{"row_index": 9, "source_identity_sha256": "same"}],
        }
        with pytest.raises(DatasetError, match="share a source identity"):
            assert_disjoint([first, second])

    def test_reusing_an_identity_inside_one_manifest_is_allowed(self):
        # One person's enrollment photos and their held-out queries share a
        # manifest on purpose; the guard only refuses the reuse across
        # manifests.
        first = {
            "manifest_id": "one",
            "split": "train",
            "samples": [
                {"row_index": 1, "source_identity_sha256": "same"},
                {"row_index": 2, "source_identity_sha256": "same"},
            ],
        }
        assert_disjoint([first])


class TestClusteringDegenerateBaselines:
    """The trivial answers, scored by the real plugin on the real gold.

    The manifests carry truth labels, so these need no images and no
    download -- but they are the real partitions the tiers score.
    """

    def test_one_giant_cluster_is_three_sevenths_on_the_test_tier(self):
        # 12 images, 3 people of 4: truth pairs 3*C(4,2) = 18, predicted
        # pairs C(12,2) = 66, TP = 18, FP = 48, FN = 0. F1 = 36/84 = 3/7.
        # A submission that does nothing scores a third of the way to
        # perfect -- which is why ARI sits next to it, at exactly 0.
        case = _manifest_clustering_case("test")
        scores = ClusteringBenchmark().score([[0] * 12], [case])
        assert scores["clustering_pairwise_f1"] == pytest.approx(3.0 / 7.0)
        assert scores["adjusted_rand_index"] == 0.0

    def test_one_giant_cluster_is_seven_nineteenths_on_the_evaluation_tier(self):
        # 32 images, 4 people of 8: truth pairs 4*C(8,2) = 112, predicted
        # pairs C(32,2) = 496. F1 = 224/608 = 7/19.
        case = _manifest_clustering_case("evaluation")
        scores = ClusteringBenchmark().score([[0] * 32], [case])
        assert scores["clustering_pairwise_f1"] == pytest.approx(7.0 / 19.0)
        assert scores["adjusted_rand_index"] == 0.0

    def test_all_singletons_score_zero_on_both_tiers(self):
        for tier, size in (("test", 12), ("evaluation", 32)):
            case = _manifest_clustering_case(tier)
            scores = ClusteringBenchmark().score([list(range(size))], [case])
            assert scores["clustering_pairwise_f1"] == 0.0, tier
            assert scores["adjusted_rand_index"] == 0.0, tier

    def test_random_partitions_land_at_the_ari_chance_level(self):
        # 300 seeded random groupings: their mean ARI is 0 (the correction
        # does its job on the real partition) while their mean F1 is
        # comfortably above zero -- the reason F1 alone cannot see laziness.
        case = _manifest_clustering_case("test")
        labels = list(case.expected_labels)
        size = len(labels)
        groups = len(set(labels))
        rng = random.Random(20261009)
        aris = []
        f1s = []
        benchmark = ClusteringBenchmark()
        for _ in range(300):
            partition = [rng.randrange(groups) for _ in range(size)]
            scores = benchmark.score([partition], [case])
            aris.append(scores["adjusted_rand_index"])
            f1s.append(scores["clustering_pairwise_f1"])
        assert abs(sum(aris) / len(aris)) < 0.05
        assert sum(f1s) / len(f1s) > 0.05


class TestRecognitionDegenerateBaselines:
    """Trivial recognizers, scored against the manifest-built gold."""

    def test_an_all_none_recognizer_scores_a_quarter(self):
        scenario = _manifest_recognition_scenario("test")
        gold = recognition_expected(scenario)
        output = {key: [None] * len(value) for key, value in gold.items()}
        scores = RecognitionBenchmark().score([output], [scenario])
        assert scores["known_identification"] == 0.0
        assert scores["unknown_rejection_recall"] == 1.0
        assert scores["post_enrollment_accuracy"] == 0.0
        assert scores["unknown_lifecycle"] == pytest.approx(0.5)
        assert scores["recognition_score"] == pytest.approx(0.25)

    def test_calling_everyone_the_stranger_ties_all_none(self):
        # The mirror-image degenerate answer: refuse nothing, and name every
        # query -- including the people already enrolled -- with the
        # stranger's id. Post-enrolment accuracy is perfect, so the score
        # lands on the same 0.25 as refusing everything. Neither half of the
        # metric can see the difference between these two lazinesses; only
        # the diagnostics name them apart.
        scenario = _manifest_recognition_scenario("test")
        gold = recognition_expected(scenario)
        stranger = scenario.unknown_person_id
        output = {key: [stranger] * len(value) for key, value in gold.items()}
        scores = RecognitionBenchmark().score([output], [scenario])
        assert scores["recognition_score"] == pytest.approx(0.25)
        assert scores["known_identification"] == 0.0
        assert scores["unknown_rejection_recall"] == 0.0
        assert scores["post_enrollment_accuracy"] == 1.0

    def test_always_answering_the_first_enrolled_name_is_one_sixth(self):
        # Three enrolled people, two queries each: the first person's two
        # queries are right (macro cell 1.0), the other two cells are 0, so
        # known identification is 1/3. The stranger is misnamed both times
        # and the post-enrolment questions are misnamed, so the unknown
        # lifecycle is 0. Score = (1/3 + 0) / 2 = 1/6.
        scenario = _manifest_recognition_scenario("test")
        gold = recognition_expected(scenario)
        first = scenario.known[0].person_id
        output = {key: [first] * len(value) for key, value in gold.items()}
        scores = RecognitionBenchmark().score([output], [scenario])
        assert scores["known_identification"] == pytest.approx(1.0 / 3.0)
        assert scores["recognition_score"] == pytest.approx(1.0 / 6.0)

    def test_a_confident_wrong_answer_is_diagnosed_as_misnaming(self):
        scenario = _manifest_recognition_scenario("test")
        gold = recognition_expected(scenario)
        first = scenario.known[0].person_id
        output = {key: [first] * len(value) for key, value in gold.items()}
        benchmark = RecognitionBenchmark()
        benchmark.score([output], [scenario])
        notes = " ".join(benchmark.last_diagnostics)
        # Known people were misnamed and the newly enrolled person was named
        # wrongly too: the descriptors/averaging message, not the cutoff one.
        assert "someone else's name" in notes
        assert "called unknown" not in notes and "came back as unknown" not in notes

    def test_a_recognizer_that_never_rejects_is_diagnosed_as_the_cutoff(self):
        # Perfect on the knowns and on the post-enrolment questions, but it
        # named every stranger. That is a cutoff that never fires, and the
        # diagnostic names it while the headline metric reads a perfectly
        # respectable (1 + 0.5) / 2 = 0.75.
        scenario = _manifest_recognition_scenario("test")
        gold = recognition_expected(scenario)
        first = scenario.known[0].person_id
        output = {
            "known": list(gold["known"]),
            "unknown_before": [first] * len(gold["unknown_before"]),
            "post_enrollment": list(gold["post_enrollment"]),
        }
        benchmark = RecognitionBenchmark()
        scores = benchmark.score([output], [scenario])
        assert scores["known_identification"] == 1.0
        assert scores["unknown_rejection_recall"] == 0.0
        assert scores["post_enrollment_accuracy"] == 1.0
        assert scores["recognition_score"] == pytest.approx(0.75)
        notes = " ".join(benchmark.last_diagnostics)
        assert "never fires" in notes or "cutoff never fires" in notes


# ---------------------------------------------------------------------------
# Seed-sweep accounting at the plugin boundary
# ---------------------------------------------------------------------------


class TestSeedSweepAccounting:
    """Sweep repeats report instability without touching the published score."""

    @staticmethod
    def _cases():
        scored = ClusteringScenario(
            images=(),
            expected_labels=[0, 0, 1, 1, 2, 2, 2, 2],
            seed=1,
            scored=True,
            scenario_key="s",
        )
        repeat_one = ClusteringScenario(
            images=(),
            expected_labels=[0, 0, 1, 1, 2, 2, 2, 2],
            seed=101,
            scored=False,
            scenario_key="s",
        )
        repeat_two = ClusteringScenario(
            images=(),
            expected_labels=[0, 0, 1, 1, 2, 2, 2, 2],
            seed=102,
            scored=False,
            scenario_key="s",
        )
        return scored, repeat_one, repeat_two

    def test_repeats_move_the_spread_and_never_the_metric(self):
        scored, repeat_one, repeat_two = self._cases()
        # Repeat one merges persons three and four into person one's label
        # (predicted-same pairs 16, true-positive 8): F1 = 16/24 = 2/3.
        # Repeat two interleaves two clusters: F1 = 4/20 = 1/5.
        outputs = [
            [0, 0, 1, 1, 2, 2, 2, 2],  # perfect: F1 1.0
            [0, 0, 1, 1, 0, 0, 0, 0],  # two people merged: F1 2/3
            [0, 1, 0, 1, 0, 1, 0, 1],  # interleaved: F1 1/5
        ]
        benchmark = ClusteringBenchmark()
        metrics = benchmark.score(outputs, [scored, repeat_one, repeat_two])
        assert metrics["clustering_pairwise_f1"] == pytest.approx(1.0)
        # The spread is measured within the scenario's group, which includes
        # the scored case's own F1: 1.0 minus 0.2.
        assert metrics["clustering_seed_spread"] == pytest.approx(0.8)
        assert any("moved the F1" in note for note in benchmark.last_diagnostics)

    def test_a_tight_sweep_is_reported_as_stable(self):
        scored, repeat_one, repeat_two = self._cases()
        outputs = [
            [0, 0, 1, 1, 2, 2, 2, 2],
            [0, 0, 1, 1, 2, 2, 2, 2],
            [0, 0, 1, 1, 2, 2, 2, 2],
        ]
        benchmark = ClusteringBenchmark()
        metrics = benchmark.score(outputs, [scored, repeat_one, repeat_two])
        assert metrics["clustering_seed_spread"] == pytest.approx(0.0)
        assert any("held" in note for note in benchmark.last_diagnostics)

    def test_repeats_without_a_scenario_key_attach_to_the_latest_scored_case(self):
        scored = ClusteringScenario(
            images=(), expected_labels=[0, 0, 1, 1], seed=1, scored=True, scenario_key=None
        )
        repeat = ClusteringScenario(
            images=(), expected_labels=[0, 0, 1, 1], seed=2, scored=False, scenario_key=None
        )
        benchmark = ClusteringBenchmark()
        # The repeat shares no pair with the truth: F1 = 0.0 against the
        # scored case's 1.0.
        metrics = benchmark.score([[0, 0, 1, 1], [1, 0, 1, 0]], [scored, repeat])
        assert metrics["clustering_seed_spread"] == pytest.approx(1.0)
        # The published F1 still comes from the scored case alone.
        assert metrics["clustering_pairwise_f1"] == pytest.approx(1.0)
