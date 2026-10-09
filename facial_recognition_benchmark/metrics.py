"""Deterministic recognition and label-invariant clustering metrics.

The pure scoring layer behind ``plugins.py``: no I/O, no model, no randomness,
so the same inputs always produce the same floats. Recognition is scored per
expected identity (a macro average) plus a two-part unknown-person lifecycle;
clustering is scored by pairwise F1 with the adjusted Rand index as a
diagnostic. Wrong-length or missing inputs degrade to zero-valued metrics,
never to an exception and never to a renormalized denominator.
"""

from __future__ import annotations

from collections import Counter
from math import comb
from typing import Dict, List, Mapping, Optional, Sequence

from .contracts import ClusterId, PersonId


def score_recognition(
    outputs: Sequence[Mapping[str, Sequence[Optional[PersonId]]]],
    expected: Sequence[Mapping[str, Sequence[Optional[PersonId]]]],
) -> Dict[str, float]:
    """Score known identification and the unknown-person lifecycle.

    The five metrics, each in [0, 1]:

    - ``known_identification``: accuracy within each expected identity, then
      the mean of those per-identity accuracies. This is a macro average, so
      a person with one query weighs exactly as much as a person with ten;
      the score cannot be raised by flooding easy identities with queries.
    - ``unknown_rejection_recall``: fraction of ``unknown_before`` queries
      answered ``None`` (the stranger was turned away).
    - ``post_enrollment_accuracy``: fraction of ``post_enrollment`` queries
      answered with the expected id. Abstentions and wrong names both count
      against it; `recognition_diagnostics` separates the two because their
      fixes differ.
    - ``unknown_lifecycle``: mean of unknown rejection recall and
      post-enrollment accuracy. Rejecting strangers and then recognizing
      those same people after enrollment are the two halves of one distance
      cutoff, so they are averaged rather than summed.
    - ``recognition_score``: mean of ``known_identification`` and
      ``unknown_lifecycle``.

    The returned mapping carries one extra key, ``_diagnostics``, whose value
    is a list of explanation strings from `recognition_diagnostics` -- the
    only non-float in the mapping. The scoring plugin pops it onto its
    ``last_diagnostics`` surface before reporting numbers; anything else
    reading the mapping as floats must pop it too.

    Malformed input voids the whole submission rather than shrinking the
    denominator: a case-count mismatch between ``outputs`` and ``expected``,
    an empty ``expected``, a wrong-length list in any tracked field, or a
    ``None`` inside the expected known labels all return the all-zero
    mapping, without a ``_diagnostics`` key. A missing field in a case
    mapping behaves as an empty list. An empty unknown or post-enrollment
    list leaves that component at 0.0 instead of dividing by zero, so a
    submission with no stranger or post-enrollment queries at all scores
    0.5, not 1.0.

    Parameters
    ----------
    outputs
        Student labels produced for each recognition scenario: one mapping
        per scenario, each holding ``known``, ``unknown_before``, and
        ``post_enrollment`` lists of predicted ids (``None`` to abstain).
    expected
        Trusted labels derived from the scenario definitions, in the same
        shape and order.

    Returns
    -------
    Dict[str, float]
        The five metrics above; see the notes on ``_diagnostics`` for the
        one value that is a list instead.
    """

    if len(outputs) != len(expected) or not expected:
        return _empty_recognition_score()

    known_by_identity: Dict[PersonId, List[bool]] = {}
    unknown_correct = unknown_total = 0
    post_correct = post_total = 0
    post_abstained = post_confused = 0
    for actual_case, expected_case in zip(outputs, expected):
        actual_known = actual_case.get("known", ())
        expected_known = expected_case.get("known", ())
        if len(actual_known) != len(expected_known):
            return _empty_recognition_score()
        for actual_label, expected_label in zip(actual_known, expected_known):
            if expected_label is None:
                return _empty_recognition_score()
            known_by_identity.setdefault(expected_label, []).append(actual_label == expected_label)

        actual_unknown = actual_case.get("unknown_before", ())
        expected_unknown = expected_case.get("unknown_before", ())
        if len(actual_unknown) != len(expected_unknown):
            return _empty_recognition_score()
        unknown_correct += sum(label is None for label in actual_unknown)
        unknown_total += len(expected_unknown)

        actual_post = actual_case.get("post_enrollment", ())
        expected_post = expected_case.get("post_enrollment", ())
        if len(actual_post) != len(expected_post):
            return _empty_recognition_score()
        for actual_label, expected_label in zip(actual_post, expected_post):
            if actual_label == expected_label:
                post_correct += 1
            elif actual_label is None:
                # Abstained on someone it was just given. Kept apart from
                # naming the wrong person because the fix differs: this is the
                # cutoff being too strict for a profile built from one photo,
                # not the descriptors being wrong.
                post_abstained += 1
            else:
                post_confused += 1
        post_total += len(expected_post)

    known_identification = _mean(
        [sum(results) / len(results) for results in known_by_identity.values() if results]
    )
    unknown_rejection = unknown_correct / unknown_total if unknown_total else 0.0
    post_enrollment = post_correct / post_total if post_total else 0.0
    unknown_lifecycle = (unknown_rejection + post_enrollment) / 2.0
    scores = {
        "known_identification": known_identification,
        "unknown_rejection_recall": unknown_rejection,
        "post_enrollment_accuracy": post_enrollment,
        "unknown_lifecycle": unknown_lifecycle,
        "recognition_score": (known_identification + unknown_lifecycle) / 2.0,
    }
    # Attached to the returned mapping rather than logged, so the caller
    # decides whether to surface it. `score_recognition` is pure otherwise.
    scores["_diagnostics"] = recognition_diagnostics(
        known_identification=known_identification,
        unknown_rejection=unknown_rejection,
        post_correct=post_correct,
        post_abstained=post_abstained,
        post_confused=post_confused,
        post_total=post_total,
    )
    return scores


def recognition_diagnostics(
    *,
    known_identification: float,
    unknown_rejection: float,
    post_correct: int,
    post_abstained: int,
    post_confused: int,
    post_total: int,
) -> List[str]:
    """Sentences naming what a number means and which knob moves it.

    A zero with no explanation is the failure mode this exists to prevent.
    Measured on a real submission: known identification 0.83, unknown
    rejection 1.00, post-enrollment 0.00 -- which reads like a broken
    recognizer and is not one. Its pipeline ranked the newly enrolled person
    first at cosine distance 0.63; its own 0.2846 cutoff then rejected them.
    Same-person distances for people enrolled from several photos sat at
    0.05-0.13, so the cutoff is right for those and too strict for a profile
    built from one image. That is a calibration finding, and it is only
    visible if the two ways of failing are counted apart.

    The branch structure, because the priority order is the meaning:

    - Post-enrollment produces at most one note. Total abstention (every
      query for the new person came back ``None``) is blamed on the cutoff;
      abstentions that merely outnumber misnamings are blamed on the cutoff
      too; otherwise any misnaming is blamed on the descriptors or the
      averaging instead. An even abstain/confuse split therefore reports the
      confusion only, and the abstentions go unmentioned.
    - Two further notes are independent and can co-fire with anything
      above: rejecting every stranger while recognizing none of the newly
      enrolled (one too-strict cutoff, seen from both sides), and
      identifying every known face while rejecting no stranger (a cutoff
      that never fires). Their 0.999 / 0.001 comparisons are
      float-exactness guards that read as "all" and "none".

    Returns
    -------
    List[str]
        Explanation sentences, possibly empty.
    """

    notes: List[str] = []
    if post_total and post_abstained == post_total:
        notes.append(
            "Every query for the newly enrolled person came back as unknown. The "
            "person was added, so this is the cutoff rather than the database: a "
            "profile built from one photo sits further from a new photo of that "
            "person than a profile built from several does. Compare the distance "
            "you get here against the distances between photos of someone already "
            "enrolled, and see whether one cutoff can serve both."
        )
    elif post_total and post_abstained > post_confused and post_abstained:
        notes.append(
            f"{post_abstained} of {post_total} queries for the newly enrolled person "
            "were called unknown rather than named. That is the cutoff being "
            "strict, not a descriptor problem."
        )
    elif post_confused:
        notes.append(
            f"{post_confused} of {post_total} queries for the newly enrolled person "
            "were given someone else's name. The cutoff let them through but the "
            "nearest profile was the wrong one, so this is the descriptors or "
            "the averaging, not the threshold."
        )
    if unknown_rejection >= 0.999 and post_total and not post_correct:
        notes.append(
            "Rejecting every stranger and recognizing none of them after "
            "enrollment is what a cutoff that is too strict looks like from both "
            "sides at once; the same number produces both halves."
        )
    if known_identification >= 0.999 and unknown_rejection <= 0.001:
        notes.append(
            "Every known face was identified and no stranger was ever rejected, "
            "which is what happens when the cutoff never fires: check that an "
            "answer above your distance threshold actually returns None."
        )
    return notes


def score_clustering(
    outputs: Sequence[Sequence[ClusterId]], expected: Sequence[Sequence[ClusterId]]
) -> Dict[str, float]:
    """Average pairwise F1 and adjusted Rand index across clustering cases.

    The two means returned here are what ``ClusteringBenchmark`` reports. A
    case whose labels do not match its truth in length, or whose truth is
    empty, contributes 0.0 to both means but stays in the denominator --
    wrong-length output is scored as a wrong answer, never dropped. A
    case-count mismatch between ``outputs`` and ``expected``, or an empty
    ``expected``, zeroes both metrics outright.

    Parameters
    ----------
    outputs
        One list of predicted cluster ids per clustering case.
    expected
        One list of trusted ids per case, in the same order.

    Returns
    -------
    Dict[str, float]
        ``clustering_pairwise_f1`` and ``adjusted_rand_index``, each the
        mean across cases.
    """

    if len(outputs) != len(expected) or not expected:
        return {"clustering_pairwise_f1": 0.0, "adjusted_rand_index": 0.0}
    f1_scores: List[float] = []
    ari_scores: List[float] = []
    for actual, truth in zip(outputs, expected):
        if len(actual) != len(truth) or not truth:
            f1_scores.append(0.0)
            ari_scores.append(0.0)
            continue
        f1_scores.append(pairwise_f1(actual, truth))
        ari_scores.append(adjusted_rand_index(actual, truth))
    return {
        "clustering_pairwise_f1": _mean(f1_scores),
        "adjusted_rand_index": _mean(ari_scores),
    }


def pairwise_f1(actual: Sequence[ClusterId], expected: Sequence[ClusterId]) -> float:
    """Measure whether pairs are grouped together, ignoring label names.

    Every pair of items is asked two questions: do the predicted labels
    agree, and do the trusted ones. Only the agreement pattern is scored, so
    ``[9, 9, 2, 2]`` and ``["a", "a", "b", "b"]`` are the same answer --
    this is what makes the metric cluster-name-invariant. With zero agreeing
    pairs the precision-recall ratio would divide by zero, so the function
    short-circuits instead: no pairs at all to score (one item, or every
    item its own cluster on both sides) is a perfect 1.0, and any
    disagreement without a single agreeing pair is 0.0.

    Parameters
    ----------
    actual
        Cluster identifiers returned by the application.
    expected
        Trusted identity partition.

    Returns
    -------
    float
        F1 over same-cluster pairs, in [0, 1].
    """

    true_positive = false_positive = false_negative = 0
    for left in range(len(expected)):
        for right in range(left + 1, len(expected)):
            predicted_same = actual[left] == actual[right]
            expected_same = expected[left] == expected[right]
            if predicted_same and expected_same:
                true_positive += 1
            elif predicted_same:
                false_positive += 1
            elif expected_same:
                false_negative += 1
    if true_positive == 0:
        return 1.0 if false_positive == 0 and false_negative == 0 else 0.0
    precision = true_positive / (true_positive + false_positive)
    recall = true_positive / (true_positive + false_negative)
    return 2.0 * precision * recall / (precision + recall)


def adjusted_rand_index(actual: Sequence[ClusterId], expected: Sequence[ClusterId]) -> float:
    """Compute the adjusted Rand index without a scikit-learn dependency.

    Chance-adjusted agreement between two partitions of the same items:
    1.0 for identical partitions, near 0.0 for partitions that agree about
    as often as random pairing would, and negative when they agree less
    often than chance (an alternating split against a 3+3 truth measures
    about -0.111). Cluster-label names never matter.

    Degenerate inputs return 1.0 rather than raising or yielding NaN: with
    fewer than two items there are no pairs to compare, and the correction
    denominator is zero exactly when both partitions are degenerate in the
    same way -- every item in one cluster on both sides, or every item alone
    on both sides. The convention is to read "nothing to disagree about" as
    a perfect score. A submission that puts everyone in one cluster
    therefore earns a 1.0 here while `pairwise_f1` scores the same answer
    strictly below 1.0, which is why the Rand index stays a diagnostic and
    pairwise F1 stays the primary.

    Parameters
    ----------
    actual
        Cluster identifiers returned by the application.
    expected
        Trusted identity partition.

    Returns
    -------
    float
        Adjusted Rand index; 1.0 on the degenerate paths above.
    """

    size = len(expected)
    if size < 2:
        return 1.0
    contingency = Counter(zip(expected, actual))
    expected_counts = Counter(expected)
    actual_counts = Counter(actual)
    sum_pairs = sum(comb(count, 2) for count in contingency.values())
    expected_pairs = sum(comb(count, 2) for count in expected_counts.values())
    actual_pairs = sum(comb(count, 2) for count in actual_counts.values())
    total_pairs = comb(size, 2)
    expected_index = expected_pairs * actual_pairs / total_pairs
    maximum_index = (expected_pairs + actual_pairs) / 2.0
    denominator = maximum_index - expected_index
    if denominator == 0:
        return 1.0
    return (sum_pairs - expected_index) / denominator


def _empty_recognition_score() -> Dict[str, float]:
    return {
        "known_identification": 0.0,
        "unknown_rejection_recall": 0.0,
        "post_enrollment_accuracy": 0.0,
        "unknown_lifecycle": 0.0,
        "recognition_score": 0.0,
    }


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0
