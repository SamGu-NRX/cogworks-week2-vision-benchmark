"""Deterministic recognition and label-invariant clustering metrics."""

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

    Malformed, missing, or wrong-length behavior produces zero-valued metrics
    rather than silently changing the denominator.

    Parameters
    ----------
    outputs
        Student labels produced for each recognition scenario.
    expected
        Trusted labels derived from the scenario definitions.
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
                # No name for someone it was just given. Kept apart from
                # naming the wrong person because the two send a student to
                # different places to look. Which one is right is not decided
                # here: a submission returns None for whatever reason it has,
                # and this never sees a distance or a cutoff.
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
    """

    notes: List[str] = []
    if post_total and post_abstained == post_total:
        notes.append(
            "Every query for the newly enrolled person came back as unknown. "
            "Check that enrolling them stored a profile, then compare the "
            "distances these queries produce against the distances between "
            "photos of someone enrolled from several images: a profile built "
            "from one photo can sit further from a new photo of that person "
            "than one built from several, so see whether one cutoff can serve "
            "both."
        )
    elif post_total and post_abstained > post_confused and post_abstained:
        notes.append(
            "{} of {} queries for the newly enrolled person came back as unknown "
            "rather than named. Compare the distances those queries produce "
            "against your cutoff, and against the distances between photos of "
            "someone enrolled from several images.".format(post_abstained, post_total)
        )
    elif post_confused:
        notes.append(
            "{} of {} queries for the newly enrolled person came back with "
            "someone else's name. Check which profile they matched and at what "
            "distance, and how you combine several photos of one person into "
            "one profile.".format(post_confused, post_total)
        )
    # Gated on abstention, not merely on nothing being correct. A run that
    # named the wrong person every time also has no correct answers, and
    # this sentence describes the opposite shape: a cutoff that is too
    # strict abstains, it does not misname.
    if unknown_rejection >= 0.999 and post_abstained and not post_correct:
        notes.append(
            "Rejecting every stranger and recognizing none of them after "
            "enrollment is what a cutoff that is too strict looks like from both "
            "sides at once. If one distance threshold decides both, that is the "
            "number to look at."
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
    """Average pairwise F1 and adjusted Rand index across clustering cases."""

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

    Parameters
    ----------
    actual
        Cluster identifiers returned by the application.
    expected
        Trusted identity partition.
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
    """Compute the adjusted Rand index without a scikit-learn dependency."""

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
