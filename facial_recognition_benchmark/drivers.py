"""Benchmark-owned execution drivers for student application behavior."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .adapters import AdapterContractError, adapt_clustering, adapt_recognition, instantiate
from .contracts import ClusterId, Image, PersonId


@dataclass(frozen=True)
class RecognitionIdentity:
    """Enrollment and held-out images for one known identity."""

    person_id: PersonId
    enrollment: Sequence[Image]
    queries: Sequence[Image]


@dataclass(frozen=True)
class ShuffledQueryBatches:
    """Query images with their labels and their grouping removed.

    A hosted evaluation cannot hand the submission ``known``,
    ``unknown_queries``, and ``post_enrollment_queries`` as three named lists,
    because those names are the answer: everything in ``unknown_queries`` is
    correctly labelled ``None``, and everything in ``post_enrollment_queries``
    is correctly labelled with the identity just enrolled. So the runner deals
    every query image into these two batches in an order the submission cannot
    predict, and un-permutes the predictions afterwards.

    Two batches rather than one, because the stranger's photos are asked about
    twice with two different correct answers, and ``enroll`` has to run between
    the asking. Both batches also carry known-identity queries, so neither can
    be answered with a single constant label.

    Each batch is still one ``recognize`` call. Clustering the whole batch and
    then labelling the clusters is a technique this week teaches, and splitting
    the batch into one call per image would forbid it.
    """

    before_enrollment: Sequence[Image]
    after_enrollment: Sequence[Image]


@dataclass(frozen=True)
class RecognitionScenario:
    """One complete known-to-unknown recognition lifecycle.

    The same fresh adapter is retained across initial enrollment, questions
    about the people it was told about, unknown rejection, enrollment of that
    unknown identity, held-out re-identification, and further questions about
    the people it already knew. Those last ones are the point of retaining one
    adapter: a submission that forgets everybody when it learns somebody new
    is a submission that has not done the task.

    ``shuffled_queries`` is absent for a locally built scenario, which is the
    case a student runs against the public manifests and the case
    ``recognition_scenarios`` returns. Then the three query lists above are
    populated and the driver runs the lifecycle in its plain form. It is
    present only for a scenario rebuilt from a hosted payload, where those
    three lists are empty and the images live in the batches instead.
    """

    known: Sequence[RecognitionIdentity]
    unknown_person_id: PersonId
    unknown_queries: Sequence[Image]
    unknown_enrollment: Sequence[Image]
    post_enrollment_queries: Sequence[Image]
    shuffled_queries: Optional[ShuffledQueryBatches] = None


@dataclass(frozen=True)
class ClusteringScenario:
    """One fixed image partition and random seed for Whispers evaluation.

    ``scored`` marks whether this case contributes to the metrics. The seed
    sweep re-runs the same images under other seeds to report how far a
    randomized clusterer's answer moves; those repeats are diagnostic and
    must not change the number a team publishes.

    ``scenario_key`` names the manifest scenario a case belongs to, so the
    sweep compares each repeat against its own scenario's scored case
    rather than against every scenario at once. ``None`` means the case
    attaches to the most recent scored case in the list, which is the
    order ``clustering_scenarios`` emits them in.
    """

    images: Sequence[Image]
    expected_labels: Sequence[ClusterId]
    seed: int
    scored: bool = True
    scenario_key: Optional[str] = None


RecognitionOutput = Dict[str, List[Optional[PersonId]]]


def run_recognition_scenario(
    factory: Any, model: Any, scenario: RecognitionScenario
) -> RecognitionOutput:
    """Run one stateful recognition lifecycle with a fresh student adapter.

    Parameters
    ----------
    factory
        Raw submission factory or compatible application object.
    model
        Benchmark-owned FaceNet model supplied to the factory.
    scenario
        Enrollment and query sequence to execute.

    Returns
    -------
    dict
        Labels for known, pre-enrollment unknown, and post-enrollment queries.

    Two ``recognize`` calls with the enrolment between them. Each person's
    held-out photos are split across the two, so a person with two or more of
    them is asked about on both sides of the enrolment. See ``_dealt_queries``
    for why, and ``ShuffledQueryBatches`` for the hosted lane it follows.
    """

    adapter = adapt_recognition(instantiate(factory, model))
    for identity in scenario.known:
        adapter.enroll(identity.person_id, identity.enrollment)

    # See `recognition_expected` on why this is a getattr: a scenario-shaped
    # object without the field is a locally built case and takes the plain path.
    batches = getattr(scenario, "shuffled_queries", None)
    if batches is not None:
        return _run_shuffled_queries(adapter, scenario, batches)

    images = canonical_query_images(scenario)
    before_slots, after_slots = query_phases(
        tuple(len(identity.queries) for identity in scenario.known),
        len(scenario.unknown_queries),
        len(scenario.post_enrollment_queries),
        local_query_seed(scenario),
    )

    answers: List[Optional[PersonId]] = [None] * len(images)
    _ask(adapter, images, before_slots, answers, "before-enrollment")
    adapter.enroll(scenario.unknown_person_id, scenario.unknown_enrollment)
    _ask(adapter, images, after_slots, answers, "after-enrollment")

    known_count = sum(len(identity.queries) for identity in scenario.known)
    unknown_count = len(scenario.unknown_queries)
    return {
        "known": answers[:known_count],
        "unknown_before": answers[known_count : known_count + unknown_count],
        "post_enrollment": answers[known_count + unknown_count :],
    }


def _ask(
    adapter: Any,
    images: Sequence[Image],
    slots: Sequence[int],
    answers: List[Optional[PersonId]],
    phase: str,
) -> None:
    """One ``recognize`` call, with each answer filed under the slot it came from."""

    batch = [images[slot] for slot in slots]
    labels = _recognition_labels(adapter.recognize(batch), len(batch), phase)
    for slot, label in zip(slots, labels):
        answers[slot] = label


def canonical_query_images(scenario: RecognitionScenario) -> List[Image]:
    """Every held-out photo in the order the gold is built in.

    Each known identity's photos in turn, then the stranger's photos from
    before the enrolment, then the ones from after. ``recognition_expected``
    builds its answer vector in exactly this order, so slot *i* here and entry
    *i* there are the same photograph. Both lanes number slots this way, which
    is what lets them share ``query_phases``.
    """

    images: List[Image] = []
    for identity in scenario.known:
        images.extend(identity.queries)
    images.extend(scenario.unknown_queries)
    images.extend(scenario.post_enrollment_queries)
    return images


def local_query_seed(scenario: RecognitionScenario) -> int:
    """The permutation a local run asks in, derived from the case's own names.

    Stable, because a submission scored twice has to see the same questions.
    Not secret, and not pretending to be: the names are in the public manifest
    and this function is readable, so a submission can recompute it, replay the
    permutation and score without opening a photograph. That was reproduced on
    both public tiers. Nothing local can close it, because the submission runs
    in this process and can read anything this knows, which is why the local
    command prints LOCAL and SELF-REPORTED. The hosted lane supplies its own
    seed, and hiding it is that lane's problem to solve.
    """

    import hashlib

    names = "|".join(
        [identity.person_id for identity in scenario.known] + [scenario.unknown_person_id]
    )
    return int(hashlib.sha256(names.encode("utf-8")).hexdigest()[:16], 16)


def query_phases(
    known_query_counts: Sequence[int],
    unknown_count: int,
    post_count: int,
    seed: int,
) -> Tuple[Tuple[int, ...], Tuple[int, ...]]:
    """Which canonical query slot is asked before the enrolment, and which after.

    One deal, used by both lanes. The local driver maps the slots back to
    photographs; the hosted controller keeps them as its private map from a
    shuffled batch to a scored answer. Before this they dealt differently, and
    the same submission could score two numbers depending on where it ran.

    Half of each person's own held-out photos go before the enrolment and half
    after, the odd one after. Recognition is not only naming somebody you were
    told about, it is still naming them after you have been told about somebody
    else, and a submission that emptied its database whenever it learned a new
    person answered every question a lifecycle asks when all the known
    questions come first. Splitting each person's own photos rather than
    pooling everybody's is what makes that measurable for every person instead
    of a randomly chosen subset.

    Then both phases are shuffled. Without it each is the known people in
    enrolment order followed by the stranger's photos, and a submission that
    keeps the names it was given and answers by position scores full marks
    without looking at a photograph: measured at 1.0 before the shuffle and
    0.125 after.

    What the shuffle is worth depends on whether the caller's ``seed`` can be
    recomputed by the submission, and that is the caller's question rather than
    this one's. See ``local_query_seed`` for what the local lane can promise.
    """

    import random

    known_count = sum(known_query_counts)
    before: List[int] = []
    after: List[int] = []
    at = 0
    for count in known_query_counts:
        split = count // 2
        before.extend(range(at, at + split))
        after.extend(range(at + split, at + count))
        at += count
    before.extend(range(known_count, known_count + unknown_count))
    after.extend(
        range(known_count + unknown_count, known_count + unknown_count + post_count)
    )

    shuffle = random.Random(seed)
    shuffle.shuffle(before)
    shuffle.shuffle(after)
    return tuple(before), tuple(after)


def _run_shuffled_queries(
    adapter: Any, scenario: RecognitionScenario, batches: ShuffledQueryBatches
) -> RecognitionOutput:
    """The same lifecycle, asked in label-free batches.

    The enrollment calls, their order, and the position of ``enroll`` between
    the two ``recognize`` calls are identical to the plain path. Only the
    contents of the two batches differ, and the caller that built them holds
    the map back to ``known``/``unknown_before``/``post_enrollment``.
    """

    before = _recognition_labels(
        adapter.recognize(batches.before_enrollment),
        len(batches.before_enrollment),
        "before-enrollment",
    )
    adapter.enroll(scenario.unknown_person_id, scenario.unknown_enrollment)
    after = _recognition_labels(
        adapter.recognize(batches.after_enrollment),
        len(batches.after_enrollment),
        "after-enrollment",
    )
    return {"before_enrollment": before, "after_enrollment": after}


def run_clustering_scenario(
    factory: Any, model: Any, scenario: ClusteringScenario
) -> List[ClusterId]:
    """Run and validate one clustering case with a fresh student adapter.

    Cluster identifiers may be arbitrary strings or integers. Only their
    partition relationships are meaningful to scoring.
    """

    adapter = adapt_clustering(instantiate(factory, model))
    labels = adapter.cluster(scenario.images, seed=scenario.seed)
    if isinstance(labels, (str, bytes)):
        raise AdapterContractError("cluster() must return one label per image, not a string.")
    try:
        output = list(labels)
    except TypeError as error:
        raise AdapterContractError("cluster() must return a sequence of labels.") from error
    if len(output) != len(scenario.images):
        raise AdapterContractError(
            f"cluster() returned {len(output)} labels for {len(scenario.images)} images."
        )
    for index, label in enumerate(output):
        if isinstance(label, np.generic):
            label = label.item()
            output[index] = label
        if isinstance(label, bool) or not isinstance(label, (str, int)):
            raise AdapterContractError(f"cluster() label {index} must be a string or integer.")
    return output


def recognition_expected(scenario: RecognitionScenario) -> RecognitionOutput:
    """Construct trusted labels for a recognition lifecycle scenario.

    Raises on a scenario carrying ``shuffled_queries``. That is the sandbox's
    own gold-free view, whose three query lists are empty by construction, so
    the answer built from it would be three empty vectors. Scored, those give a
    quiet zero that looks like a submission failing rather than like the
    controller forgetting to re-attach gold.
    """

    # `getattr` rather than attribute access: this function accepts anything
    # scenario-shaped, and `test_metrics.py` scores a SimpleNamespace built
    # from the five fields below. A structural caller that predates this field
    # is a locally built case, which is exactly the gold-bearing kind.
    if getattr(scenario, "shuffled_queries", None) is not None:
        raise ValueError(
            "This scenario came from a hosted payload and carries no labels. "
            "Re-attach the official gold before scoring it."
        )
    known: List[Optional[PersonId]] = []
    for identity in scenario.known:
        known.extend([identity.person_id] * len(identity.queries))
    return {
        "known": known,
        "unknown_before": [None] * len(scenario.unknown_queries),
        "post_enrollment": [scenario.unknown_person_id] * len(scenario.post_enrollment_queries),
    }


def _recognition_labels(value: Any, expected_count: int, phase: str) -> List[Optional[str]]:
    if isinstance(value, (str, bytes)):
        raise AdapterContractError(
            f"recognize() must return one label per image during {phase}, not a string."
        )
    try:
        output = list(value)
    except TypeError as error:
        raise AdapterContractError(f"recognize() must return a sequence during {phase}.") from error
    if len(output) != expected_count:
        raise AdapterContractError(
            f"recognize() returned {len(output)} labels for {expected_count} images during {phase}."
        )
    normalized: List[Optional[str]] = []
    for index, label in enumerate(output):
        if label is None:
            normalized.append(None)
        elif isinstance(label, str):
            normalized.append(label)
        else:
            raise AdapterContractError(
                f"recognize() label {index} during {phase} must be a string or None."
            )
    return normalized
