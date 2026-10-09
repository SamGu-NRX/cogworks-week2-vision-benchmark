"""Behavior locks for the p03-drivers documentation pass on drivers.py.

Every assertion here was verified against observed behavior before the
docstrings were written: the lifecycle call order, the two output shapes,
and the exact AdapterContractError messages. They keep the new docstrings
honest.
"""

import numpy as np
import pytest

from facial_recognition_benchmark.adapters import AdapterContractError
from facial_recognition_benchmark.drivers import (
    ClusteringScenario,
    RecognitionIdentity,
    RecognitionScenario,
    ShuffledQueryBatches,
    recognition_expected,
    run_clustering_scenario,
    run_recognition_scenario,
)


def image(value):
    return np.full((2, 2, 3), value, dtype=np.uint8)


def plain_scenario():
    return RecognitionScenario(
        known=[RecognitionIdentity("known", [image(1)], [image(1)])],
        unknown_person_id="new",
        unknown_queries=[image(2)],
        unknown_enrollment=[image(2)],
        post_enrollment_queries=[image(2)],
    )


def shuffled_scenario():
    return RecognitionScenario(
        known=[RecognitionIdentity("known", [image(1)], [])],
        unknown_person_id="new",
        unknown_queries=[],
        unknown_enrollment=[image(2)],
        post_enrollment_queries=[],
        shuffled_queries=ShuffledQueryBatches(
            before_enrollment=[image(1), image(2)],
            after_enrollment=[image(2)],
        ),
    )


class CountingAdapter:
    """Records the lifecycle call sequence and answers every query None."""

    def __init__(self):
        self.calls = []

    def enroll(self, person_id, images):
        self.calls.append(("enroll", person_id, len(images)))

    def recognize(self, images):
        self.calls.append(("recognize", len(images)))
        return [None] * len(images)


class FixedAnswer:
    """Stub whose recognize()/cluster() always return the given value."""

    def __init__(self, value):
        self.value = value

    def enroll(self, person_id, images):
        pass

    def recognize(self, images):
        return self.value

    def cluster(self, images, *, seed):
        return self.value


def test_plain_path_batches_known_queries_and_returns_three_keys():
    scenario = plain_scenario()
    adapter = CountingAdapter()
    output = run_recognition_scenario(lambda model: adapter, object(), scenario)
    assert set(output) == {"known", "unknown_before", "post_enrollment"}
    assert adapter.calls == [
        ("enroll", "known", 1),
        ("recognize", 1),  # known queries of all identities batched into one call
        ("recognize", 1),  # unknown queries, before the stranger is enrolled
        ("enroll", "new", 1),
        ("recognize", 1),  # post-enrollment queries
    ]


def test_shuffled_path_returns_disjoint_keys_with_same_lifecycle():
    scenario = shuffled_scenario()
    adapter = CountingAdapter()
    output = run_recognition_scenario(lambda model: adapter, object(), scenario)
    assert set(output) == {"before_enrollment", "after_enrollment"}
    assert output["before_enrollment"] == [None, None]
    assert output["after_enrollment"] == [None]
    assert adapter.calls == [
        ("enroll", "known", 1),
        ("recognize", 2),
        ("enroll", "new", 1),
        ("recognize", 1),
    ]


def test_recognition_expected_builds_gold_and_refuses_shuffled():
    scenario = plain_scenario()
    expected = recognition_expected(scenario)
    assert expected == {
        "known": ["known"],
        "unknown_before": [None],
        "post_enrollment": ["new"],
    }
    with pytest.raises(ValueError, match="carries no labels"):
        recognition_expected(shuffled_scenario())


def test_recognition_label_contract_messages_are_exact():
    scenario = plain_scenario()

    with pytest.raises(AdapterContractError) as info:
        run_recognition_scenario(lambda model: FixedAnswer("oops"), object(), scenario)
    assert str(info.value) == (
        "recognize() must return one label per image during known, not a string."
    )

    with pytest.raises(AdapterContractError) as info:
        run_recognition_scenario(lambda model: FixedAnswer(7), object(), scenario)
    assert str(info.value) == "recognize() must return a sequence during known."

    with pytest.raises(AdapterContractError) as info:
        run_recognition_scenario(lambda model: FixedAnswer([None, None]), object(), scenario)
    assert str(info.value) == "recognize() returned 2 labels for 1 images during known."

    # The length check runs before the per-label type check.
    with pytest.raises(AdapterContractError) as info:
        run_recognition_scenario(lambda model: FixedAnswer([42]), object(), scenario)
    assert str(info.value) == "recognize() label 0 during known must be a string or None."


def test_clustering_converts_numpy_scalars_and_rejects_bools():
    scenario = ClusteringScenario([image(1), image(2)], ["a", "b"], seed=5)

    output = run_clustering_scenario(
        lambda model: FixedAnswer(np.array(["a", "b"])), object(), scenario
    )
    assert output == ["a", "b"]
    assert all(type(label) is str for label in output)

    output = run_clustering_scenario(
        lambda model: FixedAnswer(np.array([3, 4])), object(), scenario
    )
    assert output == [3, 4]
    assert all(type(label) is int for label in output)

    with pytest.raises(AdapterContractError) as info:
        run_clustering_scenario(lambda model: FixedAnswer([True, False]), object(), scenario)
    assert str(info.value) == "cluster() label 0 must be a string or integer."

    # np.bool_ is unwrapped with .item() first, then rejected as a plain bool.
    with pytest.raises(AdapterContractError) as info:
        run_clustering_scenario(
            lambda model: FixedAnswer(np.array([np.bool_(True), np.bool_(False)])),
            object(),
            scenario,
        )
    assert str(info.value) == "cluster() label 0 must be a string or integer."


def test_clustering_contract_messages_are_exact():
    scenario = ClusteringScenario([image(1), image(2)], ["a", "b"], seed=5)

    with pytest.raises(AdapterContractError) as info:
        run_clustering_scenario(lambda model: FixedAnswer("ab"), object(), scenario)
    assert str(info.value) == "cluster() must return one label per image, not a string."

    with pytest.raises(AdapterContractError) as info:
        run_clustering_scenario(lambda model: FixedAnswer([1, 2, 3]), object(), scenario)
    assert str(info.value) == "cluster() returned 3 labels for 2 images."

    with pytest.raises(AdapterContractError) as info:
        run_clustering_scenario(lambda model: FixedAnswer(42), object(), scenario)
    assert str(info.value) == "cluster() must return a sequence of labels."


def test_clustering_passes_scenario_seed_to_cluster():
    seen = []

    class SeedProbe:
        def cluster(self, images, *, seed):
            seen.append(seed)
            return ["a", "b"]

    scenario = ClusteringScenario([image(1), image(2)], ["a", "b"], seed=5)
    run_clustering_scenario(lambda model: SeedProbe(), object(), scenario)
    assert seen == [5]
