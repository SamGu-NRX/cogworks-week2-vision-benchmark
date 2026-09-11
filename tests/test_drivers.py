from __future__ import annotations

import numpy as np
import pytest

from facial_recognition_benchmark.adapters import AdapterContractError
from facial_recognition_benchmark.drivers import (
    ClusteringScenario,
    RecognitionIdentity,
    RecognitionScenario,
    recognition_expected,
    run_clustering_scenario,
    run_recognition_scenario,
)
from facial_recognition_benchmark.metrics import score_recognition


def image(value):
    return np.full((2, 2, 3), value, dtype=np.uint8)


class ClassBasedRecognition:
    def __init__(self, model):
        self.database = {}

    def enroll(self, person_id, images):
        self.database[person_id] = {int(item[0, 0, 0]) for item in images}

    def recognize(self, images):
        output = []
        for item in images:
            value = int(item[0, 0, 0])
            output.append(
                next((name for name, values in self.database.items() if value in values), None)
            )
        return output


def function_based_factory(model):
    database = {}

    class ThinAdapter:
        def enroll(self, person_id, images):
            database[person_id] = {int(item[0, 0, 0]) for item in images}

        def recognize(self, images):
            return [
                next(
                    (name for name, values in database.items() if int(item[0, 0, 0]) in values),
                    None,
                )
                for item in images
            ]

    return ThinAdapter()


def recognition_scenario():
    return RecognitionScenario(
        known=[RecognitionIdentity("known", [image(1)], [image(1)])],
        unknown_person_id="new",
        unknown_queries=[image(2)],
        unknown_enrollment=[image(2)],
        post_enrollment_queries=[image(2)],
    )


def test_different_application_architectures_have_identical_behavior():
    scenario = recognition_scenario()
    class_result = run_recognition_scenario(ClassBasedRecognition, object(), scenario)
    function_result = run_recognition_scenario(function_based_factory, object(), scenario)
    assert class_result == function_result == recognition_expected(scenario)


def test_recognition_lifecycle_retains_one_adapter():
    assert run_recognition_scenario(ClassBasedRecognition, object(), recognition_scenario())[
        "post_enrollment"
    ] == ["new"]


def test_wrong_recognition_length_fails_contract():
    class Broken:
        def enroll(self, person_id, images):
            pass

        def recognize(self, images):
            return []

    with pytest.raises(AdapterContractError, match="returned 0 labels"):
        run_recognition_scenario(lambda model: Broken(), object(), recognition_scenario())


def test_clustering_validates_length_and_label_type():
    scenario = ClusteringScenario([image(1), image(2)], [0, 1], seed=1)

    class Broken:
        def cluster(self, images, *, seed):
            return [object()]

    with pytest.raises(AdapterContractError, match="returned 1 labels"):
        run_clustering_scenario(lambda model: Broken(), object(), scenario)


def retention_scenario():
    """Two people with two held-out photos each, so half fall either side."""

    return RecognitionScenario(
        known=[
            RecognitionIdentity("ada", [image(1)], [image(1), image(1)]),
            RecognitionIdentity("bea", [image(2)], [image(2), image(2)]),
        ],
        unknown_person_id="cass",
        unknown_queries=[image(3)],
        unknown_enrollment=[image(3)],
        post_enrollment_queries=[image(3)],
    )


class ForgetsEveryoneWhenItLearnsSomebodyNew:
    """Throws its database away when it is told about somebody after answering.

    The reproducer, and the reason the lifecycle moved. Under a lifecycle that
    asks about the people it was told about only before the stranger is
    enrolled, this answers every question correctly and scores 1.0: it knows
    everybody while the known questions are asked, and by the time it has
    forgotten them nothing asks again.
    """

    def __init__(self, model):
        self.database = {}
        self.answered = False

    def enroll(self, person_id, images):
        if self.answered:
            self.database = {}
        self.database[person_id] = {int(item[0, 0, 0]) for item in images}

    def recognize(self, images):
        self.answered = True
        return [
            next(
                (name for name, values in self.database.items() if int(item[0, 0, 0]) in values),
                None,
            )
            for item in images
        ]


def test_forgetting_everyone_costs_the_people_it_forgot():
    scenario = retention_scenario()

    output = run_recognition_scenario(ForgetsEveryoneWhenItLearnsSomebodyNew, object(), scenario)
    scores = score_recognition([output], [recognition_expected(scenario)])

    # Each person's first held-out photo is asked while this still knows them
    # and their second after it has thrown them away. It still refuses the
    # stranger and still names them afterwards, so the only thing it loses is
    # the half of the task it failed.
    assert output["known"] == ["ada", None, "bea", None]
    assert output["unknown_before"] == [None]
    assert output["post_enrollment"] == ["cass"]
    assert scores["known_identification"] == 0.5
    assert scores["recognition_score"] == 0.75


def test_remembering_them_is_worth_full_credit():
    scenario = retention_scenario()

    output = run_recognition_scenario(ClassBasedRecognition, object(), scenario)
    scores = score_recognition([output], [recognition_expected(scenario)])

    assert output == recognition_expected(scenario)
    assert scores["recognition_score"] == 1.0


def test_the_stranger_is_still_refused_before_and_named_after():
    scenario = retention_scenario()

    output = run_recognition_scenario(ClassBasedRecognition, object(), scenario)

    assert output["unknown_before"] == [None]
    assert output["post_enrollment"] == ["cass"]


def test_both_phases_ask_about_somebody_already_enrolled():
    """A batch holding only the stranger is answerable with one constant label.

    True of every person with two or more held-out photos, which is every
    person in the public manifests. A person with one is asked about only
    after the enrolment; see ``_dealt_queries``.
    """

    asked = []

    class Watching(ClassBasedRecognition):
        def recognize(self, images):
            asked.append([int(item[0, 0, 0]) for item in images])
            return super().recognize(images)

    run_recognition_scenario(Watching, object(), retention_scenario())

    assert len(asked) == 2
    for batch in asked:
        assert {1, 2} & set(batch), batch


class AnswersByPositionWithoutLookingAtAPhoto:
    """Never examines an image. Keeps the names it was handed and counts.

    Under an unshuffled deal each batch is the known people in enrollment
    order followed by the stranger's photos, so this is enough to answer every
    question correctly without recognizing anything.
    """

    def __init__(self, model):
        self.names = []
        self.stranger = None
        self.answered = False

    def enroll(self, person_id, images):
        if self.answered:
            self.stranger = person_id
        else:
            self.names.append(person_id)

    def recognize(self, images):
        self.answered = True
        known = len(images) - 1
        return [
            self.names[at] if at < known else self.stranger for at in range(len(images))
        ]


def test_answering_by_position_does_not_score():
    """The batches are shuffled, so position says nothing about who is in a photo."""

    scenario = retention_scenario()

    output = run_recognition_scenario(
        AnswersByPositionWithoutLookingAtAPhoto, object(), scenario
    )
    scores = score_recognition([output], [recognition_expected(scenario)])

    assert scores["recognition_score"] != 1.0


def test_the_same_case_is_asked_in_the_same_order_twice():
    """A submission scored twice has to see the same questions."""

    asked = []

    class Watching(ClassBasedRecognition):
        def recognize(self, images):
            asked.append([int(item[0, 0, 0]) for item in images])
            return super().recognize(images)

    run_recognition_scenario(Watching, object(), retention_scenario())
    first = list(asked)
    asked.clear()
    run_recognition_scenario(Watching, object(), retention_scenario())

    assert asked == first
