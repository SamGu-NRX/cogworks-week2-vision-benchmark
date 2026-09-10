"""Discovery for the recognition task: what binds, and what a bound one does.

Every repository here is a module written the way one of the audited 2026
teams wrote theirs, reduced to the shape that matters for the property under
test. The model is a stand-in with FaceNet's two methods, and a photo's faces
are the non-zero numbers along its first row, so a test can say exactly who is
in a picture and how many.
"""

import sys
import tempfile
import unittest
from pathlib import Path
from types import ModuleType

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "python" / "cogbench" / "src"))
sys.path.insert(0, str(ROOT / "benchmarks" / "week2"))

from facial_recognition_benchmark.drivers import (  # noqa: E402
    RecognitionIdentity,
    RecognitionScenario,
    recognition_expected,
    run_recognition_scenario,
)
from facial_recognition_benchmark.metrics import score_recognition  # noqa: E402
from facial_recognition_benchmark.roles import (  # noqa: E402
    descriptors_in,
    looks_like_an_empty_database,
    looks_like_face_descriptors,
    named,
    recognition_fixture,
)

#: Wide enough to be a descriptor rather than a detection box, narrow enough
#: that a test can read one.
WIDTH = 32


def photo(*faces: int) -> np.ndarray:
    """A picture of the people named by these numbers. No numbers, no face."""

    value = np.zeros((4, 4, 3), dtype=np.uint8)
    for index, who in enumerate(faces):
        value[0, index, 0] = who
    return value


def faces_in(image) -> list:
    return [int(who) for who in np.asarray(image)[0, :, 0] if who]


def descriptor_of(who: int) -> np.ndarray:
    """One-hot, so the cosine distance between two people is exactly 1."""

    vector = np.zeros(WIDTH, dtype=float)
    vector[who % WIDTH] = 1.0
    return vector


class FakeFaceNet:
    """The two methods the benchmark and every audited team call on the model."""

    def detect(self, image):
        faces = faces_in(image)
        return (
            np.zeros((len(faces), 4), dtype=float),
            np.full(len(faces), 0.99, dtype=float),
            np.zeros((len(faces), 5, 2), dtype=float),
        )

    def compute_descriptors(self, image, boxes):
        faces = faces_in(image)
        if not faces:
            return np.zeros((0, WIDTH), dtype=float)
        return np.stack([descriptor_of(who) for who in faces])


def _written(name: str, source: str) -> ModuleType:
    """A module whose functions look like a team's, for the search to try."""

    module = ModuleType(name)
    module.__name__ = name
    exec(compile(source, name + ".py", "exec"), module.__dict__)
    for value in list(module.__dict__.values()):
        if callable(value) and getattr(value, "__module__", None) in (None, "builtins"):
            try:
                value.__module__ = name
            except (AttributeError, TypeError):
                pass
    return module


def scenario_of(known, unknown, unknown_id="stranger"):
    """A lifecycle case built from face numbers rather than photographs.

    ``known`` is ``{name: (enrollment_faces, query_faces)}`` and each face
    number becomes one single-face photo.
    """

    identities = [
        RecognitionIdentity(
            person_id=name,
            enrollment=[photo(who) for who in enrollment],
            queries=[photo(who) for who in queries],
        )
        for name, (enrollment, queries) in known.items()
    ]
    return RecognitionScenario(
        known=identities,
        unknown_person_id=unknown_id,
        unknown_queries=[photo(unknown)],
        unknown_enrollment=[photo(unknown)],
        post_enrollment_queries=[photo(unknown)],
    )


#: Two people with three enrollment photos and two questions each, plus a
#: stranger. The shape of the public test tier, at the size a unit test wants.
FIRST = scenario_of(
    {"ada": ((1, 2, 3), (1, 2)), "bea": ((4, 5, 6), (4, 5))},
    unknown=7,
    unknown_id="cass",
)

#: The same lifecycle with different people and different photos. Every number
#: here is one no scenario above used, so an answer that comes back right can
#: only have come from this run.
SECOND = scenario_of(
    {"dov": ((11, 12, 13), (11, 12)), "eve": ((14, 15, 16), (14, 15))},
    unknown=17,
    unknown_id="finn",
)


#: The shape the course teaches: a database of your own, a step that describes
#: a photo, a call that files a descriptor under a name, and a call that looks
#: one up and can say it does not know.
NORMAL = '''
import numpy as np


def make_database():
    return {}


def describe(image, model):
    boxes, probabilities, landmarks = model.detect(image)
    return model.compute_descriptors(image, boxes)


def add_face(database, name, descriptor):
    database.setdefault(name, []).append(descriptor)


def cos_dist(one, other):
    return 1.0 - float(
        np.dot(one, other) / (np.linalg.norm(one) * np.linalg.norm(other))
    )


def whose_face(database, descriptor, cutoff=0.4):
    best, closest = "Unknown", 2.0
    for name, descriptors in database.items():
        for other in descriptors:
            distance = cos_dist(descriptor, other)
            if distance < closest:
                best, closest = name, distance
    if closest > cutoff:
        return "Unknown"
    return best
'''


def adapter_for(found):
    """The object a scored run gets, built the way the plugin builds it."""

    from facial_recognition_benchmark.plugins import RecognitionBenchmark

    return RecognitionBenchmark().submission_from_discovery(found)


class _ARecognitionSearch(unittest.TestCase):
    """Runs the plugin's own discovery fields against a repository on disk."""

    def fixture_for(self, scenario):
        made = recognition_fixture(scenario)
        self.assertIsNotNone(made, "the scenario is too small to search with")
        return made

    def resolve(self, source, scenario=FIRST, model=None, name="theirs.py"):
        from cogbench.pipeline import Fixtures
        from cogbench.resolve import resolve

        from facial_recognition_benchmark.roles import (
            DESCRIBE_ROLE,
            enrollment_arrangements,
            recognition_accepts,
            write_photos,
        )

        root = Path(tempfile.mkdtemp(prefix="cogworks-week2-repo-")).resolve()
        (root / name).write_text(source)
        fixture = self.fixture_for(scenario)
        forms = Fixtures(((fixture.photos,), (write_photos(fixture.photos),)))
        self.said = {}

        def accepts(chain, enroll, query):
            passed, detail = recognition_accepts(
                chain, forms.for_chain(chain)[0], fixture, enroll, query
            )
            self.said.setdefault(tuple(step.label for step in chain), detail)
            return passed, detail

        return resolve(
            root,
            chain_role=DESCRIBE_ROLE,
            fixture=forms,
            accepts=accepts,
            arrangements=enrollment_arrangements,
            factories=looks_like_an_empty_database,
            extras={"model": model if model is not None else FakeFaceNet()},
        )

    def why(self):
        return "\n".join(str(detail) for detail in self.said.values())

    def scored(self, found, scenario, model=None):
        """Drive the real driver over a scenario and score it, as a run does."""

        from facial_recognition_benchmark.plugins import RecognitionBenchmark

        plugin = RecognitionBenchmark()
        output = run_recognition_scenario(
            lambda *args, **kwargs: plugin.submission_from_discovery(found),
            model if model is not None else FakeFaceNet(),
            scenario,
        )
        return output, score_recognition([output], [recognition_expected(scenario)])


class AWholeLifecycleRunsOnADiscoveredRepository(_ARecognitionSearch):
    """The point of the whole thing.

    Enrol two people, name them from photos they have not been shown, refuse a
    stranger, learn the stranger, name them, and still name the first two.
    """

    def test_a_repository_written_the_way_the_course_teaches_is_found(self):
        found = self.resolve(NORMAL)

        self.assertTrue(found.ready, self.why())
        self.assertEqual(found.verdict.status, "scored")
        self.assertEqual([step.label for step in found.chain], ["theirs.describe"])
        self.assertEqual(found.attempt.enroll, "theirs.add_face")
        self.assertEqual(found.attempt.query, "theirs.whose_face")

    def test_the_lifecycle_it_runs_is_the_one_the_benchmark_asked_for(self):
        found = self.resolve(NORMAL)

        output, scores = self.scored(found, FIRST)

        self.assertEqual(output["known"], ["ada", "ada", "bea", "bea"])
        self.assertEqual(output["unknown_before"], [None])
        self.assertEqual(output["post_enrollment"], ["cass"])
        self.assertEqual(scores["recognition_score"], 1.0)

    def test_the_people_enrolled_first_are_still_known_afterwards(self):
        found = self.resolve(NORMAL)
        adapter = adapter_for(found)

        adapter.enroll("ada", [photo(1), photo(2)])
        adapter.enroll("bea", [photo(4)])
        before = adapter.recognize([photo(7)])
        adapter.enroll("cass", [photo(7)])

        self.assertEqual(before, [None])
        self.assertEqual(
            adapter.recognize([photo(7), photo(1), photo(4)]), ["cass", "ada", "bea"]
        )


class EachScenarioStartsFromADatabaseTheirOwnCodeJustMade(_ARecognitionSearch):
    """Proving the binding filled their database, and the score must not see it.

    The search enrols `fixture_a` and `fixture_b` to find out whether a store
    works. A scored run that started from that database would be answering
    about the search's people as well as the case's.
    """

    def test_the_search_s_people_are_not_in_the_scored_run(self):
        found = self.resolve(NORMAL)

        first, _ = self.scored(found, FIRST)
        second, scores = self.scored(found, SECOND)

        self.assertNotIn("fixture_a", first["known"] + first["post_enrollment"])
        self.assertEqual(second["known"], ["dov", "dov", "eve", "eve"])
        self.assertEqual(second["post_enrollment"], ["finn"])
        self.assertEqual(scores["recognition_score"], 1.0)

    def test_the_first_scenario_s_people_are_not_in_the_second(self):
        found = self.resolve(NORMAL)

        self.scored(found, FIRST)
        second, _ = self.scored(found, SECOND)

        self.assertEqual(second["unknown_before"], [None])
        self.assertNotIn("ada", second["known"])


#: The same code with its database in a module global, which is what one
#: audited repository does. Nothing here can be rebuilt, so the second
#: scenario inherits the first.
A_DATABASE_THAT_NEVER_RESETS = '''
import numpy as np

_DB = {}


def describe(image, model):
    boxes, probabilities, landmarks = model.detect(image)
    return model.compute_descriptors(image, boxes)


def add_face(name, descriptor):
    _DB.setdefault(name, []).append(descriptor)


def whose_face(descriptor, cutoff=0.4):
    best, closest = "Unknown", 2.0
    for name, descriptors in _DB.items():
        for other in descriptors:
            distance = 1.0 - float(
                np.dot(descriptor, other)
                / (np.linalg.norm(descriptor) * np.linalg.norm(other))
            )
            if distance < closest:
                best, closest = name, distance
    return "Unknown" if closest > cutoff else best
'''


class ADatabaseInAModuleGlobalIsNotIsolated(_ARecognitionSearch):
    """A limit of the platform, written down rather than worked around.

    Nothing at this layer can give a module global an empty database: the
    benchmark cannot know which of their functions writes to one before it
    calls it, and restoring a module's globals between attempts belongs to
    whatever is doing the calling. What happens instead depends on their code.
    This one reads back rows the search's probing left behind, raises, and is
    refused, which is the better of the two outcomes available; the other is a
    run scored against rows the benchmark put there itself.
    """

    def test_the_repository_is_refused_rather_than_scored_against_that(self):
        found = self.resolve(A_DATABASE_THAT_NEVER_RESETS)

        self.assertFalse(found.ready)
        self.assertEqual(found.verdict.status, "not_wired")


class EveryQueryPhotoGetsExactlyOneAnswer(_ARecognitionSearch):
    """The driver counts labels against images and refuses a mismatch.

    Zero faces and several faces are the two ways a photo stops being one
    face, and both have to come back as one answer.
    """

    def setUp(self):
        self.found = self.resolve(NORMAL)
        self.adapter = adapter_for(self.found)
        self.adapter.enroll("ada", [photo(1)])
        self.adapter.enroll("bea", [photo(4)])

    def test_a_photo_with_no_face_in_it_is_one_answer_of_none(self):
        self.assertEqual(self.adapter.recognize([photo()]), [None])

    def test_a_photo_with_two_faces_is_still_one_answer(self):
        self.assertEqual(self.adapter.recognize([photo(4, 1)]), ["bea"])

    def test_a_photo_of_two_strangers_is_one_answer_of_none(self):
        self.assertEqual(self.adapter.recognize([photo(20, 21)]), [None])

    def test_the_count_holds_across_a_mixed_batch(self):
        images = [photo(1), photo(), photo(4, 1), photo(99)]

        self.assertEqual(self.adapter.recognize(images), ["ada", None, "bea", None])

    def test_a_photo_with_no_face_enrols_nothing(self):
        self.adapter.enroll("nobody", [photo()])

        self.assertEqual(self.adapter.recognize([photo(1)]), ["ada"])


#: A query that ranks the nearest rows and never rejects, beside one that puts
#: the team's own cutoff in front of the same ranking. Bagel wrote both.
RANKS_AND_PREDICTS = '''
import numpy as np


def make_database():
    return {}


def describe(image, model):
    boxes, probabilities, landmarks = model.detect(image)
    return model.compute_descriptors(image, boxes)


def add(database, name, descriptor):
    database.setdefault(name, []).append(descriptor)


def _ranked(database, descriptor):
    rows = []
    for name, descriptors in database.items():
        for other in descriptors:
            similarity = float(
                np.dot(descriptor, other)
                / (np.linalg.norm(descriptor) * np.linalg.norm(other))
            )
            rows.append({"name": name, "similarity": similarity})
    return sorted(rows, key=lambda row: -row["similarity"])


def query(database, descriptor):
    return _ranked(database, descriptor)[:5]


def predict(database, descriptor, threshold=0.6):
    results = query(database, descriptor)
    if not results:
        return {"prediction": "unknown", "results": []}
    best = results[0]
    prediction = best["name"] if best["similarity"] >= threshold else "unknown"
    return {"prediction": prediction, "similarity": best["similarity"], "results": results}
'''


class TheAnswerThatCanRejectIsTheOneThatBinds(_ARecognitionSearch):
    """Two of their own functions answer, and only one of them says "not this".

    Binding the first one reached would score that team zero on half the
    benchmark using code they wrote. Two things keep that from happening: a
    bare ranking states no name, so it is not read as an answer at all, and a
    query that names the stranger grades half against a full mark for one that
    rejects. This is Bagel's `query` and `predict`, reduced.
    """

    def test_the_thresholded_answer_wins_over_the_bare_ranking(self):
        found = self.resolve(RANKS_AND_PREDICTS)

        self.assertTrue(found.ready, self.why())
        self.assertEqual(found.attempt.query, "theirs.predict")

    def test_their_own_rejection_is_what_the_run_reports(self):
        found = self.resolve(RANKS_AND_PREDICTS)

        _, scores = self.scored(found, FIRST)

        self.assertEqual(scores["unknown_rejection_recall"], 1.0)
        self.assertEqual(scores["recognition_score"], 1.0)


#: A team whose cutoff never rejects anybody. Their binding is fine; their
#: number is not.
A_CUTOFF_THAT_NEVER_REJECTS = '''
import numpy as np


def make_database():
    return {}


def describe(image, model):
    boxes, probabilities, landmarks = model.detect(image)
    return model.compute_descriptors(image, boxes)


def add_face(database, name, descriptor):
    database.setdefault(name, []).append(descriptor)


def whose_face(database, descriptor):
    best, closest = "Unknown", 3.0
    for name, descriptors in database.items():
        for other in descriptors:
            distance = 1.0 - float(
                np.dot(descriptor, other)
                / (np.linalg.norm(descriptor) * np.linalg.norm(other))
            )
            if distance < closest:
                best, closest = name, distance
    return best
'''


class ABadlyChosenCutoffIsScoredRatherThanRefused(_ARecognitionSearch):
    """The rule this file inherits from the clustering side of roles.py.

    Discovery decides whether a binding works. How well it works is the
    metric's question, and a team who never says "I do not know" should read
    that in their unknown rejection rather than in a wiring message.
    """

    def test_it_still_binds(self):
        found = self.resolve(A_CUTOFF_THAT_NEVER_REJECTS)

        self.assertTrue(found.ready, self.why())

    def test_the_number_says_what_is_wrong(self):
        found = self.resolve(A_CUTOFF_THAT_NEVER_REJECTS)

        _, scores = self.scored(found, FIRST)

        self.assertEqual(scores["known_identification"], 1.0)
        self.assertEqual(scores["unknown_rejection_recall"], 0.0)


#: Their describe step hands back the boxes and the descriptors together, as
#: three of the five audited repositories do.
BOXES_AND_DESCRIPTORS = '''
import numpy as np


def make_database():
    return {}


def detect_and_describe(model, image):
    boxes, probabilities, landmarks = model.detect(image)
    return boxes, model.compute_descriptors(image, boxes)


def add_descriptor(database, name, descriptor):
    database.setdefault(name, []).append(descriptor)


def find_match(database, descriptor, threshold=0.45):
    names = list(database.keys())
    if not names:
        return "Unknown"
    best, closest = "Unknown", 3.0
    for name in names:
        for other in database[name]:
            distance = 1.0 - float(
                np.dot(descriptor, other)
                / (np.linalg.norm(descriptor) * np.linalg.norm(other))
            )
            if distance < closest:
                best, closest = name, distance
    return best if closest < threshold else "Unknown"
'''


class ADetectionBoxIsNotADescriptor(_ARecognitionSearch):
    """The boxes come first out of a returned pair and pass a float-array test.

    Bound as the descriptors they score a whole run against the corners of a
    face rather than the face, which is a wrong answer that looks like a
    working pipeline.
    """

    def test_the_descriptors_are_what_binds(self):
        found = self.resolve(BOXES_AND_DESCRIPTORS)

        self.assertTrue(found.ready, self.why())
        self.assertEqual([step.label for step in found.chain], ["theirs.detect_and_describe"])
        _, scores = self.scored(found, FIRST)
        self.assertEqual(scores["recognition_score"], 1.0)

    def test_the_predicate_refuses_a_box_and_accepts_a_descriptor(self):
        self.assertFalse(looks_like_face_descriptors(np.zeros((2, 4), dtype=float)))
        self.assertFalse(looks_like_face_descriptors(np.full(2, 0.9, dtype=float)))
        self.assertTrue(looks_like_face_descriptors(np.zeros((2, WIDTH), dtype=float)))
        self.assertTrue(looks_like_face_descriptors(np.zeros(WIDTH, dtype=float)))


class TheModelTheBenchmarkOwnsReachesAStepThatAsksForIt(_ARecognitionSearch):
    """One audited team takes it as their first argument and builds none.

    Without it their describe step cannot be called at all, and the repository
    reads as having no descriptors when what it has is a parameter.
    """

    def test_the_model_is_offered_before_the_photo_as_well_as_after(self):
        found = self.resolve(BOXES_AND_DESCRIPTORS)

        self.assertTrue(found.ready, self.why())
        supplied = getattr(found.chain[0], "supplied", {})
        self.assertIn("model", supplied)


#: Their describe step reads a file, which is the form the capstone document
#: tells students to write.
A_PATH_TAKING_STEP = '''
import numpy as np
from PIL import Image


def make_database():
    return {}


def get_descriptor(image_path):
    picture = np.asarray(Image.open(str(image_path)).convert("RGB"))
    faces = [int(who) for who in picture[0, :, 0] if who]
    if not faces:
        return None
    vector = np.zeros(32, dtype=float)
    vector[faces[0] % 32] = 1.0
    return vector


def add_face(database, name, descriptor):
    database.setdefault(name, []).append(descriptor)


def whose_face(database, descriptor, cutoff=0.4):
    best, closest = "Unknown", 3.0
    for name, descriptors in database.items():
        for other in descriptors:
            distance = 1.0 - float(
                np.dot(descriptor, other)
                / (np.linalg.norm(descriptor) * np.linalg.norm(other))
            )
            if distance < closest:
                best, closest = name, distance
    return best if closest < cutoff else "Unknown"
'''


class TheScoredRunPresentsTheFormTheSearchProved(_ARecognitionSearch):
    """A chain bound on paths has to be given paths when it is scored."""

    def test_a_path_taking_step_binds_and_then_runs_on_paths(self):
        found = self.resolve(A_PATH_TAKING_STEP)

        self.assertTrue(found.ready, self.why())
        self.assertEqual(getattr(found.chain[0], "form", None), 1)
        _, scores = self.scored(found, FIRST)
        self.assertEqual(scores["recognition_score"], 1.0)


#: Their zero-argument function hands back the ten people they committed to
#: their repository, which is a loader rather than a factory.
A_DATABASE_THEY_SHIPPED_FULL = '''
import numpy as np

_SAVED = {"someone": [np.ones(32, dtype=float)]}


def load_database():
    return dict(_SAVED)


def make_database():
    return {}
'''


class ALoaderIsNotAnEmptyDatabaseFactory(unittest.TestCase):
    """Scoring a run against a database that arrived full scores their data."""

    def setUp(self):
        self.module = _written("shipped", A_DATABASE_THEY_SHIPPED_FULL)

    def test_a_function_that_hands_back_people_is_refused(self):
        self.assertFalse(looks_like_an_empty_database(self.module.load_database))

    def test_a_function_that_hands_back_nothing_is_accepted(self):
        self.assertTrue(looks_like_an_empty_database(self.module.make_database))

    def test_an_object_holding_people_is_refused_and_an_empty_one_is_not(self):
        class Full:
            def __init__(self):
                self.profiles = {"someone": [1]}

        class Empty:
            def __init__(self):
                self.profiles = {}
                self.path = None

        self.assertFalse(looks_like_an_empty_database(Full))
        self.assertTrue(looks_like_an_empty_database(Empty))

    def test_a_function_that_needs_an_argument_is_not_a_factory(self):
        self.assertFalse(looks_like_an_empty_database(lambda path: {}))


class AnAnswerIsReadAgainstTheNamesTheBenchmarkEnrolled(unittest.TestCase):
    """No table of sentinels: the benchmark chose every name it handed over.

    "Unknown", "unknown", and a dictionary whose prediction is "unknown" are
    all answers the benchmark has no name for, and so is a name it never
    enrolled.
    """

    KNOWN = frozenset({"ada", "bea"})

    def test_a_name_it_enrolled_comes_back_as_that_name(self):
        self.assertEqual(named("ada", self.KNOWN), "ada")

    def test_the_wrong_name_is_kept_rather_than_flattened_to_none(self):
        # Naming the wrong person and naming nobody are different mistakes
        # with different fixes, and the scorer reports them apart.
        self.assertEqual(named("bea", self.KNOWN), "bea")

    def test_a_name_it_did_not_enrol_is_none_whatever_the_word(self):
        # No table of sentinels: the rule is that a name this run did not
        # enrol is not an identification, and every spelling of "unknown" the
        # corpus writes falls out of it.
        for answer in ("Unknown", "unknown", "", "no match", "Nobody"):
            self.assertIsNone(named(answer, self.KNOWN), answer)

    def test_a_name_and_a_distance_is_read_from_the_pair(self):
        self.assertEqual(named(("ada", 0.2), self.KNOWN), "ada")
        self.assertIsNone(named(("Unknown", 0.9), self.KNOWN), None)

    def test_a_prediction_dictionary_is_read_from_its_decision(self):
        rejected = {
            "prediction": "unknown",
            "similarity": 0.5,
            "results": [{"name": "ada", "similarity": 0.5}],
        }
        named_one = dict(rejected, prediction="ada")

        # Their ranking still holds ada. Reading the closest match instead of
        # their decision would turn a correct rejection into an identification.
        self.assertIsNone(named(rejected, self.KNOWN))
        self.assertEqual(named(named_one, self.KNOWN), "ada")

    def test_the_decision_is_read_whichever_order_they_built_it_in(self):
        # Reading deeper than the answer's own parts found the top-ranked name
        # inside `results` and turned a rejection into an identification, so
        # the same dictionary answered two ways depending on which key its
        # author wrote first.
        rejected = {"results": [{"name": "ada", "similarity": 0.9}], "prediction": "unknown"}
        named_one = {"results": [{"name": "ada", "similarity": 0.9}], "prediction": "ada"}

        self.assertIsNone(named(rejected, self.KNOWN))
        self.assertEqual(named(named_one, self.KNOWN), "ada")

    def test_a_rejection_spelled_as_nothing_is_not_overridden_by_the_ranking(self):
        self.assertIsNone(
            named({"prediction": None, "results": [{"name": "ada"}]}, self.KNOWN)
        )

    def test_a_ranking_with_no_decision_in_it_is_not_read(self):
        # A name inside a collection is not a name the answer states. No
        # audited repository answers only this way; `DiscoverySpec.readers` is
        # what the SDK provides for running one of their own functions over it.
        self.assertIsNone(named([{"name": "ada", "similarity": 0.9}], self.KNOWN))

    def test_an_answer_with_no_name_in_it_is_none(self):
        self.assertIsNone(named(None, self.KNOWN))
        self.assertIsNone(named(0.4, self.KNOWN))
        self.assertIsNone(named([], self.KNOWN))


#: A describe step that hands back the same buffer every time, which is a
#: thing a team writes to avoid allocating, and a store that keeps what it is
#: handed, which Bagel's does.
A_REUSED_OUTPUT_BUFFER = '''
import numpy as np

_BUFFER = np.zeros((1, 32), dtype=float)


def make_database():
    return {}


def describe(image, model):
    boxes, probabilities, landmarks = model.detect(image)
    described = model.compute_descriptors(image, boxes)
    if len(described) == 0:
        return np.zeros((0, 32), dtype=float)
    _BUFFER[0] = described[0]
    return _BUFFER


def add_face(database, name, descriptor):
    database.setdefault(name, []).append(np.asarray(descriptor).ravel())


def whose_face(database, descriptor, cutoff=0.4):
    best, closest = "Unknown", 3.0
    for name, descriptors in database.items():
        for other in descriptors:
            distance = 1.0 - float(
                np.dot(descriptor, other)
                / (np.linalg.norm(descriptor) * np.linalg.norm(other))
            )
            if distance < closest:
                best, closest = name, distance
    return best if closest < cutoff else "Unknown"
'''


class WhatWasEnrolledStaysEnrolled(_ARecognitionSearch):
    """A describe step may hand back the same array every time.

    Their store keeps what it is given, so without a copy the second photo
    overwrites the first person's descriptor and everybody becomes whoever was
    described last.
    """

    def test_an_earlier_enrolment_survives_a_later_photo(self):
        found = self.resolve(A_REUSED_OUTPUT_BUFFER)
        self.assertTrue(found.ready, self.why())
        adapter = adapter_for(found)

        adapter.enroll("ada", [photo(1)])
        adapter.enroll("bea", [photo(4)])

        self.assertEqual(adapter.recognize([photo(1), photo(4)]), ["ada", "bea"])


class OnePhotoAtATimeIsHowManyFacesTheStepFound(unittest.TestCase):
    """Three shapes say the same thing about one photo, and one says nothing."""

    def test_a_vector_is_one_face(self):
        rows = descriptors_in(np.ones(WIDTH, dtype=float))

        self.assertEqual(len(rows), 1)

    def test_a_matrix_is_one_row_per_face(self):
        self.assertEqual(len(descriptors_in(np.zeros((3, WIDTH), dtype=float))), 3)

    def test_a_step_bound_per_photo_wraps_its_answer_in_a_list(self):
        self.assertEqual(len(descriptors_in([np.zeros((2, WIDTH), dtype=float)])), 2)

    def test_no_face_is_no_rows(self):
        self.assertEqual(descriptors_in(None), [])
        self.assertEqual(descriptors_in(np.zeros((0, WIDTH), dtype=float)), [])
        self.assertEqual(descriptors_in([]), [])


class TheFixtureIsSmallAndIsNotTheScoredCase(unittest.TestCase):
    """Every photo in it is described once per candidate the search tries."""

    def test_it_takes_two_people_a_question_and_a_stranger(self):
        fixture = recognition_fixture(FIRST)

        self.assertEqual(len(fixture.photos), 6)
        self.assertEqual([name for name, _ in fixture.enrollment], ["fixture_a", "fixture_b"])
        self.assertEqual(fixture.query_of, "fixture_a")
        self.assertEqual(faces_in(fixture.photos[fixture.query]), [1])
        self.assertEqual(faces_in(fixture.photos[fixture.stranger]), [7])

    def test_its_names_are_none_of_the_scored_case_s(self):
        fixture = recognition_fixture(FIRST)
        theirs = {person.person_id for person in FIRST.known} | {FIRST.unknown_person_id}

        self.assertFalse(theirs & {name for name, _ in fixture.enrollment})

    def test_a_case_too_small_to_search_with_says_so(self):
        thin = RecognitionScenario(
            known=[RecognitionIdentity("only", [photo(1)], [photo(1)])],
            unknown_person_id="new",
            unknown_queries=[photo(2)],
            unknown_enrollment=[photo(2)],
            post_enrollment_queries=[photo(2)],
        )

        self.assertIsNone(recognition_fixture(thin))


#: A repository whose describe step finds nothing in the benchmark's photos.
A_STEP_THAT_FINDS_NO_FACE = '''
import numpy as np


def make_database():
    return {}


def describe(image, model):
    return np.zeros((0, 32), dtype=float)


def add_face(database, name, descriptor):
    database.setdefault(name, []).append(descriptor)


def whose_face(database, descriptor):
    return "Unknown"
'''


class ARepositoryThatFindsNoFaceIsRefusedAtTheStepThatFailed(_ARecognitionSearch):
    """Better than a quiet score near chance.

    A submission whose descriptors are always empty scores exactly what one
    that answers None to everything scores, and the metric's diagnostics would
    read that as a strict cutoff. It does not get that far: an empty array is
    not a face descriptor, so nothing binds and the refusal points at the step.
    """

    def test_a_store_offered_nothing_is_not_a_store_that_worked(self):
        # The acceptance test used to answer "enrolled two people" after making
        # no calls at all, so every store passed the first pass and the student
        # was told their query answered "no one".
        from facial_recognition_benchmark.roles import _attempt, recognition_fixture

        fixture = recognition_fixture(FIRST)
        nothing = [[] for _ in fixture.photos]

        passed, detail = _attempt(nothing, fixture, lambda *a: None, None)

        self.assertFalse(passed)
        self.assertIn("found no face", detail)

    def test_the_refusal_points_at_the_descriptors_step(self):
        found = self.resolve(A_STEP_THAT_FINDS_NO_FACE)

        self.assertFalse(found.ready)
        self.assertEqual(found.verdict.status, "not_wired")
        self.assertIn("descriptors", found.verdict.headline)

    def test_a_photo_their_step_finds_nothing_in_is_answered_not_dropped(self):
        # The other half of the same case: a step that works on the fixture
        # and finds nothing in one scored photo still owes an answer for it.
        found = self.resolve(NORMAL)
        adapter = adapter_for(found)
        adapter.enroll("ada", [photo(1)])

        self.assertEqual(adapter.recognize([photo(1), photo(), photo(1)]),
                         ["ada", None, "ada"])


if __name__ == "__main__":
    unittest.main()
