"""A discovered chain is run the way the search proved it works.

The search binds `adj_list(paths, threshold)` by offering a cutoff, and binds
`whispers(nodes, adj, iterations)` by spreading the pair `adj_list` returned.
The acceptance test and the scored run must make those same calls, or they
score a different program than the one that passed. Measured on one 2026
repository before this existed: the acceptance test called `adj_list(paths)`,
got a TypeError, and reported that their code ran and answered wrongly.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import ModuleType

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "python" / "cogbench" / "src"))
sys.path.insert(0, str(ROOT / "benchmarks" / "week2"))

from cogbench.pipeline import Candidate  # noqa: E402

from facial_recognition_benchmark.roles import _run  # noqa: E402


def _module(name, source):
    module = ModuleType(name)
    exec(compile(source, name, "exec"), module.__dict__)
    return module


THEIRS = _module(
    "theirs",
    "class Node:\n"
    "    def __init__(self, image_path, label):\n"
    "        self.image_path = image_path\n"
    "        self.label = label\n"
    "def adj_list(paths, threshold):\n"
    "    nodes = [Node(p, i) for i, p in enumerate(paths)]\n"
    "    return nodes, {n: [] for n in nodes}\n"
    "def whispers(nodes, adj, iterations):\n"
    "    return [len(adj)] * iterations\n"
    "def connected_comps(adj, nodes):\n"
    "    return [[n] for n in nodes]\n",
)


class DiscoveredChainTests(unittest.TestCase):
    def test_a_step_bound_with_a_tuning_is_called_with_it(self):
        chain = [Candidate("theirs.adj_list", THEIRS.adj_list, "theirs", tuning=0.3)]

        nodes, adj = _run(chain, ["a.png", "b.png"])

        self.assertEqual([n.image_path for n in nodes], ["a.png", "b.png"])
        self.assertEqual(len(adj), 2)

    def test_a_step_bound_without_a_tuning_raises_as_it_did_before(self):
        chain = [Candidate("theirs.adj_list", THEIRS.adj_list, "theirs")]

        with self.assertRaises(TypeError):
            _run(chain, ["a.png"])

    def test_a_returned_pair_is_spread_into_the_next_step(self):
        chain = [
            Candidate("theirs.adj_list", THEIRS.adj_list, "theirs", tuning=0.3),
            Candidate("theirs.whispers", THEIRS.whispers, "theirs", tuning=10),
        ]

        self.assertEqual(_run(chain, ["a.png", "b.png"]), [2] * 10)

    def test_a_returned_pair_is_spread_reversed_when_that_is_their_order(self):
        chain = [
            Candidate("theirs.adj_list", THEIRS.adj_list, "theirs", tuning=0.3),
            Candidate("theirs.connected_comps", THEIRS.connected_comps, "theirs"),
        ]

        groups = _run(chain, ["a.png", "b.png"])

        self.assertEqual([[n.image_path for n in g] for g in groups], [["a.png"], ["b.png"]])


if __name__ == "__main__":
    unittest.main()


class TheScoredRunPresentsTheBoundForm(unittest.TestCase):
    """A chain bound on paths is scored on paths; one bound on arrays, on
    arrays. The search proved one of the two, and scoring the other scores a
    function that never ran."""

    def _images(self):
        import numpy as np

        return [np.zeros((4, 4, 3), dtype=np.uint8) for _ in range(3)]

    def test_a_path_taking_first_step_is_given_paths(self):
        from facial_recognition_benchmark.discovered import DiscoveredClustering

        seen = []

        def takes_paths(paths):
            seen.append([type(p).__name__ for p in paths])
            return [str(p) for p in paths]

        takes_paths.__module__ = "theirs"
        chain = [Candidate("theirs.takes_paths", takes_paths, "theirs", form=1)]

        labels = DiscoveredClustering(chain).cluster(self._images())

        self.assertEqual(len(labels), 3)
        self.assertEqual(seen, [["PosixPath"] * 3])

    def test_an_array_taking_first_step_is_given_arrays(self):
        from facial_recognition_benchmark.discovered import DiscoveredClustering

        seen = []

        def takes_arrays(images):
            seen.append([type(i).__name__ for i in images])
            return list(range(len(images)))

        takes_arrays.__module__ = "theirs"
        chain = [Candidate("theirs.takes_arrays", takes_arrays, "theirs", form=0)]

        DiscoveredClustering(chain).cluster(self._images())

        self.assertEqual(seen, [["ndarray"] * 3])


class GroupsBecomeLabelsInPhotoOrder(unittest.TestCase):
    """The driver wants one label per photo; two audited teams return groups
    of nodes. Reading them back is placement, not repair."""

    def test_groups_are_placed_by_the_photo_each_node_names(self):
        from facial_recognition_benchmark.roles import labels_in_photo_order

        class N:
            def __init__(self, p):
                self.image_path = p

        photos = ["a.png", "b.png", "c.png", "d.png"]
        answer = [[N("c.png"), N("a.png")], [N("d.png"), N("b.png")]]

        labels = labels_in_photo_order(answer, photos)

        self.assertEqual(len(labels), 4)
        self.assertEqual(labels[0], labels[2])
        self.assertEqual(labels[1], labels[3])
        self.assertNotEqual(labels[0], labels[1])

    def test_a_dropped_photo_gets_a_label_of_its_own(self):
        from facial_recognition_benchmark.roles import labels_in_photo_order

        class N:
            def __init__(self, p):
                self.image_path = p

        labels = labels_in_photo_order([[N("a.png"), N("b.png")]], ["a.png", "b.png", "c.png"])

        self.assertEqual(labels[0], labels[1])
        self.assertNotEqual(labels[2], labels[0])

    def test_a_positional_list_passes_through(self):
        from facial_recognition_benchmark.roles import labels_in_photo_order

        self.assertEqual(labels_in_photo_order([0, 0, 1], ["a", "b", "c"]), [0, 0, 1])

    def test_groups_that_name_no_photo_are_refused(self):
        from facial_recognition_benchmark.roles import labels_in_photo_order

        with self.assertRaises(TypeError):
            labels_in_photo_order([[object()], [object()]], ["a", "b"])

    def test_the_scored_run_returns_one_label_per_photo(self):
        from facial_recognition_benchmark.discovered import DiscoveredClustering

        def groups(paths):
            return [[THEIRS.Node(p, 0) for p in paths[:2]], [THEIRS.Node(p, 1) for p in paths[2:]]]

        groups.__module__ = "theirs"
        chain = [Candidate("theirs.groups", groups, "theirs", form=1)]
        import numpy as np

        labels = DiscoveredClustering(chain).cluster([np.zeros((4, 4, 3), dtype=np.uint8) for _ in range(4)])

        self.assertEqual(len(labels), 4)
        self.assertEqual(labels[0], labels[1])
        self.assertEqual(labels[2], labels[3])
        self.assertNotEqual(labels[0], labels[2])


class ACourseStyleNodeNamesItsPhotoByIndex(unittest.TestCase):
    def test_integer_ids_are_read_as_positions(self):
        from facial_recognition_benchmark.roles import labels_in_photo_order

        class N:
            def __init__(self, i):
                self.id = i
                self.file_path = None

        labels = labels_in_photo_order([[N(0), N(2)], [N(1)]], ["a.png", "b.png", "c.png"])

        self.assertEqual(labels[0], labels[2])
        self.assertNotEqual(labels[0], labels[1])


class TheScoredRunIsSeeded(unittest.TestCase):
    def test_the_same_seed_gives_the_same_labels(self):
        import random

        from facial_recognition_benchmark.discovered import DiscoveredClustering

        def draws(paths):
            return [random.randrange(1000) for _ in paths]

        draws.__module__ = "theirs"
        chain = [Candidate("theirs.draws", draws, "theirs", form=0)]
        images = [b"x", b"y", b"z"]

        first = DiscoveredClustering(chain).cluster(images, seed=7)
        second = DiscoveredClustering(chain).cluster(images, seed=7)
        other = DiscoveredClustering(chain).cluster(images, seed=8)

        self.assertEqual(first, second)
        self.assertNotEqual(first, other)


class TheWeekReadsTheAnswerTheirOwnFunctionsGiveBack(unittest.TestCase):
    """Week 2 can be answered in more shapes than a list of labels.

    Each shape below is one the 2026 corpus wrote. They are checked against
    the week's own readers rather than against a repository, because what is
    being tested is the week's description of an answer, not any team's code.
    """

    def test_a_mapping_of_groups_is_an_answer(self):
        """`sorted_images()` returns {photo: [photo, ...]}: the key names the
        group and the value lists its members, both spelled with the names the
        benchmark handed their constructor."""

        from facial_recognition_benchmark.roles import (
            labels_in_photo_order,
            looks_like_labels,
            looks_like_labels_for,
        )

        answer = {"a.png": ["a.png", "b.png"], "c.png": ["c.png", "d.png"]}

        self.assertTrue(looks_like_labels(answer))
        self.assertTrue(looks_like_labels_for(4)(answer))
        self.assertFalse(looks_like_labels_for(3)(answer))

        labels = labels_in_photo_order(answer, ["a.png", "b.png", "c.png", "d.png"])

        self.assertEqual(labels[0], labels[1])
        self.assertEqual(labels[2], labels[3])
        self.assertNotEqual(labels[0], labels[2])

    def test_a_group_of_photo_names_needs_no_node_to_read(self):
        from facial_recognition_benchmark.roles import _as_grouping

        got = _as_grouping([["a.png", "b.png"], ["c.png"]], 3)

        self.assertEqual(
            got, frozenset({frozenset({"a.png", "b.png"}), frozenset({"c.png"})})
        )

    def test_an_empty_mapping_is_not_an_answer(self):
        from facial_recognition_benchmark.roles import looks_like_labels

        self.assertFalse(looks_like_labels({}))


class TheirGraphMayBeAnObjectRatherThanAPair(unittest.TestCase):
    """One 2026 team keeps the nodes and the adjacency on a `Whispers`
    instance, so the value their graph step produces is an object. The week
    has to recognise that without recognising every object."""

    class _Graph:
        def __init__(self, vectors, nodes=()):
            self.vectors = list(vectors)
            self.nodes = list(nodes)

    class _Node:
        def __init__(self, index):
            self.id = index
            self.label = index

    def _vectors(self, count):
        import numpy as np

        return [np.zeros(4, dtype=np.float32) for _ in range(count)]

    def test_an_object_holding_a_descriptor_per_photo_is_a_graph(self):
        from facial_recognition_benchmark.roles import looks_like_graph_for

        built = self._Graph(self._vectors(3))

        self.assertTrue(looks_like_graph_for(3)(built))
        self.assertFalse(looks_like_graph_for(4)(built))

    def test_an_object_holding_the_photos_themselves_is_not(self):
        """The search offers each of their classes the photos as well, so
        `Node(photos)` keeps twelve of something too. The photos are unsigned
        bytes and a descriptor is a float vector."""

        import numpy as np

        from facial_recognition_benchmark.roles import looks_like_graph_for

        photos = [np.zeros((4, 4, 3), dtype=np.uint8) for _ in range(3)]

        self.assertFalse(looks_like_graph_for(3)(self._Graph(photos)))

    def test_a_graph_is_finished_only_once_it_holds_the_nodes(self):
        """`Whispers(vectors, names, threshold)` keeps the descriptors from
        the moment it is built and makes its nodes two calls later. Without
        the difference the search read the empty object as a finished graph
        and ran whispers over no nodes at all."""

        from facial_recognition_benchmark.roles import looks_like_graph_for

        empty = self._Graph(self._vectors(3))
        filled = self._Graph(self._vectors(3), [self._Node(i) for i in range(3)])

        self.assertFalse(looks_like_graph_for(3, finished=True)(empty))
        self.assertTrue(looks_like_graph_for(3, finished=True)(filled))

    def test_a_plain_value_is_never_a_graph(self):
        from facial_recognition_benchmark.roles import looks_like_graph

        for value in (3, 3.5, "graph", b"graph", None, ()):
            self.assertFalse(looks_like_graph(value), value)


class OnePassOfWhispersIsNotWhispers(unittest.TestCase):
    """Whispers picks its next node at random and stops when the labels stop
    changing, so a chain that has run it answers the same way twice. One 2026
    team wrote both `whispers_sweep()` and `train_sweeps()`; both leave a
    grouping the metric can read, and without this the one the search happened
    to sort first was the one that bound."""

    @staticmethod
    def _photos(count):
        import numpy as np

        return [np.zeros((4, 4, 3), dtype=np.uint8) for _ in range(count)]

    def test_a_settled_answer_is_accepted(self):
        from facial_recognition_benchmark.roles import accepts

        def settled(photos):
            return [0, 0, 1, 1]

        settled.__module__ = "theirs"
        chain = [Candidate("theirs.settled", settled, "theirs")]

        passed, detail = accepts(chain, self._photos(4), [0, 0, 1, 1])

        self.assertTrue(passed, detail)

    def test_an_answer_that_moves_with_the_visit_order_is_refused(self):
        import random

        from facial_recognition_benchmark.roles import accepts

        def one_pass(photos):
            # Two different groupings of the same four photos, chosen by the
            # visit order rather than by the faces.
            return [0, 0, 1, 1] if random.random() < 0.5 else [0, 1, 0, 1]

        one_pass.__module__ = "theirs"
        chain = [Candidate("theirs.one_pass", one_pass, "theirs")]

        passed, detail = accepts(chain, self._photos(4), [0, 0, 1, 1])

        self.assertFalse(passed)
        self.assertIn("second time", detail)


def _written(name, source):
    """A module whose functions look like a team's, for the search to try.

    Written to a real file rather than exec'd in place, because the search
    only considers callables whose ``__module__`` is the module it is reading.
    """

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


class _AWeek2Search(unittest.TestCase):
    """The week's own role, run over a handful of made-up photos.

    Everything below is a shape the 2026 corpus wrote, checked against a
    module written here rather than against a repository, so a test says what
    the week needs rather than what one team happens to contain.
    """

    #: Three photos of one person and three of another, as far as the fake
    #: descriptors below are concerned.
    GROUPS = (0, 0, 0, 1, 1, 1)

    def photos(self):
        import numpy as np

        return [
            np.full((4, 4, 3), 10 + 100 * side, dtype=np.uint8) for side in self.GROUPS
        ]

    def search(self, module, *, paths=False):
        from cogbench.pipeline import Fixtures, resolve_chain

        from facial_recognition_benchmark.roles import (
            accepts,
            cluster_role_for,
            write_photos,
        )

        images = self.photos()
        expected = list(self.GROUPS)
        forms = Fixtures(((images,), (write_photos(images),)))
        role = cluster_role_for(len(images))
        detail = {}

        def verify(chain):
            passed, said = accepts(chain, forms.for_chain(chain)[0], expected)
            detail.setdefault(tuple(s.label for s in chain), said)
            return passed

        binding, refusal = resolve_chain(role, [module], forms, verify=verify)
        return binding, refusal, detail


ONE_PART_OF_WHAT_IT_RETURNED = '''
import numpy as np

def describe(photo):
    """Boxes, how sure it was, and the descriptors: their shape exactly."""
    one = np.asarray(photo, dtype=np.float32)
    if one.ndim != 3:
        raise ValueError("one photo at a time")
    # Boxes and confidences as whole numbers, so the only part of this
    # that looks like a descriptor is the part that is one.
    return (
        np.zeros((1, 4), dtype=np.int32),
        np.ones(1, dtype=np.int32),
        np.stack([np.repeat(one.ravel()[:1] / 100.0, 8)]),
    )

class _Node:
    def __init__(self, index):
        self.id = index
        self.label = index

def group(descriptors):
    """Their graph reader and their answer in one, which two of the audited
    teams also wrote: the groups, made of their own node objects."""
    groups = {}
    for index, row in enumerate(descriptors):
        # One face per photo, flattened the way their own graph does it.
        row = np.asarray(row, dtype=np.float32).ravel()
        if row.size != 8:
            raise ValueError("a descriptor is eight numbers")
        groups.setdefault(round(float(row[0]), 3), []).append(_Node(index))
    return [groups[key] for key in sorted(groups)]
'''


class OnePartOfWhatTheirFunctionReturned(_AWeek2Search):
    """Bagel's `file_descriptors(path)` returns (boxes, probabilities,
    descriptors) for one photo. Gathering element 2 across photos is the same
    reading the search already does for a single return value; without it the
    descriptors step was handed a list of triples and refused."""

    def test_the_descriptors_are_taken_out_of_each_photos_result(self):
        binding, refusal, _detail = self.search(
            _written("theirs", ONE_PART_OF_WHAT_IT_RETURNED)
        )

        self.assertIsNone(refusal, refusal.detail if refusal else "")
        first = binding.steps[0]
        self.assertEqual(first.label, "theirs.describe")
        self.assertTrue(first.per_item)
        self.assertEqual(first.element, 2)

    def test_the_scored_run_takes_the_same_part(self):
        from facial_recognition_benchmark.discovered import DiscoveredClustering

        binding, _refusal, _detail = self.search(
            _written("theirs", ONE_PART_OF_WHAT_IT_RETURNED)
        )

        labels = DiscoveredClustering(binding.steps).cluster(self.photos())

        self.assertEqual(len(labels), len(self.GROUPS))
        self.assertEqual(labels[0], labels[1])
        self.assertNotEqual(labels[0], labels[3])


A_CLASS_THAT_DEMANDS_ITS_DATA = '''
import numpy as np

def describe(photo):
    one = np.asarray(photo, dtype=np.float32)
    if one.ndim != 3:
        raise ValueError("one photo at a time")
    return np.repeat(one.ravel()[:1] / 100.0, 8)

class Graph:
    """Their graph: built from the descriptors, the photo each came from, and
    a cutoff they never gave a default."""

    def __init__(self, vectors, names, threshold):
        self.vectors = []
        for vector in vectors:
            row = np.asarray(vector, dtype=np.float32)
            if row.ndim != 1:
                raise ValueError("a descriptor is a vector")
            self.vectors.append(row)
        self.names = list(names)
        self.threshold = threshold
        self.nodes = []

    def create_nodes(self):
        self.nodes = [Node(i, self.vectors[i], self.names[i]) for i in range(len(self.vectors))]

    def settle(self):
        for node in self.nodes:
            for other in self.nodes:
                if abs(float(node.vector[0]) - float(other.vector[0])) <= self.threshold:
                    node.label = min(node.label, other.label)

    def sorted_images(self):
        groups = {}
        for node in self.nodes:
            groups.setdefault(self.nodes[node.label].name, []).append(node.name)
        return groups

class Node:
    def __init__(self, index, vector, name):
        self.id = index
        self.label = index
        self.vector = vector
        self.name = name
'''


class AClassThatDemandsItsDataIsTheGraphStep(_AWeek2Search):
    """Bagel's `Whispers(vectors, names, threshold)` is the graph step: the
    arguments are what the step takes, the instance is what it produces, and
    its methods are the steps after it. The middle argument is one name per
    descriptor, which is the input the benchmark passed one step earlier."""

    def _binding(self):
        binding, refusal, detail = self.search(
            _written("theirs", A_CLASS_THAT_DEMANDS_ITS_DATA)
        )
        self.assertIsNone(refusal, "\n".join(str(v) for v in detail.values()))
        return binding

    def test_the_constructor_is_a_step_and_its_methods_are_the_next_ones(self):
        binding = self._binding()
        labels = [step.label for step in binding.steps]

        self.assertEqual(labels[0], "theirs.describe")
        self.assertEqual(labels[1], "theirs.Graph")
        self.assertIn("theirs.Graph.create_nodes", labels)
        self.assertIn("theirs.Graph.settle", labels)
        self.assertEqual(labels[-1], "theirs.Graph.sorted_images")

    def test_the_photo_each_descriptor_came_from_fills_the_name_argument(self):
        binding = self._binding()
        built = binding.steps[1]

        self.assertIn("identity", built.plan)
        self.assertIn("tuning", built.plan)
        self.assertIn("identity", built.supplied)

    def test_a_scored_run_names_its_own_photos_not_the_search_s(self):
        """The names go onto their nodes and come back out of
        `sorted_images`, so a run that replayed the search's names produced
        groups of photos this run had never seen."""

        from facial_recognition_benchmark.discovered import DiscoveredClustering

        binding = self._binding()

        labels = DiscoveredClustering(binding.steps).cluster(self.photos())

        self.assertEqual(len(labels), len(self.GROUPS))
        self.assertEqual(labels[0], labels[1])
        self.assertNotEqual(labels[0], labels[3])


A_PIPELINE_OVER_A_FOLDER = '''
import numpy as np
from pathlib import Path

class Album:
    """Their pipeline reads a folder rather than taking arguments."""

    def __init__(self):
        self.nodes = []
        for index, path in enumerate(sorted(Path("baseImages").iterdir())):
            self.nodes.append(Node(index, path))

    def group_nodes(self):
        for node in self.nodes:
            for other in self.nodes:
                if node.tone == other.tone:
                    node.label = min(node.label, other.label)

    def sorted_images(self):
        groups = {}
        for node in self.nodes:
            groups.setdefault(self.nodes[node.label].name, []).append(node.name)
        return groups

class Node:
    def __init__(self, index, path):
        import imageio.v3 as iio

        self.id = index
        self.label = index
        self.name = str(path)
        self.tone = int(np.asarray(iio.imread(path)).ravel()[0])
'''


class TheirPipelineReadsAFolder(_AWeek2Search):
    """CoggurtFilter's `clusterCreator()` takes no arguments and describes
    every photo in a directory. That is a different interface to the same
    work, and the honest answer is to give them a folder of the benchmark's
    own photos."""

    def test_a_zero_argument_reader_is_handed_the_benchmarks_photos(self):
        binding, refusal, detail = self.search(_written("theirs", A_PIPELINE_OVER_A_FOLDER))

        self.assertIsNone(refusal, "\n".join(str(v) for v in detail.values()))
        self.assertEqual(binding.steps[0].label, "theirs.Album")
        self.assertEqual(binding.steps[0].supplied.get("folder"), "baseImages")
        self.assertTrue(binding.steps[0].self_only)
