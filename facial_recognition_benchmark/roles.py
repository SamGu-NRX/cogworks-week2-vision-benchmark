"""What Week 2 asks for, described so a repository can be searched for it.

The capstone is two tasks over the same faces. Clustering: given a folder of
photos, put the ones showing the same person together. Recognition: be told
who someone is, then answer who is in a photo, including when the answer is
nobody you know. The course names the steps on the way to both. Find the faces
in a picture, turn each into a descriptor, measure how close two descriptors
are, and then either build a graph and run whispers over it, or file the
descriptor under a name and look one up. Every team writes those, in their own
files, under their own names.

So this says what each step does in terms of what goes in and what comes back,
never what it is called. Each task's acceptance test is the only thing that can
accept a binding: for clustering, hand it photos of two people and require that
it puts each person's photos together; for recognition, tell it about two
people and ask who is in a third photo.

Week 1 taught the shape of this file and one lesson worth repeating: the
validators are for cutting the search, not for judging. A team whose threshold
is badly tuned should be scored badly by the benchmark, not refused by
discovery.
"""

from __future__ import annotations

import atexit
import contextlib
import inspect
import io
import os
import shutil
import sys
import tempfile
from dataclasses import dataclass
from typing import Any, List, Optional, Sequence, Tuple

import numpy as np

from cogbench.pipeline import Role, Stage

__all__ = [
    "CLUSTER_ROLE",
    "cluster_role_for",
    "labels_in_photo_order",
    "accepts",
    "looks_like_labels",
    "looks_like_descriptors",
    "write_photos",
    "lay_out_folders",
    "DESCRIBE_ROLE",
    "RecognitionFixture",
    "descriptors_in",
    "enrollment_arrangements",
    "looks_like_an_empty_database",
    "looks_like_face_descriptors",
    "named",
    "readable",
    "recognition_accepts",
    "recognition_fixture",
]


def looks_like_descriptors(value: Any) -> bool:
    """One vector per face, or a batch of them.

    Deliberately loose about the dimension. The course uses FaceNet's 512, and
    a team who wrote their own encoder is answering the same question with a
    different number.
    """

    if isinstance(value, np.ndarray):
        return value.ndim in (1, 2) and value.size > 0 and value.dtype.kind == "f"
    if isinstance(value, (list, tuple)) and value:
        first = value[0]
        return isinstance(first, np.ndarray) and first.dtype.kind == "f"
    return False


def looks_like_labels(value: Any) -> bool:
    """An answer saying which photos go together.

    Three shapes, all of which the course teaches or the corpus wrote. One
    label per image, in image order, is the obvious one. Groups of nodes is
    the other: the capstone's `connected_components` "returns the groups"
    (week2-vision-capstone.md:392), and two of the three audited teams end
    there rather than flattening back to a list.

    The third is the same groups in a mapping. One 2026 team's
    `sorted_images()` returns `{photo: [photo, ...]}`, keyed and valued by
    the names the benchmark handed their constructor, which is a complete
    answer to "which photos go together" written the way a person would want
    to read it. Refusing it cost that team every one of their own readers.

    A label is whatever a team used to mean "these two are the same person": a
    number, a name, a node id. What matters is that the answer can be turned
    into a grouping, which is all the metric reads.
    """

    if isinstance(value, np.ndarray):
        return value.ndim == 1 and value.size > 0
    if isinstance(value, dict):
        return bool(value) and all(_is_group(group) for group in value.values())
    if not isinstance(value, (list, tuple)) or not value:
        return False
    if all(isinstance(item, (int, str, np.integer)) for item in value):
        return True
    # Groups: a list of lists, each holding whatever their node class is.
    return all(_is_group(item) for item in value)


def _is_group(item: Any) -> bool:
    return isinstance(item, (list, tuple, set)) and len(item) > 0


def looks_like_labels_for(count: int):
    """`looks_like_labels`, and there must be one per photo.

    The plain predicate accepts any non-empty list of ints, and one 2026
    team's `whispers` returns a list of ints: how many components there
    were after each pass. Ten ints for twelve photos read as the answer,
    the chain ended there, and their `connected_comps`, which reads the
    real labels off the graph, was never called. The search knows how many
    photos it handed over, so the answer must have that many entries, or
    groups whose sizes add up to it.
    """

    def _labels(value: Any) -> bool:
        if not looks_like_labels(value):
            return False
        if isinstance(value, np.ndarray):
            return value.size == count
        groups = value.values() if isinstance(value, dict) else value
        if not isinstance(value, dict) and all(
            isinstance(item, (int, str, np.integer)) for item in value
        ):
            return len(value) == count
        return sum(len(group) for group in groups) == count

    return _labels


def looks_like_graph_for(count: int, *, finished: bool = False):
    """`looks_like_graph`, and an object of theirs must hold one per photo.

    ``finished`` asks for the nodes rather than the descriptors, which is the
    difference between a graph object that has been made and one that has
    been filled in. Bagel's `Whispers(vectors, names, threshold)` keeps the
    descriptors from the moment it is built and makes its nodes two calls
    later; without the distinction the search read the empty object as a
    finished graph, carried it past both of those calls, and ran whispers
    over no nodes at all.

    Only the object form is sized. A pair or a list of nodes is the shape the
    course teaches and a team whose graph drops a photo with no face in it
    still built a graph; the count is applied where the test would otherwise
    accept anything at all, which is a class of theirs the search happened to
    construct out of a cutoff.

    Measured on week 2's Lashika repository the moment the object form was
    added: `Profile(0.3)` and `Node(0.3)` both passed, five chains that build
    one and throw it away entered the frontier ahead of their own
    `adj_list`, and a repository that had been binding for weeks was refused.
    """

    def _graph(value: Any) -> bool:
        if not looks_like_graph(value):
            return False
        if isinstance(value, (list, tuple, dict, np.ndarray)):
            return True
        return _holds(value, count, finished=finished)

    return _graph


def _holds(built: Any, count: int, *, finished: bool = False) -> bool:
    """Whether one of their objects is keeping a face per photo.

    Read off whatever their constructor stored, because that is the only
    thing a graph object has to have: it was given one face per photo and it
    kept them somewhere, either as the descriptors themselves or as the
    nodes it made of them. Bagel's `Whispers` keeps both (`vectors` and,
    once `create_nodes` has run, `nodes`).

    Counting anything of the right length was not enough. The search also
    offers each of their classes the photos themselves, since a team whose
    graph builder reads the photos is a shape the week allows, so
    `Node(photos)` and `Profile(photos)` both kept twelve of something and
    both passed. The photos are unsigned bytes and a descriptor is a float
    vector, which is the difference between the input and a face.
    """

    for value in getattr(built, "__dict__", {}).values():
        try:
            if len(value) != count:
                continue
        except TypeError:
            continue
        if looks_like_graph(value):
            return True
        if not finished and looks_like_descriptors(value):
            return True
    return False


def cluster_role_for(count: int) -> Role:
    """`CLUSTER_ROLE` whose last stage insists on one label per photo."""

    from dataclasses import replace as _replace

    def _sized(stage):
        # Both stages that can end the chain. Sizing only the last one left
        # `settle` accepting a list of component counts as its output, which
        # took a beam slot from the in-place chain that reads the real labels.
        if stage.name in ("settle", "labels"):
            return _replace(stage, produces=looks_like_labels_for(count))
        if stage.name == "graph":
            return _replace(stage, produces=looks_like_graph_for(count))
        if stage.name in ("edges", "nodes"):
            return _replace(
                stage, produces=looks_like_graph_for(count, finished=True)
            )
        return stage

    return Role(CLUSTER_ROLE.name, tuple(_sized(stage) for stage in CLUSTER_ROLE.stages))


#: Images in, one label per image out.
#:
#: Every step between is marked fusible, because teams divide this work
#: differently and all the divisions are correct. One 2026 team wrote
#: `whispers(images, iterations)` doing the whole thing; another wrote
#: `get_descriptor`, `adj_list`, and `whispers` separately; a third put
#: detection and description in one `process_image`. Insisting on any one
#: division would refuse working code over how it was organized.
#: Everything Python and numpy already had a name for. Anything else that
#: comes back from one of their functions is an object one of their classes
#: made, which is the only test `looks_like_graph` can apply to a design it
#: is not allowed to require.
_PLAIN = (
    bool,
    int,
    float,
    complex,
    str,
    bytes,
    bytearray,
    list,
    tuple,
    dict,
    set,
    frozenset,
    type(None),
    np.ndarray,
    np.generic,
    os.PathLike,
)


def looks_like_graph(value: Any) -> bool:
    """Nodes, or nodes and their adjacency, or the object holding both.

    The course's own design is a graph: "a list of nodes and an adjacency
    graph that describes the weighted connections between your nodes"
    (docs/capstones/week2-vision-capstone.md:386). Two of three audited teams
    return exactly that, so it is a step, not an implementation detail.

    A fourth shape is one object. One 2026 team wrote `Whispers(vectors,
    names, threshold)` and keeps the nodes and the adjacency on it, so the
    thing their graph step produces is an instance rather than a pair.
    Refusing it refused their whole pipeline, because every later step is a
    method of that object.

    Deliberately structural rather than typed: a node is whatever class they
    wrote, and requiring a shape would be requiring their design.
    """

    if isinstance(value, tuple) and len(value) == 2:
        return looks_like_graph(value[0])
    if isinstance(value, dict):
        return bool(value)
    if not isinstance(value, (list, tuple)):
        # An object of a class they wrote, holding whatever their graph is.
        return not isinstance(value, _PLAIN) and not callable(value)
    if not value:
        return False
    first = value[0]
    # Not a descriptor and not a label: something they built.
    return not isinstance(first, (int, str, float, np.integer, np.ndarray))


CLUSTER_ROLE = Role(
    "cluster",
    (
        Stage(
            "descriptors",
            prefers=("descriptor", "embed", "encode", "facenet", "process"),
            produces=looks_like_descriptors,
            arity=1,
            tunings=(0.3, 0.4, 0.5, 0.6, 0.7),
            # A team whose graph builder reads the photos itself never had a
            # separate descriptor step. One audited repository's
            # `adj_list(image_paths, threshold)` describes every face and
            # builds the graph in one function, which is the same work in one
            # place rather than two.
            fusible=True,
            # The capstone hands one photo at a time
            # (docs/capstones/week2-vision-capstone.md:176), and every audited
            # team wrote a per-photo descriptor function.
            per_item=True,
            # One 2026 team wrote the whole thing over a directory: their
            # class takes no arguments, reads a folder of photos, and
            # describes every one of them. That is not a missing function,
            # it is a different interface to the same work, and the photos
            # the benchmark hands over are a folder as readily as they are a
            # list. Asked for here and at the graph step and nowhere after,
            # because a folder is what the benchmark's own input can be
            # turned into and nothing further down the chain is one.
            folder=True,
        ),
        Stage(
            "graph",
            prefers=("adj", "node", "graph", "build", "connect"),
            produces=looks_like_graph,
            # A team who went straight from descriptors to labels never built
            # one, and that is a complete answer too.
            fusible=True,
            tunings=(0.3, 0.4, 0.5, 0.6, 0.7),
            # One 2026 team's graph is `Whispers(vectors, names, threshold)`,
            # where `names` is one label per descriptor so their nodes can say
            # which photo they came from. The benchmark knows that: it is the
            # input it passed one step ago. Handing it back is input, and it
            # is the only way their constructor can be called at all.
            identity=True,
            # The folder-reading team's constructor does not hand back
            # descriptors, it hands back the graph it built out of them, so
            # the offer has to be open here as well as at the step before.
            # Week 2's CoggurtFilter is the case: `clusterCreator()` reads a
            # directory, describes every photo in it, and keeps the nodes.
            folder=True,
        ),
        Stage(
            "edges",
            prefers=("adj", "matrix", "edge", "connect", "dist", "neighbor"),
            produces=looks_like_graph,
            # Almost every team built the whole graph in the step above; this
            # stage is skipped for them and costs nothing. It exists because
            # one 2026 team wrote the graph as a class and finishes it with
            # one method per piece: `create_matrix()` works out which faces
            # are close enough to connect and `create_nodes()` makes the
            # nodes, both returning nothing and writing on the object. A role
            # with one graph step could express the constructor or those
            # methods, never both, so their pipeline could not be reached.
            fusible=True,
            in_place=True,
            tunings=(0.3, 0.4, 0.5, 0.6, 0.7),
        ),
        Stage(
            "nodes",
            prefers=("node", "vertex", "build", "create"),
            produces=looks_like_graph,
            # The other piece of the same object; see `edges`. Both stages
            # are fusible and in place, so the search tries their methods in
            # either order and only the order that settles the graph survives
            # the acceptance test.
            fusible=True,
            in_place=True,
            tunings=(0.3, 0.4, 0.5, 0.6, 0.7),
        ),
        Stage(
            "settle",
            prefers=("whisper", "propagate", "cluster", "label"),
            produces=looks_like_labels,
            fusible=True,
            # Whispers is iterative and the course does not fix a count, so
            # their `whispers(nodes, adj, iterations)` takes one with no
            # default. The cutoffs are here too, for a team who clusters
            # straight from descriptors and takes the threshold at this step.
            #
            # Enough passes to settle on this fixture. The benchmark's own
            # driver decides how long a scored run gets; this only has to be
            # long enough to tell a working chain from a broken one.
            # Largest pass count first. What binds is what the scored run
            # uses, and on one 2026 repository the same chain scored 0.67
            # at 10 passes against 0.91 at 20 or more; the instructor's
            # hand-written adapter runs their loop for len(nodes) * 60. 200
            # passes settle a 12-photo fixture and take well under the
            # per-call clock.
            tunings=(200, 20, 10, 0.3, 0.4, 0.5, 0.6, 0.7),
            # The course says `propagate_label` "should update that node's
            # label" and has `whispers` record how the component count
            # changes as it runs (week2-vision-capstone.md:390-392). One
            # audited team's whispers does exactly that: it returns the
            # counts, and the labels are on the graph it was handed.
            in_place=True,
        ),
        Stage(
            "labels",
            prefers=("component", "group", "cluster", "label"),
            produces=looks_like_labels,
            # A team whose `whispers` returns the labels has already
            # answered; this stage is for one whose `whispers` left them on
            # the graph and whose `connected_comps` reads them back. Before
            # this stage existed the in-place step was also the last one, so
            # the graph it forwarded reached nothing and the chain was
            # refused as "does not say which photos go together".
            fusible=True,
        ),
    ),
)


def accepts(chain, images, expected):
    """Cluster photos of two people and require the two groups back.

    The weakest test that still kills a wrong binding. It does not measure
    quality: a team whose threshold splits one person into two clusters should
    lose points on the benchmark, not be refused here. What it rejects is a
    chain that returns the same label for everyone, or a different label for
    every photo, both of which a wrongly assembled pipeline produces.

    That is what it now does. It used to require the exact partition, which
    read as the same rule and is a much stronger one: it refused a pipeline
    that grouped the photos nearly right, and refused it with a wiring
    message. A grouping that is neither degenerate is an answer to this
    question, and how good an answer belongs to the metric.

    One more thing it rejects, for the same reason: a chain whose answer moves
    when nothing but the random visit order does. Whispers picks its next node
    at random and stops when the labels stop changing, so a chain that has run
    it answers the same way twice. One that has not is one turn of the loop
    rather than the loop.

    That distinction is not quality and cannot be reached by any other means
    the week has. One 2026 repository wrote both: `whispers_sweep()` makes one
    pass and `train_sweeps()` repeats it. Both are theirs, both leave a
    grouping the metric can read, and whichever the search happens to try
    first is the one that binds -- so without this the score was 0.56 with a
    0.35 spread across seeds where their own loop scores 0.91 with none, and
    which of the two it was came down to the order the candidates were sorted
    in.

    Returns ``(passed, detail)``.
    """

    import random as _random

    with _fresh_state():
        lay_out_folders(chain, images)
        try:
            # Two draws of the visit order, not two different inputs. The
            # seeds are fixed so this and a second cold resolve make exactly
            # the same calls.
            _random.seed(0)
            labels = _run(chain, images)
            _random.seed(1)
            again = _run(chain, images)
        except BaseException as error:  # noqa: BLE001 - student code raises anything
            return False, "clustering raised {}: {}".format(
                type(error).__name__, str(error)[:120]
            )

    got = _as_grouping(labels, len(images))
    if got is None:
        return False, "returned {} that does not say which photos go together".format(
            type(labels).__name__
        )
    if got != _as_grouping(again, len(images)):
        return False, (
            "grouped the same photos differently the second time, so this step is "
            "one pass of whispers rather than whispers running until the labels "
            "stop changing"
        )
    want = _grouping(expected)
    if got == want:
        return True, "grouped {} photos into {} people".format(len(images), len(want))

    # Not an exact match, which is not the same as not an answer. The
    # docstring above promised to reject a chain that says everyone is the
    # same person or that everyone is different, and to leave quality to the
    # benchmark. Requiring the exact partition broke that promise: it also
    # refused a chain that grouped the photos nearly right, which is a tuning
    # result and the single most useful thing a team can be told.
    #
    # Measured on one 2026 repository. Its pipeline binds and runs, and at
    # every cutoff the search tries it returns 4, 5 or 6 groups where the
    # fixture has 3, because its threshold splits one person in two. Refusing
    # that reported "nothing took that for the labels step", which is false
    # and sends them to look for a function they already wrote.
    #
    # So the bar here is: did this chain answer the question at all. Two
    # degenerate answers are still refused, because both are what a wrongly
    # assembled pipeline produces rather than what a working one gets wrong.
    if len(got) <= 1:
        return False, "put all {} photos in one group".format(len(images))
    if len(got) >= len(images):
        return False, "put every photo in its own group"
    return True, "grouped {} photos into {} groups where there are {} people".format(
        len(images), len(got), len(want)
    )


def _as_grouping(answer: Any, count: int):
    """Read an answer into "which photos went together", or None.

    Two shapes. One label per photo, in photo order, is read positionally.
    Groups of nodes are read by asking each node which photo it is, which the
    course's own node class answers: it carries the file path of its image
    (week2-vision-capstone.md:418). Anything else is not an answer to this
    question and is refused rather than guessed at.
    """

    if isinstance(answer, np.ndarray):
        answer = answer.tolist()
    answer = _groups_of(answer)
    if not isinstance(answer, (list, tuple)) or not answer:
        return None

    if all(isinstance(item, (int, str, np.integer)) for item in answer):
        return _grouping(answer) if len(answer) == count else None

    groups = []
    for group in answer:
        if not isinstance(group, (list, tuple, set)):
            return None
        members = frozenset(_identifies(node) for node in group)
        if any(member is None for member in members):
            return None
        groups.append(members)
    if sum(len(group) for group in groups) != count:
        return None
    return frozenset(groups)


def labels_in_photo_order(answer: Any, photos: Sequence[Any]) -> List[Any]:
    """One label per photo, in the order the photos were handed over.

    The same two shapes `_as_grouping` reads. A positional list is returned
    as given. Groups of nodes are placed by asking each node which photo it
    stands for (`_identifies`), matched against the photo list by `str()`,
    which is how a path and a node's `image_path` compare. A photo their
    pipeline dropped (no face found, file unreadable) gets a label of its
    own, because the metric compares pairs and an unplaced photo is in no
    pair with anyone; that is what their code said about it. An answer that
    does not say which photos went together raises, and the driver records
    the contract error against the scenario.
    """

    if isinstance(answer, np.ndarray):
        answer = answer.tolist()
    answer = _groups_of(answer)
    if isinstance(answer, (list, tuple)) and answer and all(
        isinstance(item, (int, str, np.integer)) for item in answer
    ):
        return list(answer)
    keys = [str(photo) for photo in photos]
    placed: dict = {}
    for index, group in enumerate(answer if isinstance(answer, (list, tuple)) else ()):
        if not isinstance(group, (list, tuple, set)):
            raise TypeError("cluster() returned something that is not groups or labels")
        for node in group:
            who = _identifies(node)
            if who is None:
                continue
            # The course's own Node takes "a unique identifier for this
            # node ... a value in [0, N-1]" and the file path separately
            # (week2-vision-capstone.md:418). One audited team fills the
            # id and leaves file_path None, so their nodes name a photo by
            # its position. That is a positional answer, read as one.
            if who.isdigit() and int(who) < len(keys) and who not in placed:
                who = keys[int(who)]
            placed[who] = index
    if not placed:
        raise TypeError("cluster() returned groups whose members do not say which photo they are")
    labels: List[Any] = []
    next_label = len(answer)
    for key in keys:
        if key in placed:
            labels.append(placed[key])
        else:
            labels.append(next_label)
            next_label += 1
    return labels


def lay_out_folders(chain: Sequence[Any], photos: Sequence[Any]) -> Optional[str]:
    """Put the photos back under the folder name their code reads.

    A step bound through `Stage.folder` was proved by writing the benchmark's
    own photos into a directory named the way their code asked for it. Every
    run after that has to present the same directory or their constructor
    looks for a folder that is not there, and the report blames them for a
    file the benchmark did not put out.

    Writes into the current working directory and nowhere else, so the caller
    is responsible for being somewhere throwaway; `_fresh_state` and
    `DiscoveredClustering.cluster` both are. Returns the folder name, or None
    when no step of this chain reads one.
    """

    import shutil
    from pathlib import Path as _Path

    wanted = None
    for step in chain:
        name = getattr(step, "supplied", {}).get("folder")
        if name:
            wanted = name
    if not wanted:
        return None
    folder = _Path.cwd() / wanted
    folder.mkdir(parents=True, exist_ok=True)
    for photo in photos:
        source = _Path(str(photo))
        if source.is_file():
            shutil.copyfile(str(source), str(folder / source.name))
    return wanted


def _groups_of(answer: Any) -> Any:
    """A mapping of groups read as the groups it holds.

    One 2026 team's `sorted_images()` returns `{photo: [photo, ...]}`: the
    key names the group and the list is its members, both spelled with the
    names the benchmark handed their constructor. The keys carry nothing the
    values do not, so the answer is the values.
    """

    if isinstance(answer, dict):
        return list(answer.values())
    return answer


def _identifies(node: Any):
    """Which photo a node stands for, read off the node itself.

    The course's node carries "the file path of the image corresponding to
    this node" (week2-vision-capstone.md:438), so that is what is asked for.
    A node that says nothing about its photo cannot be placed, and the answer
    is refused rather than guessed at.

    A group whose members are the photo names themselves needs no reading at
    all: one 2026 team's `sorted_images()` groups the very strings the
    benchmark handed their constructor as the identity of each descriptor.
    """

    if isinstance(node, (str, os.PathLike, int, np.integer)):
        # A group whose members are the identities the benchmark supplied.
        # Those are the photo paths when the chain took paths and the photo's
        # position when it took arrays, which is what `identities_for`
        # decides; both are read back here as the name of a photo, and
        # `labels_in_photo_order` turns a positional one into its photo.
        return str(node)
    for attribute in ("image_path", "file_path", "path", "filepath", "image", "id", "ID"):
        value = getattr(node, attribute, None)
        if value is None:
            continue
        return str(value)
    return None


def _grouping(labels: Sequence[Any]) -> frozenset:
    """Which photos went together, independent of what the groups are called."""

    groups: dict = {}
    for index, label in enumerate(labels):
        groups.setdefault(label, []).append(index)
    return frozenset(frozenset(members) for members in groups.values())


# ---------------------------------------------------------------------------
# Recognition
# ---------------------------------------------------------------------------
#
# Recognition remembers, so the search has three things to find rather than
# one chain: a step that turns a photo into a face descriptor, a call that
# files one under a name, and a call that takes one and answers with a name.
# The resolver pairs the last two itself, the way it finds Week 1's song
# database; what is described here is the first.
#
# The descriptor step is the half both sides share, and running one chain for
# both removes one cause of a miss: their two halves cannot disagree about
# what a descriptor is, because both were handed the same vectors. And what
# their detector did is counted separately (`plugins._describing_notes`), so
# a photo it found no face in is named as that rather than counted as a miss.
# A miss on a photo where a face was found is still theirs to look for among
# the descriptor, the matcher and the cutoff; nothing here says which of the
# three.
#
# The cost, written down because the opposite is easy to claim: this does not
# reveal a team whose enrolment describes a face one way and whose query
# describes it another. It makes that unexecutable. Such a team scores clean
# here and drifts in their own application, and the benchmark will not say so.


#: A value narrower than this is one of the things the model returns beside
#: the descriptors, not a descriptor. `model.detect` gives back a box of four
#: numbers and one probability per face, and three of the five audited 2026
#: repositories hand those back in the same tuple as the descriptors
#: (CoggurtFilter `(boxes, descriptors)`, Asterisk `(boxes, descriptors,
#: probabilities)`, Bagel `(boxes, probs, descriptors)`). The search offers
#: each part of a returned tuple in order, so a plain "float array" test binds
#: the boxes and scores a whole run against them.
#:
#: The floor is not the model's own 512, because a team who reduces the vector
#: before storing it is answering the same question with a different number,
#: and refusing that refuses working code over a representation choice.
NARROWEST_DESCRIPTOR = 16


def looks_like_face_descriptors(value: Any) -> bool:
    """One face descriptor, or one per face in a photo.

    Clustering asks a looser question with `looks_like_descriptors`, and the
    difference is not the corpus, which is the same teams. It is how much each
    acceptance test can catch. Clustering grades a whole
    six-photo grouping, so a chain bound on the boxes answers wrongly and is
    refused. Recognition grades one name and one rejection against two people,
    which boxes can pass by luck, and passing stops the search.
    """

    if isinstance(value, np.ndarray):
        if value.dtype.kind != "f" or value.ndim not in (1, 2) or value.size == 0:
            return False
        return value.shape[-1] >= NARROWEST_DESCRIPTOR
    if isinstance(value, (list, tuple)) and value:
        return looks_like_face_descriptors(value[0])
    return False


DESCRIBE_ROLE = Role(
    "describe",
    (
        Stage(
            "descriptors",
            prefers=("descriptor", "describe", "embed", "encode", "facenet", "vector"),
            produces=looks_like_face_descriptors,
            # The capstone hands one photo at a time, and four of the five
            # audited teams wrote a per-photo function. A team who wrote one
            # that takes the whole list still binds: the search offers both.
            per_item=True,
            # The FaceNet model the course hands every team. Four of the five
            # audited repositories construct their own when their module
            # loads, but CoggurtFilter's `detect_and_describe(model, image)`
            # takes it as an argument, and no amount of searching their
            # repository produces one. It is the same object the benchmark
            # already gives a submission that declares itself
            # (`adapters.instantiate`), so offering it here restores parity
            # rather than supplying anything of ours.
            extras=("model",),
        ),
    ),
)
# No tuning slot, unlike the clustering role: every detection cutoff in the
# audited corpus has a default (`add_image(..., detection_threshold=0.87)`,
# `detecting_faces(pic, threshold=0.3)`), so nothing here has a required
# argument the search would have to guess. No folder step either; this task
# asks about one photo, and a directory is not an answer to it.


@dataclass(frozen=True)
class RecognitionFixture:
    """Two people to remember and two photos to ask about.

    ``enrollment`` maps a name to positions in ``photos``; ``query`` is the
    position of a held-out photo of ``query_of``; ``stranger`` is a photo of
    nobody who was enrolled.
    """

    photos: Sequence[Any]
    enrollment: Sequence[Tuple[str, Tuple[int, ...]]]
    query: int
    query_of: str
    stranger: int


#: The names the search enrolls. Deliberately not the scored case's names: a
#: database that survives from the search into a run survives holding these,
#: and `named` hands back only names the run itself enrolled, so an answer
#: built from them reads as unknown rather than as one of the case's people.
FIXTURE_NAMES = ("fixture_a", "fixture_b")


def recognition_fixture(scenario: Any) -> Optional[RecognitionFixture]:
    """Build the search's own small lifecycle out of a scored case.

    Six photos, and the size is the point: each one is described once per
    candidate the search tries, and describing one is a forward pass of
    FaceNet. Two people, because naming the right one out of one is not
    evidence. Two enrollment photos each, because a profile built from a
    single photo can fall outside a team's own cutoff, and refusing them for
    that would refuse working code. One stranger, because saying "I do not
    know this person" is half of what this benchmark measures.
    """

    known = list(scenario.known)[:2]
    strangers = list(scenario.unknown_queries)
    # A case too small to ask the question with. `discovery()` reports that as
    # having nothing to search for, the same as an uncached dataset does.
    if (
        len(known) < 2
        or not strangers
        or not known[0].queries
        or any(len(person.enrollment) < 2 for person in known)
    ):
        return None

    photos: List[Any] = []
    enrollment = []
    for name, person in zip(FIXTURE_NAMES, known):
        start = len(photos)
        photos.extend(person.enrollment[:2])
        enrollment.append((name, tuple(range(start, len(photos)))))
    photos.append(known[0].queries[0])
    photos.append(strangers[0])
    return RecognitionFixture(
        photos=tuple(photos),
        enrollment=tuple(enrollment),
        query=len(photos) - 2,
        query_of=FIXTURE_NAMES[0],
        stranger=len(photos) - 1,
    )


def recognition_accepts(chain, photos, fixture, enroll_call, query_call):
    """Remember two people, then ask who is in a photo of one of them.

    Returns ``(passed, detail)``. ``passed`` is False, 0.5 or 1.0.

    The half mark is not a near miss, it is which of a team's own functions to
    bind when several of them answer. A query that names the right person and
    names the stranger too has done half of what this benchmark measures, so
    it grades half; one that also rejects grades full. The resolver keeps the
    best grade it has seen and stops looking once it has a full one, so a
    team who wrote both a matcher with a cutoff and one without gets the one
    that can say "not this person" rather than whichever was reached first.

    A team who names the stranger still binds, at 0.5, and the benchmark
    scores their rejection recall at zero, which is the number they can act
    on. The one tuning failure this cannot forgive is the opposite: a cutoff
    strict enough to reject somebody it has just been shown two photos of
    fails the first question, and a store that never stored anything fails it
    the same way, so there is no telling those apart from here. The fixture
    gives each person two enrolment photos rather than one so that a cutoff
    has a profile worth matching against, which is the only lever this has.
    """

    with _fresh_state():
        try:
            described = _described(chain, photos)
        except BaseException as error:  # noqa: BLE001 - student code raises anything
            return False, (
                f"describing a photo raised {type(error).__name__}: {str(error)[:120]}"
            )
        return _attempt(described, fixture, enroll_call, query_call)


def _attempt(described, fixture, enroll_call, query_call):
    """One enrolment and up to two questions, on one store."""

    enrolled = 0
    for name, positions in fixture.enrollment:
        for position in positions:
            # The first face and no other, which is what the scored run does
            # (`DiscoveredRecognition`). A search that read every face proved
            # bindings the run could not execute: a photo with a spurious
            # detection in front of the person answered during the search and
            # came back unknown when it was scored.
            for row in described[position][:1]:
                try:
                    # A copy per attempt, on top of the one `descriptors_in`
                    # made. These rows are cached and served to every store
                    # and query the search pairs, so one of them writing
                    # through what it is handed would change what the next
                    # attempt is offered. Bagel's `add` keeps what it is given
                    # (`np.asarray(v, float32).ravel()` copies nothing when
                    # the array is already float32 and contiguous).
                    enroll_call(name, row.copy())
                    enrolled += 1
                except TypeError as error:
                    # A signature mismatch is not the wrong function, it is a
                    # different arrangement of the same one.
                    return False, (
                        f"enrolling did not accept those arguments: {str(error)[:120]}"
                    )
                except BaseException as error:  # noqa: BLE001 - student code raises
                    return False, (
                        f"enrolling raised {type(error).__name__}: {str(error)[:120]}"
                    )
    if not enrolled:
        # Nothing was offered, so nothing was tested, and every store would
        # pass this and then fail at the query. Their describe step found no
        # face in any of the enrolment photos, and saying so here is the
        # difference between naming the step that failed and telling a team
        # their query answered "no one".
        offered = sum(len(positions) for _, positions in fixture.enrollment)
        who = " and ".join(name for name, _ in fixture.enrollment)
        return False, f"found no face in any of the {offered} photos of {who}"
    if query_call is None:
        who = " and ".join(name for name, _ in fixture.enrollment)
        return True, f"enrolled {who} from {enrolled} descriptors"

    known = frozenset(name for name, _ in fixture.enrollment)
    if not described[fixture.query]:
        # Their detector, not their matcher. Saying "got no one" here would
        # send them to the cutoff over a photo nothing was ever asked about.
        return False, (
            f"found no face in the photo of {fixture.query_of} it was going to ask about"
        )
    asked, said = _asked(query_call, described[fixture.query], known)
    if said is not None:
        return False, said
    if asked != fixture.query_of:
        got = "no one" if asked is None else asked
        return False, f"asked who is in a photo of {fixture.query_of} and got {got}"
    stranger, said = _asked(query_call, described[fixture.stranger], known)
    if said is not None:
        return False, said
    if stranger is None:
        return 1.0, f"named {fixture.query_of} and called a stranger unknown"
    return 0.5, f"named {fixture.query_of}, and named the stranger {stranger}"


def _asked(query_call, rows, known):
    """Who the store says is in the first face of one photo, or a refusal.

    Returns ``(name_or_None, None)`` on an answer and ``(None, detail)`` when
    their query raised or answered something this cannot read. Those are kept
    apart from a plain ``(None, None)`` because a query that raises or answers
    unreadably is a binding that does not work, and a query that answers
    "nobody" is a working binding making a decision.
    """

    if not rows:
        return None, None
    try:
        answer = query_call(rows[0].copy())
    except BaseException as error:  # noqa: BLE001 - student code raises anything
        return None, f"querying raised {type(error).__name__}: {str(error)[:120]}"
    if not readable(answer):
        # Refused here rather than left to fail mid-run. The scored adapter
        # raises on this shape, so accepting it would prove a binding the run
        # cannot execute, which is the one thing the search must never do.
        return None, (
            f"querying answered {repr(answer)[:80]}, which says neither a name "
            "nor nobody"
        )
    return named(answer, known), None


def enrollment_arrangements(store, person_id: str, descriptor: Any):
    """The two orders a team writes a call that files a descriptor by name.

    Both orders appear in the audited corpus and neither is wrong:
    CoggurtFilter's `add_descriptor(db, name, descriptor)` and Lashika's
    `FaceDatabase.add_descriptor(name, descriptor)` put the name first,
    Bagel's `VectorDatabase.add(vector, name)` puts it second. The database
    itself is already in front of both when a team keeps one as an argument;
    the resolver binds it there before this is called.

    Every entry is built without calling anything, because the resolver counts
    them once with a store that does nothing.
    """

    return (
        lambda: store(person_id, descriptor),
        lambda: store(descriptor, person_id),
    )


def looks_like_an_empty_database(candidate: Any) -> bool:
    """One of their zero-argument functions that makes an empty face database.

    Emptiness is the whole test. A function that hands back a database with
    people already in it is not a factory, it is a loader, and scoring against
    it would score their data: Asterisk's `load_database()` takes no arguments
    and returns the ten identities they committed to their repository.

    Every attribute counts, including the private ones, because the two
    mistakes here are not the same size. Missing a full database costs a run
    scored against their data, which is the reason this exists. Refusing an
    object that holds some bookkeeping this cannot size costs nothing: a team
    whose database is an object of their own is found through the shape that
    needs no factory, which is how both repositories that score were found.
    """

    call = getattr(candidate, "call", candidate)
    try:
        inspect.signature(call).bind()
    except (TypeError, ValueError):
        return False
    try:
        with _fresh_state():
            # Inside a throwaway directory because their factory may write a
            # file next to itself.
            made = call()
    except BaseException:  # noqa: BLE001 - student code raises anything
        return False
    if isinstance(made, (dict, list, set, frozenset)):
        return not made
    if made is None or isinstance(made, (str, bytes, tuple, int, float, bool, np.ndarray)):
        return False
    state = getattr(made, "__dict__", None)
    if not isinstance(state, dict):
        return False
    return not any(_holds_anything(value) for value in state.values())


def _holds_anything(value: Any) -> bool:
    """Whether an attribute of a made-fresh object counts as contents."""

    if value is None or isinstance(value, (str, bytes, int, float, bool)):
        return False
    try:
        return len(value) > 0
    except TypeError:
        # Something with no size, a lock or a path. Counted as contents,
        # because this cannot see whether it is empty and guessing wrong in
        # the other direction scores a run against their data.
        return True


def descriptors_in(answer: Any) -> List[Any]:
    """Every face descriptor a chain produced for one photo, as a list of rows.

    The benchmark asks about one photo at a time, so this never has to decide
    which photo a row came from. What it does decide is how many faces are in
    the answer: a step bound per photo hands back a one-element list holding
    that photo's answer, a step that describes every face hands back a matrix,
    and a step that picks one face hands back a vector. All three say the same
    thing about the same photo.

    Each row is copied out, because this is where the benchmark takes a value
    student code produced and starts keeping it. A describe step that writes
    into one array every call is a thing a team writes to avoid allocating,
    and reading a row of it later reads whatever photo went through last: two
    photos described before either was asked about came back as two copies of
    the second.
    """

    if isinstance(answer, np.ndarray) and looks_like_face_descriptors(answer):
        return [answer.copy()] if answer.ndim == 1 else [row.copy() for row in answer]
    if isinstance(answer, (list, tuple)):
        rows: List[Any] = []
        for item in answer:
            rows.extend(descriptors_in(item))
        return rows
    return []


def named(answer: Any, known: Any) -> Optional[str]:
    """Which of the identities the benchmark enrolled their answer names.

    The benchmark chose every name it handed over, so an answer that is not
    one of them did not identify anybody it asked about, and the contract's
    word for that is None. That is why there is no table of sentinels here:
    the corpus says "Unknown", "unknown", and once a dictionary whose
    `prediction` is "unknown", and none of them is a name this run enrolled.
    A wrong name that the benchmark did enrol is kept as that name, because
    naming the wrong person and naming nobody are different mistakes with
    different fixes, and the scorer already reports them apart.
    """

    said = _stated_name(answer)
    if not isinstance(said, str):
        return None
    said = said.strip()
    return said if said in known else None


def readable(answer: Any) -> bool:
    """Whether their answer says anything this can read as a name or as none.

    False is not a rejection. It is an answer shaped like nothing the contract
    has a word for: a bare number, an empty list, a mapping that states two
    names. It is scored as though they had said "I do not know this person",
    because ending a scenario over one is worse than scoring it, and it is
    counted so that it is not laundered into a rejection in silence.
    """

    return _stated_name(answer) is not _UNREADABLE


#: Their answer said nothing this can read, which is not the same as their
#: saying they do not know the person.
_UNREADABLE = object()


def _stated_name(answer: Any) -> Any:
    """The name their answer states, None when it states nobody, else unreadable.

    Two rules, and between them they read every shape in the corpus without
    depending on how a team happened to build it.

    A name is the answer itself, or one of the answer's own parts. It is never
    something reached by going down through a collection. Bagel's `predict`
    returns their decision under `prediction` and the ranking it came from
    under `results`, so a reader that descends into `results` finds the
    top-ranked name and turns their correct rejection into an identification.

    And the answer's parts must say one thing. A `("ada", 0.2)` pair and a
    `{"prediction": "unknown", "similarity": 0.4, "results": [...]}` mapping
    each state exactly one name-shaped part, so each is read. Something that
    states two, like a mapping holding both a decision and the nearest name
    beside it, is not an answer this can read, and guessing between them means
    the reading depends on which the team wrote first. None of the audited
    repositories answers that way; a team who does is refused at the
    acceptance test rather than scored on a coin toss.

    A team whose only query answers with a ranking and no decision is likewise
    not read. `DiscoverySpec.readers` is what the SDK provides for running one
    of their own functions over an answer, and none of the five needs it.
    """

    if isinstance(answer, str) or answer is None:
        return answer
    if not isinstance(answer, (dict, list, tuple)):
        return _UNREADABLE
    parts = answer.values() if isinstance(answer, dict) else answer
    said = [part for part in parts if part is None or isinstance(part, str)]
    return said[0] if len(said) == 1 else _UNREADABLE


#: The last chain's descriptors, so a pairing search does not describe the
#: same photos again for every store and query it tries. One entry, because
#: every attempt for one chain happens inside one call of the resolver's
#: pairing loop. Instrumented on CoggurtFilter: 110 acceptance runs, 109 of
#: them served from here, 0.2 seconds describing photos altogether against
#: 76 seconds inside their own store and query.
_DESCRIBED: List[Any] = []


def _described(chain: Sequence[Any], photos: Sequence[Any]) -> List[List[Any]]:
    """The fixture's descriptors under this exact chain, one entry per photo.

    The photos are part of what is cached, and by identity: two resolves in
    one process bind Candidates that compare equal, so the chain alone would
    hand a second fixture the first one's answers. The entry holds the photos,
    so nothing here can be freed and its identity reused.
    """

    key = tuple(chain)
    if _DESCRIBED and _DESCRIBED[0] == key and _same_photos(_DESCRIBED[1], photos):
        return _DESCRIBED[2]
    described = [descriptors_in(_run(chain, [photo])) for photo in photos]
    _DESCRIBED[:] = [key, list(photos), described]
    return described


def _same_photos(cached: Sequence[Any], photos: Sequence[Any]) -> bool:
    """The same photo objects, not merely as many of them."""

    if len(cached) != len(photos):
        return False
    return all(one is other for one, other in zip(cached, photos))


def _run(chain: Sequence[Any], images: Sequence[Any]) -> Any:
    """Push images through a resolved chain, changing nothing on the way.

    Each step is called as the search called it: `bound` carries the tuning
    that made it run, and a returned pair is spread when the next function
    takes both parts. Neither transforms an answer. Without them a chain the
    search accepted raised on its first call here and the verdict blamed
    their algorithm.
    """

    value = _step(chain[0], (images,))
    for step in chain[1:]:
        produced = _step(step, (value,))
        # A step that answered on the graph it was given returns something
        # else (their `whispers` returns how the component count moved).
        # The graph goes forward, exactly as the search carried it.
        if not getattr(step, "in_place", False):
            value = produced
    return value


def _step(step: Any, args: tuple) -> Any:
    """Call one step with the hand-off the search used for it.

    The offers come from the search's own `_handoffs`, in its order: the
    value whole, spread as arguments, the two parts reversed, then each
    part on its own. Any exception moves to the next offer, because that is
    what the search did (`_call` treats every failure as "not this one").
    A hand-written subset here drifted twice from that list, and each time
    a chain the search had accepted raised on the same input when scored.
    """

    from cogbench.pipeline import _Spread, _handoffs

    call = getattr(step, "bound", step.call)
    if len(args) != 1:
        return call(*args)
    error: BaseException = TypeError("no offer accepted")
    for offered, _note in _handoffs(args[0]):
        try:
            return call(*offered) if isinstance(offered, _Spread) else call(offered)
        except BaseException as caught:  # noqa: BLE001 - student code raises anything
            error = caught
    raise error


def write_photos(images: Sequence[Any]) -> List[Any]:
    """The same photos, written out as PNG files, returning their paths.

    The capstone document tells students to write "a function that takes in a
    list of image-paths" (docs/capstones/week2-vision-capstone.md:386), and
    all three audited 2026 repositories did. The benchmark hands over arrays,
    so a team that followed the instructions had no function the search could
    call, and every one of them was refused at the first step.

    Which is our contract's problem, not their code's. The photos are the same
    photos either way; a file is one of the two shapes the course taught, so
    the search offers both and their own function decides which it takes.
    """

    from PIL import Image

    # Kept for the life of the process rather than deleted after the probe:
    # the search calls their function many times, and a path that stopped
    # existing between calls would look like their code failing.
    from pathlib import Path as _Path

    folder = tempfile.mkdtemp(prefix="cogworks-week2-photos-")
    # Never deleted during the run: a bound step may read these paths again
    # at scoring time. Removed when the process ends. Measured before this:
    # one leaked directory per resolve, and a full disk after a corpus pass.
    atexit.register(shutil.rmtree, folder, ignore_errors=True)
    paths = []
    for index, image in enumerate(images):
        path = _Path(folder) / "photo_{:03d}.png".format(index)
        Image.fromarray(np.asarray(image, dtype=np.uint8)).save(path)
        # Path objects, not strings. One audited repository does
        # `p.parent` on what it is given, and a string has no `.parent`, so a
        # str fixture failed their code over how we spelled a path rather than
        # over anything they wrote. `str(path)` works on a Path everywhere the
        # corpus uses one; the reverse is not true.
        paths.append(path)
    return paths


@contextlib.contextmanager
def _fresh_state():
    """One attempt, one empty directory, and no narration.

    Their functions write: one audited repository sorts photos into a `result`
    directory as it clusters. And they print, at length. Neither belongs in the
    search for which of their functions to call.
    """

    previous = os.getcwd()
    saved_out, saved_err = sys.stdout, sys.stderr
    with tempfile.TemporaryDirectory(prefix="cogworks-week2-") as temporary:
        os.chdir(temporary)
        sys.stdout = io.StringIO()
        sys.stderr = io.StringIO()
        try:
            yield
        finally:
            sys.stdout, sys.stderr = saved_out, saved_err
            os.chdir(previous)
