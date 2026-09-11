"""Turn a discovered binding into the submission a Week 2 driver runs.

Discovery finds which of a team's functions do the week's work. Clustering
wants an object with ``cluster(images, *, seed)``; recognition wants one with
``enroll(person_id, images)`` and ``recognize(images)``. This is the short
piece between them, and it lives here because the protocols it writes to are
Week 2's.

Their functions are called exactly as the search called them, because the
search is what proved the chain works. Nothing is repaired: a function that
raises raises, and the driver records it against the scenario, which is how a
bug in their code stays visible as theirs.
"""

from __future__ import annotations

import contextlib

from typing import Any, List, Optional, Sequence

__all__ = [
    "DiscoveredClustering",
    "DiscoveredRecognition",
    "build",
    "build_recognition",
]


class DiscoveredClustering:
    """A team's own functions, wearing the interface the driver expects."""

    def __init__(self, chain: Sequence[Any]) -> None:
        self._chain = list(chain)

    def cluster(self, images: Sequence[Any], *, seed: int = 0) -> List[Any]:
        """Photos in, one label per photo out.

        ``seed`` is applied to Python's `random` and not passed on. Their
        functions were bound by calling them with photos alone, which is the
        signature the whole corpus wrote; a team who takes a seed is welcome
        to, and the benchmark already measures how much their answer moves
        between draws.

        The photos go in as the form their first function was bound with:
        arrays, or the same photos written to disk as paths. That is not a
        transformation of their answer. The search proved the chain on one
        of those two forms and the scored run must present the same one, or
        it scores a function that never ran.
        """

        import random

        from cogbench.pipeline import identities_for, runtime_pool

        from .roles import (
            _run,
            labels_in_photo_order,
            lay_out_folders,
            write_photos,
        )

        # The driver passes a seed so a scored run is reproducible. Their
        # `whispers` draws from `random` without seeding, so the seed has to
        # be set here, before their loop runs; the search seeds the same way
        # (`_resolve_chain`), and the instructor adapter for this corpus does
        # too. Not passed on: their functions were bound without it.
        random.seed(seed)
        photos = list(images)
        if getattr(self._chain[0], "form", None) == 1:
            photos = write_photos(photos)
        # A step bound on the item's identity takes THIS run's photos, not
        # the ones the search wrote. One 2026 team's `Whispers(vectors,
        # names, threshold)` keeps each name on its node and their
        # `sorted_images()` returns the groups keyed by them; replaying the
        # search's names made every group name a photo this run never saw,
        # and reading the answer back raised instead of scoring. Worked out
        # by the search's own rule so the two cannot drift apart.
        with runtime_pool({"identity": identities_for((), (photos,))}):
            # A step of theirs that reads a directory was bound by writing
            # the benchmark's photos into one of that name, so a scored run
            # has to present the same directory. Done from a throwaway
            # working directory: the folder is the benchmark's input, and
            # writing it wherever the runner happened to start would leave
            # a folder of photos in somebody's checkout.
            with _somewhere_throwaway():
                lay_out_folders(self._chain, photos)
                answer = _run(self._chain, photos)
        # Their answer is read the way the acceptance test read it. Two of
        # the three audited teams end at `connected_comps`, which returns
        # groups of their own node objects; the driver wants one label per
        # photo in photo order. Handing the groups over unread made the
        # driver count 4 or 8 "labels" for 12 photos and refuse a chain the
        # search had just proved.
        return labels_in_photo_order(answer, photos)


class DiscoveredRecognition:
    """A team's own functions, wearing the interface the recognition driver expects.

    Three of theirs rather than one chain: the step that describes a photo,
    the call that files a descriptor under a name, and the call that looks one
    up. The search bound all three together by running them, and this replays
    them the same way, one photo at a time.

    Their describe step is called one photo at a time because the driver asks
    for exactly one label per image. Running the whole batch through it would
    hand back a pile of descriptors with no way to say which photo each came
    from, once a photo yields no face or two.
    """

    # One photo, one face: the first their describe step returned, on both
    # sides. A contract that asks for one label per photo has to reduce their
    # per-face answer to a per-photo one somewhere, and every way of doing it
    # leans, so this picks the way that leans least and says which way.
    #
    # The alternative was to enrol every face and then answer with the first
    # one their matcher recognized, and it leans twice in the same direction:
    # a spurious box filed under a person is what a later stranger matches,
    # and reading past a face their matcher rejected can only turn a correct
    # rejection into a name, never the reverse.
    #
    # For a team who picks a face themselves this is their pick, because it
    # happened inside their own describe step; for one who returns them all
    # it is their detector's order. Extra faces are counted and never asked
    # about. The acceptance test asks the same question (`roles._attempt`), so
    # the search cannot prove a binding the run will not execute. This is part
    # of the contract, so changing it is a new benchmark version.

    def __init__(self, chain: Sequence[Any], enroll_call: Any, query_call: Any) -> None:
        self._chain = list(chain)
        self._enroll = enroll_call
        self._query = query_call
        # The names this has handed over. Not a copy of their database: the
        # benchmark never reads that. This is the benchmark's own record of
        # what it asked them to remember, and it is what lets an answer be
        # read back as an identification or as none.
        self._known: set = set()
        # What happened to a photo before their matcher saw it, and what came
        # back that this could not read. Counted rather than dropped, because
        # all of it ends as None: a run of them scores exactly what a
        # submission that answers None to everything scores, and the metric's
        # own diagnostics read that as a cutoff being strict. `score` says so.
        #
        # Enrolment and query are counted apart because the fix differs. A
        # query photo with no face is answered unknown; an enrolment photo
        # with no face leaves that person's profile thinner, or empty, and
        # every later question about them suffers for it.
        self.photos_not_enrolled = 0
        self.photos_not_answered = 0
        self.faces_not_asked_about = 0
        self.answers_not_read = 0

    def enroll(self, person_id: str, images: Sequence[Any]) -> None:
        """File the first face in each of these photos under one name."""

        self._known.add(person_id)
        for rows in self._describe(images):
            if rows:
                self._enroll(person_id, rows[0])
            else:
                self.photos_not_enrolled += 1

    def recognize(self, images: Sequence[Any]) -> List[Optional[str]]:
        """One answer per photo: what their query said about its first face.

        Their cutoff and their rejection are the only things deciding. Nothing
        here applies a threshold, and when their query named nobody the answer
        is None, which is this contract's word for "I do not know this person"
        and the honest thing to say about a photo their system could not put a
        name to.
        """

        from .roles import named, readable

        answers: List[Optional[str]] = []
        for rows in self._describe(images):
            if not rows:
                self.photos_not_answered += 1
                answers.append(None)
                continue
            answer = self._query(rows[0])
            if not readable(answer):
                self.answers_not_read += 1
            answers.append(named(answer, self._known))
        return answers

    def _describe(self, images: Sequence[Any]) -> List[List[Any]]:
        """Their describe step over these photos, in the form it was bound with."""

        # Imported here rather than at module scope, like everything else
        # this file takes from `.roles`: that module needs cogbench, and this
        # package is importable, and its other suites run, without it.
        from .roles import _run, descriptors_in, write_photos

        photos = list(images)
        # The photos go in as the form their first function was bound with:
        # arrays, or the same photos written to disk as paths. The search
        # proved the chain on one of those two forms and the scored run has to
        # present the same one, or it scores a function that never ran. Written
        # in one batch rather than one file at a time, because each call of
        # `write_photos` makes a directory that lives until the process ends.
        if getattr(self._chain[0], "form", None) == 1:
            photos = write_photos(photos)
        with _somewhere_throwaway():
            described = [descriptors_in(_run(self._chain, [photo])) for photo in photos]
        self.faces_not_asked_about += sum(len(rows) - 1 for rows in described if rows)
        return described


@contextlib.contextmanager
def _somewhere_throwaway():
    """A working directory their code may write into and read a folder from.

    Their functions write: one audited repository sorts photos into a
    `result` directory as it clusters, and another reads a directory of
    photos the benchmark has to put there. Neither belongs in whatever
    directory the runner happened to start in.
    """

    import os
    import tempfile

    with tempfile.TemporaryDirectory(prefix="cogworks-week2-run-") as temporary:
        previous = os.getcwd()
        os.chdir(temporary)
        try:
            yield
        finally:
            os.chdir(previous)


def build(submission: Any) -> DiscoveredClustering:
    if not getattr(submission, "ready", False):
        raise RuntimeError("This repository did not resolve, so there is nothing to run.")
    return DiscoveredClustering(submission.chain)


def build_recognition(submission: Any) -> DiscoveredRecognition:
    """The same repository, wearing the recognition driver's interface.

    `fresh()` is what makes each scenario start empty. Proving the binding
    works enrolled two people into the team's database, and when that database
    is an object or one of their own zero-argument factories those two are
    still in it; scoring from it would leave `fixture_a` competing with the
    people the case is actually about. The driver builds one of these per
    scenario, so this runs once per scenario and each one starts from a
    database their own code just made.

    A team whose database is a module global has nothing to rebuild, and this
    cannot give them an empty one. What happens then depends on their code:
    the search's probing writes into a global it cannot restore, and a store
    that reads those rows back raises and is refused, while one that ignores
    them binds and starts the run holding the two people the search enrolled.
    `roles.named` keeps those two out of an answer, because it hands back only
    names the run itself enrolled, but it cannot keep them from competing.
    None of the audited repositories has that shape and reaches a run. Giving
    them an empty database would mean restoring a module's globals between
    attempts, which belongs to whatever is calling their functions rather than
    to the week describing the task.
    """

    if not getattr(submission, "ready", False):
        raise RuntimeError("This repository did not resolve, so there is nothing to run.")
    ready = submission.fresh()
    return DiscoveredRecognition(ready.chain, ready.enroll, ready.query)
