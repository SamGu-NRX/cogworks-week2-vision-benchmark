"""Turn a discovered binding into the submission the clustering driver runs.

Discovery finds which of a team's functions take photos and return one label
per photo. The driver wants an object with ``cluster(images, *, seed)``. This
is the short piece between them, and it lives here because the protocol it
writes to is Week 2's.

Their functions are called exactly as the search called them, because the
search is what proved the chain works. Nothing is repaired: a function that
raises raises, and the driver records it against the scenario, which is how a
bug in their code stays visible as theirs.
"""

from __future__ import annotations

import contextlib

from typing import Any, List, Sequence

__all__ = ["DiscoveredClustering", "build"]


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
