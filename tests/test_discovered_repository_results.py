"""Pins automatic discovery results for repositories in the optional corpus.

The corpus is not part of this repository, so these tests skip when its local
checkout or the named repository is absent.

One repository per test and few tests on purpose. Importing a team's modules
puts their names into this interpreter, and two teams both writing `whispers.py`
displace each other: running all five recognition repositories in one process
left the last two with nothing bound, where each on its own binds. Both surfaces
that run this for real, `cogworks` and the hosted runner, use one process per
repository.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
CORPUS = ROOT / ".cache" / "student-repos"

sys.path.insert(0, str(ROOT / "python" / "cogbench" / "src"))
sys.path.insert(0, str(ROOT / "benchmarks" / "week2"))


@unittest.skipUnless(CORPUS.is_dir(), "student corpus is not checked out")
class DiscoveredRepositoryResults(unittest.TestCase):
    def test_lashika_vision_module_capstone(self) -> None:
        from cogbench.plugins import load_benchmark
        from cogbench.resolve import resolve
        from cogbench.runner import _facenet_model

        name = "LashikaKapoor28__Vision_Module_Capstone"
        repo = (CORPUS / name).resolve()
        if not repo.is_dir():
            self.skipTest("{} is not in this checkout".format(name))

        plugin = load_benchmark("vision-clustering")
        spec = plugin.discovery()
        found = resolve(
            repo,
            chain_role=spec.chain_role,
            fixture=spec.fixture,
            accepts=spec.accepts,
            arrangements=spec.arrangements,
            hints=spec.hints,
            benchmark="vision-clustering",
        )

        self.assertEqual(found.verdict.status, "scored")
        self.assertEqual(
            [step.label for step in found.chain],
            ["whispers.adj_list", "whispers.whispers", "whispers.connected_comps"],
        )

        cases = plugin.load_cases("test")
        model = _facenet_model()
        score = plugin.score(
            plugin.run(
                lambda *args, **kwargs: plugin.submission_from_discovery(found),
                model,
                cases,
            ),
            cases,
        )
        self.assertAlmostEqual(
            score["clustering_pairwise_f1"], 0.9090909090909091, places=4
        )


@unittest.skipUnless(CORPUS.is_dir(), "student corpus is not checked out")
class DiscoveredRecognitionResults(unittest.TestCase):
    """What recognition discovery finds in one repository, and what it scores.

    Bagel rather than the other repository that scores, because its modules
    import in a second where CoggurtFilter's run a clustering sweep at import
    and take thirty, and this pins a result rather than measuring a search.
    """

    NAME = "BagelBreaker__week2_capstone"
    SHA = "29f9cf94751d6267e00be3ae4816be98a027662e"

    def repository(self) -> Path:
        repo = (CORPUS / self.NAME).resolve()
        if not repo.is_dir():
            self.skipTest(f"{self.NAME} is not in this checkout")
        import subprocess

        head = subprocess.run(
            ["git", "-C", str(repo), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
        )
        if head.stdout.strip() != self.SHA:
            # The numbers below belong to this commit. A checkout that has
            # moved is a different submission, not a regression.
            self.skipTest(f"{self.NAME} is not at {self.SHA}")
        return repo

    def test_their_own_describe_store_and_query_are_what_binds(self) -> None:
        from cogbench.plugins import load_benchmark
        from cogbench.resolve import from_spec

        repo = self.repository()
        plugin = load_benchmark("vision-recognition")
        found = from_spec(repo, plugin.discovery(), benchmark="vision-recognition")

        self.assertEqual(found.verdict.status, "scored")
        self.assertEqual(
            [step.label for step in found.chain], ["get_descriptor.file_descriptors"]
        )
        self.assertEqual(found.attempt.enroll, "vector_db.VectorDatabase().add")
        # `query` ranks the nearest rows and never rejects; `predict` puts
        # their own cutoff in front of the same ranking. Binding the first one
        # reached would score this team zero on unknown rejection using code
        # they wrote, so the acceptance test grades a rejection above no
        # rejection and reads no name out of a bare ranking.
        self.assertEqual(found.attempt.query, "vector_db.VectorDatabase().predict")

    def test_the_lifecycle_it_scores_is_theirs(self) -> None:
        from cogbench.plugins import load_benchmark
        from cogbench.resolve import from_spec
        from cogbench.runner import _facenet_model

        repo = self.repository()
        plugin = load_benchmark("vision-recognition")
        found = from_spec(repo, plugin.discovery(), benchmark="vision-recognition")
        cases = plugin.load_cases("test")

        outputs = plugin.run(
            lambda *args, **kwargs: plugin.submission_from_discovery(found),
            _facenet_model(),
            cases,
        )
        score = plugin.score(outputs, cases)

        # The reference application misses the same known query on this case
        # and scores 0.9167, so what separates them is the half of the
        # lifecycle their cutoff refuses: a stranger enrolled from one photo.
        self.assertEqual(outputs[0]["unknown_before"], [None, None])
        self.assertEqual(outputs[0]["post_enrollment"], [None, None])
        self.assertAlmostEqual(score["known_identification"], 5 / 6, places=4)
        self.assertAlmostEqual(score["unknown_rejection_recall"], 1.0, places=4)
        self.assertAlmostEqual(score["recognition_score"], 2 / 3, places=4)


if __name__ == "__main__":
    unittest.main()
