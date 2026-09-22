"""Pins automatic discovery results for a repository in the optional corpus.

The corpus is not part of this repository, so this test skips when its local
checkout or the named repository is absent.
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


if __name__ == "__main__":
    unittest.main()
