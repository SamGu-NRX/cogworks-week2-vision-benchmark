"""Re-running the same photos under other seeds, and what it catches.

Whispers picks a random visit order. The manifests pinned one seed, so a
submission whose answer swings between draws scored whatever that single draw
happened to give, and nothing said the number was a coin flip.
"""

from __future__ import annotations

import random

from facial_recognition_benchmark.drivers import ClusteringScenario
from facial_recognition_benchmark.plugins import ClusteringBenchmark


def scenario(seed, scored=True):
    return ClusteringScenario(
        images=[],
        expected_labels=["a", "a", "a", "b", "b", "b"],
        seed=seed,
        scored=scored,
    )


def cases():
    """One scored case plus three sweep repeats, as load_cases builds them."""

    return [scenario(1729)] + [scenario(s, scored=False) for s in (20260819, 31337, 8675309)]


class TestOnlyTheScoredCaseCounts:
    def test_the_sweep_does_not_move_the_published_number(self):
        """The whole design rests on this. A team's F1 must be what the
        manifest's own seed produced, not an average over seeds we invented."""

        bench = ClusteringBenchmark()
        perfect = [0, 0, 0, 1, 1, 1]
        # The scored case is perfect; every repeat is garbage.
        outputs = [perfect, [0] * 6, [0] * 6, [0] * 6]
        metrics = bench.score(outputs, cases())
        assert metrics["clustering_pairwise_f1"] == 1.0

    def test_a_scenario_list_without_repeats_reports_no_spread(self):
        bench = ClusteringBenchmark()
        metrics = bench.score([[0, 0, 0, 1, 1, 1]], [scenario(1729)])
        assert "clustering_seed_spread" not in metrics


class TestTheSweepCatchesInstability:
    def test_an_unstable_clusterer_is_named(self):
        bench = ClusteringBenchmark()
        outputs = [
            [0, 0, 0, 1, 1, 1],   # scored: perfect
            [0, 0, 1, 1, 1, 1],   # and then the answer wanders
            [0] * 6,
            [0, 1, 2, 3, 4, 5],
        ]
        metrics = bench.score(outputs, cases())
        assert metrics["clustering_seed_spread"] > 0.15
        note = " ".join(bench.last_diagnostics)
        assert "seed" in note.lower()
        assert "visit order" in note

    def test_a_stable_clusterer_is_told_so(self):
        """Stability is a result, and a team that got it should hear it."""

        bench = ClusteringBenchmark()
        perfect = [0, 0, 0, 1, 1, 1]
        metrics = bench.score([perfect] * 4, cases())
        assert metrics["clustering_seed_spread"] == 0.0
        assert "held to within" in " ".join(bench.last_diagnostics)

    def test_a_middling_spread_gets_no_sentence(self):
        """Between the two thresholds there is nothing useful to say, and a
        sentence for every run is a sentence nobody reads.

        Sized at 32 images, which is what the evaluation manifest holds. On a
        six-image toy, one misplaced photo moves F1 by 0.38 and there is no
        middle at all; on 32 the same mistake moves it by 0.07. The thresholds
        are calibrated for the real case, so the fixture has to be too.
        """

        truth = ["p{}".format(i // 4) for i in range(32)]
        perfect = [i // 4 for i in range(32)]
        one_off = list(perfect)
        one_off[5] = one_off[0]

        big = [
            ClusteringScenario(images=[], expected_labels=truth, seed=1729),
        ] + [
            ClusteringScenario(images=[], expected_labels=truth, seed=s, scored=False)
            for s in (20260819, 31337, 8675309)
        ]
        bench = ClusteringBenchmark()
        metrics = bench.score([perfect, one_off, perfect, perfect], big)
        assert 0.02 < metrics["clustering_seed_spread"] < 0.15
        assert bench.last_diagnostics == []


class TestSpreadIsPerScenario:
    """Repeats only mean something next to their own scenario's scored case.

    Pooling every scenario's F1 into one list reports the difficulty gap
    between scenarios as instability: a clusterer that is perfectly stable
    on both an easy and a hard scenario would be told its answer swings.
    """

    @staticmethod
    def two_scenarios():
        easy = ["a", "a", "a", "b", "b", "b"]
        hard = ["a", "a", "b", "b"]
        out = []
        for key, labels in (("easy", easy), ("hard", hard)):
            out.append(
                ClusteringScenario(
                    images=[], expected_labels=labels, seed=1729, scenario_key=key
                )
            )
            out.extend(
                ClusteringScenario(
                    images=[],
                    expected_labels=labels,
                    seed=s,
                    scored=False,
                    scenario_key=key,
                )
                for s in (20260819, 31337, 8675309)
            )
        return out

    def test_a_stable_clusterer_on_scenarios_of_different_difficulty_reports_zero(self):
        bench = ClusteringBenchmark()
        perfect_easy = [0, 0, 0, 1, 1, 1]
        one_cluster_hard = [0, 0, 0, 0]  # stable everywhere, F1 0.5 on hard
        outputs = [perfect_easy] * 4 + [one_cluster_hard] * 4
        metrics = bench.score(outputs, self.two_scenarios())
        assert metrics["clustering_pairwise_f1"] == 0.75
        assert metrics["clustering_seed_spread"] == 0.0
        note = " ".join(bench.last_diagnostics)
        # One primary seed plus three sweep seeds, not a case count.
        assert "the 4 seeds" in note

    def test_the_worst_scenario_sets_the_reported_spread(self):
        bench = ClusteringBenchmark()
        perfect_easy = [0, 0, 0, 1, 1, 1]
        outputs = [perfect_easy] * 4 + [
            [0, 0, 1, 1],      # scored: perfect on hard
            [0, 0, 0, 0],      # and then the answer wanders
            [0, 1, 2, 3],
            [0, 0, 1, 1],
        ]
        metrics = bench.score(outputs, self.two_scenarios())
        assert metrics["clustering_seed_spread"] > 0.15
        assert "visit order" in " ".join(bench.last_diagnostics)


class TestSeedsAreFixed:
    def test_the_sweep_seeds_do_not_depend_on_the_clock_or_the_process(self):
        """Two runs of one submission must face the same seeds, or a team
        comparing runs is comparing noise."""

        from facial_recognition_benchmark.datasets import STABILITY_SEEDS

        random.seed(1)
        first = tuple(STABILITY_SEEDS)
        random.seed(2)
        assert tuple(STABILITY_SEEDS) == first
        assert all(isinstance(s, int) for s in STABILITY_SEEDS)
