"""CogBench plugin objects for the two Week 2 vision tracks."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

from .datasets import (
    CacheStatus,
    cache_status,
    clustering_scenarios,
    load_manifest,
    recognition_scenarios,
)
from .drivers import (
    recognition_expected,
    run_clustering_scenario,
    run_recognition_scenario,
)
from .metrics import score_clustering, score_recognition


class RecognitionBenchmark:
    """Load, execute, and score known/unknown recognition scenarios."""

    benchmark_id = "vision-recognition"
    benchmark_version = 2
    contract_version = "cogworks.submissions.v2"
    plugin_version = "0.1.0"
    dataset_version = "celeba-manifests-v1"
    scorer_version = "recognition-v1"
    primary_metric = "recognition_score"

    metric_labels = {
        "recognition_score": "Recognition score",
        "known_identification": "Known identification",
        "unknown_rejection_recall": "Unknown rejection recall",
        "post_enrollment_accuracy": "Post-enrollment accuracy",
        "unknown_lifecycle": "Unknown lifecycle",
    }

    #: What each metric measures, in the course's own vocabulary, and which
    #: part of the capstone it comes from.
    #:
    #: A number a student cannot trace back to something they were taught is a
    #: black box, and a black box teaches nothing. The words here are CogWeb's
    #: -- "descriptor vector", "cosine distance", "cutoff" -- rather than ours.
    #: Source: docs/cogweb/pages/Video/FacialRecognition.md.
    #: Filled by score(). Present before the first run so a caller reading it
    #: early gets an empty list rather than an AttributeError.
    last_diagnostics: list = []

    metric_help = {
        "recognition_score": (
            "The mean of known identification and the unknown lifecycle below. "
            "Both halves count equally because the capstone is not only "
            "recognizing a face it has seen: saying \"I do not know this person\" "
            "and then learning them is the other half of the assignment. This is "
            "the leaderboard number."
        ),
        "known_identification": (
            "Of the people already in the database, how often the right name comes "
            "back. Averaged per person rather than per image, so someone with many "
            "photos cannot carry the score for someone with few. This is the "
            "cosine distance between descriptor vectors doing its job."
        ),
        "unknown_rejection_recall": (
            "How often a face that is not in the database is correctly called "
            "unknown rather than matched to the nearest name. This is the cutoff "
            "the course asks you to choose: too generous and every stranger "
            "becomes somebody, too strict and nobody is ever recognized."
        ),
        "post_enrollment_accuracy": (
            "After a stranger is added to the database, how often they are then "
            "identified correctly. A system that rejects unknowns but cannot learn "
            "them has only solved half the problem."
        ),
        "unknown_lifecycle": (
            "The mean of unknown rejection and post-enrollment accuracy: the whole "
            "arc of meeting a stranger, saying so, and knowing them next time."
        ),
    }

    def load_cases(self, tier: str, cache_root: Optional[Path] = None) -> Sequence[Any]:
        """Load the small test or larger public-evaluation cases."""

        return recognition_scenarios(_public_manifest(tier), cache_root)

    def run(
        self, factory: Any, model: Any, cases: Sequence[Any]
    ) -> List[Mapping[str, Sequence[Optional[str]]]]:
        """Execute every case through a fresh application adapter."""

        return [run_recognition_scenario(factory, model, case) for case in cases]

    def score(
        self,
        outputs: Sequence[Mapping[str, Sequence[Optional[str]]]],
        cases: Sequence[Any],
    ) -> Dict[str, float]:
        """Compare recognition outputs with trusted lifecycle labels.

        `score_recognition` returns its diagnostics alongside the numbers;
        they are lifted onto `last_diagnostics` here, matching Weeks 1 and 3,
        so the returned mapping stays metric-name to float and the runner can
        float() every value in it without special-casing a key.
        """

        scores = dict(score_recognition(outputs, [recognition_expected(case) for case in cases]))
        self.last_diagnostics = list(scores.pop("_diagnostics", []))
        return scores

    def cache_status(self, tier: str, cache_root: Optional[Path] = None) -> CacheStatus:
        """Report whether the selected public data tier is cached and valid."""

        return cache_status(_public_manifest(tier), cache_root)


class ClusteringBenchmark:
    """Load, execute, and score fixed-seed Whispers scenarios."""

    benchmark_id = "vision-clustering"
    benchmark_version = 2
    contract_version = "cogworks.submissions.v2"
    plugin_version = "0.1.0"
    dataset_version = "celeba-manifests-v1"
    scorer_version = "clustering-v2"
    primary_metric = "clustering_pairwise_f1"

    metric_labels = {
        "clustering_pairwise_f1": "Pairwise F1",
        "adjusted_rand_index": "Adjusted Rand index",
        "clustering_seed_spread": "Spread across seeds",
    }
    #: Reported only when the seed sweep ran. Lower is better: it measures how
    #: much the answer changed when nothing about the photos did.
    lower_is_better = {"clustering_seed_spread"}

    #: See RecognitionBenchmark.metric_help. Source for the vocabulary:
    #: docs/cogweb/pages/Video/Whispers.md.
    metric_help = {
        "clustering_pairwise_f1": (
            "Over every pair of photos, how often two photos of the same person "
            "end up in the same group and two photos of different people end up "
            "apart. Cluster numbers themselves do not matter -- calling a group "
            "3 instead of 1 changes nothing -- which is why the score is built "
            "from pairs. This is the leaderboard number."
        ),
        "adjusted_rand_index": (
            "The same agreement, corrected for how much a random grouping would "
            "score by luck. Reported next to pairwise F1 because a run that "
            "puts every photo in one giant cluster can look respectable on F1 "
            "and lands near zero here."
        ),
        "clustering_seed_spread": (
            "How much the pairwise F1 moved when the same photos were clustered "
            "again under different random seeds. Whispers picks a random visit "
            "order, so this says whether your answer is about the faces or about "
            "that order. A wide spread usually means some random choice is not "
            "using the seed you were given. This is reported and never scored."
        ),
    }

    def load_cases(self, tier: str, cache_root: Optional[Path] = None) -> Sequence[Any]:
        """Load the small test or larger public-evaluation cases."""

        return clustering_scenarios(_public_manifest(tier), cache_root)

    def run(self, factory: Any, model: Any, cases: Sequence[Any]) -> List[Sequence[Any]]:
        """Execute every clustering case through a fresh adapter."""

        return [run_clustering_scenario(factory, model, case) for case in cases]

    def score(self, outputs: Sequence[Sequence[Any]], cases: Sequence[Any]) -> Dict[str, float]:
        """Compute label-invariant metrics against trusted partitions.

        Only cases marked ``scored`` reach the metrics. The rest are the seed
        sweep, read here to report how far a randomized clusterer's answer
        moves between draws and never to change the score itself.
        """

        scored = [(o, c) for o, c in zip(outputs, cases) if getattr(c, "scored", True)]
        metrics = score_clustering(
            [o for o, _ in scored], [c.expected_labels for _, c in scored]
        )

        self.last_diagnostics = []
        if any(not getattr(c, "scored", True) for c in cases):
            # The spread is per scenario: each repeat reruns one scenario's
            # images, so it only says something next to that scenario's own
            # scored case. Pooling every scenario into one list would report
            # the difficulty gap between scenarios as instability. A repeat
            # names its scenario with scenario_key; when the key is absent it
            # belongs to the most recent scored case, which is the order
            # load_cases emits them in.
            f1_groups: Dict[Any, List[float]] = {}
            seed_groups: Dict[Any, set] = {}
            swept_keys: List[Any] = []
            last_scored_key: Any = None
            for index, (output, case) in enumerate(zip(outputs, cases)):
                is_scored = getattr(case, "scored", True)
                key = getattr(case, "scenario_key", None)
                if is_scored:
                    if key is None:
                        key = ("scored-at", index)
                    last_scored_key = key
                elif key is None:
                    # Repeats before any scored case have nothing to attach
                    # to; compare them with each other rather than hiding
                    # each in its own group.
                    key = last_scored_key if last_scored_key is not None else "orphan-sweep"
                f1 = score_clustering([output], [case.expected_labels])[
                    "clustering_pairwise_f1"
                ]
                f1_groups.setdefault(key, []).append(f1)
                seed_groups.setdefault(key, set()).add(getattr(case, "seed", None))
                if not is_scored and key not in swept_keys:
                    swept_keys.append(key)
            # The reported number is the worst scenario: one unstable scenario
            # means some random choice is loose even when the others held.
            # max() keeps the first of tied keys, so ties fall to the earliest
            # scenario.
            worst = max(swept_keys, key=lambda key: max(f1_groups[key]) - min(f1_groups[key]))
            f1s = f1_groups[worst]
            seed_count = len(seed_groups[worst])
            spread = max(f1s) - min(f1s)
            metrics["clustering_seed_spread"] = spread
            # Calibrated on the 32-image evaluation scenario, where one
            # misplaced photo moves F1 by 0.07 and a genuinely unstable run
            # moves it much further. On a six-image test scenario one mistake
            # moves it by 0.38, so the small tier will trip the loud threshold
            # more readily; that is the right direction for a tier students
            # use to debug.
            if spread >= 0.15:
                self.last_diagnostics.append(
                    "Re-running the same images under {} different seeds moved the F1 by "
                    "{:.2f} (from {:.2f} to {:.2f}). Whispers picks a random visit order, "
                    "so a spread this wide means the answer depends on that order more "
                    "than on the faces. Check that every random choice uses the seed you "
                    "were given.".format(seed_count, spread, min(f1s), max(f1s))
                )
            elif spread <= 0.02:
                self.last_diagnostics.append(
                    "The clustering held to within {:.3f} F1 across {} seeds, so the "
                    "answer is about the faces rather than the visit order.".format(
                        spread, seed_count
                    )
                )
        return metrics

    def cache_status(self, tier: str, cache_root: Optional[Path] = None) -> CacheStatus:
        """Report whether the selected public data tier is cached and valid."""

        return cache_status(_public_manifest(tier), cache_root)

    def discovery(self) -> Any:
        """What to look for in a repository that never packaged itself.

        None of the 2026 capstones registered an entry point, so asking for
        one asks for a step no team took. Instead the benchmark says what its
        task is -- photos in, one label per photo out -- and
        ``cogbench.resolve`` searches their repository against that by running
        their functions.

        The fixture is the first scored scenario: real photos of two people
        with a known grouping. Built lazily, because it reads the cached
        dataset and importing a plugin should not.
        """

        from cogbench.discovery_spec import DiscoverySpec
        from cogbench.pipeline import Fixtures

        from .roles import accepts, cluster_role_for, write_photos

        scenario = next(
            (case for case in self.load_cases("test") if getattr(case, "scored", False)),
            None,
        )
        if scenario is None:
            return None

        images = list(scenario.images)
        expected = list(scenario.expected_labels)
        # Two forms of the same photos. The capstone document tells students
        # to write "a function that takes in a list of image-paths"
        # (docs/capstones/week2-vision-capstone.md:386), and all three audited
        # 2026 repositories did, so an arrays-only fixture refused every team
        # that followed the instructions. Their own function decides which it
        # takes; the photos are identical either way.
        forms = Fixtures(((images,), (write_photos(images),)))
        # The acceptance test hands the chain the form it was bound with.
        # Before this it always handed arrays, so a team whose first
        # function took paths was called with the wrong thing and reported
        # as having run and answered wrongly.
        return DiscoverySpec(
            chain_role=cluster_role_for(len(images)),
            fixture=forms,
            accepts=lambda chain, *_: accepts(chain, forms.for_chain(chain)[0], expected),
            arrangements=None,  # nothing is stored between calls
            hints=("week2", "week 2", "vision", "faces", "capstone"),
            expects="one group per person",
        )

    def submission_from_discovery(self, submission: Any) -> Any:
        """Turn a resolved repository into the object ``run`` expects."""

        from .discovered import build

        return build(submission)


def _public_manifest(tier: str) -> Mapping[str, Any]:
    if tier not in ("test", "evaluation"):
        raise ValueError("Public tier must be 'test' or 'evaluation'.")
    return load_manifest(tier)
