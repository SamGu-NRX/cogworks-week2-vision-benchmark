"""Real-image validity arms on materialized CelebA rows.

Runs the reference FaceNet recognizer and the Whispers clusterer against the
packaged public manifests, plus degenerate baselines on real embeddings, and
writes one JSON file per tier under study/results/.

Deliberately NOT part of the hermetic harness: this script touches the
network (HuggingFace rows), the GPU-less model checkpoint, and wall-clock
time. The harness pins scoring logic; this script measures real behavior.

Usage:
    PYTHONPATH=".:face_recognition_app" python study/real_arms.py test
    PYTHONPATH=".:face_recognition_app" python study/real_arms.py evaluation
"""

from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from facial_recognition_benchmark import whispers as whispers_mod
from facial_recognition_benchmark.datasets import (
    clustering_scenarios,
    load_manifest,
    recognition_scenarios,
)
from facial_recognition_benchmark.drivers import (
    recognition_expected,
    run_clustering_scenario,
    run_recognition_scenario,
)
from facial_recognition_benchmark.metrics import score_recognition
from facial_recognition_benchmark.plugins import ClusteringBenchmark

OUT_DIR = Path("study/results")
CUTOFF_SWEEP = (0.3, 0.5, 0.7, 0.9)


class EmbeddingCache:
    """Embed every scenario image once; everything downstream reuses this."""

    def __init__(self, model: Any) -> None:
        self.model = model
        self._by_id: Dict[int, np.ndarray] = {}
        self.failures: List[Dict[str, Any]] = []
        self.seconds = 0.0

    def descriptor_for(self, key: int, image: np.ndarray) -> Optional[np.ndarray]:
        if key in self._by_id:
            return self._by_id[key]
        start = time.perf_counter()
        descriptor = self._embed(image)
        self.seconds += time.perf_counter() - start
        if descriptor is None:
            self.failures.append({"key": key, "reason": "no face detected"})
            descriptor = None
        self._by_id[key] = descriptor  # None is cached too: one probe each.
        return descriptor

    def _embed(self, image: np.ndarray) -> Optional[np.ndarray]:
        boxes, probabilities, _landmarks = self.model.detect(image)
        if boxes is None or len(boxes) == 0:
            self.failures.append({"key": self._next_key, "reason": "no detections"})
            return None
        probabilities = np.asarray(probabilities)
        order = np.argsort(probabilities)[::-1]
        best = None
        for index in order:
            if probabilities[index] >= 0.9:
                best = index
                break
        if best is None:
            self.failures.append(
                {
                    "key": self._next_key,
                    "reason": "no face above detection_threshold",
                    "best_probability": round(float(probabilities.max()), 4),
                }
            )
            return None
        descriptors = self.model.compute_descriptors(image, boxes[best : best + 1])
        if descriptors is None or len(descriptors) == 0:
            self.failures.append({"key": self._next_key, "reason": "no descriptors returned"})
            return None
        vector = np.asarray(descriptors[0], dtype=float)
        norm = np.linalg.norm(vector)
        return vector / norm if norm else vector


def _cosine(left: np.ndarray, right: np.ndarray) -> float:
    return float(np.dot(left, right))


class FaceNetReference:
    """Reference recognizer: mean-descriptor prototypes plus a cosine cutoff."""

    def __init__(
        self,
        cache: EmbeddingCache,
        keys_for: Any,
        matching_threshold: float = 0.7,
    ) -> None:
        self.cache = cache
        self.keys_for = keys_for  # image -> stable int key
        self.matching_threshold = matching_threshold
        self.prototypes: Dict[str, List[np.ndarray]] = {}
        self.distances: List[Dict[str, Any]] = []
        self._next_key = 1

    def enroll(self, person_id: str, images: Sequence[np.ndarray]) -> None:
        vectors = []
        for image in images:
            key = self.keys_for(image)
            descriptor = self.cache.descriptor_for(key, image)
            if descriptor is not None:
                vectors.append(descriptor)
        if vectors:
            prototype = np.mean(vectors, axis=0)
            prototype /= np.linalg.norm(prototype)
            self.prototypes.setdefault(person_id, []).append(prototype)

    def recognize(self, images: Sequence[np.ndarray]) -> List[Optional[str]]:
        answers: List[Optional[str]] = []
        for image in images:
            key = self.keys_for(image)
            descriptor = self.cache.descriptor_for(key, image)
            if descriptor is None or not self.prototypes:
                answers.append(None)
                continue
            best_name: Optional[str] = None
            best_similarity = -math.inf
            for name, prototypes in self.prototypes.items():
                for prototype in prototypes:
                    similarity = _cosine(descriptor, prototype)
                    if similarity > best_similarity:
                        best_similarity = similarity
                        best_name = name
            distance = 1.0 - best_similarity
            answers.append(best_name if distance <= self.matching_threshold else None)
            self.distances.append(
                {
                    "named": best_name,
                    "distance": round(distance, 4),
                    "accepted": bool(distance <= self.matching_threshold),
                }
            )
        return answers


class AllNoneAdapter:
    """The recognizer that abstains on everyone, on real images."""

    def __init__(self, *_args: Any, **_kwargs: Any) -> None:
        self.prototypes: Dict[str, List[np.ndarray]] = {}

    def enroll(self, person_id: str, images: Sequence[np.ndarray]) -> None:
        self.prototypes.setdefault(person_id, []).append(np.zeros(512))

    def recognize(self, images: Sequence[np.ndarray]) -> List[Optional[str]]:
        return [None for _ in images]


class PositionalCheaterAdapter:
    """Answers names by batch position; the phase split defeats it."""

    _NAMES = ("alpha", "bravo", "charlie", "delta", "echo", "foxtrot")

    def __init__(self, *_args: Any, **_kwargs: Any) -> None:
        self._seen = 0
        self.prototypes: Dict[str, List[np.ndarray]] = {}

    def enroll(self, person_id: str, images: Sequence[np.ndarray]) -> None:
        self.prototypes.setdefault(person_id, []).append(np.zeros(512))

    def recognize(self, images: Sequence[np.ndarray]) -> List[Optional[str]]:
        answers = []
        for _offset in range(len(images)):
            answers.append(self._NAMES[self._seen % len(self._NAMES)])
            self._seen += 1
        return answers


class OneClusterAdapter:
    """Labels every image with the same cluster id."""

    def __init__(self, *_args: Any, **_kwargs: Any) -> None:
        pass

    def cluster(self, images: Sequence[np.ndarray], seed: int = 0) -> List[int]:
        return [0 for _ in images]


class SingletonAdapter:
    """Gives every image its own cluster."""

    def __init__(self, *_args: Any, **_kwargs: Any) -> None:
        pass

    def cluster(self, images: Sequence[np.ndarray], seed: int = 0) -> List[int]:
        return list(range(len(images)))


class SingleLinkageAdapter:
    """Greedy single-linkage chain over cosine similarity, seeded ordering."""

    def __init__(self, cache: EmbeddingCache, keys_for: Any, threshold: float = 0.5) -> None:
        self.cache = cache
        self.keys_for = keys_for
        self.threshold = threshold

    def cluster(self, images: Sequence[np.ndarray], seed: int = 0) -> List[int]:
        descriptors = []
        for image in images:
            descriptor = self.cache.descriptor_for(self.keys_for(image), image)
            descriptors.append(descriptor)
        labels: List[Optional[int]] = [None] * len(images)
        next_label = 0
        for index in range(len(images)):
            if descriptors[index] is None:
                labels[index] = -1
                continue
            for other in range(index):
                if (
                    labels[other] is not None
                    and labels[other] >= 0
                    and descriptors[other] is not None
                ):
                    if _cosine(descriptors[index], descriptors[other]) >= 1.0 - self.threshold:
                        labels[index] = labels[other]
                        break
            else:
                labels[index] = next_label
                next_label += 1
        return [int(label) for label in labels]


class WhispersAdapter:
    """The reference Whispers clusterer over real embeddings."""

    def __init__(self, cache: EmbeddingCache, keys_for: Any, threshold: float = 0.5) -> None:
        self.cache = cache
        self.keys_for = keys_for
        self.threshold = threshold

    def cluster(self, images: Sequence[np.ndarray], seed: int = 0) -> List[int]:
        descriptors = []
        for image in images:
            descriptor = self.cache.descriptor_for(self.keys_for(image), image)
            if descriptor is None:
                descriptor = np.zeros(512)
            descriptors.append(descriptor)
        # whispers() draws its visit order from the global numpy RNG and takes
        # no seed itself; seeding here is what makes the manifest's repeat
        # seeds meaningful.
        np.random.seed(seed)
        result = whispers_mod.whispers(descriptors, threshold=self.threshold)
        labels = result["labels"] if isinstance(result, dict) else result
        return [int(label) for label in labels]


class ImageKeyer:
    """Maps image objects to stable integer keys by content and arrival."""

    def __init__(self) -> None:
        self._by_object: Dict[int, int] = {}
        self._next = 0

    def __call__(self, image: np.ndarray) -> int:
        obj_id = id(image)
        if obj_id not in self._by_object:
            self._by_object[obj_id] = self._next
            self._next += 1
        return self._by_object[obj_id]


def _recognition_arm(manifest: Dict[str, Any]) -> Dict[str, Any]:
    from facenet_models import FacenetModel

    model = FacenetModel(device="cpu")
    cache = EmbeddingCache(model)
    keyer = ImageKeyer()
    scenarios = recognition_scenarios(manifest)
    expected = [recognition_expected(scenario) for scenario in scenarios]

    arms: Dict[str, Any] = {}

    for cutoff in CUTOFF_SWEEP:
        # A fresh recognizer per scenario: the driver builds one adapter per
        # run, so enrollments from one scenario must not leak into the next.
        # The factory appends each instance to a list so the distance log
        # survives the run.
        recognizers: List[FaceNetReference] = []

        def factory(
            m: Any, c: float = cutoff, _into: List[FaceNetReference] = recognizers
        ) -> FaceNetReference:
            recognizer = FaceNetReference(cache, keyer, matching_threshold=c)
            _into.append(recognizer)
            return recognizer

        outputs = [run_recognition_scenario(factory, model, scenario) for scenario in scenarios]
        scores = score_recognition(outputs, expected)
        arms[f"facenet_cutoff_{cutoff}"] = {
            "known_identification": scores["known_identification"],
            "unknown_rejection_recall": scores["unknown_rejection_recall"],
            "post_enrollment_accuracy": scores["post_enrollment_accuracy"],
            "recognition_score": scores["recognition_score"],
            "distance_log": [entry for recognizer in recognizers for entry in recognizer.distances],
        }

    for name, factory in (
        ("all_none", AllNoneAdapter),
        ("positional_cheater", PositionalCheaterAdapter),
    ):
        outputs = [run_recognition_scenario(factory, model, s) for s in scenarios]
        scores = score_recognition(outputs, expected)
        arms[name] = {
            "known_identification": scores["known_identification"],
            "unknown_rejection_recall": scores["unknown_rejection_recall"],
            "post_enrollment_accuracy": scores["post_enrollment_accuracy"],
            "recognition_score": scores["recognition_score"],
        }

    return {
        "arms": arms,
        "embedding_failures": cache.failures,
        "images_embedded": len(cache._by_id),
        "embedding_seconds": round(cache.seconds, 2),
    }


def _clustering_arm(manifest: Dict[str, Any]) -> Dict[str, Any]:
    from facenet_models import FacenetModel

    model = FacenetModel(device="cpu")
    cache = EmbeddingCache(model)
    keyer = ImageKeyer()
    cases = clustering_scenarios(manifest)
    benchmark = ClusteringBenchmark()

    arms: Dict[str, Any] = {}
    for name, factory in (
        ("whispers_reference", lambda m: WhispersAdapter(cache, keyer)),
        ("single_linkage", lambda m: SingleLinkageAdapter(cache, keyer)),
        ("one_cluster", OneClusterAdapter),
        ("singleton", SingletonAdapter),
    ):
        outputs = [run_clustering_scenario(factory, model, case) for case in cases]
        metrics = benchmark.score(outputs, cases)
        arms[name] = {key: value for key, value in metrics.items() if not key.startswith("_")}

    return {
        "arms": arms,
        "embedding_failures": cache.failures,
        "images_embedded": len(cache._by_id),
        "embedding_seconds": round(cache.seconds, 2),
        "case_count": len(cases),
        "diagnostics": benchmark.last_diagnostics,
    }


def main(tier: str) -> None:
    manifest = load_manifest(tier)
    started = time.perf_counter()
    report = {
        "tier": tier,
        "manifest_id": manifest["manifest_id"],
        "recognition": _recognition_arm(manifest),
        "clustering": _clustering_arm(manifest),
        "total_seconds": round(time.perf_counter() - started, 1),
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / f"real_arms_{tier}.json"
    path.write_text(json.dumps(report, indent=2, default=str))
    summary = {
        "recognition": {
            name: {k: v for k, v in arm.items() if k != "distance_log"}
            for name, arm in report["recognition"]["arms"].items()
        },
        "clustering": report["clustering"]["arms"],
        "failures": report["recognition"]["embedding_failures"],
        "seconds": report["total_seconds"],
    }
    print(json.dumps(summary, indent=2))
    print(f"wrote {path}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "test")
