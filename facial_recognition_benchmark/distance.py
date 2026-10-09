"""
Cosine Distance Calculation Benchmark Module

Exercises the reference implementations in ``utils.py``
(``cosine_distance``, ``cosine_distance_matrix``) for correctness,
throughput, and match-threshold selection. Part of the self-serve
diagnostic layer; the scored v2 benchmark
(``plugins``/``drivers``/``metrics``/``datasets``) does not import this
module.

Reference cosine behavior, verified at this commit: identical unit vectors
give 0.0 (within ~1e-16), orthogonal 1.0, opposite 2.0; a zero-norm vector
is guarded and yields 1.0 (in ``cosine_distance`` and per-row in
``cosine_distance_matrix``) instead of NaN.
"""

import time
from typing import Any, Dict, List

import numpy as np

from facial_recognition_benchmark.utils import cosine_distance, cosine_distance_matrix


def benchmark_cosine_distance_accuracy(
    num_test_vectors: int = 100,
    dimension: int = 512
) -> Dict[str, Any]:
    """
    Check ``utils.cosine_distance`` against known cosine geometry.

    Runs up to 10 cases per group regardless of ``num_test_vectors`` (the
    count is capped, not scaled): identical unit vectors must give 0,
    orthogonal basis vectors 1, opposite unit vectors 2, and a unit vector
    perturbed with 0.01 noise must land in [0, 0.5]. Exact groups use a
    1e-6 tolerance; identical vectors measure ~1e-16 rather than exactly 0
    because the reference computes ``1 - dot/(norm*norm)`` in floating
    point. Vectors come from unseeded ``np.random.randn``, so ``actual``
    values vary run to run while pass/fail stays stable.
    
    Parameters
    ----------
    num_test_vectors : int
        Cap on cases per test group: each group runs min(10,
        num_test_vectors) cases, so values above 10 change nothing.
    dimension : int
        Length of the synthetic test vectors.
    
    Returns
    -------
    Dict[str, Any]
        ``test_cases`` — per case: ``test`` name, ``expected`` (the three
        exact groups) or ``expected_range`` (similar group), ``actual``,
        ``error`` (exact groups only), ``passed``.
        ``max_error`` / ``avg_error`` — worst and mean error across the
        three exact groups; the similar group reports no error.
        ``all_passed`` — False if any case failed.
    """
    results = {
        "test_cases": [],
        "max_error": 0.0,
        "avg_error": 0.0,
        "all_passed": True
    }
    
    errors = []
    
    # Test 1: Identical vectors should have distance 0
    for i in range(min(10, num_test_vectors)):
        vector = np.random.randn(dimension)
        vector = vector / np.linalg.norm(vector)
        
        distance = cosine_distance(vector, vector)
        error = abs(distance - 0.0)
        errors.append(error)
        
        results["test_cases"].append({
            "test": "identical_vectors",
            "expected": 0.0,
            "actual": distance,
            "error": error,
            "passed": error < 1e-6
        })
        
        if error >= 1e-6:
            results["all_passed"] = False
    
    # Test 2: Orthogonal vectors should have distance 1
    for i in range(min(10, num_test_vectors)):
        # Create orthogonal vectors
        v1 = np.zeros(dimension)
        v1[0] = 1.0
        
        v2 = np.zeros(dimension)
        v2[1] = 1.0
        
        distance = cosine_distance(v1, v2)
        error = abs(distance - 1.0)
        errors.append(error)
        
        results["test_cases"].append({
            "test": "orthogonal_vectors",
            "expected": 1.0,
            "actual": distance,
            "error": error,
            "passed": error < 1e-6
        })
        
        if error >= 1e-6:
            results["all_passed"] = False
    
    # Test 3: Opposite vectors should have distance 2
    for i in range(min(10, num_test_vectors)):
        vector = np.random.randn(dimension)
        vector = vector / np.linalg.norm(vector)
        opposite_vector = -vector
        
        distance = cosine_distance(vector, opposite_vector)
        error = abs(distance - 2.0)
        errors.append(error)
        
        results["test_cases"].append({
            "test": "opposite_vectors",
            "expected": 2.0,
            "actual": distance,
            "error": error,
            "passed": error < 1e-6
        })
        
        if error >= 1e-6:
            results["all_passed"] = False
    
    # Test 4: Similar vectors should have small distance
    for i in range(min(10, num_test_vectors)):
        base_vector = np.random.randn(dimension)
        base_vector = base_vector / np.linalg.norm(base_vector)
        
        # Add small noise
        noise = np.random.randn(dimension) * 0.01
        similar_vector = base_vector + noise
        similar_vector = similar_vector / np.linalg.norm(similar_vector)
        
        distance = cosine_distance(base_vector, similar_vector)
        
        results["test_cases"].append({
            "test": "similar_vectors",
            "expected_range": (0.0, 0.5),
            "actual": distance,
            "passed": 0.0 <= distance <= 0.5
        })
        
        if not (0.0 <= distance <= 0.5):
            results["all_passed"] = False
    
    # Calculate error statistics
    if errors:
        results["max_error"] = float(np.max(errors))
        results["avg_error"] = float(np.mean(errors))
    
    return results


def benchmark_cosine_distance_performance(
    vector_sizes: List[int] = None,
    num_test_vectors: int = 100,
    dimension: int = 512,
    num_runs: int = 5
) -> Dict[str, Any]:
    """
    Time pairwise-loop vs matrix cosine distance on synthetic descriptors.

    For each set size, draws ``size`` unseeded random vectors (rows of a
    ``size x dimension`` matrix) and times ``num_runs`` rounds of (a) a
    naive double loop calling ``cosine_distance`` on every pair and (b) one
    ``cosine_distance_matrix`` call, reporting mean times and throughputs
    in distances per second (pairwise uses n*(n-1)/2 pairs, matrix n*n).
    Timing uses ``time.time`` and the data is not seeded, so only the broad
    loop-vs-matrix ratio is meaningful across machines.
    
    Parameters
    ----------
    vector_sizes : List[int], optional
        Set sizes to sweep; defaults to [10, 50, 100, 500, 1000]. The
        naive double loop makes large sizes slow (quadratic in ``size``).
    num_test_vectors : int
        ACCEPTED BUT IGNORED — the body never reads it (each set contains
        ``size`` vectors). Kept so existing call sites keep working.
    dimension : int
        Descriptor length of the synthetic vectors.
    num_runs : int
        Timing rounds averaged per set size.
    
    Returns
    -------
    Dict[str, Any]
        ``vector_sizes`` — echoed list of set sizes.
        ``pairwise_times`` / ``matrix_times`` — mean seconds per round.
        ``pairwise_throughput`` / ``matrix_throughput`` — distances per
        second, one entry per set size.
    """
    if vector_sizes is None:
        vector_sizes = [10, 50, 100, 500, 1000]
    
    results = {
        "vector_sizes": vector_sizes,
        "pairwise_times": [],
        "matrix_times": [],
        "pairwise_throughput": [],
        "matrix_throughput": []
    }
    
    for size in vector_sizes:
        # Generate test vectors
        vectors = np.random.randn(size, dimension)
        vectors = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
        
        # Benchmark pairwise distance calculation
        pairwise_times = []
        for _ in range(num_runs):
            start_time = time.time()
            for i in range(size):
                for j in range(i + 1, size):
                    cosine_distance(vectors[i], vectors[j])
            pairwise_times.append(time.time() - start_time)
        
        avg_pairwise_time = np.mean(pairwise_times)
        results["pairwise_times"].append(avg_pairwise_time)
        
        # Calculate pairwise throughput (distances per second)
        num_pairs = size * (size - 1) // 2
        pairwise_throughput = num_pairs / avg_pairwise_time if avg_pairwise_time > 0 else 0
        results["pairwise_throughput"].append(pairwise_throughput)
        
        # Benchmark matrix distance calculation
        matrix_times = []
        for _ in range(num_runs):
            start_time = time.time()
            cosine_distance_matrix(vectors, vectors)
            matrix_times.append(time.time() - start_time)
        
        avg_matrix_time = np.mean(matrix_times)
        results["matrix_times"].append(avg_matrix_time)
        
        # Calculate matrix throughput (distances per second)
        matrix_throughput = (size * size) / avg_matrix_time if avg_matrix_time > 0 else 0
        results["matrix_throughput"].append(matrix_throughput)
    
    return results


def benchmark_cosine_distance_threshold(
    same_person_distances: List[float],
    different_person_distances: List[float],
    thresholds: List[float] = None
) -> Dict[str, Any]:
    """
    Pick the cosine-distance cutoff that best separates same-person from
    different-person pairs.

    A pair matches when its distance is <= the threshold. Per threshold,
    the true-positive rate is the fraction of ``same_person_distances``
    that matches and the false-positive rate the fraction of
    ``different_person_distances``; an empty list contributes 0.0 (the
    counters are pre-initialized, so empty inputs are safe). F1 uses real
    precision, TP / (TP + FP) — an earlier version used 1 - fpr
    (specificity) and overstated F1. The default sweep is
    ``np.arange(0.0, 1.0, 0.01)`` — 0.00 to 0.99 in steps of 0.01.
    
    Parameters
    ----------
    same_person_distances : List[float]
        Descriptor distances between pairs of the same person.
    different_person_distances : List[float]
        Descriptor distances between pairs of different people.
    thresholds : List[float], optional
        Cutoffs to test; defaults to 0.00..0.99 in steps of 0.01.
    
    Returns
    -------
    Dict[str, Any]
        ``thresholds`` — echoed cutoffs.
        ``true_positive_rates`` / ``false_positive_rates`` /
        ``f1_scores`` — one entry per cutoff, same order.
        ``optimal_threshold`` — first cutoff with the highest F1, or
        ``None`` when no cutoffs were given.
        ``optimal_f1_score`` / ``optimal_tpr`` / ``optimal_fpr`` — values
        at that cutoff; on an all-zero F1 sweep the first cutoff wins with
        F1 0.0 (e.g. empty inputs).
    """
    if thresholds is None:
        thresholds = np.arange(0.0, 1.0, 0.01).tolist()
    
    results = {
        "thresholds": thresholds,
        "true_positive_rates": [],
        "false_positive_rates": [],
        "f1_scores": [],
        "optimal_threshold": None,
        "optimal_f1_score": 0.0,
        "optimal_tpr": 0.0,
        "optimal_fpr": 0.0
    }
    
    tpr_list = []
    fpr_list = []
    f1_list = []
    
    for threshold in thresholds:
        # Initialize before the branches: with an empty input list the
        # branch never assigned these, and the read below raised
        # UnboundLocalError. (Only the two counters read below are kept;
        # the true/false-negative twins were never read anywhere.)
        true_positives = 0
        false_positives = 0

        # True positive rate: same-person pairs correctly matched
        if same_person_distances:
            true_positives = sum(1 for d in same_person_distances if d <= threshold)
            tpr = true_positives / len(same_person_distances)
        else:
            tpr = 0.0

        # False positive rate: different-person pairs incorrectly matched
        if different_person_distances:
            false_positives = sum(1 for d in different_person_distances if d <= threshold)
            fpr = false_positives / len(different_person_distances)
        else:
            fpr = 0.0

        # Calculate F1 score
        # The old value here was 1 - fpr, which is specificity, not precision.
        # Real precision is TP / (TP + FP), so f1_scores and the optimal
        # threshold derived from them now reflect actual F1.
        if (true_positives + false_positives) > 0:
            precision = true_positives / (true_positives + false_positives)
        else:
            precision = 0.0
        recall = tpr
        if precision + recall > 0:
            f1 = 2 * (precision * recall) / (precision + recall)
        else:
            f1 = 0.0
        
        tpr_list.append(tpr)
        fpr_list.append(fpr)
        f1_list.append(f1)
    
    results["true_positive_rates"] = tpr_list
    results["false_positive_rates"] = fpr_list
    results["f1_scores"] = f1_list
    
    # Find optimal threshold (highest F1 score)
    if f1_list:
        optimal_idx = np.argmax(f1_list)
        results["optimal_threshold"] = thresholds[optimal_idx]
        results["optimal_f1_score"] = f1_list[optimal_idx]
        results["optimal_tpr"] = tpr_list[optimal_idx]
        results["optimal_fpr"] = fpr_list[optimal_idx]
    
    return results
