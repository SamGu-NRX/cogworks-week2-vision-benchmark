"""
Legacy face-detection diagnostics.

Profiles any detector object exposing ``detect(image) ->
(boxes, probabilities, landmarks)`` — the triple produced by the reference
application's MTCNN-based FacenetModel, but no specific model class is
required, so these run against mock detectors in tests. They belong to the
self-serve diagnostic layer (driven by ``benchmark.py:run_benchmark``); the
scored v2 benchmark (``plugins``/``drivers``/``metrics``/``datasets``) does
not import this module.

Coverage:
- Detection accuracy against per-image ground-truth face counts
- False positive rejection on non-face images
- Detection probability threshold effectiveness
- Detection speed performance
"""

import time
from typing import Any, Dict, List, Optional

import numpy as np

from facial_recognition_benchmark.utils import load_image


def benchmark_detection(
    model,
    face_images: List[str],
    non_face_images: Optional[List[str]] = None,
    # ground_truth_boxes removed: it was never read, and its positional slot
    # swallowed ground_truth_num_faces from run_benchmark, leaving accuracy None.
    ground_truth_num_faces: Optional[Dict[str, int]] = None
) -> Dict[str, Any]:
    """
    Benchmark face detection on labeled face and optional non-face images.

    For each face image the file is loaded, one ``model.detect`` call is
    timed with ``time.time``, and the detections are recorded. A face image
    counts as correct only when the detected box count EXACTLY equals the
    expected count in ``ground_truth_num_faces`` for its path; images
    without an entry are skipped from the accuracy fraction entirely, so a
    missing key is not an error — ``detection_accuracy`` simply stays
    ``None`` when no image was countable. Load and detect failures are
    printed to stdout and skipped: that shrinks the sample behind the
    probability and timing averages, while the false-positive rate keeps
    its full denominator.

    Parameters
    ----------
    model : object
        Detector exposing ``detect(image) -> (boxes, probabilities,
        landmarks)``; only ``len(boxes)`` and ``probabilities`` are read.
    face_images : List[str]
        Paths to images expected to contain faces.
    non_face_images : List[str], optional
        Paths to images expected to contain no faces; any image with at
        least one detection is recorded as a false positive.
    ground_truth_num_faces : Dict[str, int], optional
        Maps face-image paths to the expected face count. Pass it as a
        keyword: the fourth positional slot once belonged to a dead
        ``ground_truth_boxes`` parameter that silently swallowed it.
    
    Returns
    -------
    Dict[str, Any]
        ``detection_accuracy`` — exact-count accuracy over images with
        ground truth, or ``None`` when none were countable.
        ``false_positive_rate`` — fraction of non-face images with at
        least one detection, or ``None`` without non-face images.
        ``avg_detection_probability`` — mean over successful face images
        of the mean detection probability, or ``None`` on no successes.
        ``avg_detection_time`` — mean wall-clock seconds per successful
        ``detect`` call on face images, or ``None`` on no successes.
        ``detections`` — per face image: ``image_path``,
        ``num_faces_detected``, ``probabilities`` (list),
        ``detection_time`` (seconds).
        ``false_positives`` — per offending non-face image:
        ``image_path``, ``num_false_positives``, ``probabilities`` (list).
    """
    results = {
        "detection_accuracy": None,
        "false_positive_rate": None,
        "avg_detection_probability": None,
        "avg_detection_time": None,
        "detections": [],
        "false_positives": []
    }
    
    # Test face detection on face images
    correct_detections = 0
    total_detections = 0
    total_probability = 0.0
    detection_times = []
    
    for image_path in face_images:
        try:
            image = load_image(image_path)
            
            # Measure detection time
            start_time = time.time()
            boxes, probabilities, landmarks = model.detect(image)
            detection_time = time.time() - start_time
            detection_times.append(detection_time)
            
            num_detected = len(boxes) if boxes is not None else 0
            
            # Check against ground truth if provided
            if ground_truth_num_faces and image_path in ground_truth_num_faces:
                expected = ground_truth_num_faces[image_path]
                if num_detected == expected:
                    correct_detections += 1
                total_detections += 1
            
            # Record detection results
            detection_result = {
                "image_path": image_path,
                "num_faces_detected": num_detected,
                "probabilities": probabilities.tolist() if probabilities is not None else [],
                "detection_time": detection_time
            }
            results["detections"].append(detection_result)
            
            # Sum probabilities for average
            if probabilities is not None and len(probabilities) > 0:
                total_probability += np.mean(probabilities)
                
        except Exception as e:
            print(f"Error processing {image_path}: {e}")
            continue
    
    # Calculate accuracy if ground truth provided
    if total_detections > 0:
        results["detection_accuracy"] = correct_detections / total_detections
    
    # Calculate average detection probability
    if results["detections"]:
        results["avg_detection_probability"] = total_probability / len(results["detections"])
    
    # Calculate average detection time
    if detection_times:
        results["avg_detection_time"] = np.mean(detection_times)
    
    # Test false positive rejection on non-face images
    if non_face_images:
        false_positive_count = 0
        total_non_face_images = len(non_face_images)
        
        for image_path in non_face_images:
            try:
                image = load_image(image_path)
                boxes, probabilities, landmarks = model.detect(image)
                
                num_detected = len(boxes) if boxes is not None else 0
                
                if num_detected > 0:
                    false_positive_count += 1
                    
                    # Record false positive details
                    fp_result = {
                        "image_path": image_path,
                        "num_false_positives": num_detected,
                        "probabilities": probabilities.tolist() if probabilities is not None else []
                    }
                    results["false_positives"].append(fp_result)
                    
            except Exception as e:
                print(f"Error processing {image_path}: {e}")
                continue
        
        results["false_positive_rate"] = (
            false_positive_count / total_non_face_images if total_non_face_images > 0 else 0.0
        )
    
    return results


def benchmark_detection_probability_threshold(
    model,
    face_images: List[str],
    non_face_images: List[str],
    thresholds: List[float] = None
) -> Dict[str, Any]:
    """
    Sweep detection-probability thresholds and pick the best by F1.

    Counting is image-level, not box-level: a face image contributes one
    true positive when ANY of its detection probabilities reaches the
    threshold, and a non-face image contributes one false positive under
    the same rule. Rates divide by the number of images SUPPLIED, so an
    image that failed to load still counts in the denominator. The default
    sweep is ``np.arange(0.5, 1.0, 0.05)`` — 0.50 to 0.95 in steps of 0.05.

    Note: the "precision" used here is ``1 - false_positive_rate``, which
    is specificity, not classification precision, so the reported F1 and
    the ``optimal_threshold`` derived from it are optimistic. The matching
    distance-space sweep in ``distance.py`` was corrected to real
    precision; this detection-side variant was left as-is to keep its
    returned values stable.
    
    Parameters
    ----------
    model : object
        Detector exposing ``detect(image) -> (boxes, probabilities,
        landmarks)``; only ``probabilities`` are read.
    face_images : List[str]
        Paths to images expected to contain faces.
    non_face_images : List[str]
        Paths to images expected to contain no faces.
    thresholds : List[float], optional
        Detection-probability cutoffs to test; defaults to 0.50..0.95 in
        steps of 0.05.
    
    Returns
    -------
    Dict[str, Any]
        ``thresholds`` — the cutoffs actually swept (echoed back).
        ``face_detection_rates`` — per threshold, fraction of face images
        with at least one detection at or above it.
        ``false_positive_rates`` — same rule on the non-face images.
        ``optimal_threshold`` — cutoff with the highest F1 (the first one
        wins ties), or ``None`` when nothing was swept.
        ``optimal_f1_score`` — that best F1 (0.0 when nothing was swept).
    """
    if thresholds is None:
        thresholds = np.arange(0.5, 1.0, 0.05).tolist()
    
    results = {
        "thresholds": thresholds,
        "face_detection_rates": [],
        "false_positive_rates": [],
        "optimal_threshold": None,
        "optimal_f1_score": 0.0
    }
    
    face_detection_rates = []
    false_positive_rates = []
    f1_scores = []
    
    for threshold in thresholds:
        # Count true positives (faces detected above threshold)
        true_positives = 0
        for image_path in face_images:
            try:
                image = load_image(image_path)
                boxes, probabilities, landmarks = model.detect(image)
                
                if probabilities is not None:
                    # Count faces detected with probability >= threshold
                    detected_above_threshold = np.sum(probabilities >= threshold)
                    if detected_above_threshold > 0:
                        true_positives += 1
            except Exception:
                continue
        
        # Count false positives (non-faces detected above threshold)
        false_positives = 0
        for image_path in non_face_images:
            try:
                image = load_image(image_path)
                boxes, probabilities, landmarks = model.detect(image)
                
                if probabilities is not None:
                    detected_above_threshold = np.sum(probabilities >= threshold)
                    if detected_above_threshold > 0:
                        false_positives += 1
            except Exception:
                continue
        
        # Calculate rates
        face_detection_rate = true_positives / len(face_images) if face_images else 0.0
        false_positive_rate = false_positives / len(non_face_images) if non_face_images else 0.0
        
        face_detection_rates.append(face_detection_rate)
        false_positive_rates.append(false_positive_rate)
        
        # Calculate F1 score (balance between precision and recall)
        precision = 1 - false_positive_rate
        recall = face_detection_rate
        if precision + recall > 0:
            f1 = 2 * (precision * recall) / (precision + recall)
        else:
            f1 = 0.0
        f1_scores.append(f1)
    
    results["face_detection_rates"] = face_detection_rates
    results["false_positive_rates"] = false_positive_rates
    
    # Find optimal threshold (highest F1 score)
    if f1_scores:
        optimal_idx = np.argmax(f1_scores)
        results["optimal_threshold"] = thresholds[optimal_idx]
        results["optimal_f1_score"] = f1_scores[optimal_idx]
    
    return results


def benchmark_detection_consistency(
    model,
    image_paths: List[str],
    num_runs: int = 5
) -> Dict[str, Any]:
    """
    Measure run-to-run stability of the detector's face counts.

    Each image is loaded once, then ``model.detect`` runs ``num_runs``
    times on it. The per-image score is the variance of the detected face
    counts across runs (0.0 for a deterministic detector), so one flaky
    image is still visible in ``max_variance`` when the average looks
    fine. Failed images are printed to stdout and skipped.
    
    Parameters
    ----------
    model : object
        Detector exposing ``detect(image) -> (boxes, probabilities,
        landmarks)``; only ``len(boxes)`` is read.
    image_paths : List[str]
        Paths to the images to probe repeatedly.
    num_runs : int
        Detection calls per image (default 5).
    
    Returns
    -------
    Dict[str, Any]
        ``consistency_scores`` — per image: ``image_path``,
        ``face_counts`` (one entry per run), ``variance`` of the counts.
        ``avg_variance`` — mean per-image variance (0.0 when nothing ran).
        ``max_variance`` — worst per-image variance (0.0 when nothing ran).
    """
    results = {
        "consistency_scores": [],
        "avg_variance": 0.0,
        "max_variance": 0.0
    }
    
    variances = []
    
    for image_path in image_paths:
        try:
            image = load_image(image_path)
            face_counts = []
            
            for _ in range(num_runs):
                boxes, probabilities, landmarks = model.detect(image)
                num_faces = len(boxes) if boxes is not None else 0
                face_counts.append(num_faces)
            
            # Calculate variance in detected face counts
            variance = np.var(face_counts)
            variances.append(variance)
            
            results["consistency_scores"].append({
                "image_path": image_path,
                "face_counts": face_counts,
                "variance": variance
            })
            
        except Exception as e:
            print(f"Error processing {image_path}: {e}")
            continue
    
    if variances:
        results["avg_variance"] = np.mean(variances)
        results["max_variance"] = np.max(variances)
    
    return results
