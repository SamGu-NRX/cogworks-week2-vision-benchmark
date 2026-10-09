# Vision Benchmark Metric-Validity Audit

**Branch:** `obv/products-vision-validity-20261009` (based on `origin/codex/ci-global-database-isolation`, head `a3dd948` plus two study commits)
**Date:** 2026-10-09
**Scope:** `facial_recognition_benchmark` scoring and execution paths, the shipped public manifests, and the reference application's model stack.
**Deliverables:** `tests/test_metric_validity.py` (65 hermetic tests), `study/real_arms.py` (real-image arms), `study/results/real_arms_{test,evaluation}.json` (raw numbers).

---

## 1. Verdict

The scorer's semantics are sound where it matters most: macro-averaging, cluster-label invariance, chance correction, and malformed-output containment all behave exactly as documented, and the new harness now pins each of them to hand-calculated oracles. The real risks this audit found are **not in the formulas** — they are in what the formulas are fed and what happens around them:

1. **The abstain-on-everything submission scores 0.25, not 0.0**, and the all-one-cluster clustering submission scores ~0.37–0.43 pairwise F1. Both floors are *correct consequences of the metric definitions* (verified by hand and reproduced on real FaceNet embeddings), but nothing in the docs tells a team the floor exists. ARI is the antidote on the clustering side — it scored the same submission exactly 0.0 on both real tiers.
2. **One matching cutoff cannot serve both sides of the recognition metric, and the optimum is tier-dependent.** On real FaceNet embeddings, cutoff 0.7 maximizes the test tier (0.917) while 0.5 maximizes the evaluation tier (0.975). This is the single most consequential finding for anyone tuning a submission.
3. **Degenerate embeddings break recognition from both directions at once.** Three real query images produced descriptors at cosine distance 0.83–1.06 from *every* prototype: one known person's query becomes unmatched at any cutoff ≤ 0.9, and the unknown person's garbage descriptor masquerades as a known face at loose cutoffs. The metric's phase separation is what makes this readable; a single pooled accuracy number would have buried it.
4. **The disjointness guard is not wired into the load path.** `assert_disjoint` is only called from tests; `load_manifest` accepts two overlapping manifests without complaint. Every guarantee the harness pins is procedural, not structural.
5. **The "opaque" identity hash is not opaque.** `source_identity_sha256 = sha256(str(celeb_id))` over a ~10,000-value space is reversible by brute force in seconds (demonstrated offline: all 20 recorded hashes were recovered). Combined with clear-text labels in the public evaluation manifest and `decoded_pixel_sha256` keys, a submission can table-lookup its way to a perfect evaluation-tier score.

Recommendations follow each finding in §5. The official (hidden) track's separation from the public tiers still rests on the staff-side disjointness promise and cannot be verified from this repository.

---

## 2. Method

Three layers, each with a different trust level:

| Layer | What it covers | Trust |
|---|---|---|
| **Hermetic harness** (`tests/test_metric_validity.py`) | Scoring semantics against hand-calculated oracles; adapter lifecycle witnesses; manifest guard behavior. No network, no model, deterministic. | Exact — every expected value computed by hand and explained in a comment |
| **Real-image arms** (`study/real_arms.py`) | Reference FaceNet recognizer (MTCNN + resnet, checkpoint SHA-256-verified against `model-lock.json`) and reference Whispers on materialized CelebA rows from the pinned revision (`2d738f56…`, split `valid`). | Measured — single run, CPU, timings recorded |
| **Static audits** (manifest hashes, test inventory, code reading) | Cross-tier leakage, hash reversibility, guard wiring, denominator choices. | Read-only evidence, cited by file and line |

Environment notes: the sandbox runs Python 3.13.14 against a repo declaring `<3.13`, and CPU PyTorch 2.14.1 against a lock pinning 2.2.2; imports and tests ran via `PYTHONPATH` without editable install. All metric conclusions are version-independent (pure Python arithmetic), but the real-arm embeddings come from the newer torch build — treat absolute distances as indicative, not certified.

---

## 3. What the harness pins (all 65 passing, ruff-clean)

**Recognition scoring (`score_recognition`)** — macro (per-identity) averaging, not micro: a fixture where person "a" answers 1 of 4 known queries scores exactly (0 + 2/3)/2 = 1/3, while per-image pooling would give 1/2 and mixed pooling 3/4. Abstention on a known person counts as wrong but stays in the denominator. Malformed sections — wrong lengths, `None` in expected positions — zero the whole case list rather than shrinking denominators, and the zero-propagation boundary between recognition (case-list) and clustering (per-case) behavior is pinned on both sides.

**Diagnostics** — the three post-enrollment failure readings route to the right repair: all-abstained → cutoff message; mostly-abstained → strictness message; misnamed → descriptors/averaging message. A perfect-knowns/never-rejects recognizer (composite 0.75 on the real manifest arm) triggers the "cutoff never fires" note. The strings are pinned because a regression that swaps their causes sends teams to the wrong knob.

**Clustering** — pairwise F1 is label-invariant (verified exhaustively across **all 52 partitions of 5 elements** — every relabeling of every partition scores identically). ARI cross-checks against scikit-learn when that package is installed (skipped otherwise; it is not a repo dependency). Malformed cases score zero for that case only. Degenerate submissions: all-one-cluster and singleton baselines are pinned at exact values.

**Seed-sweep accounting** — repeats move `clustering_seed_spread` and never the published F1; a keyless repeat attaches to the latest scored case; orphans group together; the reported spread is the worst scenario, not the pooled one; the loud (≥ 0.15) and quiet (≤ 0.02) thresholds fire on the right side of both boundaries.

**Lifecycle drivers** — query phases split each person's held-out photos across the two `recognize` calls, so a position-based cheater cannot score full marks (0.0 on both synthetic and real data), a forgetter scores exactly half on knowns, and a content-keyed recognizer scores full marks. Returns are length- and type-validated with the exact error strings pinned.

**Manifest guard** — `assert_disjoint` refuses shared `(split, row_index)` and shared identity hashes across manifests, treats different splits as different rows, and accepts the shipped test/evaluation pair.

---

## 4. Real-image arms (materialized CelebA, verified checkpoint)

Every embedding pass on both tiers completed with **zero detection failures** (per-arm counts in the raw JSONs: 20 and 72 images for recognition, 12 and 32 for clustering — the recognition and clustering arms embed independently, so scenario images shared across arms embed twice). Recognition-arm embedding took 1.2 s (test) / 3.8 s (evaluation) on CPU. Recognition uses mean-descriptor prototypes and a cosine cutoff; clustering uses the reference Whispers over the same embeddings, with the manifest's seed sweep.

### 4.1 Recognition — the cutoff sweep

| Cutoff | test known / reject / post → **score** | evaluation known / reject / post → **score** |
|---|---|---|
| 0.3 | 0.833 / 1.00 / 0.00 → **0.667** | 0.800 / 1.00 / 0.80 → **0.850** |
| 0.5 | 0.833 / 1.00 / 0.00 → **0.667** | 0.950 / 1.00 / 1.00 → **0.975** |
| 0.7 | 0.833 / 1.00 / 1.00 → **0.917** | 1.000 / 0.833 / 1.00 → **0.958** |
| 0.9 | 0.833 / 0.50 / 1.00 → **0.792** | 1.000 / 0.167 / 1.00 → **0.792** |

- The optimal cutoff **differs by tier**: 0.7 on test, 0.5 on evaluation. A team tuning on the small tier is tuning the wrong knob for the large one, and the metric exposes the tension cleanly — tightening the cutoff trades post-enrollment recall for stranger rejection through the same number.
- Same-person distances for multi-photo prototypes sit at **0.047–0.128**; queries for a person enrolled from a single photo sit at **0.58–0.63**. This reproduces, from scratch, the exact situation the `metrics.py` docstring documents for a real submission (cutoff 0.2846, distance 0.63): the one-photo enrollment is a calibration problem, not a broken recognizer.
- Baselines: abstain-on-everything scores **exactly 0.25 on both tiers** — (0 + (1.0 + 0.0)/2)/2 — matching the synthetic harness's prediction to the digit. The position-based cheater scores 0.0 on both tiers.

### 4.2 Degenerate embeddings — one cause, two symptoms

The test tier's distance log contains three descriptors at cosine distance **0.83, 0.91, and 1.06** from every prototype (all other accepted queries: 0.047–0.13):

- One known person's query (`person_0_2`, distance 1.06) can never match at any cutoff ≤ 0.9 → that person's known-identification is capped at 0.5 (macro 5/6 = 0.833) **at every cutoff in the sweep**.
- The unknown person's query (distance 0.83) is correctly rejected at cutoffs ≤ 0.7 but **accepted at 0.9** → rejection recall collapses from 1.0 to 0.5.

Detection never failed — MTCNN reported faces with high confidence — so these are near-orthogonal descriptors from the embedding side, not missed faces. The benchmark cannot prevent degenerate embeddings, but the metric machinery (per-phase scoring, separated abstention vs. misnaming, diagnostic strings) is what turns "0.833 and a hunch" into a diagnosis. Worth documenting for students: a distance stuck above 0.8 across all queries for one identity is an embedding problem, not a threshold problem.

### 4.3 Clustering on real embeddings

| Submission | test F1 / ARI / spread | evaluation F1 / ARI / spread |
|---|---|---|
| Reference Whispers (τ=0.5) | 0.909 / 0.879 / 0.000 | 0.851 / 0.816 / 0.067 |
| Greedy single-linkage (τ=0.5) | 0.909 / 0.879 / 0.000 | **0.968 / 0.959 / 0.000** |
| All-one-cluster | 0.429 / **0.000** / — | 0.368 / **0.000** / — |
| Singleton | 0.000 / 0.000 / — | 0.000 / 0.000 / — |

- **Single-linkage dominates the reference Whispers on the evaluation tier** (0.968 vs 0.851 F1) while being perfectly seed-stable; Whispers pays a 0.067 spread (below the 0.15 loud threshold, above the 0.02 quiet one) for a worse partition. On real FaceNet embeddings, chaining is a strong baseline, and the seed sweep is measuring real instability rather than noise.
- **ARI is doing real work.** The all-one-cluster submission earns 0.37–0.43 pairwise F1 — a plausible-looking number — and exactly 0.0 ARI on both tiers. The chance correction is not decorative; F1 alone would flatter a do-nothing submission by forty points.

---

## 5. Findings and recommendations

**F1 — Wire the disjointness guard into the load path.** `assert_disjoint` exists and is correct; nothing calls it when manifests are loaded. `load_manifest` (or the benchmark's case-loading entry point) should run it over the manifest set it opens. *Cost:* a behavior change at load time — overlapping manifests become errors instead of silent coexistence. *Risk if unwired:* every cross-tier guarantee is procedural and one packaging mistake away from vanishing.

**F2 — Salt or abandon the identity hash.** `sha256(str(celeb_id))` over ~10,000 values is a lookup table away from clear text (demonstrated: all 20 shipped hashes recovered offline). Salt per manifest revision, or hash the image bytes and rely on `pixel_sha256` alone. Until then, treat `source_identity_sha256` as clear text in any threat model. *Cost:* a manifest schema-version bump and regeneration. *Reverses if:* the hash is only ever compared, never published — but it ships in the public manifests today.

**F3 — Document the score floor.** Abstain-on-everything = 0.25 composite recognition; all-one-cluster ≈ 0.4 pairwise F1. One paragraph in the benchmark README ("your floor is not zero; here is why and what ARI is for") converts both from confusion traps into calibration anchors. *Cost:* a paragraph. *Reverses if:* the composite formula changes.

**F4 — State the official-track dependency.** The public tiers are verifiably disjoint (rows 0–163 vs 184–490; zero pixel/identity overlap; verified from shipped hashes). The hidden tier's disjointness from both rests on the staff-side promise; the README should say so explicitly. Also note the intentional intra-tier reuse (21 of 32 evaluation clustering images also appear in recognition scenarios) so no one reads it as leakage.

**F5 — The public evaluation tier is a table-lookup surface.** Clear-text labels plus `decoded_pixel_sha256` keys in a wheel-shipped manifest let a submission key predictions off pixel hashes and score perfectly. If the evaluation tier is meant to rank, the wheel should ship test-tier labels only. *Cost:* packaging change; evaluation labels move to the hosted lane. *Reverses if:* the tier is explicitly formative, not ranking.

**F6 — Record the single-linkage observation** (F1 0.968, spread 0.0, evaluation tier) somewhere students can find it. It is a useful baseline, and Whispers' 0.067 spread against it is exactly the kind of honest comparison the benchmark exists to teach. No code change implied.

---

## 6. Reproduction

```bash
# Hermetic harness (no network, no model):
PYTHONPATH=".:face_recognition_app" python -m pytest tests/test_metric_validity.py -q
#   → 65 passed

# Full suite as baselined:
PYTHONPATH=".:face_recognition_app" python -m pytest tests/ -q \
  --ignore=tests/test_discovered_chain.py --ignore=tests/test_discovered_recognition.py
#   → 142 passed, 5 skipped (the two ignored files need the internal cogbench SDK)

# Real-image arms (network for CelebA rows; checkpoint verified against model-lock.json):
PYTHONPATH=".:face_recognition_app" python study/real_arms.py test
PYTHONPATH=".:face_recognition_app" python study/real_arms.py evaluation
```

Raw numbers: `study/results/real_arms_test.json`, `study/results/real_arms_evaluation.json` (per-cutoff distance logs included).

## 7. Limitations

- Single-run real-arm measurements on CPU with torch 2.14.1 against a lock pinning 2.2.2; relative orderings (cutoff trade-offs, baseline gaps) are robust, absolute distances are indicative.
- The two discovered-path test files are excluded from collection by the missing internal `cogbench` SDK — their behavior is outside this audit's scope.
- The `whispers()` reference takes its visit order from the global numpy RNG and no seed parameter; the real-arm adapter seeds the global RNG per repeat, which makes the sweep meaningful but is a side effect worth centralizing in the reference itself.
