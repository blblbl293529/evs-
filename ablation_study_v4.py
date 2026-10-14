#!/usr/bin/env python3
"""
Fusion-v4 ablation study.

This script DOES NOT modify fusion_v4.py or comparison_protocol.py.

Ablations:
    1. Full V4
    2. w/o Odd Parity
    3. w/o Even Parity
    4. w/o Valley
    5. w/o Poisson Normalization
    6. w/o Information Support
    7. Single Scale (delta=5)
    8. w/o Mesoscopic = Local + Global
    9. w/o Global CP = Local + Mesoscopic
   10. Equal Scale Fusion
   11. w/o Shared Global Reference
   12. Local Only

Fairness rules:
    - same comparison_protocol
    - same raw events
    - same official ROI / full-sensor rule
    - same 1 ms aggregation
    - same acquisition interval
    - same candidate boundary as Full V4
    - predictions are produced before GT is loaded
    - all fixed Fusion-v4 coefficients remain frozen
    - no dataset identity or GT enters prediction

Expected location:
    C:\\Users\\24686\\Desktop\\event focus\\fusion_v4\\Ablation study V4\\ablation_study_v4.py
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scipy.special import log_ndtr


# ============================================================================
# Import parent Fusion-v4 folder
# ============================================================================

SCRIPT_DIR = Path(__file__).resolve().parent
FUSION_DIR = SCRIPT_DIR.parent
PROJECT_ROOT = FUSION_DIR.parent

if str(FUSION_DIR) not in sys.path:
    sys.path.insert(0, str(FUSION_DIR))

import comparison_protocol as protocol
import fusion_v4 as fv4


# ============================================================================
# Constants
# ============================================================================

EPS = 1e-12

# Keep the SAME candidate boundary as the original Full V4 for every ablation.
REFERENCE_OFFSETS = tuple(int(x) for x in fv4.DEFAULT_OFFSETS)
REFERENCE_RADIUS = int(max(REFERENCE_OFFSETS))

DEFAULT_OUTPUT = SCRIPT_DIR / "results"

TIMESTAMP_DATASETS = {"SYN", "DAVIS", "EVK4"}
POSITION_DATASETS = {"EAD", "EvHF"}


# ============================================================================
# Ablation configuration
# ============================================================================

@dataclass(frozen=True)
class AblationConfig:
    name: str

    # physical residuals
    use_even: bool = True
    use_odd: bool = True
    use_valley: bool = True

    # statistical model
    use_poisson: bool = True
    use_information: bool = True

    # symmetric offsets used to form parity evidence
    offsets: tuple[int, ...] = REFERENCE_OFFSETS

    # scale branches
    use_mesoscopic: bool = True
    use_global_cp: bool = True

    # scale fusion
    equal_scale_fusion: bool = False

    # whether global symmetry/rate are reused as reference evidence
    # inside the local and mesoscopic branches
    use_shared_global_reference: bool = True


def make_variants() -> list[AblationConfig]:
    return [
        AblationConfig(
            name="Full V4",
        ),

        AblationConfig(
            name="w/o Odd Parity",
            use_odd=False,
        ),

        AblationConfig(
            name="w/o Even Parity",
            use_even=False,
        ),

        AblationConfig(
            name="w/o Valley",
            use_valley=False,
        ),

        AblationConfig(
            name="w/o Poisson Normalization",
            use_poisson=False,
        ),

        AblationConfig(
            name="w/o Information Support",
            use_information=False,
        ),

        AblationConfig(
            name="Single Scale d=5",
            offsets=(5,),
        ),

        # This is exactly the Local + Global scale combination.
        AblationConfig(
            name="w/o Mesoscopic",
            use_mesoscopic=False,
            use_global_cp=True,
        ),

        # This is exactly the Local + Mesoscopic scale combination.
        AblationConfig(
            name="w/o Global CP",
            use_mesoscopic=True,
            use_global_cp=False,
        ),

        AblationConfig(
            name="Equal Scale Fusion",
            equal_scale_fusion=True,
        ),

        AblationConfig(
            name="w/o Shared Global Reference",
            use_shared_global_reference=False,
        ),

        AblationConfig(
            name="Local Only",
            use_mesoscopic=False,
            use_global_cp=False,
        ),
    ]


# ============================================================================
# Generic helpers
# ============================================================================

def slugify(text: str) -> str:
    result = []
    for char in text.lower():
        if char.isalnum():
            result.append(char)
        else:
            result.append("_")
    value = "".join(result)
    while "__" in value:
        value = value.replace("__", "_")
    return value.strip("_")


def normalize_on_valid(values: np.ndarray, valid: np.ndarray) -> np.ndarray:
    result = np.asarray(values, dtype=float).copy()
    result[~np.isfinite(result)] = 0.0
    result[~valid] = 0.0

    reference = result[valid]
    if len(reference) == 0:
        return result

    lo = float(np.min(reference))
    hi = float(np.max(reference))

    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        result[:] = 0.0
        return result

    result = (result - lo) / (hi - lo)
    result = np.clip(result, 0.0, 1.0)
    result[~valid] = 0.0
    return result


def support_from_loss(
    loss_values: np.ndarray,
    valid: np.ndarray,
) -> np.ndarray:
    support = np.ones(len(valid), dtype=float)
    support[~valid] = 0.0

    current = np.asarray(loss_values, dtype=float)[valid]

    if len(current) == 0:
        return support

    minimum = float(np.min(current))

    support[valid] = np.exp(
        -np.clip(current - minimum, 0.0, 700.0)
    )

    return support


def concentration(
    values: np.ndarray,
    valid: np.ndarray,
) -> float:
    probability = np.maximum(
        np.asarray(values, dtype=float)[valid],
        0.0,
    )

    mass = float(np.sum(probability))

    if mass <= EPS or len(probability) <= 1:
        return 0.0

    probability /= mass

    entropy = -float(
        np.sum(probability * np.log(probability + EPS))
    )

    return float(
        np.clip(
            1.0 - entropy / math.log(len(probability)),
            0.0,
            1.0,
        )
    )


def entropy_concentration(
    values: np.ndarray,
    valid: np.ndarray,
) -> float:
    evidence = np.asarray(values, dtype=float)[valid]

    if len(evidence) <= 1:
        return 0.0

    evidence = np.maximum(
        evidence - float(np.min(evidence)),
        0.0,
    )

    mass = float(np.sum(evidence))

    if mass <= EPS:
        return 0.0

    probability = evidence / mass

    entropy = -float(
        np.sum(probability * np.log(probability + EPS))
    )

    return float(
        np.clip(
            1.0 - entropy / math.log(len(probability)),
            0.0,
            1.0,
        )
    )


def robust_prominence(
    values: np.ndarray,
    valid: np.ndarray,
) -> float:
    evidence = np.asarray(values, dtype=float)[valid]

    if len(evidence) == 0:
        return 0.0

    baseline = float(np.median(evidence))

    robust_sigma = 1.4826 * float(
        np.median(np.abs(evidence - baseline))
    )

    prominence = max(
        float(np.max(evidence)) - baseline,
        0.0,
    )

    return float(
        prominence
        / (prominence + robust_sigma + EPS)
    )


def peak_dominance(
    values: np.ndarray,
    valid: np.ndarray,
    distance: int,
) -> float:
    from scipy.signal import find_peaks

    finite = np.nan_to_num(
        np.asarray(values, dtype=float),
        nan=0.0,
    )

    peaks, _ = find_peaks(
        finite,
        distance=max(int(distance), 1),
    )

    peaks = np.asarray(
        [p for p in peaks if valid[p]],
        dtype=int,
    )

    if len(peaks) == 0:
        return 0.0

    levels = np.sort(finite[peaks])[::-1]

    if levels[0] <= EPS:
        return 0.0

    runner_up = (
        levels[1]
        if len(levels) > 1
        else 0.0
    )

    return float(
        np.clip(
            (levels[0] - runner_up) / levels[0],
            0.0,
            1.0,
        )
    )


def evidence_odds(probability_like: float) -> float:
    value = float(
        np.clip(
            probability_like,
            0.0,
            1.0 - 1e-6,
        )
    )

    return value / max(1.0 - value, 1e-6)


def peak_curvature(
    score: np.ndarray,
    index: int,
    valid: np.ndarray,
) -> float:
    if index <= 0 or index >= len(score) - 1:
        return fv4.RELIABILITY_FEATURE_EPSILON

    if not (
        valid[index - 1]
        and valid[index]
        and valid[index + 1]
    ):
        return fv4.RELIABILITY_FEATURE_EPSILON

    triplet = np.asarray(
        score[index - 1:index + 2],
        dtype=float,
    )

    if np.any(~np.isfinite(triplet)):
        return fv4.RELIABILITY_FEATURE_EPSILON

    return max(
        float(
            2.0 * triplet[1]
            - triplet[0]
            - triplet[2]
        ),
        fv4.RELIABILITY_FEATURE_EPSILON,
    )


def normalize_reliability(
    reliability: np.ndarray,
    allowed: np.ndarray,
) -> np.ndarray:
    reliability = np.maximum(
        np.asarray(reliability, dtype=float),
        0.0,
    )

    allowed = np.asarray(allowed, dtype=bool)

    reliability[~allowed] = 0.0

    mass = float(np.sum(reliability))

    if mass > EPS:
        return reliability / mass

    fallback = allowed.astype(float)

    fallback_mass = float(np.sum(fallback))

    if fallback_mass <= EPS:
        raise RuntimeError("No active evidence remains.")

    return fallback / fallback_mass


# ============================================================================
# Mesoscopic branch
# ============================================================================

def mesoscopic_ablation(
    positive_count: np.ndarray,
    negative_count: np.ndarray,
    activity_rate: np.ndarray,
    valid: np.ndarray,
    offsets_array: np.ndarray,
    symmetry_index: float,
    symmetry_reliability: float,
    symmetry_resolution: float,
    symmetry_length_adequacy: float,
    rate_index: float,
    config: AblationConfig,
) -> tuple[np.ndarray, dict[str, Any]]:

    positive = np.asarray(
        positive_count,
        dtype=float,
    )

    negative = np.asarray(
        negative_count,
        dtype=float,
    )

    activity = np.asarray(
        activity_rate,
        dtype=float,
    )

    n = len(activity)

    candidates = np.flatnonzero(valid)

    # IMPORTANT:
    # keep the same resolution / boundary scale as Full V4
    # even for the single-scale ablation.
    reference_radius = REFERENCE_RADIUS

    smooth_kernel = fv4.gaussian_kernel(3.0)

    activity_s = fv4.reflected_convolve(
        activity,
        smooth_kernel,
    )

    positive_activity_s = fv4.reflected_convolve(
        positive,
        smooth_kernel,
    )

    negative_activity_s = fv4.reflected_convolve(
        negative,
        smooth_kernel,
    )

    positive_s = (
        positive_activity_s
        / max(float(np.max(positive_activity_s)), EPS)
    )

    negative_s = (
        negative_activity_s
        / max(float(np.max(negative_activity_s)), EPS)
    )

    residual = activity - activity_s

    residual_median = float(
        np.median(residual)
    )

    noise_scale = 1.4826 * float(
        np.median(
            np.abs(
                residual - residual_median
            )
        )
    )

    if noise_scale <= EPS:
        if len(activity) > 1:
            noise_scale = 1.4826 * float(
                np.median(
                    np.abs(
                        np.diff(activity)
                    )
                )
            )

    noise_scale = max(
        noise_scale,
        EPS,
    )

    number_scales = len(offsets_array)

    support = np.zeros(
        n,
        dtype=float,
    )

    valley = np.zeros(
        n,
        dtype=float,
    )

    reversal = np.zeros(
        n,
        dtype=float,
    )

    valley_by_scale = np.zeros(
        (number_scales, n),
        dtype=float,
    )

    reversal_by_scale = np.zeros_like(
        valley_by_scale
    )

    side_activity = np.empty(
        (number_scales, len(candidates)),
        dtype=float,
    )

    reversal_weights = np.empty_like(
        side_activity
    )

    for scale_index, delta_value in enumerate(
        offsets_array
    ):
        delta = int(delta_value)

        left = candidates - delta
        right = candidates + delta

        side = 0.5 * (
            activity_s[left]
            + activity_s[right]
        )

        side_activity[
            scale_index
        ] = np.maximum(
            side,
            0.0,
        )

        valley_by_scale[
            scale_index,
            candidates,
        ] = np.clip(
            (
                side
                - activity_s[candidates]
            )
            / (
                np.abs(side)
                + noise_scale
            ),
            0.0,
            1.0,
        )

        mismatch = (
            np.abs(
                positive_s[left]
                - negative_s[right]
            )
            +
            np.abs(
                negative_s[left]
                - positive_s[right]
            )
        )

        denominator = (
            positive_s[left]
            + negative_s[left]
            + positive_s[right]
            + negative_s[right]
        )

        weights = np.maximum(
            denominator,
            EPS,
        )

        reversal_by_scale[
            scale_index,
            candidates,
        ] = np.clip(
            1.0
            - mismatch / weights,
            0.0,
            1.0,
        )

        reversal_weights[
            scale_index
        ] = weights

    support[candidates] = np.sqrt(
        np.mean(
            side_activity,
            axis=0,
        )
        *
        np.maximum(
            activity_s[candidates],
            0.0,
        )
    )

    valley[candidates] = np.mean(
        valley_by_scale[:, candidates],
        axis=0,
    )

    reversal[candidates] = (
        np.sum(
            reversal_by_scale[:, candidates]
            * reversal_weights,
            axis=0,
        )
        /
        np.maximum(
            np.sum(
                reversal_weights,
                axis=0,
            ),
            EPS,
        )
    )

    support = normalize_on_valid(
        support,
        valid,
    )

    def scale_stability(
        scale_evidence: np.ndarray,
    ) -> tuple[float, float, list[int]]:

        locations = []

        for values in scale_evidence:
            current = (
                values * support
            )[candidates]

            best = int(
                candidates[
                    int(
                        np.argmax(current)
                    )
                ]
            )

            locations.append(best)

        location_array = np.asarray(
            locations,
            dtype=float,
        )

        median = float(
            np.median(
                location_array
            )
        )

        mad = 1.4826 * float(
            np.median(
                np.abs(
                    location_array
                    - median
                )
            )
        )

        stability = float(
            1.0
            /
            (
                1.0
                +
                (
                    mad
                    / max(
                        reference_radius,
                        1,
                    )
                ) ** 2
            )
        )

        return (
            stability,
            mad,
            locations,
        )

    valley_stability, valley_mad, valley_locations = (
        scale_stability(
            valley_by_scale
        )
    )

    reversal_stability, reversal_mad, reversal_locations = (
        scale_stability(
            reversal_by_scale
        )
    )

    valley_reliability = 0.0

    if config.use_valley:
        valley_reliability = (
            entropy_concentration(
                valley,
                valid,
            )
            *
            robust_prominence(
                valley,
                valid,
            )
            *
            valley_stability
        )

    reversal_reliability = 0.0

    if config.use_odd:
        reversal_reliability = (
            entropy_concentration(
                reversal,
                valid,
            )
            *
            robust_prominence(
                reversal,
                valid,
            )
            *
            reversal_stability
        )

    structural_parts = []
    structural_weights = []

    if (
        config.use_valley
        and valley_reliability > 0
    ):
        structural_parts.append(
            valley
        )

        structural_weights.append(
            valley_reliability
        )

    if (
        config.use_odd
        and reversal_reliability > 0
    ):
        structural_parts.append(
            reversal
        )

        structural_weights.append(
            reversal_reliability
        )

    if structural_parts:
        w = np.asarray(
            structural_weights,
            dtype=float,
        )

        w /= np.sum(w)

        structural_log = np.zeros(
            n,
            dtype=float,
        )

        for current_weight, current_field in zip(
            w,
            structural_parts,
        ):
            structural_log += (
                current_weight
                * np.log(
                    current_field
                    + 1e-6
                )
            )

        structural_term = np.exp(
            structural_log
        )

    else:
        structural_term = np.ones(
            n,
            dtype=float,
        )

    valley_hypothesis = (
        support
        * structural_term
    )

    valley_hypothesis[
        ~valid
    ] = 0.0

    # ------------------------------------------------------------------------
    # Causal trailing-edge evidence
    # ------------------------------------------------------------------------

    center_energy = normalize_on_valid(
        activity_s,
        valid,
    )

    polarity_balance = (
        1.0
        -
        np.abs(
            positive_activity_s
            - negative_activity_s
        )
        /
        np.maximum(
            positive_activity_s
            + negative_activity_s,
            EPS,
        )
    )

    descending = normalize_on_valid(
        np.maximum(
            0.0,
            -np.gradient(
                center_energy
            ),
        ),
        valid,
    )

    edge_hypothesis = (
        center_energy
        *
        np.sqrt(
            np.maximum(
                polarity_balance,
                0.0,
            )
            *
            descending
        )
    )

    edge_hypothesis[
        ~valid
    ] = 0.0

    valley_possibility = normalize_on_valid(
        valley_hypothesis,
        valid,
    )

    edge_possibility = normalize_on_valid(
        edge_hypothesis,
        valid,
    )

    sample_grid = np.arange(
        n,
        dtype=float,
    )

    symmetry_possibility = np.exp(
        -0.5
        *
        (
            (
                sample_grid
                - symmetry_index
            )
            / symmetry_resolution
        ) ** 2
    )

    rate_possibility = np.exp(
        -0.5
        *
        (
            (
                sample_grid
                - rate_index
            )
            / reference_radius
        ) ** 2
    )

    centres = np.asarray(
        [
            float(
                candidates[
                    int(
                        np.argmax(
                            valley_possibility[
                                candidates
                            ]
                        )
                    )
                ]
            ),
            float(
                candidates[
                    int(
                        np.argmax(
                            edge_possibility[
                                candidates
                            ]
                        )
                    )
                ]
            ),
            float(symmetry_index),
            float(rate_index),
        ],
        dtype=float,
    )

    resolutions = np.asarray(
        [
            reference_radius,
            reference_radius,
            symmetry_resolution,
            reference_radius,
        ],
        dtype=float,
    )

    local_coverage_fraction = (
        reference_radius
        /
        max(
            n - 1,
            1,
        )
    )

    local_coverage = (
        local_coverage_fraction
        /
        (
            local_coverage_fraction
            + fv4.REFERENCE_COVERAGE_FRACTION
        )
    )

    symmetry_coverage = float(
        np.clip(
            symmetry_length_adequacy,
            0.0,
            1.0,
        )
    )

    enabled_stabilities = []

    if config.use_valley:
        enabled_stabilities.append(
            valley_stability
        )

    if config.use_odd:
        enabled_stabilities.append(
            reversal_stability
        )

    if enabled_stabilities:
        structural_stability = float(
            np.prod(
                enabled_stabilities
            )
            ** (
                1.0
                / len(
                    enabled_stabilities
                )
            )
        )
    else:
        structural_stability = 0.0

    structural_identifiability = (
        peak_dominance(
            valley_hypothesis,
            valid,
            reference_radius,
        )
    )

    edge_identifiability = (
        peak_dominance(
            edge_hypothesis,
            valid,
            reference_radius,
        )
    )

    base_reliability = np.asarray(
        [
            local_coverage
            * structural_stability
            * (
                structural_identifiability
                ** fv4.VALLEY_IDENTIFIABILITY_ORDER
            ),

            local_coverage
            * (
                edge_identifiability
                ** fv4.EDGE_IDENTIFIABILITY_ORDER
            ),

            float(
                symmetry_reliability
            ),

            (
                (1.0 - local_coverage)
                ** fv4.RATE_LOCAL_FAILURE_ORDER
            )
            *
            (
                (1.0 - symmetry_coverage)
                ** fv4.RATE_GLOBAL_FAILURE_ORDER
            ),
        ],
        dtype=float,
    )

    allowed = np.asarray(
        [
            True,
            True,
            config.use_shared_global_reference,
            config.use_shared_global_reference,
        ],
        dtype=bool,
    )

    base_reliability[
        ~allowed
    ] = 0.0

    distance_squared = (
        centres[:, None]
        - centres[None, :]
    ) ** 2

    variance_pair = (
        resolutions[:, None] ** 2
        + resolutions[None, :] ** 2
    )

    agreement = np.exp(
        -distance_squared
        /
        np.maximum(
            2.0 * variance_pair,
            EPS,
        )
    )

    np.fill_diagonal(
        agreement,
        0.0,
    )

    agreement[
        :,
        base_reliability <= EPS,
    ] = 0.0

    corroboration = np.max(
        agreement,
        axis=1,
    )

    # rate is an observability fallback
    if allowed[-1]:
        corroboration[-1] = 1.0

    adjusted_reliability = (
        base_reliability
        *
        corroboration
        ** fv4.CORROBORATION_ORDER
    )

    weights = normalize_reliability(
        adjusted_reliability,
        allowed,
    )

    possibilities = np.vstack(
        [
            valley_possibility,
            edge_possibility,
            symmetry_possibility,
            rate_possibility,
        ]
    )

    score = np.sum(
        weights[:, None]
        * possibilities,
        axis=0,
    )

    score[
        ~valid
    ] = np.nan

    meta = {
        "weights": {
            "valley_polarity": float(weights[0]),
            "trailing_edge": float(weights[1]),
            "global_symmetry": float(weights[2]),
            "global_rate": float(weights[3]),
        },

        "candidate_indices": {
            "valley_polarity": float(centres[0]),
            "trailing_edge": float(centres[1]),
            "global_symmetry": float(centres[2]),
            "global_rate": float(centres[3]),
        },

        "valley_reliability": float(
            valley_reliability
        ),

        "polarity_reversal_reliability": float(
            reversal_reliability
        ),

        "valley_multiscale_mad_samples": float(
            valley_mad
        ),

        "reversal_multiscale_mad_samples": float(
            reversal_mad
        ),

        "valley_scale_candidates": (
            valley_locations
        ),

        "reversal_scale_candidates": (
            reversal_locations
        ),

        "global_symmetry_length_adequacy": (
            symmetry_coverage
        ),
    }

    return score, meta


# ============================================================================
# Configurable Fusion-v4 estimator
# ============================================================================

def focus_ablation(
    positive_count: np.ndarray,
    negative_count: np.ndarray,
    duration_s: np.ndarray | float,
    config: AblationConfig,
) -> tuple[pd.DataFrame, dict[str, Any]]:

    positive = np.asarray(
        positive_count,
        dtype=float,
    )

    negative = np.asarray(
        negative_count,
        dtype=float,
    )

    if (
        positive.ndim != 1
        or negative.ndim != 1
        or len(positive) != len(negative)
    ):
        raise ValueError(
            "positive_count and negative_count "
            "must be equal 1-D arrays"
        )

    if (
        np.any(~np.isfinite(positive))
        or np.any(~np.isfinite(negative))
        or np.any(positive < 0)
        or np.any(negative < 0)
    ):
        raise ValueError(
            "Event counts must be finite "
            "and non-negative."
        )

    n = len(positive)

    if np.isscalar(duration_s):
        duration = np.full(
            n,
            float(duration_s),
            dtype=float,
        )
    else:
        duration = np.asarray(
            duration_s,
            dtype=float,
        )

    if (
        duration.shape != positive.shape
        or np.any(~np.isfinite(duration))
        or np.any(duration <= 0)
    ):
        raise ValueError(
            "duration_s must be finite and positive."
        )

    offsets_array = np.sort(
        np.unique(
            np.asarray(
                config.offsets,
                dtype=int,
            )
        )
    )

    if (
        len(offsets_array) == 0
        or np.any(offsets_array <= 0)
    ):
        raise ValueError(
            "Offsets must contain positive integers."
        )

    if int(np.max(offsets_array)) > REFERENCE_RADIUS:
        raise ValueError(
            "Ablation offsets exceed Full-V4 "
            "reference candidate boundary."
        )

    # ------------------------------------------------------------------------
    # SAME candidate region for every ablation
    # ------------------------------------------------------------------------

    valid = np.zeros(
        n,
        dtype=bool,
    )

    if (
        n
        >=
        2 * REFERENCE_RADIUS + 1
    ):
        valid[
            REFERENCE_RADIUS:
            n - REFERENCE_RADIUS
        ] = True

    observability = fv4.focus_observability(
        positive,
        negative,
        duration,
    )

    if (
        n
        <
        2 * REFERENCE_RADIUS + 1
    ):
        return (
            pd.DataFrame(),
            {
                "valid_focus": False,
                "status": "no_focus",
                "no_focus_reason": "insufficient_focus_samples",
                "variant": config.name,
                "observability": observability,
            },
        )

    if not observability["passed"]:
        return (
            pd.DataFrame(),
            {
                "valid_focus": False,
                "status": "no_focus",
                "no_focus_reason": observability.get(
                    "reason",
                    "unobservable",
                ),
                "variant": config.name,
                "observability": observability,
            },
        )

    candidates = np.flatnonzero(
        valid
    )

    # ------------------------------------------------------------------------
    # Polarity nuisance gain correction
    # ------------------------------------------------------------------------

    sum_positive = float(
        np.sum(positive)
    )

    sum_negative = float(
        np.sum(negative)
    )

    common_total = 0.5 * (
        sum_positive
        + sum_negative
    )

    gain_positive = (
        common_total
        /
        max(
            sum_positive,
            EPS,
        )
    )

    gain_negative = (
        common_total
        /
        max(
            sum_negative,
            EPS,
        )
    )

    rate_positive_raw = (
        gain_positive
        * positive
        / duration
    )

    rate_negative_raw = (
        gain_negative
        * negative
        / duration
    )

    kernel = fv4.gaussian_kernel(
        float(
            fv4.DEFAULT_SMOOTHING_SIGMA
        )
    )

    rate_positive = (
        fv4.reflected_convolve(
            rate_positive_raw,
            kernel,
        )
    )

    rate_negative = (
        fv4.reflected_convolve(
            rate_negative_raw,
            kernel,
        )
    )

    total_rate_raw = (
        rate_positive_raw
        + rate_negative_raw
    )

    total_rate = (
        rate_positive
        + rate_negative
    )

    signed_rate = (
        rate_positive
        - rate_negative
    )

    # ------------------------------------------------------------------------
    # Variance model
    # ------------------------------------------------------------------------

    if config.use_poisson:

        variance_positive_raw = (
            gain_positive ** 2
            * positive
            / duration ** 2
        )

        variance_negative_raw = (
            gain_negative ** 2
            * negative
            / duration ** 2
        )

        kernel_squared = (
            kernel ** 2
        )

        variance_positive = (
            fv4.reflected_convolve(
                variance_positive_raw,
                kernel_squared,
            )
        )

        variance_negative = (
            fv4.reflected_convolve(
                variance_negative_raw,
                kernel_squared,
            )
        )

        variance_total = (
            variance_positive
            + variance_negative
        )

        dispersion = (
            fv4.robust_overdispersion(
                total_rate_raw,
                total_rate,
                variance_total,
            )
        )

        variance_total *= dispersion

    else:
        # Ablation:
        # remove count-dependent Poisson variance,
        # but retain one sequence-level robust scale.
        residual = (
            total_rate_raw
            - total_rate
        )

        centre = float(
            np.median(
                residual
            )
        )

        robust_sigma = 1.4826 * float(
            np.median(
                np.abs(
                    residual
                    - centre
                )
            )
        )

        if (
            robust_sigma <= EPS
            and len(total_rate) > 1
        ):
            robust_sigma = 1.4826 * float(
                np.median(
                    np.abs(
                        np.diff(
                            total_rate
                        )
                    )
                )
            )

        robust_sigma = max(
            robust_sigma,
            EPS,
        )

        variance_total = np.full(
            n,
            robust_sigma ** 2,
            dtype=float,
        )

        dispersion = 1.0

    # ------------------------------------------------------------------------
    # Local parity residuals
    # ------------------------------------------------------------------------

    number_scales = len(
        offsets_array
    )

    even_loss_by_scale = np.zeros(
        (number_scales, n),
        dtype=float,
    )

    odd_loss_by_scale = np.zeros_like(
        even_loss_by_scale
    )

    valley_loss_by_scale = np.zeros_like(
        even_loss_by_scale
    )

    for scale_index, delta_value in enumerate(
        offsets_array
    ):
        delta = int(
            delta_value
        )

        left = (
            candidates
            - delta
        )

        right = (
            candidates
            + delta
        )

        side_variance = (
            variance_total[left]
            + variance_total[right]
            + EPS
        )

        z_even = (
            total_rate[left]
            - total_rate[right]
        ) / np.sqrt(
            side_variance
        )

        z_odd = (
            signed_rate[left]
            + signed_rate[right]
        ) / np.sqrt(
            side_variance
        )

        curvature_variance = (
            side_variance
            +
            4.0
            * variance_total[candidates]
        )

        z_valley = (
            total_rate[left]
            + total_rate[right]
            -
            2.0
            * total_rate[candidates]
        ) / np.sqrt(
            curvature_variance
            + EPS
        )

        if config.use_even:
            even_loss_by_scale[
                scale_index,
                candidates,
            ] = fv4.huber(
                z_even,
                fv4.DEFAULT_HUBER_DELTA,
            )

        if config.use_odd:
            odd_loss_by_scale[
                scale_index,
                candidates,
            ] = fv4.huber(
                z_odd,
                fv4.DEFAULT_HUBER_DELTA,
            )

        if config.use_valley:
            mixture = float(
                fv4.DEFAULT_VALLEY_MIXTURE
            )

            log_valley_probability = (
                np.logaddexp(
                    math.log(
                        max(
                            1.0 - mixture,
                            EPS,
                        )
                    )
                    + math.log(0.5),

                    math.log(
                        max(
                            mixture,
                            EPS,
                        )
                    )
                    + log_ndtr(
                        z_valley
                    ),
                )
            )

            valley_loss_by_scale[
                scale_index,
                candidates,
            ] = (
                -log_valley_probability
            )

    even_loss = np.mean(
        even_loss_by_scale[
            :,
            valid,
        ],
        axis=0,
    )

    odd_loss = np.mean(
        odd_loss_by_scale[
            :,
            valid,
        ],
        axis=0,
    )

    valley_loss = np.mean(
        valley_loss_by_scale[
            :,
            valid,
        ],
        axis=0,
    )

    combined_loss = np.zeros(
        n,
        dtype=float,
    )

    local_loss_valid = np.zeros(
        len(candidates),
        dtype=float,
    )

    if config.use_even:
        local_loss_valid += (
            even_loss
        )

    if config.use_odd:
        local_loss_valid += (
            odd_loss
        )

    if config.use_valley:
        local_loss_valid += (
            valley_loss
        )

    combined_loss[
        valid
    ] = local_loss_valid

    even_full = np.zeros(
        n,
        dtype=float,
    )
    even_full[valid] = (
        even_loss
    )

    odd_full = np.zeros(
        n,
        dtype=float,
    )
    odd_full[valid] = (
        odd_loss
    )

    valley_full = np.zeros(
        n,
        dtype=float,
    )
    valley_full[valid] = (
        valley_loss
    )

    even_support = support_from_loss(
        even_full,
        valid,
    )

    odd_support = support_from_loss(
        odd_full,
        valid,
    )

    valley_support = support_from_loss(
        valley_full,
        valid,
    )

    local_relative_likelihood = np.zeros(
        n,
        dtype=float,
    )

    minimum_local_loss = float(
        np.min(
            combined_loss[
                valid
            ]
        )
    )

    local_relative_likelihood[
        valid
    ] = np.exp(
        -np.clip(
            combined_loss[valid]
            - minimum_local_loss,
            0.0,
            700.0,
        )
    )

    # ------------------------------------------------------------------------
    # Information support
    # ------------------------------------------------------------------------

    if config.use_information:
        information_support = (
            normalize_on_valid(
                total_rate,
                valid,
            )
        )
    else:
        information_support = np.zeros(
            n,
            dtype=float,
        )
        information_support[
            valid
        ] = 1.0

    local_peof_support = np.zeros(
        n,
        dtype=float,
    )

    local_peof_support[
        valid
    ] = (
        information_support[
            valid
        ]
        ** fv4.INFORMATION_EXPONENT
        *
        local_relative_likelihood[
            valid
        ]
    )

    local_peak = float(
        np.max(
            local_peof_support[
                valid
            ]
        )
    )

    if local_peak > 0:
        local_peof_support /= (
            local_peak
        )

    # ------------------------------------------------------------------------
    # Whole-sequence global references
    # ------------------------------------------------------------------------

    symmetry_index, symmetry_meta = (
        fv4.global_odd_symmetry_index(
            rate_positive_raw,
            rate_negative_raw,
        )
    )

    symmetry_resolution = max(
        float(
            symmetry_meta[
                "resolution_samples"
            ]
        ),
        fv4.RELIABILITY_FEATURE_EPSILON,
    )

    symmetry_length_adequacy = float(
        symmetry_meta[
            "length_adequacy"
        ]
    )

    rate_index = (
        fv4.golden_section_rate_index(
            total_rate_raw
        )
    )

    sample_grid = np.arange(
        n,
        dtype=float,
    )

    symmetry_support = np.exp(
        -0.5
        *
        (
            (
                sample_grid
                - symmetry_index
            )
            / symmetry_resolution
        ) ** 2
    )

    rate_support = np.exp(
        -0.5
        *
        (
            (
                sample_grid
                - rate_index
            )
            / REFERENCE_RADIUS
        ) ** 2
    )

    # ------------------------------------------------------------------------
    # Causal trailing edge
    # ------------------------------------------------------------------------

    center_information = (
        normalize_on_valid(
            total_rate,
            valid,
        )
    )

    polarity_balance = np.clip(
        1.0
        -
        np.abs(
            rate_positive
            - rate_negative
        )
        /
        np.maximum(
            rate_positive
            + rate_negative,
            EPS,
        ),
        0.0,
        1.0,
    )

    negative_derivative = np.maximum(
        0.0,
        -np.gradient(
            center_information
        ),
    )

    descending_support = (
        normalize_on_valid(
            negative_derivative,
            valid,
        )
    )

    edge_support = (
        center_information
        *
        np.sqrt(
            polarity_balance
            * descending_support
        )
    )

    edge_support[
        ~valid
    ] = 0.0

    # ------------------------------------------------------------------------
    # Local evidence reliability
    # ------------------------------------------------------------------------

    local_coverage_fraction = (
        REFERENCE_RADIUS
        /
        max(
            n - 1,
            1,
        )
    )

    local_coverage = float(
        local_coverage_fraction
        /
        (
            local_coverage_fraction
            + fv4.REFERENCE_COVERAGE_FRACTION
        )
    )

    active_identifiability = []

    if config.use_even:
        active_identifiability.append(
            concentration(
                even_support
                * information_support,
                valid,
            )
        )

    if config.use_odd:
        active_identifiability.append(
            concentration(
                odd_support
                * information_support,
                valid,
            )
        )

    if active_identifiability:
        local_identifiability = float(
            np.prod(
                active_identifiability
            )
        )
    else:
        local_identifiability = 0.0

    edge_identifiability = (
        peak_dominance(
            edge_support,
            valid,
            REFERENCE_RADIUS,
        )
    )

    symmetry_coverage = (
        symmetry_length_adequacy
    )

    absolute_reliability = np.asarray(
        [
            local_coverage
            * evidence_odds(
                local_identifiability
            ),

            fv4.EDGE_DEPENDENCE_CORRECTION
            * local_coverage
            * evidence_odds(
                edge_identifiability
            ),

            evidence_odds(
                float(
                    symmetry_meta[
                        "reliability"
                    ]
                )
            ),

            (
                (1.0 - local_coverage)
                ** 2
            )
            *
            (
                (1.0 - symmetry_coverage)
                ** 2
            ),
        ],
        dtype=float,
    )

    allowed_local_evidence = np.asarray(
        [
            True,
            True,
            config.use_shared_global_reference,
            config.use_shared_global_reference,
        ],
        dtype=bool,
    )

    absolute_reliability[
        ~allowed_local_evidence
    ] = 0.0

    local_centres = np.asarray(
        [
            float(
                candidates[
                    int(
                        np.argmax(
                            local_peof_support[
                                candidates
                            ]
                        )
                    )
                ]
            ),

            float(
                candidates[
                    int(
                        np.argmax(
                            edge_support[
                                candidates
                            ]
                        )
                    )
                ]
            ),

            float(
                symmetry_index
            ),

            float(
                rate_index
            ),
        ],
        dtype=float,
    )

    local_resolutions = np.asarray(
        [
            REFERENCE_RADIUS,
            REFERENCE_RADIUS,
            symmetry_resolution,
            REFERENCE_RADIUS,
        ],
        dtype=float,
    )

    distance_squared = (
        local_centres[:, None]
        - local_centres[None, :]
    ) ** 2

    variance_pair = (
        local_resolutions[:, None] ** 2
        +
        local_resolutions[None, :] ** 2
    )

    agreement = np.exp(
        -distance_squared
        /
        np.maximum(
            2.0 * variance_pair,
            EPS,
        )
    )

    np.fill_diagonal(
        agreement,
        0.0,
    )

    agreement[
        :,
        absolute_reliability <= EPS,
    ] = 0.0

    corroboration = np.max(
        agreement,
        axis=1,
    )

    if config.use_shared_global_reference:
        corroboration[-1] = 1.0

    absolute_reliability = (
        absolute_reliability
        *
        corroboration
    )

    evidence_weights = normalize_reliability(
        absolute_reliability,
        allowed_local_evidence,
    )

    local_scale_likelihood = (
        evidence_weights[0]
        * local_peof_support
        +
        evidence_weights[1]
        * edge_support
        +
        evidence_weights[2]
        * symmetry_support
        +
        evidence_weights[3]
        * rate_support
    )

    local_scale_likelihood[
        ~valid
    ] = np.nan

    local_scale_likelihood /= max(
        float(
            np.nanmax(
                local_scale_likelihood[
                    valid
                ]
            )
        ),
        EPS,
    )

    # ------------------------------------------------------------------------
    # Mesoscopic scale
    # ------------------------------------------------------------------------

    activity_rate = (
        positive
        + negative
    ) / duration

    structural_rate_index = (
        fv4.golden_section_rate_index(
            activity_rate
        )
    )

    mesoscopic_scale_likelihood, mesoscopic_meta = (
        mesoscopic_ablation(
            positive_count=positive,
            negative_count=negative,
            activity_rate=activity_rate,
            valid=valid,
            offsets_array=offsets_array,
            symmetry_index=symmetry_index,
            symmetry_reliability=float(
                symmetry_meta[
                    "reliability"
                ]
            ),
            symmetry_resolution=symmetry_resolution,
            symmetry_length_adequacy=symmetry_length_adequacy,
            rate_index=structural_rate_index,
            config=config,
        )
    )

    mesoscopic_scale_likelihood /= max(
        float(
            np.nanmax(
                mesoscopic_scale_likelihood[
                    valid
                ]
            )
        ),
        EPS,
    )

    mesoscopic_scale_likelihood = (
        fv4.reflected_convolve(
            np.nan_to_num(
                mesoscopic_scale_likelihood,
                nan=0.0,
            ),
            fv4.gaussian_kernel(
                fv4.STRUCTURAL_POOL_SMOOTHING_SIGMA_SAMPLES
            ),
        )
    )

    mesoscopic_scale_likelihood[
        ~valid
    ] = np.nan

    # ------------------------------------------------------------------------
    # Local vs mesoscopic adaptive reliability
    # ------------------------------------------------------------------------

    local_candidate = int(
        candidates[
            int(
                np.argmax(
                    local_scale_likelihood[
                        candidates
                    ]
                )
            )
        ]
    )

    mesoscopic_candidate = int(
        candidates[
            int(
                np.argmax(
                    mesoscopic_scale_likelihood[
                        candidates
                    ]
                )
            )
        ]
    )

    candidate_disagreement = float(
        abs(
            local_candidate
            - mesoscopic_candidate
        )
    )

    local_curvature = peak_curvature(
        local_scale_likelihood,
        local_candidate,
        valid,
    )

    mesoscopic_curvature = peak_curvature(
        mesoscopic_scale_likelihood,
        mesoscopic_candidate,
        valid,
    )

    eps_r = (
        fv4.RELIABILITY_FEATURE_EPSILON
    )

    reliability_features = np.asarray(
        [
            1.0,

            math.log(
                (
                    local_coverage
                    + eps_r
                )
                /
                (
                    1.0
                    - local_coverage
                    + eps_r
                )
            ),

            math.log(
                (
                    mesoscopic_curvature
                    + eps_r
                )
                /
                (
                    local_curvature
                    + eps_r
                )
            ),

            math.log(
                local_identifiability
                + eps_r
            ),

            math.log1p(
                candidate_disagreement
            ),

            math.log1p(
                dispersion
            ),

            math.log(
                (
                    symmetry_length_adequacy
                    + eps_r
                )
                /
                (
                    1.0
                    - symmetry_length_adequacy
                    + eps_r
                )
            ),
        ],
        dtype=float,
    )

    reliability_logit = float(
        np.clip(
            fv4.STRUCTURAL_RELIABILITY_COEFFICIENTS
            @ reliability_features,
            -50.0,
            50.0,
        )
    )

    mesoscopic_consensus_weight = float(
        1.0
        /
        (
            1.0
            +
            math.exp(
                -reliability_logit
            )
        )
    )

    local_consensus_weight = (
        1.0
        - mesoscopic_consensus_weight
    )

    # ------------------------------------------------------------------------
    # Global polarity change point
    # ------------------------------------------------------------------------

    signed_phase_rate = (
        fv4.reflected_convolve(
            signed_rate,
            fv4.gaussian_kernel(
                fv4.POLARITY_CHANGE_POINT_SIGMA_SAMPLES
            ),
        )
    )

    polarity_bridge = np.cumsum(
        signed_phase_rate
    )

    polarity_bridge -= np.linspace(
        float(
            polarity_bridge[0]
        ),
        float(
            polarity_bridge[-1]
        ),
        n,
    )

    global_scale_likelihood = (
        normalize_on_valid(
            np.abs(
                polarity_bridge
            ),
            valid,
        )
    )

    local_coverage_logit = math.log(
        (
            local_coverage
            + eps_r
        )
        /
        (
            1.0
            - local_coverage
            + eps_r
        )
    )

    change_point_gate_logit = (
        fv4.POLARITY_CHANGE_POINT_GATE_SLOPE
        *
        (
            local_coverage_logit
            -
            fv4.POLARITY_CHANGE_POINT_COVERAGE_LOGIT
        )
    )

    coverage_gate = float(
        1.0
        /
        (
            1.0
            +
            math.exp(
                -float(
                    np.clip(
                        change_point_gate_logit,
                        -50.0,
                        50.0,
                    )
                )
            )
        )
    )

    length_adequacy_logit = math.log(
        (
            symmetry_length_adequacy
            + eps_r
        )
        /
        (
            1.0
            - symmetry_length_adequacy
            + eps_r
        )
    )

    required_length_logit = math.log(
        fv4.POLARITY_CHANGE_POINT_LENGTH_ADEQUACY
        /
        (
            1.0
            -
            fv4.POLARITY_CHANGE_POINT_LENGTH_ADEQUACY
        )
    )

    change_point_length_gate_logit = (
        fv4.POLARITY_CHANGE_POINT_GATE_SLOPE
        *
        (
            length_adequacy_logit
            -
            required_length_logit
        )
    )

    length_gate = float(
        1.0
        /
        (
            1.0
            +
            math.exp(
                -float(
                    np.clip(
                        change_point_length_gate_logit,
                        -50.0,
                        50.0,
                    )
                )
            )
        )
    )

    dual_polarity_adequacy = float(
        2.0
        *
        min(
            float(
                np.sum(
                    positive
                )
            ),
            float(
                np.sum(
                    negative
                )
            ),
        )
        /
        max(
            float(
                np.sum(
                    positive
                )
                +
                np.sum(
                    negative
                )
            ),
            eps_r,
        )
    )

    global_cp_weight = (
        coverage_gate
        * length_gate
        * dual_polarity_adequacy
    )

    # ------------------------------------------------------------------------
    # Scale marginalization
    # ------------------------------------------------------------------------

    if config.equal_scale_fusion:

        raw_scale_weights = np.asarray(
            [
                1.0,
                1.0 if config.use_mesoscopic else 0.0,
                1.0 if config.use_global_cp else 0.0,
            ],
            dtype=float,
        )

    else:

        raw_scale_weights = np.asarray(
            [
                (
                    1.0
                    - global_cp_weight
                )
                *
                local_consensus_weight,

                (
                    1.0
                    - global_cp_weight
                )
                *
                mesoscopic_consensus_weight,

                global_cp_weight,
            ],
            dtype=float,
        )

        if not config.use_mesoscopic:
            raw_scale_weights[1] = 0.0

        if not config.use_global_cp:
            raw_scale_weights[2] = 0.0

    scale_allowed = np.asarray(
        [
            True,
            config.use_mesoscopic,
            config.use_global_cp,
        ],
        dtype=bool,
    )

    scale_weights = normalize_reliability(
        raw_scale_weights,
        scale_allowed,
    )

    scale_likelihoods = np.vstack(
        [
            local_scale_likelihood,
            mesoscopic_scale_likelihood,
            global_scale_likelihood,
        ]
    )

    fusion_score = np.sum(
        scale_weights[:, None]
        * scale_likelihoods,
        axis=0,
    )

    fusion_score[
        ~valid
    ] = np.nan

    if not np.all(
        np.isfinite(
            fusion_score[
                candidates
            ]
        )
    ):
        return (
            pd.DataFrame(),
            {
                "valid_focus": False,
                "status": "no_focus",
                "no_focus_reason": "nonfinite_focus_score",
                "variant": config.name,
                "observability": observability,
            },
        )

    # ------------------------------------------------------------------------
    # Winner
    # ------------------------------------------------------------------------

    focus_index = int(
        candidates[
            int(
                np.nanargmax(
                    fusion_score[
                        candidates
                    ]
                )
            )
        ]
    )

    focus_index_float = float(
        focus_index
    )

    quadratic_offset = 0.0

    if (
        focus_index > 0
        and focus_index < n - 1
        and valid[
            focus_index - 1
        ]
        and valid[
            focus_index + 1
        ]
    ):
        denominator = float(
            fusion_score[
                focus_index - 1
            ]
            -
            2.0
            * fusion_score[
                focus_index
            ]
            +
            fusion_score[
                focus_index + 1
            ]
        )

        if denominator < -EPS:

            quadratic_offset = float(
                np.clip(
                    0.5
                    *
                    (
                        fusion_score[
                            focus_index - 1
                        ]
                        -
                        fusion_score[
                            focus_index + 1
                        ]
                    )
                    /
                    denominator,
                    -0.5,
                    0.5,
                )
            )

            focus_index_float += (
                quadratic_offset
            )

    likelihood_mass = float(
        np.nansum(
            fusion_score[
                valid
            ]
        )
    )

    posterior = np.zeros(
        n,
        dtype=float,
    )

    posterior[
        valid
    ] = (
        fusion_score[
            valid
        ]
        /
        max(
            likelihood_mass,
            EPS,
        )
    )

    curve = pd.DataFrame(
        {
            "sample_index": np.arange(
                n
            ),

            "positive_count": positive,

            "negative_count": negative,

            "total_rate": total_rate,

            "signed_rate": signed_rate,

            "information_support": information_support,

            "local_scale_likelihood": local_scale_likelihood,

            "mesoscopic_scale_likelihood": mesoscopic_scale_likelihood,

            "global_scale_likelihood": global_scale_likelihood,

            "fusion_score": fusion_score,

            "focus_posterior": posterior,

            "is_focus": (
                np.arange(n)
                == focus_index
            ),
        }
    )

    meta = {
        "variant": config.name,

        "valid_focus": True,

        "status": "focus_detected",

        "no_focus_reason": None,

        "focus_index": int(
            focus_index
        ),

        "focus_index_float": float(
            focus_index_float
        ),

        "quadratic_subsample_offset": float(
            quadratic_offset
        ),

        "observability": observability,

        "scale_weights": {
            "local": float(
                scale_weights[0]
            ),

            "mesoscopic": float(
                scale_weights[1]
            ),

            "global": float(
                scale_weights[2]
            ),
        },

        "local_consensus_weight": float(
            local_consensus_weight
        ),

        "mesoscopic_consensus_weight": float(
            mesoscopic_consensus_weight
        ),

        "global_cp_observability_weight": float(
            global_cp_weight
        ),

        "local_candidate_index": int(
            local_candidate
        ),

        "mesoscopic_candidate_index": int(
            mesoscopic_candidate
        ),

        "global_candidate_index": int(
            candidates[
                int(
                    np.argmax(
                        global_scale_likelihood[
                            candidates
                        ]
                    )
                )
            ]
        ),

        "candidate_disagreement_samples": float(
            candidate_disagreement
        ),

        "global_symmetry_index": float(
            symmetry_index
        ),

        "global_rate_index": float(
            rate_index
        ),

        "local_identifiability": float(
            local_identifiability
        ),

        "quasi_poisson_overdispersion": float(
            dispersion
        ),

        "polarity_gain_positive": float(
            gain_positive
        ),

        "polarity_gain_negative": float(
            gain_negative
        ),

        "mesoscopic": mesoscopic_meta,

        "config": asdict(
            config
        ),
    }

    return curve, meta


# ============================================================================
# Prediction wrapper
# ============================================================================

def predict_timestamp(
    data: protocol.SequenceData,
    frame: pd.DataFrame,
    config: AblationConfig,
) -> tuple[
    float | None,
    pd.DataFrame,
    dict[str, Any],
    float,
]:

    started = time.perf_counter()

    # Full V4 always uses the ORIGINAL estimator.
    # This guarantees the ablation baseline is exactly the existing V4.
    if config.name == "Full V4":

        curve, meta = (
            fv4.focus_from_event_counts(
                data.positive,
                data.negative,
                data.duration_us
                * 1e-6,
            )
        )

    else:

        curve, meta = focus_ablation(
            positive_count=data.positive,
            negative_count=data.negative,
            duration_s=data.duration_us
            * 1e-6,
            config=config,
        )

    runtime_ms = (
        time.perf_counter()
        - started
    ) * 1000.0

    if not bool(
        meta.get(
            "valid_focus",
            False,
        )
    ):
        return (
            None,
            curve,
            meta,
            runtime_ms,
        )

    winner = int(
        meta[
            "focus_index"
        ]
    )

    winner_float = float(
        meta.get(
            "focus_index_float",
            winner,
        )
    )

    timestamps = frame[
        "timestamp_us"
    ].to_numpy(
        dtype=float
    )

    predicted_timestamp = float(
        np.interp(
            winner_float,
            np.arange(
                len(timestamps),
                dtype=float,
            ),
            timestamps,
        )
    )

    return (
        predicted_timestamp,
        curve,
        meta,
        runtime_ms,
    )


# ============================================================================
# Summary generation
# ============================================================================

def build_summary(
    results: pd.DataFrame,
) -> pd.DataFrame:

    if results.empty:
        return pd.DataFrame()

    grouped = (
        results
        .groupby(
            [
                "dataset",
                "variant",
            ],
            as_index=False,
        )
        .agg(
            sequence_count=(
                "sequence",
                "count",
            ),

            valid_count=(
                "valid",
                "sum",
            ),

            success_rate=(
                "valid",
                "mean",
            ),

            timestamp_mae_ms=(
                "abs_error_ms",
                "mean",
            ),

            focus_position_mae=(
                "abs_position_error",
                "mean",
            ),

            mean_runtime_ms=(
                "runtime_ms",
                "mean",
            ),
        )
    )

    primary_metric = []

    primary_mae = []

    for _, row in grouped.iterrows():

        dataset = row[
            "dataset"
        ]

        if dataset in POSITION_DATASETS:

            primary_metric.append(
                "focus_position_mae"
            )

            primary_mae.append(
                row[
                    "focus_position_mae"
                ]
            )

        else:

            primary_metric.append(
                "timestamp_mae_ms"
            )

            primary_mae.append(
                row[
                    "timestamp_mae_ms"
                ]
            )

    grouped[
        "primary_metric"
    ] = primary_metric

    grouped[
        "primary_mae"
    ] = primary_mae

    grouped[
        "delta_vs_full"
    ] = np.nan

    grouped[
        "relative_change_vs_full_pct"
    ] = np.nan

    for dataset in grouped[
        "dataset"
    ].unique():

        mask_dataset = (
            grouped[
                "dataset"
            ]
            == dataset
        )

        full_rows = grouped[
            mask_dataset
            &
            (
                grouped[
                    "variant"
                ]
                == "Full V4"
            )
        ]

        if full_rows.empty:
            continue

        full_value = float(
            full_rows[
                "primary_mae"
            ].iloc[0]
        )

        if not np.isfinite(
            full_value
        ):
            continue

        for index in grouped[
            mask_dataset
        ].index:

            current = float(
                grouped.at[
                    index,
                    "primary_mae",
                ]
            )

            grouped.at[
                index,
                "delta_vs_full",
            ] = (
                current
                - full_value
            )

            if abs(
                full_value
            ) > EPS:

                grouped.at[
                    index,
                    "relative_change_vs_full_pct",
                ] = (
                    (
                        current
                        - full_value
                    )
                    /
                    full_value
                    *
                    100.0
                )

    return grouped


# ============================================================================
# Plots
# ============================================================================

def plot_dataset_summary(
    summary: pd.DataFrame,
    dataset: str,
    output_dir: Path,
) -> None:

    group = summary[
        summary.dataset
        == dataset
    ].copy()

    if group.empty:
        return

    group = group.sort_values(
        "primary_mae",
        ascending=True,
    )

    values = group[
        "primary_mae"
    ].to_numpy(
        dtype=float
    )

    labels = group[
        "variant"
    ].tolist()

    unit = (
        "focus-position units"
        if dataset in POSITION_DATASETS
        else "ms"
    )

    fig, axis = plt.subplots(
        figsize=(
            10,
            max(
                6,
                0.45 * len(group),
            ),
        ),
        constrained_layout=True,
    )

    y = np.arange(
        len(group)
    )

    bars = axis.barh(
        y,
        values,
    )

    axis.set_yticks(
        y,
        labels,
    )

    axis.invert_yaxis()

    for bar, value in zip(
        bars,
        values,
    ):
        if np.isfinite(
            value
        ):
            axis.text(
                value,
                bar.get_y()
                + bar.get_height()
                / 2.0,
                f" {value:.4g}",
                va="center",
            )

    axis.set_xlabel(
        f"Primary MAE ({unit})"
    )

    axis.set_title(
        f"{dataset} — Fusion-v4 ablation study"
    )

    axis.grid(
        True,
        axis="x",
        alpha=0.25,
    )

    fig.savefig(
        output_dir
        /
        f"{dataset}_ablation_mae.png",
        dpi=220,
        facecolor="white",
    )

    plt.close(
        fig
    )


def plot_heatmap(
    summary: pd.DataFrame,
    datasets: list[str],
    metric: str,
    title: str,
    output_path: Path,
) -> None:

    subset = summary[
        summary.dataset.isin(
            datasets
        )
    ]

    if subset.empty:
        return

    variants = [
        item.name
        for item
        in make_variants()
    ]

    matrix = np.full(
        (
            len(datasets),
            len(variants),
        ),
        np.nan,
        dtype=float,
    )

    for row_index, dataset in enumerate(
        datasets
    ):
        for column_index, variant in enumerate(
            variants
        ):
            current = subset[
                (
                    subset.dataset
                    == dataset
                )
                &
                (
                    subset.variant
                    == variant
                )
            ]

            if not current.empty:
                matrix[
                    row_index,
                    column_index,
                ] = float(
                    current[
                        metric
                    ].iloc[0]
                )

    fig, axis = plt.subplots(
        figsize=(
            max(
                14,
                1.15 * len(variants),
            ),
            5.5,
        ),
        constrained_layout=True,
    )

    image = axis.imshow(
        matrix,
        aspect="auto",
    )

    axis.set_xticks(
        np.arange(
            len(variants)
        ),
        variants,
        rotation=55,
        ha="right",
    )

    axis.set_yticks(
        np.arange(
            len(datasets)
        ),
        datasets,
    )

    axis.set_title(
        title
    )

    for i in range(
        matrix.shape[0]
    ):
        for j in range(
            matrix.shape[1]
        ):
            value = matrix[
                i,
                j,
            ]

            if np.isfinite(
                value
            ):
                axis.text(
                    j,
                    i,
                    f"{value:.3g}",
                    ha="center",
                    va="center",
                    fontsize=8,
                )

    fig.colorbar(
        image,
        ax=axis,
        label="MAE",
    )

    fig.savefig(
        output_path,
        dpi=220,
        facecolor="white",
    )

    plt.close(
        fig
    )


# ============================================================================
# Main
# ============================================================================

def main() -> int:

    parser = argparse.ArgumentParser(
        description=__doc__
    )

    parser.add_argument(
        "--data-root",
        type=Path,
        default=(
            PROJECT_ROOT
            / "_datasets"
        ),
        help=(
            "Root containing SYN, DAVIS, EVK4, EvHF and EAD. "
            "Default: project_root/_datasets"
        ),
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
    )

    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=protocol.DATASETS,
        default=list(
            protocol.DATASETS
        ),
    )

    parser.add_argument(
        "--max-sequences",
        type=int,
        default=None,
        help=(
            "Optional smoke-test limit PER DATASET. "
            "Example: --max-sequences 5"
        ),
    )

    parser.add_argument(
        "--save-curves",
        action="store_true",
        help=(
            "Save one time-series CSV per variant and sequence. "
            "This can consume substantial disk space."
        ),
    )

    args = parser.parse_args()

    output_root = (
        args.output.resolve()
    )

    output_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    variants = make_variants()

    all_rows: list[
        dict[str, Any]
    ] = []

    failures: list[
        dict[str, str]
    ] = []

    print()
    print("=" * 78)
    print("Fusion-v4 Ablation Study")
    print("=" * 78)
    print(
        f"V4 module       : {Path(fv4.__file__).resolve()}"
    )
    print(
        f"Protocol module : {Path(protocol.__file__).resolve()}"
    )
    print(
        f"Data root       : {args.data_root.resolve()}"
    )
    print(
        f"Output          : {output_root}"
    )
    print(
        f"Variants        : {len(variants)}"
    )
    print("=" * 78)
    print()

    total_processed = 0

    for dataset in args.datasets:

        dataset_root = (
            args.data_root
            / dataset
        )

        sequence_dirs = list(
            protocol.discover_event_sequences(
                dataset_root
            )
        )

        if (
            args.max_sequences
            is not None
        ):
            sequence_dirs = (
                sequence_dirs[
                    :args.max_sequences
                ]
            )

        print(
            f"[{dataset}] "
            f"{len(sequence_dirs)} sequences"
        )

        for sequence_number, directory in enumerate(
            sequence_dirs,
            1,
        ):

            try:

                # ------------------------------------------------------------
                # COMMON INPUT
                # ------------------------------------------------------------

                data = protocol.load_sequence(
                    dataset,
                    directory,
                    dataset_root,
                )

                frame = (
                    protocol.frame_from_data(
                        data
                    )
                )

                sequence = (
                    data.relative_name
                )

                safe_sequence = (
                    protocol.safe_sequence_name(
                        sequence
                    )
                )

                sequence_output = (
                    output_root
                    / dataset
                    / "sequences"
                    / safe_sequence
                )

                sequence_output.mkdir(
                    parents=True,
                    exist_ok=True,
                )

                # ------------------------------------------------------------
                # IMPORTANT:
                # ALL PREDICTIONS ARE MADE BEFORE GT IS LOADED
                # ------------------------------------------------------------

                predictions: dict[
                    str,
                    float | None
                ] = {}

                runtimes: dict[
                    str,
                    float
                ] = {}

                metadata: dict[
                    str,
                    dict[str, Any]
                ] = {}

                for variant in variants:

                    (
                        estimate,
                        curve,
                        meta,
                        runtime_ms,
                    ) = predict_timestamp(
                        data=data,
                        frame=frame,
                        config=variant,
                    )

                    predictions[
                        variant.name
                    ] = estimate

                    runtimes[
                        variant.name
                    ] = runtime_ms

                    metadata[
                        variant.name
                    ] = meta

                    if (
                        args.save_curves
                        and not curve.empty
                    ):
                        curve_dir = (
                            sequence_output
                            / "curves"
                        )

                        curve_dir.mkdir(
                            parents=True,
                            exist_ok=True,
                        )

                        curve.to_csv(
                            curve_dir
                            /
                            (
                                slugify(
                                    variant.name
                                )
                                + ".csv"
                            ),
                            index=False,
                            encoding="utf-8-sig",
                        )

                # ------------------------------------------------------------
                # LOAD GT ONLY AFTER ALL PREDICTIONS EXIST
                # ------------------------------------------------------------

                evaluation = (
                    protocol.evaluation_metadata(
                        data
                    )
                )

                mapper = evaluation.pop(
                    "position_mapper",
                    None,
                )

                gt_timestamp = float(
                    evaluation[
                        "gt_timestamp_us"
                    ]
                )

                gt_position = (
                    evaluation.get(
                        "gt_focus_position"
                    )
                )

                condition = (
                    evaluation.get(
                        "condition",
                        "",
                    )
                )

                position_unit = (
                    evaluation.get(
                        "position_unit"
                    )
                )

                sequence_rows = []

                for variant in variants:

                    estimate = predictions[
                        variant.name
                    ]

                    predicted_position = (
                        None
                        if (
                            mapper is None
                            or estimate is None
                        )
                        else mapper(
                            float(
                                estimate
                            )
                        )
                    )

                    valid_prediction = (
                        estimate is not None
                    )

                    row = {
                        "dataset": dataset,

                        "sequence": sequence,

                        "condition": condition,

                        "variant": variant.name,

                        "valid": valid_prediction,

                        "predicted_timestamp_us": estimate,

                        "gt_timestamp_us": gt_timestamp,

                        "signed_error_ms": (
                            None
                            if estimate is None
                            else (
                                float(
                                    estimate
                                )
                                -
                                gt_timestamp
                            )
                            /
                            1000.0
                        ),

                        "abs_error_ms": (
                            None
                            if estimate is None
                            else abs(
                                float(
                                    estimate
                                )
                                -
                                gt_timestamp
                            )
                            /
                            1000.0
                        ),

                        "predicted_focus_position": (
                            predicted_position
                        ),

                        "gt_focus_position": (
                            gt_position
                        ),

                        "signed_position_error": (
                            None
                            if (
                                predicted_position
                                is None
                                or gt_position
                                is None
                            )
                            else (
                                predicted_position
                                -
                                float(
                                    gt_position
                                )
                            )
                        ),

                        "abs_position_error": (
                            None
                            if (
                                predicted_position
                                is None
                                or gt_position
                                is None
                            )
                            else abs(
                                predicted_position
                                -
                                float(
                                    gt_position
                                )
                            )
                        ),

                        "position_unit": (
                            position_unit
                        ),

                        "runtime_ms": (
                            runtimes[
                                variant.name
                            ]
                        ),

                        "event_count_used": int(
                            data.total.sum()
                        ),

                        "bin_count": int(
                            len(frame)
                        ),

                        "spatial_scope": (
                            "official_roi"
                            if data.roi is not None
                            else "full_sensor"
                        ),

                        "roi": (
                            None
                            if data.roi is None
                            else str(
                                data.roi
                            )
                        ),
                    }

                    sequence_rows.append(
                        row
                    )

                    all_rows.append(
                        row
                    )

                pd.DataFrame(
                    sequence_rows
                ).to_csv(
                    sequence_output
                    /
                    "ablation_results.csv",
                    index=False,
                    encoding="utf-8-sig",
                )

                audit = {
                    "dataset": dataset,

                    "sequence": sequence,

                    "event_file": str(
                        data.event_path
                    ),

                    "protocol_version": (
                        protocol.PROTOCOL_VERSION
                    ),

                    "official_roi": (
                        data.roi
                    ),

                    "spatial_scope": (
                        "official_roi"
                        if data.roi is not None
                        else "full_sensor"
                    ),

                    "common_time_interval_us": [
                        float(
                            data.start_us
                        ),
                        float(
                            data.end_us
                        ),
                    ],

                    "common_event_count": int(
                        data.total.sum()
                    ),

                    "common_bin_count": int(
                        len(frame)
                    ),

                    "ground_truth_loaded_after_all_predictions": True,

                    "predictions": predictions,

                    "evaluation": evaluation,

                    "variant_metadata": metadata,
                }

                (
                    sequence_output
                    /
                    "ablation_audit.json"
                ).write_text(
                    json.dumps(
                        fv4.json_safe(
                            audit
                        ),
                        ensure_ascii=False,
                        indent=2,
                    ),
                    encoding="utf-8",
                )

                total_processed += 1

                if (
                    sequence_number % 10 == 0
                    or sequence_number
                    == len(sequence_dirs)
                ):
                    print(
                        f"  processed "
                        f"{sequence_number}/"
                        f"{len(sequence_dirs)}"
                    )

            except Exception as error:

                print(
                    f"  ERROR: {directory}"
                )

                print(
                    f"         "
                    f"{type(error).__name__}: "
                    f"{error}"
                )

                failures.append(
                    {
                        "dataset": dataset,
                        "sequence_dir": str(
                            directory
                        ),
                        "error": (
                            f"{type(error).__name__}: "
                            f"{error}"
                        ),
                    }
                )

    # ========================================================================
    # Save all results
    # ========================================================================

    results = pd.DataFrame(
        all_rows
    )

    results.to_csv(
        output_root
        /
        "all_ablation_results.csv",
        index=False,
        encoding="utf-8-sig",
    )

    pd.DataFrame(
        failures
    ).to_csv(
        output_root
        /
        "failures.csv",
        index=False,
        encoding="utf-8-sig",
    )

    summary = build_summary(
        results
    )

    summary.to_csv(
        output_root
        /
        "summary_by_dataset.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # ========================================================================
    # Dataset plots
    # ========================================================================

    if not summary.empty:

        for dataset in summary[
            "dataset"
        ].unique():

            dataset_output = (
                output_root
                / dataset
            )

            dataset_output.mkdir(
                parents=True,
                exist_ok=True,
            )

            plot_dataset_summary(
                summary,
                dataset,
                dataset_output,
            )

        plot_heatmap(
            summary=summary,
            datasets=[
                dataset
                for dataset
                in [
                    "SYN",
                    "DAVIS",
                    "EVK4",
                ]
                if dataset
                in summary.dataset.unique()
            ],
            metric="timestamp_mae_ms",
            title=(
                "Fusion-v4 Ablation Study — "
                "Timestamp MAE (ms)"
            ),
            output_path=(
                output_root
                /
                "timestamp_mae_heatmap.png"
            ),
        )

        plot_heatmap(
            summary=summary,
            datasets=[
                dataset
                for dataset
                in [
                    "EAD",
                    "EvHF",
                ]
                if dataset
                in summary.dataset.unique()
            ],
            metric="focus_position_mae",
            title=(
                "Fusion-v4 Ablation Study — "
                "Focus-position MAE"
            ),
            output_path=(
                output_root
                /
                "position_mae_heatmap.png"
            ),
        )

    # ========================================================================
    # Manifest
    # ========================================================================

    manifest = {
        "study": (
            "Fusion-v4 ablation study"
        ),

        "fusion_version": getattr(
            fv4,
            "VERSION",
            None,
        ),

        "fusion_file": str(
            Path(
                fv4.__file__
            ).resolve()
        ),

        "protocol_version": (
            protocol.PROTOCOL_VERSION
        ),

        "protocol_file": str(
            Path(
                protocol.__file__
            ).resolve()
        ),

        "data_root": str(
            args.data_root.resolve()
        ),

        "output_root": str(
            output_root
        ),

        "candidate_boundary_rule": (
            "All ablations retain Full-V4 "
            f"reference radius = {REFERENCE_RADIUS} samples."
        ),

        "ground_truth_rule": (
            "Ground truth is loaded only after "
            "all ablation predictions are complete."
        ),

        "input_rule": (
            "Same comparison_protocol SequenceData, "
            "same ROI, same acquisition interval, "
            "same 1 ms aggregation."
        ),

        "coefficient_rule": (
            "Fusion-v4 frozen coefficients are never "
            "retrained for an ablation."
        ),

        "variants": [
            asdict(
                variant
            )
            for variant
            in variants
        ],

        "processed_sequences": int(
            total_processed
        ),

        "failure_count": int(
            len(
                failures
            )
        ),
    }

    (
        output_root
        /
        "ablation_manifest.json"
    ).write_text(
        json.dumps(
            fv4.json_safe(
                manifest
            ),
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print()
    print("=" * 78)
    print("Finished")
    print("=" * 78)
    print(
        f"Processed sequences : {total_processed}"
    )
    print(
        f"Failures            : {len(failures)}"
    )
    print(
        f"Results             : {output_root}"
    )
    print()

    if not summary.empty:
        display_columns = [
            "dataset",
            "variant",
            "valid_count",
            "success_rate",
            "timestamp_mae_ms",
            "focus_position_mae",
            "delta_vs_full",
            "relative_change_vs_full_pct",
        ]

        print(
            summary[
                display_columns
            ].to_string(
                index=False
            )
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(
        main()
    )