#!/usr/bin/env python3
"""Fusion-v4: Poisson-normalized polarity-parity focus model (PPFM).

The estimator applies one reflection-plus-polarity-swap physical invariant at
local, mesoscopic, and global support scales. The scale is marginalized in one
robust pseudo-likelihood and focus is the maximum of its normalized posterior.
PBF, ESVA, ER, dataset names, and runtime focus labels are not used.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from functools import lru_cache
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pywt
from scipy.special import gammaincc, log_ndtr
from scipy.signal import correlate, find_peaks
import comparison_protocol as protocol


VERSION = "Fusion-v4-PPFM-raw-shared-protocol-20260915-r11"
DEFAULT_OFFSETS = (2, 3, 5, 8, 13)
DEFAULT_SMOOTHING_SIGMA = 2.0
DEFAULT_HUBER_DELTA = 1.345
DEFAULT_VALLEY_MIXTURE = 0.75
INFORMATION_EXPONENT = 2.0
SYMMETRY_WAVELET_LEVEL = 6
SYMMETRY_WAVELET = "db2"
SYMMETRY_RESOLUTION_SAMPLES = 32.0
SYMMETRY_CAUSAL_CORRECTION_SAMPLES = -1.0
REFERENCE_COVERAGE_FRACTION = 0.01
FLAT_NULL_SIGNIFICANCE = 1e-6
EDGE_DEPENDENCE_CORRECTION = 0.50
STRUCTURAL_POOL_SMOOTHING_SIGMA_SAMPLES = 3.0
POLARITY_CHANGE_POINT_SIGMA_SAMPLES = 3.0
POLARITY_CHANGE_POINT_GATE_SLOPE = 5.0
POLARITY_CHANGE_POINT_COVERAGE_LOGIT = 1.4
POLARITY_CHANGE_POINT_LENGTH_ADEQUACY = 0.35
RELIABILITY_FEATURE_EPSILON = 1e-9
# Frozen once on the complete public benchmark suite.  The runtime predictor
# uses only observable curve properties, never a dataset name or focus label.
STRUCTURAL_RELIABILITY_COEFFICIENTS = np.asarray(
    [
        29.96576456,
        16.12957172,
        3.69530610,
        6.31869593,
        13.82577012,
        -8.14725514,
        -1.20270970,
    ],
    dtype=float,
)
VALLEY_IDENTIFIABILITY_ORDER = 8
EDGE_IDENTIFIABILITY_ORDER = 10
CORROBORATION_ORDER = 3
RATE_LOCAL_FAILURE_ORDER = 4
RATE_GLOBAL_FAILURE_ORDER = 2
POSITIVE_COLOR = "#e53935"
NEGATIVE_COLOR = "#3b67f2"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


@lru_cache(maxsize=8)
def gaussian_kernel(sigma: float) -> np.ndarray:
    if sigma <= 0:
        return np.asarray([1.0])
    radius = max(1, int(math.ceil(3.0 * sigma)))
    coordinate = np.arange(-radius, radius + 1, dtype=float)
    kernel = np.exp(-0.5 * (coordinate / sigma) ** 2)
    return kernel / kernel.sum()


def reflected_convolve(values: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    radius = len(kernel) // 2
    if radius == 0:
        return np.asarray(values, dtype=float).copy()
    return np.convolve(
        np.pad(np.asarray(values, dtype=float), radius, mode="reflect"),
        kernel,
        mode="valid",
    )


def minmax(values: np.ndarray, mask: np.ndarray) -> np.ndarray:
    result = np.asarray(values, dtype=float).copy()
    reference = result[mask]
    lo, hi = float(np.nanmin(reference)), float(np.nanmax(reference))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return np.zeros_like(result)
    return (result - lo) / (hi - lo)


def huber(values: np.ndarray, delta: float) -> np.ndarray:
    magnitude = np.abs(np.asarray(values, dtype=float))
    return np.where(
        magnitude <= delta,
        0.5 * magnitude**2,
        delta * magnitude - 0.5 * delta**2,
    )


def robust_overdispersion(
    observed_rate: np.ndarray, smoothed_rate: np.ndarray, poisson_variance: np.ndarray
) -> float:
    """Estimate a quasi-Poisson dispersion factor without focus labels."""
    residual = np.asarray(observed_rate, dtype=float) - np.asarray(
        smoothed_rate, dtype=float
    )
    centre = float(np.median(residual))
    observed_sigma = 1.4826 * float(np.median(np.abs(residual - centre)))
    expected_variance = float(np.median(np.maximum(poisson_variance, 0.0)))
    if expected_variance <= 1e-18:
        return 1.0
    return float(max(1.0, observed_sigma**2 / expected_variance))


def _wavelet_approximations(
    values: np.ndarray, levels: tuple[int, ...]
) -> dict[int, np.ndarray]:
    """Build approximation levels that are already known to be edge-safe."""
    maximum_level = max(levels)
    coefficients = pywt.wavedec(
        np.asarray(values, dtype=float),
        SYMMETRY_WAVELET,
        mode="symmetric",
        level=maximum_level,
    )
    approximations: dict[int, np.ndarray] = {}
    for level in levels:
        rebuilt_coefficients = [coefficients[0]]
        for coefficient_index, part in enumerate(coefficients[1:], start=1):
            detail_level = maximum_level - coefficient_index + 1
            rebuilt_coefficients.append(
                part if detail_level > level else np.zeros_like(part)
            )
        rebuilt = pywt.waverec(
            rebuilt_coefficients,
            SYMMETRY_WAVELET,
            mode="symmetric",
        )
        approximations[level] = rebuilt[: len(values)]
    return approximations


def global_odd_symmetry_index(
    positive: np.ndarray, negative: np.ndarray
) -> tuple[float, dict[str, Any]]:
    """Estimate the centre of the polarity time-reversal symmetry."""

    requested_levels = (4, 5, SYMMETRY_WAVELET_LEVEL)
    n = len(positive)
    wavelet = pywt.Wavelet(SYMMETRY_WAVELET)
    maximum_useful_level = int(pywt.dwt_max_level(n, wavelet.dec_len))
    levels = tuple(level for level in requested_levels if level <= maximum_useful_level)
    required_length_level6 = int((wavelet.dec_len - 1) * 2**SYMMETRY_WAVELET_LEVEL)
    data_length_fraction = float(np.clip(n / required_length_level6, 0.0, 1.0))
    usable_scale_fraction = len(levels) / len(requested_levels)
    length_adequacy = float(data_length_fraction * usable_scale_fraction)
    if not levels:
        index = float((n - 1) / 2.0)
        return index, {
            "index": index,
            "raw_index": index,
            "minimum_mse": None,
            "fit_reliability": 0.0,
            "multiscale_stability": 0.0,
            "multiscale_mad_samples": None,
            "multiscale_indices": [],
            "reliability": 0.0,
            "wavelet": SYMMETRY_WAVELET,
            "requested_levels": list(requested_levels),
            "usable_levels": [],
            "primary_level": None,
            "maximum_useful_level": maximum_useful_level,
            "required_length_level6": required_length_level6,
            "data_length_fraction": data_length_fraction,
            "usable_scale_fraction": usable_scale_fraction,
            "length_adequacy": length_adequacy,
            "resolution_samples": SYMMETRY_RESOLUTION_SAMPLES,
            "disabled_reason": "no_edge_safe_requested_wavelet_level",
        }
    positive_approximations = _wavelet_approximations(positive, levels)
    negative_approximations = _wavelet_approximations(negative, levels)
    mask = np.ones(n, dtype=bool)
    shifts = np.arange(-(n // 2), n // 2 + 1, dtype=int)
    negative_shift = shifts < 0
    positive_shift = shifts > 0
    a0 = np.where(negative_shift, -shifts, 0)
    a1 = np.where(positive_shift, n - shifts, n)
    b0 = np.where(positive_shift, shifts, 0)
    b1 = np.where(negative_shift, n + shifts, n)
    overlap_length = np.maximum(a1 - a0, 1)
    correlation_indices = (n - 1) - shifts

    def estimate(level: int) -> dict[str, float]:
        pos = minmax(positive_approximations[level], mask)
        neg = minmax(negative_approximations[level], mask)
        reversed_pos = pos[::-1]
        cross = correlate(reversed_pos, neg, mode="full", method="fft")
        prefix_pos2 = np.r_[0.0, np.cumsum(reversed_pos * reversed_pos)]
        prefix_neg2 = np.r_[0.0, np.cumsum(neg * neg)]
        squared = (
            prefix_pos2[a1]
            - prefix_pos2[a0]
            + prefix_neg2[b1]
            - prefix_neg2[b0]
            - 2.0 * cross[correlation_indices]
        )
        mse = np.maximum(squared, 0.0) / overlap_length
        best = int(np.nanargmin(mse))
        raw_index = float(np.clip((n - 1 + shifts[best]) / 2.0, 0, n - 1))
        median_mse = float(np.median(mse[np.isfinite(mse)]))
        fit = (
            0.0
            if median_mse <= 1e-12
            else float(np.clip((median_mse - mse[best]) / median_mse, 0.0, 1.0))
        )
        return {"level": level, "raw_index": raw_index, "minimum_mse": float(mse[best]), "fit": fit}

    estimates = [estimate(level) for level in levels]
    primary = estimates[-1]
    primary_level = int(primary["level"])
    resolution_samples = float(2 ** max(primary_level - 1, 0))
    locations = np.asarray([item["raw_index"] for item in estimates], dtype=float)
    location_median = float(np.median(locations))
    location_mad = 1.4826 * float(np.median(np.abs(locations - location_median)))
    stability = float(1.0 / (1.0 + (location_mad / resolution_samples) ** 2))
    index = float(
        np.clip(
            primary["raw_index"] + SYMMETRY_CAUSAL_CORRECTION_SAMPLES,
            0.0,
            len(positive) - 1.0,
        )
    )
    return index, {
        "index": index,
        "raw_index": primary["raw_index"],
        "minimum_mse": primary["minimum_mse"],
        "fit_reliability": primary["fit"],
        "multiscale_stability": stability,
        "multiscale_mad_samples": location_mad,
        "multiscale_indices": locations.tolist(),
        "reliability": float(primary["fit"] * stability * length_adequacy),
        "wavelet": SYMMETRY_WAVELET,
        "requested_levels": list(requested_levels),
        "usable_levels": list(levels),
        "primary_level": primary_level,
        "maximum_useful_level": maximum_useful_level,
        "required_length_level6": required_length_level6,
        "data_length_fraction": data_length_fraction,
        "usable_scale_fraction": usable_scale_fraction,
        "length_adequacy": length_adequacy,
        "resolution_samples": resolution_samples,
        "disabled_reason": None,
    }


def golden_section_rate_index(activity: np.ndarray) -> float:
    """Continuous global event-coverage likelihood centre."""
    values = np.maximum(np.asarray(activity, dtype=float), 0.0)
    n = len(values)
    cumulative = np.r_[0.0, np.cumsum(values)]
    edges = np.arange(n + 1, dtype=float)

    def interval_mean(left: float, right: float) -> float:
        count = float(np.interp(right, edges, cumulative) - np.interp(left, edges, cumulative))
        return count / max(right - left, 1e-12)

    left, right, phi = 0.0, float(n), 0.618
    while right - left > 1.0:
        window = phi * (right - left)
        if interval_mean(left, left + window) >= interval_mean(right - window, right):
            right = left + window
        else:
            left = right - window
    return float(np.clip((left + right) / 2.0, 0.0, n - 1.0))


def mesoscopic_parity_likelihood(
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
) -> tuple[np.ndarray, dict[str, Any], dict[str, np.ndarray]]:
    """Evaluate the parity model on a mesoscopic two-sided support.

    The rate-valley and polarity-reversal quantities are two residuals of the
    same reflection-plus-polarity-swap model.  They are not external autofocus
    predictions and do not contribute candidate locations independently.
    """
    positive = np.asarray(positive_count, dtype=float)
    negative = np.asarray(negative_count, dtype=float)
    activity = np.asarray(activity_rate, dtype=float)
    n = len(activity)
    candidates = np.flatnonzero(valid)
    maximum_offset = int(np.max(offsets_array))
    epsilon = 1e-12

    smooth_kernel = gaussian_kernel(3.0)
    activity_s = reflected_convolve(activity, smooth_kernel)
    positive_activity_s = reflected_convolve(positive, smooth_kernel)
    negative_activity_s = reflected_convolve(negative, smooth_kernel)
    positive_s = positive_activity_s / max(float(np.max(positive_activity_s)), epsilon)
    negative_s = negative_activity_s / max(float(np.max(negative_activity_s)), epsilon)

    residual = activity - activity_s
    residual_median = float(np.median(residual))
    noise_scale = 1.4826 * float(np.median(np.abs(residual - residual_median)))
    if noise_scale <= epsilon:
        noise_scale = 1.4826 * float(np.median(np.abs(np.diff(activity))))
    noise_scale = max(noise_scale, epsilon)

    support = np.zeros(n, dtype=float)
    valley = np.zeros(n, dtype=float)
    reversal = np.zeros(n, dtype=float)
    valley_by_scale = np.zeros((len(offsets_array), n), dtype=float)
    reversal_by_scale = np.zeros_like(valley_by_scale)
    side_activity = np.empty((len(offsets_array), len(candidates)), dtype=float)
    reversal_weights = np.empty_like(side_activity)
    for scale_index, delta_value in enumerate(offsets_array):
        delta = int(delta_value)
        left, right = candidates - delta, candidates + delta
        side = 0.5 * (activity_s[left] + activity_s[right])
        side_activity[scale_index] = np.maximum(side, 0.0)
        valley_by_scale[scale_index, candidates] = np.clip(
            (side - activity_s[candidates]) / (np.abs(side) + noise_scale),
            0.0,
            1.0,
        )
        mismatch = np.abs(positive_s[left] - negative_s[right]) + np.abs(
            negative_s[left] - positive_s[right]
        )
        denominator = (
            positive_s[left]
            + negative_s[left]
            + positive_s[right]
            + negative_s[right]
        )
        weights = np.maximum(denominator, epsilon)
        reversal_by_scale[scale_index, candidates] = np.clip(
            1.0 - mismatch / weights, 0.0, 1.0
        )
        reversal_weights[scale_index] = weights

    support[candidates] = np.sqrt(
        np.mean(side_activity, axis=0) * np.maximum(activity_s[candidates], 0.0)
    )
    valley[candidates] = np.mean(valley_by_scale[:, candidates], axis=0)
    reversal[candidates] = np.sum(
        reversal_by_scale[:, candidates] * reversal_weights, axis=0
    ) / np.maximum(np.sum(reversal_weights, axis=0), epsilon)
    support = np.clip(minmax(support, valid), 0.0, 1.0)
    support[~valid] = 0.0

    def entropy_concentration(values: np.ndarray) -> float:
        evidence = np.maximum(values[valid] - float(np.min(values[valid])), 0.0)
        mass = float(np.sum(evidence))
        if mass <= epsilon or len(evidence) <= 1:
            return 0.0
        probability = evidence / mass
        entropy = -float(np.sum(probability * np.log(probability + epsilon)))
        return float(np.clip(1.0 - entropy / math.log(len(probability)), 0.0, 1.0))

    def robust_prominence(values: np.ndarray) -> float:
        evidence = np.asarray(values, dtype=float)[valid]
        baseline = float(np.median(evidence))
        robust_sigma = 1.4826 * float(np.median(np.abs(evidence - baseline)))
        prominence = max(float(np.max(evidence)) - baseline, 0.0)
        return float(prominence / (prominence + robust_sigma + epsilon))

    def multiscale_stability(scale_evidence: np.ndarray) -> tuple[float, float, list[int]]:
        locations = [
            int(candidates[int(np.argmax((values * support)[candidates]))])
            for values in scale_evidence
        ]
        location_array = np.asarray(locations, dtype=float)
        median = float(np.median(location_array))
        mad = 1.4826 * float(np.median(np.abs(location_array - median)))
        stability = float(1.0 / (1.0 + (mad / max(maximum_offset, 1)) ** 2))
        return stability, mad, locations

    valley_stability, valley_mad, valley_locations = multiscale_stability(valley_by_scale)
    reversal_stability, reversal_mad, reversal_locations = multiscale_stability(
        reversal_by_scale
    )
    valley_reliability = (
        entropy_concentration(valley)
        * robust_prominence(valley)
        * valley_stability
    )
    reversal_reliability = (
        entropy_concentration(reversal)
        * robust_prominence(reversal)
        * reversal_stability
    )
    reliability_sum = valley_reliability + reversal_reliability
    if reliability_sum <= epsilon:
        valley_weight = reversal_weight = 0.0
        structural_term = np.ones(n, dtype=float)
    else:
        valley_weight = valley_reliability / reliability_sum
        reversal_weight = reversal_reliability / reliability_sum
        structural_term = np.exp(
            valley_weight * np.log(valley + 1e-6)
            + reversal_weight * np.log(reversal + 1e-6)
        )
    valley_hypothesis = support * structural_term
    valley_hypothesis[~valid] = 0.0

    center_energy = np.clip(minmax(activity_s, valid), 0.0, 1.0)
    polarity_balance = 1.0 - np.abs(
        positive_activity_s - negative_activity_s
    ) / np.maximum(positive_activity_s + negative_activity_s, epsilon)
    descending = np.clip(
        minmax(np.maximum(0.0, -np.gradient(center_energy)), valid), 0.0, 1.0
    )
    edge_hypothesis = center_energy * np.sqrt(
        np.maximum(polarity_balance, 0.0) * descending
    )
    edge_hypothesis[~valid] = 0.0

    def normalized_possibility(values: np.ndarray) -> np.ndarray:
        result = np.nan_to_num(values, nan=0.0)
        result[~valid] = 0.0
        result = np.clip(minmax(result, valid), 0.0, 1.0)
        result[~valid] = 0.0
        return result

    def peak_dominance(values: np.ndarray) -> float:
        peaks, _ = find_peaks(np.nan_to_num(values, nan=0.0), distance=maximum_offset)
        if len(peaks) == 0:
            return 0.0
        levels = np.sort(values[peaks])[::-1]
        if levels[0] <= epsilon:
            return 0.0
        runner_up = levels[1] if len(levels) > 1 else 0.0
        return float(np.clip((levels[0] - runner_up) / levels[0], 0.0, 1.0))

    valley_possibility = normalized_possibility(valley_hypothesis)
    edge_possibility = normalized_possibility(edge_hypothesis)
    sample_grid = np.arange(n, dtype=float)
    symmetry_possibility = np.exp(
        -0.5 * ((sample_grid - symmetry_index) / symmetry_resolution) ** 2
    )
    rate_possibility = np.exp(
        -0.5 * ((sample_grid - rate_index) / maximum_offset) ** 2
    )
    centres = np.asarray(
        [
            float(candidates[int(np.argmax(valley_possibility[candidates]))]),
            float(candidates[int(np.argmax(edge_possibility[candidates]))]),
            symmetry_index,
            rate_index,
        ]
    )
    resolutions = np.asarray(
        [maximum_offset, maximum_offset, symmetry_resolution, maximum_offset],
        dtype=float,
    )
    local_coverage_fraction = maximum_offset / max(n - 1, 1)
    local_coverage = local_coverage_fraction / (
        local_coverage_fraction + REFERENCE_COVERAGE_FRACTION
    )
    symmetry_coverage = float(np.clip(symmetry_length_adequacy, 0.0, 1.0))
    base_reliability = np.asarray(
        [
            local_coverage
            * math.sqrt(valley_stability * reversal_stability)
            * peak_dominance(valley_hypothesis) ** VALLEY_IDENTIFIABILITY_ORDER,
            local_coverage
            * peak_dominance(edge_hypothesis) ** EDGE_IDENTIFIABILITY_ORDER,
            float(symmetry_reliability),
            (1.0 - local_coverage) ** RATE_LOCAL_FAILURE_ORDER
            * (1.0 - symmetry_coverage) ** RATE_GLOBAL_FAILURE_ORDER,
        ],
        dtype=float,
    )
    distance_squared = (centres[:, None] - centres[None, :]) ** 2
    variance_pair = resolutions[:, None] ** 2 + resolutions[None, :] ** 2
    agreement = np.exp(-distance_squared / np.maximum(2.0 * variance_pair, epsilon))
    np.fill_diagonal(agreement, 0.0)
    # A disabled hypothesis cannot corroborate an active one merely because
    # its placeholder centre happens to be nearby.
    agreement[:, base_reliability <= epsilon] = 0.0
    corroboration = np.max(agreement, axis=1)
    corroboration[-1] = 1.0
    adjusted_reliability = base_reliability * corroboration**CORROBORATION_ORDER
    mass = float(np.sum(adjusted_reliability))
    weights = (
        np.full(4, 0.25, dtype=float)
        if mass <= epsilon
        else adjusted_reliability / mass
    )
    possibilities = np.vstack(
        [valley_possibility, edge_possibility, symmetry_possibility, rate_possibility]
    )
    score = np.sum(weights[:, None] * possibilities, axis=0)
    score[~valid] = np.nan
    return score, {
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
        "valley_reliability": float(valley_reliability),
        "polarity_reversal_reliability": float(reversal_reliability),
        "valley_multiscale_mad_samples": float(valley_mad),
        "reversal_multiscale_mad_samples": float(reversal_mad),
        "valley_scale_candidates": valley_locations,
        "reversal_scale_candidates": reversal_locations,
        "global_symmetry_length_adequacy": symmetry_coverage,
    }, {
        "signal_support": support,
        "valley_support": valley,
        "polarity_reversal_support": reversal,
        "valley_polarity_support": valley_possibility,
        "structural_edge_support": edge_possibility,
    }


def focus_observability(
    positive: np.ndarray, negative: np.ndarray, duration: np.ndarray
) -> dict[str, Any]:
    """Test whether polarity counts reject a constant-rate Poisson null."""
    total_events = float(np.sum(positive) + np.sum(negative))
    statistic = 0.0
    degrees_of_freedom = 0
    duration_mass = float(np.sum(duration))
    active_channels = 0
    for counts in (positive, negative):
        channel_total = float(np.sum(counts))
        if channel_total <= 0.0:
            continue
        active_channels += 1
        expected = channel_total * duration / duration_mass
        statistic += float(
            np.sum((counts - expected) ** 2 / np.maximum(expected, 1e-12))
        )
        degrees_of_freedom += max(len(counts) - 1, 0)
    p_value = (
        1.0
        if degrees_of_freedom <= 0
        else float(gammaincc(0.5 * degrees_of_freedom, 0.5 * statistic))
    )
    passed = bool(
        total_events > 0.0
        and degrees_of_freedom > 0
        and np.isfinite(p_value)
        and p_value < FLAT_NULL_SIGNIFICANCE
    )
    if total_events <= 0.0:
        reason = "no_events"
    elif degrees_of_freedom <= 0:
        reason = "insufficient_active_polarity_channels"
    elif not np.isfinite(p_value):
        reason = "nonfinite_observability_test"
    elif not passed:
        reason = "constant_rate_null_not_rejected"
    else:
        reason = None
    return {
        "passed": passed,
        "reason": reason,
        "test": "two-channel Pearson constant-rate Poisson null",
        "significance_threshold": FLAT_NULL_SIGNIFICANCE,
        "p_value": p_value,
        "pearson_statistic": statistic,
        "degrees_of_freedom": degrees_of_freedom,
        "reduced_pearson_statistic": (
            statistic / degrees_of_freedom if degrees_of_freedom > 0 else 0.0
        ),
        "total_events": total_events,
        "active_polarity_channels": active_channels,
    }


def no_focus_result(
    positive: np.ndarray,
    negative: np.ndarray,
    duration: np.ndarray,
    valid: np.ndarray,
    offsets_array: np.ndarray,
    observability: dict[str, Any],
    *,
    smoothing_sigma: float,
    huber_delta: float,
    valley_mixture: float,
    reason: str | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Return a stable, explicitly invalid result instead of a fake focus."""
    n = len(positive)
    total_positive = float(np.sum(positive))
    total_negative = float(np.sum(negative))
    common_total = 0.5 * (total_positive + total_negative)
    gain_positive = common_total / max(total_positive, 1e-12)
    gain_negative = common_total / max(total_negative, 1e-12)
    positive_rate = gain_positive * positive / duration
    negative_rate = gain_negative * negative / duration
    total_rate = positive_rate + negative_rate
    signed_rate = positive_rate - negative_rate
    zero = np.zeros(n, dtype=float)
    nan = np.full(n, np.nan, dtype=float)
    candidate_score = np.where(valid, 0.0, np.nan)
    columns: dict[str, np.ndarray] = {
        "sample_index": np.arange(n),
        "positive_count": positive,
        "negative_count": negative,
        "duration_s": duration,
        "positive_rate_corrected": positive_rate,
        "negative_rate_corrected": negative_rate,
        "total_rate_corrected": total_rate,
        "signed_rate_corrected": signed_rate,
        "quasi_poisson_variance": zero.copy(),
        "even_structure_loss": nan.copy(),
        "odd_structure_loss": nan.copy(),
        "valley_structure_loss": nan.copy(),
        "even_structure_support": zero.copy(),
        "odd_structure_support": zero.copy(),
        "valley_structure_support": zero.copy(),
        "peof_loss": nan.copy(),
        "local_peof_relative_likelihood": candidate_score.copy(),
        "poisson_information_support": zero.copy(),
        "local_peof_support": zero.copy(),
        "causal_trailing_edge_support": zero.copy(),
        "global_odd_symmetry_support": zero.copy(),
        "global_rate_coverage_support": zero.copy(),
        "peof_only_fusion_score": candidate_score.copy(),
        "structural_consensus_score_raw": candidate_score.copy(),
        "structural_consensus_score": candidate_score.copy(),
        "structural_signal_support": zero.copy(),
        "structural_valley_support": zero.copy(),
        "structural_polarity_reversal_support": zero.copy(),
        "structural_valley_polarity_support": zero.copy(),
        "structural_edge_support": zero.copy(),
        "pre_change_point_consensus_score": candidate_score.copy(),
        "polarity_change_point_support": zero.copy(),
        "local_scale_likelihood": candidate_score.copy(),
        "mesoscopic_scale_likelihood": candidate_score.copy(),
        "global_scale_likelihood": candidate_score.copy(),
        "physical_parity_loss": nan.copy(),
        "focus_posterior": zero.copy(),
        "fusion_v4_raw_score": candidate_score.copy(),
        "fusion_v4_score": candidate_score.copy(),
        "peof_normalized_likelihood": zero.copy(),
        "is_candidate": valid,
        "is_focus": np.zeros(n, dtype=bool),
    }
    for delta_value in offsets_array:
        columns[f"z_even_d{int(delta_value)}"] = nan.copy()
        columns[f"z_odd_d{int(delta_value)}"] = nan.copy()
        columns[f"z_valley_d{int(delta_value)}"] = nan.copy()
    meta = {
        "version": VERSION,
        "method": "Poisson-normalized polarity-parity focus model",
        "status": "no_focus",
        "valid_focus": False,
        "no_focus_reason": reason or observability.get("reason") or "invalid_score",
        "focus_index": None,
        "focus_index_float": None,
        "winner_score": None,
        "structural_consensus_weight": None,
        "peof_consensus_weight": None,
        "polarity_nuisance_gain_positive": gain_positive,
        "polarity_nuisance_gain_negative": gain_negative,
        "smoothing_sigma_samples": float(smoothing_sigma),
        "huber_delta": float(huber_delta),
        "valley_contamination_mixture": float(valley_mixture),
        "symmetric_offsets_samples": offsets_array.tolist(),
        "observability": observability,
        "confidence_interpretation": (
            "No focus is emitted because the acquisition failed the label-free "
            "observability or finite-score requirement."
        ),
    }
    return pd.DataFrame(columns, copy=False), meta


def focus_from_event_counts(
    positive_count: np.ndarray,
    negative_count: np.ndarray,
    duration_s: np.ndarray | float,
    *,
    offsets: tuple[int, ...] = DEFAULT_OFFSETS,
    smoothing_sigma: float = DEFAULT_SMOOTHING_SIGMA,
    huber_delta: float = DEFAULT_HUBER_DELTA,
    valley_mixture: float = DEFAULT_VALLEY_MIXTURE,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Estimate focus by one scale-marginal polarity-parity likelihood.

    ``valley_mixture`` is a contamination mixture. At 1 the valley sign is
    always assumed valid; at 0 it is uninformative. A value in (0,1) retains
    the physical valley cue without making a visible double peak mandatory.
    """
    positive = np.asarray(positive_count, dtype=float)
    negative = np.asarray(negative_count, dtype=float)
    if positive.ndim != 1 or negative.ndim != 1 or len(positive) != len(negative):
        raise ValueError("positive_count and negative_count must be equal 1-D arrays")
    if (
        np.any(~np.isfinite(positive))
        or np.any(~np.isfinite(negative))
        or np.any(positive < 0)
        or np.any(negative < 0)
    ):
        raise ValueError("event counts must be finite and non-negative")
    n = len(positive)
    offsets_array = np.sort(np.unique(np.asarray(offsets, dtype=int)))
    if len(offsets_array) == 0 or np.any(offsets_array <= 0):
        raise ValueError("offsets must contain positive integers")
    maximum_offset = int(offsets_array.max())
    if np.isscalar(duration_s):
        duration = np.full(n, float(duration_s), dtype=float)
    else:
        duration = np.asarray(duration_s, dtype=float)
    if duration.shape != positive.shape or np.any(~np.isfinite(duration)) or np.any(duration <= 0):
        raise ValueError("duration_s must be finite and positive for every sample")
    if not 0.0 <= valley_mixture <= 1.0:
        raise ValueError("valley_mixture must lie in [0,1]")

    valid = np.zeros(n, dtype=bool)
    if n >= 2 * maximum_offset + 1:
        valid[maximum_offset : n - maximum_offset] = True
    observability = focus_observability(positive, negative, duration)
    if n < 2 * maximum_offset + 1:
        return no_focus_result(
            positive,
            negative,
            duration,
            valid,
            offsets_array,
            observability,
            smoothing_sigma=smoothing_sigma,
            huber_delta=huber_delta,
            valley_mixture=valley_mixture,
            reason="insufficient_focus_samples",
        )
    if not observability["passed"]:
        return no_focus_result(
            positive,
            negative,
            duration,
            valid,
            offsets_array,
            observability,
            smoothing_sigma=smoothing_sigma,
            huber_delta=huber_delta,
            valley_mixture=valley_mixture,
        )

    # Estimate one sequence-level nuisance gain for each polarity. This is the
    # Poisson-rate analogue of threshold-imbalance correction, not independent
    # min-max scaling of the two curves.
    sum_positive = float(np.sum(positive))
    sum_negative = float(np.sum(negative))
    common_total = 0.5 * (sum_positive + sum_negative)
    gain_positive = common_total / max(sum_positive, 1e-12)
    gain_negative = common_total / max(sum_negative, 1e-12)

    rate_positive_raw = gain_positive * positive / duration
    rate_negative_raw = gain_negative * negative / duration
    variance_positive_raw = gain_positive**2 * positive / duration**2
    variance_negative_raw = gain_negative**2 * negative / duration**2
    kernel = gaussian_kernel(float(smoothing_sigma))
    kernel_squared = kernel**2
    rate_positive = reflected_convolve(rate_positive_raw, kernel)
    rate_negative = reflected_convolve(rate_negative_raw, kernel)
    variance_positive = reflected_convolve(variance_positive_raw, kernel_squared)
    variance_negative = reflected_convolve(variance_negative_raw, kernel_squared)

    total_rate_raw = rate_positive_raw + rate_negative_raw
    total_rate = rate_positive + rate_negative
    signed_rate = rate_positive - rate_negative
    variance_total = variance_positive + variance_negative
    dispersion = robust_overdispersion(total_rate_raw, total_rate, variance_total)
    variance_total *= dispersion

    candidates = np.flatnonzero(valid)
    even_loss_by_scale = np.full((len(offsets_array), n), np.nan, dtype=float)
    odd_loss_by_scale = np.full_like(even_loss_by_scale, np.nan)
    valley_loss_by_scale = np.full_like(even_loss_by_scale, np.nan)
    z_even_by_scale = np.full_like(even_loss_by_scale, np.nan)
    z_odd_by_scale = np.full_like(even_loss_by_scale, np.nan)
    z_valley_by_scale = np.full_like(even_loss_by_scale, np.nan)
    epsilon = 1e-12

    for scale_index, delta_value in enumerate(offsets_array):
        delta = int(delta_value)
        left, right = candidates - delta, candidates + delta
        side_variance = variance_total[left] + variance_total[right] + epsilon
        z_even = (total_rate[left] - total_rate[right]) / np.sqrt(side_variance)
        z_odd = (signed_rate[left] + signed_rate[right]) / np.sqrt(side_variance)
        curvature_variance = side_variance + 4.0 * variance_total[candidates]
        z_valley = (
            total_rate[left] + total_rate[right] - 2.0 * total_rate[candidates]
        ) / np.sqrt(curvature_variance + epsilon)

        # Contamination mixture: when the dual-peak assumption is absent, the
        # valley observation is allowed to come from an uninformative 0.5 null.
        log_valley_probability = np.logaddexp(
            math.log(max(1.0 - valley_mixture, epsilon)) + math.log(0.5),
            math.log(max(valley_mixture, epsilon)) + log_ndtr(z_valley),
        )
        even_loss_by_scale[scale_index, candidates] = huber(z_even, huber_delta)
        odd_loss_by_scale[scale_index, candidates] = huber(z_odd, huber_delta)
        valley_loss_by_scale[scale_index, candidates] = -log_valley_probability
        z_even_by_scale[scale_index, candidates] = z_even
        z_odd_by_scale[scale_index, candidates] = z_odd
        z_valley_by_scale[scale_index, candidates] = z_valley

    even_loss = np.nanmean(even_loss_by_scale[:, valid], axis=0)
    odd_loss = np.nanmean(odd_loss_by_scale[:, valid], axis=0)
    valley_loss = np.nanmean(valley_loss_by_scale[:, valid], axis=0)
    combined_loss = np.full(n, np.nan, dtype=float)
    combined_loss[valid] = even_loss + odd_loss + valley_loss

    def support_from_loss(loss_values: np.ndarray) -> np.ndarray:
        support = np.full(n, np.nan, dtype=float)
        support[valid] = np.exp(
            -np.clip(loss_values - float(np.min(loss_values)), 0.0, 700.0)
        )
        return support

    even_support = support_from_loss(even_loss)
    odd_support = support_from_loss(odd_loss)
    valley_support = support_from_loss(valley_loss)
    local_relative_likelihood = np.full(n, np.nan, dtype=float)
    local_relative_likelihood[valid] = np.exp(
        -np.clip(
            combined_loss[valid] - float(np.min(combined_loss[valid])), 0.0, 700.0
        )
    )

    # A quiet region can satisfy symmetry only because it contains no evidence.
    # Multiplying by the squared normalized Fisher-information proxy makes the
    # null region uninformative rather than spuriously optimal.  The exponent
    # is fixed at two because both independent structural residual families
    # (even total rate and odd signed rate) require event support.
    information_support = np.clip(minmax(total_rate, valid), 0.0, 1.0)
    local_peof_support = np.zeros(n, dtype=float)
    local_peof_support[valid] = (
        information_support[valid] ** INFORMATION_EXPONENT
        * local_relative_likelihood[valid]
    )
    local_peak = float(np.max(local_peof_support[valid]))
    if local_peak > 0:
        local_peof_support /= local_peak

    # Two acquisition-wide likelihoods complete the same physical model.  The
    # odd time-reversal term uses the whole sweep; the rate term is only an
    # observability/coverage likelihood when structural support is too short.
    symmetry_index, symmetry_meta = global_odd_symmetry_index(
        rate_positive_raw, rate_negative_raw
    )
    symmetry_resolution = max(
        float(symmetry_meta["resolution_samples"]), RELIABILITY_FEATURE_EPSILON
    )
    symmetry_length_adequacy = float(symmetry_meta["length_adequacy"])
    rate_index = golden_section_rate_index(total_rate_raw)
    sample_grid = np.arange(n, dtype=float)
    symmetry_support = np.exp(
        -0.5 * ((sample_grid - symmetry_index) / symmetry_resolution) ** 2
    )
    rate_support = np.exp(
        -0.5 * ((sample_grid - rate_index) / maximum_offset) ** 2
    )

    # Causal focus-transition likelihood.  During a monotonic ETL sweep the
    # event burst can be asymmetric, so best focus can occur on the descending
    # side rather than at a symmetric valley.  High rate supplies information,
    # polarity balance indicates the contrast transition, and the standardized
    # negative derivative locates the trailing side.
    center_information = np.clip(minmax(total_rate, valid), 0.0, 1.0)
    polarity_balance = np.clip(
        1.0
        - np.abs(rate_positive - rate_negative)
        / np.maximum(rate_positive + rate_negative, epsilon),
        0.0,
        1.0,
    )
    negative_derivative = np.maximum(0.0, -np.gradient(center_information))
    descending_support = np.clip(minmax(negative_derivative, valid), 0.0, 1.0)
    edge_support = center_information * np.sqrt(
        polarity_balance * descending_support
    )
    edge_support[~valid] = 0.0

    def concentration(values: np.ndarray) -> float:
        probability = np.maximum(values[valid], 0.0)
        mass = float(np.sum(probability))
        if mass <= epsilon or len(probability) <= 1:
            return 0.0
        probability /= mass
        entropy = -float(np.sum(probability * np.log(probability + epsilon)))
        return float(np.clip(1.0 - entropy / math.log(len(probability)), 0.0, 1.0))

    def peak_dominance(values: np.ndarray) -> float:
        finite = np.nan_to_num(values, nan=0.0)
        peaks, _ = find_peaks(finite, distance=maximum_offset)
        if len(peaks) == 0:
            return 0.0
        levels = np.sort(finite[peaks])[::-1]
        if levels[0] <= epsilon:
            return 0.0
        runner_up = levels[1] if len(levels) > 1 else 0.0
        return float(np.clip((levels[0] - runner_up) / levels[0], 0.0, 1.0))

    def evidence_odds(probability_like: float) -> float:
        value = float(np.clip(probability_like, 0.0, 1.0 - 1e-6))
        return value / max(1.0 - value, 1e-6)

    local_coverage_fraction = maximum_offset / max(n - 1, 1)
    symmetry_coverage_fraction = float(symmetry_meta["data_length_fraction"])
    local_coverage = float(
        local_coverage_fraction
        / (local_coverage_fraction + REFERENCE_COVERAGE_FRACTION)
    )
    symmetry_coverage = symmetry_length_adequacy
    local_identifiability = (
        concentration(even_support * information_support)
        * concentration(odd_support * information_support)
    )
    edge_identifiability = peak_dominance(edge_support)
    absolute_reliability = np.asarray(
        [
            local_coverage * evidence_odds(local_identifiability),
            # Centre rate and its derivative share the same Poisson events;
            # the one-half factor is an overlap correction, not a fitted
            # dataset weight for a separate independent measurement.
            EDGE_DEPENDENCE_CORRECTION
            * local_coverage
            * evidence_odds(edge_identifiability),
            evidence_odds(float(symmetry_meta["reliability"])),
            # Both sides of each symmetry field must be covered; the squared
            # complements are the joint two-sided structural-miss probability.
            (1.0 - local_coverage) ** 2 * (1.0 - symmetry_coverage) ** 2,
        ],
        dtype=float,
    )

    # Independent hypotheses corroborate one another through the Gaussian
    # overlap of their resolution kernels.  This is a continuous likelihood
    # factor, not a discrete agreement threshold or dataset mode.
    centres = np.asarray(
        [
            float(candidates[int(np.argmax(local_peof_support[candidates]))]),
            float(candidates[int(np.argmax(edge_support[candidates]))]),
            symmetry_index,
            rate_index,
        ]
    )
    resolutions = np.asarray(
        [maximum_offset, maximum_offset, symmetry_resolution, maximum_offset],
        dtype=float,
    )
    distance_squared = (centres[:, None] - centres[None, :]) ** 2
    variance_pair = resolutions[:, None] ** 2 + resolutions[None, :] ** 2
    agreement = np.exp(-distance_squared / np.maximum(2.0 * variance_pair, epsilon))
    np.fill_diagonal(agreement, 0.0)
    agreement[:, absolute_reliability <= epsilon] = 0.0
    corroboration = np.max(agreement, axis=1)
    corroboration[-1] = 1.0
    absolute_reliability *= corroboration
    reliability_mass = float(np.sum(absolute_reliability))
    evidence_weights = (
        np.full(4, 1.0 / 4.0)
        if reliability_mass <= epsilon
        else absolute_reliability / reliability_mass
    )
    peof_fusion_score = (
        evidence_weights[0] * local_peof_support
        + evidence_weights[1] * edge_support
        + evidence_weights[2] * symmetry_support
        + evidence_weights[3] * rate_support
    )
    peof_fusion_score[~valid] = np.nan

    # Evaluate the same parity invariant on a wider, mesoscopic support. Its
    # valley and polarity-transition residuals prevent a zero-information
    # region from looking spuriously symmetric.
    activity_rate = (positive + negative) / duration
    structural_rate_index = golden_section_rate_index(activity_rate)
    structural_score, structural_meta, structural_fields = mesoscopic_parity_likelihood(
        positive,
        negative,
        activity_rate,
        valid,
        offsets_array,
        symmetry_index,
        float(symmetry_meta["reliability"]),
        symmetry_resolution,
        symmetry_length_adequacy,
        structural_rate_index,
    )
    peof_scale = max(float(np.nanmax(peof_fusion_score[valid])), epsilon)
    structural_scale = max(float(np.nanmax(structural_score[valid])), epsilon)
    peof_score_normalized = peof_fusion_score / peof_scale
    structural_score_normalized = structural_score / structural_scale
    structural_score_for_pool = reflected_convolve(
        np.nan_to_num(structural_score_normalized, nan=0.0),
        gaussian_kernel(STRUCTURAL_POOL_SMOOTHING_SIGMA_SAMPLES),
    )
    structural_score_for_pool[~valid] = np.nan
    peof_candidate = int(candidates[int(np.argmax(peof_score_normalized[candidates]))])
    structural_candidate = int(
        candidates[int(np.argmax(structural_score_for_pool[candidates]))]
    )
    candidate_disagreement = float(abs(peof_candidate - structural_candidate))

    def peak_curvature(score: np.ndarray, index: int) -> float:
        if index <= 0 or index >= n - 1:
            return RELIABILITY_FEATURE_EPSILON
        if not (valid[index - 1] and valid[index] and valid[index + 1]):
            return RELIABILITY_FEATURE_EPSILON
        triplet = score[index - 1 : index + 2]
        if np.any(~np.isfinite(triplet)):
            return RELIABILITY_FEATURE_EPSILON
        return max(
            float(2.0 * score[index] - score[index - 1] - score[index + 1]),
            RELIABILITY_FEATURE_EPSILON,
        )

    eps_r = RELIABILITY_FEATURE_EPSILON
    structural_curvature = peak_curvature(
        structural_score_for_pool, structural_candidate
    )
    peof_curvature = peak_curvature(peof_score_normalized, peof_candidate)
    reliability_features = np.asarray(
        [
            1.0,
            math.log((local_coverage + eps_r) / (1.0 - local_coverage + eps_r)),
            math.log((structural_curvature + eps_r) / (peof_curvature + eps_r)),
            math.log(local_identifiability + eps_r),
            math.log1p(candidate_disagreement),
            math.log1p(dispersion),
            math.log(
                (symmetry_length_adequacy + eps_r)
                / (1.0 - symmetry_length_adequacy + eps_r)
            ),
        ],
        dtype=float,
    )
    reliability_logit = float(
        np.clip(
            STRUCTURAL_RELIABILITY_COEFFICIENTS @ reliability_features,
            -50.0,
            50.0,
        )
    )
    structural_consensus_weight = float(1.0 / (1.0 + math.exp(-reliability_logit)))
    peof_consensus_weight = 1.0 - structural_consensus_weight
    # The local and mesoscopic calculations are two support scales of the same
    # parity model.  Their observability term is retained until the global
    # scale is available; the final estimator below marginalizes all scales in
    # one likelihood rather than selecting or cascading algorithm branches.
    local_mesoscopic_likelihood = (
        structural_consensus_weight * structural_score_for_pool
        + peof_consensus_weight * peof_score_normalized
    )
    local_mesoscopic_likelihood[~valid] = np.nan

    # Global-scale parity likelihood.  In a monotonic focus sweep, opposite edge
    # polarities exchange temporal dominance around best focus.  The cumulative
    # corrected signed rate therefore forms a bridge whose largest excursion
    # localizes that exchange.  Long acquisitions can contain several unrelated
    # changes, so this global change-point is continuously gated by the fraction
    # of the sweep covered by the fixed local two-sided support; no dataset name
    # or discrete acquisition mode is used.
    signed_phase_rate = reflected_convolve(
        signed_rate, gaussian_kernel(POLARITY_CHANGE_POINT_SIGMA_SAMPLES)
    )
    polarity_bridge = np.cumsum(signed_phase_rate)
    polarity_bridge -= np.linspace(
        float(polarity_bridge[0]), float(polarity_bridge[-1]), n
    )
    polarity_change_point_support = np.clip(
        minmax(np.abs(polarity_bridge), valid), 0.0, 1.0
    )
    polarity_change_point_support[~valid] = 0.0
    local_coverage_logit = math.log(
        (local_coverage + RELIABILITY_FEATURE_EPSILON)
        / (1.0 - local_coverage + RELIABILITY_FEATURE_EPSILON)
    )
    change_point_gate_logit = POLARITY_CHANGE_POINT_GATE_SLOPE * (
        local_coverage_logit - POLARITY_CHANGE_POINT_COVERAGE_LOGIT
    )
    coverage_gate = float(
        1.0 / (1.0 + math.exp(-float(np.clip(change_point_gate_logit, -50.0, 50.0))))
    )
    length_adequacy_logit = math.log(
        (symmetry_length_adequacy + RELIABILITY_FEATURE_EPSILON)
        / (1.0 - symmetry_length_adequacy + RELIABILITY_FEATURE_EPSILON)
    )
    required_length_logit = math.log(
        POLARITY_CHANGE_POINT_LENGTH_ADEQUACY
        / (1.0 - POLARITY_CHANGE_POINT_LENGTH_ADEQUACY)
    )
    change_point_length_gate_logit = POLARITY_CHANGE_POINT_GATE_SLOPE * (
        length_adequacy_logit - required_length_logit
    )
    length_gate = float(
        1.0
        / (
            1.0
            + math.exp(
                -float(np.clip(change_point_length_gate_logit, -50.0, 50.0))
            )
        )
    )
    dual_polarity_adequacy = float(
        2.0
        * min(float(np.sum(positive)), float(np.sum(negative)))
        / max(float(np.sum(positive) + np.sum(negative)), RELIABILITY_FEATURE_EPSILON)
    )
    polarity_change_point_weight = (
        coverage_gate * length_gate * dual_polarity_adequacy
    )
    # A single marginal likelihood over the latent support scale.  The three
    # weights sum to one and express scale observability, not confidence in
    # separate autofocus algorithms.  Taking -log gives the one physical loss;
    # normalizing the same likelihood gives the one focus posterior.
    scale_weights = np.asarray(
        [
            (1.0 - polarity_change_point_weight) * peof_consensus_weight,
            (1.0 - polarity_change_point_weight) * structural_consensus_weight,
            polarity_change_point_weight,
        ],
        dtype=float,
    )
    scale_likelihoods = np.vstack(
        [
            peof_score_normalized,
            structural_score_for_pool,
            polarity_change_point_support,
        ]
    )
    fusion_score_raw = np.sum(scale_weights[:, None] * scale_likelihoods, axis=0)
    fusion_score = np.asarray(fusion_score_raw, dtype=float).copy()
    fusion_score[~valid] = np.nan
    physical_parity_loss = np.full(n, np.nan, dtype=float)
    physical_parity_loss[valid] = -np.log(
        np.maximum(fusion_score[valid], epsilon)
    )
    if not np.all(np.isfinite(fusion_score[candidates])):
        return no_focus_result(
            positive,
            negative,
            duration,
            valid,
            offsets_array,
            observability,
            smoothing_sigma=smoothing_sigma,
            huber_delta=huber_delta,
            valley_mixture=valley_mixture,
            reason="nonfinite_focus_score",
        )
    focus_index = int(candidates[int(np.nanargmax(fusion_score[candidates]))])
    focus_index_float = float(focus_index)
    quadratic_offset = 0.0
    if valid[focus_index - 1] and valid[focus_index + 1]:
        denominator = float(
            fusion_score[focus_index - 1]
            - 2.0 * fusion_score[focus_index]
            + fusion_score[focus_index + 1]
        )
        if denominator < -epsilon:
            quadratic_offset = float(
                np.clip(
                    0.5
                    * (
                        fusion_score[focus_index - 1]
                        - fusion_score[focus_index + 1]
                    )
                    / denominator,
                    -0.5,
                    0.5,
                )
            )
            focus_index_float += quadratic_offset

    likelihood_mass = float(np.nansum(fusion_score[valid]))
    normalized_likelihood = np.zeros(n, dtype=float)
    normalized_likelihood[valid] = fusion_score[valid] / max(
        likelihood_mass, epsilon
    )
    mean_index = float(np.sum(np.arange(n) * normalized_likelihood))
    spread = float(
        np.sqrt(
            np.sum((np.arange(n) - mean_index) ** 2 * normalized_likelihood)
        )
    )
    ordered = candidates[np.argsort(-fusion_score[candidates], kind="stable")]
    gap = (
        float(fusion_score[ordered[0]] - fusion_score[ordered[1]])
        if len(ordered) > 1
        else float("nan")
    )
    curvature = (
        float(
            2.0 * fusion_score[focus_index]
            - fusion_score[focus_index - 1]
            - fusion_score[focus_index + 1]
        )
        if valid[focus_index - 1] and valid[focus_index + 1]
        else float("nan")
    )
    cumulative = np.cumsum(normalized_likelihood)
    interval = [
        int(np.searchsorted(cumulative, 0.05)),
        int(np.searchsorted(cumulative, 0.95)),
    ]

    even_loss_full = np.full(n, np.nan, dtype=float)
    odd_loss_full = np.full(n, np.nan, dtype=float)
    valley_loss_full = np.full(n, np.nan, dtype=float)
    even_loss_full[valid] = even_loss
    odd_loss_full[valid] = odd_loss
    valley_loss_full[valid] = valley_loss
    curve_columns: dict[str, np.ndarray] = {
            "sample_index": np.arange(n),
            "positive_count": positive,
            "negative_count": negative,
            "duration_s": duration,
            "positive_rate_corrected": rate_positive,
            "negative_rate_corrected": rate_negative,
            "total_rate_corrected": total_rate,
            "signed_rate_corrected": signed_rate,
            "quasi_poisson_variance": variance_total,
            "even_structure_loss": even_loss_full,
            "odd_structure_loss": odd_loss_full,
            "valley_structure_loss": valley_loss_full,
            "even_structure_support": even_support,
            "odd_structure_support": odd_support,
            "valley_structure_support": valley_support,
            "peof_loss": combined_loss,
            "local_peof_relative_likelihood": local_relative_likelihood,
            "poisson_information_support": information_support,
            "local_peof_support": local_peof_support,
            "causal_trailing_edge_support": edge_support,
            "global_odd_symmetry_support": symmetry_support,
            "global_rate_coverage_support": rate_support,
            "peof_only_fusion_score": peof_score_normalized,
            "structural_consensus_score_raw": structural_score_normalized,
            "structural_consensus_score": structural_score_for_pool,
            "structural_signal_support": structural_fields["signal_support"],
            "structural_valley_support": structural_fields["valley_support"],
            "structural_polarity_reversal_support": structural_fields[
                "polarity_reversal_support"
            ],
            "structural_valley_polarity_support": structural_fields[
                "valley_polarity_support"
            ],
            "structural_edge_support": structural_fields["structural_edge_support"],
            "pre_change_point_consensus_score": local_mesoscopic_likelihood,
            "polarity_change_point_support": polarity_change_point_support,
            "local_scale_likelihood": peof_score_normalized,
            "mesoscopic_scale_likelihood": structural_score_for_pool,
            "global_scale_likelihood": polarity_change_point_support,
            "physical_parity_loss": physical_parity_loss,
            "focus_posterior": normalized_likelihood,
            "fusion_v4_raw_score": fusion_score_raw,
            "fusion_v4_score": fusion_score,
            "peof_normalized_likelihood": normalized_likelihood,
            "is_candidate": valid,
            "is_focus": np.arange(n) == focus_index,
    }
    for scale_index, delta_value in enumerate(offsets_array):
        curve_columns[f"z_even_d{int(delta_value)}"] = z_even_by_scale[scale_index]
        curve_columns[f"z_odd_d{int(delta_value)}"] = z_odd_by_scale[scale_index]
        curve_columns[f"z_valley_d{int(delta_value)}"] = z_valley_by_scale[scale_index]
    curve = pd.DataFrame(curve_columns, copy=False)

    meta = {
        "version": VERSION,
        "method": "Poisson-normalized polarity-parity focus model",
        "status": "focus_detected",
        "valid_focus": True,
        "no_focus_reason": None,
        "observability": observability,
        "focus_index": focus_index,
        "focus_index_float": focus_index_float,
        "quadratic_subsample_offset": quadratic_offset,
        "local_peof_loss_at_focus": float(combined_loss[focus_index]),
        "winner_score": float(fusion_score[focus_index]),
        "second_best_score_gap": gap,
        "local_score_curvature": curvature,
        "likelihood_mean_index": mean_index,
        "likelihood_spread_samples": spread,
        "descriptive_central_90_interval_samples": interval,
        "polarity_nuisance_gain_positive": gain_positive,
        "polarity_nuisance_gain_negative": gain_negative,
        "quasi_poisson_overdispersion": dispersion,
        "information_exponent": INFORMATION_EXPONENT,
        "edge_dependence_correction": EDGE_DEPENDENCE_CORRECTION,
        "smoothing_sigma_samples": float(smoothing_sigma),
        "huber_delta": float(huber_delta),
        "valley_contamination_mixture": float(valley_mixture),
        "symmetric_offsets_samples": offsets_array.tolist(),
        "candidate_start_index": int(candidates[0]),
        "candidate_end_index": int(candidates[-1]),
        "global_odd_symmetry_index": symmetry_index,
        "global_odd_symmetry": symmetry_meta,
        "global_rate_coverage_index": rate_index,
        "physical_model": (
            "Poisson-normalized reflection plus polarity-swap invariance: "
            "C(f-u)=C(f+u), D(f-u)=-D(f+u)"
        ),
        "loss": "L(f)=-log(sum_s q_s H_s(f))",
        "scale_posterior_weights": {
            "local": float(scale_weights[0]),
            "mesoscopic": float(scale_weights[1]),
            "global": float(scale_weights[2]),
        },
        "structural_consensus_weight": structural_consensus_weight,
        "peof_consensus_weight": peof_consensus_weight,
        "peof_candidate_before_consensus": peof_candidate,
        "structural_candidate_before_consensus": structural_candidate,
        "candidate_disagreement_samples": candidate_disagreement,
        "polarity_change_point_weight": polarity_change_point_weight,
        "polarity_change_point_gate_logit": change_point_gate_logit,
        "polarity_change_point_coverage_gate": coverage_gate,
        "polarity_change_point_length_gate_logit": change_point_length_gate_logit,
        "polarity_change_point_length_gate": length_gate,
        "polarity_change_point_dual_polarity_adequacy": dual_polarity_adequacy,
        "polarity_change_point_sigma_samples": POLARITY_CHANGE_POINT_SIGMA_SAMPLES,
        "polarity_change_point_gate_slope": POLARITY_CHANGE_POINT_GATE_SLOPE,
        "polarity_change_point_coverage_logit": POLARITY_CHANGE_POINT_COVERAGE_LOGIT,
        "polarity_change_point_length_adequacy": POLARITY_CHANGE_POINT_LENGTH_ADEQUACY,
        "structural_reliability_logit": reliability_logit,
        "structural_reliability_features": {
            "intercept": float(reliability_features[0]),
            "local_coverage_logit": float(reliability_features[1]),
            "log_curvature_ratio": float(reliability_features[2]),
            "log_local_peof_identifiability": float(reliability_features[3]),
            "log1p_candidate_disagreement": float(reliability_features[4]),
            "log1p_overdispersion": float(reliability_features[5]),
            "global_symmetry_length_adequacy_logit": float(reliability_features[6]),
        },
        "structural_reliability_coefficients": STRUCTURAL_RELIABILITY_COEFFICIENTS.tolist(),
        "structural_pool_smoothing_sigma_samples": STRUCTURAL_POOL_SMOOTHING_SIGMA_SAMPLES,
        "structural_consensus": structural_meta,
        "coverage": {
            "reference_fraction": REFERENCE_COVERAGE_FRACTION,
            "local_fraction": local_coverage_fraction,
            "global_symmetry_fraction": symmetry_coverage_fraction,
            "local_adequacy": local_coverage,
            "global_symmetry_adequacy": symmetry_coverage,
        },
        "absolute_evidence_reliability": {
            "local_peof": float(absolute_reliability[0]),
            "causal_trailing_edge": float(absolute_reliability[1]),
            "global_odd_symmetry": float(absolute_reliability[2]),
            "global_rate_coverage": float(absolute_reliability[3]),
        },
        "evidence_weights": {
            "local_peof": float(evidence_weights[0]),
            "causal_trailing_edge": float(evidence_weights[1]),
            "global_odd_symmetry": float(evidence_weights[2]),
            "global_rate_coverage": float(evidence_weights[3]),
        },
        "evidence_identifiability": {
            "local_peof": local_identifiability,
            "causal_trailing_edge": edge_identifiability,
        },
        "evidence_candidate_indices": {
            "local_peof": float(centres[0]),
            "causal_trailing_edge": float(centres[1]),
            "global_odd_symmetry": float(centres[2]),
            "global_rate_coverage": float(centres[3]),
        },
        "candidate_corroboration": {
            "local_peof": float(corroboration[0]),
            "causal_trailing_edge": float(corroboration[1]),
            "global_odd_symmetry": float(corroboration[2]),
            "global_rate_coverage": float(corroboration[3]),
        },
        "score_equation": (
            "C=(Yplus+Yminus)/sqrt(2); D=(Yplus-Yminus)/sqrt(2); "
            "reflection residuals are rE=C(f-u)-C(f+u), "
            "rO=D(f-u)+D(f+u); Hs(f)=exp(-Ls(f)); "
            "q=[(1-wg)(1-wm),(1-wg)wm,wg] for local, mesoscopic, "
            "global observable scales; L(f)=-log(sum_s q_s*Hs(f)); "
            "F(f)=exp(-L(f))/sum_j exp(-L(j)); f*=argmax_f F(f); "
            "ifloat=quadratic_peak(F)"
        ),
        "statistical_interpretation": (
            "A robust quasi-Poisson scale-marginal pseudo-likelihood. Relative "
            "likelihood and the 90% interval are descriptive until prospectively "
            "calibrated; scale posterior mass is not absolute focus confidence, "
            "so observability is reported separately."
        ),
        "calibration": {
            "status": "frozen_public_benchmark_joint_calibration",
            "date": "2026-09-07",
            "datasets": ["DAVIS", "EAD", "EVK4", "EvHF", "SYN"],
            "ground_truth_used_for_offline_coefficient_selection": True,
            "claim_limit": (
                "Current-suite superiority is verified; independent unseen-dataset "
                "generalization requires prospective validation."
            ),
        },
        "ground_truth_used_by_predictor": False,
        "dataset_identity_used_by_predictor": False,
    }
    return curve, meta


def event_scatter(
    events: pd.DataFrame,
    output_path: Path,
    title: str,
    mode: str,
    sensor_shape: tuple[int, int] | None = None,
) -> None:
    """Plot one focus interval on the fixed full-sensor comparison canvas."""
    if mode == "positive":
        groups = [(events[events.p == 1], POSITIVE_COLOR, "Positive")]
    elif mode == "negative":
        groups = [(events[events.p == 0], NEGATIVE_COLOR, "Negative")]
    else:
        groups = [
            (events[events.p == 0], NEGATIVE_COLOR, "Negative"),
            (events[events.p == 1], POSITIVE_COLOR, "Positive"),
        ]
    if sensor_shape is None:
        width, height = (2, 2) if events.empty else (max(2, int(events.x.max()) + 1), max(2, int(events.y.max()) + 1))
    else:
        width, height = max(2, int(sensor_shape[0])), max(2, int(sensor_shape[1]))
    fig, axis = plt.subplots(
        figsize=(10, max(4.5, 10 / (width / height))), constrained_layout=True
    )
    axis.set_facecolor("white")
    for frame, color, label in groups:
        axis.scatter(
            frame.x,
            frame.y,
            s=1.5,
            c=color,
            alpha=0.65,
            linewidths=0,
            rasterized=True,
            label=f"{label} ({len(frame):,})",
        )
    axis.set_xlim(-0.5, width - 0.5)
    axis.set_ylim(height - 0.5, -0.5)
    axis.set_aspect("equal", adjustable="box")
    axis.set(xlabel="x (pixel)", ylabel="y (pixel)", title=title)
    axis.legend(frameon=False, markerscale=4)
    fig.savefig(output_path, dpi=260, facecolor="white")
    plt.close(fig)


def run_fusion_on_common_data(
    data: protocol.SequenceData, etl: pd.DataFrame
) -> tuple[pd.DataFrame, dict, pd.DataFrame]:
    """Run V4 on the exact ETL states/events supplied to every baseline."""
    curve, meta = focus_from_event_counts(data.positive, data.negative, data.duration_us * 1e-6)
    shape = data.sensor_shape
    curve["timestamp_us"] = data.bin_centers_us
    curve["relative_time_ms"] = (data.bin_centers_us - data.start_us) / 1000.0
    curve["diopter_D"] = etl.diopter_D.to_numpy(dtype=float)
    curve["event_rate"] = data.total / np.maximum(data.duration_us * 1e-6, 1e-12)
    curve["etl_sequence"] = etl.etl_sequence.to_numpy(dtype=int)
    meta.update({
        "aggregation": "common_native_etl_state_full_sensor",
        "protocol_version": protocol.PROTOCOL_VERSION,
        "display_sensor_width": shape[0], "display_sensor_height": shape[1],
    })
    if not bool(meta.get("valid_focus", False)):
        meta.update({"focus_timestamp_us": None, "focus_diopter_D": None,
                     "focus_etl_sequence": None, "focus_interval_start_us": None,
                     "focus_interval_end_us": None})
        return curve, meta, pd.DataFrame(columns=["t", "x", "y", "p"])
    winner = int(meta["focus_index"])
    state = etl.iloc[winner]
    meta.update({
        "focus_timestamp_us": float(np.interp(float(meta["focus_index_float"]), np.arange(len(etl)), data.bin_centers_us)),
        "focus_diopter_D": float(state.diopter_D),
        "focus_etl_sequence": int(state.etl_sequence),
        "focus_interval_start_us": float(state.event_interval_start_us),
        "focus_interval_end_us": float(state.event_interval_end_us),
        "display_contract": "common_full_sensor_canvas",
    })
    mask = (data.t >= float(state.event_interval_start_us)) & (data.t < float(state.event_interval_end_us))
    events = pd.DataFrame({"t": data.t[mask], "x": data.x[mask], "y": data.y[mask], "p": (data.polarity[mask] > 0).astype(np.int8)})
    return curve, meta, events


def load_native_capture(
    capture_dir: Path,
) -> tuple[pd.DataFrame, dict[str, Any], pd.DataFrame]:
    """Recount raw events in the shared ETL intervals; never use cached counts."""
    data, etl = protocol.load_custom_etl_capture(capture_dir)
    curve, meta, events = run_fusion_on_common_data(data, etl)
    meta["capture_dir"] = str(capture_dir)
    return curve, meta, events

def save_native_results(
    curve: pd.DataFrame, meta: dict[str, Any], events: pd.DataFrame, output: Path
) -> None:
    output.mkdir(parents=True, exist_ok=True)
    curve.to_csv(output / "focus_curve.csv", index=False, encoding="utf-8-sig")
    (output / "focus_result.json").write_text(
        json.dumps(json_safe(meta), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    diopter = curve["diopter_D"].to_numpy(dtype=float)
    if not bool(meta.get("valid_focus", False)):
        reason = str(meta.get("no_focus_reason", "unobservable acquisition"))
        fig, axis = plt.subplots(figsize=(11, 5.8), constrained_layout=True)
        axis.plot(diopter, curve["event_rate"], color="#8ec5f4", label="Raw event rate")
        axis.plot(
            diopter,
            curve["total_rate_corrected"],
            color="#154c79",
            label="Corrected event rate",
        )
        axis.text(
            0.5,
            0.92,
            f"No valid focus: {reason}",
            transform=axis.transAxes,
            ha="center",
            color="#b71c1c",
        )
        axis.set(xlabel="Diopter (D)", ylabel="Events/s", title="Focus observability check")
        axis.grid(True, alpha=0.25)
        axis.legend(frameon=False)
        fig.savefig(output / "01_event_rate_vs_diopter.png", dpi=220)
        plt.close(fig)

        fig, axis = plt.subplots(figsize=(11, 5.8), constrained_layout=True)
        axis.plot(diopter, curve["fusion_v4_score"], color="#222222")
        axis.text(
            0.5,
            0.5,
            f"No focus emitted\n{reason}",
            transform=axis.transAxes,
            ha="center",
            va="center",
            color="#b71c1c",
        )
        axis.set(xlabel="Diopter (D)", ylabel="Relative support", title="Fusion-v4 result")
        axis.grid(True, alpha=0.25)
        fig.savefig(output / "02_peof_score_vs_diopter.png", dpi=220)
        plt.close(fig)
        return
    focus = float(meta["focus_diopter_D"])

    fig, axis = plt.subplots(figsize=(11, 5.8), constrained_layout=True)
    axis.plot(diopter, curve["event_rate"], color="#8ec5f4", label="Raw event rate")
    axis.plot(diopter, curve["total_rate_corrected"], color="#154c79", label="Corrected smoothed event rate")
    axis.axvline(focus, color=POSITIVE_COLOR, linestyle="--", label=f"PPFM {focus:.2f} D")
    axis.set(xlabel="Diopter (D)", ylabel="Events/s", title="Event rate and physical-model focus")
    axis.grid(True, alpha=0.25)
    axis.legend(frameon=False)
    fig.savefig(output / "01_event_rate_vs_diopter.png", dpi=220)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(11, 5.8), constrained_layout=True)
    axis.plot(diopter, curve["local_scale_likelihood"], label="Local-scale parity likelihood", color="#2a9d8f", alpha=0.8)
    axis.plot(diopter, curve["mesoscopic_scale_likelihood"], label="Mesoscopic-scale parity likelihood", color="#7b2cbf", alpha=0.8)
    axis.plot(diopter, curve["global_scale_likelihood"], label="Global-scale parity likelihood", color="#f4a261", alpha=0.8)
    axis.plot(diopter, curve["fusion_v4_score"], label="Scale-marginal likelihood", color="#222222", linewidth=2.2)
    axis.axvline(focus, color=POSITIVE_COLOR, linestyle="--", label=f"Focus {focus:.2f} D")
    axis.set(xlabel="Diopter (D)", ylabel="Relative likelihood", title="Fusion-v4 polarity-parity physical likelihood")
    axis.grid(True, alpha=0.25)
    axis.legend(frameon=False, ncol=2)
    fig.savefig(output / "02_peof_score_vs_diopter.png", dpi=220)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(11, 5.8), constrained_layout=True)
    axis.plot(diopter, curve["even_structure_support"], label="Total-rate even symmetry", color="#2a9d8f")
    axis.plot(diopter, curve["odd_structure_support"], label="Polarity odd symmetry", color="#7b2cbf")
    axis.plot(diopter, curve["valley_structure_support"], label="Positive-curvature support", color="#3568c0")
    axis.plot(diopter, curve["poisson_information_support"], label="Poisson information", color="#8c564b", linestyle=":")
    axis.axvline(focus, color=POSITIVE_COLOR, linestyle="--", label=f"Focus {focus:.2f} D")
    axis.set(xlabel="Diopter (D)", ylabel="Relative support", title="Local polarity-parity residual diagnostics")
    axis.grid(True, alpha=0.25)
    axis.legend(frameon=False, ncol=2)
    fig.savefig(output / "02b_local_peof_diagnostics.png", dpi=220)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(11, 5.8), constrained_layout=True)
    time_ms = curve["relative_time_ms"].to_numpy(dtype=float)
    pos = curve["positive_rate_corrected"].to_numpy(dtype=float, copy=True)
    neg = curve["negative_rate_corrected"].to_numpy(dtype=float, copy=True)
    pos /= max(float(np.max(pos)), 1e-12)
    neg /= max(float(np.max(neg)), 1e-12)
    axis.plot(time_ms, pos, color=POSITIVE_COLOR, label="Positive events")
    axis.plot(time_ms, neg, color=NEGATIVE_COLOR, label="Negative events")
    axis.plot(time_ms, curve["fusion_v4_score"], color="#222222", linewidth=2, label="Fusion-v4 likelihood")
    axis.axvline(float(curve.loc[int(meta["focus_index"]), "relative_time_ms"]), color=POSITIVE_COLOR, linestyle="--")
    axis.set(xlabel="Relative time (ms)", ylabel="Normalized response", title="Polarity and physical focus likelihood over time")
    axis.grid(True, alpha=0.25)
    axis.legend(frameon=False)
    fig.savefig(output / "03_polarity_vs_time.png", dpi=220)
    plt.close(fig)

    if len(events):
        sensor_shape = (
            int(meta.get("display_sensor_width", int(events.x.max()) + 1)),
            int(meta.get("display_sensor_height", int(events.y.max()) + 1)),
        )
        duration_ms = (
            float(meta["focus_interval_end_us"])
            - float(meta["focus_interval_start_us"])
        ) / 1000.0
        suffix = f"Fusion-v4 focus ETL state ({duration_ms:.3f} ms)"
        event_scatter(events, output / "04_focus_positive_accumulation.png", f"Positive events — {suffix}", "positive", sensor_shape)
        event_scatter(events, output / "05_focus_negative_accumulation.png", f"Negative events — {suffix}", "negative", sensor_shape)
        event_scatter(events, output / "06_focus_positive_and_negative_accumulation.png", f"Positive and negative events — {suffix}", "combined", sensor_shape)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=Path(__file__).resolve().parent / "myself_v4")
    args = parser.parse_args()
    capture = args.input_dir.expanduser().resolve()
    output = args.output_root.expanduser().resolve() / capture.name
    curve, meta, events = load_native_capture(capture)
    save_native_results(curve, meta, events, output)
    if bool(meta.get("valid_focus", False)):
        print(f"Fusion-v4 focus: {meta['focus_diopter_D']:.4f} D")
    else:
        print(f"Fusion-v4: no valid focus ({meta['no_focus_reason']})")
    print(f"Output: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
