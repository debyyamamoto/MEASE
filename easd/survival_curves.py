"""Shared event-time grid and weighted distances for survival curves.

The grid follows the existing RuleEvaluator convention, including its optional
quantile binning. See docs/mean_pairwise_rms.md for the metric definition.
"""
from __future__ import annotations

import numpy as np

EPSILON = 1e-12
Curve = tuple[np.ndarray, np.ndarray]  # post-event survival, pre-event risk count


def kaplan_meier_from_counts(events: np.ndarray, risk: np.ndarray) -> np.ndarray:
    hazards = np.divide(events, risk, out=np.zeros_like(events), where=risk > 0.0)
    return np.cumprod(1.0 - np.clip(hazards, 0.0, 1.0))


class KaplanMeierGrid:
    """Dataset-wide grid; build each subgroup curve once on the same grid."""

    def __init__(self, times: np.ndarray, events: np.ndarray, time_bins: int | None = 512):
        times = np.asarray(times, dtype=float)
        observed = np.asarray(events, dtype=bool)
        self.n = len(times)
        self.grid_times = np.array([], dtype=float)
        self.widths = np.array([], dtype=float)
        self.weights = np.array([], dtype=float)
        self.risk_bin = np.full(self.n, -1, dtype=int)
        self.event_bin = np.full(self.n, -1, dtype=int)
        self.pooled_subject_counts = np.array([], dtype=float)
        self.pooled_event_counts = np.array([], dtype=float)
        self.pooled_risk_counts = np.array([], dtype=float)
        self.pooled_survival = np.array([], dtype=float)
        event_times = np.sort(np.unique(times[observed]))
        if not event_times.size:
            return
        exact = time_bins is None or time_bins <= 0 or event_times.size <= time_bins
        self.grid_times = (event_times if exact else np.unique(
            np.quantile(event_times, np.linspace(0.0, 1.0, int(time_bins)))))
        size = self.grid_times.size
        max_time = max(float(np.max(times)), float(self.grid_times[-1]))
        self.widths = np.maximum(np.diff(np.r_[self.grid_times, max_time]), 0.0)
        if exact:
            self.risk_bin = np.searchsorted(self.grid_times, times, side="right") - 1
            self.event_bin[observed] = np.searchsorted(self.grid_times, times[observed], side="left")
        else:
            self.risk_bin = np.clip(np.searchsorted(self.grid_times, times, side="left"), 0, size - 1)
            self.risk_bin[times < self.grid_times[0]] = -1
            self.event_bin[observed] = self.risk_bin[observed]
        self.pooled_subject_counts, self.pooled_event_counts = self.group_counts(np.ones(self.n, dtype=bool))
        self.pooled_risk_counts = np.cumsum(self.pooled_subject_counts[::-1])[::-1]
        self.pooled_survival = kaplan_meier_from_counts(self.pooled_event_counts, self.pooled_risk_counts)
        self.weights = np.divide(self.pooled_risk_counts, float(self.n),
                                 out=np.zeros_like(self.pooled_risk_counts), where=self.pooled_risk_counts > 0)

    def group_counts(self, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        risk_bins, event_bins = self.risk_bin[mask], self.event_bin[mask]
        size = self.grid_times.size
        subjects = np.bincount(risk_bins[risk_bins >= 0], minlength=size).astype(float)
        events = np.bincount(event_bins[event_bins >= 0], minlength=size).astype(float)
        return subjects, events

    def curve(self, mask: np.ndarray) -> Curve:
        subjects, events = self.group_counts(mask)
        risk = np.cumsum(subjects[::-1])[::-1]
        return kaplan_meier_from_counts(events, risk), risk


def weighted_rms_distance(first: Curve, second: Curve, widths: np.ndarray, weights: np.ndarray) -> float:
    """Weighted RMS on jointly supported grid intervals; NaN if unmeasurable.

    Both curves describe complete subgroups, including any shared individuals.
    No statistical significance test or subgroup-versus-baseline comparison.
    """
    s1, n1 = first
    s2, n2 = second
    valid = (n1 > 0) & (n2 > 0) & (widths > 0) & (weights > 0)
    weighted_widths = widths[valid] * weights[valid]
    normalizer = float(weighted_widths.sum())
    if normalizer <= EPSILON:
        return float("nan")
    difference = s1[valid] - s2[valid]
    squared = float(np.sum(weighted_widths * difference * difference) / normalizer)
    return float(np.sqrt(np.clip(squared, 0.0, 1.0)))
