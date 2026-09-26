from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd
import statsmodels.api as sm

from .dataset import Dataset

SCORE_METRICS: tuple[str, ...] = (
    "legacy_logrank",
    "fast_logrank",
    "km_cvm",
    "km_abc",
    "mdir2",
    "mdir3",
    "mdir4",
)
KM_GRID_SCORE_METRICS = ("km_cvm", "km_abc", "mdir2", "mdir3", "mdir4")
MDIR_SCORE_METRICS = ("mdir2", "mdir3", "mdir4")

MIN_RELATIVE_SUPPORT = 0.05
MAX_RELATIVE_SUPPORT = 0.55
EPSILON = 1e-12

MDIR_WEIGHT_SETS: dict[str, tuple[tuple[int, int] | str, ...]] = {
    "mdir2": ((0, 0), "cross"),
    "mdir3": ((0, 0), (0, 4), "cross"),
    "mdir4": ((0, 0), (0, 4), (4, 0), "cross"),
}


class RuleEvaluator:
    def __init__(
        self,
        dataset_obj: Dataset,
        comparacao,
        alpha,
        score_metric: str = "legacy_logrank",
        km_time_bins: int | None = 512,
        redundancy_penalty: float = 0.0,
        redundancy_similarity_threshold: float = 0.9,
    ):
        if comparacao not in ("complement", "population"):
            raise ValueError("comparacao must be either 'complement' or 'population'.")
        if score_metric not in SCORE_METRICS:
            raise ValueError(f"score_metric must be one of {SCORE_METRICS}.")
        if not 0.0 <= redundancy_penalty <= 1.0:
            raise ValueError("redundancy_penalty must be between 0 and 1.")
        if not 0.0 <= redundancy_similarity_threshold < 1.0:
            raise ValueError("redundancy_similarity_threshold must be in [0, 1).")

        self.dataset_obj = dataset_obj
        self.comparacao = comparacao
        self.score_metric = score_metric
        self.km_time_bins = None if km_time_bins is None or km_time_bins <= 0 else int(km_time_bins)
        self.redundancy_penalty = float(redundancy_penalty)
        self.redundancy_similarity_threshold = float(redundancy_similarity_threshold)
        self.sub_group_cases = dataset_obj.get_instances()
        self.p_value = None
        self._fitness = 0.0
        self._n = self.dataset_obj.size
        self._all_indices = set(range(self._n))
        self.alpha = alpha
        self._survival_times = self.dataset_obj._original_data[self.dataset_obj.surv_name]
        self._events = self.dataset_obj._original_data[self.dataset_obj._event_name]
        self._time_values = self._survival_times.to_numpy(dtype=float, copy=False)
        self._event_values = self._events.to_numpy(dtype=int, copy=False)
        self._event_observed = self._event_values.astype(bool)

        self._logrank_gamma = np.zeros(self._n, dtype=float)
        self._logrank_residual = self._event_values.astype(float)
        self._logrank_expected_total = 0.0
        if score_metric == "fast_logrank":
            self._precompute_fast_logrank()

        self._km_grid_times = np.array([], dtype=float)
        self._km_widths = np.array([], dtype=float)
        self._km_weights = np.array([], dtype=float)
        self._km_risk_bin = np.full(self._n, -1, dtype=int)
        self._km_event_bin = np.full(self._n, -1, dtype=int)
        self._km_pooled_subject_counts = np.array([], dtype=float)
        self._km_pooled_event_counts = np.array([], dtype=float)
        self._km_pooled_risk_counts = np.array([], dtype=float)
        self._km_pooled_survival = np.array([], dtype=float)
        self._mdir_weights = np.empty((0, 0), dtype=float)
        if score_metric in KM_GRID_SCORE_METRICS or self.redundancy_penalty > 0.0:
            self._precompute_km_grid()
        if score_metric in MDIR_SCORE_METRICS:
            self._precompute_mdir_weights()

    def get_covered_mask(self, rule, dataset: np.ndarray):
        if not rule or len(rule) < 2 or len(rule[0]) != len(rule[1]):
            return np.zeros(dataset.shape[0], dtype=bool)

        indices, values = rule[0], rule[1]
        n_rows = dataset.shape[0]
        mask = np.ones(n_rows, dtype=bool)

        for idx, val in zip(indices, values):
            if not val:
                return np.zeros(n_rows, dtype=bool)

            col_data = dataset[:, idx]
            if isinstance(val[0], (str, np.str_)):
                if len(val) > 1:
                    row_mask = np.isin(col_data, val)
                else:
                    row_mask = col_data == val[0]
            else:
                row_mask = (col_data >= val[0]) & (col_data <= val[1])
            mask &= row_mask

        return mask

    def get_covered_indices(self, rule, dataset: np.ndarray):
        return np.flatnonzero(self.get_covered_mask(rule, dataset)).tolist()

    def fitness(self, rule, dataset_x: np.ndarray):
        """Receives a rule and computes its fitness."""
        rule_mask = self.get_covered_mask(rule, dataset_x)
        covered_count = int(rule_mask.sum())
        if covered_count < 1:
            return 0.0

        relative_support = covered_count / self._n
        if relative_support > MAX_RELATIVE_SUPPORT or relative_support < MIN_RELATIVE_SUPPORT:
            return 0.0

        if covered_count >= self._n:
            return 0.0

        if self.score_metric == "legacy_logrank":
            discrepancy = self._legacy_logrank_score(rule_mask)
        elif self.score_metric == "fast_logrank":
            discrepancy = self._fast_logrank_score(rule_mask)
        elif self.score_metric in ("km_cvm", "km_abc"):
            discrepancy = self._km_distance_score(rule_mask)
        elif self.score_metric in MDIR_SCORE_METRICS:
            discrepancy = self._mdir_score(rule_mask)
        else:
            discrepancy = 0.0

        if not np.isfinite(discrepancy) or discrepancy <= 0.0:
            return 0.0
        return float(discrepancy * (relative_support**self.alpha))

    def get_fitness(self, population, dataset_x: np.ndarray):
        fitness_array = np.asarray([self.fitness(individual, dataset_x) for individual in population], dtype=float)
        sorted_indices = np.argsort(-fitness_array, kind="stable")
        sorted_population = [population[index] for index in sorted_indices]
        return fitness_array[sorted_indices].tolist(), sorted_population

    def penalize_score(
        self,
        p_fitness_list: list,
        p_top_k_rules: list,
        p_population: list,
        p_dataset_x: np.ndarray,
        candidate_count: int | None = None,
    ) -> list[float]:
        """Penalize candidates whose survival curve is too similar to the Top-K."""
        penalized = [float(score) for score in p_fitness_list]
        if self.redundancy_penalty <= 0.0 or not p_top_k_rules or not penalized:
            return penalized

        top_k_curves = [self._km_curve(np.asarray(top_rule[2], dtype=bool)) for top_rule in p_top_k_rules]
        top_k_keys = [tuple(sorted(top_rule[1][0])) for top_rule in p_top_k_rules]
        limit = len(p_population) if candidate_count is None else max(0, candidate_count)
        limit = min(limit, len(p_population), len(penalized))
        threshold = self.redundancy_similarity_threshold

        for idx in range(limit):
            if penalized[idx] <= 0.0:
                continue
            # The archive replaces rules with the same attribute key. Compare
            # against the other entries, which can coexist with this candidate.
            candidate_key = tuple(sorted(p_population[idx][0]))
            candidate_mask = self.get_covered_mask(p_population[idx], p_dataset_x)
            candidate_curve = self._km_curve(candidate_mask)
            max_similarity = max(
                (
                    self._km_curve_similarity_from_estimates(candidate_curve, top_k_curve)
                    for top_key, top_k_curve in zip(top_k_keys, top_k_curves)
                    if top_key != candidate_key
                ),
                default=0.0,
            )
            if max_similarity <= threshold:
                continue

            normalized_similarity = (max_similarity - threshold) / (1.0 - threshold)
            penalty_factor = 1.0 - (self.redundancy_penalty * normalized_similarity)
            penalized[idx] = max(0.0, penalized[idx] * penalty_factor)

        return penalized

    def _legacy_logrank_score(self, rule_mask: np.ndarray) -> float:
        try:
            group_id = np.full(self._n, "pop", dtype=object)
            group_id[rule_mask] = "sg"
            if self.comparacao == "complement":
                group_id[~rule_mask] = "complement"

            p_value_result = sm.duration.survdiff(
                time=self._time_values,
                status=self._event_values,
                group=group_id,
            )
            p_value = float(p_value_result[1])
            if pd.isna(p_value) or p_value < 0.0 or p_value > 1.0:
                return 0.0
            return 1.0 - p_value
        except (ValueError, ZeroDivisionError, Exception):
            return 0.0

    def _precompute_fast_logrank(self) -> None:
        event_times = np.sort(np.unique(self._time_values[self._event_observed]))
        if event_times.size == 0:
            return

        event_positions = np.searchsorted(event_times, self._time_values[self._event_observed], side="left")
        event_counts = np.bincount(event_positions, minlength=event_times.size).astype(float)

        last_risk_positions = np.searchsorted(event_times, self._time_values, side="right") - 1
        valid_risk = last_risk_positions >= 0
        subject_counts = np.bincount(last_risk_positions[valid_risk], minlength=event_times.size).astype(float)
        risk_counts = np.cumsum(subject_counts[::-1])[::-1]

        alpha_t = np.divide(event_counts, risk_counts, out=np.zeros_like(event_counts), where=risk_counts > 0.0)
        cumulative_alpha = np.cumsum(alpha_t)

        self._logrank_gamma = np.zeros(self._n, dtype=float)
        self._logrank_gamma[valid_risk] = cumulative_alpha[last_risk_positions[valid_risk]]
        self._logrank_residual = self._event_values.astype(float) - self._logrank_gamma
        self._logrank_expected_total = float(self._logrank_gamma.sum())

    def _fast_logrank_score(self, rule_mask: np.ndarray) -> float:
        expected_sg = float(self._logrank_gamma[rule_mask].sum())
        if expected_sg <= EPSILON:
            return 0.0

        numerator = float(self._logrank_residual[rule_mask].sum())
        if self.comparacao == "population":
            statistic = (numerator * numerator) / expected_sg
        else:
            expected_ref = self._logrank_expected_total - expected_sg
            if expected_ref <= EPSILON:
                return 0.0
            statistic = (numerator * numerator) * ((1.0 / expected_sg) + (1.0 / expected_ref))

        if statistic <= 0.0:
            return 0.0
        return float(statistic / (1.0 + statistic))

    def _precompute_km_grid(self) -> None:
        event_times = np.sort(np.unique(self._time_values[self._event_observed]))
        if event_times.size == 0:
            return

        exact_grid = self.km_time_bins is None or event_times.size <= self.km_time_bins
        if exact_grid:
            grid_times = event_times
        elif self.km_time_bins is not None:
            quantiles = np.linspace(0.0, 1.0, self.km_time_bins)
            grid_times = np.unique(np.quantile(event_times, quantiles))
        else:
            grid_times = event_times

        if grid_times.size == 0:
            return

        self._km_grid_times = grid_times.astype(float, copy=False)
        grid_size = self._km_grid_times.size
        max_time = max(float(np.max(self._time_values)), float(self._km_grid_times[-1]))
        self._km_widths = np.diff(np.r_[self._km_grid_times, max_time])
        self._km_widths = np.maximum(self._km_widths, 0.0)

        if exact_grid:
            risk_bin = np.searchsorted(self._km_grid_times, self._time_values, side="right") - 1
            event_bin = np.full(self._n, -1, dtype=int)
            event_bin[self._event_observed] = np.searchsorted(
                self._km_grid_times,
                self._time_values[self._event_observed],
                side="left",
            )
        else:
            risk_bin = np.searchsorted(self._km_grid_times, self._time_values, side="left")
            risk_bin = np.clip(risk_bin, 0, grid_size - 1)
            risk_bin[self._time_values < self._km_grid_times[0]] = -1
            event_bin = np.full(self._n, -1, dtype=int)
            event_bin[self._event_observed] = risk_bin[self._event_observed]

        self._km_risk_bin = risk_bin.astype(int, copy=False)
        self._km_event_bin = event_bin.astype(int, copy=False)

        valid_risk = self._km_risk_bin >= 0
        self._km_pooled_subject_counts = np.bincount(self._km_risk_bin[valid_risk], minlength=grid_size).astype(float)
        valid_events = self._km_event_bin >= 0
        self._km_pooled_event_counts = np.bincount(self._km_event_bin[valid_events], minlength=grid_size).astype(float)
        self._km_pooled_risk_counts = np.cumsum(self._km_pooled_subject_counts[::-1])[::-1]
        self._km_pooled_survival = self._kaplan_meier_from_counts(
            self._km_pooled_event_counts,
            self._km_pooled_risk_counts,
        )
        self._km_weights = np.divide(
            self._km_pooled_risk_counts,
            float(self._n),
            out=np.zeros_like(self._km_pooled_risk_counts),
            where=self._km_pooled_risk_counts > 0.0,
        )

    def _km_distance_score(self, rule_mask: np.ndarray) -> float:
        if self._km_grid_times.size == 0:
            return 0.0

        group_subject_counts, group_event_counts = self._group_km_counts(rule_mask)
        group_risk_counts = np.cumsum(group_subject_counts[::-1])[::-1]
        group_survival = self._kaplan_meier_from_counts(group_event_counts, group_risk_counts)

        if self.comparacao == "population":
            ref_risk_counts = self._km_pooled_risk_counts
            ref_survival = self._km_pooled_survival
        else:
            ref_subject_counts = self._km_pooled_subject_counts - group_subject_counts
            ref_event_counts = self._km_pooled_event_counts - group_event_counts
            ref_risk_counts = np.cumsum(ref_subject_counts[::-1])[::-1]
            ref_survival = self._kaplan_meier_from_counts(ref_event_counts, ref_risk_counts)

        valid = (
            (group_risk_counts > 0.0) & (ref_risk_counts > 0.0) & (self._km_widths > 0.0) & (self._km_weights > 0.0)
        )
        if not np.any(valid):
            return 0.0

        weighted_widths = self._km_widths[valid] * self._km_weights[valid]
        normalizer = float(weighted_widths.sum())
        if normalizer <= EPSILON:
            return 0.0

        diff = group_survival[valid] - ref_survival[valid]
        if self.score_metric == "km_abc":
            distance = float(np.sum(weighted_widths * np.abs(diff)) / normalizer)
        else:
            distance = float(np.sum(weighted_widths * diff * diff) / normalizer)

        return min(max(distance, 0.0), 1.0)

    def _km_curve(self, rule_mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        subject_counts, event_counts = self._group_km_counts(rule_mask)
        risk_counts = np.cumsum(subject_counts[::-1])[::-1]
        survival = self._kaplan_meier_from_counts(event_counts, risk_counts)
        return survival, risk_counts

    def _km_curve_similarity_from_estimates(
        self,
        first_curve: tuple[np.ndarray, np.ndarray],
        second_curve: tuple[np.ndarray, np.ndarray],
    ) -> float:
        first_survival, first_risk_counts = first_curve
        second_survival, second_risk_counts = second_curve

        valid = (
            (first_risk_counts > 0.0) & (second_risk_counts > 0.0) & (self._km_widths > 0.0) & (self._km_weights > 0.0)
        )
        if not np.any(valid):
            return 0.0

        weighted_widths = self._km_widths[valid] * self._km_weights[valid]
        normalizer = float(weighted_widths.sum())
        if normalizer <= EPSILON:
            return 0.0

        difference = first_survival[valid] - second_survival[valid]
        mean_squared_distance = float(np.sum(weighted_widths * difference * difference) / normalizer)
        rms_distance = np.sqrt(min(max(mean_squared_distance, 0.0), 1.0))
        return float(1.0 - rms_distance)

    def _group_km_counts(self, rule_mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        grid_size = self._km_grid_times.size
        rule_indices = np.flatnonzero(rule_mask)

        rule_risk_bins = self._km_risk_bin[rule_indices]
        valid_risk = rule_risk_bins >= 0
        subject_counts = np.bincount(rule_risk_bins[valid_risk], minlength=grid_size).astype(float)

        rule_event_bins = self._km_event_bin[rule_indices]
        valid_events = rule_event_bins >= 0
        event_counts = np.bincount(rule_event_bins[valid_events], minlength=grid_size).astype(float)

        return subject_counts, event_counts

    def _precompute_mdir_weights(self) -> None:
        if self._km_grid_times.size == 0:
            return

        survival_before = np.r_[1.0, self._km_pooled_survival[:-1]]
        pooled_failure_before = np.clip(1.0 - survival_before, 0.0, 1.0)

        weights = []
        for direction in MDIR_WEIGHT_SETS[self.score_metric]:
            if direction == "cross":
                weights.append(1.0 - (2.0 * pooled_failure_before))
            else:
                r, g = direction
                weights.append((pooled_failure_before**r) * ((1.0 - pooled_failure_before) ** g))

        self._mdir_weights = np.vstack(weights).astype(float, copy=False)

    def _mdir_score(self, rule_mask: np.ndarray) -> float:
        if self._mdir_weights.size == 0:
            return 0.0

        group_subject_counts, group_event_counts = self._group_km_counts(rule_mask)
        group_risk_counts = np.cumsum(group_subject_counts[::-1])[::-1]

        pooled_risk_counts = self._km_pooled_risk_counts
        pooled_event_counts = self._km_pooled_event_counts

        if self.comparacao == "complement":
            ref_subject_counts = self._km_pooled_subject_counts - group_subject_counts
            ref_risk_counts = np.cumsum(ref_subject_counts[::-1])[::-1]
            has_reference = ref_risk_counts > 0.0
        else:
            has_reference = pooled_risk_counts > group_risk_counts

        valid = (
            (group_risk_counts > 0.0)
            & has_reference
            & (pooled_risk_counts > 1.0)
            & (pooled_event_counts > 0.0)
            & (group_risk_counts < pooled_risk_counts)
        )
        if not np.any(valid):
            return 0.0

        total_risk = pooled_risk_counts[valid]
        total_events = pooled_event_counts[valid]
        group_risk = group_risk_counts[valid]
        group_events = group_event_counts[valid]

        risk_fraction = np.divide(group_risk, total_risk, out=np.zeros_like(group_risk), where=total_risk > 0.0)
        expected_events = risk_fraction * total_events
        observed_minus_expected = group_events - expected_events

        finite_population_correction = np.divide(
            total_risk - total_events,
            total_risk - 1.0,
            out=np.zeros_like(total_risk),
            where=total_risk > 1.0,
        )
        variance_base = total_events * risk_fraction * (1.0 - risk_fraction) * finite_population_correction
        informative = variance_base > EPSILON
        if not np.any(informative):
            return 0.0

        weights = self._mdir_weights[:, valid][:, informative]
        residual = observed_minus_expected[informative]
        variance_base = variance_base[informative]

        linear_statistic = weights @ residual
        covariance = (weights * variance_base) @ weights.T
        if not np.any(np.abs(covariance) > EPSILON):
            return 0.0

        try:
            statistic = float(linear_statistic @ np.linalg.pinv(covariance, hermitian=True) @ linear_statistic)
        except np.linalg.LinAlgError:
            return 0.0

        if statistic <= 0.0 or not np.isfinite(statistic):
            return 0.0
        return float(statistic / (1.0 + statistic))

    def _kaplan_meier_from_counts(self, event_counts: np.ndarray, risk_counts: np.ndarray) -> np.ndarray:
        hazards = np.divide(event_counts, risk_counts, out=np.zeros_like(event_counts), where=risk_counts > 0.0)
        hazards = np.clip(hazards, 0.0, 1.0)
        return np.cumprod(1.0 - hazards)
