import os
import time
import math
import copy
from random import seed
from heapq import heapify, heappush, heappop
from typing import Literal
from rich.console import Console
import numpy as np
import pandas as pd

from .population import PopulationGenerator
from .evaluation import RuleEvaluator
from .operators import GeneticOperators
from .dataset import Dataset
from .visualization import RulesPlotter, plot_topk_convergency
from .performance import ProcessResourceMonitor

EPSILON = 1e-12
DEFAULT_CROSSOVER_RATE = 60
DEFAULT_MUTATION_RATE = 40
RATES_CHANGE_FATOR = 20
RATEPOLICY = Literal["adaptive", "fixed"]
console = Console()


def _normalize_restart_percentage(restart_percentage: float) -> float:
    if restart_percentage < 0:
        raise ValueError("restart_percentage must be non-negative.")
    if restart_percentage > 100:
        raise ValueError("restart_percentage must be between 0 and 100 percent.")
    if restart_percentage > 1:
        return restart_percentage / 100
    return restart_percentage


class MEASE:
    def __init__(
        self,
        data: pd.DataFrame,
        time_col: str,
        event_col: str,
        max_generations: int,
        population_size: int,
        max_generations_no_improve: int,
        max_pop_restarts: int,
        restart_percentage: float,
        seed_val: int,
        comparacao: str,
        alpha: float,
        ksize: int,
        plot_n_rules: int,
        coverage_threshold: float = 0.8,
        debug_performance: bool = False,
        rate_policy: RATEPOLICY = "adaptive",
        score_metric: str = "legacy_logrank",
        km_time_bins: int | None = 512,
        redundancy_penalty: float = 0.0,
        redundancy_similarity_threshold: float = 0.9,
        archive_selection: Literal["historical", "greedy"] = "historical",
    ):
        if rate_policy not in ("adaptive", "fixed"):
            raise ValueError("rate_policy must be either 'adaptive' or 'fixed'.")
        if archive_selection not in ("historical", "greedy"):
            raise ValueError("archive_selection must be either 'historical' or 'greedy'.")

        self.survival_event_col = event_col
        self.survival_time_col = time_col
        self.rate_policy = rate_policy
        self.crossover_rate = DEFAULT_CROSSOVER_RATE
        self.mutation_rate = DEFAULT_MUTATION_RATE
        self.initial_crossover_rate = DEFAULT_CROSSOVER_RATE
        self.initial_mutation_rate = DEFAULT_MUTATION_RATE
        self.max_generations = max_generations
        self.population_size = population_size
        self.no_improvement_counter = 0
        self.restart_counter_consecutive = 0
        self.restart_percentage = _normalize_restart_percentage(restart_percentage)
        self.max_generations_no_improve = max_generations_no_improve
        self.max_pop_restarts = max_pop_restarts
        self.seed = seed_val
        self.ksize = ksize
        self.top_k_heap = []
        self.best_by_key = {}
        self.prev_best_by_key = {}
        heapify(self.top_k_heap)
        self.dataset_obj = Dataset(data, time_col, event_col)
        self.generator = PopulationGenerator()
        self.alpha = alpha
        self.score_metric = score_metric
        self.km_time_bins = km_time_bins
        self.redundancy_penalty = redundancy_penalty
        self.redundancy_similarity_threshold = redundancy_similarity_threshold
        self.archive_selection = archive_selection
        self.evaluation = RuleEvaluator(
            self.dataset_obj,
            comparacao,
            self.alpha,
            score_metric=self.score_metric,
            km_time_bins=self.km_time_bins,
            redundancy_penalty=self.redundancy_penalty,
            redundancy_similarity_threshold=self.redundancy_similarity_threshold,
        )
        self.operators = GeneticOperators(self.evaluation, self._get_best)
        self.top_n_plot = plot_n_rules
        self.coverage_threshold = coverage_threshold
        seed(self.seed)
        self.debug_performance = debug_performance

    def _get_mask(self, rule: list[list]):
        return self.dataset_obj.get_rule_mask(rule)

    def _jaccard_test(self, mask1, mask2):
        intersection = np.logical_and(mask1, mask2).sum()
        union = np.logical_or(mask1, mask2).sum()
        return intersection / union if union > 0.0 else 0.0

    def _overlap_coefficient(self, mask1, mask2):
        intersection = np.logical_and(mask1, mask2).sum()
        minimum = min(mask1.sum(), mask2.sum())
        return intersection / minimum if minimum > 0 else 0.0

    def _is_redundant(self, new_rule, new_fitness):
        # Base case: the top-k group is empty.
        if not self.best_by_key:
            return False, self._get_mask(new_rule)
        new_mask = self._get_mask(new_rule)
        keys_to_remove = []

        for existing_key, (existing_fitness, _, existing_mask) in self.best_by_key.items():
            iou = self._jaccard_test(new_mask, existing_mask)
            if iou >= self.coverage_threshold:
                if new_fitness <= existing_fitness + EPSILON:
                    return True, new_mask
                else:
                    keys_to_remove.append(existing_key)

        for i in keys_to_remove:
            del self.best_by_key[i]

        return False, new_mask

    def _adjust_interval(self, rule, dataset):

        for i in range(len(rule[0])):
            if type(rule[1][i][0]) == str:
                continue
            col_index = rule[0][i]
            col_values = np.asarray(dataset[:, col_index], dtype=np.float64)
            min_val = col_values.min()
            max_val = col_values.max()
            c_min, c_max = True, True

            if rule[1][i][0] < min_val:
                rule[1][i][0] = min_val
                c_min = False

            if rule[1][i][1] > max_val:
                rule[1][i][1] = max_val
                c_max = False

            int_max_val = rule[1][i][1]

            if c_max:
                nearest_idx = np.argmin(np.abs(col_values - int_max_val))
                rule[1][i][1] = col_values[nearest_idx]

            if c_min:
                int_min_val = rule[1][i][0]
                valid_mask = col_values < int_max_val
                if np.any(valid_mask):
                    candidates = col_values[valid_mask]
                    nearest_idx = np.argmin(np.abs(candidates - int_min_val))
                    rule[1][i][0] = candidates[nearest_idx]

        rule = self._label_rules(rule)

        return rule

    def _get_best(self, fitness_list):
        if not fitness_list:
            return -1
        try:
            return int(np.argmax(fitness_list))
        except (ValueError, TypeError):
            return -1

    def _update_mutation_crossover_rates(self, p_prev_topk: pd.DataFrame, p_curr_topk: pd.DataFrame):
        if self.rate_policy == "fixed":
            return

        if p_curr_topk.equals(p_prev_topk) and self.mutation_rate <= 80 and self.crossover_rate >= RATES_CHANGE_FATOR:
            self.mutation_rate += RATES_CHANGE_FATOR
            self.crossover_rate -= RATES_CHANGE_FATOR

        if (
            not p_curr_topk.equals(p_prev_topk)
            and self.mutation_rate >= RATES_CHANGE_FATOR
            and self.crossover_rate <= 80
        ):
            self.mutation_rate -= RATES_CHANGE_FATOR
            self.crossover_rate += RATES_CHANGE_FATOR

    def _check_stop(self, gen_count):
        prev_top_k_df = pd.DataFrame(self.prev_best_by_key)
        curr_top_k_df = pd.DataFrame(self.best_by_key)
        if gen_count >= self.max_generations:
            return False

        if self.restart_counter_consecutive >= self.max_pop_restarts:
            print(f"\n{'='*70}")
            console.log(
                f"  Stop criterion reached after {self.max_pop_restarts} population restarts.",
                style="bold green",
            )
            print(f"\n{'='*70}")

            return False

        if prev_top_k_df.empty:
            self.prev_best_by_key = self.best_by_key.copy()

            return True

        self._update_mutation_crossover_rates(prev_top_k_df, curr_top_k_df)
        self.prev_best_by_key = self.best_by_key.copy()

        return True

    def _evaluate_improvement(self, population, fitness_list, restart_prct, dataset):
        prev_top_k_df = pd.DataFrame(self.prev_best_by_key)
        curr_top_k_df = pd.DataFrame(self.best_by_key)
        if curr_top_k_df.equals(prev_top_k_df) and self.mutation_rate == 100:
            self.no_improvement_counter += 1

        elif not curr_top_k_df.equals(prev_top_k_df):
            self.no_improvement_counter = 0
            self.restart_counter_consecutive = 0

        if self.no_improvement_counter >= self.max_generations_no_improve and self.mutation_rate == 100:
            new_population = self._population_restart(population, fitness_list, restart_prct, dataset)
            self.no_improvement_counter = 0
            self.restart_counter_consecutive += 1

            self.crossover_rate = DEFAULT_CROSSOVER_RATE
            self.mutation_rate = DEFAULT_MUTATION_RATE

            self.prev_best_by_key = {}

            return new_population

        return None

    def _get_top_k(self, k, fitness_list):
        ordered_top_k_index = []
        fitness_list_copy = list(copy.deepcopy(fitness_list))

        for i in range(k):
            ind = fitness_list_copy.index(max(fitness_list_copy))
            ordered_top_k_index.append(ind)
            fitness_list_copy[ind] = -100

        return ordered_top_k_index

    def _population_restart(self, population, fitness_list, restart_prct, dataset):
        new_population = []
        replacement_qtd = int(math.ceil(len(population) * restart_prct))
        remain_index = self._get_top_k((len(population) - replacement_qtd), fitness_list)
        pop = self.generator.gen_population(replacement_qtd, dataset)

        for i in range(len(remain_index)):
            new_population.append(population[remain_index[i]])
        for i in range(len(pop)):
            new_population.append(pop[i])

        return new_population

    def _update_top_k(self, p_population, p_fitness_list):
        if self.archive_selection == "greedy":
            self._select_greedy_top_k(p_population, p_fitness_list)
            return
        for individual, fitness in zip(p_population, p_fitness_list):
            key = self._rule_key(individual)

            if key in self.best_by_key:
                previous_fit = self.best_by_key[key][0]
                if fitness > previous_fit + EPSILON:
                    mask = self._get_mask(individual)
                    self.best_by_key[key] = (fitness, individual, mask)
                    self.top_k_heap.remove((previous_fit, key))
                    heappush(self.top_k_heap, (fitness, key))
                continue
            is_redundant, new_mask = self._is_redundant(individual, fitness)
            if is_redundant:
                continue
            if len(self.best_by_key) < self.ksize or (self.top_k_heap and fitness > self.top_k_heap[0][0] + EPSILON):
                self._add_rule_to_top_k(key, fitness, individual, new_mask)

                while len(self.best_by_key) > self.ksize:
                    self._prune_heap()
                    if not self.top_k_heap:
                        break
                    _, worst_key = heappop(self.top_k_heap)
                    if worst_key in self.best_by_key:
                        del self.best_by_key[worst_key]

        if len(self.top_k_heap) > len(self.best_by_key):
            self._rebuild_heap_from_topk()

    def _select_greedy_top_k(self, population, raw_fitness):
        """Rebuild the archive with marginal scores against one selected prefix."""
        dataset_x = self.dataset_obj.data
        pool = [(rule, self.evaluation.fitness(rule, dataset_x)) for _, rule, _ in self.best_by_key.values()]
        pool.extend(zip(population, raw_fitness))
        unique = {}
        for rule, quality in pool:
            if not np.isfinite(quality) or quality <= 0.0:
                continue
            # Keep interval alternatives until their attribute key is selected.
            signature = repr(sorted((int(attr), tuple(values)) for attr, values in zip(*rule)))
            unique[signature] = (float(quality), rule, self._get_mask(rule))
        entries = [entry for _, entry in sorted(unique.items(), key=lambda item: (-item[1][0], item[0]))]
        selected = {}
        available = np.ones(len(entries), dtype=bool)
        quality = np.asarray([entry[0] for entry in entries])
        similarity = np.zeros(len(entries))
        curves = [self.evaluation._km_curve(entry[2]) for entry in entries] if self.redundancy_penalty > 0.0 else []
        threshold = self.redundancy_similarity_threshold
        while available.any() and len(selected) < self.ksize:
            redundancy = np.maximum(0.0, (similarity - threshold) / (1.0 - threshold))
            marginal = quality * (1.0 - self.redundancy_penalty * redundancy)
            index = int(np.argmax(np.where(available, marginal, -np.inf)))
            _, rule, mask = entries[index]
            key = self._rule_key(rule)
            selected[key] = (float(max(0.0, marginal[index])), copy.deepcopy(rule), mask.copy())
            for other in np.flatnonzero(available):
                _, other_rule, other_mask = entries[other]
                if (
                    self._rule_key(other_rule) == key
                    or self._jaccard_test(mask, other_mask) >= self.coverage_threshold
                ):
                    available[other] = False
                elif curves:
                    similarity[other] = max(
                        similarity[other],
                        self.evaluation._km_curve_similarity_from_estimates(curves[index], curves[other]),
                    )
        self.best_by_key = selected
        self._rebuild_heap_from_topk()

    def _population_fitness(self, population, dataset_x):
        fitness, population = self.evaluation.get_fitness(population, dataset_x)
        if self.archive_selection == "historical":
            fitness = self.evaluation.penalize_score(
                fitness,
                list(self.best_by_key.values()),
                population,
                dataset_x,
            )
        return fitness, population

    def _rebuild_heap_from_topk(self):
        self.top_k_heap = []
        for key, fit in self.best_by_key.items():
            self.top_k_heap.append((fit[0], key))

        heapify(self.top_k_heap)

    def _add_rule_to_top_k(self, p_rule, p_fitness, p_individual, p_mask):
        self.best_by_key[p_rule] = (p_fitness, p_individual, p_mask)
        heappush(self.top_k_heap, (p_fitness, p_rule))

    def _rule_key(self, p_individual):
        return tuple(sorted(p_individual[0]))

    def _prune_heap(self):
        while self.top_k_heap:
            fit, key = self.top_k_heap[0]
            current = self.best_by_key.get(key)
            if current is not None and abs(current[0] - fit) <= EPSILON:
                return

            heappop(self.top_k_heap)

    def _label_rules(self, p_rules: list) -> list[list]:
        columns_names = list(self.dataset_obj.attr_values.keys())
        for i, atribute in enumerate(p_rules[0]):
            p_rules[0][i] = columns_names[atribute]

        return p_rules

    def run(self):
        start_time = time.time()
        pid = os.getpid()
        profiler = ProcessResourceMonitor(pid)
        if self.debug_performance:
            profiler.start()

        dataset_x = np.array(self.dataset_obj.data)

        mean_fitness_history = []
        best_fitness_history = []
        print(f"\n{'='*70}")
        console.print("--- Starting Top-K Search ---")
        print(f"\n{'='*70}")
        console.print("Configuration:")
        console.print(f"   - Population: {self.population_size}")
        console.print(f"   - Generations: {self.max_generations}")
        console.print(f"   - Top-K: {self.ksize} best rules")
        console.print(f"   - Score metric: {self.score_metric}")
        console.print(f"   - Archive selection: {self.archive_selection}")
        console.print(
            f"   - Redundancy penalty: {self.redundancy_penalty:.2f} "
            f"(similarity > {self.redundancy_similarity_threshold:.2f})"
        )
        console.print(
            f"   - Rate policy: {self.rate_policy} "
            f"(crossover={self.crossover_rate}%, mutation={self.mutation_rate}%)"
        )
        print(f"{'='*70}")

        gen_count = 0
        population = self.generator.gen_population(self.population_size, dataset_x)
        gen_mean_fitness, gen_best_fitness = [], []
        topk_best_fitness_history = []
        restart_generations = []

        with console.status("[bold green] Evolving generations...") as status:
            while self._check_stop(gen_count):
                fitness_list, population = self._population_fitness(population, dataset_x)

                population, fitness_list = self.operators.crossover(
                    population, (self.crossover_rate / 100), fitness_list, dataset_x
                )

                population = self.operators.mutation(population, (self.mutation_rate / 100), fitness_list, dataset_x)

                fitness_list, population = self._population_fitness(population, dataset_x)

                if not fitness_list:
                    console.print(
                        f"WARNING G{gen_count}: empty population - stopping execution.",
                        style="bold red",
                    )
                    break

                self._update_top_k(population, fitness_list)

                new_population = self._evaluate_improvement(
                    population, fitness_list, self.restart_percentage, dataset_x
                )
                if new_population is not None:
                    population = new_population
                    restart_generations.append(gen_count + 1)
                    fitness_list, population = self._population_fitness(population, dataset_x)
                    del new_population

                mean_fit = float(np.mean(fitness_list))
                best_fit = float(np.max(fitness_list))
                topk_best_fit = max((element[0] for element in self.best_by_key.values()), default=float("nan"))
                gen_mean_fitness.append(mean_fit)
                gen_best_fitness.append(best_fit)
                topk_best_fitness_history.append(float(topk_best_fit))
                if (gen_count + 1) % 50 == 0:
                    console.log(
                        f"   Gen {gen_count + 1:3d}: Best={best_fit:.4f} | "
                        f"Mean={mean_fit:.4f} | Top-K={len(self.best_by_key)}/{self.ksize}"
                    )

                gen_count += 1

            mean_fitness_history.append(gen_mean_fitness)
            best_fitness_history.append(gen_best_fitness)
            if gen_best_fitness:
                final_best = gen_best_fitness[-1]
                console.log(
                    f"   Completed: {gen_count} generations | Final Best Fitness = {final_best:.4f}\n",
                    style="bold green",
                )
        print(f"\n{'='*70}")
        console.log(f" Finished in {time.time() - start_time:.2f} s", style="bold green")
        print(f"{'='*70}")

        top_k_values = list(self.best_by_key.values())  # List of tuples (fitness, rule).

        if top_k_values:
            fitnesses = [v[0] for v in top_k_values]
            avg_fit = np.mean(fitnesses)
            best_fit = np.max(fitnesses)
            worst_fit = np.min(fitnesses)
            std_fit = np.std(fitnesses)
            console.print(f"\n Top-{len(top_k_values)} Rule Statistics:")
            console.print(f"   - Mean Fitness: {avg_fit:.4f} (+/-{std_fit:.4f})")
            console.print(f"   - Best: {best_fit:.4f}")
            console.print(f"   - Worst: {worst_fit:.4f}")
        else:
            console.log("\n No valid rules found.", style="bold red")
            avg_fit, best_fit, worst_fit, std_fit = 0.0, 0.0, 0.0, 0.0

        final_metrics = [avg_fit, best_fit, worst_fit, std_fit]

        final_rules_found = []
        sorted_rules = sorted(self.best_by_key.values(), key=lambda x: x[0], reverse=True)

        rules_sizes = []
        raw_rules_scores = []
        penalized_rules_scores = []
        for fit, rule_raw, _ in sorted_rules:
            raw_rules_scores.append(self.evaluation.fitness(rule_raw, dataset_x))
            rule_adjusted = self._adjust_interval(copy.deepcopy(rule_raw), dataset_x)
            final_rules_found.append(rule_adjusted)
            rules_sizes.append(len(rule_adjusted[0]))
            penalized_rules_scores.append(fit)

        total_time = time.time() - start_time
        rules_qtd = len(final_rules_found)
        mean_size = np.mean(rules_sizes) if rules_sizes else 0.0
        console.print("\n Final Results:")
        console.print(f"   - Rules found: {rules_qtd}")
        console.print(f"   - Mean size: {mean_size:.2f} attributes")
        print(f"{'='*70}\n")
        detailed_rules_df = pd.DataFrame(
            {
                "Rule_Obj": [str(r) for r in final_rules_found],
                "Rule_Score": raw_rules_scores,
                "Penalized_Rule_Score": penalized_rules_scores,
            }
        )
        figures_list = RulesPlotter(
            self.dataset_obj._original_data, final_rules_found, self.survival_event_col, self.survival_time_col
        ).kaplan_meier(self.top_n_plot)
        figures_list.append(
            plot_topk_convergency(
                topk_best_fitness_history,
                gen_mean_fitness,
                restart_generations,
            )
        )

        # Info: basic summary.
        common_info = {
            "rules_count": [rules_qtd],
            "total_time": [total_time],
            "mean_size": [mean_size],
            "best_fitness": [final_metrics[1]],
            "score_metric": [self.score_metric],
            "km_time_bins": [self.km_time_bins if self.km_time_bins is not None else 0],
            "redundancy_penalty": [self.redundancy_penalty],
            "redundancy_similarity_threshold": [self.redundancy_similarity_threshold],
            "archive_selection": [self.archive_selection],
            "archive_score_context": ["selection_prefix" if self.archive_selection == "greedy" else "historical"],
            "rate_policy": [self.rate_policy],
            "initial_crossover_rate": [self.initial_crossover_rate],
            "initial_mutation_rate": [self.initial_mutation_rate],
            "final_crossover_rate": [self.crossover_rate],
            "final_mutation_rate": [self.mutation_rate],
        }
        if self.debug_performance:
            performance_stats = profiler.stop()
            info_df = pd.DataFrame(
                {
                    **common_info,
                    "cpu_mean_percent": [performance_stats.cpu_mean_percent],
                    "cpu_peak_percent": [performance_stats.cpu_peak_percent],
                    "ram_mean_mb": [performance_stats.ram_mean_mb],
                    "ram_peak_mb": [performance_stats.ram_peak_mb],
                    "ram_incremental_peak_mb": [performance_stats.ram_incremental_peak_mb],
                    "ram_baseline_mb": [performance_stats.ram_baseline_mb],
                }
            )
        else:
            info_df = pd.DataFrame(common_info)

        return (
            final_metrics,
            mean_fitness_history,
            best_fitness_history,
            total_time,
            rules_qtd,
            info_df,
            detailed_rules_df,
            final_rules_found,
            mean_size,
            figures_list,
        )
