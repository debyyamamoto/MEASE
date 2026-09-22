from dataclasses import dataclass
import numpy as np
import pandas as pd
import pandas.api.types as ptypes
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from lifelines import KaplanMeierFitter

plt.rcParams.update({"font.size": 12})

POPULATION = "Population"
COMPLEMENT = "Complement"


@dataclass
class RulesPlotter:
    dataset: pd.DataFrame
    rules: list[list]
    events_column: str
    time_column: str

    def kaplan_meier(self, num_top: int) -> list[Figure]:
        """
        Plots Kaplan-Meier curves for the rules registered in the Top-K.

        :param num_top: Number of rules shown in the top-n plot.
        :type num_top: int
        """
        if num_top > 0:
            figures_list = []

            figures_list.append(self._multiple_rules_curves(num_top))
            individual_rules_plots = self._individual_rules_curves()
            figures_list.extend(individual_rules_plots)

            return figures_list
        else:
            return []

    def _multiple_rules_curves(self, p_num_top: int):
        fig, ax = plt.subplots(figsize=(12, 10))

        fitter = KaplanMeierFitter(label=POPULATION)
        fitter.fit(self.dataset[self.time_column], self.dataset[self.events_column])
        fitter.plot_survival_function(ax=ax, linestyle="dashed", color="#000000", ci_show=False)

        for idx, rule in enumerate(self.rules[:p_num_top]):
            self._plot_rules_curves(rule, ax, idx)
        ax.set_title(f"Top-{p_num_top} Kaplan-Meier Survival Curves")
        ax.set_xlabel("Time (e.g., months, days)")
        ax.set_ylabel("Survival Probability")
        ax.grid()
        ax.legend()
        ax.set_ylim(0, 1.1)

        return fig

    def _individual_rules_curves(self):
        figures_list = []
        for i, rule in enumerate(self.rules):
            fig, ax = plt.subplots(figsize=(12, 10))

            self._plot_rule_and_complement(rule, ax)
            ax.set_title(f"Top-{i+1} Kaplan-Meier Survival Curves")
            ax.set_xlabel("Time (e.g., months, days)")
            ax.set_ylabel("Survival Probability")
            ax.grid()
            ax.legend()
            ax.set_ylim(0, 1.1)

            figures_list.append(fig)

        return figures_list

    def _plot_rules_curves(self, p_rule: list[list], p_ax: Axes, p_rule_idx: int):
        rule_string = ""
        rule_df = self.dataset.copy()
        atributes_list, constraints_list = p_rule
        for idx, (atribute, constraint) in enumerate(zip(atributes_list, constraints_list)):
            if not ptypes.is_string_dtype(rule_df[atribute].dtype):
                rule_df = rule_df[(rule_df[atribute] >= constraint[0]) & (rule_df[atribute] <= constraint[1])]
                if len(constraints_list) > 1 and idx != len(constraints_list) - 1:
                    rule_string = f"{rule_string} {constraint[0]:.4f}≤{atribute}≤{constraint[1]:.4f} ^"
                else:
                    rule_string = f"{rule_string} {constraint[0]:.4f}≤{atribute}≤{constraint[1]:.4f}"
            else:
                rule_df = rule_df[rule_df[atribute].isin(constraint)]
                if len(constraints_list) > 1 and idx != len(constraints_list) - 1:
                    rule_string = f"{rule_string} {atribute}∈{set(constraint)} ^"
                else:
                    rule_string = f"{rule_string} {atribute}∈{set(constraint)}"

        fitter = KaplanMeierFitter(label=f"Rule {p_rule_idx}")
        fitter.fit(rule_df[self.time_column], rule_df[self.events_column])
        fitter.plot_survival_function(ax=p_ax, ci_show=False)

    def _plot_rule_and_complement(self, p_rule: list[list], p_ax: Axes):
        rule_string = ""
        rule_df = self.dataset.copy()
        rule_complement_df = self.dataset.copy()
        atributes_list, constraints_list = p_rule
        for idx, (atribute, constraint) in enumerate(zip(atributes_list, constraints_list)):
            if not ptypes.is_string_dtype(rule_df[atribute].dtype):
                rule_df = rule_df[(rule_df[atribute] >= constraint[0]) & (rule_df[atribute] <= constraint[1])]
                if len(constraints_list) > 1 and idx != len(constraints_list) - 1:
                    rule_string = f"{rule_string} {constraint[0]:.4f}≤{atribute}≤{constraint[1]:.4f} ^"
                else:
                    rule_string = f"{rule_string} {constraint[0]:.4f}≤{atribute}≤{constraint[1]:.4f}"
            else:
                rule_df = rule_df[rule_df[atribute].isin(constraint)]
                if len(constraints_list) > 1 and idx != len(constraints_list) - 1:
                    rule_string = f"{rule_string} {atribute}∈{set(constraint)} ^"
                else:
                    rule_string = f"{rule_string} {atribute}∈{set(constraint)}"

        complement_indices = self.dataset.index.difference(rule_df.index)
        rule_complement_df = self.dataset.loc[complement_indices]

        fitter = KaplanMeierFitter(label=POPULATION)
        fitter.fit(self.dataset[self.time_column], self.dataset[self.events_column])
        fitter.plot_survival_function(ax=p_ax, linestyle="dashed", ci_show=False)

        if rule_complement_df.empty:
            rule_complement_df = self.dataset
        fitter = KaplanMeierFitter(label=COMPLEMENT)
        fitter.fit(rule_complement_df[self.time_column], rule_complement_df[self.events_column])
        fitter.plot_survival_function(ax=p_ax)

        fitter = KaplanMeierFitter(label=rule_string)
        fitter.fit(rule_df[self.time_column], rule_df[self.events_column])
        fitter.plot_survival_function(ax=p_ax)


def plot_topk_convergency(
    p_topk_best_fit: list,
    p_gen_mean_fit: list | None = None,
    p_restart_gens: list[int] | None = None,
):
    fig, ax = plt.subplots(figsize=(12, 10))
    if p_gen_mean_fit is not None:
        mean_gens_array = np.arange(1, len(p_gen_mean_fit) + 1)
        ax.plot(mean_gens_array, p_gen_mean_fit, label="Population mean score")

    topk_gens_array = np.arange(1, len(p_topk_best_fit) + 1)
    ax.plot(topk_gens_array, p_topk_best_fit, label="Historical Top-K best score")

    restart_label = "Population restart"
    for restart_gen in p_restart_gens or []:
        ax.axvline(
            restart_gen,
            color="#777777",
            linestyle=":",
            linewidth=1.5,
            alpha=0.75,
            label=restart_label,
        )
        restart_label = "_nolegend_"

    ax.grid()
    ax.set_title("Score Convergence")
    ax.legend()
    ax.set_xlabel("Generations")
    ax.set_ylabel("Score")

    return fig
