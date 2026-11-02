from __future__ import annotations

import argparse
from itertools import combinations
import math
import re
from glob import glob
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from scipy import stats

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

EXPERIMENTS_DIR = Path(__file__).resolve().parent
DEFAULT_DESIGNS = [
    EXPERIMENTS_DIR / "designs" / "factorial_runtime.csv",
    # EXPERIMENTS_DIR / "designs" / "factorial_score.csv",
    EXPERIMENTS_DIR / "designs" / "factorial_topk_mean_score.csv",
]
DEFAULT_RESULTS_DIR = EXPERIMENTS_DIR / "factorial_2k"
DEFAULT_OUTPUT_DIR = EXPERIMENTS_DIR / "factorial_analysis"
DEFAULT_BASELINE = "complement"
ALL_METRICS = "all"
EXCLUDED_DESIGN_COLUMNS = {"experiment", "factor", "response", "dataset", "results_subdir"}
INFO_FILE = "Info"
METRICS_FILE = "RulesMetricsResult"
DETAILED_RULES_FILE = "DetailedRules"
RULE_SCORE_COLUMN = "Rule_Score"
MEAN_RULE_SCORE_RESPONSE = "mean_rule_score"
RULE_SCORE_RESPONSES = {MEAN_RULE_SCORE_RESPONSE, "rule_score_mean", RULE_SCORE_COLUMN}
IDENTITY_TRANSFORM = "identity"
SUPPORTED_TRANSFORMS = {
    IDENTITY_TRANSFORM,
    "log",
    "sqrt",
    "arcsin_sqrt",
    "logit",
    "box_cox",
    "yeo_johnson",
}
TRANSFORM_ALIASES = {
    "none": IDENTITY_TRANSFORM,
    "raw": IDENTITY_TRANSFORM,
}
LOGIT_EPSILON = 1e-6


def normalize_transform_name(transform: str) -> str:
    transform = str(transform).strip().lower().replace("-", "_")
    transform = TRANSFORM_ALIASES.get(transform, transform)
    if transform not in SUPPORTED_TRANSFORMS:
        supported = ", ".join(sorted(SUPPORTED_TRANSFORMS | set(TRANSFORM_ALIASES)))
        raise ValueError(f"Unsupported response transform '{transform}'. Supported values: {supported}")
    return transform


def parse_response_transforms(transform_specs: list[str] | None) -> dict[str, str] | None:
    if transform_specs is None:
        return None

    transforms = {}
    for spec in transform_specs:
        if "=" not in spec:
            raise ValueError("Response transforms must use 'response=transform' syntax. " f"Got: {spec}")
        response, transform = spec.split("=", 1)
        transforms[response.strip()] = normalize_transform_name(transform)
    return transforms


def select_response_transform(
    response: str,
    response_transforms: dict[str, str] | None,
) -> str:
    if response_transforms is not None:
        if response in response_transforms:
            return response_transforms[response]
        if "all" in response_transforms:
            return response_transforms["all"]
        return IDENTITY_TRANSFORM

    return IDENTITY_TRANSFORM


def transform_output_suffix(transform: str, explicit_transforms: bool) -> str:
    if not explicit_transforms:
        return ""
    return "" if transform == IDENTITY_TRANSFORM else f"_{transform}"


def require_range(values: np.ndarray, transform: str, lower: float, upper: float) -> None:
    if ((values < lower) | (values > upper)).any():
        raise ValueError(f"The '{transform}' transform requires response values in [{lower}, {upper}].")


def apply_response_transform(values: pd.Series, transform: str) -> tuple[np.ndarray, dict[str, float | str]] | None:
    transform = normalize_transform_name(transform)
    numeric_values = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    if np.isnan(numeric_values).any():
        print("Response values contain NaN after numeric conversion.")

    metadata: dict[str, float | str] = {"response_transform": transform}
    if transform == IDENTITY_TRANSFORM:
        return numeric_values, metadata

    if transform == "log":
        if (numeric_values <= 0).any():
            print("The 'log' transform requires strictly positive response values.")
        return np.log(numeric_values), metadata

    if transform == "sqrt":
        if (numeric_values < 0).any():
            print("The 'sqrt' transform requires non-negative response values.")
        return np.sqrt(numeric_values), metadata

    if transform == "arcsin_sqrt":
        require_range(numeric_values, transform, 0.0, 1.0)
        return np.arcsin(np.sqrt(numeric_values)), metadata

    if transform == "logit":
        require_range(numeric_values, transform, 0.0, 1.0)
        clipped = np.clip(numeric_values, LOGIT_EPSILON, 1 - LOGIT_EPSILON)
        metadata["transform_epsilon"] = LOGIT_EPSILON
        return np.log(clipped / (1 - clipped)), metadata

    # if transform == "box_cox":
    #     if (numeric_values <= 0).any():
    #         raise ValueError("The 'box_cox' transform requires strictly positive response values.")
    #     transformed, lambda_value = stats.boxcox(numeric_values)
    #     metadata["transform_lambda"] = float(lambda_value)
    #     return transformed, metadata

    # transformed, lambda_value = stats.yeojohnson(numeric_values)
    # metadata["transform_lambda"] = float(lambda_value)
    # return transformed, metadata


def parse_dataset_spec(dataset_spec: str) -> tuple[str, Path, str, str]:
    parts = str(dataset_spec).split("|")
    if len(parts) != 5:
        raise ValueError(
            "Dataset specifications must use 'name|path|time_col|event_col|label'. " f"Got: {dataset_spec}"
        )
    dataset_name, dataset_path, time_col, event_col, _ = parts
    return dataset_name, Path(dataset_path), time_col, event_col


def factor_columns(design_df: pd.DataFrame) -> list[str]:
    factors = []
    for column in design_df.columns:
        if column in EXCLUDED_DESIGN_COLUMNS:
            continue
        if design_df[column].nunique(dropna=False) > 1:
            factors.append(column)
    return factors


def factor_aliases(factors: list[str]) -> dict[str, str]:
    alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    if len(factors) > len(alphabet):
        raise ValueError("This script supports at most 26 factors for alias generation.")
    return {factor: alphabet[index] for index, factor in enumerate(factors)}


def decode_factor_levels(factor_label, factors: list[str]) -> dict[str, int]:
    levels = [int(value) for value in str(factor_label).split("_")]
    if len(levels) != len(factors):
        raise ValueError(
            f"Factor label '{factor_label}' has {len(levels)} levels, "
            f"but the design has {len(factors)} varying factors: {factors}"
        )
    return dict(zip(factors, levels))


def interaction_terms(factors: list[str]) -> list[tuple[str, ...]]:
    return [term for order in range(1, len(factors) + 1) for term in combinations(factors, order)]


def term_alias(term: tuple[str, ...], aliases: dict[str, str]) -> str:
    return "".join(aliases[factor] for factor in term)


def term_sign(levels: dict[str, int], term: tuple[str, ...]) -> int:
    sign = 1
    for factor in term:
        sign *= levels[factor]
    return sign


def execution_from_file(path: Path, dataset_name: str, baseline: str, suffix: str) -> int:
    pattern = rf"^{re.escape(dataset_name)}_(\d+)_{re.escape(baseline)}_{suffix}\.csv$"
    match = re.match(pattern, path.name)
    if not match:
        raise ValueError(f"Could not parse execution number from {path.name}")
    return int(match.group(1))


def response_file_suffix(response: str) -> str:
    if response == "total_time":
        return INFO_FILE
    if response in RULE_SCORE_RESPONSES:
        return DETAILED_RULES_FILE
    return METRICS_FILE


def response_value_from_file(result_df: pd.DataFrame, response: str, response_file: Path) -> float:
    if response in result_df.columns:
        return float(pd.to_numeric(result_df[response], errors="coerce").mean())

    if response in RULE_SCORE_RESPONSES:
        if RULE_SCORE_COLUMN not in result_df.columns:
            raise KeyError(f"Missing '{RULE_SCORE_COLUMN}' column in {response_file}")
        return float(pd.to_numeric(result_df[RULE_SCORE_COLUMN], errors="coerce").mean())

    raise KeyError(f"Missing response column '{response}' in {response_file}")


def resolve_treatment_dir(
    results_dir: Path,
    response: str,
    treatment: pd.Series,
    dataset_name: str,
    baseline: str,
    suffix: str,
) -> Path:
    config = str(treatment["factor"])
    response_dirs = [response]
    if "results_subdir" in treatment.index and not pd.isna(treatment["results_subdir"]):
        response_dirs.insert(0, str(treatment["results_subdir"]))

    for response_dir in dict.fromkeys(response_dirs):
        treatment_dir = results_dir / response_dir / config / dataset_name / baseline
        if treatment_dir.exists():
            return treatment_dir

    if response in RULE_SCORE_RESPONSES:
        candidates = []
        for treatment_dir in results_dir.glob(f"*/{config}/{dataset_name}/{baseline}"):
            if any(treatment_dir.glob(f"{dataset_name}_*_{baseline}_{suffix}.csv")):
                candidates.append(treatment_dir)
        if len(candidates) == 1:
            print(f"Using detailed-rule outputs from: {candidates[0]}")
            return candidates[0]

    return results_dir / response / config / dataset_name / baseline


def collect_response_rows(
    design_df: pd.DataFrame,
    results_dir: Path,
    baseline: str,
    response_transform: str,
) -> pd.DataFrame:
    rows = []
    response = str(design_df.loc[0, "response"])
    suffix = response_file_suffix(response)

    for _, treatment in design_df.iterrows():
        dataset_name, dataset_path, time_col, event_col = parse_dataset_spec(treatment["dataset"])
        treatment_dir = resolve_treatment_dir(results_dir, response, treatment, dataset_name, baseline, suffix)
        if not treatment_dir.exists():
            print(f"Missing treatment directory: {treatment_dir}")
            continue

        for response_file in sorted(treatment_dir.glob(f"{dataset_name}_*_{baseline}_{suffix}.csv")):
            result_df = pd.read_csv(response_file)

            row = {
                "experiment": treatment["experiment"],
                "response": response,
                "config": treatment["factor"],
                "dataset_name": dataset_name,
                "dataset_path": str(dataset_path),
                "time_col": time_col,
                "event_col": event_col,
                "execution": execution_from_file(response_file, dataset_name, baseline, suffix),
                "response_file": str(response_file),
                "y_raw": response_value_from_file(result_df, response, response_file),
            }
            for column in design_df.columns:
                if column not in {"experiment", "response", "factor", "dataset"}:
                    row[column] = treatment[column]
            rows.append(row)

    rows_df = pd.DataFrame(rows)
    if rows_df.empty:
        return rows_df

    transformed_y, transform_metadata = apply_response_transform(rows_df["y_raw"], response_transform)
    rows_df["y"] = transformed_y
    for column, value in transform_metadata.items():
        rows_df[column] = value

    return rows_df


def build_factor_mapping(design_df: pd.DataFrame, factors: list[str]) -> pd.DataFrame:
    aliases = factor_aliases(factors)
    rows = []
    for factor in factors:
        levels = (
            design_df[["factor", factor]]
            .assign(coded_level=design_df["factor"].map(lambda label: decode_factor_levels(label, factors)[factor]))
            .drop_duplicates(subset=["coded_level"])
            .sort_values("coded_level")
        )
        low = levels.loc[levels["coded_level"] == -1, factor].iloc[0]
        high = levels.loc[levels["coded_level"] == 1, factor].iloc[0]
        rows.append(
            {
                "alias": aliases[factor],
                "factor": factor,
                "low_level_minus_1": low,
                "high_level_plus_1": high,
            }
        )
    return pd.DataFrame(rows)


def build_factorial_df(
    responses_df: pd.DataFrame,
    design_df: pd.DataFrame,
    factors: list[str],
) -> pd.DataFrame:
    aliases = factor_aliases(factors)
    terms = interaction_terms(factors)
    rows = []

    for _, treatment in design_df.iterrows():
        config = str(treatment["factor"])
        treatment_responses = responses_df[responses_df["config"] == config].sort_values("execution")
        y_values = treatment_responses["y"].to_numpy(dtype=float)
        levels = decode_factor_levels(config, factors)
        row = {
            "config": config,
            "I": 1,
            "y": tuple(float(value) for value in y_values),
            "n": len(y_values),
            "y_total": float(y_values.sum()),
            "y_mean": float(y_values.mean()) if len(y_values) else math.nan,
            "SSY_i": float((y_values**2).sum()),
            "SSE_i": float(((y_values - y_values.mean()) ** 2).sum()) if len(y_values) else math.nan,
        }
        for factor in factors:
            row[f"{aliases[factor]}_factor"] = factor
            row[f"{aliases[factor]}_level"] = treatment[factor]
            row[aliases[factor]] = levels[factor]
        for term in terms:
            alias = term_alias(term, aliases)
            if len(term) > 1:
                row[alias] = term_sign(levels, term)
        for index, value in enumerate(y_values, start=1):
            row[f"y_{index}"] = float(value)
        rows.append(row)

    return pd.DataFrame(rows)


def build_factorial_display_df(factorial_df: pd.DataFrame, factors: list[str]) -> pd.DataFrame:
    aliases = factor_aliases(factors)
    sign_columns = ["I", *[term_alias(term, aliases) for term in interaction_terms(factors)]]
    total_row = {"config": "Total"}
    total_row: dict

    coefficient_row = {"config": "Total/2^k"}

    for column in sign_columns:
        total = float((factorial_df[column] * factorial_df["y_mean"]).sum())
        total_row[column] = total
        coefficient_row[column] = total / (2 ** len(factors))

    display_df = factorial_df.copy()
    return pd.concat([display_df, pd.DataFrame([total_row, coefficient_row])], ignore_index=True)


def balanced_replications(factorial_df: pd.DataFrame) -> tuple[bool, float]:
    counts = sorted(factorial_df["n"].dropna().unique())
    if len(counts) == 1:
        return True, int(counts[0])
    return False, np.nan


def sum_squares_summary(factorial_df: pd.DataFrame) -> dict[str, float | int]:
    balanced, r = balanced_replications(factorial_df)

    n_total = int(factorial_df["n"].sum())
    y_total = float(factorial_df["y_total"].sum())
    treatment_count = int(factorial_df.shape[0])

    ssy = float(factorial_df["SSY_i"].sum())

    q0 = factorial_df["y_mean"].sum() / treatment_count
    ss0 = treatment_count * r * (q0**2)

    sse = float(factorial_df["SSE_i"].sum())
    sst = float(ssy - ss0)
    df = int(treatment_count * (r - 1)) if balanced else int(n_total - treatment_count)
    mse = sse / df if df > 0 else math.nan
    s_e = math.sqrt(mse) if math.isfinite(mse) else math.nan
    return {
        "treatment_count": treatment_count,
        "n_total": n_total,
        "y_total": y_total,
        "q0": q0,
        "SSY": ssy,
        "SS0": ss0,
        "SSE": sse,
        "SST": sst,
        "df_error": df,
        "MSE": mse,
        "s_e": s_e,
    }


def variance_fraction(sum_squares: float, total_sum_squares: float) -> float:
    if not math.isfinite(sum_squares) or total_sum_squares <= 0:
        return math.nan
    return float(sum_squares / total_sum_squares)


def build_effects_df(
    factorial_df: pd.DataFrame,
    factors: list[str],
    alpha: float,
) -> pd.DataFrame:
    aliases = factor_aliases(factors)
    terms = interaction_terms(factors)
    summary = sum_squares_summary(factorial_df)
    balanced, r = balanced_replications(factorial_df)
    treatment_count = int(summary["treatment_count"])
    n_total = int(summary["n_total"])
    mse = float(summary["MSE"])
    s_e = float(summary["s_e"])
    df_error = int(summary["df_error"])
    t_critical = stats.t.ppf(1 - alpha / 2, df_error) if df_error > 0 else math.nan
    s_qi = s_e / math.sqrt(n_total) if df_error > 0 and n_total > 0 else math.nan
    rows = []

    for term in terms:
        alias = term_alias(term, aliases)
        contrast_total = float((factorial_df[alias] * factorial_df["y_total"]).sum())
        contrast_mean_total = float((factorial_df[alias] * factorial_df["y_mean"]).sum())

        q_i = contrast_mean_total / treatment_count
        ss_i = n_total * (q_i**2) if balanced else math.nan
        ms_i = ss_i
        f_value = ms_i / mse if mse > 0 and math.isfinite(ms_i) else math.nan
        p_value = stats.f.sf(f_value, 1, df_error) if math.isfinite(f_value) and df_error > 0 else math.nan
        ci_margin = t_critical * s_qi if math.isfinite(t_critical) and math.isfinite(s_qi) else math.nan
        ci_low = q_i - ci_margin if math.isfinite(ci_margin) else math.nan
        ci_high = q_i + ci_margin if math.isfinite(ci_margin) else math.nan
        ci_contains_zero = (
            bool(ci_low <= 0 <= ci_high) if math.isfinite(ci_low) and math.isfinite(ci_high) else math.nan
        )
        relevant_by_ci = not ci_contains_zero if isinstance(ci_contains_zero, bool) else math.nan
        decision_by_ci = (
            "relevant" if relevant_by_ci is True else "not_relevant" if relevant_by_ci is False else "undefined"
        )

        rows.append(
            {
                "term": alias,
                "factors": ":".join(term),
                "order": len(term),
                "contrast_total": contrast_total,
                "contrast_mean_total": contrast_mean_total,
                "q_i": q_i,
                "SS_i": ss_i,
                "df_i": 1,
                "df_error": df_error,
                "MS_i": ms_i,
                "F": f_value,
                "p_value": p_value,
                "MSE": mse,
                "s_e": s_e,
                "s_qi": s_qi,
                "alpha_sig": alpha,
                "confidence_level": 1 - alpha,
                "t_critical": t_critical,
                "ci_margin": ci_margin,
                "effect_ci_low": ci_low,
                "effect_ci_high": ci_high,
                "ci_contains_zero": ci_contains_zero,
                "relevant_by_ci": relevant_by_ci,
                "decision_by_ci": decision_by_ci,
                "variance_explained": variance_fraction(ss_i, float(summary["SST"])),
                "variance_explained_percent": 100 * variance_fraction(ss_i, float(summary["SST"])),
                "balanced_replications": balanced,
                "r": r,
            }
        )

    return pd.DataFrame(rows)


def build_confidence_intervals_df(effects_df: pd.DataFrame, factorial_df: pd.DataFrame, alpha: float) -> pd.DataFrame:
    summary = sum_squares_summary(factorial_df)
    balanced, r = balanced_replications(factorial_df)
    rows = []
    for _, effect in effects_df.iterrows():
        rows.append(
            {
                "term": effect["term"],
                "factors": effect["factors"],
                "q_i": effect["q_i"],
                "SSE": summary["SSE"],
                "df_error_formula": "2^k * (r - 1)",
                "df_error": summary["df_error"],
                "MSE_formula": "SSE / df_error",
                "MSE": summary["MSE"],
                "s_e_formula": "sqrt(MSE)",
                "s_e": summary["s_e"],
                "n_total_formula": "2^k * r",
                "n_total": summary["n_total"],
                "s_qi_formula": "s_e / sqrt(2^k * r)",
                "s_qi": effect["s_qi"],
                "alpha_sig": alpha,
                "confidence_level": 1 - alpha,
                "t_critical_formula": "t.ppf(1 - alpha_sig / 2, df_error)",
                "t_critical": effect["t_critical"],
                "ci_margin_formula": "t_critical * s_qi",
                "ci_margin": effect["ci_margin"],
                "ci_low_formula": "q_i - ci_margin",
                "effect_ci_low": effect["effect_ci_low"],
                "ci_high_formula": "q_i + ci_margin",
                "effect_ci_high": effect["effect_ci_high"],
                "ci_contains_zero": effect["ci_contains_zero"],
                "relevant_by_ci": effect["relevant_by_ci"],
                "balanced_replications": balanced,
                "r": r,
            }
        )
    return pd.DataFrame(rows)


def build_ss_steps_df(factorial_df: pd.DataFrame, effects_df: pd.DataFrame) -> pd.DataFrame:
    summary = sum_squares_summary(factorial_df)
    sst = float(summary["SST"])
    rows = [
        {
            "step": "SSY",
            "source": "response",
            "formula": "SSY = sum(y_ij^2)",
            "value": summary["SSY"],
            "df": math.nan,
            "mean_square": math.nan,
            "variance_explained": math.nan,
            "variance_explained_percent": math.nan,
        },
        {
            "step": "SS0",
            "source": "correction",
            "formula": "SS0 = (sum(y_ij)^2) / (2^k r)",
            "value": summary["SS0"],
            "df": 1,
            "mean_square": math.nan,
            "variance_explained": math.nan,
            "variance_explained_percent": math.nan,
        },
    ]

    for _, effect in effects_df.iterrows():
        rows.append(
            {
                "step": "SS_i",
                "source": effect["term"],
                "formula": "SS_i = (2^k r) * q_i^2",
                "value": effect["SS_i"],
                "df": effect["df_i"],
                "df_error": effect["df_error"],
                "mean_square": effect["MS_i"],
                "q_i": effect["q_i"],
                "s_e": effect["s_e"],
                "s_qi": effect["s_qi"],
                "t_critical": effect["t_critical"],
                "ci_margin": effect["ci_margin"],
                "effect_ci_low": effect["effect_ci_low"],
                "effect_ci_high": effect["effect_ci_high"],
                "relevant_by_ci": effect["relevant_by_ci"],
                "variance_explained": effect["variance_explained"],
                "variance_explained_percent": effect["variance_explained_percent"],
            }
        )

    error_fraction = variance_fraction(float(summary["SSE"]), sst)
    rows.extend(
        [
            {
                "step": "SSE",
                "source": "experimental_error",
                "formula": "SSE = sum_i sum_j (y_ij - mean(y_i))^2",
                "value": summary["SSE"],
                "df": summary["df_error"],
                "mean_square": summary["MSE"],
                "s_e": summary["s_e"],
                "variance_explained": error_fraction,
                "variance_explained_percent": 100 * error_fraction,
            },
            {
                "step": "SST",
                "source": "total",
                "formula": "SST = SSY - SS0 = sum(SS_i) + SSE",
                "value": summary["SST"],
                "df": int(summary["n_total"] - 1),
                "mean_square": math.nan,
                "variance_explained": 1.0,
                "variance_explained_percent": 100.0,
            },
        ]
    )
    return pd.DataFrame(rows)


def build_variance_explained_df(effects_df: pd.DataFrame, factorial_df: pd.DataFrame) -> pd.DataFrame:
    summary = sum_squares_summary(factorial_df)
    rows = (
        effects_df[["term", "factors", "order", "SS_i", "variance_explained", "variance_explained_percent"]]
        .rename(columns={"SS_i": "sum_squares"})
        .to_dict("records")
    )

    error_fraction = variance_fraction(float(summary["SSE"]), float(summary["SST"]))
    rows.append(
        {
            "term": "SSE",
            "factors": "experimental_error",
            "order": math.nan,
            "sum_squares": summary["SSE"],
            "variance_explained": error_fraction,
            "variance_explained_percent": 100 * error_fraction,
        }
    )
    rows.append(
        {
            "term": "SST",
            "factors": "total",
            "order": math.nan,
            "sum_squares": summary["SST"],
            "variance_explained": 1.0,
            "variance_explained_percent": 100.0,
        }
    )
    return pd.DataFrame(rows).sort_values("variance_explained_percent", ascending=False)


def build_coefficients_df(factorial_df: pd.DataFrame, effects_df: pd.DataFrame) -> pd.DataFrame:
    summary = sum_squares_summary(factorial_df)
    rows = [
        {
            "term": "I",
            "factors": "intercept",
            "order": 0,
            "q_i": summary["q0"],
            "SS_i": summary["SS0"],
            "df_i": 1,
            "MS_i": math.nan,
            "F": math.nan,
            "p_value": math.nan,
            "MSE": summary["MSE"],
            "s_e": summary["s_e"],
            "s_qi": math.nan,
            "df_error": summary["df_error"],
            "alpha_sig": math.nan,
            "confidence_level": math.nan,
            "t_critical": math.nan,
            "ci_margin": math.nan,
            "effect_ci_low": math.nan,
            "effect_ci_high": math.nan,
            "ci_contains_zero": math.nan,
            "relevant_by_ci": math.nan,
            "decision_by_ci": math.nan,
            "variance_explained": math.nan,
            "variance_explained_percent": math.nan,
            "balanced_replications": effects_df["balanced_replications"].iloc[0] if not effects_df.empty else math.nan,
            "r": effects_df["r"].iloc[0] if not effects_df.empty else math.nan,
        }
    ]
    rows: list
    rows.extend(effects_df.to_dict("records"))
    return pd.DataFrame(rows)


def build_residuals_df(
    responses_df: pd.DataFrame,
    coefficients_df: pd.DataFrame,
    factors: list[str],
) -> pd.DataFrame:
    aliases = factor_aliases(factors)
    terms = interaction_terms(factors)
    coefficients = coefficients_df.set_index("term")["q_i"].to_dict()
    q0 = float(coefficients["I"])
    residuals_df = responses_df.copy().sort_values(["config", "execution"]).reset_index(drop=True)
    residuals_df["I"] = 1
    residuals_df["q0"] = q0
    residuals_df["y_hat_main_effects"] = q0
    residuals_df["y_hat_full_factorial"] = q0

    for factor in factors:
        alias = aliases[factor]
        residuals_df[alias] = residuals_df["config"].map(lambda label: decode_factor_levels(label, factors)[factor])

    for term in terms:
        alias = term_alias(term, aliases)
        if alias not in residuals_df.columns:
            residuals_df[alias] = residuals_df["config"].map(
                lambda label, current_term=term: term_sign(decode_factor_levels(label, factors), current_term)
            )
        q_i = float(coefficients[alias])
        residuals_df["y_hat_full_factorial"] += residuals_df[alias] * q_i
        if len(term) == 1:
            residuals_df["y_hat_main_effects"] += residuals_df[alias] * q_i

    residuals_df["residual_main_effects"] = residuals_df["y"] - residuals_df["y_hat_main_effects"]
    residuals_df["treatment_mean"] = residuals_df.groupby("config")["y"].transform("mean")
    residuals_df["full_factorial_minus_treatment_mean"] = (
        residuals_df["y_hat_full_factorial"] - residuals_df["treatment_mean"]
    )
    residuals_df["residual_full_factorial"] = residuals_df["y"] - residuals_df["y_hat_full_factorial"]
    residuals_df["experiment_order"] = np.arange(1, len(residuals_df) + 1)
    residuals_df["abs_residual_full_factorial"] = residuals_df["residual_full_factorial"].abs()
    residuals_df["squared_residual_full_factorial"] = residuals_df["residual_full_factorial"] ** 2
    return residuals_df


def breusch_pagan_manual(
    residuals_df: pd.DataFrame,
    design_columns: list[str],
    residual_column: str = "residual_full_factorial",
) -> tuple[float, float]:
    residuals_squared = residuals_df[residual_column].to_numpy(dtype=float) ** 2
    design = residuals_df[design_columns].to_numpy(dtype=float)
    aux_design = np.column_stack([np.ones(len(design)), design])
    beta, *_ = np.linalg.lstsq(aux_design, residuals_squared, rcond=None)
    fitted = aux_design @ beta
    ss_res = float(((residuals_squared - fitted) ** 2).sum())
    ss_tot = float(((residuals_squared - residuals_squared.mean()) ** 2).sum())
    r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
    statistic = float(len(residuals_squared) * r_squared)
    df = int(aux_design.shape[1] - 1)
    return statistic, float(stats.chi2.sf(statistic, df))


def build_assumption_checks_df(
    residuals_df: pd.DataFrame,
    factorial_df: pd.DataFrame,
    effects_df: pd.DataFrame,
    factors: list[str],
    alpha: float,
) -> pd.DataFrame:
    aliases = factor_aliases(factors)
    factorial_terms = [term_alias(term, aliases) for term in interaction_terms(factors)]
    residuals = residuals_df["residual_full_factorial"].to_numpy(dtype=float)
    fitted = residuals_df["y_hat_full_factorial"].to_numpy(dtype=float)
    checks = []

    balanced, _ = balanced_replications(factorial_df)
    checks.append(
        {
            "check_type": "model_assumption",
            "assumption": "balanced_2kr_design",
            "statistic": factorial_df["n"].nunique(),
            "p_value": math.nan,
            "passed": balanced,
            "notes": "The classroom 2^k r formulas assume the same number of replications in every treatment.",
        }
    )

    max_reconstruction_error = float(residuals_df["full_factorial_minus_treatment_mean"].abs().max())
    checks.append(
        {
            "check_type": "model_assumption",
            "assumption": "full_factorial_reconstructs_treatment_means",
            "statistic": max_reconstruction_error,
            "p_value": math.nan,
            "passed": bool(max_reconstruction_error <= 1e-10),
            "notes": (
                "The complete 2^k model uses I, main effects, and interactions as additive coded terms "
                "on the selected response scale."
            ),
        }
    )

    interactions = effects_df[effects_df["order"] > 1]
    if not interactions.empty:
        min_interaction_p = float(interactions["p_value"].min())
        checks.append(
            {
                "check_type": "interpretation_diagnostic",
                "assumption": "main_effects_only_additivity_screen",
                "statistic": math.nan,
                "p_value": min_interaction_p,
                "passed": bool(min_interaction_p >= alpha),
                "notes": (
                    "Significant interactions do not invalidate the complete factorial model; they mean main effects "
                    "should not be interpreted alone."
                ),
            }
        )

    if len(residuals) >= 3:
        shapiro_stat, shapiro_p = stats.shapiro(residuals)
        checks.append(
            {
                "check_type": "model_assumption",
                "assumption": "normal_residuals",
                "statistic": shapiro_stat,
                "p_value": shapiro_p,
                "passed": bool(shapiro_p >= alpha),
                "notes": "Uses pure experimental error y_ij - y_hat_full_factorial. Also inspect qq_plot.png.",
            }
        )

    bp_stat, bp_p = breusch_pagan_manual(residuals_df, factorial_terms)
    checks.append(
        {
            "check_type": "model_assumption",
            "assumption": "homoscedastic_residuals_breusch_pagan",
            "statistic": bp_stat,
            "p_value": bp_p,
            "passed": bool(bp_p >= alpha),
            "notes": "Manual BP test using squared full-factorial residuals and coded factorial terms.",
        }
    )

    groups = [group["y"].to_numpy(dtype=float) for _, group in residuals_df.groupby("config") if len(group) > 1]
    if len(groups) >= 2:
        levene_stat, levene_p = stats.levene(*groups, center="median")
        checks.append(
            {
                "check_type": "model_assumption",
                "assumption": "homoscedastic_response_by_treatment_levene",
                "statistic": levene_stat,
                "p_value": levene_p,
                "passed": bool(levene_p >= alpha),
                "notes": "Compares response variance across treatment cells.",
            }
        )

    corr_order = np.corrcoef(residuals_df["experiment_order"], residuals)[0, 1]
    checks.append(
        {
            "check_type": "visual_diagnostic",
            "assumption": "residuals_are_cloud_over_execution_order",
            "statistic": corr_order,
            "p_value": math.nan,
            "passed": math.nan,
            "notes": "Visual check: inspect residuals_by_execution.png for trends or blocks.",
        }
    )

    corr_fitted = np.corrcoef(fitted, residuals)[0, 1]
    checks.append(
        {
            "check_type": "visual_diagnostic",
            "assumption": "residuals_are_cloud_over_fitted",
            "statistic": corr_fitted,
            "p_value": math.nan,
            "passed": math.nan,
            "notes": "Primary visual check for homoscedasticity: inspect residuals_vs_fitted.png.",
        }
    )

    corr_observed = np.corrcoef(residuals_df["y"], residuals)[0, 1]
    checks.append(
        {
            "check_type": "supplementary_diagnostic",
            "assumption": "residuals_are_cloud_over_observed_response",
            "statistic": corr_observed,
            "p_value": math.nan,
            "passed": math.nan,
            "notes": (
                "Supplementary plot only: observed response equals fitted plus residual, "
                "so residuals_vs_response.png can show a mechanical pattern."
            ),
        }
    )

    checks.append(
        {
            "check_type": "protocol_assumption",
            "assumption": "independent_experiments_protocol",
            "statistic": math.nan,
            "p_value": math.nan,
            "passed": math.nan,
            "notes": "This cannot be proven from the CSVs; it depends on randomized, fresh executions without cache/warm starts.",
        }
    )
    return pd.DataFrame(checks)


def save_diagnostic_plots(residuals_df: pd.DataFrame, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    residuals = residuals_df["residual_full_factorial"].to_numpy(dtype=float)
    fitted = residuals_df["y_hat_full_factorial"].to_numpy(dtype=float)
    fitted_main_effects = residuals_df["y_hat_main_effects"].to_numpy(dtype=float)
    main_effects_residuals = residuals_df["residual_main_effects"].to_numpy(dtype=float)
    y = residuals_df["y"].to_numpy(dtype=float)
    residual_std = residuals.std(ddof=1)
    standardized = residuals / residual_std if residual_std > 0 else residuals
    residuals_by_config_df = residuals_df.groupby("config")["residual_full_factorial"].agg(list)

    plt.figure(figsize=(7, 5))
    plt.scatter(residuals_df["experiment_order"], residuals, alpha=0.8)
    plt.axhline(0, color="black", linewidth=1)
    plt.xlabel("Experiment order")
    plt.ylabel("Full-factorial residual")
    plt.title("Full-factorial residuals by experiment order")
    plt.tight_layout()
    plt.savefig(output_dir / "residuals_by_execution.png", dpi=200)
    plt.close()

    fig, ax = plt.subplots()
    ax.axhline(0, color="black", linewidth=1)
    idx_list = []
    for idx, values in enumerate(residuals_by_config_df):
        idx_array = np.ones_like(values) * idx
        ax.scatter(idx_array, values)
        idx_list.append(idx)
    ax.set_xticks(idx_list)
    ax.set_xticklabels(residuals_by_config_df.index)
    plt.title("Residual by Config")
    ax.set_xlabel("config")
    ax.set_ylabel("Full-factorial residual")
    fig.savefig(output_dir / "residual_by_config.png")
    plt.close()

    plt.figure(figsize=(7, 5))
    plt.scatter(y, fitted, alpha=0.8)
    plt.xlabel("Observed response")
    plt.ylabel("Full-factorial fitted response")
    plt.title("Observed response vs fitted response")
    plt.tight_layout()
    plt.savefig(output_dir / "response_vs_fitted.png", dpi=200)
    plt.close()

    plt.figure(figsize=(7, 5))
    plt.scatter(fitted, residuals, alpha=0.8)
    plt.axhline(0, color="black", linewidth=1)
    plt.xlabel("Full-factorial fitted response")
    plt.ylabel("Full-factorial residual")
    plt.title("Full-factorial residuals vs fitted response")
    plt.tight_layout()
    plt.savefig(output_dir / "residuals_vs_fitted.png", dpi=200)
    plt.close()

    plt.figure(figsize=(7, 5))
    plt.scatter(y, residuals, alpha=0.8)
    plt.axhline(0, color="black", linewidth=1)
    plt.xlabel("Observed response")
    plt.ylabel("Full-factorial residual")
    plt.title("Full-factorial residuals vs observed response")
    plt.tight_layout()
    plt.savefig(output_dir / "residuals_vs_response.png", dpi=200)
    plt.close()

    plt.figure(figsize=(7, 5))
    plt.scatter(fitted_main_effects, main_effects_residuals, alpha=0.8)
    plt.axhline(0, color="black", linewidth=1)
    plt.xlabel("Main-effects fitted response")
    plt.ylabel("Main-effects residual")
    plt.title("Main-effects residuals vs fitted response")
    plt.tight_layout()
    plt.savefig(output_dir / "additive_residuals_vs_fitted.png", dpi=200)
    plt.close()

    plt.figure(figsize=(7, 5))
    stats.probplot(residuals, dist="norm", plot=plt)
    plt.title("Q-Q plot")
    plt.tight_layout()
    plt.savefig(output_dir / "qq_plot.png", dpi=200)
    plt.close()

    plt.figure(figsize=(7, 5))
    plt.scatter(fitted, np.sqrt(np.abs(standardized)), alpha=0.8)
    plt.xlabel("Full-factorial fitted response")
    plt.ylabel("sqrt(|standardized residual|)")
    plt.title("Scale-location")
    plt.tight_layout()
    plt.savefig(output_dir / "scale_location.png", dpi=200)
    plt.close()


def analyze_design(
    design_file: Path,
    results_dir: Path,
    output_dir: Path,
    baseline: str,
    alpha: float,
    response_transforms: dict[str, str] | None,
) -> None:
    design_df = pd.read_csv(design_file)
    response = str(design_df.loc[0, "response"])
    response_transform = select_response_transform(response, response_transforms)
    factors = factor_columns(design_df)
    if len(factors) == 1:
        response = str(design_df.loc[0, "response"])
        if response == ALL_METRICS:
            levels_table = {}
            levels_table: dict[str, dict[str, dict]]
            repetitions = 0
            for _, treatment in design_df.iterrows():
                dataset_name, _, _, _ = parse_dataset_spec(treatment["dataset"])
                config = str(treatment["factor"])
                experiment = treatment["experiment"]
                treatment_dir = Path(results_dir / experiment / response / config / dataset_name / baseline)

                levels_table[config] = {}

                stats_metrics_df = pd.read_csv(treatment_dir / f"{dataset_name}_{baseline}_RulesMetricsResult.csv")
                if "metric" not in stats_metrics_df.columns:
                    print("\n")
                    print(treatment_dir / f"{dataset_name}_{baseline}_RulesMetricsResult.csv")
                    continue
                for metric in stats_metrics_df["metric"][:-3]:
                    metrics_files = glob(f"{treatment_dir}/*RulesMetricsResult.csv")
                    metrics_files.remove(f"{treatment_dir}/{dataset_name}_{baseline}_RulesMetricsResult.csv")
                    repetitions = len(metrics_files)

                    y_mean = stats_metrics_df[stats_metrics_df["metric"] == metric]["mean"].values[0]
                    y_ij_list = []
                    errors_list = []
                    for file in metrics_files:
                        metrics_value = pd.read_csv(file)[metric].values[0]
                        y_ij_list.append(metrics_value)
                        errors_list.append(metrics_value - y_mean)

                    levels_table[config][metric] = {"y_j_mean": y_mean, "errors_ij": errors_list, "y_ij": y_ij_list}

            global_info = {}
            metric_info_dict = {}
            f_critical = stats.f.ppf(
                1 - alpha, len(levels_table.keys()) - 1, len(levels_table.keys()) * (repetitions - 1)
            )
            for level in levels_table.keys():
                for metric in levels_table.get(level, {}).keys():
                    if metric_info_dict.get(metric, None) is None:
                        metric_info_dict[metric] = {}
                        metric_info_dict[metric]["y_ij"] = levels_table[level][metric]["y_ij"]
                        metric_info_dict[metric]["mean"] = [levels_table[level][metric]["y_j_mean"]]
                        metric_info_dict[metric]["error"] = levels_table[level][metric]["errors_ij"]
                    else:
                        metric_info_dict[metric]["y_ij"].extend(levels_table[level][metric]["y_ij"])
                        metric_info_dict[metric]["mean"].append(levels_table[level][metric]["y_j_mean"])
                        metric_info_dict[metric]["error"].extend(levels_table[level][metric]["errors_ij"])
            for metric, metric_values in metric_info_dict.items():
                global_info[metric] = {
                    "mean": np.mean(metric_values["mean"]),
                    "SSY": np.sum(np.square(metric_values["y_ij"])),
                    "SSE": np.sum(np.square(metric_values["error"])),
                    "SSA": 0.0,
                }
                global_info[metric]["SS0"] = (
                    len(levels_table.keys()) * repetitions * np.square(global_info[metric]["mean"])
                )
                global_info[metric]["SST"] = global_info[metric]["SSY"] - global_info[metric]["SS0"]

            for level in levels_table.keys():
                for metric in levels_table.get(level, {}).keys():
                    levels_table[level][metric]["alpha_j"] = (
                        levels_table[level][metric]["y_j_mean"] - global_info[metric]["mean"]
                    )
                    global_info[metric]["SSA"] += np.square(levels_table[level][metric]["alpha_j"])

            for metric, metric_values in metric_info_dict.items():
                global_info[metric]["SSA"] *= repetitions
                global_info[metric]["Explained Var"] = (
                    global_info[metric]["SSA"] / global_info[metric]["SST"]
                    if global_info[metric]["SST"] > 0.0
                    else 0.0
                )
                global_info[metric]["se^2"] = global_info[metric]["SSE"] / (
                    len(levels_table.keys()) * (repetitions - 1)
                )
                global_info[metric]["MSA"] = global_info[metric]["SSA"] / (len(levels_table.keys()) - 1)
                global_info[metric]["MSE"] = global_info[metric]["SSE"] / (
                    len(levels_table.keys()) * (repetitions - 1)
                )
                global_info[metric]["F_calc"] = (
                    global_info[metric]["MSA"] / global_info[metric]["MSE"]
                    if global_info[metric]["MSE"] > 0.0
                    else -np.inf
                )
                global_info[metric]["significant"] = global_info[metric]["F_calc"] > f_critical

            print(global_info)
    else:
        responses_df = collect_response_rows(design_df, results_dir, baseline, response_transform)
        if responses_df.empty:
            print(f"No rows collected for {design_file}. Skipping.")
            return

        experiment = str(design_df.loc[0, "experiment"])
        output_dir.mkdir(parents=True, exist_ok=True)

        factorial_df = build_factorial_df(responses_df, design_df, factors)
        factorial_display_df = build_factorial_display_df(factorial_df, factors)
        effects_df = build_effects_df(factorial_df, factors, alpha)
        coefficients_df = build_coefficients_df(factorial_df, effects_df)
        confidence_intervals_df = build_confidence_intervals_df(effects_df, factorial_df, alpha)
        ss_steps_df = build_ss_steps_df(factorial_df, effects_df)
        variance_df = build_variance_explained_df(effects_df, factorial_df)
        residuals_df = build_residuals_df(responses_df, coefficients_df, factors)
        assumptions_df = build_assumption_checks_df(residuals_df, factorial_df, effects_df, factors, alpha)
        response_label = f"{response}{transform_output_suffix(response_transform, response_transforms is not None)}"
        diagnostics_dir = output_dir / f"{experiment}_{response_label}_diagnostics"

        prefix = output_dir / f"{experiment}_{response_label}"
        responses_df.to_csv(f"{prefix}_raw_responses.csv", index=False)
        factorial_df.to_csv(f"{prefix}_treatments.csv", index=False)
        factorial_display_df.to_csv(f"{prefix}_factorial_df.csv", index=False)
        coefficients_df.to_csv(f"{prefix}_coefficients.csv", index=False)
        effects_df.to_csv(f"{prefix}_effects.csv", index=False)
        confidence_intervals_df.to_csv(f"{prefix}_confidence_intervals.csv", index=False)
        ss_steps_df.to_csv(f"{prefix}_ss_steps.csv", index=False)
        variance_df.to_csv(f"{prefix}_variance_explained.csv", index=False)
        residuals_df.to_csv(f"{prefix}_residuals.csv", index=False)
        assumptions_df.to_csv(f"{prefix}_assumption_checks.csv", index=False)
        save_diagnostic_plots(residuals_df, diagnostics_dir)

        print(f"\n=== {experiment}: {response} ===")
        print(f"Response transform: {response_transform}")
        print(f"Factors: {', '.join(factors)}")
        print(f"Rows: {len(responses_df)} responses, {len(factorial_df)} treatments")
        print(f"Saved classroom factorial table: {prefix}_factorial_df.csv")
        print(f"Saved coefficients: {prefix}_coefficients.csv")
        print(f"Saved confidence intervals: {prefix}_confidence_intervals.csv")
        print(f"Saved SS steps: {prefix}_ss_steps.csv")
        print(f"Saved variance explained: {prefix}_variance_explained.csv")
        print(f"Saved assumption checks: {prefix}_assumption_checks.csv")
        print(f"Saved diagnostic plots: {diagnostics_dir}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Manual 2^k r factorial analysis for MEASE experiment outputs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--designs", nargs="+", type=Path, default=DEFAULT_DESIGNS)
    parser.add_argument("--results_dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--baseline", default=DEFAULT_BASELINE)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument(
        "--response_transforms",
        nargs="*",
        default=None,
        help=(
            "Explicit response transforms using response=transform syntax. "
            "Supported transforms: identity, log, sqrt, arcsin_sqrt, logit, box_cox, yeo_johnson. "
            "Use all=transform to apply the same transform to every design. "
            "For bounded fractional responses such as mean_rule_score, prefer logit over log."
        ),
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    response_transforms = parse_response_transforms(args.response_transforms)
    for design_file in args.designs:
        analyze_design(
            design_file=design_file,
            results_dir=args.results_dir,
            output_dir=args.output_dir,
            baseline=args.baseline,
            alpha=args.alpha,
            response_transforms=response_transforms,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
