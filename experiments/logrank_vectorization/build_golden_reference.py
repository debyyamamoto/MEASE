"""Fase 1 da vetorização do log-rank: gabarito de referência.

Roda o `RuleEvaluator.fitness()` ATUAL (baseado em `statsmodels.duration.survdiff`)
sobre um conjunto fixo de regras que cobrem os casos-limite relevantes para o
teste log-rank, e salva p-valor / estatística / fitness resultantes em CSV.

Esse gabarito deve ser gerado e congelado ANTES de qualquer mudança em
`easd/evaluation.py`. A Fase 5 da vetorização compara a implementação nova
contra este arquivo, não contra uma nova execução do statsmodels.

Uso:
    uv run python experiments/logrank_vectorization/build_golden_reference.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import statsmodels.api as sm

from easd.dataset import Dataset
from easd.evaluation import RuleEvaluator

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT = SCRIPT_DIR / "golden_reference_logrank.csv"
ALPHA = 0.5
SEED = 42
N_ROWS = 2000


def make_synthetic_dataset(n: int = N_ROWS, seed: int = SEED) -> pd.DataFrame:
    """Dataset sintético com atributos numéricos e categóricos e tempo de
    sobrevivência influenciado por eles, para que as regras abaixo produzam
    separações reais (não só ruído) entre subgrupo e o resto.
    """
    rng = np.random.default_rng(seed)

    age = rng.uniform(20, 90, n)
    biomarker1 = rng.normal(0, 1, n)
    biomarker2 = rng.normal(5, 2, n)
    sex = rng.choice(["M", "F"], size=n)
    stage = rng.choice(["I", "II", "III", "IV"], size=n, p=[0.4, 0.3, 0.2, 0.1])
    treatment = rng.choice(["A", "B"], size=n, p=[0.5, 0.5])

    stage_effect = pd.Series(stage).map({"I": 0.0, "II": 0.3, "III": 0.6, "IV": 1.0}).to_numpy()
    linear_pred = 0.02 * (age - 55) + stage_effect - 0.3 * (treatment == "B")
    event_scale = np.exp(-linear_pred) * 500
    event_time = rng.exponential(scale=event_scale, size=n)

    # Censura bem mais forte no braço "B", pra existir subgrupo com poucos eventos.
    censor_scale = np.where(treatment == "B", 200.0, 800.0)
    censor_time = rng.exponential(scale=censor_scale, size=n)

    time = np.minimum(event_time, censor_time)
    status = (event_time <= censor_time).astype(int)

    # Arredondar pra forçar empates de tempo (evento x evento e evento x censura
    # no mesmo instante) — é exatamente o caso-limite que a Fase 5 precisa cobrir.
    time = np.clip(np.round(time), 1.0, None)

    return pd.DataFrame(
        {
            "age": age,
            "biomarker1": biomarker1,
            "biomarker2": biomarker2,
            "sex": pd.Series(sex, dtype="string"),
            "stage": pd.Series(stage, dtype="string"),
            "treatment": pd.Series(treatment, dtype="string"),
            "time": time,
            "status": status,
        }
    )


def make_rule(dataset_obj: Dataset, conditions: dict[str, list]) -> list:
    """Converte {nome_coluna: intervalo_ou_valores} pro formato interno de regra
    ([indices_de_atributo], [intervalos]) esperado por `get_rule_mask`.
    """
    attr_idx = [dataset_obj.get_col_index(col) for col in conditions]
    intervals = list(conditions.values())
    return [attr_idx, intervals]


def threshold_for_support(values: np.ndarray, target_support: float) -> float:
    """Acha o menor valor tal que `col >= valor` cobre ~target_support da população."""
    sorted_vals = np.sort(values)
    n = len(sorted_vals)
    k = int(round(target_support * n))
    k = max(1, min(n - 1, k))
    return float(sorted_vals[n - k])


def build_rule_cases(dataset_obj: Dataset, data: pd.DataFrame) -> list[tuple[str, list]]:
    age = data["age"].to_numpy()
    age_max = float(age.max())

    near_5pct = threshold_for_support(age, 0.05)
    near_55pct = threshold_for_support(age, 0.55)

    return [
        (
            "one_numeric_attr",
            make_rule(dataset_obj, {"age": [60.0, 90.0]}),
        ),
        (
            "one_categorical_attr",
            make_rule(dataset_obj, {"stage": ["III", "IV"]}),
        ),
        (
            "four_attrs_mixed",
            make_rule(
                dataset_obj,
                {
                    "age": [40.0, 75.0],
                    "biomarker1": [-2.0, 2.0],
                    "sex": ["F"],
                    "treatment": ["A"],
                },
            ),
        ),
        (
            "near_5pct_support",
            make_rule(dataset_obj, {"age": [near_5pct, age_max + 1.0]}),
        ),
        (
            "near_55pct_support",
            make_rule(dataset_obj, {"age": [near_55pct, age_max + 1.0]}),
        ),
        (
            "empty_group_unknown_category",
            make_rule(dataset_obj, {"stage": ["STAGE_NAO_EXISTE"]}),
        ),
        (
            "full_population_100pct",
            make_rule(dataset_obj, {"age": [0.0, age_max + 1.0]}),
        ),
        (
            "high_censoring_subgroup",
            make_rule(dataset_obj, {"treatment": ["B"]}),
        ),
    ]


def _describe_tail_risk_set(times: np.ndarray) -> tuple[int, int]:
    """Diagnóstico (não entra no CSV): quantos tempos de evento únicos têm
    conjunto de risco combinado <= 1 -- é aí que a variância do log-rank
    degenera (0/0) e precisa de tratamento explícito na Fase 5.
    """
    unique_times = np.unique(times)
    n_j = np.array([(times >= t).sum() for t in unique_times])
    return int((n_j <= 1).sum()), int(len(unique_times))


def evaluate_case(
    evaluator: RuleEvaluator,
    dataset_obj: Dataset,
    case_id: str,
    rule: list,
    comparacao: str,
) -> dict:
    n_total = dataset_obj.size
    rule_group_indices = evaluator.get_covered_indices(rule, dataset_obj)
    n_covered = len(rule_group_indices)
    relative_support = n_covered / n_total if n_total else 0.0

    real_survdiff = sm.duration.survdiff
    captured: dict[str, tuple] = {}

    def _spy(*args, **kwargs):
        result = real_survdiff(*args, **kwargs)
        captured["value"] = result
        return result

    with patch("statsmodels.api.duration.survdiff", side_effect=_spy):
        fitness_value = evaluator.fitness(rule, dataset_obj)

    stat, p_value = captured.get("value", (float("nan"), float("nan")))

    return {
        "case_id": case_id,
        "comparacao": comparacao,
        "alpha": ALPHA,
        "n_total": n_total,
        "n_covered": n_covered,
        "relative_support": relative_support,
        "survdiff_called": "value" in captured,
        "chi2_stat": stat,
        "p_value": p_value,
        "fitness": fitness_value,
        "rule_json": json.dumps(rule),
    }


def build_golden_reference(output_file: Path = DEFAULT_OUTPUT) -> pd.DataFrame:
    data = make_synthetic_dataset()
    dataset_obj = Dataset(data, "time", "status")

    n_degenerate, n_unique_times = _describe_tail_risk_set(data["time"].to_numpy())
    print(
        f"[diagnóstico] {n_degenerate}/{n_unique_times} tempos de evento únicos têm "
        f"risk-set combinado <= 1 (variância degenerada) -- caso presente naturalmente "
        f"em qualquer regra com suporte válido, não precisa de caso dedicado."
    )

    rows = []
    for comparacao in ("population", "complement"):
        evaluator = RuleEvaluator(dataset_obj, comparacao, ALPHA)
        for case_id, rule in build_rule_cases(dataset_obj, data):
            rows.append(evaluate_case(evaluator, dataset_obj, case_id, rule, comparacao))

    df = pd.DataFrame(rows)

    # Checagem de sanidade: hoje "population" e "complement" particionam o
    # dataset da mesma forma (ver ressalva na avaliação da Fase 4), então o
    # p-valor deve bater entre os dois comparacao para a mesma regra.
    pivot = df.pivot(index="case_id", columns="comparacao", values="p_value")
    mismatches = pivot[(pivot["population"] - pivot["complement"]).abs() > 1e-9]
    if not mismatches.empty:
        print("[aviso] population != complement para os casos:")
        print(mismatches)
    else:
        print("[ok] population e complement produzem o mesmo p-valor em todos os casos.")

    data.to_parquet(output_file.with_name(output_file.stem + "_dataset.parquet"))
    output_file.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_file, index=False)
    print(df.to_string(index=False))
    print(f"\nGabarito salvo em {output_file}")
    print(f"Dataset sintético salvo em {output_file.with_name(output_file.stem + '_dataset.parquet')}")
    return df


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Gera o gabarito de referência (Fase 1) da vetorização do log-rank.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--output_file", type=Path, default=DEFAULT_OUTPUT)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    build_golden_reference(args.output_file)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
