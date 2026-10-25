"""Friedman → Nemenyi for individual metrics and their mean rank, in repeated or single mode."""

from __future__ import annotations

import argparse
import itertools
import json
import os
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import scikit_posthocs as sp
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.analyze_random_search import METRICS, markdown_table, rank_panel, read_results
from experiments.repeat_random_search import (
    DEFAULT_OUTPUT,
    command_for,
    completed,
    lock,
    outputs_for,
    read_source_manifest,
    sha256,
    validate_metrics,
    verify_shell_run,
    write_json,
)


def friedman(matrix, ascending):
    """Rank dataset means; use the Iman–Davenport F approximation (Demšar §3.2.2)."""
    matrix = np.round(np.asarray(matrix, dtype=float), 12)
    if matrix.ndim != 2 or min(matrix.shape) < 3 or not np.isfinite(matrix).all():
        raise ValueError("Friedman requires >=3 datasets/configurations and finite values")
    n, k = matrix.shape
    ranks = stats.rankdata(matrix if ascending else -matrix, axis=1)
    # scipy returns NaN when every configuration is tied within every dataset.
    if np.all(np.ptp(ranks, axis=1) == 0):
        chi2, p_chi2, f, p_f = 0.0, 1.0, 0.0, 1.0
    else:
        chi2, p_chi2 = stats.friedmanchisquare(*ranks.T)
        denominator = n * (k - 1) - chi2
        f = (n - 1) * chi2 / denominator if denominator > 1e-12 else np.inf
        p_f = stats.f.sf(f, k - 1, (k - 1) * (n - 1))
    return ranks, dict(
        friedman_chi2=float(chi2),
        p_chi2=float(p_chi2),
        iman_davenport_f=float(f),
        df1=k - 1,
        df2=(k - 1) * (n - 1),
        p_value=float(p_f),
    )


def load_panel(directory):
    plan = json.loads((directory / "plan.json").read_text())
    if plan.get("protocol") != "mease-repetitions-v1":
        raise ValueError("Unsupported repetition protocol")
    ids, seeds = plan["config_ids"], plan["seeds"]
    datasets = [d["name"] for d in plan["datasets"]]
    expected = set(itertools.product(ids, datasets, seeds))
    if (
        len(ids) < 3
        or len(datasets) < 3
        or len(seeds) < 2
        or len(seeds) != plan["repetitions"]
        or len(set(ids)) != len(ids)
        or len(set(datasets)) != len(datasets)
        or len(set(seeds)) != len(seeds)
        or len(plan["jobs"]) != len(expected)
        or {tuple(j) for j in plan["jobs"]} != expected
    ):
        raise ValueError("Expected a balanced panel, >=3 configurations/datasets and >=2 distinct seeds")
    # for name, digest in plan["hashes"].items():
    #     if sha256(ROOT / name) != digest:
    #         raise ValueError(f"Experiment code/data fingerprint differs: {name}")
    rows, missing, hashes = [], [], {}
    for cid, dataset, seed in plan["jobs"]:
        _, folder = command_for(plan, directory, cid, dataset, seed)
        required = [
            folder / "command.sh",
            *outputs_for(folder, dataset, plan["baseline"]),
            folder / "metrics_full.json",
        ]
        if not completed(folder, required):
            missing.append(f"config_{cid}/{dataset}/seed_{seed}")
            continue
        values = json.loads((folder / "metrics_full.json").read_text())
        validate_metrics(values)
        rows.append(dict(config_id=cid, dataset=dataset, seed=seed, **values))
        hashes[str(folder.relative_to(directory) / ".complete.json")] = sha256(folder / ".complete.json")
    if missing:
        raise ValueError(
            f"Incomplete/invalid panel: {len(missing)} of {len(expected)} runs. "
            f"Resume repeat_random_search.sh before analyzing. Examples: {', '.join(missing[:5])}"
        )
    return plan, pd.DataFrame(rows), hashes


def load_single_panel(directory):
    """Select the original top 10; use only the saved single-run search metrics."""
    fields, datasets, _ = read_source_manifest(directory)
    if not {"baseline", "num_configs", "executions", "run_seed"}.issubset(fields):
        raise ValueError("Incomplete Random Search protocol metadata")
    configs, runs = read_results(directory, fields["baseline"])
    names = [d["name"] for d in datasets]
    expected = set(itertools.product(range(len(configs)), names, [0]))
    actual = set(runs[["config_id", "dataset", "run"]].itertuples(index=False, name=None))
    if (
        len(configs) < 3
        or len(names) < 3
        or len(set(names)) != len(names)
        or len(configs) != int(fields["num_configs"])
        or int(fields["executions"]) != 1
        or actual != expected
        or len(runs) != len(expected)
        or not runs.metadata_matches.all()
    ):
        raise ValueError("Expected a complete single-run Random Search panel with consistent metadata")
    hashes = {name: sha256(directory / name) for name in ("protocol.txt", "configs.txt")}
    for row in runs.to_dict("records"):
        validate_metrics(row)
        folder = Path(row["source"]).parent
        verify_shell_run(folder)
        marker = folder / ".complete"
        hashes[str(marker.relative_to(directory))] = sha256(marker)
    # Same selection policy as repeat_random_search; then re-rank within the top 10.
    _, ranking = rank_panel(runs)
    ids = ranking.config_id.head(10).astype(int).tolist()
    seed = int(fields["run_seed"])
    selected = runs.loc[runs.config_id.isin(ids), ["config_id", "dataset", *METRICS, "#sg"]].copy()
    selected["seed"] = seed
    plan = dict(
        config_ids=ids,
        datasets=datasets,
        seeds=[seed],
        repetitions=1,
        baseline=fields["baseline"],
        source=str(directory),
        selection="original Random Search top 10 by equal metric/dataset mean ranks; ties ordered by ID",
    )
    return plan, selected, hashes


def analyze(runs, ids, datasets, seeds, alpha=0.05):
    if not 0 < alpha < 0.5:
        raise ValueError("Require 0 < alpha < 0.5")
    expected = set(itertools.product(ids, datasets, seeds))
    keys = ["config_id", "dataset", "seed"]
    if (
        len(ids) < 3
        or len(datasets) < 3
        or len(seeds) < 1
        or len(expected) != len(ids) * len(datasets) * len(seeds)
        or runs.duplicated(keys).any()
        or set(runs[keys].itertuples(index=False, name=None)) != expected
    ):
        raise ValueError("Expected a complete balanced panel with distinct configurations/datasets/seeds")
    if not np.isfinite(runs[list(METRICS)].to_numpy(dtype=float)).all():
        raise ValueError("Metrics must be finite")
    means = runs.groupby(["config_id", "dataset"], as_index=False)[[*METRICS, "#sg"]].mean()

    # Average repetitions BEFORE ranking: datasets remain the independent blocks.
    rank_cells = means.copy()
    rank_cells[list(METRICS)] = rank_cells[list(METRICS)].round(12)
    per_dataset, ranking = rank_panel(rank_cells)
    # The composite is a score: Friedman must rank it again within each dataset.
    per_dataset["rank_mean_metric_rank"] = (
        per_dataset.mean_metric_rank.round(12).groupby(per_dataset.dataset).rank(method="average")
    )
    rank_columns = [column for _, column in METRICS.values()]
    ranking = ranking.merge(per_dataset.groupby("config_id")[rank_columns].mean(), on="config_id")
    ranking = ranking[["config_id", "position", *rank_columns, "mean_rank"]]
    global_rows, rank_rows, pair_rows = [], [], []
    n, k = len(datasets), len(ids)

    # Descriptive CD for tables/plots; pairwise decisions use scikit-posthocs p-values.
    cd = float(stats.studentized_range.ppf(1 - alpha, k, np.inf) * np.sqrt(k * (k + 1) / (12 * n)))
    outcomes = {metric: ascending for metric, (ascending, _) in METRICS.items()}
    outcomes["mean_metric_rank"] = True
    for metric, ascending in outcomes.items():
        matrix = per_dataset.pivot(index="dataset", columns="config_id", values=metric).loc[datasets, ids].to_numpy()
        ranks, test = friedman(matrix, ascending)
        reject = test["p_value"] < alpha

        # Reuse Friedman's ranks to preserve metric directions and rounded ties.
        pvalues = sp.posthoc_nemenyi_friedman(ranks) if reject else None
        global_rows.append(
            dict(
                metric=metric,
                datasets=n,
                configs=k,
                repetitions=len(seeds),
                **test,
                alpha=alpha,
                reject_global=reject,
                nemenyi_cd=cd,
            )
        )
        avg = ranks.mean(axis=0)
        for cid, value in zip(ids, avg):
            rank_rows.append(dict(metric=metric, config_id=cid, mean_rank=float(value)))
        for a, b in itertools.combinations(range(k), 2):
            diff = float(avg[a] - avg[b])  # Negative = a has better rank.
            # A non-rejected omnibus test does not authorize post-hoc conclusions.
            p = float(pvalues.iloc[a, b]) if reject else np.nan
            significant = reject and p < alpha
            pair_rows.append(
                dict(
                    metric=metric,
                    config_a=ids[a],
                    config_b=ids[b],
                    mean_rank_a=float(avg[a]),
                    mean_rank_b=float(avg[b]),
                    rank_difference_a_minus_b=diff,
                    nemenyi_cd=cd,
                    p_nemenyi=p,
                    posthoc_performed=reject,
                    significant=significant,
                    decision=("significant" if significant else "not_significant") if reject else "not_tested",
                )
            )
    pairs = pd.DataFrame(pair_rows)
    return dict(
        runs=runs,
        dataset_means=means,
        ranking=ranking,
        ranks_per_dataset=per_dataset,
        metric_ranks=pd.DataFrame(rank_rows),
        friedman=pd.DataFrame(global_rows),
        nemenyi=pairs,
        nemenyi_significant=pairs[pairs.significant].copy(),
        nemenyi_not_significant=pairs[pairs.decision.eq("not_significant")].copy(),
    )


def plots(tables, output):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(12, 10), layout="constrained")
    for ax, metric in zip(axes.flat, tables["friedman"].metric):
        frame = tables["metric_ranks"].query("metric == @metric").sort_values("mean_rank")
        ax.scatter(frame.mean_rank, np.arange(len(frame)))
        ax.set(
            yticks=np.arange(len(frame)),
            yticklabels=[f"config_{c}" for c in frame.config_id],
            xlabel="Rank médio (menor é melhor)",
            title=metric,
            xlim=(0.5, len(frame) + 0.5),
        )
        ax.invert_yaxis()
        ax.grid(axis="x", alpha=0.25)
        cd = tables["friedman"].set_index("metric").loc[metric, "nemenyi_cd"]
        ax.text(0.02, -0.16, f"Diferença crítica Nemenyi: {cd:.3f}", transform=ax.transAxes)
    fig.savefig(output / "mean_ranks.png", dpi=160)
    fig.savefig(output / "mean_ranks.pdf")
    plt.close(fig)


def report(plan, tables, alpha):
    tests, pairs = tables["friedman"], tables["nemenyi"]
    ranking = tables["ranking"]
    mode = plan.get("mode", "repeated")
    observations = (
        "Usamos diretamente as métricas salvas em RulesMetricsResult.csv da execução original do Random Search, "
        "com a precisão disponível nesses arquivos. Não são carregadas execuções repetidas. "
        if mode == "single"
        else "Primeiro calculamos a média das repetições de cada métrica por configuração/dataset. "
    )
    findings = []
    for row in tests.itertuples():
        group = pairs[pairs.metric.eq(row.metric)]
        result = (
            f"pares com diferença significativa: {int(group.significant.sum())}; "
            f"sem diferença detectada por Nemenyi: {int(group.decision.eq('not_significant').sum())}."
            if row.reject_global
            else "H₀ não rejeitada; Nemenyi não aplicado."
        )
        findings.append(f"- **{row.metric}**: p de Friedman (Iman–Davenport)={row.p_value:.6g}; {result}")
    significant = tables["nemenyi_significant"]
    return (
        "\n\n".join(
            [
                "# Comparação das configurações: Friedman → Nemenyi",
                f"Painel: {len(plan['config_ids'])} configurações × {len(plan['datasets'])} datasets × "
                f"{plan['repetitions']} execução(ões) por configuração/dataset = {len(tables['runs'])} execuções. "
                f"Modo: `{mode}`. Referência: `{plan['baseline']}`. Seleção: `{plan['selection']}`.",
                "## Mean rank",
                observations
                + "Em cada dataset, maior exceptionality e setCov recebem melhor rank; menor model redundancy recebe melhor rank. "
                "Empates recebem rank médio (arredondamento a 12 casas somente para atribuir ranks). "
                "As colunas rank_exceptionality, rank_redundancy e rank_coverage são médias entre datasets. "
                "mean_rank é a média desses três ranks, com pesos iguais; menor é melhor. "
                "Além dos testes individuais, avaliamos o escore combinado mean_metric_rank.",
                markdown_table(ranking),
                "Em cada dataset, mean_metric_rank=(rank_exceptionality+rank_redundancy+rank_coverage)/3. "
                "Friedman ordena novamente esses escores entre configurações, atribuindo rank médio aos empates. "
                "Esses novos ranks estão em rank_mean_metric_rank de `ranks_per_dataset.csv`; suas médias "
                "alimentam Nemenyi e a linha mean_metric_rank de `metric_ranks.csv`. "
                "Elas podem diferir do mean_rank descritivo de `ranking.csv`, que é a média dos escores compostos. "
                "O teste combinado avalia essa combinação com pesos iguais; não exige diferenças nas três métricas individualmente.",
                "## Friedman por métrica e escore combinado",
                "\n".join(findings),
                f"Cada teste usa datasets como blocos: N={len(plan['datasets'])}, K={len(plan['config_ids'])}. "
                "Cada configuração contribui com um valor por dataset: a execução original em single, "
                "ou a média das repetições em repeated. "
                "Usamos Friedman corrigido por empates e sua aproximação F de Iman–Davenport "
                "(Demšar, seção 3.2.2): F=(N−1)χ²/[N(K−1)−χ²]. "
                f"A decisão usa p_value < α={alpha:g}; p_chi2 é apenas referência. "
                "Iman–Davenport aproxima a distribuição do teste; não é ajuste de multiplicidade.",
                markdown_table(
                    tests.assign(
                        p_value=tests.p_value.map(lambda p: f"{p:.6g}"),
                        p_chi2=tests.p_chi2.map(lambda p: f"{p:.6g}"),
                    )
                ),
                "## Nemenyi: comparação de todos os pares",
                "Nemenyi é aplicado somente nas métricas em que Friedman rejeita H₀. "
                "Os p-valores são calculados por scikit_posthocs.posthoc_nemenyi_friedman sobre a matriz de ranks "
                "(linhas=datasets, colunas=configurações); a decisão usa p_nemenyi < α. "
                "CD=qα√[K(K+1)/(6N)], onde qα é o quantil da amplitude studentizada dividido por √2. "
                "Um par é significativo quando |rank(A)−rank(B)| > CD. "
                f"Usamos α={alpha:g} por métrica, sem Holm ou Bonferroni adicional. "
                "Nemenyi já controla as comparações entre todos os pares dentro da métrica; "
                "não se afirma controle conjunto de erro entre as três métricas e o escore combinado.",
                (
                    markdown_table(
                        significant[["metric", "config_a", "config_b", "rank_difference_a_minus_b", "p_nemenyi"]]
                    )
                    if len(significant)
                    else "Nenhum par com diferença significativa detectada."
                ),
                "`nemenyi.csv` lista todos os pares, seus ranks, diferença crítica, p e decisão. "
                "`nemenyi_significant.csv` contém os pares significativos; `nemenyi_not_significant.csv` contém "
                "os pares testados sem diferença detectada. `not_tested` indica que Friedman não rejeitou H₀ "
                "e o p de Nemenyi fica vazio. Diferença negativa favorece A. "
                "Um resultado não significativo não demonstra equivalência entre configurações.",
                "## Interpretação e arquivos",
                "O protocolo segue as seções 3.2.2 e 3.2.3 de "
                "[Demšar (2006)](https://jmlr.org/papers/v7/demsar06a.html). "
                "Os datasets devem ser independentes. Com poucas bases, as aproximações e o poder dos testes são limitados. "
                "Esta é uma análise pós-seleção: repetir seeds nas mesmas bases não elimina o viés de escolher "
                "as melhores configurações nessas bases.",
                "`runs.csv` guarda as execuções; `dataset_means.csv`, as médias das repetições; "
                "`ranking.csv`, o ranking combinado; `ranks_per_dataset.csv` e `metric_ranks.csv`, os ranks; "
                "`friedman.csv` e `nemenyi*.csv`, os testes. `mean_ranks.png/pdf` mostram os ranks por métrica. "
                "`analysis_protocol.json` registra o método, as opções, os arquivos gerados e hashes dos insumos.",
            ]
        )
        + "\n"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=("repeated", "single"),
        default="repeated",
        help="repeated: average fresh repetitions (default); single: original Random Search top 10",
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=None,
        help="Input directory; defaults to repeated_random_search or random_search_shell for the chosen mode",
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--confidence", type=float, default=0.95)
    args = parser.parse_args()
    if not 0.5 < args.confidence < 1:
        parser.error("Require .5 < confidence < 1")
    os.chdir(ROOT)
    default_input = DEFAULT_OUTPUT if args.mode == "repeated" else "experiments/random_search_shell"
    directory = (args.input or Path(default_input)).resolve()
    output = args.output_dir.resolve() if args.output_dir else directory / "analysis"
    if output == directory:
        parser.error("Analysis output must differ from the experiment root")
    lock_name = ".repetitions.lock" if args.mode == "repeated" else ".random_search.lock"
    with lock(directory, lock_name):
        loader = load_panel if args.mode == "repeated" else load_single_panel
        plan, runs, hashes = loader(directory)
        plan["mode"] = args.mode
        tables = analyze(
            runs,
            plan["config_ids"],
            [d["name"] for d in plan["datasets"]],
            plan["seeds"],
            1 - args.confidence,
        )
        output.mkdir(parents=True, exist_ok=True)
        for name, frame in tables.items():
            frame.to_csv(output / (name + ".csv"), index=False)
        plots(tables, output)
        write_json(
            output / "analysis_protocol.json",
            dict(
                confidence=args.confidence,
                protocol="mease-friedman-nemenyi-v4",
                mode=args.mode,
                input_directory=str(directory),
                config_ids=plan["config_ids"],
                omnibus="Friedman with Iman-Davenport F approximation",
                tested_outcomes=[*METRICS, "mean_metric_rank"],
                composite="equal-weight mean of three within-dataset metric ranks; re-ranked within each dataset for testing",
                block="dataset",
                repetition_aggregation="mean before ranking" if args.mode == "repeated" else "single original run",
                posthoc="Nemenyi after omnibus rejection, alpha per metric",
                additional_multiplicity_correction=None,
                artifacts=[name + ".csv" for name in tables] + ["mean_ranks.png", "mean_ranks.pdf", "relatorio.md"],
                plan_sha256=sha256(directory / "plan.json") if args.mode == "repeated" else None,
                inputs=hashes,
                analysis_sha256=sha256(Path(__file__)),
                python=sys.version,
                numpy=np.__version__,
                pandas=pd.__version__,
                scipy=__import__("scipy").__version__,
                scikit_posthocs=sp.__version__,
            ),
        )
        (output / "relatorio.md").write_text(report(plan, tables, 1 - args.confidence))
    print(f"Report: {output / 'relatorio.md'}")
    print(tables["friedman"][["metric", "p_value", "reject_global"]].to_string(index=False))


if __name__ == "__main__":
    try:
        main()
    except (ValueError, OSError, RuntimeError) as exc:
        sys.exit(f"Error: {exc}")
