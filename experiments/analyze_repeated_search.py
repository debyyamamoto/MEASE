"""Dataset-blocked Friedman/Nemenyi analysis and conditional seed uncertainty."""
from __future__ import annotations

import argparse
import itertools
import json
import os
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.analyze_random_search import METRICS, markdown_table, rank_panel
from experiments.repeat_random_search import (DEFAULT_OUTPUT, command_for, completed, lock,
                                               outputs_for, sha256, validate_metrics, write_json)


def holm(pvalues):
    pvalues = np.asarray(pvalues, dtype=float)
    order = np.argsort(pvalues)
    adjusted = np.empty_like(pvalues)
    adjusted[order] = np.minimum(1, np.maximum.accumulate(pvalues[order] * np.arange(len(order), 0, -1)))
    return adjusted


def friedman(matrix, ascending, permutations, rng):
    """One row per dataset. Within-row label permutations retain the observed ties."""
    matrix = np.round(np.asarray(matrix, dtype=float), 12)
    n, k = matrix.shape
    ranks = stats.rankdata(matrix if ascending else -matrix, axis=1)
    tie_sum = 0
    for row in ranks:
        _, counts = np.unique(row, return_counts=True)
        tie_sum += np.sum(counts**3 - counts)
    correction = 1 - tie_sum / (n * (k**3 - k))
    if correction <= 1e-14:
        return ranks, dict(friedman_chi2=0., p_chi2=1., iman_davenport_f=0., p_iman_davenport=1.,
                           p_permutation=1., permutation_mc_se=0., tie_correction=0.)
    observed = np.sum((ranks.sum(axis=0) - n * (k + 1) / 2)**2)
    q = 12 * observed / (n * k * (k + 1) * correction)
    denom = n * (k - 1) - q
    f = (n - 1) * q / denom if denom > 1e-12 else np.inf
    count = 0
    for start in range(0, permutations, 2048):
        size = min(2048, permutations - start)
        shuffled = rng.permuted(np.broadcast_to(ranks, (size, n, k)), axis=2)
        null = np.sum((shuffled.sum(axis=1) - n * (k + 1) / 2)**2, axis=1)
        count += np.count_nonzero(null >= observed - 1e-10)
    p = (count + 1) / (permutations + 1)
    return ranks, dict(friedman_chi2=float(q), p_chi2=float(stats.chi2.sf(q, k - 1)),
                       iman_davenport_f=float(f), p_iman_davenport=float(stats.f.sf(f, k - 1, (k - 1) * (n - 1))),
                       p_permutation=p, permutation_mc_se=float(np.sqrt(p * (1 - p) / (permutations + 1))),
                       tie_correction=float(correction))


def mean_ci(values, alpha):
    values = np.asarray(values, dtype=float)
    mean = float(values.mean())
    sd = float(values.std(ddof=1))
    se = sd / np.sqrt(len(values))
    width = float(stats.t.ppf(1 - alpha / 2, len(values) - 1) * se)
    return mean, sd, mean - width, mean + width


def load_panel(directory):
    plan = json.loads((directory / "plan.json").read_text())
    if plan.get("protocol") != "mease-repetitions-v1":
        raise ValueError("Unsupported repetition protocol")
    ids, seeds = plan["config_ids"], plan["seeds"]
    datasets = [d["name"] for d in plan["datasets"]]
    expected = set(itertools.product(ids, datasets, seeds))
    if (len(ids) < 3 or len(datasets) < 3 or len(seeds) < 2 or len(seeds) != plan["repetitions"]
            or len(set(ids)) != len(ids) or len(set(datasets)) != len(datasets) or len(set(seeds)) != len(seeds)
            or len(plan["jobs"]) != len(expected) or {tuple(j) for j in plan["jobs"]} != expected):
        raise ValueError("Expected a balanced panel, >=3 configurations/datasets and >=2 distinct seeds")
    for name, digest in plan["hashes"].items():
        if sha256(ROOT / name) != digest:
            raise ValueError(f"Experiment code/data fingerprint differs: {name}")
    rows, missing, hashes = [], [], {}
    for cid, dataset, seed in plan["jobs"]:
        _, folder = command_for(plan, directory, cid, dataset, seed)
        required = [folder / "command.sh", *outputs_for(folder, dataset, plan["baseline"]), folder / "metrics_full.json"]
        if not completed(folder, required):
            missing.append(f"config_{cid}/{dataset}/seed_{seed}")
            continue
        values = json.loads((folder / "metrics_full.json").read_text())
        validate_metrics(values)
        rows.append(dict(config_id=cid, dataset=dataset, seed=seed, **values))
        hashes[str(folder.relative_to(directory) / ".complete.json")] = sha256(folder / ".complete.json")
    if missing:
        raise ValueError(f"Incomplete/invalid panel: {len(missing)} of {len(expected)} runs. "
                         f"Resume repeat_random_search.sh before analyzing. Examples: {', '.join(missing[:5])}")
    return plan, pd.DataFrame(rows), hashes


def analyze(runs, ids, datasets, seeds, alpha=.05, permutations=99999, analysis_seed=2026):
    means = runs.groupby(["config_id", "dataset"], as_index=False)[[*METRICS, "#sg"]].mean()
    # Only suppress machine-precision artifacts when assigning ranks, not reported raw metrics.
    rank_cells = means.copy()
    rank_cells[list(METRICS)] = rank_cells[list(METRICS)].round(12)
    per_dataset, ranking = rank_panel(rank_cells)
    cell_rows, global_rows, rank_rows, pair_rows, seed_pairs = [], [], [], [], []
    n, k = len(datasets), len(ids)
    families = len(METRICS)
    n_pairs = k * (k - 1) // 2
    # Nemenyi already adjusts for all configuration pairs; Bonferroni adds the three outcomes.
    cd = float(stats.studentized_range.ppf(1 - alpha / families, k, np.inf) * np.sqrt(k * (k + 1) / (12 * n)))
    rng = np.random.default_rng(analysis_seed)
    for metric, (ascending, _) in METRICS.items():
        matrix = means.pivot(index="dataset", columns="config_id", values=metric).loc[datasets, ids].to_numpy()
        ranks, test = friedman(matrix, ascending, permutations, rng)
        global_rows.append(dict(metric=metric, datasets=n, configs=k, repetitions=len(seeds),
                                **test, nemenyi_cd=cd))
        avg = ranks.mean(axis=0)
        for cid, value in zip(ids, avg):
            rank_rows.append(dict(metric=metric, config_id=cid, mean_rank=float(value)))
        for (cid, dataset), group in runs.groupby(["config_id", "dataset"]):
            mean, sd, low, high = mean_ci(group[metric], alpha)
            cell_rows.append(dict(metric=metric, config_id=cid, dataset=dataset, r=len(group),
                                  mean=mean, sd=sd, ci_low=low, ci_high=high, zero_variance=sd == 0))
        macro = runs.groupby(["seed", "config_id"])[metric].mean().unstack("config_id").loc[seeds, ids]
        for a, b in itertools.combinations(range(k), 2):
            diff = float(avg[a] - avg[b])  # Negative = a has better rank.
            q = abs(diff) / np.sqrt(k * (k + 1) / (12 * n))
            p = float(stats.studentized_range.sf(q, k, np.inf))
            pair_rows.append(dict(metric=metric, config_a=ids[a], config_b=ids[b],
                                  rank_difference_a_minus_b=diff, rank_ci_low=diff - cd, rank_ci_high=diff + cd,
                                  p_nemenyi=p, p_nemenyi_three_metrics=min(1., p * families),
                                  excludes_zero=abs(diff) > cd))
            # Conditional effect on these fixed datasets, paired by seed, NOT a Demsar test.
            differences = macro[ids[a]] - macro[ids[b]]
            mean, sd, low, high = mean_ci(differences, alpha)
            _, _, sim_low, sim_high = mean_ci(differences, alpha / (families * n_pairs))
            seed_pairs.append(dict(metric=metric, config_a=ids[a], config_b=ids[b],
                                   raw_difference_a_minus_b=mean, paired_sd=sd, ci_low=low, ci_high=high,
                                   simultaneous_ci_low=sim_low, simultaneous_ci_high=sim_high,
                                   zero_variance=sd == 0))
    tests = pd.DataFrame(global_rows)
    tests["p_permutation_holm"] = holm(tests.p_permutation)
    tests["reject_global"] = tests.p_permutation_holm <= alpha
    pairs = pd.DataFrame(pair_rows).merge(tests[["metric", "reject_global"]], on="metric")
    pairs["significant"] = pairs.reject_global & pairs.excludes_zero
    return dict(runs=runs, dataset_means=means, cell_intervals=pd.DataFrame(cell_rows),
                ranking=ranking, ranks_per_dataset=per_dataset, metric_ranks=pd.DataFrame(rank_rows),
                friedman=tests, nemenyi=pairs, paired_seed_intervals=pd.DataFrame(seed_pairs))


def plots(tables, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), layout="constrained")
    for ax, metric in zip(axes, METRICS):
        frame = tables["metric_ranks"].query("metric == @metric").sort_values("mean_rank")
        ax.scatter(frame.mean_rank, np.arange(len(frame)))
        ax.set(yticks=np.arange(len(frame)), yticklabels=[f"config_{c}" for c in frame.config_id],
               xlabel="Rank médio (menor é melhor)", title=metric, xlim=(.5, len(frame) + .5))
        ax.invert_yaxis()
        ax.grid(axis="x", alpha=.25)
        cd = tables["friedman"].set_index("metric").loc[metric, "nemenyi_cd"]
        ax.text(.02, -.16, f"Diferença crítica Nemenyi: {cd:.3f}", transform=ax.transAxes)
    fig.savefig(output / "mean_ranks.png", dpi=160)
    fig.savefig(output / "mean_ranks.pdf")
    plt.close(fig)


def report(plan, tables, alpha, permutations, analysis_seed):
    tests, pairs = tables["friedman"], tables["nemenyi"]
    ranking = tables["ranking"]
    significant = pairs[pairs.significant]
    intervals = tables["cell_intervals"].copy()
    intervals["média [IC]"] = intervals.apply(lambda x: f"{x['mean']:.5f} [{x.ci_low:.5f}; {x.ci_high:.5f}]", axis=1)
    interval_table = intervals.pivot(index=["dataset", "config_id"], columns="metric", values="média [IC]").reset_index()
    comparisons = len(plan["config_ids"]) * (len(plan["config_ids"]) - 1) // 2
    confidence = 100 * (1 - alpha)
    findings = []
    for row in tests.itertuples():
        decision = "rejeita" if row.reject_global else "não rejeita"
        count = int(significant.metric.eq(row.metric).sum())
        findings.append(f"- **{row.metric}**: Friedman por permutação {decision} H₀ após Holm "
                        f"(p ajustado={row.p_permutation_holm:.6g}); {count} pares significativos no pós-teste.")
    return "\n\n".join([
        "# Repetição das configurações do Random Search",
        f"Painel completo: {len(plan['config_ids'])} configurações × {len(plan['datasets'])} datasets × "
        f"{plan['repetitions']} seeds = {len(tables['runs'])} execuções novas. "
        f"Seeds: {plan['seeds']}. Referência: `{plan['baseline']}`. Seleção: `{plan['selection']}`.",
        "## Resultado",
        "\n".join(findings),
        f"O primeiro colocado descritivo é config_{int(ranking.iloc[0].config_id)}. "
        "Esse ranking combina três objetivos com pesos iguais; não é um teste nem demonstra um vencedor estatístico global.",
        markdown_table(ranking[["config_id", "position", "mean_rank", "worst_dataset_rank", "exceptionality", "model_redundancy", "setCov"]]),
        "## Unidade experimental e hipóteses",
        "Cada célula configuração × dataset é a média das seeds. Friedman recebe uma matriz datasets × configurações "
        "por métrica: ranks decrescentes para exceptionality/setCov e crescentes para model redundancy. "
        "Empates recebem rank médio; arredondamento a 12 casas somente na atribuição dos ranks elimina ruído numérico. "
        "H₀ representa ausência de diferenças sistemáticas de desempenho/rank entre configurações nos datasets. "
        "Os datasets são os blocos independentes: seis bases continuam sendo N=6, mesmo com r=30. "
        "As repetições estimam a aleatoriedade do MEASE; não reamostram pacientes nem produzem novos datasets.",
        "Datasets precisam ser independentes e representar o domínio da conclusão. Parentesco ou sobreposição entre bases "
        "deve ser verificado pelo pesquisador. Este experimento não comprova essa premissa.",
        "## Friedman, multiplicidade e pós-teste",
        f"Estatística de Friedman corrigida por empates. P principal: {permutations:,} permutações Monte Carlo "
        f"independentes dos rótulos dentro de cada dataset, seed de análise {analysis_seed}; p=(b+1)/(B+1). "
        "A permutação pressupõe permutabilidade dos rótulos sob H₀. O CSV inclui o erro padrão Monte Carlo; "
        "resultados junto do limiar exigem mais permutações. Também são apresentados χ² assintótico e correção "
        "Iman–Davenport como referências, sem escolher o p mais favorável.",
        f"Holm controla os três testes globais em α={alpha:g}. Nemenyi compara todos os {comparisons} pares por métrica; "
        "a distribuição da amplitude studentizada já ajusta esses pares. Bonferroni adicional entre as três métricas "
        f"usa α/3={alpha/3:.6g}. Um par só é declarado significativo quando seu teste global passa Holm e a diferença "
        "de ranks supera a diferença crítica. Os ICs simultâneos de diferenças de ranks usam a mesma diferença crítica "
        "(sem correção adicional de empates, abordagem conservadora). Nemenyi continua sendo uma aproximação; "
        "a permutação do teste global não torna o pós-teste exato. Poucos datasets e muitos candidatos reduzem o poder.",
        markdown_table(tests),
        "### Pares significativos entre datasets",
        markdown_table(significant[["metric", "config_a", "config_b", "rank_difference_a_minus_b", "rank_ci_low", "rank_ci_high", "p_nemenyi_three_metrics"]])
        if len(significant) else "Nenhum par passou o critério conjunto. Isso não demonstra equivalência nem ausência de efeito.",
        "Todos os pares estão em `nemenyi.csv`. Diferença de ranks = rank(A) − rank(B); negativa favorece A. "
        "Um resultado não significativo não demonstra equivalência entre configurações.",
        "## Intervalos de confiança e tamanho das diferenças",
        f"`cell_intervals.csv`: ICs t pontuais de {confidence:g}% para a média de cada configuração/dataset sobre "
        "as seeds (média ± t × s/√r). `paired_seed_intervals.csv`: em cada seed calcula-se a média com pesos iguais "
        "entre os datasets fixos e então A − B; o IC t usa essas diferenças pareadas entre seeds. "
        f"Há ICs pontuais de {confidence:g}% e simultâneos via Bonferroni sobre {3*comparisons} contrastes "
        "(pares × métricas). Isso mede incerteza da aleatoriedade do MEASE neste painel, não generalização a novos datasets "
        "e não substitui Friedman/Nemenyi. Para diferenças brutas, positivo favorece A em exceptionality/setCov e "
        "negativo favorece A em model redundancy.",
        "IC t assume repetições independentes e uma aproximação adequada para a distribuição da média; r=30 não garante "
        "normalidade. Métricas limitadas, discretas e próximas dos extremos exigem cautela. Os limites não são truncados; "
        "podem ultrapassar [0,1]. Variância observada zero produz IC degenerado, sinalizado nos CSVs, e não prova ausência "
        "de incerteza. A sobreposição dos ICs de duas médias não decide significância: examine o IC da diferença e "
        "a família de comparações pertinente. Os controles de multiplicidade são separados para inferência entre "
        "datasets e para os ICs condicionais às seeds.",
        markdown_table(interval_table),
        "## Procedência e limites da interpretação",
        "As configurações foram congeladas antes das novas execuções; a seed da busca foi excluída. "
        "Parâmetros originais, hashes de código/dados, seeds e ordem aleatória estão em `plan.json`; "
        "`selected_configs.csv` registra a seleção. Cada execução chama main.py com executions=1 e uma seed explícita, "
        "pois executions>1 com seed explícita reutiliza essa seed no runner atual. PYTHONHASHSEED=0 é fixado para as novas "
        "execuções. Versões de dependências vêm de uv.lock; diferenças de hardware ainda podem afetar resultados.",
        "As métricas são recalculadas sem arredondamento a partir de DetailedRules.csv usando a implementação original "
        "e conferidas contra RulesMetricsResult.csv (que arredonda para duas casas). O ranking de seleção mantém os CSVs "
        "originais da busca. Arquivos ausentes, resultados sem marcador válido ou painel incompleto impedem a análise. "
        "Não há imputação nem descarte silencioso de seeds/datasets.",
        "Selecionar as melhores configurações e avaliá-las novamente nas mesmas bases ainda é uma análise pós-seleção. "
        "Seeds novas reduzem a dependência da sorte da execução original, mas não eliminam o viés de escolha das bases/configurações. "
        "Conclusões confirmatórias requerem datasets de validação independentes ou um desenho de seleção aninhado. "
        "Não há holdout de registros neste protocolo: os resultados descrevem as regras mineradas nos próprios dados.",
        "A implementação do MEASE é preservada, incluindo limitações registradas no diagnóstico da config9: "
        "min_rule_support/max_rule_support não são encaminhados ao RuleEvaluator e o ajuste final de intervalos "
        "pode alterar a cobertura. Repetição não corrige esses problemas; corrigir o algoritmo requer uma nova versão "
        "do experimento. Model redundancy mantém seu significado atual (p>0,05 entre partes exclusivas, incluindo "
        "fallback para grupos vazios); não é uma distância geométrica entre curvas. Exceptionality é uma proporção "
        "de testes significativos e setCov cobre os registros observados.",
        "## Artefatos e referências",
        "`runs.csv`, `dataset_means.csv`, `cell_intervals.csv`, `ranking.csv`, `ranks_per_dataset.csv`, "
        "`metric_ranks.csv`, `friedman.csv`, `nemenyi.csv`, `paired_seed_intervals.csv`; "
        "`mean_ranks.png/pdf` mostram ranks descritivos. `analysis_protocol.json` registra opções e hashes dos insumos.",
        "[Demšar (2006), Statistical Comparisons of Classifiers over Multiple Data Sets, JMLR 7:1–30](https://jmlr.org/papers/v7/demsar06a.html). "
        "O uso de Friedman/Nemenyi e datasets como blocos segue esse trabalho; permutação, controle entre três métricas "
        "e ICs condicionais às seeds são escolhas adicionais deste protocolo. "
        "[SciPy: Friedman e limites da aproximação χ²](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.friedmanchisquare.html); "
        "[amplitude studentizada](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.studentized_range.html).",
    ]) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path(DEFAULT_OUTPUT))
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--confidence", type=float, default=.95)
    parser.add_argument("--permutations", type=int, default=99999)
    parser.add_argument("--analysis-seed", type=int, default=2026)
    args = parser.parse_args()
    if not .5 < args.confidence < 1 or args.permutations < 999 or args.analysis_seed < 0:
        parser.error("Require .5 < confidence < 1, permutations >= 999 and analysis-seed >= 0")
    os.chdir(ROOT)
    directory = args.input.resolve()
    output = args.output_dir.resolve() if args.output_dir else directory / "analysis"
    if output == directory:
        parser.error("Analysis output must differ from the experiment root")
    with lock(directory):
        plan, runs, hashes = load_panel(directory)
        tables = analyze(runs, plan["config_ids"], [d["name"] for d in plan["datasets"]], plan["seeds"],
                         1 - args.confidence, args.permutations, args.analysis_seed)
        output.mkdir(parents=True, exist_ok=True)
        for name, frame in tables.items():
            frame.to_csv(output / (name + ".csv"), index=False)
        plots(tables, output)
        write_json(output / "analysis_protocol.json", dict(
            confidence=args.confidence, permutations=args.permutations, analysis_seed=args.analysis_seed,
            plan_sha256=sha256(directory / "plan.json"), inputs=hashes,
            analysis_sha256=sha256(Path(__file__)), python=sys.version,
            numpy=np.__version__, pandas=pd.__version__, scipy=__import__("scipy").__version__))
        (output / "relatorio.md").write_text(report(plan, tables, 1 - args.confidence, args.permutations, args.analysis_seed))
    print(f"Report: {output / 'relatorio.md'}")
    print(tables["friedman"][["metric", "p_permutation_holm", "reject_global"]].to_string(index=False))


if __name__ == "__main__":
    try:
        main()
    except (ValueError, OSError, RuntimeError) as exc:
        sys.exit(f"Error: {exc}")
