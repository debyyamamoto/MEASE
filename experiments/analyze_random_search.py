"""Audit Random Search provenance and rank its saved metrics (no new MEASE runs).

Usage: uv run python experiments/analyze_random_search.py
The panel ranking is exploratory: Info.csv records only part of the parameters.
"""

from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path

import numpy as np
import pandas as pd

METRICS = {
    "exceptionality": (False, "rank_exceptionality"),
    "model redundancy": (True, "rank_redundancy"),
    "setCov": (False, "rank_coverage"),
}
RECORDED = ["score_metric", "redundancy_penalty", "redundancy_similarity_threshold"]
CONTEXT = ["archive_selection", "km_time_bins", "rate_policy"]


def parse_config(text: str) -> dict:
    """Read old NumPy scalar reprs without executing arbitrary Python."""

    class Scalars(ast.NodeTransformer):
        def visit_Call(self, node):
            if (
                isinstance(node.func, ast.Attribute)
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id == "np"
                and node.func.attr in {"str_", "float64", "int64"}
                and len(node.args) == 1
                and not node.keywords
            ):
                return self.visit(node.args[0])
            raise ValueError("Unexpected expression in configs.txt")

    value = ast.literal_eval(Scalars().visit(ast.parse(text, mode="eval")))
    if not isinstance(value, dict):
        raise ValueError("Expected a configuration dictionary")
    return value


def read_results(root: Path, baseline: str):
    configs = [parse_config(line) for line in (root / "configs.txt").read_text().splitlines()]
    rows = []
    for path in sorted(root.glob(f"config_*/*/{baseline}/*_RulesMetricsResult.csv")):
        cid = int(path.parents[2].name.removeprefix("config_"))
        dataset = path.parents[1].name
        run = int(path.name.removeprefix(dataset + "_").split("_")[0])
        metrics = pd.read_csv(path)
        info = pd.read_csv(path.with_name(path.name.replace("RulesMetricsResult", "Info")))
        if len(metrics) != 1 or len(info) != 1:
            raise ValueError(f"Expected one row: {path}")
        conf = configs[cid]
        recorded = info.iloc[0].to_dict()
        missing = [key for key in RECORDED + CONTEXT if key not in recorded or pd.isna(recorded[key])]
        mismatches = [key for key in RECORDED if key not in missing and recorded[key] != conf[key]]
        rows.append(
            {
                "config_id": cid,
                "dataset": dataset,
                "run": run,
                **metrics.iloc[0].to_dict(),
                **{key: recorded.get(key) for key in RECORDED + CONTEXT},
                "total_time": recorded.get("total_time"),
                "manifest_mismatches": ",".join(mismatches),
                "missing_metadata": ",".join(missing),
                "metadata_matches": not mismatches and not missing,
                "source": str(path),
            }
        )
    frame = pd.DataFrame(rows)
    if frame.empty:
        raise ValueError("No result files found")
    if frame.duplicated(["config_id", "dataset", "run"]).any():
        raise ValueError("Duplicate runs")
    values = frame[list(METRICS)].to_numpy(dtype=float)
    if not np.isfinite(values).all() or ((values < 0) | (values > 1)).any():
        raise ValueError("Metrics must be finite and in [0, 1]")
    return configs, frame


def rank_panel(cells: pd.DataFrame, weights=(1, 1, 1)):
    ranked = cells.copy()
    weights = np.asarray(weights, dtype=float)
    weights /= weights.sum()
    for metric, (ascending, column) in METRICS.items():
        ranked[column] = ranked.groupby("dataset")[metric].rank(method="average", ascending=ascending)
    rank_cols = [column for _, column in METRICS.values()]
    ranked["mean_metric_rank"] = ranked[rank_cols].to_numpy() @ weights
    ranked["dataset_position"] = ranked.groupby("dataset").mean_metric_rank.rank(method="min")
    summary = ranked.groupby("config_id").agg(
        mean_rank=("mean_metric_rank", "mean"),
        worst_dataset_rank=("mean_metric_rank", "max"),
        exceptionality=("exceptionality", "mean"),
        model_redundancy=("model redundancy", "mean"),
        setCov=("setCov", "mean"),
        mean_rules=("#sg", "mean"),
        min_exceptionality=("exceptionality", "min"),
        max_redundancy=("model redundancy", "max"),
        min_setCov=("setCov", "min"),
        datasets=("dataset", "nunique"),
    )
    summary["position"] = summary.mean_rank.rank(method="min").astype(int)
    summary = summary.reset_index().sort_values(["mean_rank", "config_id"])
    # Pareto status on the three macro averages, not on individual datasets.
    objectives = summary[["exceptionality", "model_redundancy", "setCov"]].to_numpy() * [1, -1, 1]
    summary["pareto_macro"] = [
        not (np.all(objectives >= row, axis=1) & np.any(objectives > row, axis=1)).any() for row in objectives
    ]
    return ranked, summary


def markdown_table(frame: pd.DataFrame) -> str:
    def fmt(value):
        return f"{value:.4f}" if isinstance(value, float) else str(value)

    lines = ["| " + " | ".join(frame.columns) + " |", "| " + " | ".join(["---"] * len(frame.columns)) + " |"]
    lines.extend("| " + " | ".join(map(fmt, row)) + " |" for row in frame.itertuples(index=False, name=None))
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("experiments/random_search_shell"))
    parser.add_argument("--output", type=Path, default=Path("experiments/results_random_search_analysis"))
    parser.add_argument("--baseline", choices=["complement", "population"], default="complement")
    args = parser.parse_args()
    configs, runs = read_results(args.input, args.baseline)
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    runs.to_csv(out / "audit_runs.csv", index=False)
    audit = runs.groupby("dataset").agg(
        runs=("config_id", "size"), configs=("config_id", "nunique"), matching_runs=("metadata_matches", "sum")
    )
    audit["incompatible_runs"] = audit.runs - audit.matching_runs
    audit.to_csv(out / "audit_datasets.csv")
    # Do not average repeated runs whose recorded settings differ.
    inconsistent = runs.groupby(["config_id", "dataset"])[RECORDED + CONTEXT].nunique().gt(1).any(axis=1)
    if inconsistent.any():
        raise ValueError("Different recorded parameters among repetitions of the same cell; see audit_runs.csv")
    cells = (
        runs.groupby(["config_id", "dataset"], as_index=False)
        .agg(
            {
                **{key: "mean" for key in list(METRICS) + ["#sg"]},
                "run": "size",
                "metadata_matches": "all",
            }
        )
        .rename(columns={"run": "repetitions"})
    )
    local, _ = rank_panel(cells)
    local.sort_values(["dataset", "dataset_position", "config_id"]).to_csv(
        out / "ranking_per_dataset.csv", index=False
    )
    # Only retain datasets whose entire manifest panel passes available checks.
    # Passing three recorded parameters does NOT verify the other ten parameters.
    eligible = audit.index[(audit.incompatible_runs == 0) & (audit.configs == len(configs))].tolist()
    context_ok = runs[runs.dataset.isin(eligible)][CONTEXT].nunique().le(1).all()
    if not context_ok:
        eligible = []
    varying = runs.groupby("config_id")[RECORDED + CONTEXT].nunique().gt(1).any(axis=1).sum()
    report = [
        "# Análise exploratória do Random Search do MEASE",
        f"Resultados: {len(runs)} execuções, {runs.config_id.nunique()} IDs, {runs.dataset.nunique()} datasets; referência `{args.baseline}`.",
        "## Auditoria antes do ranking",
        markdown_table(audit.reset_index()),
        f"{varying} IDs têm parâmetros registrados diferentes entre datasets. Não se deve interpretá-los como configurações globais únicas.",
        "Na versão de random_search.py lida no início desta análise, o sorteio e a sobrescrita de configs.txt ocorriam antes de pular pastas já existentes. Isso permite misturar buscas diferentes sob o mesmo ID. A divergência é observada nos arquivos; esse mecanismo é uma explicação provável, não uma reconstrução do histórico. O gerador foi alterado externamente durante a análise; esta observação descreve a versão inicialmente inspecionada.",
        "Info.csv permite confrontar apenas score_metric, redundancy_penalty e redundancy_similarity_threshold com o manifesto. Os outros dez parâmetros sorteados não foram registrados ali. Coincidência nesses três campos não comprova a configuração completa; inclusive os poucos IDs sem divergência podem coincidir por acaso.",
        "## Métricas e critério",
        "No código atual, exceptionality é a proporção de regras com p ≤ 0,05 no teste contra o grupo de referência. Não mede tamanho de efeito. O alpha sorteado é peso do fitness, separado desse nível de significância. Model redundancy é a proporção de pares com p > 0,05, comparando as partes exclusivas dos subgrupos; ausência de significância não prova equivalência dos modelos. setCov = |união das coberturas das regras| / |dataset|: cobre os registros observados, não todo o espaço possível de atributos.",
        "1. Calcular médias das repetições por configuração e dataset, se existirem e forem consistentes.\n2. Dentro de cada dataset, ordenar exceptionality e setCov em ordem decrescente, model redundancy em crescente; empates recebem a posição média.\n3. Tirar a média dos três ranks (pesos iguais, escolha explícita).\n4. Tirar a média entre datasets, dando o mesmo peso a cada base. Menor mean_rank é melhor. Empates no ranking final permanecem empatados; o ID apenas ordena a apresentação.\n5. Inspecionar também o pior rank entre datasets, os valores originais e a sensibilidade aos pesos.",
        "Fórmula: R(c) = (1/D) Σ_d [r_exceptionality(c,d) + r_redundancy(c,d) + r_setCov(c,d)] / 3. Não é teste de significância nem prova de ótimo global. Ranks descartam magnitude das diferenças; raw metrics e uma alternativa de utilidade são apresentados para avaliar essa limitação.",
    ]
    if len(eligible) >= 2:
        panel, summary = rank_panel(cells[cells.dataset.isin(eligible)])
        panel.to_csv(out / "panel_per_dataset.csv", index=False)
        manifest = pd.DataFrame(configs).rename_axis("config_id").reset_index()
        summary.merge(manifest, on="config_id", validate="one_to_one").to_csv(
            out / "ranking_panel_provisional.csv", index=False
        )
        alternatives = []
        for name, weights in [
            ("equal", (1, 1, 1)),
            ("exceptionality_50pct", (2, 1, 1)),
            ("redundancy_50pct", (1, 2, 1)),
            ("coverage_50pct", (1, 1, 2)),
        ]:
            _, ranked = rank_panel(cells[cells.dataset.isin(eligible)], weights)
            alternatives.append(ranked[["config_id", "position", "mean_rank"]].assign(scenario=name))
        pd.concat(alternatives).to_csv(out / "weight_sensitivity.csv", index=False)
        utility = summary.copy()
        utility["utility"] = (utility.exceptionality + 1 - utility.model_redundancy + utility.setCov) / 3
        utility = utility.rename(columns={"position": "rank_method_position"})
        utility["utility_position"] = utility.utility.rank(ascending=False, method="min").astype(int)
        utility.sort_values("utility", ascending=False).to_csv(out / "alternative_raw_utility.csv", index=False)
        winner = int(summary.iloc[0].config_id)
        (out / "candidate_manifest_parameters.json").write_text(
            json.dumps(
                {
                    "config_id": winner,
                    "status": "provisional; only three manifest parameters audited",
                    "datasets": eligible,
                    "parameters_from_current_manifest": configs[winner],
                },
                indent=2,
            )
            + "\n"
        )
        report.extend(
            [
                "## Ranking provisório do painel sem divergências detectadas",
                f"Datasets: {', '.join(eligible)}. {len(summary)} candidatos. Não representa validação nas seis bases. Parâmetros completos condicionados à confirmação da procedência do manifesto.",
                markdown_table(
                    summary[
                        [
                            "position",
                            "config_id",
                            "mean_rank",
                            "worst_dataset_rank",
                            "exceptionality",
                            "model_redundancy",
                            "setCov",
                            "mean_rules",
                        ]
                    ].head(10)
                ),
                f"Candidato pelo critério de ranks: config_{winner}. Resultados individuais:",
                markdown_table(
                    panel.loc[
                        panel.config_id.eq(winner),
                        ["dataset", "exceptionality", "model redundancy", "setCov", "#sg", "mean_metric_rank"],
                    ]
                ),
                "Parâmetros nominais no manifesto atual (não integralmente verificados):\n\n```json\n"
                + json.dumps(configs[winner], indent=2)
                + "\n```",
                "Vencedores sob pesos alternativos:\n\n"
                + markdown_table(
                    pd.concat(alternatives).query("position == 1")[["scenario", "config_id", "mean_rank"]]
                ),
                "O arquivo alternative_raw_utility.csv usa U = (exceptionality + 1 − model redundancy + setCov)/3; como todas estão em [0,1], é uma alternativa com trocas absolutas explicitamente iguais. Sua ordenação pode diferir da média de ranks. pareto_macro indica não dominância nas três médias entre datasets; não implica superioridade em cada dataset e não foi usado como filtro oculto.",
            ]
        )
        print(
            summary[["position", "config_id", "mean_rank", "exceptionality", "model_redundancy", "setCov"]]
            .head(10)
            .to_string(index=False)
        )
    else:
        report.append(
            "Não há pelo menos dois datasets completos sem divergências detectadas para gerar um painel provisório."
        )
    report.extend(
        [
            "## Limitações e próximos experimentos",
            f"Repetições por célula observadas: {sorted(cells.repetitions.unique().tolist())}. Com uma execução por célula, o ranking não estima variabilidade entre sementes. O gerador não passa seed e o runner atual usa o índice da execução (0 para uma execução); isso deve ser confirmado para o código usado na busca.",
            "As métricas foram salvas arredondadas a duas casas pelo runner, mesmo aparecendo com quatro casas no CSV; os empates devem ser preservados. Reter maior precisão nas próximas execuções. #sg/ksize varia e afeta a cobertura e a granularidade das proporções; para comparações com baselines, definir orçamento de regras e de busca comparável, ou explicitar essa variação no protocolo.",
            "Recuperar o manifesto original e ligar cada resultado a todos os parâmetros; se não for possível, executar novamente as combinações necessárias em uma pasta nova. Não é possível reconstruir parâmetros ausentes apenas pelas métricas. Salvar um manifesto imutável, configuração completa por execução, seed e versão do código; ao retomar, carregar esse manifesto e conferir arquivos concluídos em vez de apenas a existência da pasta.",
            "Após corrigir a procedência, selecionar uma lista curta (por exemplo, cinco candidatas) e repetir cada uma com sementes distintas e o mesmo orçamento nas bases. O runner atual repete a mesma seed se --seed e --executions > 1 forem usados juntos: usar processos separados com sementes explícitas distintas, ou corrigir a política antes de confirmar resultados. Mais sementes avaliam a variabilidade do algoritmo, mas não removem o viés de selecionar e avaliar na mesma coleção de dados.",
            "Para sustentar generalização entre datasets, selecionar parâmetros em bases de desenvolvimento e avaliar em bases reservadas; uma avaliação externa deixando um dataset de fora por vez estima o procedimento de seleção. Se a afirmação for generalização a novos indivíduos, separar também descoberta e avaliação das regras em dados distintos. Após seleção, congelar a configuração antes da comparação confirmatória com outros algoritmos.",
            "Variar um fator por vez em torno da configuração escolhida mede sensibilidade local. Não identifica todos os fatores mais importantes globalmente e pode perder interações (por exemplo, population × generations ou ksize × threshold). Complementar com desenho fatorial para os fatores prioritários. Não é necessário provar um ótimo global para começar uma análise de sensibilidade; é necessário um ponto de referência justificado e reproduzível.",
            "## Reprodução e arquivos",
            "```bash\nuv run python experiments/analyze_random_search.py\n```",
            "audit_runs.csv contém fontes e divergências; audit_datasets.csv resume a auditoria; ranking_per_dataset.csv ordena os resultados de cada base (IDs locais não garantem uma configuração global); ranking_panel_provisional.csv contém o painel elegível e parâmetros nominais; panel_per_dataset.csv mostra todos os componentes do score; weight_sensitivity.csv e alternative_raw_utility.csv mostram escolhas alternativas. Os resultados originais não são modificados.",
            "## Referências metodológicas",
            "A agregação por ranks entre datasets se inspira em [Demšar (2006)](https://www.jmlr.org/papers/v7/demsar06a.html); a combinação dos três objetivos e seus pesos é uma decisão desta análise. Sobre separar seleção e avaliação, ver [Cawley e Talbot (2010)](https://www.jmlr.org/papers/v11/cawley10a.html).",
        ]
    )
    (out / "relatorio.md").write_text("\n\n".join(report) + "\n")
    print(f"\nAudit and report: {out}")


if __name__ == "__main__":
    main()
