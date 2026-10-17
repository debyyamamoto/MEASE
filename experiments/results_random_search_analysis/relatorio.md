# Análise exploratória do Random Search do MEASE

Resultados: 1800 execuções, 300 IDs, 6 datasets; referência `complement`.

## Auditoria antes do ranking

| dataset | runs | configs | matching_runs | incompatible_runs |
| --- | --- | --- | --- | --- |
| breast-cancer | 300 | 300 | 300 | 0 |
| cancer | 300 | 300 | 300 | 0 |
| carcinoma | 300 | 300 | 300 | 0 |
| lung | 300 | 300 | 300 | 0 |
| mgus2 | 300 | 300 | 300 | 0 |
| veteran | 300 | 300 | 300 | 0 |

0 IDs têm parâmetros registrados diferentes entre datasets. Não se deve interpretá-los como configurações globais únicas.

Na versão de random_search.py lida no início desta análise, o sorteio e a sobrescrita de configs.txt ocorriam antes de pular pastas já existentes. Isso permite misturar buscas diferentes sob o mesmo ID. A divergência é observada nos arquivos; esse mecanismo é uma explicação provável, não uma reconstrução do histórico. O gerador foi alterado externamente durante a análise; esta observação descreve a versão inicialmente inspecionada.

Info.csv permite confrontar apenas score_metric, redundancy_penalty e redundancy_similarity_threshold com o manifesto. Os outros dez parâmetros sorteados não foram registrados ali. Coincidência nesses três campos não comprova a configuração completa; inclusive os poucos IDs sem divergência podem coincidir por acaso.

## Métricas e critério

No código atual, exceptionality é a proporção de regras com p ≤ 0,05 no teste contra o grupo de referência. Não mede tamanho de efeito. O alpha sorteado é peso do fitness, separado desse nível de significância. Model redundancy é a proporção de pares com p > 0,05, comparando as partes exclusivas dos subgrupos; ausência de significância não prova equivalência dos modelos. setCov = |união das coberturas das regras| / |dataset|: cobre os registros observados, não todo o espaço possível de atributos.

1. Calcular médias das repetições por configuração e dataset, se existirem e forem consistentes.
2. Dentro de cada dataset, ordenar exceptionality e setCov em ordem decrescente, model redundancy em crescente; empates recebem a posição média.
3. Tirar a média dos três ranks (pesos iguais, escolha explícita).
4. Tirar a média entre datasets, dando o mesmo peso a cada base. Menor mean_rank é melhor. Empates no ranking final permanecem empatados; o ID apenas ordena a apresentação.
5. Inspecionar também o pior rank entre datasets, os valores originais e a sensibilidade aos pesos.

Fórmula: R(c) = (1/D) Σ_d [r_exceptionality(c,d) + r_redundancy(c,d) + r_setCov(c,d)] / 3. Não é teste de significância nem prova de ótimo global. Ranks descartam magnitude das diferenças; raw metrics e uma alternativa de utilidade são apresentados para avaliar essa limitação.

## Ranking provisório do painel sem divergências detectadas

Datasets: breast-cancer, cancer, carcinoma, lung, mgus2, veteran. 300 candidatos. Não representa validação nas seis bases. Parâmetros completos condicionados à confirmação da procedência do manifesto.

| position | config_id | mean_rank | worst_dataset_rank | exceptionality | model_redundancy | setCov | mean_rules |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 55 | 85.1667 | 134.6667 | 0.9333 | 0.2167 | 0.9933 | 5.0000 |
| 2 | 172 | 90.1944 | 131.0000 | 0.9833 | 0.2850 | 0.9950 | 10.0000 |
| 3 | 150 | 91.5833 | 124.8333 | 0.9283 | 0.2783 | 0.9967 | 7.0000 |
| 4 | 9 | 91.9167 | 111.0000 | 0.9667 | 0.2000 | 0.9817 | 5.0000 |
| 5 | 43 | 95.0278 | 118.3333 | 0.9767 | 0.2700 | 0.9717 | 7.0000 |
| 6 | 103 | 95.6944 | 144.8333 | 0.9000 | 0.2167 | 0.9850 | 5.0000 |
| 7 | 79 | 95.7778 | 148.8333 | 0.9517 | 0.2850 | 0.9950 | 7.0000 |
| 8 | 234 | 96.0000 | 142.6667 | 0.8567 | 0.2400 | 0.9917 | 7.0000 |
| 9 | 21 | 97.1667 | 147.6667 | 1.0000 | 0.2833 | 0.9617 | 5.0000 |
| 10 | 182 | 98.2778 | 138.5000 | 0.9767 | 0.2383 | 0.9417 | 7.0000 |

Candidato pelo critério de ranks: config_55. Resultados individuais:

| dataset | exceptionality | model redundancy | setCov | #sg | mean_metric_rank |
| --- | --- | --- | --- | --- | --- |
| breast-cancer | 1.0000 | 0.3000 | 0.9800 | 5.0000 | 55.5000 |
| cancer | 0.6000 | 0.4000 | 1.0000 | 5.0000 | 111.1667 |
| carcinoma | 1.0000 | 0.2000 | 0.9900 | 5.0000 | 79.5000 |
| lung | 1.0000 | 0.0000 | 1.0000 | 5.0000 | 69.1667 |
| mgus2 | 1.0000 | 0.0000 | 0.9900 | 5.0000 | 61.0000 |
| veteran | 1.0000 | 0.4000 | 1.0000 | 5.0000 | 134.6667 |

Parâmetros nominais no manifesto atual (não integralmente verificados):

```json
{
  "score_metric": "fast_logrank",
  "threshold": 0.6,
  "generations": 500,
  "population": 500,
  "restart_gen": 10,
  "restart_pop": 10,
  "restart_pct": 0.3,
  "alpha": 0.7,
  "redundancy_penalty": 0.5,
  "redundancy_similarity_threshold": 0.9,
  "ksize": 5,
  "min_rule_support": 0.1,
  "max_rule_support": 0.55
}
```

Vencedores sob pesos alternativos:

| scenario | config_id | mean_rank |
| --- | --- | --- |
| equal | 55 | 85.1667 |
| exceptionality_50pct | 172 | 92.5417 |
| redundancy_50pct | 101 | 79.2083 |
| coverage_50pct | 55 | 79.4792 |

O arquivo alternative_raw_utility.csv usa U = (exceptionality + 1 − model redundancy + setCov)/3; como todas estão em [0,1], é uma alternativa com trocas absolutas explicitamente iguais. Sua ordenação pode diferir da média de ranks. pareto_macro indica não dominância nas três médias entre datasets; não implica superioridade em cada dataset e não foi usado como filtro oculto.

## Limitações e próximos experimentos

Repetições por célula observadas: [1]. Com uma execução por célula, o ranking não estima variabilidade entre sementes. O gerador não passa seed e o runner atual usa o índice da execução (0 para uma execução); isso deve ser confirmado para o código usado na busca.

As métricas foram salvas arredondadas a duas casas pelo runner, mesmo aparecendo com quatro casas no CSV; os empates devem ser preservados. Reter maior precisão nas próximas execuções. #sg/ksize varia e afeta a cobertura e a granularidade das proporções; para comparações com baselines, definir orçamento de regras e de busca comparável, ou explicitar essa variação no protocolo.

Recuperar o manifesto original e ligar cada resultado a todos os parâmetros; se não for possível, executar novamente as combinações necessárias em uma pasta nova. Não é possível reconstruir parâmetros ausentes apenas pelas métricas. Salvar um manifesto imutável, configuração completa por execução, seed e versão do código; ao retomar, carregar esse manifesto e conferir arquivos concluídos em vez de apenas a existência da pasta.

Após corrigir a procedência, selecionar uma lista curta (por exemplo, cinco candidatas) e repetir cada uma com sementes distintas e o mesmo orçamento nas bases. O runner atual repete a mesma seed se --seed e --executions > 1 forem usados juntos: usar processos separados com sementes explícitas distintas, ou corrigir a política antes de confirmar resultados. Mais sementes avaliam a variabilidade do algoritmo, mas não removem o viés de selecionar e avaliar na mesma coleção de dados.

Para sustentar generalização entre datasets, selecionar parâmetros em bases de desenvolvimento e avaliar em bases reservadas; uma avaliação externa deixando um dataset de fora por vez estima o procedimento de seleção. Se a afirmação for generalização a novos indivíduos, separar também descoberta e avaliação das regras em dados distintos. Após seleção, congelar a configuração antes da comparação confirmatória com outros algoritmos.

Variar um fator por vez em torno da configuração escolhida mede sensibilidade local. Não identifica todos os fatores mais importantes globalmente e pode perder interações (por exemplo, population × generations ou ksize × threshold). Complementar com desenho fatorial para os fatores prioritários. Não é necessário provar um ótimo global para começar uma análise de sensibilidade; é necessário um ponto de referência justificado e reproduzível.

## Reprodução e arquivos

```bash
uv run python experiments/analyze_random_search.py
```

audit_runs.csv contém fontes e divergências; audit_datasets.csv resume a auditoria; ranking_per_dataset.csv ordena os resultados de cada base (IDs locais não garantem uma configuração global); ranking_panel_provisional.csv contém o painel elegível e parâmetros nominais; panel_per_dataset.csv mostra todos os componentes do score; weight_sensitivity.csv e alternative_raw_utility.csv mostram escolhas alternativas. Os resultados originais não são modificados.

## Referências metodológicas

A agregação por ranks entre datasets se inspira em [Demšar (2006)](https://www.jmlr.org/papers/v7/demsar06a.html); a combinação dos três objetivos e seus pesos é uma decisão desta análise. Sobre separar seleção e avaliação, ver [Cawley e Talbot (2010)](https://www.jmlr.org/papers/v11/cawley10a.html).
