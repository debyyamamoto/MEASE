A análise de sensibilidade com `response=all` usa somente a ANOVA de um fator.
A seleção original foi preservada: as três últimas métricas da coluna `metric`
do CSV agregado são excluídas (`metric[:-3]`). As médias e somas de quadrados
são calculadas a partir das execuções individuais, sem usar médias arredondadas.

O teste compara as médias dos níveis do fator, com:

```text
SSA = sum(n_j * (mean_j - mean)**2)
SSE = sum((y_ij - mean_j)**2)
df_factor = k - 1
df_error = N - k
F = (SSA / df_factor) / (SSE / df_error)
p_value = scipy.stats.f.sf(F, df_factor, df_error)
significant = p_value < alpha
Explained Var = SSA / (SSA + SSE)  # eta²
```

Os tamanhos efetivos dos grupos são usados, inclusive quando há quantidades
diferentes de repetições. Com variância residual zero e médias diferentes,
`F=+inf` e `p=0`; quando todas as observações são iguais, o teste é indefinido
(`NaN`) e a métrica não é selecionada.

A saída de cada fator é salva na subpasta `sensitivity_analysis` de `--output_dir`
(por padrão, `experiments/factorial_design/factorial_analysis/sensitivity_analysis`)
e fica limitada a:

- `<experiment>_all_anova.csv`: uma linha por métrica incluída, com fator,
  graus de liberdade, F calculado, F crítico, p-valor, nível de significância,
  decisão `significant`, variância explicada (`Explained Var`, η² como proporção
  de 0 a 1) e transformação da resposta. Quando não há variância total, η² é `NaN`.
- `<experiment>_all_significant_impact.png`: médias por nível, apenas para
  métricas com `global_info[metric]["significant"] == True`. Não é gerado
  gráfico quando nenhuma métrica é significativa ou quando o fator é categórico.

Fatores com níveis não numéricos, como `score_metric`, são tratados como
categóricos. Para eles, a saída contém somente a tabela da ANOVA, indicando
em quais métricas o fator tem efeito significativo e a variância explicada em
cada métrica. Gráficos anteriores desse fator são removidos ao executar novamente.

O terminal exibe a mesma tabela de decisões da ANOVA, com η² em porcentagem.
Se houver transformação da resposta, η² se refere à variância nessa escala.
Não são exportadas
cópias das respostas individuais nem tabelas intermediárias. Não há testes
Shapiro, Levene, correção de Holm ou intervalos baseados em t nesse fluxo.
O teste é global: a decisão não identifica diferenças entre pares de níveis.

Para executar, indicando a pasta com os resultados extraídos:

```bash
uv run python experiments/factorial_design/factorial_analysis.py \
  --designs experiments/factorial_design/designs/factorial_population.csv \
  --results_dir experiments/factorial_design/factorial_2k
```

Os testes automatizados conferem o cálculo contra `scipy.stats.f_oneway`,
casos balanceados e desbalanceados, variância zero, a exclusão das três últimas
métricas e a geração somente dos arquivos previstos.
