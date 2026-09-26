# Documentacao das Metricas de Score do MEASE

Data: 2026-07-27

Este documento explica a implementacao das funcoes objetivo usadas no `evaluate`
dos individuos do MEASE, com foco nas novas metricas adicionadas para quantificar
discrepancia entre curvas de sobrevivencia sem necessariamente executar um teste
estatistico formal a cada regra.

As metricas disponiveis em `main.py` sao:

- `legacy_logrank`
- `fast_logrank`
- `km_cvm`
- `km_abc`
- `mdir2`
- `mdir3`
- `mdir4`

O uso via CLI e:

```bash
uv run python main.py experiments/artificial_datsets/generated/teste_100_000.parquet \
  -time time \
  -event event \
  -label subgroup \
  -comp complement \
  --alpha 0.10 \
  --score_metric km_cvm
```

Se `--score_metric` nao for informado, o padrao atual permanece
`legacy_logrank`, para preservar compatibilidade com os experimentos anteriores.

## Motivacao

O gargalo historico do MEASE estava no calculo da fitness de cada regra. Para
cada individuo da populacao, o algoritmo:

1. calcula quais subjects sao cobertos pela regra;
2. compara a curva de sobrevivencia do subgrupo coberto contra uma referencia;
3. combina a discrepancia encontrada com o suporte relativo da regra.

A fitness final segue a forma:

```text
fitness(rule) = discrepancy(rule) * support(rule)^alpha
```

com filtro de suporte aplicado antes do score:

```text
0.05 <= support(rule) <= 0.55
```

Isso aparece em `RuleEvaluator.fitness`, em `easd/evaluation.py`.

A motivacao dos novos scores vem de uma observacao metodologica importante: no
MEASE, a funcao objetivo nao precisa necessariamente produzir um p-value correto.
Ela precisa ranquear regras promissoras durante a busca. Portanto, uma estatistica
de discrepancia entre curvas pode ser mais adequada e muito mais barata do que
recalcular um teste de log-rank completo para cada regra.

## A afirmativa esta correta?

Sim, com uma distincao importante.

Para `km_cvm` e `km_abc`, a afirmativa esta correta: a fitness deixa de chamar
uma rotina estatistica de alto nivel a cada regra e passa a estimar curvas
Kaplan-Meier por contagens NumPy. Depois disso, calcula diretamente a diferenca
entre a curva do subgrupo coberto pela regra e a curva de referencia escolhida
por `-comp`.

O gargalo legado observado no caminho da fitness vem principalmente de:

```python
statsmodels.duration.survdiff(...)
```

chamado dentro de `_legacy_logrank_score` para cada regra avaliada. O `lifelines`
aparece no projeto para visualizacao de curvas Kaplan-Meier em
`easd/visualization.py`, mas nao e usado no hot path de `RuleEvaluator.fitness`.

Tambem ha uma distincao entre as novas metricas:

| metrica | estima Kaplan-Meier na fitness? | chama `survdiff`/`lifelines` por regra? | o que mede |
| --- | --- | --- | --- |
| `legacy_logrank` | nao diretamente no codigo do MEASE | sim, `statsmodels.survdiff` | `1 - p_value` do teste log-rank |
| `fast_logrank` | nao | nao | proxy rapido baseado em residuos tipo log-rank |
| `km_abc` | sim, via NumPy | nao | area ponderada absoluta entre curvas |
| `km_cvm` | sim, via NumPy | nao | distancia quadratica ponderada entre curvas |
| `mdir2`/`mdir3`/`mdir4` | usa a curva pooled para pesos | nao | forma quadratica de weighted log-rank statistics |

Portanto, a explicacao curta e:

- `km_cvm` e `km_abc` aceleram porque transformam a comparacao de curvas em
  contagens por bin, `np.bincount`, `np.cumsum` reverso e `np.cumprod`.
- `fast_logrank` acelera porque precomputa contribuicoes individuais do log-rank
  e depois soma essas contribuicoes por regra.
- `mdir2`, `mdir3` e `mdir4` aceleram porque usam a mesma grade de tempos e as
  mesmas contagens agregadas das metricas KM, combinando varios weighted log-rank
  scores por algebra linear pequena.
- `legacy_logrank` e mais caro porque recria grupos e executa uma rotina completa
  de teste estatistico para cada regra.

## Arquitetura no Codigo

O ponto central da implementacao e `RuleEvaluator`, em `easd/evaluation.py`.

Durante a inicializacao, o avaliador recebe:

```python
RuleEvaluator(
    dataset_obj,
    comparacao,
    alpha,
    score_metric="legacy_logrank",
    km_time_bins=512,
)
```

O parametro chega pelo CLI em `main.py`, e e propagado por:

- `main.config_from_args`
- `RunConfig` em `easd/runner.py`
- `MEASE.__init__` em `easd/core.py`
- `RuleEvaluator.__init__` em `easd/evaluation.py`

As colunas de tempo e evento sao convertidas uma vez para arrays NumPy:

```python
self._time_values = ...
self._event_values = ...
self._event_observed = ...
```

Esse detalhe e importante: as novas metricas evitam criar `pd.Series`, concatenar
grupos e chamar `statsmodels.duration.survdiff` a cada regra.

Os principais campos internos criados no avaliador sao:

| campo | usado por | significado |
| --- | --- | --- |
| `_time_values` | todas | tempos de sobrevivencia como `np.ndarray` |
| `_event_values` | todas | indicador de evento/censura como `np.ndarray` inteiro |
| `_event_observed` | todas | mascara booleana dos subjects com evento observado |
| `_logrank_gamma` | `fast_logrank` | contribuicao esperada aproximada de cada subject |
| `_logrank_residual` | `fast_logrank` | diferenca `evento_observado - esperado` por subject |
| `_km_grid_times` | `km_cvm`, `km_abc`, `mdir*` | grade de tempos onde curvas e weighted log-rank scores serao avaliados |
| `_km_widths` | `km_cvm`, `km_abc` | largura temporal associada a cada ponto da grade |
| `_km_weights` | `km_cvm`, `km_abc` | peso de estabilidade baseado no total em risco |
| `_km_risk_bin` | `km_cvm`, `km_abc`, `mdir*` | bin ate o qual cada subject contribui para o risco |
| `_km_event_bin` | `km_cvm`, `km_abc`, `mdir*` | bin onde o evento do subject e contado, ou `-1` se censurado |
| `_km_pooled_subject_counts` | `km_cvm`, `km_abc`, `mdir*` | contagem global de subjects por bin de risco |
| `_km_pooled_event_counts` | `km_cvm`, `km_abc`, `mdir*` | contagem global de eventos por bin |
| `_km_pooled_risk_counts` | `km_cvm`, `km_abc`, `mdir*` | numero global em risco em cada bin |
| `_km_pooled_survival` | `km_cvm`, `km_abc`, `mdir*` | curva Kaplan-Meier da populacao completa |
| `_mdir_weights` | `mdir2`, `mdir3`, `mdir4` | matriz de pesos das direcoes weighted log-rank |

## Cobertura da Regra

Toda metrica comeca com:

```python
rule_mask = self.get_covered_mask(rule, dataset_x)
```

`rule_mask` e um array booleano com tamanho `n_subjects`, onde `True` indica que
o subject foi coberto pela regra.

Para atributos numericos:

```text
lower <= x <= upper
```

Para atributos categoricos:

```text
x in accepted_values
```

A decisao de usar mask booleana em vez de lista de indices e essencial para
escala, porque permite usar operacoes vetorizadas, somas por mascara e
`np.bincount`.

Esta etapa ainda precisa varrer as colunas usadas pela regra. Ou seja, se a regra
tem tres condicoes, a mask e refinada tres vezes. O ganho esta em fazer essa
varredura em arrays NumPy, evitando loops Python por subject.

## `legacy_logrank`

Esta e a metrica original preservada por compatibilidade.

Implementacao:

```python
_legacy_logrank_score(rule_mask)
```

Ela monta um vetor de grupos:

```text
sg          para subjects cobertos
complement  para subjects nao cobertos, quando -comp complement
pop         para referencia population
```

Depois chama:

```python
statsmodels.duration.survdiff(...)
```

e usa:

```text
discrepancy = 1 - p_value
```

### Relacao com a literatura

O log-rank e o padrao classico para comparar curvas de sobrevivencia, sendo
especialmente eficiente quando a hipotese de proportional hazards e adequada.
Dormuth et al. (2022) destacam esse ponto, mas tambem mostram que o log-rank pode
perder poder em cenarios de non-proportional hazards e curvas cruzadas.

### Limite pratico

Para MEASE, o problema nao e apenas estatistico: e computacional. O `survdiff`
precisa varrer tempos de evento e construir estruturas internas a cada regra.
Em datasets grandes, isso multiplica o custo por:

```text
population_size * generations * chamadas extras no crossover
```

## `fast_logrank`

Esta metrica e um proxy rapido do log-rank.

Implementacao:

```python
_precompute_fast_logrank()
_fast_logrank_score(rule_mask)
```

A ideia e precomputar, uma unica vez, um valor esperado por subject. Primeiro sao
obtidos os tempos de evento distintos, os eventos por tempo e o numero em risco
por tempo:

```text
d_t = numero de eventos no tempo t
Y_t = numero de subjects em risco no tempo t
alpha_t = d_t / Y_t
```

Depois, para cada subject `i`, calcula-se:

```text
gamma_i = sum alpha_t para tempos t em que subject i ainda estava em risco
residual_i = D_i - gamma_i
```

Durante o evaluate de uma regra, basta somar os residuos dos subjects cobertos:

```text
numerator(rule) = sum residual_i, para i em SG
expected_sg = sum gamma_i, para i em SG
```

Para `-comp complement`, o score aproximado e:

```text
statistic = numerator^2 * (1 / expected_sg + 1 / expected_ref)
```

Depois a estatistica e comprimida para a faixa `[0, 1)`:

```text
discrepancy = statistic / (1 + statistic)
```

Essa compressao evita que valores extremos dominem completamente o termo de
suporte.

### Relacao com a literatura

Sverdrup, Yang e LeBlanc (2026), no artigo `Efficient Log-Rank Updates for Random
Survival Forests`, mostram que o numerador do log-rank pode ser reescrito como
uma soma de quantidades por individuo. O artigo usa essa propriedade para fazer
atualizacoes em tempo constante ao varrer splits ordenados em survival forests.

No MEASE, nao estamos varrendo splits ordenados de uma unica variavel. Estamos
avaliando regras arbitrarias. Ainda assim, a ideia central e reaproveitada:
precomputar a contribuicao individual e, para cada regra, somar apenas os
subjects cobertos pela mask.

O artigo tambem discute uma aproximacao de variancia por Poissonizacao. A
normalizacao por `expected_sg` e `expected_ref` implementada aqui segue esse
espirito, mas deve ser lida como proxy de ranking, nao como teste formal.

## `mdir2`, `mdir3` e `mdir4`

Estas metricas implementam uma versao escalavel da estatistica Omnibus
Multi-direction Logrank para uso como score de ranking do MEASE.

Implementacao:

```python
_precompute_km_grid()
_precompute_mdir_weights()
_mdir_score(rule_mask)
```

O teste mdir formal combina varios weighted log-rank statistics em uma forma
quadratica studentizada. No MEASE, usamos a estatistica bruta como funcao
objetivo, sem calcular p-value por aproximacao qui-quadrado e sem permutacoes.
Isso e importante para escala: o pacote R `mdir.logrank`, por exemplo, tambem
oferece p-value por permutacao, mas isso seria caro demais dentro do loop
evolutivo.

### Pesos implementados

A documentacao do pacote R define direcoes do tipo:

```text
w(x) = x^r * (1 - x)^g
```

e uma direcao de crossing:

```text
w_cross(x) = 1 - 2x
```

No codigo, usamos:

```text
x_j = F_hat(t_j-) = 1 - S_pooled(t_j-)
```

ou seja, a probabilidade acumulada de falha pooled imediatamente antes do ponto
da grade. A curva pooled vem de `_km_pooled_survival`.

As variantes disponiveis sao:

| metrica | direcoes usadas |
| --- | --- |
| `mdir2` | proportional `(0, 0)` + crossing |
| `mdir3` | proportional `(0, 0)` + early `(0, 4)` + crossing |
| `mdir4` | proportional `(0, 0)` + early `(0, 4)` + late `(4, 0)` + crossing |

O `mdir2` e a opcao mais fiel ao default do pacote R e a recomendacao mais clara
do artigo de Dormuth et al. (2022): log-rank + crossing weight. Como o texto
principal do artigo nao detalha exatamente quais direcoes extras foram usadas nos
rotulos `mdir3` e `mdir4`, esta implementacao explicita os pesos extras acima.
Assim, os resultados de `mdir3` e `mdir4` devem ser lidos como extensoes
experimentais controladas, nao como uma reproducao bit-a-bit do pacote R.

### Score calculado por regra

Para cada bin `j`, o codigo tem:

```text
Y_j     = numero pooled em risco
d_j     = numero pooled de eventos
Y_sg_j  = numero do subgrupo em risco
d_sg_j  = numero de eventos do subgrupo
```

O evento esperado no subgrupo, sob a referencia pooled, e:

```text
E_sg_j = (Y_sg_j / Y_j) * d_j
```

O residuo de weighted log-rank em cada ponto e:

```text
r_j = d_sg_j - E_sg_j
```

Para cada direcao de peso `w_k`, calcula-se:

```text
U_k = sum_j w_kj * r_j
```

Esses valores formam o vetor:

```text
U = [U_1, U_2, ..., U_m]
```

A matriz de covariancia empirica e aproximada por:

```text
V_ab = sum_j w_aj * w_bj * d_j * p_j * (1 - p_j) * (Y_j - d_j) / (Y_j - 1)
```

onde:

```text
p_j = Y_sg_j / Y_j
```

Por fim, a estatistica mdir usada como discrepancia e:

```text
statistic = U^T * pinv(V) * U
```

em que `pinv(V)` e a pseudo-inversa de Moore-Penrose. O uso de pseudo-inversa e
importante porque algumas direcoes podem ficar quase colineares ou pouco
informativas em uma regra especifica.

Assim como em `fast_logrank`, a estatistica e comprimida:

```text
discrepancy = statistic / (1 + statistic)
```

Depois, a fitness final continua sendo:

```text
fitness = discrepancy * support^alpha
```

### Relacao com a literatura

Dormuth et al. (2022) descrevem o mdir como teste omnibus que combina weighted
log-rank statistics para cobrir alternativas proporcionais, nao proporcionais e
com cruzamento de curvas. O artigo relata desempenho consistente em poder e erro
tipo I, e observa que `mdir2`, com log-rank + crossing weight, foi tao poderoso
quanto ou mais poderoso que `mdir3`/`mdir4` na maioria dos cenarios avaliados.

A documentacao do pacote R `mdir.logrank` confirma a configuracao default
`cross = TRUE` e `rg = list(c(0, 0))`, isto e, proporcional + crossing. A
implementacao do MEASE usa essa ideia, mas nao calcula os p-values do pacote R.

### Escalabilidade

O custo caro do mdir formal seria recalcular weighted log-rank tests completos e,
principalmente, p-values por permutacao para cada regra. Isso nao foi
implementado.

No MEASE, o caminho escalavel e:

1. precomputar a grade de tempos e a curva pooled uma vez;
2. precomputar a matriz de pesos `_mdir_weights` uma vez;
3. para cada regra, obter contagens por bin com `np.bincount`;
4. calcular `U`, `V` e `U^T pinv(V) U`.

Como `m` e pequeno (`2`, `3` ou `4`), a pseudo-inversa e desprezivel perto do
custo de avaliar milhares de masks.

## `km_abc`

Esta metrica mede a area entre as curvas Kaplan-Meier.

Implementacao:

```python
_precompute_km_grid()
_km_distance_score(rule_mask)
```

Matematicamente, a ideia e:

```text
ABC = integral w(t) * |S_sg(t) - S_ref(t)| dt
```

No codigo, a integral e aproximada por soma discreta em uma grade de tempos:

```text
distance =
  sum width_j * weight_j * abs(S_sg_j - S_ref_j)
  ------------------------------------------------
  sum width_j * weight_j
```

onde:

- `S_sg_j` e a Kaplan-Meier do subgrupo no ponto da grade;
- `S_ref_j` e a Kaplan-Meier do complemento ou populacao;
- `width_j` e o intervalo ate o proximo ponto da grade;
- `weight_j` e proporcional ao numero pooled de subjects em risco.

### Relacao com a literatura

Dormuth et al. (2022) discutem o teste ABC, baseado na area entre curvas, como uma
alternativa capaz de capturar diferencas em curvas de sobrevivencia que podem
cruzar. Isso e relevante porque uma metrica assinada, como diferenca de RMST, pode
sofrer cancelamento: uma curva fica acima em parte do tempo e abaixo em outra.

Pepe e Fleming (1989), em `Weighted Kaplan-Meier Statistics`, propoem uma classe
de estatisticas baseadas em diferencas ponderadas entre estimadores Kaplan-Meier.
O `km_abc` implementado aqui e inspirado diretamente nessa familia de metricas de
distancia entre curvas, mas sem calcular distribuicao assintotica ou p-value.

## `km_cvm`

Esta foi a metrica que voce observou como mais forte nos seus testes.

Ela usa a mesma infraestrutura de Kaplan-Meier do `km_abc`, mas troca modulo por
quadrado:

```text
CvM = integral w(t) * (S_sg(t) - S_ref(t))^2 dt
```

No codigo:

```text
distance =
  sum width_j * weight_j * (S_sg_j - S_ref_j)^2
  ------------------------------------------------
  sum width_j * weight_j
```

### Por que `km_cvm` pode funcionar melhor?

Ha algumas razoes praticas:

1. Ela mede discrepancia entre curvas diretamente, que e exatamente o objetivo da
   funcao de fitness neste contexto.
2. Por elevar a diferenca ao quadrado, regioes com separacao forte entre curvas
   ganham mais peso do que pequenas diferencas ruidosas.
3. Ela nao depende de p-value. Portanto, evita saturacoes e custos associados a
   testes formais repetidos milhares de vezes.
4. Ela evita o cancelamento de sinais. Se curvas cruzam, diferencas positivas e
   negativas continuam contribuindo positivamente para a distancia.
5. Em datasets sinteticos com subgrupo injetado, uma regra que realmente separa
   uma curva de sobrevivencia tende a produzir diferencas persistentes em varios
   pontos da grade, o que favorece a estatistica integrada.

### Relacao com Cramer-von Mises

O nome `km_cvm` vem da familia de estatisticas do tipo Cramer-von Mises, que
medem discrepancia integrada quadratica entre funcoes de distribuicao ou curvas.
No caso do MEASE, usamos uma versao de engenharia:

```text
distancia quadratica ponderada entre curvas Kaplan-Meier
```

Nao e uma implementacao completa do teste de Schumacher (1994) ou de um teste
Cramer-von Mises com distribuicao nula, variancia e p-value. Isso e proposital: o
objetivo e ranquear individuos na busca evolutiva, nao fazer inferencia formal a
cada fitness.

## Como a Kaplan-Meier e calculada

Para `km_cvm`, `km_abc` e `mdir*`, o codigo evita bibliotecas externas de
plotting, fitting ou testes estatisticos a cada regra. Em vez disso, usa
contagens vetorizadas.

Na inicializacao:

1. obtem os tempos de evento observados;
2. cria uma grade de tempos;
3. mapeia cada subject para:
   - um bin de risco;
   - um bin de evento, se houve evento;
4. calcula contagens pooled de subjects e eventos por bin;
5. calcula o numero em risco por soma cumulativa reversa;
6. calcula a curva pooled de Kaplan-Meier.

A formula de Kaplan-Meier usada e:

```text
S(t_j) = product_{k <= j} (1 - d_k / Y_k)
```

No codigo:

```python
hazards = event_counts / risk_counts
survival = np.cumprod(1.0 - hazards)
```

Em termos de implementacao, `_kaplan_meier_from_counts` recebe dois vetores:

```text
event_counts[j] = numero de eventos no bin j
risk_counts[j]  = numero de subjects em risco no bin j
```

e transforma isso em:

```text
hazard_j = event_counts[j] / risk_counts[j]
S_j = S_{j-1} * (1 - hazard_j)
```

O `np.cumprod` e exatamente esse produto acumulado.

Para cada regra em `km_cvm` e `km_abc`:

1. `rule_mask` seleciona subjects cobertos;
2. `np.bincount` calcula eventos e subjects cobertos por bin;
3. o complemento e obtido por subtracao das contagens pooled;
4. as curvas `S_sg` e `S_ref` sao calculadas;
5. a distancia integrada e computada na grade.

### Passo a passo de `_precompute_km_grid`

Esta funcao roda uma vez na criacao do `RuleEvaluator`, antes da busca avaliar
regras.

1. Seleciona apenas os tempos em que houve evento observado:

```python
event_times = np.sort(np.unique(self._time_values[self._event_observed]))
```

Subjects censurados entram na conta de risco, mas nao criam eventos.

2. Define a grade de tempos:

- se o numero de tempos de evento e menor ou igual a `km_time_bins`, usa todos os
  tempos de evento;
- se ha muitos tempos distintos, usa quantis dos tempos de evento para limitar a
  grade;
- se `--km_time_bins 0`, usa todos os tempos de evento.

3. Calcula `_km_widths`, que representa a largura temporal associada a cada ponto
da grade:

```python
self._km_widths = np.diff(np.r_[self._km_grid_times, max_time])
```

Esse vetor e usado na integral discreta. Uma diferenca entre curvas que dura
muito tempo deve pesar mais do que uma diferenca que aparece em um intervalo
muito curto.

4. Mapeia cada subject para um bin de risco:

```text
_km_risk_bin[i] = ultimo bin em que o subject i ainda contribui para o conjunto em risco
```

Se um subject tem tempo 10, ele contribui para o conjunto em risco ate os bins
que representam tempos menores ou iguais a 10.

5. Mapeia cada evento para um bin de evento:

```text
_km_event_bin[i] = bin onde o evento do subject i e contado
```

Se o subject foi censurado, `_km_event_bin[i] = -1`. Isso significa que ele
contribui para o denominador de risco ate o tempo de censura, mas nao contribui
para o numerador de eventos.

6. Agrega contagens globais com `np.bincount`:

```text
_km_pooled_subject_counts[j] = subjects cujo ultimo bin de risco e j
_km_pooled_event_counts[j]   = eventos observados no bin j
```

7. Calcula o numero em risco por soma cumulativa reversa:

```python
self._km_pooled_risk_counts = np.cumsum(self._km_pooled_subject_counts[::-1])[::-1]
```

A soma e reversa porque o numero em risco no bin `j` inclui todos os subjects
cujo tempo observado chega ate `j` ou passa de `j`.

8. Calcula a Kaplan-Meier pooled:

```python
self._km_pooled_survival = self._kaplan_meier_from_counts(
    self._km_pooled_event_counts,
    self._km_pooled_risk_counts,
)
```

Essa curva pooled fica guardada para ser reutilizada por todas as regras.

9. Calcula pesos por estabilidade:

```python
self._km_weights = self._km_pooled_risk_counts / n_subjects
```

Assim, regioes muito tardias da curva, com poucos subjects ainda em risco, tem
menor influencia na distancia final. Isso segue a intuicao das estatisticas
Kaplan-Meier ponderadas: pontos da curva com mais informacao devem ter mais peso.

### Passo a passo de `_km_distance_score`

Esta funcao roda a cada regra quando `score_metric` e `km_cvm` ou `km_abc`.

1. Obtem as contagens do subgrupo:

```python
group_subject_counts, group_event_counts = self._group_km_counts(rule_mask)
```

Internamente, `_group_km_counts` faz:

```python
rule_indices = np.flatnonzero(rule_mask)
```

e depois usa `np.bincount` sobre os bins desses subjects. Assim, o codigo nao
precisa montar um DataFrame do subgrupo nem ajustar um `KaplanMeierFitter`.

2. Calcula o numero em risco do subgrupo:

```python
group_risk_counts = np.cumsum(group_subject_counts[::-1])[::-1]
```

3. Calcula a curva Kaplan-Meier do subgrupo:

```python
group_survival = self._kaplan_meier_from_counts(group_event_counts, group_risk_counts)
```

4. Define a referencia:

```python
if self.comparacao == "population":
    ref_survival = self._km_pooled_survival
else:
    ref_subject_counts = self._km_pooled_subject_counts - group_subject_counts
    ref_event_counts = self._km_pooled_event_counts - group_event_counts
```

Com `-comp complement`, a referencia e construida por subtracao das contagens do
subgrupo das contagens globais. Isso evita recalcular tudo do zero para os
subjects nao cobertos.

5. Filtra pontos validos da grade:

```text
grupo tem subjects em risco
referencia tem subjects em risco
largura temporal e positiva
peso pooled e positivo
```

Esse filtro evita divisao por zero e evita comparar regioes em que uma das curvas
nao tem suporte empirico.

6. Calcula a diferenca entre curvas:

```python
diff = group_survival[valid] - ref_survival[valid]
```

7. Calcula a distancia:

Para `km_abc`:

```text
sum(width * weight * abs(diff)) / sum(width * weight)
```

Para `km_cvm`:

```text
sum(width * weight * diff^2) / sum(width * weight)
```

8. Limita o score ao intervalo `[0, 1]`:

```python
return min(max(distance, 0.0), 1.0)
```

Depois, `RuleEvaluator.fitness` ainda aplica:

```text
fitness = distance * support^alpha
```

Portanto, a metrica de curva mede discrepancia, e o termo de suporte controla a
preferencia por regras que cobrem uma quantidade minima e nao trivial de
subjects.

Para `mdir*`, a mesma grade e as mesmas contagens sao usadas para calcular
weighted log-rank statistics, e nao para integrar uma distancia entre curvas.

## O Papel de `km_time_bins`

O parametro `--km_time_bins` controla o tamanho maximo da grade de tempos para
`km_cvm`, `km_abc` e `mdir*`.

Padrao:

```text
--km_time_bins 512
```

Se o dataset tem muitos tempos de evento distintos, o codigo usa quantis dos
tempos de evento para criar uma grade com no maximo 512 pontos. Isso reduz custo
e memoria em datasets grandes.

Para usar todos os tempos de evento:

```bash
--km_time_bins 0
```

Essa opcao tende a ser mais fiel, mas pode ser mais cara em datasets grandes.
Para `mdir*`, `--km_time_bins 0` fica mais proximo do weighted log-rank em tempos
de evento exatos; com bins, o score deve ser lido como uma aproximacao
deliberada para escala.

## Complexidade Computacional

### `legacy_logrank`

Por regra:

```text
O(n + M) aproximadamente
```

onde `n` e o numero de subjects e `M` e o numero de tempos de evento distintos.
Na pratica, tambem ha overhead de objetos Python, pandas/statsmodels e alocacao do
vetor de grupos.

### `fast_logrank`

Precomputacao:

```text
O(n + M)
```

Por regra:

```text
O(n)
```

O custo principal e aplicar a mask e somar arrays NumPy. Embora o resultado use
apenas os subjects cobertos, a indexacao booleana precisa inspecionar a mask de
tamanho `n`.

### `km_cvm` e `km_abc`

Precomputacao:

```text
O(n + B)
```

onde `B` e o numero de bins da grade de tempo.

Por regra:

```text
O(n + n_covered + B)
```

O termo `n` vem da construcao da mask e de `np.flatnonzero(rule_mask)`. Depois
disso, as contagens usam os subjects cobertos e a parte temporal fica limitada a
`B`. Na pratica, `np.bincount`, `np.cumsum` e `np.cumprod` tornam esse custo
baixo, especialmente com `B = 512`.

### `mdir2`, `mdir3` e `mdir4`

Precomputacao:

```text
O(n + B*m)
```

onde `m` e o numero de direcoes weighted log-rank.

Por regra:

```text
O(n + n_covered + B*m^2 + m^3)
```

Como `m` e no maximo `4`, os termos de algebra linear sao pequenos. O custo
pratico fica proximo das metricas KM: construir a mask, agregar contagens por bin
e percorrer a grade de tamanho `B`.

## Interpretacao dos Scores

Os valores de `Rule_Score` nao devem ser comparados diretamente entre metricas.

Exemplo:

- `legacy_logrank` usa `1 - p_value`;
- `fast_logrank` usa uma estatistica aproximada comprimida;
- `km_abc` usa distancia L1 normalizada entre curvas;
- `km_cvm` usa distancia L2 quadratica normalizada entre curvas.
- `mdir2`/`mdir3`/`mdir4` usam uma forma quadratica de weighted log-rank
  statistics, tambem comprimida para `[0, 1)`.

Portanto, `Rule_Score = 0.001` em `km_cvm` nao significa regra pior do que
`Rule_Score = 0.20` em `legacy_logrank`. A comparacao deve ser feita dentro da
mesma metrica, ou por metricas externas como F1, cobertura, runtime e validacao
visual das curvas.

## Relacao com `-comp complement` e `-comp population`

Para `-comp complement`, a referencia e o conjunto `D \ SG`, ou seja, subjects
nao cobertos pela regra. Essa e a opcao mais coerente para comparar duas curvas
disjuntas.

Para as metricas KM com `-comp population`, a referencia e a curva pooled do
dataset inteiro. Essa opcao mede o quanto o subgrupo se desvia da curva global,
mas a referencia inclui o proprio subgrupo. Portanto, ela nao e uma comparacao
disjunta de duas amostras.

Para `mdir*`, a interpretacao e semelhante. Com `-comp complement`, o score
opera como comparacao subgrupo versus complemento. Com `-comp population`, ele
mede os eventos observados do subgrupo contra os eventos esperados pela
experiencia pooled do dataset. Isso e util como score de desvio, mas nao deve ser
lido como o teste mdir formal de duas amostras independentes.

Para o uso atual recomendado nos seus experimentos:

```bash
-comp complement --score_metric km_cvm
```

## Relacao com as Metricas Finais do Runner

A metrica usada no `evaluate` controla a busca evolutiva e aparece em:

```text
DetailedRules.csv -> Rule_Score
Info.csv -> score_metric
```

Entretanto, algumas metricas finais em `compute_run_metrics`, como
`exceptionality`, ainda usam a logica historica de p-value/log-rank da camada de
avaliacao final. Isso significa que:

- `Rule_Score` vem da metrica escolhida em `--score_metric`;
- `max_f1_score` vem da comparacao com a coluna `-label`, quando ela existe;
- `exceptionality` ainda e uma metrica estatistica separada, nao o score
  escolhido em `--score_metric`.

## Por que a Implementacao Ficou Mais Complexa

A complexidade adicional vem de tres decisoes de performance:

1. Precomputar tudo que independe da regra.
2. Representar cobertura por mask booleana em NumPy.
3. Calcular curvas KM por contagens agregadas, nao por objetos de alto nivel.

Uma implementacao mais curta poderia chamar `lifelines.KaplanMeierFitter` ou
`statsmodels.survdiff` dentro de cada fitness. Isso seria mais legivel, mas
voltaria ao problema original: milhares de alocacoes e rotinas caras dentro do
loop evolutivo.

Em outras palavras, o codigo ficou mais verboso porque ele separa explicitamente
coisas que bibliotecas de alto nivel fariam internamente:

- definicao da grade de tempos;
- contagem de eventos;
- contagem de subjects em risco;
- construcao da curva Kaplan-Meier;
- escolha da referencia;
- integracao da distancia entre curvas.

Essa separacao e justamente o que permite reutilizar precomputacoes e evitar que
cada regra seja tratada como um novo problema estatistico completo.

## Recomendacao Atual

Com base nos testes relatados por voce e nos benchmarks ja feitos no projeto, a
recomendacao pratica e:

```bash
--score_metric km_cvm -comp complement --alpha 0.10
```

Use `km_cvm` quando o objetivo principal for minerar subgrupos com curvas de
sobrevivencia claramente discrepantes, especialmente em datasets grandes.

Use `fast_logrank` quando quiser manter comportamento mais proximo ao log-rank
legado, mas com runtime bem menor.

Use `km_abc` quando houver interesse em uma distancia mais robusta e linear entre
curvas, especialmente se diferencas moderadas e distribuidas no tempo forem
importantes.

Use `mdir2` quando quiser testar uma alternativa omnibus ainda proxima da familia
log-rank, especialmente se houver suspeita de cruzamento de curvas. Use `mdir4`
como opcao experimental quando quiser combinar crossing com pesos early/late,
mas compare runtime e F1 contra `km_cvm`, pois os seus testes atuais ainda
favoreceram `km_cvm`.

## Referencias

1. Pepe, M. S.; Fleming, T. R. (1989). "Weighted Kaplan-Meier Statistics: A
   Class of Distance Tests for Censored Survival Data". Biometrics, 45(2),
   497-507. Arquivo local: `cramer.pdf`.

2. Dormuth, I.; Liu, T.; Xu, J.; Pauly, M.; Ditzhaus, M. (2022). "A comparative
   study to alternatives to the log-rank test". arXiv:2210.13258v1. Arquivo
   local: `2210.13258v1.pdf`.

3. Sverdrup, E.; Yang, J.; LeBlanc, M. (2026). "Efficient Log-Rank Updates for
   Random Survival Forests". arXiv:2510.03665v2. Arquivo local:
   `2510.03665v2.pdf`.

4. Hothorn, T.; Hornik, K.; Zeileis, A. (2006). "Unbiased Recursive
   Partitioning: A Conditional Inference Framework". Journal of Computational
   and Graphical Statistics, 15(3), 651-674. Arquivo local:
   `CTREE-2006-.pdf`.

5. Ditzhaus, M.; Friedrich, S. (2018). `mdir.logrank`: Multiple-Direction
   Logrank Test. Pacote R, versao 0.0.4. Documentacao CRAN:
   `https://CRAN.R-project.org/package=mdir.logrank`.

6. Dormuth, I.; Herrmann, C.; Konietschke, F.; Pauly, M.; Wirth, M.; et al.
   (2026). "Beyond Bonferroni: new multiple contrast tests for time-to-event
   data under non-proportional hazards". Lifetime Data Analysis, 32, Article 8.
   `https://doi.org/10.1007/s10985-025-09676-9`.

## Arquivos Relacionados

- `easd/evaluation.py`: implementacao das metricas.
- `main.py`: argumentos `--score_metric` e `--km_time_bins`.
- `easd/runner.py`: propagacao via `RunConfig`.
- `easd/core.py`: criacao do `RuleEvaluator` e registro da metrica no `Info.csv`.
- `experiments/artificial_datsets/score_metric_benchmark.py`: benchmark
  reproduzivel em datasets sinteticos.
- `experiments/artificial_datsets/score_metric_benchmark_summary.md`: resumo dos
  primeiros benchmarks de runtime e F1.
