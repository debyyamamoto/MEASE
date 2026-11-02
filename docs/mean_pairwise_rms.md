# Distância RMS média ponderada entre curvas de sobrevivência de subgrupos completos

Nome no código e nos CSVs: `mean_pairwise_rms`. Direção desejada, quando o objetivo
é diversidade de modelos: **maximizar**. É uma medida descritiva de separação
geométrica, não um teste de hipóteses nem uma medida de qualidade preditiva.

## 1. Curvas comparadas

Seja o conjunto de regras retornado $\mathcal R=\{r_1,\ldots,r_K\}$ e seja $G_i$
o conjunto de indivíduos cobertos por $r_i$. Estimamos a sobrevivência em cada
**subgrupo completo**: os indivíduos em $G_i\cap G_j$ permanecem nas duas curvas.
Não comparamos somente $G_i\setminus G_j$ e $G_j\setminus G_i$.

Com censura à direita, o estimador de Kaplan–Meier é

$$
\widehat S_i(t)=\prod_{u\leq t}\left(1-\frac{d_i(u)}{Y_i(u)}\right),
$$

em que $d_i(u)$ é o número de eventos em $u$ e $Y_i(u)$ é o número de indivíduos
do subgrupo em risco imediatamente antes de $u$. Produtos consideram tempos
com eventos e denominadores positivos; censuras reduzem os conjuntos em risco,
mas não são eventos. Na interface de métricas, eventos devem estar codificados
como 0 (censura) e 1 (evento), como em `Dataset.get_data()`.

Referência de implementação independente do estimador:
[lifelines, KaplanMeierFitter](https://lifelines.readthedocs.io/en/stable/fitters/univariate/KaplanMeierFitter.html).
A ponderação e a agregação RMS abaixo são escolhas operacionais do MEASE;
essa referência não é uma proposta desta métrica específica.

## 2. Grade, duração dos intervalos e pesos

Utilizamos uma grade comum de tempos $t_1<\cdots<t_M$ para todo o dataset $D$,
com $n=|D|$. Definimos $t_{M+1}=\max_{x\in D} T_x$ e

$$
\Delta t_m=t_{m+1}-t_m,\qquad
w_m=\frac{Y_D(t_m)}{n}.
$$

Aqui $T_x$ é o tempo observado até evento ou censura. O peso $w_m$ é a fração
da **população inteira** ainda em risco, e não a fração do par de subgrupos.
O fator $\Delta t_m$ leva em conta a duração do intervalo; simplesmente tirar
uma média dos pontos da curva daria mais influência a regiões com mais pontos.
O peso de risco reduz a influência de regiões tardias com poucos indivíduos
observados. Não é uma ponderação pela variância nem uma correção completa para
a incerteza das curvas dos subgrupos.

No modo exato (`--km_time_bins 0` ou `None` na API), a grade contém todos os
tempos distintos de eventos da população. Sobre essa grade, os valores de
Kaplan–Meier são exatos, inclusive com empates entre eventos e censuras.
O peso de risco é avaliado no início de cada intervalo: esta soma não equivale,
em geral, a integrar um peso de risco continuamente atualizado nas censuras.
O período anterior ao primeiro evento não entra no numerador **nem no
denominador**, por compatibilidade com o diagnóstico original.

O padrão é `--km_time_bins 512`. Quando há mais de 512 tempos distintos de
eventos, a implementação usa até 512 quantis desses tempos. As contagens são
agrupadas nessa grade (tempos mapeados ao limite superior correspondente),
produzindo aproximações das curvas e dos conjuntos em risco. As equações
seguintes usam então essas estimativas discretizadas. Fixe a mesma resolução
para comparar configurações. “Completo” refere-se aos membros dos subgrupos;
não significa grade exata, seguimento infinito ou extrapolação das curvas.

## 3. Distância entre duas curvas

Somente intervalos em que os dois subgrupos ainda possuem indivíduos em risco
no início são comparáveis:

$$
\mathcal V_{ij}=\{m:Y_i(t_m)>0,\ Y_j(t_m)>0,\ \Delta t_m>0,\ w_m>0\}.
$$

Normalizando os pesos no domínio desse par,

$$
a_{ijm}=\frac{\Delta t_m\,w_m}
{\sum_{\ell\in\mathcal V_{ij}}\Delta t_\ell\,w_\ell},
\qquad m\in\mathcal V_{ij},
$$

a distância RMS (*root mean square*) é

$$
d_{ij}^{\mathrm{RMS}}=
\sqrt{\sum_{m\in\mathcal V_{ij}}a_{ijm}
\left[\widehat S_i(t_m)-\widehat S_j(t_m)\right]^2}.
$$

Os valores da sobrevivência são os valores **após os eventos** de $t_m$;
os números em risco são os valores imediatamente antes. O critério de
validade é avaliado na grade, não em cada instante dentro do intervalo.
Os domínios válidos e a normalização podem variar entre pares.

A distância está em $[0,1]$, é simétrica e não tem unidade temporal. Uma
distância zero indica coincidência nos intervalos considerados, não prova de
igualdade das distribuições. Uma distância de 0,20 representa uma separação
quadrática média de 20 pontos percentuais de probabilidade de sobrevivência;
não implica uma diferença constante de 20 pontos nem maior sobrevivência de
um subgrupo em todos os tempos.

O quadrado impede que diferenças com sinais opostos se cancelem quando as
curvas se cruzam e dá maior influência a separações grandes. A raiz retorna
à escala de probabilidade. Para os mesmos pesos e intervalos, RMS é pelo
menos tão grande quanto a distância ABC normalizada, que usa $|S_i-S_j|$.

Exemplo com dois intervalos de pesos normalizados iguais: se as diferenças
são $0,10$ e $-0,30$, a média com sinal é $-0,10$, a ABC é $0,20$, e

$$
d^{\mathrm{RMS}}=\sqrt{(0,10^2+(-0,30)^2)/2}\approx0,2236.
$$

## 4. Qualidade do conjunto de regras

Quando todos os pares são comparáveis, a métrica exportada é

$$
\operatorname{mean\_pairwise\_rms}(\mathcal R)=
\frac{2}{K(K-1)}\sum_{i<j}d_{ij}^{\mathrm{RMS}}.
$$

Cada par recebe o mesmo peso, independentemente do tamanho dos seus subgrupos.
Primeiro calculamos a raiz em cada par; depois fazemos a média entre pares.
Não é a raiz de uma média global de todas as diferenças quadráticas.

Se alguns pares não tiverem denominador positivo, usamos o conjunto $\mathcal P$
dos pares mensuráveis:

$$
\operatorname{mean\_pairwise\_rms}(\mathcal R)=
\frac{1}{|\mathcal P|}\sum_{(i,j)\in\mathcal P}d_{ij}^{\mathrm{RMS}}.
$$

`rms_valid_pairs` registra $|\mathcal P|$. Compare-o sempre com $K(K-1)/2$,
usando o número **efetivo** de regras exportadas em `#sg`. Se não houver pares
mensuráveis, inclusive conjuntos com menos de duas regras, o resultado é
`NaN` (campo vazio no CSV). Não se atribui separação máxima à falta de dados.
Um dataset sem eventos ou sem intervalos de duração positiva também não
permite essa avaliação na grade definida. Pares com subgrupos idênticos,
quando mensuráveis, contribuem com zero.

Essa convenção resolve um caso degenerado do diagnóstico exploratório: nele,
`1 - similaridade` podia ser 1 quando a similaridade retornava zero por falta
de intervalos válidos. Os dez pares de cada execução da config9 são válidos,
por isso os valores publicados no diagnóstico são reproduzidos sem alteração.
A regra de seleção do arquivo de soluções mantém seu comportamento anterior.

## 5. Diferença para model redundancy e para fitness

`model redundancy` é a proporção de pares classificados como redundantes por
um teste log-rank entre as **partes exclusivas** dos subgrupos, com o fallback
existente quando uma parte fica vazia. A RMS usa as **curvas completas** e
quantifica magnitude geométrica, sem teste, p-valor ou limiar de significância.
Por isso, uma configuração pode ter simultaneamente RMS maior e
`model redundancy` maior.

No diagnóstico da config9, em mgus2 e seed 0:

| Fitness usado na busca | mean_pairwise_rms | Pares válidos | model redundancy |
| --- | ---: | ---: | ---: |
| fast_logrank | 0,187704213773 | 10/10 | 0,10 |
| km_abc | 0,468775451092 | 10/10 | 0,20 |

Isso sustenta a observação de maior separação geométrica naquela execução de
`km_abc`. Não demonstra superioridade estatística, causalidade nem que todas
as suas curvas se cruzem menos. RMS não conta cruzamentos: curvas que se
cruzam podem ter RMS alta, e curvas paralelas muito próximas podem ter RMS baixa.

`mean_pairwise_rms` avalia a diversidade **entre regras**. O fitness `km_abc`
avalia a separação de **uma regra e sua referência**, usa diferença absoluta
e ainda incorpora o peso de suporte. São objetivos distintos. Para pares
mensuráveis, a similaridade geométrica usada pelo arquivo é $1-d_{ij}^{RMS}$;
a métrica exportada é calculada sobre as regras finais retornadas.

## 6. Limites e uso experimental

- Valor alto pode resultar de curvas estimadas com poucos indivíduos. Use
  conjuntamente suporte, cobertura, excepcionalidade e incerteza das curvas.
- A normalização torna a distância invariante a uma troca multiplicativa de
  unidade temporal, como meses para dias, salvo tolerâncias numéricas. Não
  elimina diferenças de censura, seguimento ou composição entre datasets.
- Como o domínio válido depende do par, não se presume que esta medida
  satisfaça todos os axiomas de uma distância matemática, especialmente a
  desigualdade triangular. Também não é a distância RMS irrestrita entre
  distribuições completas.
- Pares omitidos podem tornar a média pouco representativa. Informe quantos
  pares foram avaliados; não compare médias sem avaliar essa disponibilidade.
- Os pares compartilham regras e indivíduos, portanto não são réplicas
  independentes para construir um IC. Para a variabilidade da busca, calcule
  uma métrica por execução e depois analise as repetições. Para generalização
  entre datasets, mantenha os datasets como blocos.

## 7. Integração e reprodução

`compute_run_metrics(..., km_time_bins=512)` calcula a métrica para qualquer
fitness e baseline. O runner encaminha a resolução do CLI e a exporta em
`*_RulesMetricsResult.csv`, com oito casas decimais na saída por execução.
O resumo de múltiplas execuções a inclui automaticamente; o coletor
`experiments/collect_results.py` também a inclui e deixa valores ausentes
para CSVs antigos. `rms_valid_pairs` é um contador de disponibilidade, não um
objetivo de qualidade.

Os rankings e as famílias de hipóteses já definidos para três métricas não
foram alterados. Os resultados históricos também não foram recalculados.
Os protocolos congelados verificam hashes do código: esta versão não deve ser
misturada com execuções da versão anterior na mesma campanha. Para incorporar
RMS à análise histórica, faça uma reavaliação identificada separadamente a
partir de `DetailedRules.csv`, sem modificar os resultados originais.

Validação automatizada: `python -m unittest discover -s tests -v`. Inclui cálculo
manual, estimador independente de Kaplan–Meier, cruzamentos, subgrupos
aninhados, casos sem dados comparáveis e agregação de CSVs novos e antigos.
A reprodução dos valores da config9 é verificada quando os artefatos locais
do diagnóstico estão disponíveis.

## Texto sugerido para o artigo

> Para caracterizar a diversidade geométrica dos modelos de sobrevivência
> associados às regras, calculamos a distância RMS média ponderada entre
> pares de curvas de Kaplan–Meier dos subgrupos completos. Em cada par, as
> diferenças quadráticas entre probabilidades de sobrevivência foram
> ponderadas pela duração dos intervalos da grade temporal e pela proporção
> da população em risco no início de cada intervalo. Os pesos foram
> normalizados nos intervalos em que ambos os subgrupos apresentavam
> indivíduos em risco. A medida do conjunto corresponde à média aritmética
> das distâncias RMS dos pares avaliáveis. Valores maiores indicam maior
> separação entre curvas, sem implicar significância estatística. Reportamos
> também o número de pares avaliáveis, e resultados sem pares avaliáveis
> foram tratados como ausentes.

Acrescente a resolução realmente utilizada (por exemplo, até 512 pontos por
quantis dos tempos distintos de eventos) e as equações das seções 2–4.
