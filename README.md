# HIN Empresas — Grafos Heterogeneos de Empresas

[![CI](https://github.com/brunokobi/experimento2026/actions/workflows/ci.yml/badge.svg)](https://github.com/brunokobi/experimento2026/actions/workflows/ci.yml)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![uv](https://img.shields.io/badge/package%20manager-uv-de5fe9)](https://docs.astral.sh/uv/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![PyTorch Geometric](https://img.shields.io/badge/PyG-2.5%2B-3C2179)](https://pytorch-geometric.readthedocs.io/)
[![Neo4j](https://img.shields.io/badge/Neo4j-graph%20export-008CC1?logo=neo4j&logoColor=white)](https://neo4j.com/)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![Status](https://img.shields.io/badge/status-research%20%2F%20WIP-yellow)](#)
[![License](https://img.shields.io/badge/license-Proprietary-lightgrey)](#)

Projeto de pesquisa academica para construcao e analise de uma **Rede Heterogenea de
Informacao (HIN)** representando empresas, socios, sancoes administrativas e vinculos
politicos da Grande Vitoria (ES), com foco em extracao de **metapaths** e modelos de
GNN (via PyTorch Geometric), sobre dados reais de
[`projeto_grande_vitoria_empresas`](https://github.com/brunokobi/projeto_grande_vitoria_empresas).

## O trabalho, em linguagem simples

A pergunta de fundo: **uma empresa conectada a uma empresa ja sancionada — via socio
em comum, mesmo endereco, ou vinculo politico do socio — tem risco maior de tambem
estar envolvida em irregularidade**, mesmo quando isso nao aparece nos dados tabulares
comuns (CNAE, porte, capital social)? E, se sim, **qual desses tipos de conexao carrega
mais sinal**?

Pra responder, as empresas e seus socios sao montados como um **grafo** (nao uma
planilha) e uma rede neural de grafos (GNN) aprende esse padrao de conexao, comparando
contra um modelo tabular tradicional (que so ve cada empresa isolada, sem a rede).

Como so **188 empresas em 344.130 (0,055%)** tem sancao administrativa confirmada, o
problema nao e classificacao balanceada — e **deteccao de anomalia / ranking de risco**
(PR-AUC, Precision@k: "das 20 empresas mais suspeitas segundo o modelo, quantas eram
mesmo sancionadas"). E o rotulo e **positivo-incompleto**: essas 188 sao as que ja
foram *pegas*, nao "toda irregularidade que existe" — isso e dito explicitamente no
texto, nao escondido.

**Achado que mudou o escopo (08/08/2026)**: ao reconferir o banco real, das 188
empresas positivas, **41 (22%) nao tem sancao na propria empresa** — foram marcadas
como positivas so por ter **socio em comum com outra empresa sancionada**. Isso e
risco de circularidade: e o mesmo mecanismo (socio comum) que a hipotese da pesquisa
quer testar. Mitigacao: o resultado principal usa so as 148 empresas com sancao
**direta** como rotulo; as 41 via-socio entram como analise de sensibilidade separada,
nunca misturadas sem declarar qual definicao de rotulo foi usada. Detalhe completo em
[`docs/research_plan.md`](docs/research_plan.md), secoes 5 e 9.

## Plano de pesquisa (dissertacao de mestrado)

Objetivo: produzir um artigo publicavel em veiculo de classificacao (Qualis/CAPES)
alta. Plano completo em [`docs/research_plan.md`](docs/research_plan.md) — resumo:

- **Pergunta de pesquisa**: metapaths estruturais (socio comum, endereco comum,
  vinculo politico) melhoram a identificacao de empresas com sancao administrativa
  confirmada, em relacao a um baseline tabular — e quais metapaths carregam mais sinal?
- **Fonte de dados (confirmada e reconferida no banco real em 08/08/2026)**: 344.130
  empresas, 231.890 socios, 322 sancoes administrativas (188 empresas distintas), 7
  municipios.
- **Rotulo (achado critico)**: so **188 empresas em 344.130 (0,055%)** tem sancao
  administrativa confirmada — desbalanceamento extremo. Tratado como deteccao de
  anomalia/ranking (PR-AUC, Precision@k), nao classificacao balanceada. Dessas 188,
  **148 sao rotulo direto e 41 via socio comum** (risco de circularidade — ver acima).
  *(pesquisa conduzida sem orientador formal — ver nota na secao inicial do plano)*
- **`processos_judiciais` e ruidoso, nao e rotulo**: casado por nome via DJEN
  (`match_confianca='nome'`), nao por CNPJ direto; pipeline ainda em andamento no
  repo do dataset (110.489 registros / 2.374 empresas casadas at 08/08/2026).
- **Neo4j vs. PyTorch Geometric**: Neo4j para exploracao/validacao/visualizacao
  (Cypher, GDS, figuras); PyTorch Geometric para o treino da GNN em si — Neo4j nao
  treina modelo.
- **Infraestrutura**: Neo4j hospedado numa VPS pessoal (ja orquestrada via
  Coolify) — treino da GNN e notebooks ficam na maquina local (VPS e
  CPU-only/ARM64, sem GPU). Segredos de acesso nunca vao neste repositorio.
- **Metodologia**: validacao cruzada estratificada (nao split temporal simples, dado
  o N pequeno de positivos), extracao de metapath via matriz esparsa, baselines
  tabular/GNN homogenea/HAN-HGT, multiplas seeds + teste estatistico.
- **Venues-alvo**: BRACIS/SBBD como rede de seguranca; periodicos como *Expert
  Systems with Applications*, *Knowledge-Based Systems* ou *Decision Support
  Systems* como alvo principal, compativel com o prazo do mestrado.

## Etapas do trabalho

| # | Etapa | Onde | Status |
|---|---|---|---|
| 1 | ETL do dataset (RFB, JUCEES, CEIS/CNEP, TCEES, PGFN, IBAMA, TSE, DJEN) | repo [`projeto_grande_vitoria_empresas`](https://github.com/brunokobi/projeto_grande_vitoria_empresas) | ✅ nucleo pronto (`cnpj`, `jucees`, `sancoes`, `dividas_ativas`, `ibama`); ⏸️ `djen` (processos judiciais) e `geo` ainda rodando em background, nao bloqueiam as proximas etapas |
| 2 | Definir pergunta de pesquisa, rotulo e metodologia | [`docs/research_plan.md`](docs/research_plan.md) | ✅ travado (com o achado de circularidade do item acima) |
| 3 | Adaptar `settings.py` / `loaders.py` ao schema real (`empresas`, `socios`, `sancoes_administrativas`, `dividas_ativas`, `vinculos_politicos`) | `src/config`, `src/data/loaders.py` (`GrandeVitoriaLoader`) | ✅ primeira versao — inclui `rotulo_sancao()` (y_direto/y_qualquer, respeitando o risco de circularidade) |
| 4 | Construir a HIN real em `HeteroData` a partir das tabelas do banco | `src/graph/build_hin.py` (`build_empresas_hin`) | ✅ primeira versao — nos `empresa`/`socio`/`endereco`/`municipio`/`vinculo_politico`; `processos_judiciais` ainda de fora (pipeline `djen` em andamento) |
| 5 | Extracao de metapath via produto de matriz esparsa (escala para 344k empresas) | `src/graph/metapaths.py` (`SparseMetaPathExtractor`) | ✅ feito e **validado contra o banco real**: HIN completa (344.130 empresas) construida em 23,5s / 0,44 GB; os 3 metapaths de hipotese extraidos em <0,2s cada. Dois bugs reais encontrados e corrigidos so ao rodar contra dado de verdade — ver `scripts/validar_hin_real.py` e a nota abaixo |
| 6 | Exportar a HIN para Neo4j (exploracao Cypher/GDS, figuras da dissertacao) | `src/graph/neo4j_export.py` + VPS pessoal | ✅ feito — Neo4j 5 + GDS rodando na VPS; HIN real exportada (344k+ nos). Exposto publicamente via HTTPS (11/08/2026, decisao explicita do pesquisador) — ver secao abaixo |
| 7.1 | Feature engineering tabular | `src/features/tabular.py` (`build_feature_matrix`) | ✅ feito e **validado contra o banco real**: 344.130 empresas, 124 colunas, 0 `NaN` — inclui infração ambiental, contratos públicos, renúncia/benefício fiscal (10/08/2026) e features de grafo explícitas + risco de corrupção baseadas em literatura (11/08/2026, ver nota abaixo) |
| 7.2 | Harness de avaliacao (PR-AUC, Precision@k, CV estratificada repetida, seeds+Wilcoxon) | `src/evaluation/harness.py` | ✅ feito — generico para os 3 baselines (so precisa de uma funcao `fit_predict`) |
| 7.3 | Baseline tabular (XGBoost + class weighting) | `src/models/tabular_baseline.py` | ✅ feito e **rodado contra o banco real** — primeiro resultado quantitativo: PR-AUC ~18,8× a taxa-base em `y_direto` (ver nota abaixo) |
| 7.4 | Baseline GNN homogenea (via `SparseMetaPathExtractor`) | `src/models/gnn_homogeneous.py` | ✅ feito e rodado — **pior que o tabular em `y_direto`** (13,0x vs 18,8x de lift), ver nota abaixo |
| 7.5 | HAN/HGT (heterogenea de verdade) | `src/models/han_hgt.py` | ✅ feito e rodado — recupera quase todo o sinal perdido na GNN homogenea (18,2x vs 13,0x em `y_direto`), quase empata com o tabular |
| 7.6 | Comparacao estatistica dos 3 modelos (resultado do Marco 1) | `scripts/comparar_baselines.py` | ✅ **finalizado** — rerodado 4x (30 folds, 107→117→124 colunas + busca de hiperparametros do HAN/HGT) — **HAN/HGT e significativamente PIOR que o tabular em `y_direto` nas 4 rodadas, inclusive apos tuning dedicado** (p=0,007 / p<0,0001 / p=0,0001 / p=0,0024), ver nota abaixo |
| 7.7 | Analise de sensibilidade: rotulo `direto` (148) vs. `direto+socio` (188) | — | ⏳ pendente — decidido, nao implementado |
| 8 | Publicacao: resultado parcial em workshop/BRACIS, depois artigo principal em periodico-alvo | `docs/manuscrito/paper_en.md` (GIQ), `dissertacao_pt.md` | ✅ rascunho completo (Introducao->Conclusao+Referencias+2 figuras), revisado como parecer simulado de revisor experiente + ablation de features + ablation de tipo de no no HGT (15/08/2026) — falta so revisao de tom pelo pesquisador antes de submeter |

Cronograma por marcos (Marco 1–4) e riscos declarados: ver secoes 9 e 11 de
[`docs/research_plan.md`](docs/research_plan.md).

**Primeiro resultado quantitativo (baseline tabular, 08/08/2026)**: PR-AUC
0,0081 (`y_direto`) / 0,0085 (`y_qualquer`) — parecem baixos, mas a taxa-base
e 0,043%/0,055%, entao isso e **~18,8x e ~15,5x melhor que o acaso**,
respectivamente. O modelo tabular tem lift *menor* em `y_qualquer` (as
empresas que so sao suspeitas por sócio comum) — consistente com a hipotese
da tese: dado tabular isolado nao enxerga risco por associacao. Ver
`scripts/rodar_baseline_tabular.py`.

**Segundo resultado (GNN homogenea, 08/08/2026)** — reportado sem maquiar,
mesmo nao sendo a narrativa esperada: PR-AUC 0,0056 (`y_direto`, lift
~13,0x) / 0,0101 (`y_qualquer`, lift ~18,5x). **No rotulo principal
(`y_direto`), a GNN homogenea ficou pior que o tabular** (13,0x vs 18,8x).
Ja em `y_qualquer` ela ganha do tabular — mas isso e quase esperado, nao uma
vitoria limpa: os 40 positivos extras de `y_qualquer` foram *rotulados* via
socio comum, e a GNN usa exatamente essa aresta, entao tem vantagem "de
dentro" pra achar esses casos especificos (reforca por que `y_direto` e o
rotulo principal, nao `y_qualquer`). **Limitacao a corrigir antes da
comparacao estatistica (7.6)**: o tabular rodou com 50 folds e a GNN so com
10 (custo computacional -- ~29min pra 10 folds, 50 epochs; escalar pra 50
folds levaria ~2h30) -- ainda nao da pra rodar Wilcoxon comparando os dois,
precisa padronizar o numero de folds primeiro. Ver
`scripts/rodar_baseline_gnn_homogenea.py`.

**Terceiro resultado (HAN/HGT, 08/08/2026)**: PR-AUC 0,0078 (`y_direto`,
lift ~18,2x) / 0,0208 (`y_qualquer`, lift ~38,1x). Tratar cada metapath
como relacao distinta (em vez de colapsar num grafo homogeneo) **recupera
quase todo o sinal perdido na 7.4** (18,2x vs 13,0x) e chega quase no
empate com o tabular no rotulo principal (18,2x vs 18,8x, dentro da margem
de ruido com so 5 folds). Em `y_qualquer` o salto e grande (38,1x) — mas
amplifica a confusao de circularidade ja registrada: modelar a relacao
socio-empresa explicitamente da vantagem "de dentro" enorme pra achar
justamente os casos rotulados via essa mesma relacao. **Achado de
infraestrutura**: a configuracao original (hidden_channels=64, 2 cabecas
de atencao, incluindo `municipio`) estourou memoria (OOM killer) na
maquina local de desenvolvimento (7,8 GB de RAM — bem menos que a VPS);
resolvido reduzindo dimensoes e excluindo `municipio` (que ja nao era
metapath de hipotese) — pico de memoria caiu para ~5 GB, estavel entre
folds (nao e vazamento). Ver `scripts/rodar_baseline_han_hgt.py`.

**Comparacao estatistica final (etapa 7.6)**: primeira rodada (5 folds,
08/08/2026) foi inconclusiva por baixo poder estatistico (Wilcoxon p entre
0,62 e 1,00 em todos os 6 pares) — nao usada como resultado final.
**Escalada para 30 folds (5x6), rodada 09-10/08/2026, ~8h45**: agora com
poder estatistico real, o resultado e claro — ver
`docs/resultados/comparar_baselines_30folds_2026-08-10.log`:

| Rotulo | Tabular | GNN homogenea | HAN/HGT |
|---|---|---|---|
| `y_direto` (principal) | 18,6x | 16,5x | **14,2x** |
| `y_qualquer` (confundido) | 15,7x | 24,4x | **33,5x** |

**No rotulo principal, o HAN/HGT e estatisticamente PIOR que o tabular**
(Wilcoxon p=0,0066) **e pior que a GNN homogenea** (p=0,0293) — com so 5
folds isso parecia empate (ruido mascarando o efeito real). Em
`y_qualquer` os dois modelos de rede vencem o tabular (p=0,002 e p=0,036)
— mas esse rotulo e o confundido por circularidade, nao e evidencia limpa
a favor da hipotese.

**Isso e um resultado negativo genuino pra pergunta de pesquisa central**:
nesta implementacao, GNN heterogenea nao supera o baseline tabular na
deteccao de sancao direta — pelo contrario, o modelo mais sofisticado
piora. Nao e erro de codigo (o bug de reprodutibilidade da secao anterior
ja foi corrigido antes desta rodada). Pode ser resultado real, ou artefato
de hiperparametros/epocas de primeira versao — decisao de investir em
tuning ou reportar como esta em aberto com o pesquisador.

**Resultado v2, com as 4 features novas da etapa 7.1 (117 colunas), 30
folds, 11/08/2026, ~7h10** — ver
`docs/resultados/comparar_baselines_30folds_v2_2026-08-11.log`:

| Rotulo | Tabular | GNN homogenea | HAN/HGT |
|---|---|---|---|
| `y_direto` (principal) | 75,8x | **81,2x** | 29,3x |
| `y_qualquer` | **62,3x** | 41,6x | 45,4x |

Todos os 3 modelos saltaram ~4x em PR-AUC absoluto — as features novas
(infracao ambiental, contratos publicos, beneficios fiscais) carregam
sinal real e forte, nao ruido. **A conclusao central fica mais forte, nao
mudou de direcao**: no rotulo principal, HAN/HGT continua estatisticamente
PIOR que tabular e GNN homogenea, agora com `p<0,0001` nos dois casos
(antes 0,0066/0,0293) — o efeito ficou mais claro, nao mais fraco, com
features melhores. Mudanca qualitativa em `y_qualquer`: com as features
novas, **nenhuma diferenca e significativa** ali (antes os modelos de rede
venciam por causa da circularidade) — o tabular deixou de perder pra rede
mesmo no rotulo confundido. Reforca, com evidencia mais forte, que a
hipotese central nao se confirma nesta implementacao.

**Bug de reprodutibilidade encontrado e corrigido nesta etapa**:
`torch.manual_seed(random_state)` era chamado so na construcao da fabrica
`make_*_fit_predict`, nao a cada fold -- rodar outro modelo torch antes no
mesmo processo consumia o RNG global e mudava o resultado, mesmo com o
mesmo `random_state`. Corrigido movendo a chamada pra dentro do
`fit_predict` (mesma seed a cada fold, isola variancia de inicializacao
da variancia dos dados). Tem teste de regressao que "suja" o RNG global de
proposito antes de comparar. Ver `src/models/gnn_homogeneous.py` e
`src/models/han_hgt.py`.

**Features novas adicionadas (10/08/2026)**: infração ambiental (IBAMA/IEMA,
match direto por CNPJ), contratos com órgãos públicos, renúncia fiscal
federal e habilitação a benefício fiscal, e imune/isento de IRPJ — 4
cruzamentos válidos identificados a partir do dashboard do dataset,
conferidos linha a linha contra o banco antes de implementar (2 candidatos
adicionais, contrato via PNCP e marca registrada no INPI, ainda não têm
tabela populada no banco — não usáveis agora). Matriz foi de 107 para 117
colunas.

**Segunda rodada de features novas (11/08/2026), baseadas em literatura**:
pesquisado antes de implementar (corrupção em compras públicas, deteção de
shell company, grafo-features-vs-GNN em fraude — ver `docs/research_plan.md`
seção 7 para as referências completas). 5 features novas:
- `grau_socio_comum` / `grau_endereco_comum` / `grau_vinculo_politico_comum`
  — grau da empresa em cada metapath de hipótese, calculado direto da HIN
  (feature de grafo *explícita* pro tabular, em vez de só deixar a GNN
  aprender a estrutura implicitamente).
- `grau_do_socio` — concentração do sócio mais conectado da empresa.
- `tem_contrato_sem_competicao` / `sobrepreco_contrato_max` — red flags
  clássicos de risco de corrupção em compras públicas (cobertura baixa: só
  5 empresas têm modalidade sem competição no banco real).
- `idade_empresa_anos` — indicador de shell company (25,7% de cobertura via
  `registros_jucees`; sentinela `-1` para empresa sem registro).

Matriz: 117 → 124 colunas. **Teste rápido (11/08/2026, só tabular, 50 folds)
confirmou ganho real antes de rodar o experimento completo**: PR-AUC
`y_direto` 0,0242 (lift **~56×**, era 18,8× com as features antigas);
`y_qualquer` 0,0218 (lift **~40×**, era 15,5×). Ver
`scripts/rodar_baseline_tabular.py`.

**Resultado v3, com as 5 features de literatura (124 colunas), 30 folds,
12/08/2026, ~7h11 (concluído sem interrupção, protegido por checkpoint por
etapa após 3 reboots terem perdido rodadas anteriores)** — ver
`docs/resultados/comparar_baselines_30folds_v3_2026-08-12.log`:

| Rotulo | Tabular | GNN homogenea | HAN/HGT |
|---|---|---|---|
| `y_direto` (principal) | 50,7x | **76,3x** | 23,5x |
| `y_qualquer` | 43,2x | 40,1x | 42,8x |

**Resultado contraintuitivo, reportado sem maquiar**: o PR-AUC absoluto de
**todos os 3 modelos caiu** em relação ao v2 (ex.: tabular em `y_direto`
0,0326→0,0218; GNN homogênea 0,0349→0,0328; HAN/HGT 0,0126→0,0101) —
apesar de as 5 features novas terem sido validadas com um teste rápido
antes de comprometer o experimento completo. **Esse teste rápido comparou
contra a baseline errada**: comparou o ganho contra o resultado v1 original
(18,8x, antes de qualquer feature de dashboard) em vez do v2 mais recente
(75,8x, já com as 4 features de dashboard) — dando a falsa impressão de
melhora quando na verdade, frente ao v2, o resultado caiu. Fica registrado
como lição: sempre comparar contra o *último* resultado completo, não
contra o primeiro.

Dito isso, a **conclusão central da tese continua de pé, e mais um achado
novo apareceu**:

- `y_direto` (rótulo principal): HAN/HGT continua **estatisticamente PIOR**
  que os outros dois (p=0,0001 vs. tabular, p<0,0001 vs. GNN homogênea) —
  quarta rodada seguida confirmando esse resultado negativo central.
- **Achado novo em `y_direto`**: pela primeira vez, a GNN homogênea é
  **estatisticamente melhor que o tabular** (p=0,0145) — em v1 (p=0,38) e
  v2 (p=0,72) essa diferença nunca foi significativa. Ranking completo e
  significativo em todos os pares: homogênea > tabular > HAN/HGT.
- `y_qualquer` (confundido por circularidade): **nenhuma diferença é
  significativa** entre os 3 modelos (p=0,84 / 0,87 / 0,89) — mesma
  conclusão qualitativa do v2, mas agora os 3 modelos convergem para
  praticamente o mesmo lift (~40-43x) em vez de o tabular vencer com
  margem. Consistente com a leitura de que dar sinal de grafo explícito
  (`grau_socio_comum` etc.) ao tabular reduz a vantagem que os modelos de
  rede tinham nesse rótulo especificamente — a literatura de
  graph-features-vs-GNN prevê exatamente esse tipo de "empate técnico"
  quando o sinal estrutural já está disponível como feature.

**Interpretação de conjunto (v1→v2→v3)**: a hipótese central da tese — GNN
heterogênea (HAN/HGT) supera baseline tabular na detecção de sanção direta
— **não se confirma em nenhuma das três rodadas**, e o efeito negativo é
estatisticamente robusto (não é ruído de poucos folds). O que mudou com as
features de literatura não foi essa conclusão central, e sim uma nuance
nova: a GNN *homogênea* (mais simples, sem tipagem de relação) passou a
vencer o tabular de forma significativa — sugerindo que o ganho de ter
*algum* sinal de grafo (mesmo colapsado) supera o tabular puro, mas
tipar explicitamente cada relação (HAN/HGT) piora em vez de ajudar, mesmo
com features melhores.

**Busca de hiperparâmetros do HAN/HGT (12-13/08/2026)**: antes de aceitar o
v3 como definitivo, testamos se o resultado era artefato de configuração
subótima — os defaults (`hidden=32`, `heads=1`, `epochs=50`) tinham sido
reduzidos por causa de OOM na máquina local, não por ajuste. 6 candidatos
testados (5 folds, só `y_direto`): **o modelo estava subtreinado, não
subdimensionado** — só aumentar épocas (50→150) quase triplicou o PR-AUC
(0,0105→0,0244); aumentar `hidden_channels` isoladamente não ajudou;
`hidden=64+heads=2+epochs=100` deu OOM de novo. Config final escolhida:
`epochs=150` isolado (o candidato com `heads=2` empatava dentro do ruído
por ~3x mais custo computacional — preferida a opção mais simples).

**Resultado final v4, com o HAN/HGT tunado (124 colunas, 30 folds,
14/08/2026, ~15h52 corridas)** — ver
`docs/resultados/comparar_baselines_30folds_v4_han_hgt_tunado_2026-08-14.log`:

| Rotulo | Tabular | GNN homogenea | HAN/HGT (tunado) |
|---|---|---|---|
| `y_direto` (principal) | 50,7x | **76,3x** | 32,6x |
| `y_qualquer` | 43,2x | 40,1x | **60,8x** |

**O tuning funcionou de verdade — e o resultado negativo central sobreviveu
mesmo assim.** Em `y_direto`, o HAN/HGT melhorou 39% em relação ao v3
(23,5x→32,6x), confirmando que o achado do subtreinamento era real. **Mas
continua estatisticamente PIOR que tabular (p=0,0024) e que GNN homogênea
(p<0,0001)** — a defesa mais forte possível contra "não tentaram treinar
direito" foi feita, e o efeito negativo se sustenta. Essa é a quinta rodada
completa (107→117→124 colunas + tuning dedicado) confirmando a mesma
direção no rótulo principal.

**Mudança qualitativa em `y_qualquer`**: o HAN/HGT tunado agora **vence** a
GNN homogênea (p=0,0277) e quase empata com o tabular (p=0,158, não
significativo, mas o lift saltou de 42,8x pra 60,8x). Interpretação: mais
épocas de treino deram ao HAN/HGT mais capacidade de explorar exatamente a
relação sócio-empresa que é o mecanismo de circularidade desse rótulo (as
40 empresas rotuladas via sócio comum, não sanção direta — seção 5 do
`docs/research_plan.md`) — reforça, não contradiz, a leitura de que o
"sucesso" do HAN/HGT em `y_qualquer` é sobre explorar melhor o vazamento de
rótulo, não descobrir sinal novo.

**Conclusão prática**: com a busca de hiperparâmetros feita e documentada,
o resultado negativo do HAN/HGT no rótulo principal não é mais uma lacuna
de rigor a fechar — é a conclusão final da comparação empírica. Próximo
passo: etapa 7.7 (sensibilidade `y_direto`/`y_qualquer`, agora com essa
leitura mais precisa da circularidade) e etapa 8 (escrita/publicação — ver
`docs/manuscrito/`).

**Dois bugs reais encontrados só ao validar contra o banco de verdade** (nenhum
aparecia no dado sintético dos testes — registrado para não repetir):

1. `pandas.read_sql_query` devolve `NaN` (float) para coluna de texto nula, não
   `None`/string vazia — `nan or ""` não pega isso (`NaN` é *truthy* em Python).
   Quebrava `_chave_socio`/`_normalizar_texto` em `build_hin.py`.
2. `municipio` é um nó "hub" de baixíssima cardinalidade (7 nós para 344k
   empresas) — o produto esparso do metapath `empresa_municipio_empresa`
   tentou alocar **187 GiB**. Corrigido com `MetapathExplosionError`: estima o
   tamanho do produto antes de calcular e recusa com mensagem clara em vez de
   estourar memória. A própria estimativa tinha um bug de overflow silencioso
   em `int32` (`np.dot` sem cast para `int64`) — só apareceu com a distribuição
   real e desigual dos municípios, não com números sintéticos uniformes.

## Estrutura do projeto

```
.
├── CLAUDE.md             # resumo das decisoes travadas, para retomar contexto em qualquer maquina
├── .github/workflows/
│   └── ci.yml            # CI: lint (ruff) + testes (pytest) via uv
├── docs/
│   └── research_plan.md  # plano de pesquisa da dissertacao (pergunta, metodologia, cronograma)
├── data/
│   ├── raw/              # dados brutos (SQLite/Parquet do projeto original) — nao versionado
│   ├── processed/        # dados limpos/transformados — nao versionado
│   └── graph_exports/    # HIN serializada (.pt), exports para Neo4j etc. — nao versionado
├── notebooks/
│   └── 01_eda.ipynb      # EDA e inspecao visual da HIN
├── src/
│   ├── config/
│   │   └── settings.py   # Pydantic Settings (.env) — caminhos, Neo4j, parametros de memoria
│   ├── data/
│   │   └── loaders.py    # SQLiteLoader / ParquetLoader / GrandeVitoriaLoader (schema real)
│   ├── graph/
│   │   ├── hin_builder.py   # HINBuilder (HeteroData -> networkx -> Neo4j/export) — generico
│   │   ├── build_hin.py     # build_empresas_hin() — monta a HIN real a partir do dataset
│   │   └── metapaths.py     # MetaPath / MetaPathExtractor
│   └── tests/
│       ├── conftest.py           # fixtures: HIN sintetica + SQLite sintetico no schema real
│       ├── test_connectivity.py  # sem nos isolados, indices validos, simetria de arcos reversos
│       ├── test_memory.py        # pico de memoria na construcao vs. orcamento configurado
│       ├── test_quality.py       # NaN/Inf, duplicidade de arcos, schema minimo esperado
│       └── test_real_schema.py   # GrandeVitoriaLoader + build_empresas_hin (rotulo, metapaths reais)
├── pyproject.toml
├── .env.example
└── .gitignore
```

## Setup do ambiente

Este projeto usa [`uv`](https://docs.astral.sh/uv/) como gerenciador de pacotes (compativel
tambem com Poetry, ver secao `[tool.poetry]` do `pyproject.toml`).

```bash
# 1. Instalar dependencias (cria .venv automaticamente)
uv sync

# 2. Copiar e ajustar as variaveis de ambiente
cp .env.example .env
# edite .env com as credenciais reais do Neo4j e caminhos dos dados

# 3. Rodar os testes
uv run pytest

# 4. Abrir o notebook de EDA
uv run jupyter lab notebooks/01_eda.ipynb
```

Alternativa com Poetry:

```bash
poetry install
cp .env.example .env
poetry run pytest
```

## Configuracao (`src/config/settings.py`)

Todos os parametros sao lidos do `.env` via `pydantic-settings`:

| Variavel                  | Descricao                                              |
|----------------------------|----------------------------------------------------------|
| `DATA_RAW_DIR`             | diretorio dos dados brutos                                |
| `DATA_PROCESSED_DIR`       | diretorio dos dados processados                           |
| `DATA_GRAPH_EXPORTS_DIR`   | diretorio de exportacao da HIN (.pt)                       |
| `SQLITE_DB_PATH`           | caminho do banco SQLite de origem                          |
| `NEO4J_URI` / `NEO4J_USER` / `NEO4J_PASSWORD` / `NEO4J_DATABASE` | credenciais do Neo4j |
| `MAX_MEMORY_GB`            | orcamento de memoria monitorado em `test_memory.py`       |
| `NUM_WORKERS`, `RANDOM_SEED`, `TORCH_DEVICE`, `BATCH_SIZE` | parametros de execucao |

```python
from src.config import get_settings

settings = get_settings()
settings.ensure_data_dirs()
```

## Neo4j (exploracao/visualizacao — VPS pessoal)

Neo4j 5 + GDS rodam num container na VPS pessoal (Oracle Cloud, ja orquestrada via
Coolify). **Decisao revista em 11/08/2026**: inicialmente so acesso via tunel SSH
(nenhuma porta publica — o grafo carrega dado pessoal, nome de sócio/endereço); depois
exposto publicamente por decisao explicita do pesquisador (aceitando o risco: protegido
só por usuário/senha, sem 2FA/rate-limit), via Traefik com HTTPS automática
(Let's Encrypt):

- **Browser**: https://neo4j.brunokobi.duckdns.org
- **Bolt** (driver Python): `bolt+s://neo4j.brunokobi.duckdns.org:7687` — já é o
  default do `.env` real. Bolt tem TLS próprio (mesmo certificado Let's Encrypt do
  domínio) — necessário porque o Browser carrega via HTTPS e o navegador bloqueia
  Bolt sem TLS a partir de página segura ("mixed content"); `bolt://` sem TLS ainda
  funciona por compatibilidade, mas use `bolt+s://`.

```bash
# Exportar a HIN real para o Neo4j (idempotente — usa MERGE por id)
uv run python scripts/exportar_hin_neo4j.py

# Explorar via Neo4j Browser
#    https://neo4j.brunokobi.duckdns.org
```

Alternativa mais segura, ainda disponível: voltar ao túnel SSH (as portas do
container também aceitam conexão via `127.0.0.1` na VPS) e usar
`NEO4J_URI=bolt://localhost:7687`:

```bash
ssh -L 7687:localhost:7687 -L 7474:localhost:7474 -i ~/ssh-key-2026-07-18.key ubuntu@<vps>
```

Convenção de nomes na exportação (`src/graph/neo4j_export.py`): tipo de nó
`empresa` → label Cypher `Empresa`; relação `participa_de` → tipo de
relacionamento `PARTICIPA_DE`. Arcos reversos (`rev_*`) não são exportados —
um relacionamento Cypher já é navegável nos dois sentidos. Nenhuma feature
numérica de treino (`x`) vai para o Neo4j, só o rótulo (`sancionada_direto`/
`sancionada_qualquer`) como propriedade booleana, para poder filtrar/colorir
na exploração.

## Proximos passos sugeridos

Ver tambem a tabela [Etapas do trabalho](#etapas-do-trabalho) acima para o roteiro
completo. Tecnicamente, o proximo passo (etapa 3) e:

1. Implementar loaders especificos das tabelas reais (`empresas`, `socios`,
   `sancoes_administrativas`, `dividas_ativas`, `vinculos_politicos`) em
   `src/data/loaders.py`, no lugar do schema generico de exemplo.
2. Definir o schema completo de nos/relacoes do dominio real em `src/graph/hin_builder.py`.
3. Adicionar exportador para Neo4j (`neo4j` driver ja incluido nas dependencias).
4. Expandir `COMMON_METAPATHS` conforme as hipoteses da pesquisa (socio comum, endereco
   comum, vinculo politico) e reescrever a extracao para produto de matriz esparsa.
