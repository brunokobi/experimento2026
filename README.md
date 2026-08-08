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
| 6 | Exportar a HIN para Neo4j (exploracao Cypher/GDS, figuras da dissertacao) | `src/graph/neo4j_export.py` + VPS pessoal | ✅ feito — Neo4j 5 + GDS rodando na VPS, **acesso so via tunel SSH** (nunca porta publica: o grafo tem dado pessoal); HIN real exportada (344k+ nos) |
| 7.1 | Feature engineering tabular | `src/features/tabular.py` (`build_feature_matrix`) | ✅ feito e **validado contra o banco real**: 344.130 empresas, 5,5s, 107 colunas, 0 `NaN` |
| 7.2 | Harness de avaliacao (PR-AUC, Precision@k, CV estratificada repetida, seeds+Wilcoxon) | `src/evaluation/harness.py` | ✅ feito — generico para os 3 baselines (so precisa de uma funcao `fit_predict`) |
| 7.3 | Baseline tabular (XGBoost + class weighting) | `src/models/tabular_baseline.py` | ✅ feito e **rodado contra o banco real** — primeiro resultado quantitativo: PR-AUC ~18,8× a taxa-base em `y_direto` (ver nota abaixo) |
| 7.4 | Baseline GNN homogenea (via `SparseMetaPathExtractor`) | `src/models/gnn_homogeneous.py` | ✅ feito e rodado — **pior que o tabular em `y_direto`** (13,0x vs 18,8x de lift), ver nota abaixo |
| 7.5 | HAN/HGT (heterogenea de verdade) | — | ⏳ pendente |
| 7.6 | Comparacao estatistica dos 3 modelos (resultado do Marco 1) | — | ⏳ pendente |
| 7.7 | Analise de sensibilidade: rotulo `direto` (148) vs. `direto+socio` (188) | — | ⏳ pendente — decidido, nao implementado |
| 8 | Publicacao: resultado parcial em workshop/BRACIS, depois artigo principal em periodico-alvo | — | ⏳ pendente |

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
Coolify) — **acesso exclusivamente via tunel SSH**, nenhuma porta exposta publicamente
(decisao de seguranca: o grafo carrega dado pessoal — nome de sócio, endereço). Sem o
túnel aberto, `NEO4J_URI=bolt://localhost:7687` do `.env` não conecta em lugar nenhum.

```bash
# 1. Abrir o tunel (mantenha rodando em outro terminal enquanto for usar o Neo4j)
ssh -L 7687:localhost:7687 -L 7474:localhost:7474 \
    -i ~/ssh-key-2026-07-18.key ubuntu@<vps>

# 2. Exportar a HIN real para o Neo4j (idempotente — usa MERGE por id)
uv run python scripts/exportar_hin_neo4j.py

# 3. Explorar via Neo4j Browser (com o tunel aberto)
#    http://localhost:7474
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
