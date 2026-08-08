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
| 5 | Reescrever extracao de metapath para produto de matriz esparsa (o DFS atual em `networkx` nao escala para 344k empresas) | `src/graph/metapaths.py` | ⏳ pendente — `COMMON_METAPATHS` ja atualizado para os nos/relacoes reais (`empresa_socio_empresa`, `empresa_endereco_empresa`, `empresa_vinculo_politico_empresa`), extracao ainda por DFS |
| 6 | Exportar a HIN para Neo4j (exploracao Cypher/GDS, figuras da dissertacao) | `src/graph/hin_builder.py` + VPS pessoal | ⏳ pendente |
| 7 | Baselines: tabular (XGBoost/LightGBM com class weighting), GNN homogenea, HAN/HGT | notebooks/ + novo modulo de treino | ⏳ pendente |
| 8 | Avaliacao: validacao cruzada estratificada repetida, PR-AUC/Precision@k, multiplas seeds + teste estatistico (Wilcoxon) | idem | ⏳ pendente |
| 9 | Analise de sensibilidade: rotulo `direto` (148) vs. `direto+socio` (188) | idem | ⏳ pendente — decidido, nao implementado |
| 10 | Publicacao: resultado parcial em workshop/BRACIS, depois artigo principal em periodico-alvo | — | ⏳ pendente |

Cronograma por marcos (Marco 1–4) e riscos declarados: ver secoes 9 e 11 de
[`docs/research_plan.md`](docs/research_plan.md).

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
