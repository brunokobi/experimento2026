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
Informacao (HIN)** representando empresas, socios, CNAEs e municipios, com foco em
extracao de **metapaths** e preparacao para modelos de GNN (via PyTorch Geometric).

## Estrutura do projeto

```
.
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
│   │   └── loaders.py    # SQLiteLoader / ParquetLoader
│   ├── graph/
│   │   ├── hin_builder.py   # HINBuilder (HeteroData -> networkx -> Neo4j/export)
│   │   └── metapaths.py     # MetaPath / MetaPathExtractor
│   └── tests/
│       ├── conftest.py           # fixture de HIN sintetica
│       ├── test_connectivity.py  # sem nos isolados, indices validos, simetria de arcos reversos
│       ├── test_memory.py        # pico de memoria na construcao vs. orcamento configurado
│       └── test_quality.py       # NaN/Inf, duplicidade de arcos, schema minimo esperado
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

1. Implementar loaders especificos das tabelas reais do projeto original em `src/data/loaders.py`.
2. Definir o schema completo de nos/relacoes do dominio em `src/graph/hin_builder.py`.
3. Adicionar exportador para Neo4j (`neo4j` driver ja incluido nas dependencias).
4. Expandir `COMMON_METAPATHS` conforme as hipoteses da pesquisa.
