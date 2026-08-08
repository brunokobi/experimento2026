"""Script manual: constroi a HIN real e exporta para o Neo4j (exploracao
Cypher/GDS, figuras da dissertacao) -- fora da suite de testes, ja que
depende do banco real e de um Neo4j de verdade acessivel via tunel SSH.

Pre-requisito: abrir o tunel antes de rodar (Neo4j na VPS pessoal nao expoe
porta publica -- decisao de seguranca, ver src/graph/neo4j_export.py)::

    ssh -L 7687:localhost:7687 -L 7474:localhost:7474 \\
        -i ~/ssh-key-2026-07-18.key ubuntu@<vps>

Uso:
    uv run python scripts/exportar_hin_neo4j.py
"""

from __future__ import annotations

import time

from loguru import logger

from src.data.loaders import GrandeVitoriaLoader
from src.graph.build_hin import build_empresas_hin
from src.graph.neo4j_export import export_hin_to_neo4j


def main() -> None:
    logger.info("Construindo a HIN a partir do banco real...")
    t0 = time.perf_counter()
    builder = build_empresas_hin(GrandeVitoriaLoader())
    builder.build()
    logger.info(f"HIN construida em {time.perf_counter() - t0:.1f}s: {builder.stats()}")

    logger.info("Exportando para o Neo4j (tunel SSH deve estar aberto)...")
    t0 = time.perf_counter()
    export_hin_to_neo4j(builder)
    logger.info(f"Export concluido em {time.perf_counter() - t0:.1f}s.")


if __name__ == "__main__":
    main()
