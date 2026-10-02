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
from neo4j import GraphDatabase

from src.config import get_settings
from src.data.loaders import GrandeVitoriaLoader
from src.graph.build_hin import build_empresas_hin
from src.graph.neo4j_export import (
    export_detalhes_processos_judiciais,
    export_detalhes_vinculos_politicos,
    export_hin_to_neo4j,
)


def main() -> None:
    loader = GrandeVitoriaLoader()

    logger.info("Construindo a HIN a partir do banco real...")
    t0 = time.perf_counter()
    builder = build_empresas_hin(loader)
    builder.build()
    logger.info(f"HIN construida em {time.perf_counter() - t0:.1f}s: {builder.stats()}")

    logger.info("Exportando para o Neo4j (tunel SSH deve estar aberto, se aplicavel)...")
    t0 = time.perf_counter()
    export_hin_to_neo4j(builder)
    logger.info(f"Export concluido em {time.perf_counter() - t0:.1f}s.")

    # Enriquecimento de propriedades (fonte, ano, valor, cargo, partido,
    # tribunal/classe/status etc.) -- fora do HeteroData de proposito, ver
    # docstring de export_detalhes_vinculos_politicos.
    logger.info("Enriquecendo vinculos politicos e processos judiciais com propriedades...")
    t0 = time.perf_counter()
    settings = get_settings()
    driver = GraphDatabase.driver(
        settings.neo4j_uri, auth=(settings.neo4j_user, settings.neo4j_password.get_secret_value())
    )
    try:
        export_detalhes_vinculos_politicos(
            driver, settings.neo4j_database, loader.vinculos_politicos(), loader.candidatos_perfil()
        )
        export_detalhes_processos_judiciais(
            driver, settings.neo4j_database, loader.processos_judiciais(match_confianca="nome")
        )
    finally:
        driver.close()
    logger.info(f"Enriquecimento concluido em {time.perf_counter() - t0:.1f}s.")


if __name__ == "__main__":
    main()
