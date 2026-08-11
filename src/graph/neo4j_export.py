"""Exportacao da HIN para o Neo4j -- exploracao Cypher/GDS, figuras da
dissertacao. Neo4j **nao treina** o modelo, so explora/valida/visualiza (ver
docs/research_plan.md, secao 6); PyTorch Geometric continua sendo o motor de
treino de fato.

Acesso ao Neo4j da VPS pessoal (decisao revista em 11/08/2026, ver
``docs/research_plan.md`` secao 6): inicialmente so via tunel SSH (grafo tem
dado pessoal -- nome de socio, endereco); depois exposto publicamente por
decisao explicita do pesquisador (aceitando o risco), via Traefik com
HTTPS/Let's Encrypt:

    https://neo4j.brunokobi.duckdns.org   (Browser, HTTP/HTTPS via Traefik)
    bolt://neo4j.brunokobi.duckdns.org:7687   (Bolt, porta publicada direto)

Protegido so por usuario/senha (sem 2FA/rate-limit) -- ``NEO4J_URI`` no
``.env`` real ja aponta pro dominio publico. Alternativa mais segura, ainda
disponivel, e voltar ao tunel SSH quando quiser (as portas do container
tambem aceitam conexao via ``127.0.0.1`` na VPS)::

    ssh -L 7687:localhost:7687 -L 7474:localhost:7474 \\
        -i ~/ssh-key-2026-07-18.key ubuntu@<vps>
    # com o tunel aberto, usar NEO4J_URI=bolt://localhost:7687

Convencao de nomes: tipo de no ``empresa`` -> label Cypher ``Empresa``
(PascalCase); relacao ``participa_de`` -> tipo de relacionamento
``PARTICIPA_DE`` (UPPER_SNAKE_CASE) -- convencoes usuais do Neo4j. Arcos
reversos (``rev_*``, criados por ``HINBuilder.add_edge_type(bidirectional=True)``)
NAO sao exportados -- um relacionamento Cypher ja e navegavel nos dois
sentidos, exportar os dois duplicaria a aresta.

O rotulo (``y_direto``/``y_qualquer``) entra como propriedade booleana do no
``Empresa``, so para poder filtrar/colorir na exploracao -- nenhuma feature
numerica de treino (``x``) e exportada, o Neo4j aqui e para inspecao, nao
para alimentar o modelo.
"""

from __future__ import annotations

from typing import Any

from loguru import logger
from neo4j import Driver, GraphDatabase

from src.config import Settings, get_settings
from src.graph.hin_builder import HINBuilder


def _to_label(node_type: str) -> str:
    """``empresa`` -> ``Empresa``; ``vinculo_politico`` -> ``VinculoPolitico``."""
    return "".join(part.capitalize() for part in node_type.split("_"))


def _to_rel_type(relation: str) -> str:
    """``participa_de`` -> ``PARTICIPA_DE``."""
    return relation.upper()


def _extra_node_props(builder: HINBuilder, node_type: str) -> dict[str, list[Any]]:
    """Propriedades escalares extras por tipo de no -- hoje so o rotulo do no
    ``empresa`` (para filtrar/colorir na exploracao), nunca feature numerica
    de treino."""
    props: dict[str, list[Any]] = {}
    if node_type == "empresa":
        store = builder.data[node_type]
        if "y_direto" in store:
            props["sancionada_direto"] = store.y_direto.tolist()
        if "y_qualquer" in store:
            props["sancionada_qualquer"] = store.y_qualquer.tolist()
    return props


def _export_node_type(driver: Driver, database: str, builder: HINBuilder, node_type: str, batch_size: int) -> None:
    label = _to_label(node_type)
    ids = builder.external_ids(node_type)
    extra_props = _extra_node_props(builder, node_type)

    with driver.session(database=database) as session:
        session.run(f"CREATE CONSTRAINT IF NOT EXISTS FOR (n:{label}) REQUIRE n.id IS UNIQUE")

        for start in range(0, len(ids), batch_size):
            end = min(start + batch_size, len(ids))
            batch = []
            for i in range(start, end):
                row: dict[str, Any] = {"id": str(ids[i])}
                for prop_name, values in extra_props.items():
                    row[prop_name] = values[i]
                batch.append(row)
            session.run(f"UNWIND $rows AS row MERGE (n:{label} {{id: row.id}}) SET n += row", rows=batch)

    logger.info(f"Neo4j: {len(ids)} nos '{label}' exportados.")


def _export_edge_type(
    driver: Driver, database: str, builder: HINBuilder, edge_type: tuple[str, str, str], batch_size: int
) -> None:
    src_type, relation, dst_type = edge_type
    src_label, dst_label, rel_type = _to_label(src_type), _to_label(dst_type), _to_rel_type(relation)

    src_ids = builder.external_ids(src_type)
    dst_ids = builder.external_ids(dst_type)
    edge_index = builder.data[edge_type].edge_index
    rows_src, rows_dst = edge_index[0].tolist(), edge_index[1].tolist()
    total = len(rows_src)

    with driver.session(database=database) as session:
        for start in range(0, total, batch_size):
            end = min(start + batch_size, total)
            batch = [{"src": str(src_ids[rows_src[i]]), "dst": str(dst_ids[rows_dst[i]])} for i in range(start, end)]
            session.run(
                f"UNWIND $rows AS row "
                f"MATCH (a:{src_label} {{id: row.src}}), (b:{dst_label} {{id: row.dst}}) "
                f"MERGE (a)-[:{rel_type}]->(b)",
                rows=batch,
            )

    logger.info(f"Neo4j: {total} arcos '{src_label}-{rel_type}->{dst_label}' exportados.")


def export_hin_to_neo4j(builder: HINBuilder, settings: Settings | None = None, batch_size: int = 5000) -> None:
    """Exporta a HIN inteira (nos + arcos, sem os reversos ``rev_*``) para o
    Neo4j configurado em ``Settings`` -- ver docstring do modulo para a
    convencao de nomes e o motivo de precisar do tunel SSH.

    Idempotente: usa ``MERGE`` por ``id``, pode ser rodado de novo com
    seguranca (ex.: apos reconstruir a HIN com dado atualizado).
    """
    settings = settings or get_settings()
    driver = GraphDatabase.driver(
        settings.neo4j_uri, auth=(settings.neo4j_user, settings.neo4j_password.get_secret_value())
    )
    try:
        driver.verify_connectivity()
        for node_type in builder.data.node_types:
            _export_node_type(driver, settings.neo4j_database, builder, node_type, batch_size)
        for edge_type in builder.data.edge_types:
            _src, relation, _dst = edge_type
            if relation.startswith("rev_"):
                continue
            _export_edge_type(driver, settings.neo4j_database, builder, edge_type, batch_size)
    finally:
        driver.close()
    logger.info("Export para Neo4j concluido.")
