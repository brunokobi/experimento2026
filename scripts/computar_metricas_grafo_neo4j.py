"""Calcula métricas estruturais do grafo (centralidade + comunidade) via
Neo4j GDS (Graph Data Science, já habilitado no Neo4j da VPS -- ver
docker-compose em /opt/coolify/apps/neo4j/) e escreve como propriedades nos
nós ``Empresa``, pro kobi-intelligence-hub consumir com um lookup simples
(sem precisar rodar GDS a cada request).

Roda DEPOIS de ``exportar_hin_neo4j.py`` (precisa do grafo já exportado) --
chamado por ``scripts/atualizar_neo4j_dinamico.sh`` logo em seguida, então
fica sempre em sincronia com o dataset publicado mais recente.

Três métricas, adicionadas em 12/09/2026 (ver conversa sobre "explorar ao
máximo a GNN/Neo4j"):

1. **Centralidade (PageRank + betweenness aproximado)** -- "essa empresa é
   um hub ou é periférica na rede?". PageRank é exato (barato, ~10s no
   grafo todo). Betweenness exato é inviável nesse tamanho de grafo
   (O(V*E) -- Brandes) -- usa a aproximação RA-Brandes do GDS
   (``samplingSize``), que já levou ~6-8min no teste real. Ambos viram
   também um PERCENTIL entre as Empresas (0-100), mais legível que o valor
   bruto do algoritmo.
2. **Comunidade (Louvain)** -- agrupa a rede inteira em clusters densos,
   sem precisar partir de uma empresa específica. Cada Empresa recebe o id
   da comunidade + o tamanho da comunidade + a % de sancionadas ali dentro
   -- permite ver "essa empresa pertence a um cluster que concentra
   sanções?", coisa que a visão 1-hop (rede_societaria.py) não enxerga.
3. Risco indireto multi-hop (2-3 saltos) é uma query Cypher direta, sem
   pré-cálculo -- ver ``kobi-intelligence-hub/src/dossie/rede_societaria.py``
   (mostrou ser rápida o suficiente, <2s até no pior caso testado, pra
   rodar ao vivo a cada request).

Índice em ``comunidade_louvain`` é OBRIGATÓRIO antes do passo de estatística
por comunidade -- sem ele, o MATCH de re-agrupamento faz table scan repetido
(testado: timeout >120s sem índice; com índice, roda em segundos).
"""
from __future__ import annotations

import time

from loguru import logger
from neo4j import GraphDatabase

from src.config import get_settings

GRAPH_NAME = "redeSocietaria"
BETWEENNESS_SAMPLING_SIZE = 2000
BETWEENNESS_SAMPLING_SEED = 42
BATCH_SIZE = 5000


def _driver(settings):
    return GraphDatabase.driver(
        settings.neo4j_uri, auth=(settings.neo4j_user, settings.neo4j_password.get_secret_value())
    )


def _projetar_grafo(session) -> None:
    existe = session.run(
        f"CALL gds.graph.exists('{GRAPH_NAME}') YIELD exists RETURN exists"
    ).single()["exists"]
    if existe:
        session.run(f"CALL gds.graph.drop('{GRAPH_NAME}')")
    session.run(f"""
        CALL gds.graph.project(
          '{GRAPH_NAME}',
          ['Empresa','Socio','Endereco','VinculoPolitico','Municipio'],
          {{
            PARTICIPA_DE: {{orientation: 'UNDIRECTED'}},
            SEDIADA_EM: {{orientation: 'UNDIRECTED'}},
            LOCALIZADA_EM: {{orientation: 'UNDIRECTED'}},
            TEM_VINCULO_POLITICO: {{orientation: 'UNDIRECTED'}}
          }}
        )
    """)


def _computar_percentis(session, propriedade: str, propriedade_percentil: str) -> None:
    """Percentil (0-100, maior = mais central) entre as Empresas com a
    propriedade calculada -- lido em memória (só cnpj + valor, ~350k floats,
    leve) e escrito de volta em lotes (mesmo padrão de ``neo4j_export.py``)."""
    valores = session.run(
        f"MATCH (e:Empresa) WHERE e.{propriedade} IS NOT NULL RETURN e.id AS id, e.{propriedade} AS v"
    ).data()
    valores.sort(key=lambda r: r["v"], reverse=True)
    n = len(valores)
    linhas = [{"id": r["id"], "p": round(100.0 * (n - i) / n, 2)} for i, r in enumerate(valores)]
    for inicio in range(0, n, BATCH_SIZE):
        lote = linhas[inicio:inicio + BATCH_SIZE]
        session.run(
            f"UNWIND $rows AS row MATCH (e:Empresa {{id: row.id}}) SET e.{propriedade_percentil} = row.p",
            rows=lote,
        )
    logger.info(f"Percentil de {propriedade} escrito em {n} empresas.")


def main() -> None:
    settings = get_settings()
    driver = _driver(settings)
    try:
        with driver.session(database=settings.neo4j_database) as session:
            logger.info("Projetando grafo pro GDS...")
            t0 = time.perf_counter()
            _projetar_grafo(session)
            logger.info(f"Grafo projetado em {time.perf_counter() - t0:.1f}s.")

            logger.info("PageRank (centralidade)...")
            t0 = time.perf_counter()
            r = session.run(f"""
                CALL gds.pageRank.write('{GRAPH_NAME}', {{writeProperty: 'centralidade_pagerank'}})
                YIELD nodePropertiesWritten, ranIterations
                RETURN nodePropertiesWritten, ranIterations
            """).single()
            logger.info(f"PageRank: {dict(r)} em {time.perf_counter() - t0:.1f}s.")

            logger.info("Louvain (comunidade)...")
            t0 = time.perf_counter()
            r = session.run(f"""
                CALL gds.louvain.write('{GRAPH_NAME}', {{writeProperty: 'comunidade_louvain'}})
                YIELD communityCount, modularity
                RETURN communityCount, modularity
            """).single()
            logger.info(f"Louvain: {dict(r)} em {time.perf_counter() - t0:.1f}s.")

            logger.info(f"Betweenness aproximado (samplingSize={BETWEENNESS_SAMPLING_SIZE})... isso demora minutos.")
            t0 = time.perf_counter()
            r = session.run(f"""
                CALL gds.betweenness.write('{GRAPH_NAME}', {{
                  writeProperty: 'centralidade_betweenness',
                  samplingSize: {BETWEENNESS_SAMPLING_SIZE},
                  samplingSeed: {BETWEENNESS_SAMPLING_SEED}
                }})
                YIELD nodePropertiesWritten
                RETURN nodePropertiesWritten
            """).single()
            logger.info(f"Betweenness: {dict(r)} em {time.perf_counter() - t0:.1f}s.")

            logger.info("Calculando percentis (PageRank e betweenness)...")
            _computar_percentis(session, "centralidade_pagerank", "centralidade_pagerank_percentil")
            _computar_percentis(session, "centralidade_betweenness", "centralidade_betweenness_percentil")

            logger.info("Índice em comunidade_louvain (necessário antes da agregação por comunidade)...")
            session.run("CREATE INDEX IF NOT EXISTS FOR (e:Empresa) ON (e.comunidade_louvain)")

            logger.info("Estatísticas por comunidade (tamanho + % sancionada)...")
            t0 = time.perf_counter()
            r = session.run("""
                MATCH (e:Empresa)
                WITH e.comunidade_louvain AS com, count(*) AS tamanho,
                     sum(CASE WHEN e.sancionada_direto THEN 1 ELSE 0 END) AS sancionadas
                MATCH (e2:Empresa {comunidade_louvain: com})
                SET e2.comunidade_tamanho = tamanho,
                    e2.comunidade_pct_sancionada = round(100.0 * sancionadas / tamanho, 2)
                RETURN count(DISTINCT com) AS comunidades, count(e2) AS nos_atualizados
            """).single()
            logger.info(f"Comunidades: {dict(r)} em {time.perf_counter() - t0:.1f}s.")

            logger.info("Limpando projeção GDS (só ocupa memória em memória, não precisa persistir)...")
            session.run(f"CALL gds.graph.drop('{GRAPH_NAME}')")
    finally:
        driver.close()
    logger.info("Métricas de grafo calculadas e persistidas no Neo4j.")


if __name__ == "__main__":
    main()
