#!/bin/bash
# Atualiza o Neo4j "dinâmico" (neo4j.brunokobi.duckdns.org) com o dataset
# público mais recente do grande_vitoria_empresas — roda direto do terminal
# (cron ou manual), sem tunel SSH (NEO4J_URI já é o endpoint público
# bolt+s://, ver .env).
#
# NÃO mexe no snapshot congelado do artigo (ver docs/backups/README.md) —
# esse script só afeta o grafo AO VIVO, que a partir de 07/09/2026 deixou
# de ser um snapshot fixo por decisão do pesquisador.
#
# export_hin_to_neo4j (src/graph/neo4j_export.py) usa MERGE — idempotente,
# rodar de novo não duplica nó/aresta, só atualiza/adiciona.
set -euo pipefail
cd "$(dirname "$0")/.."

RELEASE_URL="https://github.com/brunokobi/projeto_grande_vitoria_empresas/releases/download/dataset-latest/grande_vitoria.db.gz"
DB_PATH="data/raw/grande_vitoria.db"
GZ_TMP="${DB_PATH}.gz.new"
DB_TMP="${DB_PATH}.new"

echo "$(date -Iseconds) [atualizar_neo4j] baixando dataset publicado mais recente ..."
curl -sL -o "$GZ_TMP" "$RELEASE_URL"
gunzip -c "$GZ_TMP" > "$DB_TMP"
mv "$DB_TMP" "$DB_PATH"
rm -f "$GZ_TMP"
echo "$(date -Iseconds) [atualizar_neo4j] dataset atualizado ($(du -h "$DB_PATH" | cut -f1))."

echo "$(date -Iseconds) [atualizar_neo4j] reconstruindo HIN + exportando pro Neo4j (MERGE, idempotente) ..."
.venv/bin/python scripts/exportar_hin_neo4j.py

echo "$(date -Iseconds) [atualizar_neo4j] calculando métricas de grafo (centralidade/comunidade via GDS) ..."
.venv/bin/python scripts/computar_metricas_grafo_neo4j.py

echo "$(date -Iseconds) [atualizar_neo4j] concluído."
