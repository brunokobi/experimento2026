"""Treina e persiste a versão final da GNN homogênea (etapa 7.4) pra servir
inferência em produção, ao lado do modelo tabular já finalizado (ver
``treinar_modelo_final.py``) -- o `kobi-intelligence-hub` passa a mostrar os
dois scores lado a lado no dossiê, em vez de só o XGBoost.

Escolhida a variante HOMOGÊNEA, não HAN/HGT: o preprint mostrou a HAN/HGT
("mais sofisticada") perdendo estatisticamente pra homogênea em várias
rodadas de comparação -- não faz sentido escolher a mais cara/complexa pra
produção quando a mais simples já não perde (ver README.md).

Diferente do harness de CV (``rodar_baseline_gnn_homogenea.py``): aqui
treina numa passada só, com o grafo INTEIRO como treino e como alvo de
inferência -- a GNN é transdutiva (a passagem de mensagem usa o grafo
inteiro de qualquer jeito, ver docstring de
``src.models.gnn_homogeneous.make_gnn_fit_predict``), então isso não é
"vazamento de dado" no sentido de medir generalização (isso já foi feito e
publicado no harness de CV) -- é maximizar o sinal do modelo SERVIDO, o
mesmo raciocínio de ``treinar_modelo_final.py`` pro baseline tabular.

Salva em ``models/gnn_final/``:
  - ``scores_gnn.csv`` (cnpj, score_0_100): probabilidade prevista JÁ pra
    TODAS as empresas do dataset de treino. O projeto irmão
    (kobi-intelligence-hub) só faz lookup por CNPJ nesse CSV, igual já faz
    com ``models/tabular_final/scores.csv``.
  - ``metadata.json``: data de treino, hiperparâmetros, total de empresas,
    positivos (documentação/auditoria).

Não salva os pesos do PyTorch em si (diferente do ``modelo.joblib`` do
baseline tabular): a GNN é transdutiva -- reaproveitar os pesos sem
reconstruir o EXATO mesmo grafo usado no treino não tem sentido (o grafo
muda conforme o dataset cresce, e diferente do XGBoost tabular a GNN não
generaliza pra um nó que nunca viu no treino). Pra atualizar o score,
roda esse script de novo (mesmo ciclo do tabular, ver README.md).

Uso:
    uv run python scripts/treinar_modelo_final_gnn.py
"""
from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from loguru import logger

from src.data.loaders import GrandeVitoriaLoader
from src.features.tabular import build_feature_matrix
from src.graph.build_hin import build_empresas_hin
from src.models.gnn_homogeneous import make_gnn_fit_predict

SAIDA_DIR = Path(__file__).resolve().parent.parent / "models" / "gnn_final"
ROTULO = "y_direto"
MAX_GRAU_ENDERECO = 20
HIDDEN_CHANNELS = 64
EPOCHS = 100
LR = 0.01
RANDOM_STATE = 42


def main() -> None:
    loader = GrandeVitoriaLoader()

    logger.info("Construindo a HIN a partir do banco real...")
    t0 = time.perf_counter()
    builder = build_empresas_hin(loader)
    builder.build()
    logger.info(f"HIN construída em {time.perf_counter() - t0:.1f}s.")

    logger.info("Construindo a matriz de features a partir do banco real...")
    t0 = time.perf_counter()
    features = build_feature_matrix(loader)
    logger.info(f"Matriz construída em {time.perf_counter() - t0:.1f}s: {features.shape}")

    colunas_x = [c for c in features.columns if c not in ("y_direto", "y_qualquer")]
    x_full = features[colunas_x]
    y_full = features[ROTULO].to_numpy()

    n_pos = int(y_full.sum())
    n_neg = len(y_full) - n_pos
    logger.info(f"Treinando com {len(y_full)} empresas, {n_pos} positivas ({n_neg} negativas)...")

    fit_predict = make_gnn_fit_predict(
        builder, features,
        max_grau_endereco=MAX_GRAU_ENDERECO, hidden_channels=HIDDEN_CHANNELS,
        epochs=EPOCHS, lr=LR, random_state=RANDOM_STATE,
    )

    t0 = time.perf_counter()
    # Treina no grafo inteiro E prediz pro grafo inteiro (transdutivo -- não
    # há separação treino/teste aqui, ver docstring do módulo acima).
    scores = fit_predict(x_full, y_full, x_full)
    logger.info(f"Treino + inferência concluídos em {time.perf_counter() - t0:.1f}s.")

    scores_df = pd.DataFrame({"cnpj": x_full.index, "score_0_100": (scores * 100).round(2)})

    SAIDA_DIR.mkdir(parents=True, exist_ok=True)
    scores_df.to_csv(SAIDA_DIR / "scores_gnn.csv", index=False)
    (SAIDA_DIR / "metadata.json").write_text(json.dumps({
        "treinado_em": datetime.now(timezone.utc).isoformat(),
        "modelo": "GNN homogênea (GraphSAGE, 2 camadas)",
        "rotulo": ROTULO,
        "total_empresas": len(y_full),
        "positivos": n_pos,
        "max_grau_endereco": MAX_GRAU_ENDERECO,
        "hidden_channels": HIDDEN_CHANNELS,
        "epochs": EPOCHS,
        "lr": LR,
        "random_state": RANDOM_STATE,
    }, ensure_ascii=False, indent=2, default=str))
    logger.info(f"Scores salvos em {SAIDA_DIR}/")


if __name__ == "__main__":
    main()
