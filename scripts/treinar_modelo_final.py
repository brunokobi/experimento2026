"""Treina e persiste a versão final do baseline tabular (XGBoost, etapa
7.3) pra servir inferência em produção -- até 07/09/2026 esse modelo só
existia dentro do harness de cross-validation (avaliação estatística),
nunca foi salvo em disco.

Diferente do harness de CV: aqui treina numa transação SÓ (todo o dataset
disponível), sem separar treino/teste -- o objetivo não é mais avaliar
generalização (já feito e publicado, ver README.md), é maximizar o sinal
usado pro modelo SERVIDO (Kobi Intelligence Hub, projeto irmão). Rótulo
principal ``y_direto`` (ver docs/research_plan.md, seção 5/9 -- risco de
circularidade do y_qualquer).

Salva em ``models/tabular_final/``:
  - ``modelo.joblib``: o XGBClassifier treinado.
  - ``colunas.json``: ordem exata das colunas de feature (a matriz de
    inferência precisa reproduzir isso à risca).
  - ``metadata.json``: data de treino, total de empresas, positivos, e o
    scale_pos_weight usado (documentação/auditoria).

Uso:
    uv run python scripts/treinar_modelo_final.py
"""
from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
from loguru import logger
from xgboost import XGBClassifier

from src.data.loaders import GrandeVitoriaLoader
from src.features.tabular import build_feature_matrix
from src.models.tabular_baseline import _DEFAULT_PARAMS

SAIDA_DIR = Path(__file__).resolve().parent.parent / "models" / "tabular_final"
ROTULO = "y_direto"


def main() -> None:
    logger.info("Construindo a matriz de features a partir do banco real...")
    t0 = time.perf_counter()
    features = build_feature_matrix(GrandeVitoriaLoader())
    logger.info(f"Matriz construída em {time.perf_counter() - t0:.1f}s: {features.shape}")

    colunas_x = [c for c in features.columns if c not in ("y_direto", "y_qualquer")]
    x = features[colunas_x]
    y = features[ROTULO].to_numpy()

    n_pos = int(y.sum())
    n_neg = len(y) - n_pos
    scale_pos_weight = n_neg / n_pos if n_pos > 0 else 1.0
    logger.info(f"Treinando com {len(y)} empresas, {n_pos} positivas (scale_pos_weight={scale_pos_weight:.1f})...")

    params = {**_DEFAULT_PARAMS, "random_state": 42, "scale_pos_weight": scale_pos_weight}
    modelo = XGBClassifier(**params)
    t0 = time.perf_counter()
    modelo.fit(x, y)
    logger.info(f"Treino concluído em {time.perf_counter() - t0:.1f}s.")

    SAIDA_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(modelo, SAIDA_DIR / "modelo.joblib")
    (SAIDA_DIR / "colunas.json").write_text(json.dumps(colunas_x, ensure_ascii=False, indent=2))
    (SAIDA_DIR / "metadata.json").write_text(json.dumps({
        "treinado_em": datetime.now(timezone.utc).isoformat(),
        "rotulo": ROTULO,
        "total_empresas": len(y),
        "positivos": n_pos,
        "scale_pos_weight": scale_pos_weight,
        "colunas": colunas_x,
        "params": params,
    }, ensure_ascii=False, indent=2, default=str))
    logger.info(f"Modelo salvo em {SAIDA_DIR}/")


if __name__ == "__main__":
    main()
