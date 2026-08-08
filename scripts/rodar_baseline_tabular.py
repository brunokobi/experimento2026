"""Script manual: roda o baseline tabular (etapa 7.3) contra o banco real --
primeiro numero quantitativo real da dissertacao. Fora da suite de testes
(depende do banco real e demora minutos, nao segundos).

Resultado principal usa `y_direto` (148 empresas, sem risco de circularidade,
ver docs/research_plan.md secao 5/9); `y_qualquer` (188, inclui as via-socio)
roda so como contexto/sensibilidade, nunca como o numero citado no texto sem
qualificar qual rotulo foi usado.

Uso:
    uv run python scripts/rodar_baseline_tabular.py
"""

from __future__ import annotations

import time

from loguru import logger

from src.data.loaders import GrandeVitoriaLoader
from src.evaluation.harness import evaluate_repeated_cv
from src.features.tabular import build_feature_matrix
from src.models.tabular_baseline import xgboost_fit_predict

K_VALUES = (10, 20, 50)


def _rodar(features, coluna_rotulo: str, n_splits: int = 5, n_repeats: int = 10):
    x = features.drop(columns=["y_direto", "y_qualquer"])
    y = features[coluna_rotulo].to_numpy()

    logger.info(f"Rodando CV para rotulo '{coluna_rotulo}' ({int(y.sum())} positivos em {len(y)})...")
    t0 = time.perf_counter()
    resultado = evaluate_repeated_cv(
        x, y, xgboost_fit_predict(), n_splits=n_splits, n_repeats=n_repeats, k_values=K_VALUES
    )
    logger.info(f"CV para '{coluna_rotulo}' concluida em {time.perf_counter() - t0:.1f}s.")
    return resultado


def main() -> None:
    logger.info("Construindo a matriz de features a partir do banco real...")
    features = build_feature_matrix(GrandeVitoriaLoader())

    resultado_direto = _rodar(features, "y_direto")
    resultado_qualquer = _rodar(features, "y_qualquer")

    print()
    print("=== Resultado principal: rotulo y_direto (148 empresas, sem circularidade) ===")
    print(resultado_direto[["pr_auc", *[f"precision_at_{k}" for k in K_VALUES]]].agg(["mean", "std"]))

    print()
    print("=== Sensibilidade: rotulo y_qualquer (188 empresas, inclui via-socio) ===")
    print(resultado_qualquer[["pr_auc", *[f"precision_at_{k}" for k in K_VALUES]]].agg(["mean", "std"]))


if __name__ == "__main__":
    main()
