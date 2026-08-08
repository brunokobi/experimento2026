"""Script manual: roda o baseline GNN homogenea (etapa 7.4) contra o banco
real -- segundo numero quantitativo real da dissertacao, primeiro que de
fato usa a estrutura de rede. Fora da suite de testes (banco real + treino
de GNN, demora minutos).

Comeca com n_repeats/epochs modestos (custo computacional de GNN e bem maior
que XGBoost -- 50 folds x 100 epochs de passagem de mensagem num grafo de
milhoes de arestas seria inviavel numa sessao interativa); ver
docs/research_plan.md secao 7 para a trilha de escalar depois.

Uso:
    uv run python scripts/rodar_baseline_gnn_homogenea.py
"""

from __future__ import annotations

import time

from loguru import logger

from src.data.loaders import GrandeVitoriaLoader
from src.evaluation.harness import evaluate_repeated_cv
from src.features.tabular import build_feature_matrix
from src.graph.build_hin import build_empresas_hin
from src.models.gnn_homogeneous import make_gnn_fit_predict

K_VALUES = (10, 20, 50)
N_SPLITS = 5
N_REPEATS = 2
EPOCHS = 50


def _rodar(x, y_full, fit_predict, coluna_rotulo: str):
    y = y_full[coluna_rotulo].to_numpy()
    logger.info(f"Rodando CV para rotulo '{coluna_rotulo}' ({int(y.sum())} positivos em {len(y)})...")
    t0 = time.perf_counter()
    resultado = evaluate_repeated_cv(x, y, fit_predict, n_splits=N_SPLITS, n_repeats=N_REPEATS, k_values=K_VALUES)
    logger.info(f"CV para '{coluna_rotulo}' concluida em {time.perf_counter() - t0:.1f}s.")
    return resultado


def main() -> None:
    loader = GrandeVitoriaLoader()

    logger.info("Construindo a HIN e a matriz de features a partir do banco real...")
    builder = build_empresas_hin(loader)
    builder.build()
    features = build_feature_matrix(loader)

    x = features.drop(columns=["y_direto", "y_qualquer"])
    fit_predict = make_gnn_fit_predict(builder, features, epochs=EPOCHS)

    resultado_direto = _rodar(x, features, fit_predict, "y_direto")
    resultado_qualquer = _rodar(x, features, fit_predict, "y_qualquer")

    print()
    print(f"=== GNN homogenea -- y_direto ({N_SPLITS}x{N_REPEATS} folds, {EPOCHS} epochs) ===")
    print(resultado_direto[["pr_auc", *[f"precision_at_{k}" for k in K_VALUES]]].agg(["mean", "std"]))

    print()
    print(f"=== GNN homogenea -- y_qualquer ({N_SPLITS}x{N_REPEATS} folds, {EPOCHS} epochs) ===")
    print(resultado_qualquer[["pr_auc", *[f"precision_at_{k}" for k in K_VALUES]]].agg(["mean", "std"]))


if __name__ == "__main__":
    main()
