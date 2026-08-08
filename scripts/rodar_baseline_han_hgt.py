"""Script manual: roda o baseline HAN/HGT (etapa 7.5) contra o banco real --
terceiro numero quantitativo real da dissertacao, o modelo "heterogeneo de
verdade" (nao colapsa os metapaths, ao contrario da etapa 7.4). Fora da
suite de testes (banco real + treino de HGT, demora minutos).

n_repeats menor que o da GNN homogenea (1 em vez de 2) -- HGTConv e mais caro
por epoch (multiplos tipos de relacao, atencao por aresta) que a GraphSAGE
simples da etapa 7.4; ainda mais distante dos 50 folds do baseline tabular
(ver pendencia de rigor metodologico na secao 7 do plano, a resolver na 7.6).

Defaults de memoria conservadores herdados de make_han_hgt_fit_predict
(hidden_channels=32, num_heads=1, municipio excluido) -- ver docstring la:
a configuracao original (64/2 cabecas, com municipio) estourou memoria (OOM
killer) na maquina local de desenvolvimento (7,8 GB de RAM).

Uso:
    uv run python scripts/rodar_baseline_han_hgt.py
"""

from __future__ import annotations

import time

from loguru import logger

from src.data.loaders import GrandeVitoriaLoader
from src.evaluation.harness import evaluate_repeated_cv
from src.features.tabular import build_feature_matrix
from src.graph.build_hin import build_empresas_hin
from src.models.han_hgt import make_han_hgt_fit_predict

K_VALUES = (10, 20, 50)
N_SPLITS = 5
N_REPEATS = 1
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
    fit_predict = make_han_hgt_fit_predict(builder, features, epochs=EPOCHS)

    resultado_direto = _rodar(x, features, fit_predict, "y_direto")
    resultado_qualquer = _rodar(x, features, fit_predict, "y_qualquer")

    print()
    print(f"=== HAN/HGT -- y_direto ({N_SPLITS}x{N_REPEATS} folds, {EPOCHS} epochs) ===")
    print(resultado_direto[["pr_auc", *[f"precision_at_{k}" for k in K_VALUES]]].agg(["mean", "std"]))

    print()
    print(f"=== HAN/HGT -- y_qualquer ({N_SPLITS}x{N_REPEATS} folds, {EPOCHS} epochs) ===")
    print(resultado_qualquer[["pr_auc", *[f"precision_at_{k}" for k in K_VALUES]]].agg(["mean", "std"]))


if __name__ == "__main__":
    main()
