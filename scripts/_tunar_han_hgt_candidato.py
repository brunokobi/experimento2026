"""Worker de um unico candidato da busca de hiperparametros do HAN/HGT
(chamado pelo processo pai, ``scripts/tunar_han_hgt.py``, via subprocess --
nao para ser rodado direto). Roda isolado em processo proprio de proposito:
se o OOM killer matar esse processo (SIGKILL, sem exception capturavel em
Python), o processo pai sobrevive e so perde esse candidato, nao a busca
inteira -- mesmo raciocinio do checkpoint por etapa de
``scripts/comparar_baselines.py``, aplicado aqui em granularidade de
subprocesso em vez de arquivo.

Uso (via CLI, chamado pelo processo pai):
    uv run python scripts/_tunar_han_hgt_candidato.py \
        --hidden 32 --heads 1 --layers 2 --epochs 50 --out <caminho.csv>
"""

from __future__ import annotations

import argparse

from loguru import logger

from src.data.loaders import GrandeVitoriaLoader
from src.evaluation.harness import evaluate_repeated_cv
from src.features.tabular import build_feature_matrix
from src.graph.build_hin import build_empresas_hin
from src.models.han_hgt import make_han_hgt_fit_predict

N_SPLITS = 5
N_REPEATS = 1
RANDOM_STATE = 42
K_VALUES = (10, 20, 50)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hidden", type=int, required=True)
    parser.add_argument("--heads", type=int, required=True)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--epochs", type=int, required=True)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--out", type=str, required=True)
    args = parser.parse_args()

    loader = GrandeVitoriaLoader()
    builder = build_empresas_hin(loader)
    builder.build()
    features = build_feature_matrix(loader, builder=builder)
    x = features.drop(columns=["y_direto", "y_qualquer"])
    y = features["y_direto"].to_numpy()

    fit_predict = make_han_hgt_fit_predict(
        builder,
        features,
        hidden_channels=args.hidden,
        num_heads=args.heads,
        num_layers=args.layers,
        epochs=args.epochs,
        lr=args.lr,
        random_state=RANDOM_STATE,
    )

    logger.info(
        f"Candidato hidden={args.hidden} heads={args.heads} layers={args.layers} "
        f"epochs={args.epochs} lr={args.lr} -- rodando {N_SPLITS}x{N_REPEATS} folds em y_direto..."
    )
    resultado = evaluate_repeated_cv(
        x, y, fit_predict, n_splits=N_SPLITS, n_repeats=N_REPEATS, k_values=K_VALUES, random_state=RANDOM_STATE
    )
    resultado.to_csv(args.out, index=False)
    logger.info(f"Candidato concluido: PR-AUC medio={resultado['pr_auc'].mean():.4f}. Salvo em {args.out}")


if __name__ == "__main__":
    main()
