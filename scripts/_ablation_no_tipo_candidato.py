"""Worker de um unico candidato do ablation de tipo de no no HGT (chamado
pelo processo pai, scripts/ablation_tipo_no_hgt.py, via subprocess -- nao
para ser rodado direto). Mesma razao de isolamento em subprocesso que
scripts/_tunar_han_hgt_candidato.py: sobrevive a um OOM killer sem perder
os outros candidatos.

Uso (via CLI, chamado pelo processo pai):
    uv run python scripts/_ablation_no_tipo_candidato.py \
        --excluir localizada_em,rev_localizada_em,participa_de,rev_participa_de \
        --out <caminho.csv>
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
EPOCHS = 150  # config final tunada (scripts/tunar_han_hgt.py)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--excluir", type=str, required=True, help="relacoes a excluir, separadas por virgula")
    parser.add_argument("--out", type=str, required=True)
    args = parser.parse_args()
    excluir_relacoes = tuple(args.excluir.split(","))

    loader = GrandeVitoriaLoader()
    builder = build_empresas_hin(loader)
    builder.build()
    features = build_feature_matrix(loader, builder=builder)
    x = features.drop(columns=["y_direto", "y_qualquer"])
    y = features["y_direto"].to_numpy()

    fit_predict = make_han_hgt_fit_predict(
        builder, features, epochs=EPOCHS, random_state=RANDOM_STATE, excluir_relacoes=excluir_relacoes
    )

    logger.info(f"Candidato excluir={excluir_relacoes} epochs={EPOCHS} -- rodando {N_SPLITS}x{N_REPEATS} folds em y_direto...")
    resultado = evaluate_repeated_cv(
        x, y, fit_predict, n_splits=N_SPLITS, n_repeats=N_REPEATS, k_values=K_VALUES, random_state=RANDOM_STATE
    )
    resultado.to_csv(args.out, index=False)
    logger.info(f"Candidato concluido: PR-AUC medio={resultado['pr_auc'].mean():.4f}. Salvo em {args.out}")


if __name__ == "__main__":
    main()
