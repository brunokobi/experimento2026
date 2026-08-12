"""Script manual: etapa 7.6 -- padroniza n_splits/n_repeats/random_state
entre os 3 baselines e roda a comparacao estatistica (Wilcoxon) de verdade.

Fold count padronizado em 5x6=30 -- escala acordada com o pesquisador em
08/08/2026 apos o resultado inconclusivo com 5 folds (Wilcoxon p entre 0.625
e 1.0 em todos os pares, sem poder estatistico suficiente para decidir).
Estimativa de tempo: ~7h30 nesta maquina (HAN/HGT domina o custo, ~278s por
fold). Ainda e uma comparacao valida (mesmo random_state, folds pareados
por construcao) -- ver docs/research_plan.md, secao 7, pendencia de rigor
metodologico.

Checkpoint por etapa (adicionado apos o 3o reboot interromper uma rodada no
meio): cada combinacao modelo/rotulo e salva em disco assim que termina, em
``~/checkpoints_comparar_baselines/``. Se o script for rodado de novo (ex.:
apos reboot), pula direto as etapas ja concluidas em vez de recomputar do
zero -- so perde a etapa que estava rodando no momento da interrupcao, nao
tudo. O nome do arquivo inclui o numero de colunas da matriz de features
(``x.shape[1]``) para nao reusar por engano um checkpoint de uma versao
antiga da matriz (com menos/outras features) -- limitacao conhecida: nao
detecta mudanca de conteudo com a MESMA contagem de colunas.

Uso:
    uv run python scripts/comparar_baselines.py
"""

from __future__ import annotations

import time
from pathlib import Path

import pandas as pd
from loguru import logger

from src.data.loaders import GrandeVitoriaLoader
from src.evaluation.harness import compare_models, evaluate_repeated_cv
from src.features.tabular import build_feature_matrix
from src.graph.build_hin import build_empresas_hin
from src.models.gnn_homogeneous import make_gnn_fit_predict
from src.models.han_hgt import make_han_hgt_fit_predict
from src.models.tabular_baseline import xgboost_fit_predict

N_SPLITS = 5
N_REPEATS = 6
RANDOM_STATE = 42
K_VALUES = (10, 20, 50)
CHECKPOINT_DIR = Path.home() / "checkpoints_comparar_baselines"


def _checkpoint_path(nome: str, num_colunas: int) -> Path:
    CHECKPOINT_DIR.mkdir(exist_ok=True)
    return CHECKPOINT_DIR / f"{nome.replace('/', '_')}_ncols{num_colunas}.csv"


def _rodar(nome: str, x, y, fit_predict):
    checkpoint = _checkpoint_path(nome, x.shape[1])
    if checkpoint.exists():
        logger.info(f"Checkpoint encontrado para '{nome}' ({checkpoint}) -- pulando recomputo.")
        return pd.read_csv(checkpoint)

    logger.info(f"Rodando CV para '{nome}'...")
    t0 = time.perf_counter()
    resultado = evaluate_repeated_cv(
        x, y, fit_predict, n_splits=N_SPLITS, n_repeats=N_REPEATS, k_values=K_VALUES, random_state=RANDOM_STATE
    )
    logger.info(f"'{nome}' concluido em {time.perf_counter() - t0:.1f}s.")
    resultado.to_csv(checkpoint, index=False)
    return resultado


def _comparar(nome_a: str, resultado_a, nome_b: str, resultado_b) -> None:
    teste = compare_models(resultado_a["pr_auc"].to_numpy(), resultado_b["pr_auc"].to_numpy())
    media_a, media_b = resultado_a["pr_auc"].mean(), resultado_b["pr_auc"].mean()
    veredito = "diferenca significativa (p<0.05)" if teste.pvalue < 0.05 else "sem diferenca significativa"
    print(f"{nome_a} (media={media_a:.4f}) vs {nome_b} (media={media_b:.4f}): Wilcoxon p={teste.pvalue:.4f} -- {veredito}")


def main() -> None:
    loader = GrandeVitoriaLoader()
    builder = build_empresas_hin(loader)
    builder.build()
    features = build_feature_matrix(loader, builder=builder)
    x = features.drop(columns=["y_direto", "y_qualquer"])

    modelos = {
        "tabular": xgboost_fit_predict(random_state=RANDOM_STATE),
        "gnn_homogenea": make_gnn_fit_predict(builder, features, random_state=RANDOM_STATE),
        "han_hgt": make_han_hgt_fit_predict(builder, features, random_state=RANDOM_STATE),
    }

    for rotulo in ["y_direto", "y_qualquer"]:
        print()
        print(f"========== Rotulo: {rotulo} ({N_SPLITS}x{N_REPEATS} folds) ==========")
        y = features[rotulo].to_numpy()
        resultados = {}
        for nome, fit_predict in modelos.items():
            resultados[nome] = _rodar(f"{nome}/{rotulo}", x, y, fit_predict)
            r = resultados[nome]
            print(f"{nome}: PR-AUC {r['pr_auc'].mean():.4f} +- {r['pr_auc'].std():.4f}")

        print()
        print("--- Comparacoes (Wilcoxon pareado, pr_auc por fold) ---")
        for a, b in [("tabular", "gnn_homogenea"), ("tabular", "han_hgt"), ("gnn_homogenea", "han_hgt")]:
            _comparar(a, resultados[a], b, resultados[b])


if __name__ == "__main__":
    main()
