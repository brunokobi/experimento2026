"""Ablation barato (so XGBoost, sem GNN/HGT) isolando a contribuicao
marginal de dois grupos de feature adicionados juntos na rodada 3 (ver
docs/research_plan.md, secao 7) -- limitacao explicitamente declarada nos
manuscritos (Secao 5.5 de docs/manuscrito/paper_en.md e dissertacao_pt.md):
nao dava pra saber quanto do fechamento da diferenca tabular-vs-GNN
(Secao 5.2) vinha de cada grupo separadamente.

Grupos (7 colunas da rodada 3, 117->124):
- GRAFO (4 colunas): grau_socio_comum, grau_endereco_comum,
  grau_vinculo_politico_comum, grau_do_socio -- vem da estrutura da HIN.
- COMPRAS (3 colunas): tem_contrato_sem_competicao, sobrepreco_contrato_max,
  idade_empresa_anos -- vem de dado externo (contratos, JUCEES), nao da HIN.

4 variantes, mesmos 30 folds/seed do harness principal (barato o bastante
pra nao precisar reduzir):
- base: matriz da rodada 2 (117 colunas, sem os 7 novos)
- so_grafo: rodada 2 + so as 4 colunas de grafo
- so_compras: rodada 2 + so as 3 colunas de compras/fachada
- completa: rodada 3 completa (124 colunas = 122 features + 2 rotulos, todas as 7 novas)

Uso:
    uv run python scripts/ablation_features_tabular.py
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
from src.models.tabular_baseline import xgboost_fit_predict

N_SPLITS = 5
N_REPEATS = 6
RANDOM_STATE = 42
K_VALUES = (10, 20, 50)

COLUNAS_GRAFO = ["grau_socio_comum", "grau_endereco_comum", "grau_vinculo_politico_comum", "grau_do_socio"]
COLUNAS_COMPRAS = ["tem_contrato_sem_competicao", "sobrepreco_contrato_max", "idade_empresa_anos"]

CHECKPOINT_DIR = Path.home() / "checkpoints_ablation_features"


def _rodar(nome: str, x: pd.DataFrame, y) -> pd.DataFrame:
    CHECKPOINT_DIR.mkdir(exist_ok=True)
    checkpoint = CHECKPOINT_DIR / f"{nome}.csv"
    if checkpoint.exists():
        logger.info(f"Checkpoint encontrado para '{nome}' -- pulando recomputo.")
        return pd.read_csv(checkpoint)

    logger.info(f"Rodando CV para '{nome}' ({x.shape[1]} colunas)...")
    t0 = time.perf_counter()
    resultado = evaluate_repeated_cv(
        x, y, xgboost_fit_predict(random_state=RANDOM_STATE),
        n_splits=N_SPLITS, n_repeats=N_REPEATS, k_values=K_VALUES, random_state=RANDOM_STATE,
    )
    logger.info(f"'{nome}' concluido em {time.perf_counter() - t0:.1f}s.")
    resultado.to_csv(checkpoint, index=False)
    return resultado


def main() -> None:
    loader = GrandeVitoriaLoader()
    builder = build_empresas_hin(loader)
    builder.build()
    features = build_feature_matrix(loader, builder=builder)

    faltando = [c for c in COLUNAS_GRAFO + COLUNAS_COMPRAS if c not in features.columns]
    if faltando:
        raise RuntimeError(f"Colunas esperadas da rodada 3 nao encontradas: {faltando}")

    x_completa = features.drop(columns=["y_direto", "y_qualquer"])
    x_base = x_completa.drop(columns=COLUNAS_GRAFO + COLUNAS_COMPRAS)
    x_so_grafo = x_completa.drop(columns=COLUNAS_COMPRAS)
    x_so_compras = x_completa.drop(columns=COLUNAS_GRAFO)

    variantes = {
        "base_117cols": x_base,
        "so_grafo_121cols": x_so_grafo,
        "so_compras_120cols": x_so_compras,
        "completa_124cols": x_completa,
    }
    logger.info({nome: v.shape[1] for nome, v in variantes.items()})

    resumo = []
    for rotulo in ["y_direto", "y_qualquer"]:
        y = features[rotulo].to_numpy()
        resultados = {}
        for nome_variante, x in variantes.items():
            nome_completo = f"{nome_variante}_{rotulo}"
            resultados[nome_variante] = _rodar(nome_completo, x, y)
            r = resultados[nome_variante]
            resumo.append(
                {"rotulo": rotulo, "variante": nome_variante, "n_cols": x.shape[1],
                 "pr_auc_medio": r["pr_auc"].mean(), "pr_auc_std": r["pr_auc"].std()}
            )

        print()
        print(f"--- Comparacoes pareadas (Wilcoxon, {rotulo}) ---")
        pares = [
            ("base_117cols", "so_grafo_121cols"),
            ("base_117cols", "so_compras_120cols"),
            ("base_117cols", "completa_124cols"),
            ("so_grafo_121cols", "completa_124cols"),
            ("so_compras_120cols", "completa_124cols"),
        ]
        for a, b in pares:
            teste = compare_models(resultados[a]["pr_auc"].to_numpy(), resultados[b]["pr_auc"].to_numpy())
            media_a, media_b = resultados[a]["pr_auc"].mean(), resultados[b]["pr_auc"].mean()
            veredito = "diferenca significativa (p<0.05)" if teste.pvalue < 0.05 else "sem diferenca significativa"
            print(f"{a} (media={media_a:.4f}) vs {b} (media={media_b:.4f}): Wilcoxon p={teste.pvalue:.4f} -- {veredito}")

    print()
    print("=== Resumo ===")
    print(pd.DataFrame(resumo).to_string(index=False))


if __name__ == "__main__":
    main()
