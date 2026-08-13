"""Busca pequena de hiperparametros do HAN/HGT (etapa 7.5), motivada pela
maior vulnerabilidade do resultado v3 (12/08/2026): os defaults
(``hidden_channels=32``, ``num_heads=1``, ``epochs=50``) foram reduzidos por
causa de OOM na maquina local, nao por tuning -- um revisor de periodico Qualis
alto vai perguntar exatamente isso antes de aceitar "HAN/HGT e pior" como
conclusao. Objetivo aqui nao e achar o otimo global, e sim descartar a
hipotese de undertraining/undercapacity com uma busca pequena e defensavel.

5 candidatos, variando um eixo por vez a partir do baseline atual, so em
``y_direto`` (o rotulo principal) e com so 5 folds (nao 30) -- suficiente
pra RANQUEAR candidatos, nao pra decidir estatisticamente (isso e feito
depois, rodando o vencedor pela comparacao completa de 30 folds em
``scripts/comparar_baselines.py``).

Cada candidato roda em subprocesso isolado (``_tunar_han_hgt_candidato.py``)
para sobreviver a um OOM killer sem perder os outros candidatos -- mesma
logica do checkpoint por etapa de ``comparar_baselines.py``, em granularidade
de processo em vez de arquivo.

Uso:
    uv run python scripts/tunar_han_hgt.py
"""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

import pandas as pd
from loguru import logger

CHECKPOINT_DIR = Path.home() / "checkpoints_tunar_han_hgt"

CANDIDATOS = [
    {"nome": "baseline_h32_hd1_e50", "hidden": 32, "heads": 1, "epochs": 50},
    {"nome": "mais_epocas_h32_hd1_e150", "hidden": 32, "heads": 1, "epochs": 150},
    {"nome": "mais_hidden_h64_hd1_e50", "hidden": 64, "heads": 1, "epochs": 50},
    {"nome": "mais_heads_h32_hd2_e50", "hidden": 32, "heads": 2, "epochs": 50},
    {"nome": "maior_h64_hd2_e100", "hidden": 64, "heads": 2, "epochs": 100},
]


def _rodar_candidato(c: dict) -> pd.DataFrame | None:
    CHECKPOINT_DIR.mkdir(exist_ok=True)
    out = CHECKPOINT_DIR / f"{c['nome']}.csv"
    if out.exists():
        logger.info(f"Checkpoint encontrado para '{c['nome']}' -- pulando recomputo.")
        return pd.read_csv(out)

    logger.info(f"Rodando candidato '{c['nome']}' em subprocesso isolado...")
    t0 = time.perf_counter()
    resultado = subprocess.run(
        [
            sys.executable,
            "scripts/_tunar_han_hgt_candidato.py",
            "--hidden", str(c["hidden"]),
            "--heads", str(c["heads"]),
            "--epochs", str(c["epochs"]),
            "--out", str(out),
        ],
        cwd=Path(__file__).resolve().parent.parent,
    )
    dt = time.perf_counter() - t0
    if resultado.returncode != 0:
        logger.warning(
            f"Candidato '{c['nome']}' FALHOU (exit code {resultado.returncode}, "
            f"provavel OOM killer se 137) apos {dt:.1f}s -- pulando, sem checkpoint."
        )
        return None
    logger.info(f"Candidato '{c['nome']}' concluido em {dt:.1f}s.")
    return pd.read_csv(out)


def main() -> None:
    resumo = []
    for c in CANDIDATOS:
        resultado = _rodar_candidato(c)
        if resultado is None:
            resumo.append({**c, "pr_auc_medio": None, "pr_auc_std": None, "status": "falhou"})
            continue
        resumo.append(
            {
                **c,
                "pr_auc_medio": resultado["pr_auc"].mean(),
                "pr_auc_std": resultado["pr_auc"].std(),
                "status": "ok",
            }
        )

    df = pd.DataFrame(resumo).sort_values("pr_auc_medio", ascending=False, na_position="last")
    print()
    print("========== Ranking dos candidatos (y_direto, 5 folds) ==========")
    print(df.to_string(index=False))

    vencedor = df.iloc[0]
    if pd.isna(vencedor["pr_auc_medio"]):
        print("\nNenhum candidato completou com sucesso -- revisar antes de prosseguir.")
    else:
        print(
            f"\nVencedor: '{vencedor['nome']}' (hidden={vencedor['hidden']}, heads={vencedor['heads']}, "
            f"epochs={vencedor['epochs']}) -- PR-AUC {vencedor['pr_auc_medio']:.4f}"
        )


if __name__ == "__main__":
    main()
