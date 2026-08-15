"""Ablation de tipo de no auxiliar no HGT -- fecha a lacuna deixada em aberto
na Secao 5.2/5.5 dos manuscritos: a explicacao de "sobrecarga de
informacao" (por que o HGT especificamente tem desempenho pior que
alternativas mais simples) era so uma interpretacao fundamentada em
literatura (Wang et al., 2025), nunca testada isolando qual tipo de no
auxiliar (socio, endereco, vinculo politico) especificamente causa o
efeito.

3 candidatos, config final tunada (hidden=32/heads=1/epochs=150, ver
scripts/tunar_han_hgt.py), 5 folds, so y_direto (rankeamento/diagnostico,
nao decisao estatistica final -- mesmo escopo enxuto da busca de
hiperparametros):
- sem_socio: exclui o tipo de no socio (edges participa_de)
- sem_endereco: exclui o tipo de no endereco (edges sediada_em)
- sem_vinculo_politico: exclui o tipo de no vinculo_politico (edges
  tem_vinculo_politico)

O candidato "completo" (todos os 3 tipos de no auxiliar, mesma config) NAO
e rerodado aqui -- reaproveita o valor ja conhecido da busca de
hiperparametros (candidato "mais_epocas_h32_hd1_e150", PR-AUC 0,0244,
mesma config exata: hidden=32/heads=1/epochs=150/5 folds/y_direto/mesmo
random_state=42), documentado em docs/research_plan.md e CLAUDE.md.

Cada candidato roda em subprocesso isolado (scripts/_ablation_no_tipo_candidato.py)
-- sobrevive a um OOM killer sem perder os outros candidatos.

Uso:
    uv run python scripts/ablation_tipo_no_hgt.py
"""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

import pandas as pd
from loguru import logger

CHECKPOINT_DIR = Path.home() / "checkpoints_ablation_tipo_no_hgt"

# Baseline conhecido (busca de hiperparametros, candidato "mais_epocas_h32_hd1_e150"):
# hidden=32, heads=1, epochs=150, 5 folds, y_direto, random_state=42.
# Nao rerodado aqui -- ver docstring do modulo.
BASELINE_PR_AUC = 0.0244

CANDIDATOS = [
    {
        "nome": "sem_socio",
        "excluir": "localizada_em,rev_localizada_em,participa_de,rev_participa_de",
    },
    {
        "nome": "sem_endereco",
        "excluir": "localizada_em,rev_localizada_em,sediada_em,rev_sediada_em",
    },
    {
        "nome": "sem_vinculo_politico",
        "excluir": "localizada_em,rev_localizada_em,tem_vinculo_politico,rev_tem_vinculo_politico",
    },
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
        [sys.executable, "scripts/_ablation_no_tipo_candidato.py", "--excluir", c["excluir"], "--out", str(out)],
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
    resumo = [{"nome": "completo_3_tipos_no", "pr_auc_medio": BASELINE_PR_AUC, "status": "reaproveitado da busca de hiperparametros"}]
    for c in CANDIDATOS:
        resultado = _rodar_candidato(c)
        if resultado is None:
            resumo.append({"nome": c["nome"], "pr_auc_medio": None, "status": "falhou"})
            continue
        resumo.append({"nome": c["nome"], "pr_auc_medio": resultado["pr_auc"].mean(), "status": "ok"})

    df = pd.DataFrame(resumo)
    print()
    print("========== Ablation de tipo de no auxiliar no HGT (y_direto, 5 folds) ==========")
    print(df.to_string(index=False))
    print()
    print(f"Baseline (todos os 3 tipos de no): PR-AUC {BASELINE_PR_AUC:.4f}")
    for r in resumo[1:]:
        if r["pr_auc_medio"] is not None:
            delta = r["pr_auc_medio"] - BASELINE_PR_AUC
            print(f"{r['nome']}: PR-AUC {r['pr_auc_medio']:.4f} (delta vs. completo: {delta:+.4f})")


if __name__ == "__main__":
    main()
