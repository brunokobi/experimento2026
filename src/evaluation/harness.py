"""Harness de avaliacao para os baselines (etapa 7.2, ver
``docs/research_plan.md``, secao 7). Infraestrutura compartilhada pelos 3
modelos (tabular, GNN homogenea, HAN/HGT) -- nao especifica a nenhum deles.

Por que validacao cruzada estratificada repetida, nao split simples/temporal:
com so 148 (``y_direto``) ou 188 (``y_qualquer``) positivos em 344 mil, um
unico holdout arrisca deixar poucos ou nenhum positivo no teste (ver secao 7
do plano). ``RepeatedStratifiedKFold`` preserva a proporcao de positivos em
cada fold, e repetir com particoes diferentes da variancia real da
estimativa -- essencial pra comparar 3 modelos sem concluir diferenca por
sorte de um unico split.

Metricas: **PR-AUC** (``average_precision_score``) e **Precision@k** --
nunca acuracia, que engana com 0,055% de positivos (um modelo que sempre diz
"nao sancionada" acerta 99,945% e nao aprendeu nada, ver secao 5 do plano).
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
import pandas as pd
from loguru import logger
from scipy.stats import wilcoxon
from sklearn.metrics import average_precision_score
from sklearn.model_selection import RepeatedStratifiedKFold

FitPredict = Callable[[pd.DataFrame, np.ndarray, pd.DataFrame], np.ndarray]
"""Assinatura esperada do modelo: recebe ``(x_train, y_train, x_test)``,
devolve um score continuo (quanto maior, mais arriscado) para ``x_test`` --
nao a classe predita. Cada baseline (tabular/GNN homogenea/HAN-HGT)
implementa isso do seu jeito e passa pro harness; e o unico ponto de
acoplamento entre o harness e o modelo."""


def precision_at_k(y_true: np.ndarray, y_score: np.ndarray, k: int) -> float:
    """Fracao de positivos reais entre os ``k`` maiores scores.

    Ex.: ``k=20`` -> "das 20 empresas mais bem ranqueadas, quantas eram
    sancionadas de verdade" (ver secao 5 do plano de pesquisa). Se ``k``
    exceder o numero de amostras, usa todas (nao lanca erro).
    """
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    k = min(k, len(y_score))
    if k == 0:
        return float("nan")
    top_k_idx = np.argsort(-y_score)[:k]
    return float(np.mean(y_true[top_k_idx]))


def evaluate_repeated_cv(
    x: pd.DataFrame,
    y: np.ndarray,
    fit_predict: FitPredict,
    n_splits: int = 5,
    n_repeats: int = 10,
    k_values: Sequence[int] = (10, 20, 50),
    random_state: int = 42,
) -> pd.DataFrame:
    """Roda validacao cruzada estratificada repetida e retorna um
    ``DataFrame`` (uma linha por fold) com ``pr_auc`` e
    ``precision_at_{k}`` para cada ``k`` em ``k_values``.

    Folds sem nenhum positivo no teste sao pulados (PR-AUC indefinido nesse
    caso) e um aviso e logado -- nao deveria acontecer com ``n_splits``
    razoavel dado o numero de positivos, mas e melhor pular explicitamente
    do que propagar ``NaN``/erro silencioso.

    **Importante para comparar modelos** (``compare_models``): rode
    ``evaluate_repeated_cv`` com o **mesmo** ``random_state``/``n_splits``/
    ``n_repeats`` para os modelos que for comparar -- o teste de Wilcoxon e
    pareado por fold, exige que a linha ``i`` do resultado de A corresponda
    exatamente ao mesmo fold da linha ``i`` do resultado de B.
    """
    cv = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=random_state)
    y = np.asarray(y)
    rows: list[dict[str, Any]] = []

    for fold_idx, (train_idx, test_idx) in enumerate(cv.split(x, y)):
        repeat, fold = divmod(fold_idx, n_splits)
        x_train, x_test = x.iloc[train_idx], x.iloc[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        if y_test.sum() == 0:
            logger.warning(f"repeat={repeat} fold={fold}: 0 positivos no teste, PR-AUC indefinido -- pulado.")
            continue

        y_score = fit_predict(x_train, y_train, x_test)
        row: dict[str, Any] = {
            "repeat": repeat,
            "fold": fold,
            "pr_auc": average_precision_score(y_test, y_score),
            "n_positivos_teste": int(y_test.sum()),
        }
        for k in k_values:
            row[f"precision_at_{k}"] = precision_at_k(y_test, y_score, k)
        rows.append(row)

    resultado = pd.DataFrame(rows)
    logger.info(
        f"CV: {len(resultado)} folds validos, PR-AUC medio={resultado['pr_auc'].mean():.4f} "
        f"(+-{resultado['pr_auc'].std():.4f})"
    )
    return resultado


def compare_models(scores_a: np.ndarray, scores_b: np.ndarray, alternative: str = "two-sided"):
    """Teste de Wilcoxon (pareado, nao-parametrico) comparando as
    distribuicoes de uma metrica (ex.: coluna ``pr_auc`` por fold) entre dois
    modelos -- ver secao 7 do plano ("multiplas seeds + teste estatistico").

    Os dois vetores devem vir dos **mesmos folds**, na mesma ordem
    (comparacao pareada, nao amostras independentes) -- ver nota em
    ``evaluate_repeated_cv``.
    """
    scores_a = np.asarray(scores_a)
    scores_b = np.asarray(scores_b)
    if len(scores_a) != len(scores_b):
        raise ValueError("scores_a e scores_b devem ter o mesmo numero de folds (comparacao pareada).")
    return wilcoxon(scores_a, scores_b, alternative=alternative)
