"""Testes do harness de avaliacao (etapa 7.2) -- independentes do schema do
dataset (usam x/y sinteticos, nao os fixtures do banco real), ja que o
harness e generico para qualquer um dos 3 baselines.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.evaluation.harness import compare_models, evaluate_repeated_cv, precision_at_k


def test_precision_at_k_conta_positivos_entre_os_k_maiores_scores() -> None:
    y_true = np.array([0, 1, 0, 1, 1, 0, 0, 0])
    y_score = np.array([0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2])
    # top 3 por score: indices 0,1,2 -> y_true [0,1,0] -> 1/3
    assert precision_at_k(y_true, y_score, 3) == pytest.approx(1 / 3)


def test_precision_at_k_com_k_maior_que_o_dataset_usa_tudo() -> None:
    y_true = np.array([0, 1, 0, 1])
    y_score = np.array([0.1, 0.2, 0.3, 0.4])
    assert precision_at_k(y_true, y_score, 100) == pytest.approx(y_true.mean())


def test_precision_at_k_zero_retorna_nan() -> None:
    assert np.isnan(precision_at_k(np.array([0, 1]), np.array([0.1, 0.2]), 0))


def test_evaluate_repeated_cv_recupera_ranking_quase_perfeito() -> None:
    """Um score que e literalmente o sinal usado para gerar o rotulo deve dar
    PR-AUC ~1.0 em (quase) todo fold -- confirma que o harness nao introduz
    vies/erro na hora de fatiar/agregar."""
    rng = np.random.default_rng(0)
    n = 300
    sinal = rng.normal(size=n)
    x = pd.DataFrame({"sinal": sinal})
    y = (sinal > np.quantile(sinal, 0.9)).astype(int)  # ~10% positivos, perfeitamente separavel por 'sinal'

    def fit_predict(_x_train: pd.DataFrame, _y_train: np.ndarray, x_test: pd.DataFrame) -> np.ndarray:
        return x_test["sinal"].to_numpy()

    resultado = evaluate_repeated_cv(x, y, fit_predict, n_splits=5, n_repeats=2, k_values=(5,))

    assert len(resultado) == 10  # 5 splits * 2 repeats
    assert {"repeat", "fold", "pr_auc", "n_positivos_teste", "precision_at_5"}.issubset(resultado.columns)
    assert resultado["pr_auc"].mean() > 0.99
    assert (resultado["n_positivos_teste"] > 0).all()  # nenhum fold vazio deveria sobrar


def test_evaluate_repeated_cv_pula_fold_sem_positivo_no_teste(capsys: pytest.CaptureFixture) -> None:
    # so 1 positivo em 20 amostras + n_splits alto o suficiente para garantir
    # que pelo menos 1 fold acabe sem nenhum positivo no teste
    y = np.zeros(20, dtype=int)
    y[0] = 1
    x = pd.DataFrame({"x": np.arange(20)})

    def fit_predict(_x_train: pd.DataFrame, _y_train: np.ndarray, x_test: pd.DataFrame) -> np.ndarray:
        return x_test["x"].to_numpy()

    resultado = evaluate_repeated_cv(x, y, fit_predict, n_splits=5, n_repeats=1)
    assert len(resultado) < 5  # pelo menos 1 fold sem positivo foi pulado


def test_compare_models_detecta_diferenca_consistente() -> None:
    rng = np.random.default_rng(1)
    scores_pior = 0.5 + rng.normal(0, 0.01, 15)
    scores_melhor = 0.8 + rng.normal(0, 0.01, 15)
    resultado = compare_models(scores_pior, scores_melhor)
    assert resultado.pvalue < 0.05


def test_compare_models_exige_mesmo_numero_de_folds() -> None:
    with pytest.raises(ValueError, match="mesmo numero de folds"):
        compare_models([1, 2, 3], [1, 2])
