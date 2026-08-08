"""Testes de ``xgboost_fit_predict`` (etapa 7.3) -- dado sintetico
(independente do schema do dataset), integrado com o harness (7.2) pra
confirmar que a assinatura ``FitPredict`` e respeitada de ponta a ponta.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.evaluation.harness import evaluate_repeated_cv
from src.models.tabular_baseline import xgboost_fit_predict


def test_xgboost_fit_predict_recupera_sinal_separavel() -> None:
    """Duas features, uma com sinal real e uma so ruido -- o XGBoost deve
    aprender a usar a que importa e dar PR-AUC bem acima do acaso."""
    rng = np.random.default_rng(0)
    n = 400
    sinal = rng.normal(size=n)
    ruido = rng.normal(size=n)
    x = pd.DataFrame({"sinal": sinal, "ruido": ruido})
    y = (sinal > np.quantile(sinal, 0.9)).astype(int)  # ~10% positivos

    resultado = evaluate_repeated_cv(x, y, xgboost_fit_predict(), n_splits=5, n_repeats=2, k_values=(10,))

    assert len(resultado) == 10
    assert resultado["pr_auc"].mean() > 0.7, "deveria superar bastante o acaso (~0.10 pra 10% positivos)"


def test_scale_pos_weight_usa_o_fold_de_treino_nao_o_dataset_inteiro() -> None:
    """Fold de treino sem nenhum positivo nao deve quebrar (scale_pos_weight
    cai para 1.0 em vez de dividir por zero)."""
    fit_predict = xgboost_fit_predict()
    x_train = pd.DataFrame({"a": np.arange(20, dtype=float)})
    y_train = np.zeros(20, dtype=int)  # nenhum positivo no treino
    x_test = pd.DataFrame({"a": np.arange(5, dtype=float)})

    scores = fit_predict(x_train, y_train, x_test)
    assert len(scores) == 5
    assert np.all(np.isfinite(scores))


def test_xgboost_fit_predict_aceita_kwargs_extras() -> None:
    """kwargs extras devem sobrescrever os defaults (ex.: menos arvores para teste rapido)."""
    fit_predict = xgboost_fit_predict(n_estimators=5, max_depth=2)
    rng = np.random.default_rng(1)
    x_train = pd.DataFrame({"a": rng.normal(size=50)})
    y_train = (x_train["a"] > 0).astype(int).to_numpy()
    x_test = pd.DataFrame({"a": rng.normal(size=10)})

    scores = fit_predict(x_train, y_train, x_test)
    assert len(scores) == 10
    assert np.all((scores >= 0) & (scores <= 1))
