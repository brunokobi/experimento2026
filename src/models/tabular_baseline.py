"""Baseline tabular (etapa 7.3, ver ``docs/research_plan.md``, secao 7).

XGBoost com **class weighting** (``scale_pos_weight``), nao classificacao
balanceada -- e o "chao" de comparacao: se a GNN homogenea (7.4) ou o
HAN/HGT (7.5) nao superarem isso, a estrutura de rede nao estava agregando
sinal, so o dado tabular de cada empresa isolada.

``scale_pos_weight`` em vez de SMOTE/undersampling: preserva a distribuicao
real dos dados. Reamostragem sintetica com so ~150 positivos arrisca criar
padroes artificiais que nao generalizam (ver risco de overfitting em amostra
pequena, secao 9 do plano).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from xgboost import XGBClassifier

from src.evaluation.harness import FitPredict

_DEFAULT_PARAMS: dict[str, Any] = {
    "n_estimators": 300,
    "max_depth": 4,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "eval_metric": "aucpr",
    "n_jobs": -1,
}


def xgboost_fit_predict(random_state: int = 42, **xgb_kwargs: Any) -> FitPredict:
    """Cria uma funcao ``fit_predict`` (assinatura esperada por
    ``src.evaluation.harness.evaluate_repeated_cv``) que treina um
    ``XGBClassifier`` por fold.

    ``scale_pos_weight`` e calculado a partir do **fold de treino em si**,
    nao do dataset inteiro -- evita vazamento estatistico entre treino/teste
    do CV (o fold de teste nunca influencia como o modelo pondera as
    classes).
    """

    def fit_predict(x_train: pd.DataFrame, y_train: np.ndarray, x_test: pd.DataFrame) -> np.ndarray:
        y_train = np.asarray(y_train)
        n_pos = int(y_train.sum())
        n_neg = len(y_train) - n_pos
        scale_pos_weight = n_neg / n_pos if n_pos > 0 else 1.0

        params = {**_DEFAULT_PARAMS, "random_state": random_state, "scale_pos_weight": scale_pos_weight, **xgb_kwargs}
        model = XGBClassifier(**params)
        model.fit(x_train, y_train)
        return model.predict_proba(x_test)[:, 1]

    return fit_predict
