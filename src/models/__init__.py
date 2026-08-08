"""Modelos/baselines para a tarefa de risco de sancao (etapa 7 do plano de pesquisa)."""

from src.models.gnn_homogeneous import make_gnn_fit_predict
from src.models.han_hgt import make_han_hgt_fit_predict
from src.models.tabular_baseline import xgboost_fit_predict

__all__ = ["xgboost_fit_predict", "make_gnn_fit_predict", "make_han_hgt_fit_predict"]
