"""Testes do baseline HAN/HGT (etapa 7.5) -- HIN pequena do fixture do
schema real, nao o banco de verdade (treino de HGT e caro, fica pra
``scripts/rodar_baseline_han_hgt.py``).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.data.loaders import GrandeVitoriaLoader
from src.evaluation.harness import evaluate_repeated_cv
from src.graph.build_hin import build_empresas_hin
from src.models.han_hgt import make_han_hgt_fit_predict


def test_han_hgt_integra_com_harness(grande_vitoria_loader: GrandeVitoriaLoader) -> None:
    """So confirma integracao de ponta a ponta (formato/contrato) com todos
    os tipos de no/relacao da HIN real (empresa/socio/endereco/
    vinculo_politico/municipio) -- nao qualidade do modelo, a fixture e
    pequena demais pra aprender algo de verdade."""
    builder = build_empresas_hin(grande_vitoria_loader)
    builder.build()

    empresas = grande_vitoria_loader.empresas().set_index("cnpj")
    features = pd.DataFrame({"capital_social": empresas["capital_social"]}, index=empresas.index)
    y = grande_vitoria_loader.rotulo_sancao().set_index("cnpj_empresa")["y_direto"].reindex(features.index)

    fit_predict = make_han_hgt_fit_predict(builder, features, epochs=2, hidden_channels=4, num_heads=1)
    resultado = evaluate_repeated_cv(features, y.to_numpy(), fit_predict, n_splits=2, n_repeats=1, k_values=(1,))

    assert not resultado.empty
    assert resultado["pr_auc"].between(0, 1).all()


def test_han_hgt_usa_indice_nao_posicao() -> None:
    """Mesma garantia da GNN homogenea: embaralhar as linhas de x nao deve
    trocar quais nos sao avaliados."""
    from src.graph.hin_builder import HINBuilder

    builder = HINBuilder()
    empresas = [f"emp_{i}" for i in range(10)]
    builder.add_node_type("empresa", empresas)
    builder.add_node_type("socio", ["soc_1"])
    builder.add_edge_type(
        "socio", "participa_de", "empresa", edges=[("soc_1", "emp_0"), ("soc_1", "emp_1")], bidirectional=True
    )
    builder.build()

    features = pd.DataFrame({"x": np.arange(10, dtype=float)}, index=empresas)
    fit_predict = make_han_hgt_fit_predict(builder, features, epochs=2, hidden_channels=4, num_heads=1)

    embaralhado = features.sample(frac=1.0, random_state=0)
    x_test = embaralhado.iloc[:3]
    x_train = embaralhado.iloc[3:]
    y_train = np.zeros(len(x_train), dtype=int)

    scores = fit_predict(x_train, y_train, x_test)
    assert len(scores) == 3
    assert np.all(np.isfinite(scores))
