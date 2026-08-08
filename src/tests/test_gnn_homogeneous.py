"""Testes do baseline GNN homogenea (etapa 7.4) -- HIN pequena e sintetica
(via ``HINBuilder`` direto ou o fixture do schema real), nao o banco de
verdade (treino de GNN e caro, fica pra ``scripts/rodar_baseline_gnn_homogenea.py``).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data.loaders import GrandeVitoriaLoader
from src.evaluation.harness import evaluate_repeated_cv
from src.graph.build_hin import build_empresas_hin
from src.graph.hin_builder import HINBuilder
from src.models.gnn_homogeneous import build_combined_adjacency, make_gnn_fit_predict


def test_build_combined_adjacency_poda_hub_de_endereco() -> None:
    """25 empresas no mesmo endereco 'hub' nao devem virar aresta entre si
    (grau > limiar=20); 2 empresas num endereco normal devem."""
    builder = HINBuilder()
    empresas = [f"emp_{i}" for i in range(25)]
    builder.add_node_type("empresa", empresas)
    builder.add_node_type("endereco", ["end_hub", "end_normal"])

    edges = [(e, "end_hub") for e in empresas[:23]] + [(empresas[23], "end_normal"), (empresas[24], "end_normal")]
    builder.add_edge_type("empresa", "sediada_em", "endereco", edges=edges, bidirectional=True)
    builder.build()

    adjacencia = build_combined_adjacency(builder, max_grau_endereco=20)

    # nenhum par dentro do hub (23 empresas) deve estar conectado
    assert adjacencia[0, 1] == 0
    assert adjacencia[5, 10] == 0
    # o par do endereco normal (abaixo do limiar) deve estar conectado
    assert adjacencia[23, 24] == 1
    assert adjacencia[24, 23] == 1  # simetrico


def test_build_combined_adjacency_tolera_metapath_ausente() -> None:
    """Sem vinculo_politico na HIN (node_type nem existe), nao deve quebrar --
    so ignora esse componente."""
    builder = HINBuilder()
    builder.add_node_type("empresa", ["emp_1", "emp_2"])
    builder.add_node_type("socio", ["soc_1"])
    builder.add_edge_type(
        "socio", "participa_de", "empresa", edges=[("soc_1", "emp_1"), ("soc_1", "emp_2")], bidirectional=True
    )
    builder.build()

    adjacencia = build_combined_adjacency(builder)
    assert adjacencia[0, 1] == 1  # ligados via socio comum


def test_build_combined_adjacency_sem_nenhum_metapath_levanta_erro() -> None:
    builder = HINBuilder()
    builder.add_node_type("empresa", ["emp_1", "emp_2"])
    builder.build()

    with pytest.raises(ValueError, match="Nenhum dos metapaths"):
        build_combined_adjacency(builder)


def test_gnn_fit_predict_integra_com_harness(grande_vitoria_loader: GrandeVitoriaLoader) -> None:
    """So confirma integracao de ponta a ponta (formato/contrato), nao
    qualidade do modelo -- 5 empresas e treino tao pequeno nao permite
    aprender nada de verdade (ver script para validacao real)."""
    builder = build_empresas_hin(grande_vitoria_loader)
    builder.build()

    empresas = grande_vitoria_loader.empresas().set_index("cnpj")
    features = pd.DataFrame(
        {"capital_social": empresas["capital_social"]},
        index=empresas.index,
    )
    y = grande_vitoria_loader.rotulo_sancao().set_index("cnpj_empresa")["y_direto"].reindex(features.index)

    fit_predict = make_gnn_fit_predict(builder, features, epochs=3, hidden_channels=4)
    resultado = evaluate_repeated_cv(
        features, y.to_numpy(), fit_predict, n_splits=2, n_repeats=1, k_values=(1,)
    )
    assert not resultado.empty
    assert resultado["pr_auc"].between(0, 1).all()


def test_gnn_fit_predict_usa_indice_nao_posicao() -> None:
    """fit_predict deve mapear x_train/x_test pelo indice (cnpj), nao pela
    posicao -- embaralhar a ordem das linhas nao deve quebrar nem trocar os
    nos avaliados."""
    builder = HINBuilder()
    empresas = [f"emp_{i}" for i in range(10)]
    builder.add_node_type("empresa", empresas)
    builder.add_node_type("socio", ["soc_1"])
    builder.add_edge_type(
        "socio", "participa_de", "empresa", edges=[("soc_1", "emp_0"), ("soc_1", "emp_1")], bidirectional=True
    )
    builder.build()

    features = pd.DataFrame({"x": np.arange(10, dtype=float)}, index=empresas)
    fit_predict = make_gnn_fit_predict(builder, features, epochs=2, hidden_channels=4)

    # embaralha o dataframe antes de fatiar em train/test
    embaralhado = features.sample(frac=1.0, random_state=0)
    x_test = embaralhado.iloc[:3]
    x_train = embaralhado.iloc[3:]
    y_train = np.zeros(len(x_train), dtype=int)

    scores = fit_predict(x_train, y_train, x_test)
    assert len(scores) == 3
    assert np.all(np.isfinite(scores))
