"""Testes de ``build_feature_matrix`` (etapa 7.1) contra o fixture do schema
real (``grande_vitoria_loader``, ver conftest.py) -- mesmo cenario usado para
travar o achado de circularidade do rotulo (sancao via socio comum).
"""

from __future__ import annotations

import numpy as np

from src.data.loaders import GrandeVitoriaLoader
from src.features.tabular import build_feature_matrix


def test_cobre_todo_universo_de_empresas(grande_vitoria_loader: GrandeVitoriaLoader) -> None:
    features = build_feature_matrix(grande_vitoria_loader)
    assert len(features) == 5
    assert set(features.index) == set(grande_vitoria_loader.empresas()["cnpj"])


def test_num_socios_e_dividas_ativas_com_reindex_correto(grande_vitoria_loader: GrandeVitoriaLoader) -> None:
    features = build_feature_matrix(grande_vitoria_loader)

    # emp_1..emp_4 tem exatamente 1 socio cada; emp_5 nao tem nenhum (reindex fill_value=0)
    assert features.loc["11111111000101", "num_socios"] == 1
    assert features.loc["55555555000105", "num_socios"] == 0

    # so emp_3 tem dividas ativas (2 registros, 500+250)
    assert features.loc["33333333000103", "num_dividas_ativas"] == 2
    assert features.loc["33333333000103", "valor_dividas_ativas_log1p"] == np.log1p(750.0)
    assert features.loc["11111111000101", "num_dividas_ativas"] == 0
    assert features.loc["11111111000101", "valor_dividas_ativas_log1p"] == 0.0


def test_tem_vinculo_politico_so_para_emp4(grande_vitoria_loader: GrandeVitoriaLoader) -> None:
    features = build_feature_matrix(grande_vitoria_loader)
    assert features.loc["44444444000104", "tem_vinculo_politico"]
    for cnpj in ["11111111000101", "22222222000102", "33333333000103", "55555555000105"]:
        assert not features.loc[cnpj, "tem_vinculo_politico"]


def test_one_hot_categoricas_presentes(grande_vitoria_loader: GrandeVitoriaLoader) -> None:
    features = build_feature_matrix(grande_vitoria_loader)
    for coluna in ["porte_ME", "porte_EPP", "regime_tributario_SIMPLES", "municipio_VITORIA", "cnae_segmento_47"]:
        assert coluna in features.columns, f"coluna esperada ausente: {coluna}"
    assert features.loc["11111111000101", "porte_ME"]
    assert not features.loc["11111111000101", "porte_EPP"]


def test_rotulo_preserva_circularidade_direto_vs_qualquer(grande_vitoria_loader: GrandeVitoriaLoader) -> None:
    """Mesmo achado de test_real_schema.py::test_rotulo_sancao_separa_direto_de_qualquer,
    conferido agora tambem na matriz de features (nao so no loader isolado)."""
    features = build_feature_matrix(grande_vitoria_loader)
    assert features.loc["11111111000101", "y_direto"]
    assert not features.loc["22222222000102", "y_direto"]
    assert features.loc["22222222000102", "y_qualquer"]
    assert not features.loc["33333333000103", "y_qualquer"]


def test_nenhuma_coluna_de_vazamento_de_sancao(grande_vitoria_loader: GrandeVitoriaLoader) -> None:
    """Nenhuma feature deve vir de sancoes_administrativas alem do rotulo em si."""
    features = build_feature_matrix(grande_vitoria_loader)
    colunas_suspeitas = {"tipo", "match_confianca", "data_fim", "valor_multa", "orgao_sancionador"}
    assert colunas_suspeitas.isdisjoint(features.columns)


def test_infracao_ambiental_e_contrato_governamental_so_para_emp5(
    grande_vitoria_loader: GrandeVitoriaLoader,
) -> None:
    features = build_feature_matrix(grande_vitoria_loader)

    assert features.loc["55555555000105", "tem_infracao_ambiental"]
    assert features.loc["55555555000105", "num_infracoes_ambientais"] == 2
    assert features.loc["55555555000105", "valor_multas_ambientais_log1p"] == np.log1p(1500.0)

    assert features.loc["55555555000105", "tem_contrato_governamental"]
    assert features.loc["55555555000105", "num_contratos_governamentais"] == 1
    assert features.loc["55555555000105", "valor_contratos_governamentais_log1p"] == np.log1p(20000.0)

    for cnpj in ["11111111000101", "22222222000102", "33333333000103", "44444444000104"]:
        assert not features.loc[cnpj, "tem_infracao_ambiental"]
        assert not features.loc[cnpj, "tem_contrato_governamental"]


def test_beneficios_fiscais_por_tipo(grande_vitoria_loader: GrandeVitoriaLoader) -> None:
    features = build_feature_matrix(grande_vitoria_loader)

    # emp_5: renuncia fiscal (valor 300.0)
    assert features.loc["55555555000105", "tem_renuncia_fiscal"]
    assert features.loc["55555555000105", "valor_renuncia_fiscal_log1p"] == np.log1p(300.0)
    assert not features.loc["11111111000101", "tem_renuncia_fiscal"]

    # emp_1: habilitado a beneficio fiscal
    assert features.loc["11111111000101", "tem_beneficio_fiscal_habilitado"]
    assert not features.loc["22222222000102", "tem_beneficio_fiscal_habilitado"]

    # emp_2: imune/isento de IRPJ
    assert features.loc["22222222000102", "tem_imune_isento_irpj"]
    assert not features.loc["11111111000101", "tem_imune_isento_irpj"]
