"""Testes do pipeline real: GrandeVitoriaLoader + build_empresas_hin.

Usam o fixture ``grande_vitoria_loader`` (SQLite sintetico no schema real,
ver ``conftest.py``), que reproduz deliberadamente o achado de circularidade
do rotulo (sancao atribuida via socio comum) encontrado no banco de verdade
-- ver ``docs/research_plan.md``, secoes 5 e 9.
"""

from __future__ import annotations

from src.data.loaders import GrandeVitoriaLoader
from src.graph.build_hin import _chave_endereco, _chave_socio, _normalizar_texto, build_empresas_hin
from src.graph.metapaths import COMMON_METAPATHS, MetaPathExtractor, SparseMetaPathExtractor


def test_normalizacao_trata_nan_do_pandas_como_vazio() -> None:
    """``pandas.read_sql_query`` devolve ``NaN`` (float), nao ``None``, para
    colunas de texto nulas -- achado ao validar contra o banco real
    (``scripts/validar_hin_real.py``). ``nan or ""`` NAO pega esse caso
    (``NaN`` e *truthy* em Python); regressao para esse bug."""
    nan = float("nan")
    assert _normalizar_texto(nan) == ""
    assert _normalizar_texto(None) == ""
    assert _chave_socio(nan, "FULANO") == "|FULANO"
    assert _chave_socio(nan, nan) == "|"
    assert _chave_endereco(nan, "10", nan) == "|10|"


def test_rotulo_sancao_separa_direto_de_qualquer(grande_vitoria_loader: GrandeVitoriaLoader) -> None:
    """y_direto so deve marcar emp_1; y_qualquer deve marcar emp_1 e emp_2 (via socio)."""
    rotulo = grande_vitoria_loader.rotulo_sancao().set_index("cnpj_empresa")

    assert rotulo.loc["11111111000101", "y_direto"]
    assert not rotulo.loc["22222222000102", "y_direto"], "emp_2 nao tem sancao direta"
    assert rotulo.loc["22222222000102", "y_qualquer"], "emp_2 deve entrar em y_qualquer (via socio)"
    assert not rotulo.loc["33333333000103", "y_qualquer"], "emp_3 nao tem nenhuma sancao"

    # cobre todo o universo de empresas, nao so as sancionadas
    assert len(rotulo) == 5


def test_sancoes_administrativas_filtra_por_match_confianca(grande_vitoria_loader: GrandeVitoriaLoader) -> None:
    diretas = grande_vitoria_loader.sancoes_administrativas(match_confianca="direto")
    via_socio = grande_vitoria_loader.sancoes_administrativas(match_confianca="socio")

    assert diretas["cnpj_empresa"].tolist() == ["11111111000101"]
    assert via_socio["cnpj_empresa"].tolist() == ["22222222000102"]


def test_build_empresas_hin_node_and_edge_counts(grande_vitoria_loader: GrandeVitoriaLoader) -> None:
    builder = build_empresas_hin(grande_vitoria_loader)
    data = builder.build()

    assert data["empresa"].num_nodes == 5
    assert data["socio"].num_nodes == 3  # soc comum (emp_1/emp_2) + 2 outros
    assert data["endereco"].num_nodes == 4  # emp_1/emp_3 compartilham endereco
    assert data["municipio"].num_nodes == 3  # VITORIA, VILA VELHA, SERRA
    assert data["vinculo_politico"].num_nodes == 1

    assert data["socio", "participa_de", "empresa"].edge_index.shape[1] == 4
    assert data["empresa", "sediada_em", "endereco"].edge_index.shape[1] == 5


def test_build_empresas_hin_label_preserva_circularidade(grande_vitoria_loader: GrandeVitoriaLoader) -> None:
    """y_direto/y_qualquer no HeteroData devem bater com o rotulo do loader, na
    mesma ordem de ``empresas.cnpj``."""
    builder = build_empresas_hin(grande_vitoria_loader)
    data = builder.build()

    cnpjs_em_ordem = grande_vitoria_loader.empresas()["cnpj"].tolist()
    idx_emp1 = cnpjs_em_ordem.index("11111111000101")
    idx_emp2 = cnpjs_em_ordem.index("22222222000102")
    idx_emp3 = cnpjs_em_ordem.index("33333333000103")

    assert bool(data["empresa"].y_direto[idx_emp1]) is True
    assert bool(data["empresa"].y_direto[idx_emp2]) is False
    assert bool(data["empresa"].y_qualquer[idx_emp2]) is True
    assert bool(data["empresa"].y_qualquer[idx_emp3]) is False


def test_metapath_socio_comum_liga_emp1_e_emp2(grande_vitoria_loader: GrandeVitoriaLoader) -> None:
    """O metapath empresa-socio-empresa deve encontrar o par emp_1/emp_2 (mesmo socio)."""
    builder = build_empresas_hin(grande_vitoria_loader)
    graph = builder.to_networkx()
    extractor = MetaPathExtractor(graph)

    pares = extractor.commuting_matrix_pairs(COMMON_METAPATHS["empresa_socio_empresa"])
    empresas = grande_vitoria_loader.empresas()["cnpj"].tolist()
    idx_emp1, idx_emp2 = empresas.index("11111111000101"), empresas.index("22222222000102")

    assert (("empresa", idx_emp1), ("empresa", idx_emp2)) in pares


def test_metapath_endereco_comum_liga_emp1_e_emp3(grande_vitoria_loader: GrandeVitoriaLoader) -> None:
    """O metapath empresa-endereco-empresa deve encontrar o par emp_1/emp_3 (mesmo endereco)."""
    builder = build_empresas_hin(grande_vitoria_loader)
    graph = builder.to_networkx()
    extractor = MetaPathExtractor(graph)

    pares = extractor.commuting_matrix_pairs(COMMON_METAPATHS["empresa_endereco_empresa"])
    empresas = grande_vitoria_loader.empresas()["cnpj"].tolist()
    idx_emp1, idx_emp3 = empresas.index("11111111000101"), empresas.index("33333333000103")

    assert (("empresa", idx_emp1), ("empresa", idx_emp3)) in pares


def test_sparse_commuting_matrix_socio_comum_bate_com_dfs(grande_vitoria_loader: GrandeVitoriaLoader) -> None:
    """A matriz de comutacao esparsa deve concordar com o DFS: emp_1/emp_2 tem
    exatamente 1 socio em comum (fora da diagonal), e a matriz e simetrica."""
    builder = build_empresas_hin(grande_vitoria_loader)
    data = builder.build()
    extractor = SparseMetaPathExtractor(data)

    empresas = grande_vitoria_loader.empresas()["cnpj"].tolist()
    idx_emp1, idx_emp2 = empresas.index("11111111000101"), empresas.index("22222222000102")

    matrix = extractor.commuting_matrix(COMMON_METAPATHS["empresa_socio_empresa"])
    assert matrix.shape == (5, 5)
    assert matrix[idx_emp1, idx_emp2] == 1
    assert matrix[idx_emp2, idx_emp1] == 1  # simetrica
    assert matrix[idx_emp1, idx_emp1] == 1  # diagonal: emp_1 tem 1 socio


def test_sparse_top_pairs_ignora_diagonal_e_acha_endereco_comum(grande_vitoria_loader: GrandeVitoriaLoader) -> None:
    builder = build_empresas_hin(grande_vitoria_loader)
    data = builder.build()
    extractor = SparseMetaPathExtractor(data)

    empresas = grande_vitoria_loader.empresas()["cnpj"].tolist()
    idx_emp1, idx_emp3 = empresas.index("11111111000101"), empresas.index("33333333000103")

    pares = extractor.top_pairs(COMMON_METAPATHS["empresa_endereco_empresa"])
    assert all(origem != destino for origem, destino, _peso in pares), "diagonal nao deveria aparecer"
    assert (idx_emp1, idx_emp3, 1.0) in pares or (idx_emp3, idx_emp1, 1.0) in pares
