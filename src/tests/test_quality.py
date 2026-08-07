"""Testes de qualidade estrutural e de schema da HIN.

Cobrem invariantes que nao aparecem em erros de execucao, mas que indicam
problemas silenciosos de qualidade dos dados/pipeline: features com NaN/Inf,
arcos duplicados, mapeamento de ids nao-injetivo e schema minimo esperado.
"""

from __future__ import annotations

import torch

from src.graph.hin_builder import HINBuilder
from src.graph.metapaths import COMMON_METAPATHS, MetaPathExtractor


def test_node_id_mapping_is_injective(sample_hin: HINBuilder) -> None:
    """Cada id externo deve mapear para um indice interno unico, por tipo de no."""
    for node_type, id_map in sample_hin._node_id_maps.items():  # noqa: SLF001 - teste whitebox
        indices = list(id_map.values())
        assert len(indices) == len(set(indices)), f"Indices duplicados para o tipo '{node_type}'"


def test_no_nan_or_inf_in_features(sample_hin: HINBuilder) -> None:
    """Tensores de features nao devem conter NaN ou Inf."""
    data = sample_hin.build()
    for node_type in data.node_types:
        if "x" in data[node_type]:
            x = data[node_type].x
            assert not torch.isnan(x).any(), f"NaN encontrado nas features de '{node_type}'"
            assert not torch.isinf(x).any(), f"Inf encontrado nas features de '{node_type}'"


def test_no_duplicate_edges(sample_hin: HINBuilder) -> None:
    """Nao deve haver arestas duplicadas dentro do mesmo tipo de relacao."""
    data = sample_hin.build()
    for edge_type in data.edge_types:
        edge_index = data[edge_type].edge_index
        pairs = set(zip(edge_index[0].tolist(), edge_index[1].tolist(), strict=True))
        assert len(pairs) == edge_index.shape[1], f"Arcos duplicados em {edge_type}"


def test_expected_schema_is_present(sample_hin: HINBuilder) -> None:
    """O schema minimo de tipos de no e relacao esperado para o dominio deve existir."""
    data = sample_hin.build()
    expected_node_types = {"empresa", "socio", "cnae", "municipio"}
    expected_relations = {"participa_de", "atua_em", "localizada_em"}

    assert expected_node_types.issubset(set(data.node_types))
    present_relations = {relation for (_src, relation, _dst) in data.edge_types}
    assert expected_relations.issubset(present_relations)


def test_metapath_extraction_returns_valid_instances(sample_hin: HINBuilder) -> None:
    """Instancias extraidas de um metapath devem comecar e terminar no tipo esperado."""
    graph = sample_hin.to_networkx()
    extractor = MetaPathExtractor(graph)
    metapath = COMMON_METAPATHS["empresa_cnae_empresa"]

    instances = extractor.extract_instances(metapath)
    assert len(instances) > 0, "Esperava-se ao menos uma instancia de empresa-cnae-empresa"
    for instance in instances:
        assert instance[0][0] == "empresa"
        assert instance[-1][0] == "empresa"
