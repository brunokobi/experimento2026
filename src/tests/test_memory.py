"""Testes de consumo de memoria na montagem da HIN.

Usam ``tracemalloc`` para medir o pico de memoria Python alocado durante a
construcao de uma HIN sintetica e comparam contra o limite configurado em
``settings.max_memory_gb``. Servem como guarda-corpo antes de escalar para
os dados reais (milhoes de empresas/socios).
"""

from __future__ import annotations

import tracemalloc

import pytest
import scipy.sparse as sp
import torch

from src.config import get_settings
from src.graph.hin_builder import HINBuilder
from src.graph.metapaths import MetaPath, MetapathExplosionError, SparseMetaPathExtractor

GB = 1024**3


def _build_synthetic_hin(num_empresas: int, num_socios: int, num_cnaes: int = 20) -> HINBuilder:
    builder = HINBuilder()

    empresas = [f"emp_{i}" for i in range(num_empresas)]
    socios = [f"soc_{i}" for i in range(num_socios)]
    cnaes = [f"cnae_{i}" for i in range(num_cnaes)]

    builder.add_node_type("empresa", empresas, features=torch.rand(num_empresas, 8))
    builder.add_node_type("socio", socios, features=torch.rand(num_socios, 4))
    builder.add_node_type("cnae", cnaes)

    socio_edges = [(socios[i % num_socios], empresas[i % num_empresas]) for i in range(num_empresas)]
    cnae_edges = [(empresas[i], cnaes[i % num_cnaes]) for i in range(num_empresas)]

    builder.add_edge_type("socio", "participa_de", "empresa", edges=socio_edges, bidirectional=True)
    builder.add_edge_type("empresa", "atua_em", "cnae", edges=cnae_edges, bidirectional=True)
    return builder


def test_hin_construction_peak_memory_within_budget() -> None:
    """O pico de memoria Python alocado na construcao deve respeitar max_memory_gb."""
    settings = get_settings()
    budget_bytes = settings.max_memory_gb * GB

    tracemalloc.start()
    try:
        _build_synthetic_hin(num_empresas=20_000, num_socios=10_000)
        _current, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()

    assert peak < budget_bytes, (
        f"Pico de memoria ({peak / GB:.3f} GB) excedeu o orcamento configurado "
        f"({settings.max_memory_gb} GB)."
    )


def test_sparse_metapath_scales_and_respects_memory_budget() -> None:
    """A extracao via matriz esparsa (``SparseMetaPathExtractor``) deve rodar
    sem densificar a matriz de comutacao, mesmo numa HIN sintetica de dezenas
    de milhares de nos -- ao contrario do DFS de ``MetaPathExtractor``, que
    nao escala para os volumes reais (ver docs/research_plan.md, secao 6)."""
    settings = get_settings()
    budget_bytes = settings.max_memory_gb * GB
    metapath = MetaPath(
        name="empresa_socio_empresa",
        node_sequence=("empresa", "socio", "empresa"),
        relation_sequence=("rev_participa_de", "participa_de"),
    )

    tracemalloc.start()
    try:
        builder = _build_synthetic_hin(num_empresas=50_000, num_socios=20_000)
        data = builder.build()
        matrix = SparseMetaPathExtractor(data).commuting_matrix(metapath)
        _current, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()

    assert matrix.shape == (50_000, 50_000)
    assert sp.issparse(matrix), "a matriz de comutacao nao deve ser densificada"
    assert peak < budget_bytes, (
        f"Pico de memoria ({peak / GB:.3f} GB) excedeu o orcamento configurado "
        f"({settings.max_memory_gb} GB)."
    )


def test_sparse_metapath_explosion_guard_para_no_hub_baixa_cardinalidade() -> None:
    """Um no 'hub' de baixissima cardinalidade (poucos nos concentrando muitas
    arestas -- como ``municipio`` no dataset real: so 7 nos para 344k
    empresas) faz o produto esparso explodir para perto de denso. Achado real
    ao validar contra o banco de verdade: ``empresa_municipio_empresa``
    tentou alocar ~187 GiB (ver ``scripts/validar_hin_real.py``).
    ``commuting_matrix`` deve recusar com ``MetapathExplosionError`` em vez de
    tentar materializar e estourar memoria.

    Tamanhos dos hubs = distribuicao real dos 7 municipios da Grande Vitoria
    (README.md: Vila Velha 90.865, Serra 85.636, Vitoria 78.872, Cariacica
    51.479, Guarapari 23.710, Viana 11.038, Fundao 2.530). Escolhido de
    proposito, nao arbitrario: com hubs uniformes (ex.: 2x50k) o overflow de
    ``int32`` ainda da um numero "errado" que por coincidencia fica acima do
    limite (o teste passaria mesmo sem o cast para ``int64``); com essa
    distribuicao desigual, o overflow vira **negativo** (``-618548786``), que
    passaria batido pelo limite de 50M sem o cast -- reproduz exatamente o
    bug real encontrado ao validar contra o banco de verdade (ver
    ``scripts/validar_hin_real.py``).
    """
    builder = HINBuilder()
    tamanhos_municipios = [90_865, 85_636, 78_872, 51_479, 23_710, 11_038, 2_530]
    num_empresas = sum(tamanhos_municipios)
    empresas = [f"emp_{i}" for i in range(num_empresas)]
    hubs = [f"municipio_{i}" for i in range(len(tamanhos_municipios))]
    municipio_por_empresa = [
        idx for idx, tamanho in enumerate(tamanhos_municipios) for _ in range(tamanho)
    ]

    builder.add_node_type("empresa", empresas)
    builder.add_node_type("hub", hubs)
    edges = [(empresas[i], hubs[municipio_por_empresa[i]]) for i in range(num_empresas)]
    builder.add_edge_type("empresa", "pertence_a", "hub", edges=edges, bidirectional=True)
    data = builder.build()

    metapath = MetaPath(
        name="empresa_hub_empresa",
        node_sequence=("empresa", "hub", "empresa"),
        relation_sequence=("pertence_a", "rev_pertence_a"),
    )

    with pytest.raises(MetapathExplosionError):
        SparseMetaPathExtractor(data).commuting_matrix(metapath)


def test_memory_scales_sublinearly_with_feature_dtype() -> None:
    """Usar float32 (padrao) deve consumir a metade da memoria de float64 nas features."""
    n = 5_000
    f32 = torch.rand(n, 16, dtype=torch.float32)
    f64 = f32.to(torch.float64)
    assert f64.element_size() == 2 * f32.element_size()
    assert f64.numel() * f64.element_size() == 2 * (f32.numel() * f32.element_size())
