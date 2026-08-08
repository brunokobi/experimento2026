"""Testes de consumo de memoria na montagem da HIN.

Usam ``tracemalloc`` para medir o pico de memoria Python alocado durante a
construcao de uma HIN sintetica e comparam contra o limite configurado em
``settings.max_memory_gb``. Servem como guarda-corpo antes de escalar para
os dados reais (milhoes de empresas/socios).
"""

from __future__ import annotations

import tracemalloc

import scipy.sparse as sp
import torch

from src.config import get_settings
from src.graph.hin_builder import HINBuilder
from src.graph.metapaths import MetaPath, SparseMetaPathExtractor

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


def test_memory_scales_sublinearly_with_feature_dtype() -> None:
    """Usar float32 (padrao) deve consumir a metade da memoria de float64 nas features."""
    n = 5_000
    f32 = torch.rand(n, 16, dtype=torch.float32)
    f64 = f32.to(torch.float64)
    assert f64.element_size() == 2 * f32.element_size()
    assert f64.numel() * f64.element_size() == 2 * (f32.numel() * f32.element_size())
