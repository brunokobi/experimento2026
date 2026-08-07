"""Testes de conectividade estrutural da HIN.

Verificam que a HIN construida forma uma estrutura navegavel: sem nos
isolados inesperados, indices de arco dentro dos limites, e que a conversao
para ``networkx`` preserva a topologia.
"""

from __future__ import annotations

import networkx as nx

from src.graph.hin_builder import HINBuilder


def test_no_isolated_nodes(sample_hin: HINBuilder) -> None:
    """Nenhum no deve ficar isolado (sem nenhuma arco de entrada ou saida)."""
    graph = sample_hin.to_networkx()
    isolated = list(nx.isolates(graph))
    assert not isolated, f"Nos isolados encontrados: {isolated}"


def test_weakly_connected_as_undirected(sample_hin: HINBuilder) -> None:
    """A HIN de exemplo deve formar um unico componente quando tratada como nao-dirigida."""
    graph = sample_hin.to_networkx()
    undirected = graph.to_undirected()
    components = list(nx.connected_components(undirected))
    assert len(components) == 1, f"Esperado 1 componente conectado, encontrados {len(components)}"


def test_edge_indices_within_bounds(sample_hin: HINBuilder) -> None:
    """Todos os indices em edge_index devem ser < num_nodes do tipo correspondente."""
    data = sample_hin.build()
    for edge_type in data.edge_types:
        src_type, _relation, dst_type = edge_type
        edge_index = data[edge_type].edge_index
        assert edge_index.numel() == 0 or edge_index[0].max().item() < data[src_type].num_nodes
        assert edge_index.numel() == 0 or edge_index[1].max().item() < data[dst_type].num_nodes


def test_reverse_edges_are_symmetric(sample_hin: HINBuilder) -> None:
    """Para cada arco bidirecional, o arco reverso deve ter o mesmo numero de arestas."""
    data = sample_hin.build()
    forward = data["socio", "participa_de", "empresa"].edge_index
    reverse = data["empresa", "rev_participa_de", "socio"].edge_index
    assert forward.shape[1] == reverse.shape[1]
