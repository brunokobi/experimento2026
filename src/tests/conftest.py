"""Fixtures compartilhadas para os testes da HIN."""

from __future__ import annotations

import torch
from pytest import fixture

from src.graph.hin_builder import HINBuilder


@fixture
def sample_hin() -> HINBuilder:
    """HIN sintetica pequena: 4 empresas, 3 socios, 2 cnaes, 2 municipios.

    Usada para validar a mecanica de construcao/consulta sem depender de dados
    reais - os testes de qualidade/conectividade/memoria rodam sobre este fixture.
    """
    builder = HINBuilder()

    empresas = ["emp_1", "emp_2", "emp_3", "emp_4"]
    socios = ["soc_1", "soc_2", "soc_3"]
    cnaes = ["cnae_A", "cnae_B"]
    municipios = ["mun_X", "mun_Y"]

    builder.add_node_type("empresa", empresas, features=torch.rand(len(empresas), 4))
    builder.add_node_type("socio", socios, features=torch.rand(len(socios), 2))
    builder.add_node_type("cnae", cnaes)
    builder.add_node_type("municipio", municipios)

    builder.add_edge_type(
        "socio", "participa_de", "empresa",
        edges=[
            ("soc_1", "emp_1"), ("soc_1", "emp_2"),
            ("soc_2", "emp_2"), ("soc_2", "emp_3"),  # soc_2 e ponte entre os dois clusters
            ("soc_3", "emp_3"),
        ],
        bidirectional=True,
    )
    builder.add_edge_type(
        "empresa", "atua_em", "cnae",
        edges=[("emp_1", "cnae_A"), ("emp_2", "cnae_A"), ("emp_3", "cnae_B"), ("emp_4", "cnae_B")],
        bidirectional=True,
    )
    builder.add_edge_type(
        "empresa", "localizada_em", "municipio",
        edges=[("emp_1", "mun_X"), ("emp_2", "mun_X"), ("emp_3", "mun_Y"), ("emp_4", "mun_Y")],
        bidirectional=True,
    )

    return builder
