"""Fixtures compartilhadas para os testes da HIN."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import torch
from pytest import fixture

from src.data.loaders import GrandeVitoriaLoader, SQLiteLoader
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


# --- Schema real (subconjunto) do dataset projeto_grande_vitoria_empresas -- #
# Colunas conferidas contra database/schema.sql daquele repo (ver
# docs/research_plan.md, secao 4, e o achado de match_confianca='socio' na
# secao 5). So as colunas usadas por GrandeVitoriaLoader/build_empresas_hin.
_SCHEMA_REAL_SQL = """
CREATE TABLE empresas (
    cnpj TEXT PRIMARY KEY, capital_social REAL, municipio TEXT,
    logradouro TEXT, numero TEXT, cep TEXT,
    porte TEXT, regime_tributario TEXT, cnae_principal TEXT
);
CREATE TABLE socios (
    id INTEGER PRIMARY KEY AUTOINCREMENT, cnpj_empresa TEXT, nome_socio TEXT, cpf_parcial TEXT
);
CREATE TABLE sancoes_administrativas (
    id INTEGER PRIMARY KEY AUTOINCREMENT, cnpj_empresa TEXT, tipo TEXT, match_confianca TEXT
);
CREATE TABLE dividas_ativas (
    id INTEGER PRIMARY KEY AUTOINCREMENT, cnpj_empresa TEXT, valor REAL
);
CREATE TABLE vinculos_politicos (
    id INTEGER PRIMARY KEY AUTOINCREMENT, cnpj_empresa TEXT, nome_socio_vinculado TEXT
);
"""

# 5 empresas: emp_1 e emp_2 compartilham o socio "soc_A" (metapath socio
# comum); emp_1 e emp_3 compartilham endereco (metapath endereco comum);
# emp_1 tem sancao 'direto'; emp_2 tem sancao 'socio' (via emp_1, mesmo socio
# -- reproduz o achado real de circularidade); emp_4 tem vinculo politico;
# emp_5 nao tem nenhum sinal (empresa "limpa").
_EMPRESAS = [
    ("11111111000101", 1000.0, "VITORIA", "RUA A", "10", "29000-000", "ME", "SIMPLES", "4711301"),
    ("22222222000102", 2000.0, "VILA VELHA", "RUA B", "20", "29100-000", "EPP", "SIMPLES", "4711301"),
    ("33333333000103", 3000.0, "SERRA", "RUA A", "10", "29000-000", "ME", "MEI", "6201501"),
    ("44444444000104", 4000.0, "VITORIA", "RUA C", "30", "29200-000", "DEMAIS", "NORMAL", "8112500"),
    ("55555555000105", 5000.0, "SERRA", "RUA D", "40", "29300-000", "ME", "SIMPLES", "4711301"),
]
_SOCIOS = [
    ("11111111000101", "FULANO DE TAL", "123.***.***-45"),
    ("22222222000102", "FULANO DE TAL", "123.***.***-45"),  # mesmo socio de emp_1
    ("33333333000103", "CICLANO SILVA", "987.***.***-65"),
    ("44444444000104", "BELTRANO SOUZA", "555.***.***-11"),
]
_SANCOES = [
    ("11111111000101", "CEIS", "direto"),
    ("22222222000102", "CEIS", "socio"),
]
_DIVIDAS = [("33333333000103", 500.0), ("33333333000103", 250.0)]
_VINCULOS = [("44444444000104", "BELTRANO SOUZA")]


@fixture
def grande_vitoria_loader(tmp_path: Path) -> GrandeVitoriaLoader:
    """``GrandeVitoriaLoader`` sobre um SQLite sintetico com o schema real
    (subconjunto), reproduzindo o achado de circularidade (sancao via socio
    comum) encontrado no banco de verdade -- ver docstring dos dados acima.
    """
    db_path = tmp_path / "grande_vitoria_test.db"
    conn = sqlite3.connect(db_path)
    try:
        conn.executescript(_SCHEMA_REAL_SQL)
        conn.executemany("INSERT INTO empresas VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)", _EMPRESAS)
        conn.executemany("INSERT INTO socios (cnpj_empresa, nome_socio, cpf_parcial) VALUES (?, ?, ?)", _SOCIOS)
        conn.executemany(
            "INSERT INTO sancoes_administrativas (cnpj_empresa, tipo, match_confianca) VALUES (?, ?, ?)",
            _SANCOES,
        )
        conn.executemany("INSERT INTO dividas_ativas (cnpj_empresa, valor) VALUES (?, ?)", _DIVIDAS)
        conn.executemany(
            "INSERT INTO vinculos_politicos (cnpj_empresa, nome_socio_vinculado) VALUES (?, ?)", _VINCULOS
        )
        conn.commit()
    finally:
        conn.close()
    return GrandeVitoriaLoader(loader=SQLiteLoader(db_path=db_path))
