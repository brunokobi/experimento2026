"""Testes das partes puras da exportacao para Neo4j (conversao de nomes,
``HINBuilder.external_ids``) -- sem depender de um Neo4j de verdade rodando.
A exportacao ponta a ponta e validada manualmente com
``scripts/exportar_hin_neo4j.py`` contra a instancia real (tunel SSH), nao em
CI -- ver docs/research_plan.md, secao 6.
"""

from __future__ import annotations

from src.graph.hin_builder import HINBuilder
from src.graph.neo4j_export import _extra_node_props, _to_label, _to_rel_type


def test_to_label_converte_snake_case_para_pascal_case() -> None:
    assert _to_label("empresa") == "Empresa"
    assert _to_label("vinculo_politico") == "VinculoPolitico"


def test_to_rel_type_converte_para_upper_snake_case() -> None:
    assert _to_rel_type("participa_de") == "PARTICIPA_DE"
    assert _to_rel_type("tem_vinculo_politico") == "TEM_VINCULO_POLITICO"


def test_external_ids_preserva_ordem_do_indice_interno(sample_hin: HINBuilder) -> None:
    ids = sample_hin.external_ids("empresa")
    assert ids == ["emp_1", "emp_2", "emp_3", "emp_4"]
    assert sample_hin.external_ids("municipio") == ["mun_X", "mun_Y"]


def test_extra_node_props_so_existe_para_empresa_e_nunca_e_feature_de_treino(sample_hin: HINBuilder) -> None:
    """`x` (feature de treino) nunca deve ir para o Neo4j -- so o rotulo,
    quando presente."""
    sample_hin.build(validate=False)
    assert _extra_node_props(sample_hin, "socio") == {}
    assert _extra_node_props(sample_hin, "empresa") == {}  # fixture sintetica nao tem y_direto/y_qualquer
