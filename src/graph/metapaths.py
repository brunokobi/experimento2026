"""Definicao e extracao de metapaths sobre a HIN.

Um metapath e uma sequencia de tipos de no conectados por tipos de arco
especificos, ex.: ``empresa -atua_em-> cnae -atua_em-1-> empresa`` (empresas
que compartilham o mesmo CNAE) ou ``empresa -participa_de-1-> socio
-participa_de-> empresa`` (empresas com socios em comum).

``MetaPathExtractor`` percorre a HIN (via ``networkx.MultiDiGraph``, gerado por
``HINBuilder.to_networkx``) seguindo a sequencia de relacoes do metapath e
retorna os pares de nos-alvo alcancados, alem de poder materializar a matriz
de adjacencia commuting resultante.
"""

from __future__ import annotations

from dataclasses import dataclass

import networkx as nx
from loguru import logger


@dataclass(frozen=True)
class MetaPath:
    """Representa um metapath como uma sequencia de (tipo_no, relacao, tipo_no).

    Attributes:
        name: identificador legivel do metapath (ex.: "empresa_cnae_empresa").
        node_sequence: sequencia de tipos de no, ex.: ["empresa", "cnae", "empresa"].
        relation_sequence: sequencia de relacoes entre nos consecutivos,
            com o mesmo tamanho de ``node_sequence`` menos 1.
    """

    name: str
    node_sequence: tuple[str, ...]
    relation_sequence: tuple[str, ...]

    def __post_init__(self) -> None:
        if len(self.relation_sequence) != len(self.node_sequence) - 1:
            raise ValueError(
                "relation_sequence deve ter exatamente len(node_sequence) - 1 elementos."
            )

    def __len__(self) -> int:
        return len(self.relation_sequence)


class MetaPathExtractor:
    """Extrai instancias de metapaths de um ``networkx.MultiDiGraph`` heterogeneo."""

    def __init__(self, graph: nx.MultiDiGraph) -> None:
        self.graph = graph

    def extract_instances(self, metapath: MetaPath, max_instances: int | None = None) -> list[tuple]:
        """Retorna todas as instancias (caminhos completos) que satisfazem o metapath.

        Cada instancia e uma tupla de nos ``(node_type, idx)`` com o mesmo tamanho
        de ``metapath.node_sequence``. A busca e feita por expansao em profundidade
        (DFS) respeitando o tipo de no e a relacao esperados em cada passo.
        """
        start_type = metapath.node_sequence[0]
        candidates = [n for n, data in self.graph.nodes(data=True) if data.get("node_type") == start_type]

        instances: list[tuple] = []
        for start_node in candidates:
            self._expand(start_node, metapath, step=0, path=(start_node,), out=instances)
            if max_instances is not None and len(instances) >= max_instances:
                break

        logger.info(f"Metapath '{metapath.name}': {len(instances)} instancias encontradas.")
        return instances[:max_instances] if max_instances is not None else instances

    def _expand(self, current: tuple, metapath: MetaPath, step: int, path: tuple, out: list[tuple]) -> None:
        if step == len(metapath):
            out.append(path)
            return

        expected_relation = metapath.relation_sequence[step]
        expected_next_type = metapath.node_sequence[step + 1]

        for _, neighbor, data in self.graph.out_edges(current, data=True):
            if data.get("relation") != expected_relation:
                continue
            if self.graph.nodes[neighbor].get("node_type") != expected_next_type:
                continue
            self._expand(neighbor, metapath, step + 1, path + (neighbor,), out)

    def commuting_matrix_pairs(self, metapath: MetaPath, max_instances: int | None = None) -> list[tuple]:
        """Retorna apenas os pares (no_inicial, no_final) de cada instancia do metapath.

        Util para construir a matriz de adjacencia "commuting" (ex.: empresas
        conectadas via CNAE comum), usada como entrada para GNNs homogeneas
        ou para analises de similaridade.
        """
        instances = self.extract_instances(metapath, max_instances=max_instances)
        return [(instance[0], instance[-1]) for instance in instances]


# --- Metapaths de referencia para o dominio de empresas -----------------------
COMMON_METAPATHS = {
    "empresa_cnae_empresa": MetaPath(
        name="empresa_cnae_empresa",
        node_sequence=("empresa", "cnae", "empresa"),
        relation_sequence=("atua_em", "rev_atua_em"),
    ),
    "empresa_socio_empresa": MetaPath(
        name="empresa_socio_empresa",
        node_sequence=("empresa", "socio", "empresa"),
        relation_sequence=("rev_participa_de", "participa_de"),
    ),
    "empresa_municipio_empresa": MetaPath(
        name="empresa_municipio_empresa",
        node_sequence=("empresa", "municipio", "empresa"),
        relation_sequence=("localizada_em", "rev_localizada_em"),
    ),
}
