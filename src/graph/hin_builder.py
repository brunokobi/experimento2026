"""Construcao da HIN (Heterogeneous Information Network) de empresas.

Usa ``torch_geometric.data.HeteroData`` como estrutura principal (eficiente para
treinamento de GNNs) e oferece exportacao/conversao para ``networkx`` (inspecao
visual e algoritmos classicos de grafos) e para o Neo4j (persistencia/consulta).

Exemplo tipico de schema para o dominio de empresas:
    Tipos de no:    "empresa", "socio", "cnae", "municipio"
    Tipos de arco:  ("socio", "participa_de", "empresa")
                    ("empresa", "atua_em", "cnae")
                    ("empresa", "localizada_em", "municipio")
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import networkx as nx
import torch
from loguru import logger
from torch_geometric.data import HeteroData

from src.config import Settings, get_settings


class HINBuilder:
    """Monta e manipula uma HIN de empresas usando ``HeteroData`` como backend."""

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()
        self.data = HeteroData()
        self._node_id_maps: dict[str, dict[Any, int]] = {}

    # ------------------------------------------------------------------ #
    # Construcao
    # ------------------------------------------------------------------ #
    def add_node_type(
        self,
        node_type: str,
        ids: list[Any],
        features: torch.Tensor | None = None,
    ) -> None:
        """Registra um tipo de no e seu mapeamento id-externo -> indice interno.

        Args:
            node_type: nome do tipo de no (ex.: "empresa", "socio", "cnae").
            ids: lista de identificadores externos (ex.: CNPJ, CPF hash, codigo CNAE).
            features: tensor de features (shape ``[num_nodes, num_features]``); opcional.
        """
        id_map = {external_id: idx for idx, external_id in enumerate(ids)}
        self._node_id_maps[node_type] = id_map
        self.data[node_type].num_nodes = len(ids)
        if features is not None:
            if features.shape[0] != len(ids):
                raise ValueError(
                    f"features com {features.shape[0]} linhas != {len(ids)} ids para '{node_type}'"
                )
            self.data[node_type].x = features
        logger.info(f"No '{node_type}': {len(ids)} nos registrados.")

    def add_edge_type(
        self,
        src_type: str,
        relation: str,
        dst_type: str,
        edges: list[tuple[Any, Any]],
        edge_attr: torch.Tensor | None = None,
        bidirectional: bool = False,
    ) -> None:
        """Adiciona um tipo de arco a partir de pares de ids externos (src, dst).

        Ids desconhecidos (fora dos mapeamentos registrados via ``add_node_type``)
        sao ignorados com um aviso, em vez de falhar silenciosamente.
        """
        if src_type not in self._node_id_maps or dst_type not in self._node_id_maps:
            raise KeyError(
                f"Registre os tipos de no '{src_type}' e '{dst_type}' antes do arco '{relation}'."
            )

        src_map = self._node_id_maps[src_type]
        dst_map = self._node_id_maps[dst_type]

        src_idx: list[int] = []
        dst_idx: list[int] = []
        skipped = 0
        for src_id, dst_id in edges:
            if src_id not in src_map or dst_id not in dst_map:
                skipped += 1
                continue
            src_idx.append(src_map[src_id])
            dst_idx.append(dst_map[dst_id])

        if skipped:
            logger.warning(f"{skipped} arcos ({src_type}-{relation}-{dst_type}) ignorados por ids desconhecidos.")

        edge_index = torch.tensor([src_idx, dst_idx], dtype=torch.long)
        self.data[src_type, relation, dst_type].edge_index = edge_index
        if edge_attr is not None:
            self.data[src_type, relation, dst_type].edge_attr = edge_attr
        logger.info(f"Arco '{src_type}-{relation}-{dst_type}': {edge_index.shape[1]} arestas.")

        if bidirectional:
            reverse_relation = f"rev_{relation}"
            self.data[dst_type, reverse_relation, src_type].edge_index = edge_index.flip(0)

    def build(self, validate: bool = True) -> HeteroData:
        """Finaliza a construcao, opcionalmente validando o objeto ``HeteroData``."""
        if validate:
            self.data.validate(raise_on_error=True)
        return self.data

    def external_ids(self, node_type: str) -> list[Any]:
        """Retorna os ids externos de um tipo de no, na ordem do indice
        interno (posicao ``i`` da lista == indice interno ``i`` em
        ``HeteroData``) -- inverso do mapeamento criado em ``add_node_type``.
        Usado na exportacao para Neo4j, onde os nos precisam do id externo
        (CNPJ, chave de socio/endereco etc.), nao do indice interno.
        """
        id_map = self._node_id_maps[node_type]
        ordered: list[Any] = [None] * len(id_map)
        for external_id, idx in id_map.items():
            ordered[idx] = external_id
        return ordered

    # ------------------------------------------------------------------ #
    # Estatisticas / qualidade
    # ------------------------------------------------------------------ #
    def stats(self) -> dict[str, Any]:
        """Retorna um resumo estrutural da HIN (contagens por tipo de no/arco)."""
        return {
            "node_types": {nt: self.data[nt].num_nodes for nt in self.data.node_types},
            "edge_types": {
                "__".join(et): self.data[et].edge_index.shape[1] for et in self.data.edge_types
            },
        }

    # ------------------------------------------------------------------ #
    # Conversao / exportacao
    # ------------------------------------------------------------------ #
    def to_networkx(self) -> nx.MultiDiGraph:
        """Converte a HIN para um ``networkx.MultiDiGraph`` (uso em EDA/algoritmos)."""
        graph = nx.MultiDiGraph()
        for node_type in self.data.node_types:
            for idx in range(self.data[node_type].num_nodes):
                graph.add_node((node_type, idx), node_type=node_type)
        for edge_type in self.data.edge_types:
            src_type, relation, dst_type = edge_type
            edge_index = self.data[edge_type].edge_index
            for src, dst in edge_index.t().tolist():
                graph.add_edge((src_type, src), (dst_type, dst), relation=relation)
        return graph

    def export(self, filename: str = "hin.pt") -> Path:
        """Serializa o ``HeteroData`` em ``settings.data_graph_exports_dir``."""
        out_dir = self._settings.data_graph_exports_dir
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / filename
        torch.save(self.data, path)
        logger.info(f"HIN exportada para: {path}")
        return path

    @classmethod
    def load(cls, path: Path | str, settings: Settings | None = None) -> "HINBuilder":
        """Carrega uma HIN previamente exportada com ``export()``."""
        builder = cls(settings=settings)
        builder.data = torch.load(Path(path), weights_only=False)
        return builder
