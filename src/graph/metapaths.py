"""Definicao e extracao de metapaths sobre a HIN.

Um metapath e uma sequencia de tipos de no conectados por tipos de arco
especificos, ex.: ``empresa -participa_de-1-> socio -participa_de-> empresa``
(empresas com socios em comum).

Duas implementacoes, papeis diferentes (ver docs/research_plan.md, secoes 6/7):

- ``MetaPathExtractor``: percorre a HIN via DFS num ``networkx.MultiDiGraph``
  (gerado por ``HINBuilder.to_networkx``). Da os caminhos completos
  (instancias), uteis para depuracao/inspecao e para cruzar contra uma query
  Cypher no Neo4j -- mas **nao escala**: so serve para amostras pequenas.
- ``SparseMetaPathExtractor``: calcula a matriz de comutacao via produto de
  matrizes de adjacencia esparsas (CSR), direto do ``HeteroData`` -- e o que
  escala para os volumes reais do dataset (344k empresas) e o que de fato
  alimenta a extracao de features/adjacencia para os baselines e a GNN.
"""

from __future__ import annotations

from dataclasses import dataclass

import networkx as nx
import numpy as np
import scipy.sparse as sp
from loguru import logger
from torch_geometric.data import HeteroData


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
    """Extrai instancias (caminhos completos) de metapaths via DFS, para
    depuracao/inspecao em amostras pequenas -- ver ``SparseMetaPathExtractor``
    para a extracao que escala para o dataset real."""

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


def _estimate_product_nnz(a: sp.spmatrix, b: sp.spmatrix) -> int:
    """Estima (limite superior) o numero de entradas nao-nulas de ``a @ b``
    sem materializar o produto -- soma, para cada indice da dimensao
    compartilhada, o produto entre o numero de entradas na coluna
    correspondente de ``a`` e na linha correspondente de ``b``. E o mesmo
    numero que implementacoes de SpGEMM (ex.: scipy) calculam internamente
    antes de alocar memoria -- usado aqui como guarda-corpo, ver
    ``SparseMetaPathExtractor.commuting_matrix``.
    """
    # cast explicito para int64: indptr do scipy vem em int32, e np.dot em
    # int32 estoura silenciosamente (produto de duas colunas ~90k ja excede
    # o limite de int32) -- achado real, ver docstring de commuting_matrix.
    nnz_por_coluna_a = np.diff(a.tocsc().indptr).astype(np.int64)
    nnz_por_linha_b = np.diff(b.tocsr().indptr).astype(np.int64)
    return int(np.dot(nnz_por_coluna_a, nnz_por_linha_b))


class MetapathExplosionError(RuntimeError):
    """Levantado quando um metapath produziria uma matriz de comutacao proxima
    de densa -- tipicamente por atravessar um no "hub" de baixa cardinalidade
    (ex.: municipio: so 7 nos para 344k empresas) -- em vez de deixar o
    processo estourar memoria silenciosamente no meio do produto esparso.
    Achado real ao validar contra o banco de verdade (ver
    ``scripts/validar_hin_real.py``): ``empresa_municipio_empresa`` tentou
    alocar ~187 GiB.
    """


class SparseMetaPathExtractor:
    """Extrai a matriz de comutacao ("commuting matrix") de um metapath via
    produto de matrizes de adjacencia esparsas -- e o que escala para os
    volumes reais do dataset (344k empresas), ao contrario do DFS de
    ``MetaPathExtractor`` acima (que so serve para depuracao em amostras
    pequenas, cruzando o resultado contra uma query Cypher no Neo4j -- ver
    docs/research_plan.md, secao 6).

    Cada passo do metapath vira uma matriz esparsa CSR (linhas = indice do no
    de origem, colunas = indice do no de destino, valor = 1 por arco). O
    resultado e o produto dessas matrizes, na ordem do metapath: a entrada
    ``(i, j)`` conta quantos caminhos ligam o no ``i`` ao no ``j`` seguindo o
    metapath -- ex.: para ``empresa_socio_empresa``, e o numero de socios em
    comum entre as empresas ``i`` e ``j``. A diagonal conta os caminhos de um
    no de volta a ele mesmo (ex.: numero de socios da propria empresa) -- nao
    e sinal de conexao entre empresas distintas, ver ``top_pairs``.

    Nota: o ``indptr`` do scipy vem em ``int32`` (mesmo para matrizes com
    muitas linhas/colunas) -- ``_estimate_product_nnz`` faz cast explicito
    para ``int64`` antes do produto interno para nao estourar silenciosamente
    (achado real: ~90k² sozinho ja excede o limite de ``int32`` e "some" numa
    soma sem cast, deixando a guarda passar batido -- ver
    ``scripts/validar_hin_real.py``).
    """

    def __init__(self, data: HeteroData) -> None:
        self.data = data
        self._adjacency_cache: dict[tuple[str, str, str], sp.csr_matrix] = {}

    def _adjacency(self, edge_type: tuple[str, str, str]) -> sp.csr_matrix:
        """Matriz de adjacencia esparsa (CSR) de um tipo de arco, com cache."""
        if edge_type not in self._adjacency_cache:
            src_type, _relation, dst_type = edge_type
            edge_index = self.data[edge_type].edge_index
            num_src = self.data[src_type].num_nodes
            num_dst = self.data[dst_type].num_nodes
            rows = edge_index[0].numpy()
            cols = edge_index[1].numpy()
            values = np.ones(edge_index.shape[1], dtype=np.float32)
            self._adjacency_cache[edge_type] = sp.csr_matrix((values, (rows, cols)), shape=(num_src, num_dst))
        return self._adjacency_cache[edge_type]

    def commuting_matrix(self, metapath: MetaPath, max_result_nnz: int = 50_000_000) -> sp.csr_matrix:
        """Produto das matrizes de adjacencia esparsas ao longo do metapath.

        Args:
            max_result_nnz: limite de entradas nao-nulas estimadas para o
                produto -- acima disso, levanta ``MetapathExplosionError`` em
                vez de tentar materializar (e provavelmente estourar
                memoria). O default cobre confortavelmente os metapaths de
                hipotese da pesquisa (socio/endereco/vinculo politico) nos
                344k empresas reais; metapaths que atravessam um no de baixa
                cardinalidade (ex.: municipio) devem virar feature
                categorica direta no no ``empresa``, nao matriz de comutacao.
        """
        result: sp.csr_matrix | None = None
        for step, relation in enumerate(metapath.relation_sequence):
            src_type = metapath.node_sequence[step]
            dst_type = metapath.node_sequence[step + 1]
            matrix = self._adjacency((src_type, relation, dst_type))
            if result is None:
                result = matrix
                continue
            estimativa = _estimate_product_nnz(result, matrix)
            if estimativa > max_result_nnz:
                raise MetapathExplosionError(
                    f"Metapath '{metapath.name}': produto via '{relation}' geraria "
                    f"~{estimativa:,} entradas (limite: {max_result_nnz:,}) -- provavel "
                    f"no 'hub' de baixa cardinalidade no caminho (poucos nos concentrando "
                    f"muitas arestas, ex.: municipio). Nao compute esse metapath como "
                    f"matriz de comutacao completa; use agregacao (groupby) ou feature "
                    f"categorica direta no no 'empresa' em vez disso."
                )
            result = result @ matrix
        result = result.tocsr()
        logger.info(
            f"Metapath '{metapath.name}': matriz de comutacao {result.shape}, "
            f"{result.nnz} pares (contando a diagonal) com pelo menos um caminho."
        )
        return result

    def top_pairs(
        self, metapath: MetaPath, k: int | None = 20, min_weight: float = 1.0
    ) -> list[tuple[int, int, float]]:
        """Pares ``(idx_origem, idx_destino, peso)`` com mais caminhos pelo
        metapath, ignorando a diagonal (no ligado a ele mesmo) -- util para
        inspecao/depuracao sem materializar a matriz inteira.
        """
        matrix = self.commuting_matrix(metapath).tocoo()
        mask = (matrix.data >= min_weight) & (matrix.row != matrix.col)
        rows, cols, weights = matrix.row[mask], matrix.col[mask], matrix.data[mask]
        order = np.argsort(-weights)
        if k is not None:
            order = order[:k]
        return list(
            zip(rows[order].tolist(), cols[order].tolist(), weights[order].tolist(), strict=True)
        )


# --- Metapaths de referencia para o dominio de empresas -----------------------
# Os tres primeiros sao os metapaths de hipotese da pergunta de pesquisa
# (docs/research_plan.md, secao 2): socio comum, endereco comum e vinculo
# politico. Nomes de no/relacao devem casar com os produzidos por
# ``src.graph.build_hin.build_empresas_hin``.
#
# "empresa_municipio_empresa" NAO e metapath de hipotese (so contexto/
# visualizacao) e NAO deve ser passado para SparseMetaPathExtractor no banco
# real: municipio e um no "hub" de baixissima cardinalidade (so 7 nos para
# 344k empresas), e o produto esparso explode para perto de denso --
# confirmado ao validar contra o banco real (~187 GiB, ver
# MetapathExplosionError e scripts/validar_hin_real.py). So serve para o DFS
# de MetaPathExtractor em amostras pequenas.
COMMON_METAPATHS = {
    "empresa_socio_empresa": MetaPath(
        name="empresa_socio_empresa",
        node_sequence=("empresa", "socio", "empresa"),
        relation_sequence=("rev_participa_de", "participa_de"),
    ),
    "empresa_endereco_empresa": MetaPath(
        name="empresa_endereco_empresa",
        node_sequence=("empresa", "endereco", "empresa"),
        relation_sequence=("sediada_em", "rev_sediada_em"),
    ),
    "empresa_vinculo_politico_empresa": MetaPath(
        name="empresa_vinculo_politico_empresa",
        node_sequence=("empresa", "vinculo_politico", "empresa"),
        relation_sequence=("tem_vinculo_politico", "rev_tem_vinculo_politico"),
    ),
    "empresa_municipio_empresa": MetaPath(
        name="empresa_municipio_empresa",
        node_sequence=("empresa", "municipio", "empresa"),
        relation_sequence=("localizada_em", "rev_localizada_em"),
    ),
}
