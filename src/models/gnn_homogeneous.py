"""Baseline GNN homogenea (etapa 7.4, ver ``docs/research_plan.md``, secao 7).

Colapsa os 3 metapaths de hipotese (socio comum, endereco comum, vinculo
politico) numa unica matriz de adjacencia empresa-empresa (soma binarizada
das 3 matrizes de comutacao), e treina uma GNN homogenea simples
(GraphSAGE) por cima -- mesmas features tabulares da etapa 7.1 como ``x`` de
cada no, para isolar o efeito da estrutura de rede (nao misturar com
diferenca de features em relacao ao baseline tabular, etapa 7.3).

Por que "GNN homogenea" e nao HAN/HGT aqui: colapsar os 3 tipos de relacao
numa unica testa se so ter QUALQUER conexao de rede ja ajuda, antes de gastar
a complexidade extra de tratar cada metapath como uma relacao distinta
(HAN/HGT, etapa 7.5) -- degrau intermediario da comparacao, nao redundante.

**Achado real que exige poda de hub (ver ``scripts/validar_hin_real.py`` e a
investigacao de 08/08/2026)**: 878 enderecos (~0,5% do total) concentram
~10,8 milhoes das ~12,85 milhoes de arestas empresa-endereco -- sao predios
comerciais grandes/galpoes industriais com centenas de empresas registradas
(o maior tem 1.419), nao "endereco de fachada" compartilhado por poucas
empresas. Sem podar, o grafo homogeneo fica denso demais pra treinar GNN por
passagem de mensagem em tempo viavel (mesmo sem estourar memoria como no
caso de ``municipio`` na etapa 5 -- ali era produto de matriz, aqui e custo
de treino). ``max_grau_endereco`` zera colunas (enderecos) com mais conexoes
que o limiar antes de calcular a comutacao -- decisao documentada, nao
escondida.

Integracao com o harness (``evaluate_repeated_cv``, etapa 7.2): a GNN e
**transdutiva** (passagem de mensagem usa o grafo inteiro; treino e teste
compartilham a mesma vizinhanca) -- diferente do XGBoost tabular, que trata
cada fold como treino/teste totalmente independentes. Por isso
``make_gnn_fit_predict`` constroi o grafo + tensor de features **uma vez**
(fora do fold), e a funcao ``fit_predict`` devolvida usa o indice
(``cnpj``) de ``x_train``/``x_test`` -- que vem do harness ja fatiado por
fold -- so para saber **quais nos** sao treino/teste, nunca por posicao. A
loss e mascarada para os nos de treino; a mensagem passa pelo grafo inteiro
normalmente (transdutivo, como e usual em GNN semi-supervisionada).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
import torch.nn.functional as F  # noqa: N812 - convencao universal do PyTorch
from loguru import logger
from torch_geometric.nn import SAGEConv

from src.evaluation.harness import FitPredict
from src.graph.hin_builder import HINBuilder


class _GraphSAGE(torch.nn.Module):
    """2 camadas de SAGEConv + cabeca linear -- classificador binario
    (logits, nao probabilidade; ``fit_predict`` aplica sigmoid no final)."""

    def __init__(self, in_channels: int, hidden_channels: int = 64) -> None:
        super().__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, hidden_channels)
        self.lin = torch.nn.Linear(hidden_channels, 1)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        h = F.relu(self.conv1(x, edge_index))
        h = F.dropout(h, p=0.3, training=self.training)
        h = F.relu(self.conv2(h, edge_index))
        return self.lin(h).squeeze(-1)


def _adjacency_empresa_para(builder: HINBuilder, node_type: str, relation: str) -> sp.csr_matrix:
    """Adjacencia esparsa ``empresa -> node_type`` direto do ``HeteroData``.

    Duplica parte da logica de ``SparseMetaPathExtractor._adjacency`` de
    proposito: esta funcao precisa podar colunas de alto grau antes do
    produto (``_podar_colunas_de_alto_grau``), o que a API generica do
    extrator nao expoe -- ``SparseMetaPathExtractor`` continua sendo o
    caminho certo quando nao e preciso podar nada.
    """
    data = builder.data
    edge_index = data["empresa", relation, node_type].edge_index
    num_empresas = data["empresa"].num_nodes
    num_dst = data[node_type].num_nodes
    rows, cols = edge_index[0].numpy(), edge_index[1].numpy()
    values = np.ones(len(rows), dtype=np.float32)
    return sp.csr_matrix((values, (rows, cols)), shape=(num_empresas, num_dst))


def _podar_colunas_de_alto_grau(matriz: sp.csr_matrix, grau_maximo: int) -> sp.csr_matrix:
    """Zera colunas (nos do tipo destino) com mais de ``grau_maximo``
    conexoes -- ver achado de hub de endereco na docstring do modulo."""
    csc = matriz.tocsc()
    grau_por_coluna = np.diff(csc.indptr)
    manter = (grau_por_coluna <= grau_maximo).astype(np.float32)
    n_podadas = int((grau_por_coluna > grau_maximo).sum())
    if n_podadas:
        logger.info(f"Podando {n_podadas} colunas com grau > {grau_maximo} antes da comutacao.")
    mascara = sp.diags(manter)
    return (csc @ mascara).tocsr()


_METAPATHS_HOMOGENEO = (
    # (tipo_no, relacao empresa->tipo_no, poda de grau maximo ou None)
    ("socio", "rev_participa_de", None),
    ("endereco", "sediada_em", "max_grau_endereco"),
    ("vinculo_politico", "tem_vinculo_politico", None),
)


def build_combined_adjacency(builder: HINBuilder, max_grau_endereco: int = 20) -> sp.csr_matrix:
    """Soma binarizada das comutacoes dos 3 metapaths de hipotese -- colapsa
    em UM grafo homogeneo empresa-empresa, com os enderecos de altissimo
    grau podados antes (ver docstring do modulo).

    Tolerante a metapath ausente na HIN (ex.: ``vinculo_politico`` so existe
    se ``build_empresas_hin`` recebeu algum vinculo): ignora com aviso em vez
    de falhar, e levanta erro so se **nenhum** dos 3 estiver presente.
    """
    componentes = []
    for node_type, relation, poda in _METAPATHS_HOMOGENEO:
        edge_type = ("empresa", relation, node_type)
        if node_type not in builder.data.node_types or edge_type not in builder.data.edge_types:
            logger.warning(f"No '{node_type}' ausente da HIN -- metapath correspondente ignorado no grafo homogeneo.")
            continue
        adjacencia = _adjacency_empresa_para(builder, node_type, relation)
        if poda == "max_grau_endereco":
            adjacencia = _podar_colunas_de_alto_grau(adjacencia, max_grau_endereco)
        componentes.append(adjacencia @ adjacencia.T)

    if not componentes:
        raise ValueError("Nenhum dos metapaths de hipotese esta presente na HIN -- nada para combinar.")

    total = componentes[0]
    for componente in componentes[1:]:
        total = total + componente
    total.setdiag(0)
    total.eliminate_zeros()
    binaria = (total > 0).astype(np.float32).tocsr()
    logger.info(f"Grafo homogeneo empresa-empresa: {binaria.nnz:,} arestas (direcionadas, grafo simetrico).")
    return binaria


def _edge_index_from_adjacency(adjacency: sp.csr_matrix) -> torch.Tensor:
    coo = adjacency.tocoo()
    return torch.tensor(np.vstack([coo.row, coo.col]), dtype=torch.long)


def make_gnn_fit_predict(
    builder: HINBuilder,
    features: pd.DataFrame,
    max_grau_endereco: int = 20,
    hidden_channels: int = 64,
    epochs: int = 100,
    lr: float = 0.01,
    random_state: int = 42,
) -> FitPredict:
    """Constroi o grafo homogeneo + tensor de features (todos os nos) uma
    unica vez, e devolve uma funcao ``fit_predict`` compativel com
    ``evaluate_repeated_cv`` -- cada chamada treina do zero (pesos
    reiniciados), mascara a loss para os nos que nao sao de treino naquele
    fold, e prediz para os nos de teste no final.

    ``features`` deve ser a matriz de ``src.features.tabular.build_feature_matrix``
    (mesmas colunas ``x`` do baseline tabular) -- ver docstring do modulo.
    """
    torch.manual_seed(random_state)
    cnpjs = builder.external_ids("empresa")

    adjacency = build_combined_adjacency(builder, max_grau_endereco=max_grau_endereco)
    edge_index = _edge_index_from_adjacency(adjacency)

    colunas_x = [c for c in features.columns if c not in ("y_direto", "y_qualquer")]
    x_full = torch.tensor(features.loc[cnpjs, colunas_x].to_numpy(dtype=np.float32))
    cnpj_to_idx = {cnpj: i for i, cnpj in enumerate(cnpjs)}

    def fit_predict(x_train: pd.DataFrame, y_train: np.ndarray, x_test: pd.DataFrame) -> np.ndarray:
        train_idx = torch.tensor([cnpj_to_idx[c] for c in x_train.index], dtype=torch.long)
        test_idx = torch.tensor([cnpj_to_idx[c] for c in x_test.index], dtype=torch.long)

        y_train_t = torch.tensor(np.asarray(y_train), dtype=torch.float32)
        n_pos = y_train_t.sum().item()
        n_neg = len(y_train_t) - n_pos
        pos_weight = torch.tensor(n_neg / n_pos if n_pos > 0 else 1.0)

        model = _GraphSAGE(in_channels=x_full.shape[1], hidden_channels=hidden_channels)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)

        model.train()
        for _ in range(epochs):
            optimizer.zero_grad()
            logits = model(x_full, edge_index)
            loss = F.binary_cross_entropy_with_logits(logits[train_idx], y_train_t, pos_weight=pos_weight)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            scores = torch.sigmoid(model(x_full, edge_index)[test_idx])
        return scores.numpy()

    return fit_predict
