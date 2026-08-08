"""Baseline HAN/HGT (etapa 7.5, ver ``docs/research_plan.md``, secao 7) --
"heterogenea de verdade": ao contrario da GNN homogenea (etapa 7.4, que
colapsa os 3 metapaths num unico tipo de aresta empresa-empresa), aqui cada
tipo de no (``empresa``/``socio``/``endereco``/``vinculo_politico``) e cada
tipo de relacao sao tratados **distintamente**, via ``HGTConv``
(Heterogeneous Graph Transformer, Hu et al. 2020) do PyTorch Geometric --
atencao especifica por tipo de relacao, nao um peso unico compartilhado.

So o no ``empresa`` tem features tabulares reais (mesma matriz da etapa
7.1, para isolar o efeito da estrutura -- ver docstring de
``gnn_homogeneous.py``); os demais tipos de no (``socio``/``endereco``/
``vinculo_politico``/``municipio``) entram so com um embedding aprendido
(``nn.Embedding`` por indice), ja que nao tem feature tabular propria no
dataset -- limitacao usual de qualquer HAN/HGT sobre nos sem atributo.

Mesma integracao transdutiva com o harness que ``gnn_homogeneous.py`` (ver
docstring la para o raciocinio completo): grafo/embeddings/modelo
reinicializados do zero a cada fold, ``fit_predict`` usa o indice (``cnpj``)
de ``x_train``/``x_test`` para saber quais nos treinar/avaliar.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F  # noqa: N812 - convencao universal do PyTorch
from loguru import logger
from torch_geometric.nn import HGTConv, Linear

from src.evaluation.harness import FitPredict
from src.graph.hin_builder import HINBuilder


class _HeteroGNN(torch.nn.Module):
    """``HGTConv`` empilhado, com projecao linear por tipo de no na entrada
    (equaliza a dimensao de features tabulares vs. embeddings aprendidos)."""

    def __init__(
        self,
        metadata: tuple[list[str], list[tuple[str, str, str]]],
        hidden_channels: int = 64,
        num_heads: int = 2,
        num_layers: int = 2,
    ) -> None:
        super().__init__()
        node_types, _edge_types = metadata
        self.lin_dict = torch.nn.ModuleDict({node_type: Linear(-1, hidden_channels) for node_type in node_types})
        self.convs = torch.nn.ModuleList(
            [HGTConv(hidden_channels, hidden_channels, metadata, heads=num_heads) for _ in range(num_layers)]
        )
        self.out = torch.nn.Linear(hidden_channels, 1)

    def forward(
        self, x_dict: dict[str, torch.Tensor], edge_index_dict: dict[tuple[str, str, str], torch.Tensor]
    ) -> torch.Tensor:
        h_dict = {node_type: self.lin_dict[node_type](x).relu() for node_type, x in x_dict.items()}
        for conv in self.convs:
            h_dict = conv(h_dict, edge_index_dict)
        return self.out(h_dict["empresa"]).squeeze(-1)


def make_han_hgt_fit_predict(
    builder: HINBuilder,
    features: pd.DataFrame,
    hidden_channels: int = 32,
    num_heads: int = 1,
    num_layers: int = 2,
    epochs: int = 50,
    lr: float = 0.01,
    random_state: int = 42,
    excluir_relacoes: tuple[str, ...] = ("localizada_em", "rev_localizada_em"),
) -> FitPredict:
    """Constroi a HIN heterogenea + tensor de features do no ``empresa`` uma
    unica vez, e devolve uma funcao ``fit_predict`` compativel com
    ``evaluate_repeated_cv``. Cada chamada reinicializa o modelo (HGTConv +
    projecoes + embeddings dos outros tipos de no) do zero, treina mascarando
    a loss para fora do fold de treino, e prediz para o fold de teste.

    ``features`` deve ser a matriz de ``src.features.tabular.build_feature_matrix``
    (mesmas colunas ``x`` dos outros baselines).

    Defaults conservadores de memoria (``hidden_channels=32``, ``num_heads=1``,
    ``municipio`` excluido por padrao): full-batch HGTConv sobre 344k nos e
    ~1,8M arestas com ``hidden_channels=64``/``num_heads=2`` estourou memoria
    (OOM killer) na maquina local de desenvolvimento (7,8 GB de RAM, bem menos
    que a VPS) -- ``excluir_relacoes`` tira ``municipio`` por padrao (que ja
    nao era metapath de hipotese, mesma decisao da etapa 7.4) para caber na
    memoria disponivel; ajuste os defaults numa maquina com mais RAM/GPU.
    """
    torch.manual_seed(random_state)
    data = builder.data
    cnpjs = builder.external_ids("empresa")
    cnpj_to_idx = {cnpj: i for i, cnpj in enumerate(cnpjs)}

    colunas_x = [c for c in features.columns if c not in ("y_direto", "y_qualquer")]
    x_empresa = torch.tensor(features.loc[cnpjs, colunas_x].to_numpy(dtype=np.float32))

    edge_types = [et for et in data.edge_types if et[1] not in excluir_relacoes]
    tipos_usados = {"empresa"} | {src for src, _rel, _dst in edge_types} | {dst for _src, _rel, dst in edge_types}
    node_types = [nt for nt in data.node_types if nt in tipos_usados]
    metadata = (node_types, edge_types)
    edge_index_dict = {edge_type: data[edge_type].edge_index for edge_type in edge_types}
    outros_tipos = [nt for nt in node_types if nt != "empresa"]

    logger.info(
        f"HAN/HGT: {len(node_types)} tipos de no, {len(edge_types)} tipos de relacao "
        f"({', '.join(outros_tipos)} via embedding aprendido)."
    )

    def fit_predict(x_train: pd.DataFrame, y_train: np.ndarray, x_test: pd.DataFrame) -> np.ndarray:
        train_idx = torch.tensor([cnpj_to_idx[c] for c in x_train.index], dtype=torch.long)
        test_idx = torch.tensor([cnpj_to_idx[c] for c in x_test.index], dtype=torch.long)

        y_train_t = torch.tensor(np.asarray(y_train), dtype=torch.float32)
        n_pos = y_train_t.sum().item()
        n_neg = len(y_train_t) - n_pos
        pos_weight = torch.tensor(n_neg / n_pos if n_pos > 0 else 1.0)

        embeddings = torch.nn.ModuleDict(
            {nt: torch.nn.Embedding(data[nt].num_nodes, hidden_channels) for nt in outros_tipos}
        )
        modelo = _HeteroGNN(metadata, hidden_channels=hidden_channels, num_heads=num_heads, num_layers=num_layers)
        optimizer = torch.optim.Adam(
            list(modelo.parameters()) + list(embeddings.parameters()), lr=lr, weight_decay=5e-4
        )

        def montar_x_dict() -> dict[str, torch.Tensor]:
            x_dict = {"empresa": x_empresa}
            for nt in outros_tipos:
                x_dict[nt] = embeddings[nt].weight
            return x_dict

        modelo.train()
        for _ in range(epochs):
            optimizer.zero_grad()
            logits = modelo(montar_x_dict(), edge_index_dict)
            loss = F.binary_cross_entropy_with_logits(logits[train_idx], y_train_t, pos_weight=pos_weight)
            loss.backward()
            optimizer.step()

        modelo.eval()
        with torch.no_grad():
            scores = torch.sigmoid(modelo(montar_x_dict(), edge_index_dict)[test_idx])
        return scores.numpy()

    return fit_predict
