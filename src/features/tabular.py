"""Feature engineering tabular para os baselines (etapa 7.1, ver
``docs/research_plan.md``, secao 7).

Extrai uma matriz de features por empresa a partir do ``GrandeVitoriaLoader``,
pronta para os tres baselines: alimenta direto o baseline tabular (etapa 7.3)
e serve de features de no para a GNN homogenea/HAN-HGT (etapas 7.4/7.5).

Features incluidas (nenhuma derivada de sancao):
- ``capital_social_log1p`` -- capital social, log1p (distribuicao bem
  assimetrica, log estabiliza).
- ``num_socios`` -- contagem de socios da empresa.
- ``valor_dividas_ativas_log1p`` / ``num_dividas_ativas`` -- agregado de
  ``dividas_ativas`` (mesma logica de ``src.graph.build_hin``, reaproveitada
  aqui para o baseline tabular).
- ``tem_vinculo_politico`` -- booleano, empresa tem qualquer vinculo
  politico (TSE) registrado.
- ``porte``, ``regime_tributario``, ``municipio``, ``cnae_segmento`` (prefixo
  de 2 digitos do CNAE principal) -- categoricas, one-hot. Nota: no banco
  real ``porte`` vem como **codigo** da Receita (``"01"``, ``"03"``, ``"05"``
  -- nao texto tipo "ME"/"EPP"), confirmado ao validar contra o banco
  (``uv run python -c "from src.features.tabular import build_feature_matrix; ..."``);
  interpretar resultados por ``porte_XX`` exige consultar a tabela de
  codigos da Receita, nao e auto-explicativo pela coluna.
- ``num_socios``/``num_dividas_ativas`` tem cauda longa (validado contra o
  banco real: maximo de 1035 socios e 986 dividas ativas numa unica
  empresa) -- plausivel (holding/conglomerado, empresa antiga com muitas
  inscricoes), nao e erro de dado, mas considerar normalizacao/clipping
  antes de usar em modelos sensiveis a outlier (ex.: regressao logistica).

Exclusoes deliberadas (risco de vazamento ou ruido -- ver
``docs/research_plan.md``, secoes 5 e 9):
- ``situacao_cadastral`` -- pode estar entrelacada com o proprio desfecho de
  sancao (ex.: CEPIM = entidades sem fins lucrativos *impedidas*); o universo
  já é filtrado para empresas ativas na origem, risco de vazamento sem ganho
  de sinal.
- qualquer coluna de ``sancoes_administrativas`` alem do rotulo em si (nunca
  ``data_fim``, ``valor_multa`` etc. como feature).
- ``processos_judiciais`` -- ainda fora da HIN, ruidoso por design.
- datas de atualizacao cadastral (``data_situacao``,
  ``data_ultima_atualizacao``) -- refletem quando o cadastro foi tocado pela
  ultima vez, nao sinal de risco.
- idade da empresa (``registros_jucees.data_constituicao``) -- fica de fora
  desta primeira versao (tabela ainda nao tem loader dedicado); TODO natural
  para uma proxima iteracao de feature engineering.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from loguru import logger

from src.data.loaders import GrandeVitoriaLoader

_COLUNAS_CATEGORICAS = ("porte", "regime_tributario", "municipio", "cnae_segmento")


def _cnae_segmento(cnae_principal: pd.Series) -> pd.Series:
    """Prefixo de 2 digitos do CNAE principal -- reduz cardinalidade (mesma
    convencao de ``config.SEGMENTOS_CNAE`` no repo do dataset)."""
    return cnae_principal.fillna("").astype(str).str.slice(0, 2).replace("", "desconhecido")


def build_feature_matrix(loader: GrandeVitoriaLoader | None = None) -> pd.DataFrame:
    """Monta a matriz de features tabulares por empresa (index = ``cnpj``),
    junto com os rotulos ``y_direto``/``y_qualquer`` (ver
    ``GrandeVitoriaLoader.rotulo_sancao`` para a definicao de cada um e o
    risco de circularidade do segundo).

    Cobre todo o universo de ``empresas``, nao so as com dado auxiliar
    (sócio/dívida/vínculo ausente vira ``0``/``False`` via ``reindex``).
    """
    loader = loader or GrandeVitoriaLoader()

    empresas = loader.empresas().set_index("cnpj")
    socios = loader.socios()
    dividas = loader.dividas_ativas()
    vinculos = loader.vinculos_politicos()
    rotulo = loader.rotulo_sancao().set_index("cnpj_empresa")

    index = empresas.index
    features = pd.DataFrame(index=index)

    capital_social = pd.to_numeric(empresas["capital_social"], errors="coerce").fillna(0.0).clip(lower=0)
    features["capital_social_log1p"] = np.log1p(capital_social)

    num_socios = socios.groupby("cnpj_empresa").size()
    features["num_socios"] = num_socios.reindex(index, fill_value=0).astype(int)

    dividas_por_empresa = (
        dividas.groupby("cnpj_empresa")["valor"].agg(valor_dividas_ativas="sum", num_dividas_ativas="count")
    )
    valor_dividas = dividas_por_empresa["valor_dividas_ativas"].reindex(index, fill_value=0.0).clip(lower=0)
    features["valor_dividas_ativas_log1p"] = np.log1p(valor_dividas)
    features["num_dividas_ativas"] = dividas_por_empresa["num_dividas_ativas"].reindex(index, fill_value=0).astype(
        int
    )

    empresas_com_vinculo = set(vinculos["cnpj_empresa"])
    features["tem_vinculo_politico"] = index.isin(empresas_com_vinculo)

    categoricas = pd.DataFrame(
        {
            "porte": empresas["porte"].fillna("desconhecido"),
            "regime_tributario": empresas["regime_tributario"].fillna("desconhecido"),
            "municipio": empresas["municipio"].fillna("desconhecido"),
            "cnae_segmento": _cnae_segmento(empresas["cnae_principal"]),
        },
        index=index,
    )
    features = features.join(pd.get_dummies(categoricas, columns=list(_COLUNAS_CATEGORICAS)))

    rotulo_por_empresa = rotulo.reindex(index)
    features["y_direto"] = rotulo_por_empresa["y_direto"].fillna(False).astype(bool)
    features["y_qualquer"] = rotulo_por_empresa["y_qualquer"].fillna(False).astype(bool)

    logger.info(
        f"Matriz de features: {features.shape[0]} empresas x "
        f"{features.shape[1] - 2} features (+ 2 colunas de rotulo)."
    )
    return features
