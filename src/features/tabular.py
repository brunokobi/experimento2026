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
- ``tem_infracao_ambiental`` / ``num_infracoes_ambientais`` /
  ``valor_multas_ambientais_log1p`` -- agregado de ``infracoes_ambientais``
  (IBAMA/IEMA), 100% match direto por CNPJ no banco real.
- ``tem_contrato_governamental`` / ``num_contratos_governamentais`` /
  ``valor_contratos_governamentais_log1p`` -- agregado de
  ``contratos_governamentais`` (valor final do contrato).
- ``tem_renuncia_fiscal`` / ``valor_renuncia_fiscal_log1p`` -- renuncia fiscal
  federal (``beneficios_fiscais`` tipo ``RENUNCIA``, unico dos 3 tipos com
  valor monetario por linha).
- ``tem_beneficio_fiscal_habilitado`` -- booleano, habilitada a regime de
  beneficio fiscal federal (tipo ``HABILITADO`` -- ex.: Reidi/Recap/Reporto).
- ``tem_imune_isento_irpj`` -- booleano, imune/isenta de IRPJ (tipo
  ``IMUNE_ISENTO``). **Nota de interpretacao**: majoritariamente entidades
  sem fins lucrativos (associacoes, entidades filantropicas, sindicatos --
  ver ``tipo_entidade`` em ``beneficios_fiscais``) -- e a mesma populacao
  elegivel a sancao CEPIM (que so se aplica a entidades sem fins
  lucrativos). Usar como feature e legitimo (nao e vazamento -- e uma
  caracteristica da empresa que existe independente de qualquer sancao),
  mas capta em parte "a empresa e do tipo que PODE ser CEPIM-sancionada",
  nao necessariamente risco de irregularidade em si -- mencionar essa
  ressalva ao interpretar importancia de feature no modelo.
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


def _agregar_valor_e_contagem(
    df: pd.DataFrame, index: pd.Index, coluna_valor: str, nome_valor_log1p: str, nome_contagem: str
) -> dict[str, pd.Series]:
    """Agrega ``df`` por ``cnpj_empresa`` (soma de ``coluna_valor`` + contagem
    de linhas), reindexado para cobrir todo ``index`` (ausente vira 0) --
    mesmo padrao usado para ``dividas_ativas``, ``infracoes_ambientais`` e
    ``contratos_governamentais``.
    """
    agregado = df.groupby("cnpj_empresa")[coluna_valor].agg(valor="sum", num="count")
    valor = agregado["valor"].reindex(index, fill_value=0.0).clip(lower=0)
    num = agregado["num"].reindex(index, fill_value=0).astype(int)
    return {nome_valor_log1p: np.log1p(valor), nome_contagem: num}


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
    infracoes = loader.infracoes_ambientais()
    contratos = loader.contratos_governamentais()
    renuncias = loader.beneficios_fiscais(tipo="RENUNCIA")
    habilitados = loader.beneficios_fiscais(tipo="HABILITADO")
    imunes = loader.beneficios_fiscais(tipo="IMUNE_ISENTO")
    rotulo = loader.rotulo_sancao().set_index("cnpj_empresa")

    index = empresas.index
    features = pd.DataFrame(index=index)

    capital_social = pd.to_numeric(empresas["capital_social"], errors="coerce").fillna(0.0).clip(lower=0)
    features["capital_social_log1p"] = np.log1p(capital_social)

    num_socios = socios.groupby("cnpj_empresa").size()
    features["num_socios"] = num_socios.reindex(index, fill_value=0).astype(int)

    for coluna, valor in _agregar_valor_e_contagem(
        dividas, index, "valor", "valor_dividas_ativas_log1p", "num_dividas_ativas"
    ).items():
        features[coluna] = valor

    empresas_com_vinculo = set(vinculos["cnpj_empresa"])
    features["tem_vinculo_politico"] = index.isin(empresas_com_vinculo)

    for coluna, valor in _agregar_valor_e_contagem(
        infracoes, index, "valor_multa", "valor_multas_ambientais_log1p", "num_infracoes_ambientais"
    ).items():
        features[coluna] = valor
    features["tem_infracao_ambiental"] = features["num_infracoes_ambientais"] > 0

    for coluna, valor in _agregar_valor_e_contagem(
        contratos, index, "valor_final", "valor_contratos_governamentais_log1p", "num_contratos_governamentais"
    ).items():
        features[coluna] = valor
    features["tem_contrato_governamental"] = features["num_contratos_governamentais"] > 0

    valor_renuncia = (
        renuncias.groupby("cnpj_empresa")["valor"].sum().reindex(index, fill_value=0.0).clip(lower=0)
    )
    features["valor_renuncia_fiscal_log1p"] = np.log1p(valor_renuncia)
    features["tem_renuncia_fiscal"] = index.isin(set(renuncias["cnpj_empresa"]))

    features["tem_beneficio_fiscal_habilitado"] = index.isin(set(habilitados["cnpj_empresa"]))
    features["tem_imune_isento_irpj"] = index.isin(set(imunes["cnpj_empresa"]))

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
