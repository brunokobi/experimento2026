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
- ``grau_socio_comum`` / ``grau_endereco_comum`` / ``grau_vinculo_politico_comum``
  -- grau da empresa em cada metapath de hipotese (numero de empresas
  DISTINTAS conectadas via sócio/endereco/vinculo politico comum), calculado
  direto da HIN (``SparseMetaPathExtractor``). Feature de grafo *explicita*
  pro modelo tabular, em vez de so deixar a GNN aprender a estrutura
  implicitamente -- literatura de deteccao de fraude mostra que features de
  grafo (grau, centralidade) alimentando gradient boosting costumam
  igualar ou superar GNN end-to-end nesse tipo de tarefa (ver
  docs/research_plan.md, secao 7, etapa 7.7). ``grau_endereco_comum`` usa a
  mesma poda de hub de alto grau da GNN homogenea (``max_grau_endereco=20``,
  etapa 7.4) -- sem isso, predios comerciais grandes dominariam a feature
  com ruido, nao sinal.
- ``grau_do_socio`` -- concentracao do socio mais conectado da empresa (em
  quantas OUTRAS empresas ele tambem aparece) -- proxy de "poucos diretores
  controlando muitas empresas", indicador de risco de empresa de fachada
  citado na literatura (ex.: indicadores de shell company da Moody's).
  Diferente de ``grau_socio_comum``: este conta so o socio mais conectado,
  nao a uniao de empresas alcancaveis por qualquer socio.
- ``tem_contrato_sem_competicao`` -- booleano, tem contrato publico via
  modalidade sem concorrencia (``Dispensa de Licitação``/``Inexigibilidade
  de Licitação``) -- red flag classico na literatura de risco de corrupcao
  em compras publicas (Fazekas & Kocsis, 2020, British Journal of Political
  Science). **Cobertura muito baixa no banco
  real**: so 5 empresas -- a maioria dos contratos (834/894) nao tem
  modalidade informada (``"Sem Informação"``); nao esperar impacto grande
  no modelo so por essa feature.
- ``sobrepreco_contrato_max`` -- maior divergencia relativa entre
  ``valor_final`` e ``valor_inicial`` dos contratos da empresa
  (``(valor_final - valor_inicial) / valor_inicial``) -- sobrepreco/aditivo
  contratual, outro red flag classico da mesma literatura.
- ``idade_empresa_anos`` -- idade da empresa em anos, a partir de
  ``registros_jucees.data_constituicao`` (referencia fixa: 31/07/2026, mes
  de extracao do dataset -- nao ``hoje()``, pra nao mudar o valor a cada
  vez que o codigo roda). Empresa muito nova e indicador de risco de
  "empresa de fachada" na literatura (Moody's). Empresas sem registro em
  ``registros_jucees`` (255.781 das 344.130 -- so 25,7% tem esse dado)
  recebem ``-1`` como sinalizador de "desconhecido" (nao ``0``, que seria
  confundido com "empresa constituida ha 0 dias").
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
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import scipy.sparse as sp
from loguru import logger

from src.data.loaders import GrandeVitoriaLoader
from src.graph.build_hin import _chave_socio, build_empresas_hin
from src.graph.hin_builder import HINBuilder

_COLUNAS_CATEGORICAS = ("porte", "regime_tributario", "municipio", "cnae_segmento")
_MAX_GRAU_ENDERECO = 20  # mesma poda de hub de alto grau da GNN homogenea (etapa 7.4)
_DATA_REFERENCIA_IDADE_EMPRESA = pd.Timestamp("2026-07-31")  # mes de extracao do dataset, nao hoje()


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


def _grau_do_socio(socios: pd.DataFrame, index: pd.Index) -> pd.Series:
    """Concentracao do socio mais conectado da empresa (em quantas OUTRAS
    empresas ele tambem aparece) -- ver docstring do modulo. Usa a mesma
    chave de identidade de socio do ``build_hin.py`` (CPF mascarado + nome
    normalizado)."""
    socios = socios.copy()
    socios["chave_socio"] = [
        _chave_socio(cpf, nome) for cpf, nome in zip(socios["cpf_parcial"], socios["nome_socio"], strict=True)
    ]
    num_empresas_por_socio = socios.groupby("chave_socio")["cnpj_empresa"].nunique()
    socios["grau_do_socio"] = socios["chave_socio"].map(num_empresas_por_socio)
    maior_grau_por_empresa = socios.groupby("cnpj_empresa")["grau_do_socio"].max()
    # -1: nao contar a propria empresa no grau do socio mais conectado
    return (maior_grau_por_empresa - 1).reindex(index, fill_value=0).clip(lower=0).astype(int)


def _adjacencia_empresa_para(builder: HINBuilder, node_type: str, relation: str) -> sp.csr_matrix:
    """Duplica ``src.models.gnn_homogeneous._adjacency_empresa_para`` de
    proposito -- ver docstring la para o motivo (poda de coluna de alto grau
    antes do produto nao e suportado pela API generica de
    ``SparseMetaPathExtractor``)."""
    data = builder.data
    edge_index = data["empresa", relation, node_type].edge_index
    num_empresas = data["empresa"].num_nodes
    num_dst = data[node_type].num_nodes
    rows, cols = edge_index[0].numpy(), edge_index[1].numpy()
    values = np.ones(len(rows), dtype=np.float32)
    return sp.csr_matrix((values, (rows, cols)), shape=(num_empresas, num_dst))


def _podar_colunas_de_alto_grau(matriz: sp.csr_matrix, grau_maximo: int) -> sp.csr_matrix:
    """Duplica ``src.models.gnn_homogeneous._podar_colunas_de_alto_grau``."""
    csc = matriz.tocsc()
    grau_por_coluna = np.diff(csc.indptr)
    manter = (grau_por_coluna <= grau_maximo).astype(np.float32)
    return (csc @ sp.diags(manter)).tocsr()


def _grau_metapath(builder: HINBuilder, index: pd.Index) -> pd.DataFrame:
    """Grau de cada empresa em cada metapath de hipotese (numero de empresas
    DISTINTAS conectadas, nao soma de caminhos) -- ver docstring do modulo."""
    cnpjs = builder.external_ids("empresa")
    graus = pd.DataFrame(index=cnpjs)

    especificacoes = [
        ("socio", "rev_participa_de", None, "grau_socio_comum"),
        ("endereco", "sediada_em", _MAX_GRAU_ENDERECO, "grau_endereco_comum"),
        ("vinculo_politico", "tem_vinculo_politico", None, "grau_vinculo_politico_comum"),
    ]
    for node_type, relation, poda, nome_coluna in especificacoes:
        edge_type = ("empresa", relation, node_type)
        if node_type not in builder.data.node_types or edge_type not in builder.data.edge_types:
            graus[nome_coluna] = 0
            continue
        adjacencia = _adjacencia_empresa_para(builder, node_type, relation)
        if poda is not None:
            adjacencia = _podar_colunas_de_alto_grau(adjacencia, poda)
        comutacao = (adjacencia @ adjacencia.T).tocsr()
        comutacao.setdiag(0)
        comutacao.eliminate_zeros()
        graus[nome_coluna] = np.asarray((comutacao > 0).sum(axis=1)).flatten()

    return graus.reindex(index, fill_value=0)


def build_feature_matrix(
    loader: GrandeVitoriaLoader | None = None, builder: HINBuilder | None = None
) -> pd.DataFrame:
    """Monta a matriz de features tabulares por empresa (index = ``cnpj``),
    junto com os rotulos ``y_direto``/``y_qualquer`` (ver
    ``GrandeVitoriaLoader.rotulo_sancao`` para a definicao de cada um e o
    risco de circularidade do segundo).

    Cobre todo o universo de ``empresas``, nao so as com dado auxiliar
    (sócio/dívida/vínculo ausente vira ``0``/``False`` via ``reindex``).

    Args:
        loader: fonte dos dados tabulares; se omitido, cria um a partir de
            ``Settings``.
        builder: HIN ja construida, usada para as features de grafo
            (``grau_socio_comum`` etc.); se omitido, constroi uma internamente
            via ``build_empresas_hin(loader)`` (custo extra de ~8s -- se o
            chamador ja tem uma HIN construida por outro motivo, ex.:
            ``make_gnn_fit_predict``, passe-a aqui para nao duplicar).
    """
    loader = loader or GrandeVitoriaLoader()
    if builder is None:
        builder = build_empresas_hin(loader)
    builder.build(validate=False)

    empresas = loader.empresas().set_index("cnpj")
    socios = loader.socios()
    dividas = loader.dividas_ativas()
    vinculos = loader.vinculos_politicos()
    infracoes = loader.infracoes_ambientais()
    contratos = loader.contratos_governamentais()
    renuncias = loader.beneficios_fiscais(tipo="RENUNCIA")
    habilitados = loader.beneficios_fiscais(tipo="HABILITADO")
    imunes = loader.beneficios_fiscais(tipo="IMUNE_ISENTO")
    jucees = loader.registros_jucees().set_index("cnpj_empresa")
    rotulo = loader.rotulo_sancao().set_index("cnpj_empresa")

    index = empresas.index
    features = pd.DataFrame(index=index)

    capital_social = pd.to_numeric(empresas["capital_social"], errors="coerce").fillna(0.0).clip(lower=0)
    features["capital_social_log1p"] = np.log1p(capital_social)

    num_socios = socios.groupby("cnpj_empresa").size()
    features["num_socios"] = num_socios.reindex(index, fill_value=0).astype(int)
    features["grau_do_socio"] = _grau_do_socio(socios, index)

    for coluna, valor in _grau_metapath(builder, index).items():
        features[coluna] = valor

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

    _modalidades_sem_competicao = {"Dispensa de Licitação", "Inexigibilidade de Licitação"}
    empresas_sem_competicao = set(
        contratos.loc[contratos["modalidade_compra"].isin(_modalidades_sem_competicao), "cnpj_empresa"]
    )
    features["tem_contrato_sem_competicao"] = index.isin(empresas_sem_competicao)

    contratos_validos = contratos[contratos["valor_inicial"] > 0].copy()
    contratos_validos["sobrepreco"] = (
        (contratos_validos["valor_final"] - contratos_validos["valor_inicial"]) / contratos_validos["valor_inicial"]
    )
    sobrepreco_max = contratos_validos.groupby("cnpj_empresa")["sobrepreco"].max()
    features["sobrepreco_contrato_max"] = sobrepreco_max.reindex(index, fill_value=0.0).clip(lower=0)

    valor_renuncia = (
        renuncias.groupby("cnpj_empresa")["valor"].sum().reindex(index, fill_value=0.0).clip(lower=0)
    )
    features["valor_renuncia_fiscal_log1p"] = np.log1p(valor_renuncia)
    features["tem_renuncia_fiscal"] = index.isin(set(renuncias["cnpj_empresa"]))

    features["tem_beneficio_fiscal_habilitado"] = index.isin(set(habilitados["cnpj_empresa"]))
    features["tem_imune_isento_irpj"] = index.isin(set(imunes["cnpj_empresa"]))

    data_constituicao = pd.to_datetime(jucees["data_constituicao"], errors="coerce")
    idade_dias = (_DATA_REFERENCIA_IDADE_EMPRESA - data_constituicao).dt.days
    features["idade_empresa_anos"] = (idade_dias / 365.25).reindex(index, fill_value=-1).fillna(-1)

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
