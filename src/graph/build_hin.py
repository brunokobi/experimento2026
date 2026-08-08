"""Construcao da HIN real de empresas a partir do dataset
``projeto_grande_vitoria_empresas`` (ver ``docs/research_plan.md``).

Le as tabelas via ``GrandeVitoriaLoader`` e monta a HIN com ``HINBuilder``,
seguindo o schema de nos/arcos alinhado a pergunta de pesquisa (secao 2 do
plano: metapaths de socio comum, endereco comum e vinculo politico)::

    Nos:  empresa, socio, endereco, municipio, vinculo_politico
    Arcos:
        socio   -participa_de->          empresa           (bidirecional)
        empresa -sediada_em->            endereco          (bidirecional)
        empresa -localizada_em->         municipio         (bidirecional)
        empresa -tem_vinculo_politico->  vinculo_politico  (bidirecional)

O rotulo (``sancoes_administrativas``) NUNCA entra como feature de no -- so
como ``data["empresa"].y_direto`` / ``.y_qualquer`` (ver
``GrandeVitoriaLoader.rotulo_sancao``), para nao vazar informacao e para
preservar a distincao rotulo-direto vs. rotulo-via-socio (risco de
circularidade, secao 5/9 do plano de pesquisa).

``dividas_ativas`` entra agregada por empresa (valor total, numero de
inscricoes) -- sinal auxiliar, nao rotulo.

``processos_judiciais`` **nao** entra ainda: o pipeline `djen` que a
popula (no repo do dataset) ainda esta em andamento, e o campo e ruidoso por
design (casado por nome, nao CNPJ) -- ver riscos no plano de pesquisa.

Limitacoes conhecidas desta primeira versao (documentadas, nao escondidas):

- Identidade de socio (``_chave_socio``) combina CPF mascarado + nome
  normalizado; quando o CPF vem vazio (socio estrangeiro/PJ), cai so no nome
  -- risco de colisao entre homonimos.
- Identidade de endereco (``_chave_endereco``) usa logradouro+numero+CEP
  normalizados; nao resolve variacoes de grafia/abreviacao.
- Identidade de vinculo politico usa o nome do socio vinculado normalizado,
  sem casar contra a identidade de socio da empresa (``socios.nome_socio``)
  -- ligar as duas exigiria resolucao de nome mais cuidadosa; fica como
  proximo passo.
"""

from __future__ import annotations

import re
import unicodedata

import pandas as pd
import torch
from loguru import logger

from src.data.loaders import GrandeVitoriaLoader
from src.graph.hin_builder import HINBuilder


def _normalizar_texto(valor: str | None) -> str:
    """Normaliza uma string para uso como chave de identidade (maiusculas,
    sem acento, espacos colapsados). Nao resolve homonimos/variacoes de
    grafia -- limitacao conhecida, ver docstring do modulo.

    Aceita ``NaN`` (``float``) alem de ``None``/``str`` -- colunas de texto
    lidas via ``pandas.read_sql_query`` viram ``NaN`` quando nulas no SQLite,
    nao ``None`` nem string vazia (achado ao validar contra o banco real, ver
    ``scripts/validar_hin_real.py``). ``nan or ""`` NAO pega esse caso: ``NaN``
    e *truthy* em Python, entao usar ``pd.isna`` explicitamente.
    """
    if pd.isna(valor):
        return ""
    sem_acento = unicodedata.normalize("NFKD", str(valor)).encode("ascii", "ignore").decode()
    return re.sub(r"\s+", " ", sem_acento.strip().upper())


def _chave_socio(cpf_parcial: str | None, nome_socio: str | None) -> str:
    """Chave de identidade de um socio (pessoa), combinando CPF mascarado
    (ja vem mascarado da Receita -- LGPD resolvida na fonte) e nome
    normalizado. Sem CPF, cai no nome isoladamente.
    """
    cpf = "" if pd.isna(cpf_parcial) else str(cpf_parcial).strip()
    nome_norm = _normalizar_texto(nome_socio)
    return f"{cpf}|{nome_norm}"


def _chave_endereco(logradouro: str | None, numero: str | None, cep: str | None) -> str:
    """Chave de identidade de um endereco: logradouro + numero + CEP normalizados."""
    partes = [_normalizar_texto(logradouro), _normalizar_texto(numero), _normalizar_texto(cep)]
    return "|".join(partes)


def build_empresas_hin(loader: GrandeVitoriaLoader | None = None) -> HINBuilder:
    """Monta a HIN real de empresas da Grande Vitoria a partir do dataset.

    Args:
        loader: ``GrandeVitoriaLoader`` a usar; se omitido, cria um a partir
            de ``Settings`` (le ``settings.sqlite_db_path``). Em testes, passe
            um loader apontando para um banco sintetico com o schema real.
    """
    loader = loader or GrandeVitoriaLoader()
    builder = HINBuilder()

    empresas = loader.empresas()
    socios = loader.socios()
    dividas = loader.dividas_ativas()
    vinculos = loader.vinculos_politicos()
    rotulo = loader.rotulo_sancao()

    cnpjs = empresas["cnpj"].tolist()

    # --- no "empresa" ---------------------------------------------------- #
    dividas_por_empresa = (
        dividas.groupby("cnpj_empresa")["valor"]
        .agg(valor_dividas_ativas="sum", num_dividas_ativas="count")
        .reindex(cnpjs, fill_value=0.0)
    )
    capital_social = pd.to_numeric(empresas["capital_social"], errors="coerce").fillna(0.0).to_numpy()
    features_empresa = torch.tensor(
        pd.DataFrame(
            {
                "capital_social": capital_social,
                "valor_dividas_ativas": dividas_por_empresa["valor_dividas_ativas"].to_numpy(),
                "num_dividas_ativas": dividas_por_empresa["num_dividas_ativas"].to_numpy(),
            }
        ).to_numpy(dtype="float32")
    )
    builder.add_node_type("empresa", cnpjs, features=features_empresa)

    # Rotulo: NUNCA como feature de no -- atributo dedicado (y_*), preservando
    # a distincao direto/qualquer (secao 5/9 do plano de pesquisa).
    rotulo_por_cnpj = rotulo.set_index("cnpj_empresa").reindex(cnpjs)
    builder.data["empresa"].y_direto = torch.tensor(
        rotulo_por_cnpj["y_direto"].fillna(False).to_numpy(dtype="bool")
    )
    builder.data["empresa"].y_qualquer = torch.tensor(
        rotulo_por_cnpj["y_qualquer"].fillna(False).to_numpy(dtype="bool")
    )

    # --- no "socio" (metapath: socio comum) ------------------------------ #
    socios = socios.copy()
    socios["chave_socio"] = [
        _chave_socio(cpf, nome)
        for cpf, nome in zip(socios["cpf_parcial"], socios["nome_socio"], strict=True)
    ]
    chaves_socio = socios["chave_socio"].drop_duplicates().tolist()
    builder.add_node_type("socio", chaves_socio)
    builder.add_edge_type(
        "socio",
        "participa_de",
        "empresa",
        edges=list(zip(socios["chave_socio"], socios["cnpj_empresa"], strict=True)),
        bidirectional=True,
    )

    # --- no "endereco" (metapath: endereco comum) ------------------------ #
    empresas = empresas.copy()
    empresas["chave_endereco"] = [
        _chave_endereco(logr, num, cep)
        for logr, num, cep in zip(empresas["logradouro"], empresas["numero"], empresas["cep"], strict=True)
    ]
    chaves_endereco = empresas["chave_endereco"].drop_duplicates().tolist()
    builder.add_node_type("endereco", chaves_endereco)
    builder.add_edge_type(
        "empresa",
        "sediada_em",
        "endereco",
        edges=list(zip(empresas["cnpj"], empresas["chave_endereco"], strict=True)),
        bidirectional=True,
    )

    # --- no "municipio" (contexto/visualizacao -- nao e metapath primario) #
    municipios = empresas["municipio"].dropna().drop_duplicates().tolist()
    builder.add_node_type("municipio", municipios)
    empresas_com_municipio = empresas.dropna(subset=["municipio"])
    builder.add_edge_type(
        "empresa",
        "localizada_em",
        "municipio",
        edges=list(zip(empresas_com_municipio["cnpj"], empresas_com_municipio["municipio"], strict=True)),
        bidirectional=True,
    )

    # --- no "vinculo_politico" (metapath: vinculo politico) -------------- #
    if not vinculos.empty:
        vinculos = vinculos.copy()
        vinculos["chave_politico"] = vinculos["nome_socio_vinculado"].map(_normalizar_texto)
        chaves_politico = vinculos["chave_politico"].drop_duplicates().tolist()
        builder.add_node_type("vinculo_politico", chaves_politico)
        builder.add_edge_type(
            "empresa",
            "tem_vinculo_politico",
            "vinculo_politico",
            edges=list(zip(vinculos["cnpj_empresa"], vinculos["chave_politico"], strict=True)),
            bidirectional=True,
        )

    logger.info(f"HIN real construida: {builder.stats()}")
    return builder
