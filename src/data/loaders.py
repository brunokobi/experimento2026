"""Carregadores de dados do projeto original.

Duas fontes sao suportadas, conforme os requisitos da pesquisa:

- ``SQLiteLoader``: le tabelas de um banco SQLite (ex.: exportacao relacional
  de empresas, socios, CNAEs) para ``pandas.DataFrame``.
- ``ParquetLoader``: le/escreve arquivos Parquet (via PyArrow), formato usado
  para os dados processados e para os artefatos intermediarios da HIN.

Ambos respeitam os caminhos definidos em ``Settings`` mas tambem aceitam
overrides explicitos, para uso em notebooks e testes.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq
from loguru import logger

from src.config import Settings, get_settings


class SQLiteLoader:
    """Leitor de tabelas/queries de um banco SQLite."""

    def __init__(self, db_path: Path | str | None = None, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()
        self.db_path = Path(db_path) if db_path is not None else self._settings.sqlite_db_path

    def _connect(self) -> sqlite3.Connection:
        if not self.db_path.exists():
            raise FileNotFoundError(f"Banco SQLite nao encontrado: {self.db_path}")
        return sqlite3.connect(self.db_path)

    def list_tables(self) -> list[str]:
        """Lista as tabelas disponiveis no banco."""
        with self._connect() as conn:
            cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table';")
            tables = [row[0] for row in cursor.fetchall()]
        logger.debug(f"{len(tables)} tabelas encontradas em {self.db_path}")
        return tables

    def read_table(self, table_name: str, columns: list[str] | None = None, limit: int | None = None) -> pd.DataFrame:
        """Le uma tabela inteira (ou um subconjunto de colunas/linhas) como DataFrame."""
        cols = ", ".join(columns) if columns else "*"
        query = f"SELECT {cols} FROM {table_name}"  # noqa: S608 - nomes controlados internamente
        if limit is not None:
            query += f" LIMIT {limit}"
        return self.read_query(query)

    def read_query(self, query: str, params: tuple | dict | None = None) -> pd.DataFrame:
        """Executa uma query arbitraria e retorna o resultado como DataFrame."""
        logger.debug(f"Executando query SQLite: {query}")
        with self._connect() as conn:
            return pd.read_sql_query(query, conn, params=params)


class ParquetLoader:
    """Leitor/escritor de arquivos Parquet (dados processados / exports da HIN)."""

    def __init__(self, base_dir: Path | str | None = None, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()
        self.base_dir = Path(base_dir) if base_dir is not None else self._settings.data_processed_dir

    def _resolve(self, filename: str) -> Path:
        path = Path(filename)
        return path if path.is_absolute() else self.base_dir / path

    def read(self, filename: str, columns: list[str] | None = None) -> pd.DataFrame:
        """Le um arquivo Parquet como DataFrame, com possibilidade de projetar colunas."""
        path = self._resolve(filename)
        if not path.exists():
            raise FileNotFoundError(f"Arquivo Parquet nao encontrado: {path}")
        logger.debug(f"Lendo Parquet: {path}")
        return pq.read_table(path, columns=columns).to_pandas()

    def write(self, df: pd.DataFrame, filename: str, overwrite: bool = False) -> Path:
        """Escreve um DataFrame como Parquet no diretorio base."""
        path = self._resolve(filename)
        if path.exists() and not overwrite:
            raise FileExistsError(f"Arquivo ja existe (use overwrite=True): {path}")
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(path, index=False)
        logger.debug(f"Parquet escrito em: {path} ({len(df)} linhas)")
        return path


class GrandeVitoriaLoader:
    """Leitor das tabelas reais do dataset ``projeto_grande_vitoria_empresas``.

    Encapsula os nomes de tabela/coluna do schema real (ver
    ``docs/research_plan.md``, secao 4, e ``database/schema.sql`` naquele
    repo) para nao espalhar strings magicas pelo resto do codigo. Usa
    ``SQLiteLoader`` por baixo -- aceita um ``SQLiteLoader`` explicito (util
    em testes, com um banco sintetico) ou usa o caminho de ``Settings``.
    """

    def __init__(self, loader: SQLiteLoader | None = None, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()
        self._sqlite = loader or SQLiteLoader(settings=self._settings)

    def empresas(self) -> pd.DataFrame:
        """Universo de empresas ativas (chave: ``cnpj``)."""
        return self._sqlite.read_table("empresas")

    def socios(self) -> pd.DataFrame:
        """Vinculos socio-empresa (uma linha por par; ``cnpj_empresa`` + ``nome_socio``/``cpf_parcial``)."""
        return self._sqlite.read_table("socios")

    def dividas_ativas(self) -> pd.DataFrame:
        """Divida ativa (PGFN/Sefaz-ES) -- sinal auxiliar, nao e rotulo."""
        return self._sqlite.read_table("dividas_ativas")

    def vinculos_politicos(self) -> pd.DataFrame:
        """Vinculos politicos (TSE) por empresa -- sinal/no auxiliar, nao e rotulo."""
        return self._sqlite.read_table("vinculos_politicos")

    def infracoes_ambientais(self) -> pd.DataFrame:
        """Infracoes ambientais (IBAMA/IEMA) -- sinal auxiliar, match direto por
        CNPJ (confirmado: 100% ``match_confianca='direto'`` no banco real, nao
        e rotulo)."""
        return self._sqlite.read_table("infracoes_ambientais")

    def contratos_governamentais(self) -> pd.DataFrame:
        """Contratos com orgaos publicos -- sinal auxiliar (nao e rotulo).
        Sem coluna ``match_confianca`` no schema (fonte traz CNPJ direto)."""
        return self._sqlite.read_table("contratos_governamentais")

    def registros_jucees(self) -> pd.DataFrame:
        """Metadados de constituicao (JUCEES) -- sinal auxiliar (nao e rotulo).
        ``data_constituicao`` cobre 100% das 88.349 linhas no banco real."""
        return self._sqlite.read_table("registros_jucees")

    def beneficios_fiscais(self, tipo: str | None = None) -> pd.DataFrame:
        """Beneficios/renuncias fiscais federais -- sinal auxiliar (nao e rotulo).

        Args:
            tipo: se informado, filtra por ``'IMUNE_ISENTO'`` (imune/isenta de
                IRPJ -- majoritariamente entidades sem fins lucrativos, mesma
                populacao elegivel a sancao CEPIM: usar como feature e
                legitimo, mas capta "elegibilidade a CEPIM", nao
                necessariamente risco em si), ``'RENUNCIA'`` (renuncia fiscal
                federal, tem valor monetario) ou ``'HABILITADO'`` (habilitada
                a regime de beneficio fiscal, ex.: Reidi/Recap/Reporto).
                ``None`` retorna todos os tipos.
        """
        df = self._sqlite.read_table("beneficios_fiscais")
        if tipo is not None:
            df = df[df["tipo"] == tipo]
        return df

    def sancoes_administrativas(self, match_confianca: str | None = None) -> pd.DataFrame:
        """Sancoes administrativas (CEIS/CNEP/CEPIM/TCEES/TRABALHO_ESCRAVO).

        Args:
            match_confianca: se informado, filtra por ``'direto'`` (sancao na
                propria empresa) ou ``'socio'`` (atribuida via socio em comum
                com entidade sancionada -- ver risco de circularidade na
                secao 5/9 do plano de pesquisa). ``None`` retorna as duas.
        """
        df = self._sqlite.read_table("sancoes_administrativas")
        if match_confianca is not None:
            df = df[df["match_confianca"] == match_confianca]
        return df

    def processos_judiciais(self, match_confianca: str | None = None) -> pd.DataFrame:
        """Processos judiciais via DJEN -- ruidoso (casado por nome), nao e rotulo.

        Args:
            match_confianca: ``'nome'`` (casado por razao social) ou ``'socio'``
                (via socio em comum); nunca ``'direto'`` nesta tabela.
        """
        df = self._sqlite.read_table("processos_judiciais")
        if match_confianca is not None:
            df = df[df["match_confianca"] == match_confianca]
        return df

    def rotulo_sancao(self) -> pd.DataFrame:
        """Rotulo binario por empresa, nas duas granularidades travadas no plano
        de pesquisa (``docs/research_plan.md``, secoes 5 e 9):

        - ``y_direto``: sancao confirmada na propria empresa
          (``match_confianca='direto'``) -- rotulo **primario**, sem risco de
          circularidade com o metapath de socio comum.
        - ``y_qualquer``: ``y_direto`` OU sancao atribuida via socio em comum
          (``match_confianca='socio'``) -- inclui as ~41 empresas com risco de
          circularidade; usar so como analise de sensibilidade separada, nunca
          misturado com ``y_direto`` sem declarar qual foi usado.

        Retorna um DataFrame com uma linha por empresa (``cnpj_empresa``,
        ``y_direto``, ``y_qualquer``), cobrindo todo o universo de
        ``empresas`` (nao so as sancionadas).
        """
        sancoes = self.sancoes_administrativas()
        universo = self.empresas()[["cnpj"]].rename(columns={"cnpj": "cnpj_empresa"})
        diretas = set(sancoes.loc[sancoes["match_confianca"] == "direto", "cnpj_empresa"])
        quaisquer = set(sancoes["cnpj_empresa"])
        universo["y_direto"] = universo["cnpj_empresa"].isin(diretas)
        universo["y_qualquer"] = universo["cnpj_empresa"].isin(quaisquer)
        return universo
