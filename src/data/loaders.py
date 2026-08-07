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
