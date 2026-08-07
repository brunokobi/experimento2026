"""Carregadores de dados a partir das fontes do projeto original (SQLite/Parquet)."""

from src.data.loaders import ParquetLoader, SQLiteLoader

__all__ = ["SQLiteLoader", "ParquetLoader"]
