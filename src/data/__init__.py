"""Carregadores de dados a partir das fontes do projeto original (SQLite/Parquet)."""

from src.data.loaders import GrandeVitoriaLoader, ParquetLoader, SQLiteLoader

__all__ = ["SQLiteLoader", "ParquetLoader", "GrandeVitoriaLoader"]
