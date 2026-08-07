"""Configuracao central do projeto via Pydantic Settings.

Le parametros de um arquivo ``.env`` na raiz do projeto (ver ``.env.example``)
e expoe um objeto ``Settings`` unico e validado, acessivel via ``get_settings()``.

Uso:
    from src.config import get_settings

    settings = get_settings()
    print(settings.neo4j_uri)
    print(settings.data_raw_dir)
"""

from __future__ import annotations

from enum import Enum
from functools import lru_cache
from pathlib import Path

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# Raiz do projeto = dois niveis acima deste arquivo (src/config/settings.py -> raiz)
PROJECT_ROOT = Path(__file__).resolve().parents[2]


class AppEnv(str, Enum):
    """Ambientes de execucao suportados."""

    DEVELOPMENT = "development"
    TESTING = "testing"
    PRODUCTION = "production"


class TorchDevice(str, Enum):
    """Dispositivos suportados para tensores/treinamento em PyTorch."""

    CPU = "cpu"
    CUDA = "cuda"
    MPS = "mps"


class Settings(BaseSettings):
    """Parametros de configuracao do projeto, lidos de variaveis de ambiente/.env."""

    model_config = SettingsConfigDict(
        env_file=str(PROJECT_ROOT / ".env"),
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # --- Ambiente / logging ---
    app_env: AppEnv = Field(default=AppEnv.DEVELOPMENT, description="Ambiente de execucao.")
    log_level: str = Field(default="INFO", description="Nivel de log (loguru).")

    # --- Caminhos de dados ---
    data_raw_dir: Path = Field(default=Path("data/raw"))
    data_processed_dir: Path = Field(default=Path("data/processed"))
    data_graph_exports_dir: Path = Field(default=Path("data/graph_exports"))

    # --- Fonte de dados original (projeto anterior) ---
    sqlite_db_path: Path = Field(default=Path("data/raw/empresas.db"))
    parquet_dir: Path = Field(default=Path("data/raw"))

    # --- Neo4j ---
    neo4j_uri: str = Field(default="bolt://localhost:7687")
    neo4j_user: str = Field(default="neo4j")
    neo4j_password: SecretStr = Field(default=SecretStr("changeme"))
    neo4j_database: str = Field(default="neo4j")

    # --- Parametros de memoria / performance para montagem da HIN ---
    max_memory_gb: float = Field(default=8.0, gt=0, description="Limite de memoria (GB) monitorado nos testes.")
    num_workers: int = Field(default=4, ge=0)
    random_seed: int = Field(default=42)
    torch_device: TorchDevice = Field(default=TorchDevice.CPU)
    batch_size: int = Field(default=1024, gt=0)

    @field_validator("data_raw_dir", "data_processed_dir", "data_graph_exports_dir", "sqlite_db_path", "parquet_dir")
    @classmethod
    def _resolve_relative_to_project_root(cls, value: Path) -> Path:
        """Resolve caminhos relativos em relacao a raiz do projeto."""
        if value.is_absolute():
            return value
        return (PROJECT_ROOT / value).resolve()

    def ensure_data_dirs(self) -> None:
        """Cria os diretorios de dados caso nao existam (idempotente)."""
        for directory in (self.data_raw_dir, self.data_processed_dir, self.data_graph_exports_dir):
            directory.mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Retorna a instancia (singleton em processo) de ``Settings``."""
    return Settings()
