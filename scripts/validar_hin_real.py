"""Script de validacao manual: roda o pipeline real (GrandeVitoriaLoader ->
build_empresas_hin -> SparseMetaPathExtractor) contra o banco de verdade
(``data/raw/grande_vitoria.db``), fora da suite de testes (pytest) porque o
banco tem ~280 MB e dado pessoal -- nunca comitar o banco, so este script.

Reporta tempo/memoria de construcao da HIN, taxa de colisao das heuristicas
de identidade (socio, endereco) e o tamanho de cada metapath -- para validar
o pipeline contra o dado real antes de avancar para Neo4j/baselines (ver
docs/research_plan.md, secao 12).

Uso:
    uv run python scripts/validar_hin_real.py
"""

from __future__ import annotations

import time
import tracemalloc

from loguru import logger

from src.data.loaders import GrandeVitoriaLoader
from src.graph.build_hin import _chave_endereco, _chave_socio, build_empresas_hin
from src.graph.metapaths import COMMON_METAPATHS, MetapathExplosionError, SparseMetaPathExtractor

GB = 1024**3


def main() -> None:
    loader = GrandeVitoriaLoader()

    logger.info("Carregando tabelas...")
    t0 = time.perf_counter()
    empresas = loader.empresas()
    socios = loader.socios()
    logger.info(f"empresas={len(empresas)} socios={len(socios)} ({time.perf_counter() - t0:.1f}s)")

    # --- qualidade das heuristicas de identidade (nao dava para medir com dado sintetico) ---
    chaves_socio = [
        _chave_socio(cpf, nome) for cpf, nome in zip(socios["cpf_parcial"], socios["nome_socio"], strict=True)
    ]
    cpf_vazio = (socios["cpf_parcial"].fillna("").str.strip() == "").mean()
    logger.info(
        f"socios: {len(socios)} linhas -> {len(set(chaves_socio))} chaves distintas "
        f"(cpf_parcial vazio em {cpf_vazio:.1%} das linhas)"
    )

    chaves_endereco = [
        _chave_endereco(logr, num, cep)
        for logr, num, cep in zip(empresas["logradouro"], empresas["numero"], empresas["cep"], strict=True)
    ]
    logger.info(f"empresas: {len(empresas)} linhas -> {len(set(chaves_endereco))} enderecos distintos")

    # --- construcao da HIN completa ---
    logger.info("Construindo a HIN completa...")
    tracemalloc.start()
    t0 = time.perf_counter()
    builder = build_empresas_hin(loader)
    data = builder.build()
    elapsed = time.perf_counter() - t0
    _current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    logger.info(f"HIN construida em {elapsed:.1f}s, pico de memoria (Python) {peak / GB:.2f} GB")
    logger.info(f"stats: {builder.stats()}")
    logger.info(
        f"rotulo: y_direto={int(data['empresa'].y_direto.sum())} "
        f"y_qualquer={int(data['empresa'].y_qualquer.sum())} "
        f"(universo: {data['empresa'].num_nodes} empresas)"
    )

    # --- extracao de cada metapath via matriz esparsa ---
    logger.info("Extraindo metapaths (matriz esparsa)...")
    extractor = SparseMetaPathExtractor(data)
    for nome, metapath in COMMON_METAPATHS.items():
        t0 = time.perf_counter()
        try:
            matrix = extractor.commuting_matrix(metapath)
        except (KeyError, MetapathExplosionError) as exc:
            logger.warning(f"{nome}: pulado ({exc})")
            continue
        logger.info(f"{nome}: shape={matrix.shape} nnz={matrix.nnz} ({time.perf_counter() - t0:.1f}s)")


if __name__ == "__main__":
    main()
