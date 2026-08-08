# CLAUDE.md — Contexto do projeto para o Claude Code

Este arquivo é carregado automaticamente pelo Claude Code. Resume o estado da
pesquisa e as decisões já tomadas — para não repetir investigação/discussão em
outra máquina. **Fonte da verdade completa**: `docs/research_plan.md` — este
arquivo é só o resumo rápido.

## O que é

Dissertação de mestrado (sem orientador formal): uma HIN (Rede Heterogênea de
Informação) de empresas da Grande Vitória (ES), com GNN (PyTorch Geometric) para
detectar risco de sanção administrativa via metapaths societários (sócio comum,
endereço comum, vínculo político). Objetivo: publicação em veículo Qualis alto.

## Decisões já travadas (não reabrir sem motivo novo)

1. **Fonte de dados**: dataset já existente do pesquisador,
   [`projeto_grande_vitoria_empresas`](https://github.com/brunokobi/projeto_grande_vitoria_empresas)
   (GitHub Release `dataset-latest`, arquivo `grande_vitoria.db.gz`) — 344.130
   empresas, 231.890 sócios, 7 municípios. Não é preciso buscar/validar rótulo
   externo (CEIS/CNEP/TCU já vêm ingeridos e casados por CNPJ nesse dataset).
2. **Rótulo**: `sancoes_administrativas` — verificado no banco real, só **188
   empresas em 344.130 (0,055%)** têm sanção confirmada. Desbalanceamento extremo
   → tratado como detecção de anomalia/ranking (PR-AUC, Precision@k), **não**
   classificação balanceada. O rótulo é positivo-incompleto (PU), não exaustivo —
   dizer isso explicitamente no texto, não escondido.
3. **Sinais auxiliares, não rótulo**: `dividas_ativas` e `vinculos_politicos`
   (TSE) entram como features/nós extras na HIN, não como rótulo principal.
4. **Ruído a isolar**: `processos_judiciais` é casado por nome (fuzzy) em 99,9%
   dos casos, não por CNPJ — não confiável como rótulo.
5. **Neo4j vs. PyTorch Geometric**: Neo4j só explora/valida/visualiza (Cypher,
   GDS, figuras da dissertação) — **não treina modelo**. PyTorch Geometric
   (`HeteroData`) é o motor de treino de fato.
6. **Onde cada peça roda**: Neo4j na VPS pessoal do pesquisador (Oracle Cloud, já
   orquestrada via Coolify) — mesmo padrão dos outros serviços pessoais dele.
   Treino da GNN e notebooks ficam na máquina **local** — a VPS é CPU-only/ARM64
   (free tier), sem GPU, e as extensões nativas do PyTorch Geometric são
   instáveis em ARM. **Segredos de acesso (IP/chaves/credenciais) nunca vão neste
   repositório** — é público; ficam só no `.env` local e no repositório privado
   de infraestrutura pessoal do pesquisador.
7. **LGPD**: CPF de sócios já vem mascarado na fonte (Receita Federal) — sem
   trabalho adicional de pseudonimização necessário para uso nesta pesquisa.

## Pendente (próximo passo real)

**Feito (08/08/2026)**: `src/config/settings.py` e `src/data/loaders.py`
(`GrandeVitoriaLoader`) já adaptados ao schema real (`empresas`, `socios`,
`sancoes_administrativas`, `dividas_ativas`, `vinculos_politicos`);
`src/graph/build_hin.py` (`build_empresas_hin`) monta a HIN real (nós
`empresa`/`socio`/`endereco`/`municipio`/`vinculo_politico`), com rótulo
exposto como `data["empresa"].y_direto`/`.y_qualquer` — nunca como feature.
Testado em `src/tests/test_real_schema.py` (banco sintético no schema real,
não o banco de verdade). Ver seção 12 de `docs/research_plan.md` para
detalhes e limitações conhecidas (identidade de sócio/endereço por
heurística, vínculo político ainda não ligado ao sócio por nome).

**Próximo passo real agora**: reescrever `src/graph/metapaths.py` para
extração via produto de matriz esparsa (o DFS atual não escala para 344k
empresas) e então rodar a extração sobre o banco real pela primeira vez.

## Armadilhas já identificadas (não repetir)

- **Não commitar o `.db`/`.db.gz` do dataset** — chega por download da GitHub
  Release (ver seção 4 de `docs/research_plan.md`), fica em `data/raw/`
  (ignorado pelo `.gitignore`). Contém dado pessoal (nome de sócio, endereço),
  mesmo com CPF mascarado.
- **Extração de metapath deve usar produto de matrizes esparsas**, não DFS em
  `networkx` (`src/graph/metapaths.py` hoje usa DFS — é só protótipo sobre dado
  sintético; não escala para os volumes reais).
- **Não usar `processos_judiciais` como rótulo** — é ruidoso (match fuzzy por
  nome, não CNPJ direto).
- **Split temporal simples é arriscado** dado o N pequeno de positivos (188) —
  preferir validação cruzada estratificada repetida.
