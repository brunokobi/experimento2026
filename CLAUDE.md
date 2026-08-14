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

**Feito (08/08/2026, continuação)**: `src/graph/metapaths.py` ganhou
`SparseMetaPathExtractor` (produto de matrizes de adjacência esparsas via
`scipy.sparse`) — é o que escala para os 344k empresas; `MetaPathExtractor`
(DFS) foi mantido só para depuração/cruzamento com Cypher em amostras
pequenas, não é mais o caminho de produção. Testado com HIN sintética de
50k+20k nós (`test_memory.py`) sem densificar a matriz.

**Feito (08/08/2026, validação final)**: rodei tudo contra o banco real
(`scripts/validar_hin_real.py`, banco copiado — não movido — pra
`data/raw/grande_vitoria.db`, ignorado pelo git). HIN completa em 23,5s /
0,44 GB; metapaths de hipótese extraídos em <0,2s cada. Dois bugs reais só
apareceram aqui (documentados em detalhe no README e no plano de pesquisa):
`pandas` devolve `NaN` (não `None`) pra texto nulo — quebrava
`_chave_socio`; e `município` (7 nós pra 344k empresas) faz o produto
esparso explodir (~187 GiB) — resolvido com `MetapathExplosionError`
(estima o tamanho antes de calcular), que por sua vez tinha um bug de
overflow silencioso em `int32` só visível com a distribuição real
desigual dos municípios. Os dois têm teste de regressão.

**Feito (08/08/2026, Neo4j)**: Neo4j 5 + GDS rodando num container na VPS
pessoal (Oracle Cloud/Coolify, `/opt/coolify/apps/neo4j/`) — inicialmente
acesso só via túnel SSH, nenhuma porta pública (decisão: o grafo tem dado
pessoal). `src/graph/neo4j_export.py` (`export_hin_to_neo4j`) exporta a
HIN real inteira (idempotente, via `MERGE`) — rodado com sucesso contra o
banco de verdade (`scripts/exportar_hin_neo4j.py`).

**Decisão revista (11/08/2026)**: exposto publicamente por decisão
explícita do pesquisador (aceitando o risco — protegido só por
usuário/senha, sem 2FA). Browser via Traefik/HTTPS/Let's Encrypt em
`https://neo4j.brunokobi.duckdns.org`; Bolt publicado direto em
`neo4j.brunokobi.duckdns.org:7687` (porta 7687 aberta no iptables e na NSG
da OCI — NSG precisou de ação manual do pesquisador na OCI Console, fora
do alcance de SSH/CLI). `NEO4J_URI` do `.env` real já aponta pro domínio
público; túnel SSH continua funcionando como alternativa mais segura.
**Achado de infra**: Traefik desse Coolify tem tendência a ficar com
estado travado após mudança de labels de um container (mesmo padrão de
erro "router definido múltiplas vezes" já visto nos logs antigos de outro
app) — `docker restart coolify-proxy` resolve; vale lembrar disso da
próxima vez que outro serviço for adicionado/alterado atrás dele.

**Fix do Neo4j Browser (11/08/2026)**: Browser (HTTPS) não conseguia abrir
conexão Bolt sem TLS (`bolt://`) — navegador bloqueia por "mixed content"
(página segura tentando conexão insegura). Resolvido configurando TLS
próprio no Bolt (`dbms.ssl.policy.bolt`), reaproveitando o certificado
Let's Encrypt do domínio (extraído do `acme.json` do Traefik, convertido
pra PEM, montado em `/ssl/bolt` no container). `server.bolt.tls_level=OPTIONAL`
mantém `bolt://` funcionando por compatibilidade; `bolt+s://` é o
recomendado agora — `NEO4J_URI` do `.env` real já usa `bolt+s://`.

**Feito (08/08/2026, etapa 7.1)**: `src/features/tabular.py`
(`build_feature_matrix`) — feature engineering tabular por empresa (capital
social, sócios, dívida ativa agregada, vínculo político, porte/regime/CNAE
one-hot), sem nenhuma coluna derivada de sanção. Validado contra o banco
real: 344.130 empresas, 5,5s, 107 colunas, 0 `NaN`. Achado ao validar: `porte`
no banco é código numérico da Receita (`"01"`/`"03"`/`"05"`), não texto —
documentado no módulo. Etapa 7 quebrada em subetapas no README (7.1 a 7.7,
mais a etapa 8 de publicação).

**Feito (08/08/2026, etapa 7.2)**: `src/evaluation/harness.py` —
`precision_at_k`, `evaluate_repeated_cv` (RepeatedStratifiedKFold, PR-AUC +
Precision@k por fold) e `compare_models` (Wilcoxon pareado). Genérico pros 3
baselines — só exige uma função `fit_predict(x_train, y_train, x_test) ->
scores`. Testado com dado sintético (independente do schema do dataset).

**Feito (08/08/2026, etapa 7.3 — primeiro resultado quantitativo)**:
`src/models/tabular_baseline.py` (`xgboost_fit_predict`, `scale_pos_weight`
calculado por fold, nunca do dataset inteiro). Rodado contra o banco real
(`scripts/rodar_baseline_tabular.py`, ~5min): PR-AUC 0,0081 (`y_direto`) /
0,0085 (`y_qualquer`) — **~18,8×/~15,5× a taxa-base** (0,043%/0,055%), não
"baixo" como parece à primeira vista. Achado: lift menor em `y_qualquer`
(empresas só suspeitas via sócio comum) — bate com a hipótese central: dado
tabular isolado não vê risco por associação, é o que a HIN/GNN deve capturar.
Nota metodológica: desvio-padrão do Precision@k costuma superar a média (só
~30 positivos por fold de teste) — considerar k maiores (50/100) nas
próximas rodadas.

**Feito (08/08/2026, etapa 7.4 — GNN homogênea)**:
`src/models/gnn_homogeneous.py` — colapsa os 3 metapaths de hipótese numa
matriz empresa-empresa (GraphSAGE por cima, mesmas features tabulares da
7.1). Achado real ao construir o grafo: 878 endereços concentram ~11M das
~13M arestas empresa-endereço (prédios comerciais grandes, não "endereço de
fachada") — podados (`max_grau_endereco=20`) antes de virar grafo, senão
inviável treinar. Rodado contra o banco real (~29min, 5×2 folds, 50 epochs):

| Rótulo | Tabular (7.3) | GNN homogênea (7.4) |
|---|---|---|
| `y_direto` (principal) | 18,8× lift | **13,0× lift — pior** |
| `y_qualquer` (sensibilidade) | 15,5× lift | 18,5× lift — melhor, mas confundido (ver plano) |

**Reportado sem maquiar**: GNN homogênea perdeu do tabular no rótulo
principal. Causas plausíveis: poucas épocas/folds (custo computacional alto:
~29min pra só 10 folds vs 50 do tabular), e colapsar os 3 metapaths num só
tipo de aresta pode diluir sinal que o HAN/HGT deveria recuperar.
**Pendência de rigor**: fold count diferente entre tabular (50) e GNN (10)
— ainda não dá pra rodar Wilcoxon comparando os dois, precisa padronizar
antes da etapa 7.6.

**Feito (08/08/2026, etapa 7.5 — HAN/HGT)**: `src/models/han_hgt.py` —
`HGTConv` (2 camadas) tratando cada tipo de nó/relação distintamente (não
colapsa como a 7.4). Rodado contra o banco real (~44min, 5×1 folds, 50
epochs):

| Rótulo | Tabular (7.3) | GNN homogênea (7.4) | HAN/HGT (7.5) |
|---|---|---|---|
| `y_direto` | 18,8× | 13,0× | **18,2× — quase empata** |
| `y_qualquer` | 15,5× | 18,5× | **38,1× — amplifica a confusão de circularidade** |

Tratar cada metapath como relação distinta recupera quase todo o sinal
perdido ao colapsar (7.4): 18,2× vs 13,0× — exatamente a razão de a 7.5
existir separada da 7.4. No rótulo principal, empata (dentro do ruído) com
o tabular; ainda não supera claramente.

**Achado de infraestrutura**: configuração original (hidden=64, 2 cabeças,
com `município`) deu **OOM killer** na máquina local (7,8 GB RAM). Corrigido
reduzindo dimensões + excluindo `município` (não era metapath de hipótese
mesmo) — pico de memória ~5 GB, estável entre folds (confirmado não é
vazamento). Rodar a versão maior/tunada precisa de máquina com mais RAM/GPU.

**Pendência de rigor (acumulada)**: fold count diferente nos 3 modelos —
tabular (50), GNN homogênea (10), HAN/HGT (5) — bloqueia Wilcoxon até
padronizar (etapa 7.6).

**Feito (08/08/2026, etapa 7.6 — comparação estatística)**:
`scripts/comparar_baselines.py` padronizou 5×1=5 folds (o menor já usado,
evitando rodar o HAN/HGT em dobro), mesmo `random_state`, mesmo dado —
folds pareados de verdade. **Resultado: nenhuma diferença estatisticamente
significativa entre os 3 modelos, em nenhum rótulo** (Wilcoxon p entre 0,62
e 1,00 nos 6 pares). Inconclusivo pelo poder estatístico atual (5 folds,
~30 positivos por fold), não uma refutação da hipótese — precisa de mais
folds/repeats (mais recursos computacionais) pra decidir de verdade.

**Bug de reprodutibilidade achado e corrigido nessa etapa**:
`torch.manual_seed` só era chamado na construção da fábrica
`make_*_fit_predict`, não a cada fold — rodar outro modelo torch antes no
mesmo processo mudava o resultado mesmo com `random_state` igual (achado ao
comparar o HAN/HGT isolado vs. dentro do script de comparação: 0,0078 vs.
0,0059 com config "idêntica"). Corrigido: seed resetada dentro do
`fit_predict`. Teste de regressão que "suja" o RNG global de propósito
antes de comparar.

**Primeira tentativa de 30 folds perdida** (iniciada 08/08/2026 ~17:33,
checada 09/08/2026 23:09): a máquina reiniciou entre as duas datas —
`/tmp` foi limpo no reboot, matando o processo `nohup` e o log junto.

**Feito (relançada 09/08/2026 ~23:11, concluída 10/08/2026 ~06:23,
~8h45)**: log completo salvo em
`docs/resultados/comparar_baselines_30folds_2026-08-10.log`. **Resultado
final, com poder estatístico real**:

| Rótulo | Tabular | GNN homogênea | HAN/HGT | Wilcoxon |
|---|---|---|---|---|
| `y_direto` (principal) | 18,6× | 16,5× | **14,2×** | tabular>HAN/HGT p=0,0066; homogênea>HAN/HGT p=0,0293 |
| `y_qualquer` (confundido) | 15,7× | 24,4× | **33,5×** | homogênea>tabular p=0,0020; HAN/HGT>tabular p=0,0364 |

**No rótulo principal, HAN/HGT é estatisticamente PIOR que o tabular e que
a GNN homogênea** — com 5 folds parecia empate (ruído mascarando o efeito
real; não rerodar decisão em cima de N pequeno). Em `y_qualquer` os
modelos de rede vencem, mas é o rótulo confundido por circularidade — não
é evidência limpa a favor da hipótese. **Resultado negativo genuíno pra
pergunta de pesquisa central**, reportado sem maquiar — pode ser efeito
real ou artefato de hiperparâmetros/épocas de primeira versão (50 épocas,
sem tuning).

**Próximo passo real**: decidir com o pesquisador — (a) investir em tuning
de hiperparâmetros do HAN/HGT antes de aceitar esse resultado como final,
ou (b) reportar como está e seguir pra etapa 7.7 (sensibilidade `y_direto`
vs `y_qualquer`, agora com diferença estatística confirmada, não só
indício) e depois etapa 8 (publicação) com esse resultado negativo
discutido no texto.

**Feito (10/08/2026, novas features)**: `build_feature_matrix` (7.1) ganhou
4 cruzamentos novos — infração ambiental (IBAMA/IEMA), contratos com
órgãos públicos, renúncia fiscal federal, habilitação a benefício fiscal,
e imune/isento de IRPJ (esse último com ressalva de interpretação:
colinear com elegibilidade a CEPIM, não é vazamento). Identificados a
partir dos filtros do dashboard do outro repo, conferidos um a um contra o
banco antes de implementar — 2 candidatos (`contratos_pncp`, `marcas_inpi`)
não têm tabela populada ainda, não usáveis. Matriz: 107 → 117 colunas.
**Os resultados de 7.3–7.6 são com a matriz antiga (107 colunas)** — não
foram rerodados ainda com os novos sinais.

**Feito (concluído 11/08/2026 ~07:22, ~7h10)**: log movido para
`docs/resultados/comparar_baselines_30folds_v2_2026-08-11.log`. **Resultado
final v2, com as features novas (117 colunas)**:

| Rótulo | Tabular | GNN homogênea | HAN/HGT |
|---|---|---|---|
| `y_direto` (principal) | 75,8× | **81,2×** | 29,3× |
| `y_qualquer` | **62,3×** | 41,6× | 45,4× |

Todos os 3 modelos saltaram ~4× em PR-AUC absoluto — as features novas
carregam sinal real e forte. **Conclusão central fica mais forte, mesma
direção**: no rótulo principal, HAN/HGT continua estatisticamente PIOR
que tabular e GNN homogênea, agora com `p<0,0001` nos dois casos (antes
0,0066/0,0293). Em `y_qualquer`, mudança qualitativa: nenhuma diferença é
significativa agora (antes os modelos de rede venciam por causa da
circularidade) — o tabular deixou de perder pra rede até no rótulo
confundido. Reforça, com evidência mais forte, que a hipótese central da
tese não se confirma nesta implementação.

**Feito (11/08/2026, segunda rodada de features — com base em
literatura)**: pesquisei antes de implementar (não "achismo") — literatura
de grafo-features-vs-GNN em fraude, risco de corrupção em compras públicas
(Fazekas & Kocsis, 2020), detecção de shell company (Moody's). Ver referências
completas em `docs/research_plan.md`, seção 7. 5 features novas:
`grau_socio_comum`/`grau_endereco_comum`/`grau_vinculo_politico_comum`
(grau explícito de cada empresa em cada metapath, direto da HIN),
`grau_do_socio` (concentração do sócio mais conectado),
`tem_contrato_sem_competicao`/`sobrepreco_contrato_max` (red flags de
compras públicas — cobertura baixa, só 5 empresas), `idade_empresa_anos`
(via `registros_jucees`, 25,7% de cobertura, sentinela `-1`). Matriz:
117 → 124 colunas, validada contra o banco real (0 `NaN`).

**Teste rápido antes de comprometer ~7h**: só o tabular, 50 folds — PR-AUC
`y_direto` 0,0242 (lift ~56×, era 18,8×) / `y_qualquer` 0,0218 (lift ~40×,
era 15,5×). Ganho real confirmado — vale rodar o experimento completo.

**Interrompido de novo (3º reboot, entre 11/08 ~22:46 e 12/08 ~06:51 —
uptime resetado pra "2 min" quando checado)**: a rodada v3 acima morreu no
meio (chegou a completar 4 das 6 etapas: `tabular`/`gnn_homogenea` em
`y_direto`, `tabular` em `y_qualquer`, mais uma métrica agregada de
`han_hgt`/`y_direto` — sem dado por fold reaproveitável pra Wilcoxon).
Terceira vez que um `nohup` de longa duração é matado por reboot
imprevisto desta máquina local (WSL2) — as duas primeiras vezes perderam
o progresso inteiro.

**Checkpoint adicionado (12/08/2026) antes de relançar**:
`scripts/comparar_baselines.py` agora salva cada combinação modelo×rótulo
(6 no total) em `~/checkpoints_comparar_baselines/<nome>_ncols<N>.csv` assim
que termina — se rodar de novo, pula direto as etapas já concluídas em vez
de recomputar do zero (só perde a etapa que estava rodando no momento exato
da interrupção). Nome do arquivo inclui a contagem de colunas da matriz de
features pra não reusar por engano um checkpoint de uma versão antiga (não
detecta mudança de conteúdo com a MESMA contagem de colunas — limitação
conhecida, aceitável pro uso atual). Também corrigido nessa mexida:
`main()` construía a HIN (`builder`) e depois `build_feature_matrix`
reconstruía outra por dentro (não recebia `builder=builder`) — agora passa
a HIN já construída, evitando custo duplicado (~8s) a cada relançamento.

**Relançado e concluído (12/08/2026, ~06:57→14:09, ~7h11, sem
interrupção)**: PID 2021, log movido para
`docs/resultados/comparar_baselines_30folds_v3_2026-08-12.log`. Checkpoint
por etapa funcionou como planejado (nenhum reboot ocorreu durante essa
rodada, mas o mecanismo ficou validado para a próxima vez);
`~/checkpoints_comparar_baselines/` apagado após a conclusão. **Resultado
final v3, com as 5 features de literatura (124 colunas)**:

| Rótulo | Tabular | GNN homogênea | HAN/HGT |
|---|---|---|---|
| `y_direto` (principal) | 50,7× | **76,3×** | 23,5× |
| `y_qualquer` | 43,2× | 40,1× | 42,8× |

**Achado contraintuitivo**: o PR-AUC absoluto de todos os 3 modelos *caiu*
em relação ao v2 (117 colunas) — apesar do teste rápido antes de rodar ter
indicado ganho. O teste rápido comparou contra a baseline errada (v1
original, 18,8×, não o v2 mais recente, 75,8×) — lição registrada: sempre
comparar contra o último resultado completo, não o primeiro.

**Conclusão central (repetida pela 4ª vez, cada vez com mais dados)**:
HAN/HGT continua estatisticamente PIOR que tabular (p=0,0001) e que GNN
homogênea (p<0,0001) em `y_direto` — a hipótese central da tese (GNN
heterogênea supera tabular via metapaths) não se confirma em nenhuma das
3 versões de feature set testadas (107/117/124 colunas), e o efeito é
estatisticamente robusto, não ruído de poucos folds.

**Achado novo nesta rodada**: pela primeira vez, GNN homogênea vence o
tabular de forma significativa em `y_direto` (p=0,0145 — em v1 p=0,38, em
v2 p=0,72, nunca significativo antes). Em `y_qualquer`, nenhuma diferença
é significativa entre os 3 (p=0,84/0,87/0,89) — os 3 convergem para ~40-43×
em vez de o tabular vencer com folga como no v2, consistente com a ideia
de que dar sinal de grafo explícito ao tabular (`grau_socio_comum` etc.)
reduz a vantagem que os modelos de rede tinham nesse rótulo confundido.

**Decidido com o pesquisador (12/08/2026)**: investir numa busca pequena de
hiperparâmetros do HAN/HGT antes de aceitar o resultado como final —
motivação: os defaults atuais (`hidden_channels=32`, `num_heads=1`,
`epochs=50`) foram reduzidos por causa de OOM na máquina local, não por
tuning, e isso é a vulnerabilidade mais provável a ser atacada por um
revisor de periódico Qualis alto ("vocês só concluíram que é pior porque
não tentaram direito?").

**Feito (lançado 12/08/2026 ~21:41, concluído 13/08/2026 ~00:57, ~3h15)**:
`scripts/tunar_han_hgt.py` (+ worker `scripts/_tunar_han_hgt_candidato.py`,
em subprocesso isolado por candidato). 5 candidatos, variando um eixo por
vez a partir do baseline (`hidden=32/heads=1/epochs=50`), só `y_direto`, só
5 folds:

| Candidato | hidden | heads | epochs | PR-AUC (5 folds) |
|---|---|---|---|---|
| baseline | 32 | 1 | 50 | 0,0105 |
| mais_epocas | 32 | 1 | **150** | **0,0244** |
| mais_heads | 32 | **2** | 50 | 0,0182 |
| mais_hidden | 64 | 1 | 50 | 0,0100 (sem ganho) |
| maior | 64 | 2 | 100 | **OOM (falhou)** |

**Achado real: o modelo estava subtreinado**, não subdimensionado — só
aumentar épocas (50→150) quase triplicou o PR-AUC; aumentar `hidden`
isoladamente não ajudou nada; `heads=2` ajudou por si só; a combinação
`hidden=64+heads=2` confirma de novo o limite de memória da máquina local
(mesmo problema já visto na etapa 7.5 original). **Em andamento (2a
rodada, lançada 13/08/2026 ~07:14)**: testando o 6º candidato
(`hidden=32/heads=2/epochs=150` — combina os dois fatores que ajudaram,
evitando a zona de OOM) antes de travar a config final. Log em
`~/tunar_han_hgt_v2.log`, checkpoints em
`~/checkpoints_tunar_han_hgt/<candidato>.csv`.

**Busca concluída (13/08/2026 ~09:00, 2ª rodada)**: 6º candidato
(`hidden=32/heads=2/epochs=150`) venceu por margem dentro do ruído (PR-AUC
0,0249 vs. 0,0244 do candidato só-com-mais-épocas, desvio-padrão
~0,025-0,029 nos dois) — mas custava ~3x mais tempo de treino. Decisão:
descartar o `heads=2` (empate estatístico não justifica o custo) e usar
**`epochs=150` isolado** (`heads=1`, `hidden=32` default) — mais simples,
mais barato, e isola `epochs` como a alavanca real do ganho (a busca já
mostrou que `hidden`/`heads` sozinhos não ajudam muito).

**Em andamento (relançado 13/08/2026 ~23:43)**: `comparar_baselines.py`
com `han_hgt` usando `epochs=150` (commit `affe891`). PID 36441, log em
`~/comparar_baselines_v4_han_hgt_tunado.log`. Estimativa corrigida: ~17-
17,5h total (tabular ~5min + GNN homogênea ~2,8h + HAN/HGT ~14,4h — o
HAN/HGT sozinho já é ~7,2h **por rótulo**, 2 rótulos). Isso vira o
resultado v4, substituindo o v3 como resultado final reportado — se o
HAN/HGT tunado ainda perder pro tabular/GNN homogênea, o resultado
negativo fica muito mais defensável ("buscamos hiperparâmetros e não
adiantou" é mais forte que "usamos uma config só"). Depois disso: etapa
7.7 (sensibilidade) e etapa 8 (publicação).

## Armadilhas já identificadas (não repetir)

- **Não commitar o `.db`/`.db.gz` do dataset** — chega por download da GitHub
  Release (ver seção 4 de `docs/research_plan.md`), fica em `data/raw/`
  (ignorado pelo `.gitignore`). Contém dado pessoal (nome de sócio, endereço),
  mesmo com CPF mascarado.
- **Extração de metapath deve usar produto de matrizes esparsas**, não DFS em
  `networkx` — **resolvido em 08/08/2026**: `SparseMetaPathExtractor` em
  `src/graph/metapaths.py` faz isso via `scipy.sparse`; o DFS (`MetaPathExtractor`)
  foi mantido só para depuração/cruzamento com Cypher em amostras pequenas, não
  é mais o caminho de produção.
- **Não usar `processos_judiciais` como rótulo** — é ruidoso (match fuzzy por
  nome, não CNPJ direto).
- **Split temporal simples é arriscado** dado o N pequeno de positivos (188) —
  preferir validação cruzada estratificada repetida.
