# Plano de Pesquisa — Dissertação de Mestrado

> **Objetivo do documento**: registrar a pergunta de pesquisa, o escopo e as decisões
> metodológicas da dissertação, orientadas para produzir um artigo publicável em
> veículo de classificação (Qualis/CAPES) alta. Este documento é vivo — deve ser
> atualizado a cada decisão relevante tomada.
>
> **Nota sobre orientação**: pesquisa conduzida sem orientador formal — as decisões
> registradas aqui são de responsabilidade do próprio pesquisador (Bruno Kobi),
> com apoio técnico de IA.

**Status (08/08/2026)**: fonte de dados e tarefa-fim confirmadas com números reais
(seções 4 e 5); schema real da HIN e extração de metapath via matriz esparsa já
implementados em código (seção 12) — próximo passo é validar essa extração contra
o banco real (não só dado sintético) e então os baselines/treino da GNN (seções 7 e 8).

---

## 1. Motivação e lacuna na literatura

Modelos tabulares (ex.: XGBoost sobre dados cadastrais da Receita Federal) e GNNs
homogêneas ignoram um sinal estrutural importante em risco corporativo: padrões de
**sócio, endereço e contador compartilhados** entre empresas, e — ângulo pouco
explorado na literatura — **vínculo político do sócio** (candidatura/doação eleitoral).
Esse sinal é exatamente o que um metapath em uma Rede Heterogênea de Informação (HIN)
captura.

A literatura brasileira de risco/fraude corporativa é majoritariamente tabular.
Trabalhos internacionais com HAN/HGT/R-GCN em grafos corporativos existem, mas
concentrados em previsão de insolvência/risco de crédito — um espaço já saturado.
**Detecção de risco de sanção administrativa via metapaths societários e de conexão
política, no contexto de dados públicos brasileiros, é pouco explorada** — esse é o
gap que esta dissertação ataca.

## 2. Pergunta de pesquisa

> Metapaths estruturais em uma HIN de empresas (sócio comum, endereço comum, vínculo
> político) melhoram a identificação de empresas com sanção administrativa confirmada,
> em relação a um baseline tabular — e quais metapaths carregam mais sinal?

**Título provisório**: *Detecção de Risco de Sanção Administrativa em Redes
Heterogêneas de Empresas via Metapaths e Graph Neural Networks: um Estudo com Dados
Reais da Grande Vitória (ES)*.

## 3. Contribuições reivindicadas

1. Uma HIN construída sobre dados reais de empresas da Grande Vitória (ES), com schema
   e metapaths desenhados para sinais de risco corporativo (sócio comum, endereço
   comum, vínculo político via TSE).
2. Comparação empírica entre HAN/HGT, GNN homogênea e baseline tabular, avaliada com
   métricas apropriadas a rótulo raro (PR-AUC, Precision@k) — não acurácia, que engana
   nesse regime.
3. Análise interpretável de quais metapaths carregam mais sinal — inclui testar
   explicitamente se vínculo político (sinal com pouquíssima interseção direta com o
   rótulo, ver seção 5) carrega sinal indireto via rede, algo que só um modelo
   relacional captura.

## 4. Fonte de dados (confirmada e verificada)

A tese usa um dataset já existente e mantido pelo próprio pesquisador:
[`projeto_grande_vitoria_empresas`](https://github.com/brunokobi/projeto_grande_vitoria_empresas)
— um pipeline ETL que consolida fontes públicas oficiais num SQLite único, cruzadas
por CNPJ. **Verificado diretamente no banco real** (não só na documentação do repo):

| Tabela | Linhas | Empresas distintas | Papel na HIN |
|---|---|---|---|
| `empresas` | 344.130 | — (universo) | nó central |
| `socios` | 231.890 | — | nó — liga a empresas via `participa_de` |
| `sancoes_administrativas` | 322 | **188** | **rótulo principal** (ver seção 5) |
| `dividas_ativas` | 158.675 | 32.754 | sinal auxiliar (não é o rótulo) |
| `vinculos_politicos` (TSE) | 4.557 | ~4.000 | nó/sinal auxiliar — ângulo de novidade |
| `processos_judiciais` | 110.489 (parcial — pipeline `djen` ainda rodando) | 2.374 | **ruidoso**: casado por nome via DJEN, `match_confianca='nome'`, não CNPJ — ver riscos |
| `registros_jucees` | 88.349 | 88.349 | metadado (natureza jurídica, constituição) |
| `marcas_inpi` / `beneficios_fiscais` / `contratos_pncp` / `contratos_governamentais` | 36.963 / 32.006 / 7.013 / 894 | — | sinais auxiliares exploratórios |

**Escopo geográfico**: Grande Vitória (ES) — 7 municípios (Vitória, Vila Velha, Serra,
Cariacica, Viana, Guarapari, Fundão). Escala **totalmente tratável em memória** — não
há necessidade de infraestrutura de sampling distribuído nesta fase da pesquisa.

**Nota de reverificação (2026-08-08)**: números re-conferidos direto no
`grande_vitoria.db` real (não só na documentação do repo do dataset, que estava
desatualizada em um ponto: a etapa `datajud` foi descontinuada e substituída por
`djen`, que casa processo por nome via Comunica API do CNJ/PJe — ver
`src/djen_client.py` do dataset). `empresas`, `socios` e `sancoes_administrativas`
bateram exatamente com o já documentado. `processos_judiciais` mudou bastante: o
número antigo (1.314.602) era de antes da migração; o pipeline `djen` reconstruiu a
tabela do zero e ainda está rodando (110.489 linhas / 2.374 empresas casadas até o
momento desta verificação) — isso não afeta o rótulo (seção 5), só o volume desse
sinal auxiliar de baixa confiança.

**LGPD já resolvida na origem**: `cpf_parcial` já vem mascarado da Receita Federal no
próprio dataset — não é preciso pseudonimizar nada adicionalmente para uso na tese.

## 5. Rótulo e desenho do problema (decisão travada a partir dos números reais)

**Achado crítico**: apenas **188 empresas em 344.130 (0,055%)** têm qualquer sanção
administrativa registrada (`CEIS`: 139 empresas · `CNEP`: 29 · `TCEES`: 16 · `CEPIM`:
10 · `TRABALHO_ESCRAVO`: 2). Isso é desbalanceamento extremo — mais severo do que
qualquer estimativa inicial deste plano. Isso muda o desenho do problema:

- **Rótulo principal**: as 188 empresas com sanção administrativa confirmada
  (`sancoes_administrativas`) são tratadas como **positivos confirmados** — não como
  "toda a fraude que existe", só a que já foi pega. É um cenário de **rótulo
  positivo-incompleto (PU — positive/unlabeled)**, comum e aceito na literatura de
  detecção de fraude/risco.
- **Achado crítico adicional (2026-08-08) — parte do rótulo é indireta, via sócio**:
  o campo `match_confianca` de `sancoes_administrativas` revela que das 188 empresas
  positivas, **148 são "direto"** (a própria empresa está listada em CEIS/CNEP/TCEES/
  CEPIM/TRABALHO_ESCRAVO) e **41 são "socio"** (a empresa em si não está sancionada —
  foi rotulada como positiva porque um dos seus sócios está ligado a uma entidade
  sancionada; ocorre só dentro do CEIS: 46 registros / 41 empresas distintas; 148+41=189
  vs. 188 distintos por haver 1 empresa com sanção nos dois tipos). **Isso é um risco de
  circularidade/vazamento para a pergunta de pesquisa**: o metapath `empresa-sócio-empresa`
  é justamente a hipótese estrutural central da dissertação (seção 2) — para essas 41
  empresas, o rótulo já foi *construído* usando sócio comum, então um modelo que "descubra"
  que sócio comum prediz risco não estaria descobrindo sinal novo nesse subconjunto, só
  reproduzindo a regra de rotulagem. Mitigação a decidir antes do experimento principal:
  reportar métricas com o rótulo restrito às 148 "direto" como análise primária, e as 188
  completas (ou o ganho marginal das 41 "socio") como análise de sensibilidade separada —
  não misturar as duas sem declarar.
- **Framing do problema**: não é classificação binária balanceada — é **detecção de
  anomalia / ranking de risco**. Métrica principal: **PR-AUC** e **Precision@k** (ex.:
  "das 20 empresas mais bem ranqueadas pelo modelo, quantas eram sancionadas de
  verdade"), não acurácia.
- **Sinais auxiliares, não rótulo**: `dividas_ativas` (mais denso, 32.754 empresas,
  mas é dívida fiscal — não necessariamente fraude) e `vinculos_politicos` entram como
  *features*/nós extras na HIN, não como rótulo. A interseção direta entre sanção e
  vínculo político é de só **4 empresas** — ou seja, se houver sinal de vínculo
  político, ele é **indireto** (via rede, não correlação direta), que é exatamente a
  hipótese que uma GNN testa e um modelo tabular não consegue.
- **Ruído a isolar**: `processos_judiciais` é casado por nome da parte via DJEN, não
  por CNPJ direto — não deve ser tratado como sinal confiável de fraude; entra, quando
  muito, como feature de baixa confiança. **Correção (2026-08-08)**: o valor real do
  campo é `match_confianca='nome'` (confirmado contra os destinatários da publicação
  para evitar homônimo) — não existe granularidade `'direto'`/`'fuzzy'` nessa tabela
  como o `schema.sql` do dataset chegou a sugerir (comentário desatualizado, já
  corrigido lá); na prática o campo aqui é categórico e serve para **filtrar**, não
  para ponderar continuamente.

## 6. Papel do grafo (HIN), Neo4j e PyTorch Geometric

Divisão de responsabilidade decidida explicitamente (para não virar dependência morta
nem duplicação de esforço):

- **HIN (a estrutura central)**: sem ela, o problema vira só "planilha com uma coluna
  de sanção" — a HIN é o que permite perguntar "essa empresa tem risco pelo que ela
  é, ou por quem ela conhece" (sócio comum, mesmo contador, mesmo endereço, vínculo
  político).
- **Neo4j** — exploração, validação e prova visual, **não treino**:
  - Consultas Cypher exploratórias (ex.: "empresas a até 2 saltos de uma empresa
    sancionada via sócio compartilhado").
  - Figuras da dissertação/defesa (Neo4j Bloom) — casos reais ilustrados como grafo.
  - Métricas clássicas de rede via Neo4j GDS (centralidade, comunidades) — usadas como
    comparação: "a GNN aprende algo que uma métrica clássica de rede já não entregava
    de graça?" — pergunta que revisor de peso costuma fazer.
  - Depuração da extração de metapath (conferir o resultado do código Python contra
    uma query Cypher direta).
- **PyTorch Geometric (`HeteroData`)** — o motor de fato:
  - Recebe o grafo (via export do `HINBuilder`) e treina o modelo GNN (HAN/HGT).
  - É o que produz o resultado quantitativo (PR-AUC, Precision@k) reportado no artigo.

### 6.1 Onde cada peça roda (decisão de infraestrutura)

- **Neo4j**: hospedado na VPS pessoal do pesquisador (Oracle Cloud, ARM64, já
  orquestrada via Coolify/Docker/Traefik para outros serviços pessoais) — mesmo
  padrão operacional já em uso, sem impacto nos serviços existentes. Justificativa:
  o grafo é pequeno (344k nós — ver seção 4), a imagem oficial do Neo4j é multi-arch
  (sem a armadilha de build ARM64 já vista com outras imagens naquela VPS), e ganha-se
  acesso persistente de qualquer máquina, sem depender do notebook local estar ligado.
- **PyTorch Geometric / treino da GNN / notebooks**: máquina **local**. A VPS é
  CPU-only/ARM64 (free tier) — sem GPU — e as extensões nativas do PyTorch Geometric
  (`torch-scatter`/`torch-sparse`) são historicamente instáveis em ARM. Treinar ali
  não traria ganho de velocidade e arriscaria consumir tempo de pesquisa debugando
  build em vez de rodando experimento.
- **Segredos de acesso** (IP, chaves SSH, credenciais) **nunca** vão neste
  repositório — este é público. Ficam exclusivamente no `.env` local (já ignorado
  pelo `.gitignore`) e no repositório privado de infraestrutura pessoal do
  pesquisador. Este documento registra a decisão arquitetural, não os segredos.

Fluxo: `Neo4j (explorar/validar/visualizar) → HeteroData (PyTorch Geometric) → GNN →
ranking de risco + quais metapaths pesaram mais`.

## 7. Metodologia

- **Dados**: `grande_vitoria.db` (ver seção 4) — não é mais necessário buscar/validar
  fonte externa de rótulo (CEIS/CNEP/TCU já vêm ingeridos e casados por CNPJ).
- **Split**: dado o N pequeno de positivos (188), avaliar com **validação cruzada
  estratificada repetida**, não um único split temporal — um único holdout temporal
  arrisca deixar poucos ou nenhum positivo no teste. Continua sendo necessário
  verificar que features usadas no treino não "vazam" informação posterior à sanção
  (ex.: não usar `data_fim` da sanção como feature).
- **Extração de metapath via matriz esparsa** (produto de matrizes de adjacência),
  não busca em profundidade (DFS) — DFS não escala e reviewer de sistemas percebe essa
  limitação.
- **Baselines**: tabular (XGBoost/LightGBM, com técnicas de desbalanceamento — ex.:
  class weighting, focal loss), GNN homogênea, HAN/HGT.
- **Rigor estatístico**: múltiplas seeds + teste estatístico (ex.: Wilcoxon) nas
  comparações entre modelos.

## 8. Ética e LGPD

- CPF de sócios **já vem mascarado** na fonte (Receita Federal) — sem trabalho
  adicional de pseudonimização necessário para uso na tese.
- Uso de dado público, com justificativa documentada; verificar exigência de parecer
  de comitê de ética da instituição, se houver.
- Vínculo político (TSE) é dado público de candidatura/doação eleitoral — mesmo
  assim, tratar com cautela na exposição de nomes individuais nos resultados
  publicados (agregar/anonimizar exemplos usados como ilustração).

## 9. Riscos declarados

- **Desbalanceamento extremo (188/344.130)** — mitigação: framing de anomalia/ranking
  (seção 5), não classificação balanceada; validação cruzada estratificada repetida.
- **Rótulo é positivo-incompleto, não exaustivo** — a ausência de sanção não significa
  ausência de irregularidade, só que não foi pega. Isso deve ser dito explicitamente
  no texto da dissertação, não escondido.
- **`processos_judiciais` é ruidoso** (casado por nome via DJEN, `match_confianca='nome'`,
  não CNPJ direto) — não usar como rótulo; usar com cautela como feature de baixa
  confiança (filtrar, não ponderar continuamente — o campo é categórico).
- **Rótulo parcialmente circular com a hipótese estrutural (novo, 2026-08-08)**: 41 das
  188 empresas positivas (`sancoes_administrativas.match_confianca='socio'`) foram
  rotuladas via sócio comum com entidade sancionada, não por sanção direta na própria
  empresa — mesmo mecanismo do metapath central da pergunta de pesquisa (seção 2).
  Mitigação: reportar resultado primário com rótulo restrito a `match_confianca='direto'`
  (148 empresas) e tratar as 41 "socio" como análise de sensibilidade separada, deixando
  explícito no texto qual definição de rótulo foi usada em cada resultado.
- **Risco de novidade "comida"** por publicação concorrente — mitigação: publicar um
  resultado parcial em workshop/preprint antes do artigo principal.

## 10. Venues-alvo (realistas para o prazo de mestrado)

- **Rede de segurança nacional**: BRACIS, SBBD.
- **Alvo principal**: periódicos de bom impacto e ciclo de revisão compatível com o
  prazo (ex.: *Expert Systems with Applications*, *Knowledge-Based Systems*,
  *Decision Support Systems*).
- Evitar KDD/WWW/CIKM como alvo primário — ciclo e barreira de aceitação
  incompatíveis com o tempo disponível.

## 11. Cronograma orientado a publicação

1. **Marco 1** — schema da HIN implementado a partir do banco real + metapaths
   validados numa amostra, com resultado preliminar.
2. **Marco 2** — submissão de resultado parcial em workshop/BRACIS.
3. **Marco 3** — experimento completo com todos os baselines e ablation por metapath.
4. **Marco 4** — submissão do artigo principal ao periódico-alvo.

## 12. Próximo passo imediato

**Status (08/08/2026): primeira versão implementada.** `src/config/settings.py` e
`src/data/loaders.py` (`GrandeVitoriaLoader`) já apontam para o schema real
(`empresas`, `socios`, `sancoes_administrativas`, `dividas_ativas`,
`vinculos_politicos`); `src/graph/build_hin.py` (`build_empresas_hin`) monta a HIN
real com os nós `empresa`/`socio`/`endereço`/`município`/`vínculo_político` e expõe
o rótulo como `data["empresa"].y_direto` / `.y_qualquer` (nunca como feature, para
não vazar informação e para preservar a distinção direto vs. via-sócio da seção 5).
`docs/research_plan.md` e o código foram atualizados juntos — ver `README.md`,
seção "Etapas do trabalho", para o estado de cada etapa.

**Feito também (08/08/2026, continuação)**: `SparseMetaPathExtractor`
(`src/graph/metapaths.py`) substitui o DFS como caminho de produção — calcula a
matriz de comutação via produto de matrizes de adjacência esparsas (`scipy.sparse`),
testado com HIN sintética de 50k+20k nós sem densificar. O DFS (`MetaPathExtractor`)
foi mantido só para depuração/cruzamento com Cypher em amostras pequenas (seção 6).

**Feito também (08/08/2026, validação contra o banco real)**: rodei o pipeline
completo contra `grande_vitoria.db` de verdade (`scripts/validar_hin_real.py`).
HIN construída em 23,5s / 0,44 GB de pico; os 3 metapaths de hipótese extraídos
em <0,2s cada, sem densificar. Números reais: 142.844 sócios distintos (de
231.890 linhas — esperado, é o mecanismo de sócio comum funcionando, não
colisão espúria), 181.268 endereços distintos, 519 pessoas com vínculo
político, rótulo `y_direto=148`/`y_qualquer=188`.

Dois bugs reais só apareceram aqui (nenhum aparecia no dado sintético dos
testes — documentando para não repetir):
1. `pandas.read_sql_query` devolve `NaN` (float) para coluna de texto nula, não
   `None`/string vazia — `nan or ""` não pega isso (`NaN` é *truthy* em
   Python). Quebrava `_chave_socio`/`_normalizar_texto` em `build_hin.py`.
   Corrigido com `pd.isna()` explícito; tem teste de regressão.
2. `município` é nó "hub" de baixíssima cardinalidade (7 nós para 344k
   empresas): o produto esparso de `empresa_municipio_empresa` tentou alocar
   **187 GiB**. Corrigido com `MetapathExplosionError` (`SparseMetaPathExtractor`
   estima o tamanho do produto antes de calcular e recusa com mensagem clara).
   A própria estimativa tinha um bug de overflow silencioso em `int32` (o
   `indptr` do scipy vem nesse tipo; `np.dot` sem cast para `int64` estourava
   pra um número negativo, que passava batido pelo limite) — só detectável com
   a distribuição real e desigual dos municípios, não com números sintéticos
   uniformes. Tem teste de regressão que reproduz os dois bugs.

**Feito também (08/08/2026, Neo4j)**: `src/graph/neo4j_export.py`
(`export_hin_to_neo4j`) exporta a HIN real inteira para o Neo4j — rodado com
sucesso (`scripts/exportar_hin_neo4j.py`, 3min47s) contra um Neo4j 5 + GDS na
VPS pessoal (Oracle Cloud/Coolify), **acesso só via túnel SSH**, nenhuma
porta pública (o grafo tem dado pessoal). Conferido via query direta:
344.130 `Empresa`, 142.844 `Socio`, 181.268 `Endereco`, 7 `Municipio`, 519
`VinculoPolitico`; arcos `PARTICIPA_DE` (231.890), `SEDIADA_EM` (344.130),
`LOCALIZADA_EM` (344.130), `TEM_VINCULO_POLITICO` (833, não 866 — 33 pares
empresa-político colapsam por `MERGE`, ver nota abaixo).

**Observação (não é bug, mas anotar)**: 33 dos 866 vínculos políticos têm o
mesmo par (empresa, político-normalizado) repetido — candidatura em anos
diferentes, ou variação mínima de grafia que a normalização de nome já
junta. No Neo4j isso vira 1 relacionamento só (`MERGE` é idempotente por
natureza — correto). No `HeteroData`/`SparseMetaPathExtractor`, porém, o
`edge_index` **não** deduplica esses 33 pares, e `scipy.sparse` soma
entradas duplicadas — logo a matriz de comutação de
`empresa_vinculo_politico_empresa` conta peso 2 (não 1) para esses 33
pares específicos. Imprecisão pequena (33 em 519 vínculos), não bloqueia
nada agora, mas vale deduplicar o `edge_index` de `vinculo_politico` antes
de reportar qualquer análise fina desse metapath especificamente.

**Feito também (08/08/2026, etapa 7.1 — feature engineering tabular)**:
`src/features/tabular.py` (`build_feature_matrix`) monta a matriz de
features por empresa (capital social, contagem de sócios, dívida ativa
agregada, vínculo político booleano, porte/regime tributário/CNAE/município
one-hot) junto com `y_direto`/`y_qualquer`, sem nenhuma coluna de
`sancoes_administrativas` além do rótulo. Validado contra o banco real:
344.130 empresas, 5,5s, 107 colunas, 0 `NaN`. Achado: `porte` no banco é
código numérico da Receita (`"01"`/`"03"`/`"05"`), não texto descritivo —
importa para interpretar resultado depois. A etapa 7 (seção 7) foi quebrada
em subetapas (7.1–7.7) no `README.md` para acompanhar o progresso.

**Feito também (08/08/2026, etapa 7.2 — harness de avaliação)**:
`src/evaluation/harness.py` implementa `precision_at_k`, `evaluate_repeated_cv`
(validação cruzada estratificada repetida via `RepeatedStratifiedKFold`,
retorna PR-AUC + Precision@k por fold) e `compare_models` (teste de Wilcoxon
pareado). Único ponto de acoplamento com cada modelo é uma função
`fit_predict(x_train, y_train, x_test) -> scores` — o mesmo harness serve
para o baseline tabular, a GNN homogênea e o HAN/HGT sem modificação.

**Feito também (08/08/2026, etapa 7.3 — primeiro resultado quantitativo)**:
`src/models/tabular_baseline.py` (`xgboost_fit_predict`) rodado contra o
banco real via validação cruzada estratificada repetida (5 splits × 10
repeats = 50 folds, `scripts/rodar_baseline_tabular.py`, ~5min total):

| Rótulo | Positivos | PR-AUC | Taxa-base | Lift sobre o acaso |
|---|---|---|---|---|
| `y_direto` | 148 | 0,0081 ± 0,0036 | 0,043% | **~18,8×** |
| `y_qualquer` | 188 | 0,0085 ± 0,0029 | 0,055% | **~15,5×** |

**Interpretação**: os valores absolutos de PR-AUC parecem baixos, mas isso é
esperado com taxa-base tão extrema — o número relevante é o *lift* sobre o
acaso, não o valor absoluto (é exatamente por isso que a seção 5 já mandava
usar PR-AUC/Precision@k em vez de acurácia). O baseline tabular tem lift
**menor** em `y_qualquer` do que em `y_direto` — consistente com a hipótese
central da dissertação (seção 2): dado tabular isolado não enxerga risco por
associação (as 41 empresas via-sócio de `y_qualquer` não têm nenhum sinal
tabular próprio que as distinga, só a conexão de rede) — é exatamente o que
a HIN/GNN (etapas 7.4/7.5) deveria capturar e o tabular não consegue.

**Nota metodológica**: o desvio-padrão do Precision@k frequentemente supera
a própria média (ex.: Precision@10 em `y_direto` = 0,012 ± 0,039) — com só
~30 positivos por fold de teste, acertar exatamente os k primeiros é muito
ruidoso. Considerar k maiores (50, 100) nas rodadas dos próximos baselines
para uma estimativa mais estável, sem descartar k pequenos (são os que
importam pra "leitura de investigador": poucos casos pra checar manualmente).

**Pendente a seguir**:
- Etapa 7.4 (baseline GNN homogênea via `SparseMetaPathExtractor`) — primeiro
  modelo que de fato usa a estrutura de rede, comparar contra o número acima.
- `processos_judiciais` ainda não entra na HIN (pipeline `djen`, no repo do dataset,
  ainda em andamento; o campo é ruidoso por design — ver seção 9).
- Identidade de sócio (CPF mascarado + nome) e de endereço (logradouro+número+CEP
  normalizados) são heurísticas de primeira versão — não resolvem homônimos nem
  variação de grafia; documentado como limitação em `src/graph/build_hin.py`.
- Vínculo político ainda não é ligado ao sócio da própria empresa por nome (fica
  como nó auxiliar ligado direto à empresa) — juntar as duas identidades exigiria
  resolução de nome mais cuidadosa.
- Baselines (tabular/GNN homogênea/HAN-HGT) e avaliação (seções 7-8) ainda não
  implementados — é o próximo passo depois da extração rodar no banco real.

---

*Ver também o scaffold de código em `src/` (config, loaders, HIN builder/build_hin,
extração de metapaths DFS+esparsa, testes) — schema real e extração escalável já
implementados (seção acima); o próximo passo são os baselines/treino da GNN
(seções 7-8), depois de validar a extração contra o banco real.*
