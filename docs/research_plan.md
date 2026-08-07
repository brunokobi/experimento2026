# Plano de Pesquisa — Dissertação de Mestrado

> **Objetivo do documento**: registrar a pergunta de pesquisa, o escopo e as decisões
> metodológicas da dissertação, orientadas para produzir um artigo publicável em
> veículo de classificação (Qualis/CAPES) alta. Este documento é vivo — deve ser
> atualizado a cada decisão relevante tomada.
>
> **Nota sobre orientação**: pesquisa conduzida sem orientador formal — as decisões
> registradas aqui são de responsabilidade do próprio pesquisador (Bruno Kobi),
> com apoio técnico de IA.

**Status**: fonte de dados e tarefa-fim confirmadas com números reais (seções 4 e 5) —
próximo passo é desenhar o schema da HIN em código a partir do banco real (seção 10).

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
| `processos_judiciais` | 1.314.602 | — | **ruidoso**: 99,9% casado por nome (fuzzy), não CNPJ — ver riscos |
| `registros_jucees` | 88.349 | 88.349 | metadado (natureza jurídica, constituição) |
| `marcas_inpi` / `beneficios_fiscais` / `contratos_pncp` / `contratos_governamentais` | 36.963 / 32.006 / 7.013 / 894 | — | sinais auxiliares exploratórios |

**Escopo geográfico**: Grande Vitória (ES) — 7 municípios (Vitória, Vila Velha, Serra,
Cariacica, Viana, Guarapari, Fundão). Escala **totalmente tratável em memória** — não
há necessidade de infraestrutura de sampling distribuído nesta fase da pesquisa.

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
- **Ruído a isolar**: `processos_judiciais` é casado por nome da parte (fuzzy) em
  99,9% dos casos, não por CNPJ direto — não deve ser tratado como sinal confiável de
  fraude; entra, quando muito, como feature de baixa confiança (usar `match_confianca`
  para filtrar/ponderar).

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
- **`processos_judiciais` é ruidoso** (99,9% casado por nome, fuzzy) — não usar como
  rótulo; usar com cautela como feature, ponderado por `match_confianca`.
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

Com fonte de dados, rótulo e divisão Neo4j/PyG decididos, o próximo passo é técnico:
**adaptar o schema em código** (`src/config/settings.py`, `src/data/loaders.py`,
`src/graph/hin_builder.py`) ao banco real (`empresas`, `socios`,
`sancoes_administrativas`, `vinculos_politicos`, `dividas_ativas`) no lugar do schema
genérico de exemplo usado até aqui — isso ainda não foi feito neste repositório.

---

*Ver também o scaffold de código em `src/` (config, loaders, HIN builder, extração
de metapaths, testes) — construído com schema genérico de exemplo; ainda pendente de
adaptação ao schema real descrito na seção 4.*
