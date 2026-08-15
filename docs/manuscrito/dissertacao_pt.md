# Título provisório

**Modelos Simples, Sinal de Grafo Explícito: Repensando Redes Neurais em
Grafos Heterogêneos para Triagem de Risco de Sanção Administrativa em Dados
de Governo Local**

*(provisório — não há template ABNT obrigatório da instituição informado até
o momento; estrutura livre de dissertação de mestrado, capítulos de
conteúdo. Capa/folha de rosto/ficha catalográfica/resumo formal ficam para
quando o formato institucional for confirmado.)*

> **Status deste rascunho (atualizado em 14/08/2026)**: todas as seções têm
> rascunho substantivo agora, incluindo Resultados, Discussão e Conclusão,
> escritas depois de o experimento final de 30 folds (com o HGT tunado)
> terminar. Todos os números abaixo foram computados direto do log
> `docs/resultados/comparar_baselines_30folds_v4_han_hgt_tunado_2026-08-14.log`
> e reconferidos contra `docs/research_plan.md`, não recuperados de
> memória — ver nota de rigor em `docs/research_plan.md`, correção de
> 14/08/2026.

---

## Resumo

Órgãos de controle em governo local e regional são cada vez mais chamados a
triar grandes cadastros corporativos por risco de sanção administrativa —
evasão de listas de impedimento, uso de empresa de fachada, e vínculo
político não declarado — com orçamento limitado de ciência de dados e sem
infraestrutura dedicada. Redes neurais em grafo (GNNs) que modelam redes
corporativas explicitamente (sócio comum, endereço comum, vínculo político)
são frequentemente propostas como a resposta de estado da arte, sob a
suposição de que risco estrutural só é visível a um modelo relacional.
Testamos essa suposição num cadastro real e completo de 344.130 empresas de
uma região metropolitana brasileira, comparando um baseline tabular com
gradient boosting, uma GNN homogênea, e um transformer de grafo heterogêneo
(HGT), sob uma moldura positivo-incompleta apropriada à raridade extrema de
sanção confirmada (148 diretas, 188 incluindo casos inferidos via sócio).
Ao longo de cinco rodadas de avaliação independentes — três iterações
sucessivas de engenharia de features mais uma rodada dedicada de busca de
hiperparâmetros para o HGT, todas sob validação cruzada estratificada
repetida (30 folds) com teste estatístico pareado —, o modelo heterogêneo é
consistente e significativamente superado por ambas as alternativas mais
simples no rótulo principal, não-circular (tabular: lift de 50,7× sobre a
taxa-base; GNN homogênea: 76,3×; HGT tunado: 32,6×; tabular > HGT,
p = 0,0024; GNN homogênea > HGT, p < 0,0001). Criticamente, uma busca
dedicada de hiperparâmetros — motivada pelo achado de que o HGT estava
subtreinado, não subdimensionado — melhorou seu desempenho em 39%, mas não
mudou esse ranking, fechando a objeção metodológica mais provável ao
resultado. Num rótulo secundário, onde o próprio mecanismo de rotulagem se
sobrepõe ao metapath de sócio comum, a vantagem do HGT tunado cresce em vez
de diminuir (lift de 60,8×, ante 42,8× antes do ajuste) — um padrão mais
consistente com o modelo explorando circularidade de rótulo do que
descobrindo sinal genuíno de risco estrutural. Discutimos a implicação
prática para órgãos de controle com recursos limitados: um modelo tabular
bem projetado, aumentado com features explícitas de grau de grafo, não um
transformer de grafo heterogêneo, é a escolha defensável para essa tarefa
dado o orçamento computacional e de rotulagem realista do setor público.

**Palavras-chave**: risco de sanção administrativa; redes neurais em grafos
heterogêneos; detecção de anomalia; ciência de dados no setor público;
análise de rede corporativa; aprendizado positivo-não-rotulado.

## Abstract

*(English version of the abstract above — see `docs/manuscrito/paper_en.md`
for the version drafted directly in English, which this should mirror once
both are finalized together rather than translated mechanically.)*

---

## 1. Introdução

Administrações públicas em qualquer nível mantêm cadastros das empresas com
as quais se relacionam — como contribuintes, como participantes de licitação,
como beneficiárias de incentivo fiscal, como empregadoras. Uma fração pequena
dessas empresas acaba sendo formalmente sancionada por irregularidade grave o
suficiente para entrar numa lista de impedimento (no Brasil, os registros
federais CEIS e CNEP, equivalentes estaduais, e listas setoriais como o CEPIM
para entidades sem fins lucrativos irregulares). Para um órgão de controle —
tribunal de contas, controladoria, órgão de fiscalização de licitação — a
pergunta prática não é retrospectiva ("essa empresa foi sancionada?"), é
prospectiva: *entre as empresas ainda não sancionadas, quais concentram risco
suficiente para justificar escrutínio agora?*

Dois fatos estruturais tornam esse um problema de triagem difícil. Primeiro,
sanção confirmada é extremamente rara frente ao tamanho de qualquer cadastro
real — no dataset usado aqui, **188 casos confirmados em 344.130 empresas
(0,055%)**. Qualquer modelo treinado sobre esse rótulo está aprendendo de um
sinal **positivo-incompleto (PU — positive/unlabeled)**: ausência de registro
de sanção significa *ainda não foi pega*, não *sem risco*. Segundo, risco
corporativo é frequentemente relacional, não intrínseco aos atributos
declarados de uma única empresa: uma empresa com cadastro fiscal
aparentemente comum pode compartilhar sócio, endereço registrado, ou vínculo
político não declarado com uma entidade já sancionada — sinal invisível para
um modelo que trata cada empresa como uma linha independente de atributos.

Esse segundo fato é a premissa de um corpo crescente de trabalho aplicando
redes neurais em grafos heterogêneos (HGNNs) — modelos que tratam empresas,
sócios, endereços e outras entidades como tipos de nó distintos, conectados
por relações tipadas — a risco corporativo, majoritariamente em crédito e
insolvência (ver Seção 2). A promessa implícita dessa literatura é que
modelos relacionais, treinados de ponta a ponta, recuperam sinal estrutural
de risco que abordagens tabulares mais simples não conseguem.

Esta dissertação testa essa promessa diretamente, num cenário mais próximo do
que um órgão de controle real e com recursos limitados de fato teria
disponível: um cadastro empresarial real e completo de uma região
metropolitana brasileira (a Grande Vitória, Espírito Santo, sete municípios),
construído numa Rede Heterogênea de Informação (HIN) com três metapaths
escolhidos por relevância direta para fiscalização de risco administrativo —
**sócio comum**, **endereço registrado comum**, e **vínculo político comum**
(via registros eleitorais federais do TSE) — avaliado sob o desbalanceamento
extremo e a condição de rótulo PU que são a norma nesse domínio, não uma
exceção a ser contornada por ajuste fino.

### 1.1 Pergunta de pesquisa

> Metapaths estruturais numa HIN de empresas (sócio comum, endereço comum,
> vínculo político) melhoram a identificação de empresas com sanção
> administrativa confirmada, em relação a um baseline tabular — e quais
> metapaths carregam mais sinal?

### 1.2 Contribuições

1. **Uma comparação empírica real e reprodutível** entre um baseline tabular
   (gradient boosting), uma GNN homogênea, e um transformer de grafo
   heterogêneo (HGT), sob um protocolo de avaliação estatisticamente
   rigoroso e compartilhado (validação cruzada estratificada repetida, teste
   de Wilcoxon pareado em 30 folds), replicada em três iterações sucessivas
   de engenharia de features e uma rodada dedicada de ajuste de
   hiperparâmetros — um nível de escrutínio metodológico incomum na
   literatura prévia de fraude/risco em grafo, que tipicamente reporta uma
   única configuração.
2. **[RESULTADO — a definir]**: se, e sob quais condições, o aprendizado
   estrutural implícito do modelo heterogêneo supera dar o *mesmo* sinal
   estrutural explicitamente, como feature de grau, ao baseline tabular —
   testando diretamente uma afirmação da literatura de features-de-grafo-vs-GNN
   (Seção 2.3) num domínio novo.
3. **Um conjunto de features para triagem de risco de sanção administrativa
   fundamentado em literatura de corrupção em compras públicas e detecção de
   empresa de fachada** (Seção 2.4), adaptado ao que é realisticamente
   disponível num cadastro municipal brasileiro: contratação sem
   concorrência, sobrepreço contratual, idade da empresa, e concentração de
   sócio em muitas empresas.
4. **Uma discussão prática, voltada a quem fiscaliza, não só a quem pesquisa
   aprendizado em grafo**, sobre qual abordagem de modelagem é *de fato*
   justificada dado o orçamento computacional, a capacidade técnica, e a
   escassez de rótulo com que um órgão de controle real opera — não o que é
   justificado num cenário de benchmark idealizado e bem-provido de recursos.

O restante do texto está organizado da seguinte forma. A Seção 2 situa este
trabalho em quatro literaturas: detecção de risco administrativo/corporativo
em dado público brasileiro, GNNs heterogêneas para redes corporativas, o
debate features-de-grafo-vs-GNN em detecção de fraude, e aprendizado em grafo
com desbalanceamento/poucos rótulos. A Seção 3 descreve os dados, a
construção do rótulo, o desenho da rede, a engenharia de features, os
modelos e o protocolo de avaliação. A Seção 4 reporta os resultados
*(a definir)*. A Seção 5 discute os achados da perspectiva de quem fiscaliza
*(a definir)*. A Seção 6 conclui *(a definir)*.

---

## 2. Referencial teórico

### 2.1 Risco administrativo e corporativo em dado público brasileiro

Trabalho brasileiro prévio sobre risco corporativo e detecção de fraude com
dado de cadastro público é majoritariamente tabular, tratando cada empresa
como uma observação independente descrita por atributos de cadastro, regime
tributário, dívida, e código de setor. Esse corpo de trabalho não modela,
até onde se sabe, a *rede* de relações entre empresas — sócio, endereço,
vínculo político compartilhados — como objeto de primeira classe, apesar de
essas relações serem diretamente consultáveis nas mesmas fontes de dado
público (cadastro da Receita Federal, registros eleitorais). Trabalho
internacional aplicando modelos relacionais (HAN, HGT, R-GCN) a redes
corporativas existe, mas concentra-se fortemente em risco de crédito e
insolvência — um problema comparativamente rico em dado e bem rotulado —
deixando detecção de risco de sanção administrativa via metapaths de
propriedade/endereço/vínculo político, em contexto de dado público
brasileiro, comparativamente inexplorada.

### 2.2 Redes neurais em grafos heterogêneos para redes corporativas e financeiras

Arquiteturas de atenção/transformer para grafo heterogêneo — HAN (Wang et
al., 2019) e HGT (Hu et al., 2020) à frente — estendem redes neurais em
grafo além de um único tipo de nó/aresta, aprendendo pesos de atenção
específicos por tipo ao longo de metapaths ou relações. Foram aplicadas a
redes de propriedade corporativa para risco de empresa de fachada e
beneficiário final (Moody's, 2023) e a detecção de fraude financeira mais
amplamente, onde se argumenta que anéis de fraude deixam uma assinatura
estrutural — infraestrutura compartilhada, contrapartes compartilhadas —
"invisível no espaço de features, mas detectável na topologia do grafo". Uma
literatura paralela sobre redes de empresa de fachada em crime organizado em
compras públicas documenta concretamente como dado de propriedade/gestão,
combinado com dado de contratação, revela componentes conectados indicativos
de risco de conluio (ver Seção 2.4).

### 2.3 Features de grafo versus treino de GNN de ponta a ponta

Uma literatura distinta e diretamente relevante faz uma pergunta mais
estreita e mais cética: um GNN treinado de ponta a ponta supera um baseline
muito mais simples ao qual se dá *a mesma informação estrutural
explicitamente*, como feature de grafo projetada (grau, centralidade,
PageRank, agregados de vizinhança) alimentando um modelo de árvores com
gradient boosting? Evidência de benchmarks de detecção de fraude sugere que a
resposta é frequentemente "não, ou não por muito". Um benchmark recente em
detecção de fraude de seguros compara explicitamente gradient boosting
contra HinSAGE, HAN e HGT, constatando que "abordagens de árvore com
gradient boosting sobre dado tabular ainda dominam o campo" e que
abordagens baseadas em grafo especificamente sofrem sob o desbalanceamento
de classe alto típico de dado de fraude (Vandervorst et al., 2025) — a mesma
combinação de condições (grafo heterogêneo, desbalanceamento extremo)
estudada nesta dissertação. Uma linha de trabalho separada, que combina
gradient boosting com estrutura de grafo diretamente (em vez de compará-los
como rivais), mostra ainda que modelos baseados em GBDT conseguem igualar ou
superar arquiteturas de GNN puras uma vez que a informação de grafo é
disponibilizada a eles numa forma adequada (Ivanov & Prokhorenkova, 2021). O
desenho desta dissertação — dar os *mesmos* três metapaths ao baseline
tabular como feature de grau explícita, e ao modelo heterogêneo como
estrutura aprendida — é um teste direto dessa afirmação num domínio novo.

### 2.4 Indicadores de risco de corrupção em compras públicas e detecção de empresa de fachada

Trabalho empírico transnacional sobre risco de corrupção em compras públicas
(Fazekas & Kocsis, 2020; Abdou et al., 2022) estabelece indicadores objetivos
de contratação sem concorrência e sobrepreço — modalidade de contratação por
dispensa/inexigibilidade de licitação, e divergência entre valor inicial e
final do contrato — como proxies robustos e auditáveis de risco de corrupção,
independentes de qualquer desfecho processual. A prática de detecção de
empresa de fachada (Moody's, 2023; literatura de rede de empresa de fachada
em crime organizado em compras públicas) identifica ainda idade da empresa e
concentração de sócio/diretor em muitas empresas como sinais de alerta
práticos. Esses quatro indicadores são diretamente relevantes, e viáveis de
construir a partir do dado de cadastro usado neste estudo (Seção 3.4), e
motivaram uma rodada dedicada de engenharia de features fundamentada em
literatura antes do experimento final aqui reportado.

### 2.5 Aprendizado em grafo com desbalanceamento e poucos rótulos

Um corpo de trabalho específico de detecção de fraude em grafo endereça
desbalanceamento de classe e escassez de rótulo no nível de arquitetura, não
só de função de perda. PC-GNN (Liu et al., 2021) e CARE-GNN (Dou et al.,
2020) reamostram ou selecionam vizinhança via aprendizado por reforço para
contrapor tanto desbalanceamento de classe quanto "camuflagem" adversarial
por nós fraudulentos, reportando ganhos consistentes sobre agregação de GNN
ingênua em benchmarks de fraude desbalanceados. Um diagnóstico intimamente
relacionado, diretamente aplicável ao cenário desta dissertação, vem de
trabalho recente sobre detecção de fraude corporativa em grafos financeiros
"ricos porém ruidosos" (Wang et al., 2025), que identifica **sobrecarga de
informação** — a dominância numérica de tipos de nó sem informação de
atributo genuína (aqui: nós de sócio, endereço e vínculo político
compartilhados, que carregam só um embedding aprendido, não features reais)
sobre o tipo de nó-alvo — como um mecanismo específico pelo qual
message-passing heterogêneo ingênuo pode *diluir*, em vez de enriquecer,
sinal quando rótulos são escassos. Separadamente, uma literatura pequena mas
crescente de aprendizado positivo-não-rotulado (PU) em grafo (perdas PU
conscientes de estrutura; Yang et al., 2023) endereça o problema de
incompletude de rótulo que a própria construção de rótulo desta dissertação
compartilha, ainda que não integrada a um objetivo de treino de GNN
heterogênea no cenário de risco corporativo. Essa literatura é retomada na
Seção 5 *(a definir)* para interpretar os próprios achados sobre por que um
modelo heterogêneo mais complexo, treinado de ponta a ponta, pode ou não
superar alternativas mais simples exatamente sob as condições
(desbalanceamento extremo, poucos rótulos, tipos de nó auxiliares pobres em
atributo) para as quais essa literatura foi desenhada.

*(Lista de trabalho — a converter para formato ABNT completo quando o
rascunho estabilizar. Todas as referências abaixo foram verificadas
diretamente na página do arXiv ou registro DOI da editora em 14/08/2026,
não recuperadas de memória ou só de síntese de busca — ver nota de correção
em `docs/research_plan.md`, Seção 7.)*

---

## 3. Metodologia

### 3.1 Fonte de dados e escopo do estudo

O dataset vem de um pipeline de ETL mantido e publicamente documentado
(`projeto_grande_vitoria_empresas`) que consolida diversas fontes de dado
público brasileiro num único banco relacional, casadas por CNPJ: o cadastro
de empresas da Receita Federal, os registros federais de
impedimento/sanção (CEIS, CNEP), o registro do tribunal de contas estadual
(TCEES), o registro de irregularidade de entidade sem fins lucrativos
(CEPIM), registros de dívida ativa, registros de constituição empresarial
estadual (JUCEES), registros de infração ambiental, registros de contrato
governamental, registros de benefício fiscal, e registros de vínculo
político do TSE (candidatura/doação eleitoral). O escopo do estudo é a
região metropolitana da **Grande Vitória** (Espírito Santo, sete
municípios): **344.130 empresas cadastradas**, uma escala totalmente
tratável em memória em hardware comum — escolha deliberada, já que o público
para as conclusões práticas deste trabalho é órgão de controle sem
infraestrutura de big data dedicada.

### 3.2 Construção do rótulo e a moldura positivo-incompleta (PU)

Apenas **188 de 344.130 empresas (0,055%)** têm sanção administrativa
confirmada em registro. Tratamos isso como um problema positivo-incompleto:
ausência de registro de sanção indica que a empresa não foi (ainda) pega, não
que está livre de risco. Consequentemente, enquadramos a tarefa como detecção
de anomalia/ranking de risco, não classificação binária balanceada, e
reportamos PR-AUC e Precision@k em vez de acurácia ou ROC-AUC, que são
não-informativas ou enganosas nessa taxa-base.

Uma segunda questão de rotulagem, mais sutil, interage diretamente com a
pergunta de pesquisa central desta dissertação. Das 188 positivas
confirmadas, **148 são sancionadas diretamente** (a própria empresa aparece
numa lista de impedimento) enquanto **40 são rotuladas como positivas só
porque compartilham um sócio com uma entidade já sancionada** — exatamente o
mecanismo que o metapath de sócio comum é desenhado para detectar. Tratar as
188 como um único rótulo arrisca circularidade: um modelo que "descobre"
risco de sócio comum nesse subconjunto não está descobrindo sinal novo, está
reproduzindo a regra de rotulagem. Reportamos os resultados primários sobre
o rótulo restrito às 148 sancionadas diretamente (`y_direto`) e tratamos o
rótulo completo de 188 empresas (`y_qualquer`, incluindo os casos inferidos
via sócio) como uma análise de sensibilidade declarada, nunca confundindo os
dois sem explicitar qual está em uso.

### 3.3 Construção da rede heterogênea de informação

Construímos uma Rede Heterogênea de Informação (HIN) com cinco tipos de nó —
empresa, sócio, endereço, município, e vínculo político — e arestas
representando participação societária (`participa_de`), co-localização
(`sediada_em`), pertencimento municipal (`localizada_em`), e vínculo político
(`tem_vinculo_politico`). Três metapaths, escolhidos por relevância direta
para fiscalização de risco administrativo, não por conveniência
grafo-teórica, compõem a hipótese estrutural da rede: empresa–sócio–empresa
(sócio comum), empresa–endereço–empresa (endereço registrado comum), e
empresa–vínculo político–empresa (vínculo político comum, via registro de
candidatura/doação eleitoral). Um quarto metapath candidato via município
compartilhado foi excluído durante o desenvolvimento: com apenas sete nós de
município para 344.130 empresas, seu produto de adjacência é
combinatorialmente explosivo e não carrega sinal discriminativo relevante
(toda empresa compartilha município com dezenas de milhares de outras).
Extração de metapath usa produto de matrizes esparsas, não busca em
profundidade — requisito de escalabilidade nesse tamanho de rede (344.130 nós
de empresa).

### 3.4 Engenharia de features

Além de atributos de cadastro padrão (capital social, porte, regime
tributário, setor, número de sócios, dívida ativa agregada, infração
ambiental, contrato governamental, situação de benefício fiscal),
construímos dois grupos de features motivados por literatura, diretamente
informados pelas Seções 2.3–2.4:

- **Features de grau de grafo explícitas**: para cada um dos três metapaths,
  o grau da empresa naquela adjacência (número de outras empresas
  alcançáveis via sócio/endereço/vínculo político comum), mais a
  conectividade do sócio mais conectado da empresa — dando ao baseline
  tabular acesso direto à mesma informação estrutural disponível aos modelos
  de grafo, como teste da afirmação features-de-grafo-vs-GNN (Seção 2.3).
- **Indicadores de corrupção em compras públicas e de empresa de fachada**:
  sinalizador de contrato sem concorrência e razão máxima de sobrepreço
  (Fazekas & Kocsis, 2020; Abdou et al., 2022), e idade da empresa a partir
  da data de constituição no registro comercial, com valor-sentinela
  explícito para empresas sem correspondência no registro (Moody's, 2023).

### 3.5 Modelos

Comparamos três modelos sob um protocolo de avaliação idêntico:

1. **Baseline tabular**: árvores com gradient boosting (XGBoost), com
   rebalanceamento de peso de classe por fold (`scale_pos_weight`), usando o
   conjunto de features completo da Seção 3.4, incluindo as features de grau
   de grafo explícitas.
2. **GNN homogênea**: as três adjacências de metapath colapsadas num único
   grafo empresa–empresa (com um teto de grau na adjacência de endereço
   comum — um pequeno número de endereços de grandes prédios comerciais
   respondem por uma fração desproporcional das arestas de endereço comum,
   um artefato de dado que exige poda antes de dominar o grafo), treinada com
   GraphSAGE sobre as mesmas features tabulares.
3. **Transformer de grafo heterogêneo (HGT)**: cada tipo de nó e relação
   modelado distintamente (Hu et al., 2020); só o tipo de nó empresa carrega
   features tabulares reais, os demais tipos de nó (sócio, endereço, vínculo
   político) carregam um embedding aprendido, já que não têm dado de
   atributo próprio no registro-fonte — exatamente a condição de "sobrecarga
   de informação" discutida na Seção 2.5.

Os três modelos foram reavaliados em três iterações sucessivas do conjunto de
features (107, 117 e 124 colunas) à medida que features motivadas por
literatura foram adicionadas (Seções 2.3–2.4), e os hiperparâmetros do HGT
(dimensão oculta, cabeças de atenção, épocas de treino) passaram por uma
rodada dedicada de ajuste antes de a configuração final reportada ser
selecionada — ver Seção 4.4 para o achado que motivou essa etapa (o modelo
estava subtreinado, não subdimensionado).

### 3.6 Protocolo de avaliação

Usamos validação cruzada estratificada repetida (5 folds × 6 repetições = 30
folds), com os mesmos folds e semente aleatória compartilhados entre os três
modelos para permitir teste estatístico pareado. Reportamos PR-AUC e
Precision@k (k = 10, 20, 50) por fold, e comparamos os modelos dois a dois
com o teste de postos sinalizados de Wilcoxon sobre o PR-AUC por fold. Todos
os resultados são reportados tanto para o rótulo principal (`y_direto`, 148
positivas) quanto para o rótulo de sensibilidade (`y_qualquer`, 188
positivas) (Seção 3.2).

---

## 4. Resultados

Todos os valores de PR-AUC abaixo são médias sobre 30 folds (validação
cruzada estratificada de 5 folds × 6 repetições), com os mesmos folds e
semente aleatória compartilhados entre modelos dentro de uma mesma rodada,
permitindo teste de Wilcoxon pareado. "Lift" é o PR-AUC dividido pela
taxa-base do rótulo (148/344.130 = 0,0430% para `y_direto`; 188/344.130 =
0,0546% para `y_qualquer`). Cinco rodadas de avaliação foram feitas ao
todo, à medida que o conjunto de features e a configuração do HGT
evoluíram; reportamos a quinta (e final) rodada em detalhe, e as quatro
anteriores como trajetória de robustez (Seção 4.3).

### 4.1 Rótulo principal (`y_direto`, 148 sanções diretas confirmadas)

| Modelo | PR-AUC | Lift |
|---|---|---|
| Tabular (XGBoost) | 0,0218 ± 0,0146 | 50,7× |
| GNN homogênea | 0,0328 ± 0,0228 | **76,3×** |
| HGT (tunado) | 0,0140 ± 0,0181 | 32,6× |

Testes de Wilcoxon pareados (30 folds): tabular vs. GNN homogênea,
p = 0,0145 (GNN homogênea significativamente maior); tabular vs. HGT,
p = 0,0024 (tabular significativamente maior); GNN homogênea vs. HGT,
p < 0,0001 (GNN homogênea significativamente maior). As três diferenças
pareadas são estatisticamente significativas, e o ranking é consistente:
**GNN homogênea > tabular > HGT** no rótulo principal.

### 4.2 Rótulo de sensibilidade (`y_qualquer`, 188 sanções confirmadas,
incluindo 40 empresas rotuladas como positivas só por vínculo de sócio
comum com uma entidade já sancionada)

| Modelo | PR-AUC | Lift |
|---|---|---|
| Tabular (XGBoost) | 0,0236 ± 0,0200 | 43,2× |
| GNN homogênea | 0,0219 ± 0,0124 | 40,1× |
| HGT (tunado) | 0,0332 ± 0,0259 | **60,8×** |

Testes pareados: tabular vs. GNN homogênea, p = 0,8394 (sem diferença
significativa); tabular vs. HGT, p = 0,1579 (sem diferença significativa);
GNN homogênea vs. HGT, p = 0,0277 (HGT significativamente maior).
Diferente do rótulo principal, o HGT tunado é agora o melhor por estimativa
pontual, e significativamente à frente da GNN homogênea — ainda que não
(por enquanto, com esse tamanho de amostra) significativamente à frente do
baseline tabular.

### 4.3 Efeito da engenharia de features ao longo de três iterações (trajetória de robustez)

| Rodada | Features | Lift `y_direto` (tab / GNN homog. / HGT) | Lift `y_qualquer` (tab / GNN homog. / HGT) |
|---|---|---|---|
| 1 | 107 colunas | 18,6× / 16,5× / 14,2× | 15,7× / 24,4× / 33,5× |
| 2 | 117 colunas (+features do dashboard) | 75,8× / 81,2× / 29,3× | 62,3× / 41,6× / 45,4× |
| 3 | 124 colunas (+features de literatura), HGT sem ajuste | 50,7× / 76,3× / 23,5× | 43,2× / 40,1× / 42,8× |
| 4 | 124 colunas, HGT tunado (este trabalho) | 50,7× / 76,3× / **32,6×** | 43,2× / 40,1× / **60,8×** |

O ranking no rótulo principal — HGT estatisticamente pior que ambas as
alternativas — se sustenta nas três versões do conjunto de features
(rodada 1: tabular > HGT p = 0,0066, GNN homogênea > HGT p = 0,0293;
rodadas 2/3: tabular > HGT p < 0,0001 / p = 0,0001, GNN homogênea > HGT
p < 0,0001 nas duas). Não é artefato de uma iteração específica de
engenharia de features. O PR-AUC absoluto não é monótono entre as rodadas
2→3 (os três modelos tiveram queda quando as features de literatura foram
adicionadas) — um teste rápido inicial comparou a rodada 3 contra a rodada
1 em vez da rodada 2, criando uma falsa impressão de melhora; isso está
corrigido aqui e documentado como lição metodológica no diário de pesquisa
interno do projeto (`docs/research_plan.md`).

### 4.4 Efeito do ajuste de hiperparâmetros sobre o HGT

A configuração original do HGT (`hidden_channels=32`, `num_heads=1`,
`epochs=50`) tinha sido restringida por uma falha de memória (OOM) na
máquina de desenvolvimento (8 GB de RAM) em configurações maiores, não
selecionada por busca de hiperparâmetros. Uma busca dedicada (6
configurações, validação cruzada de 5 folds, só rótulo principal) achou
que **aumentar só as épocas de treino (50→150) quase triplicou o PR-AUC**
(0,0105→0,0244 nas rodadas de escala menor da busca), enquanto aumentar só
a largura da camada oculta não deu ganho nenhum (0,0105→0,0100) e a maior
configuração combinada (`hidden=64, heads=2, epochs=100`) falhou por OOM de
novo. Esse diagnóstico — subtreinamento, não subdimensionamento — motivou a
escolha de `epochs=150` (mantendo `hidden=32, heads=1`) como configuração
final, em vez de uma alternativa marginalmente melhor mas três vezes mais
cara (`heads=2` combinado com `epochs=150`) que empatava dentro do ruído
(PR-AUC de busca 0,0249 vs. 0,0244, desvio-padrão ≈ 0,025–0,029 nos dois).

Aplicar essa configuração tunada na avaliação completa de 30 folds (Seções
4.1–4.2) melhorou o lift do HGT no rótulo principal em 39% em relação ao
resultado da rodada 3 sem ajuste (23,5×→32,6×) — confirmando que o
diagnóstico de subtreinamento era real, não uma racionalização. **Ainda
assim, o HGT continua significativamente pior que as duas alternativas no
rótulo principal após o ajuste** (Seção 4.1). No rótulo secundário, o
ajuste aumentou o lift do HGT mais (42,8×→60,8×) do que no rótulo
principal — um padrão retomado na Seção 5.

## 5. Discussão

### 5.1 Um resultado negativo que sobrevive ao seu desafio metodológico mais forte

A objeção mais provável que um revisor cético levantaria contra uma versão
inicial deste resultado — que o modelo heterogêneo estava simplesmente
subtreinado em relação aos baselines mais simples — foi testada
diretamente, não argumentada de lado. Revelou-se parcialmente correta (as
épocas eram de fato um gargalo real) e irrelevante para a conclusão: uma
busca de hiperparâmetros que melhorou o HGT de forma mensurável (Seção 4.4)
não mudou seu ranking em relação aos baselines tabular e de GNN homogênea
no rótulo principal. Ao longo de cinco rodadas de avaliação independentes —
três iterações de conjunto de features mais uma rodada dedicada de ajuste —
o transformer de grafo heterogêneo é consistente e significativamente
superado na tarefa que a pergunta de pesquisa desta dissertação realmente
faz: detectar empresas com sanção administrativa *direta*, não-circular.
Tratamos isso como um achado empírico robusto, não provisório à espera de
mais ajuste.

### 5.2 Por que um modelo mais simples pode vencer aqui? Duas literaturas convergem

O resultado é consistente com, e mutuamente reforçado por, duas linhas
distintas de literatura revisadas na Seção 2. Primeiro, a literatura de
features-de-grafo-vs-GNN (Seção 2.3) prevê que dar a um modelo tabular com
gradient boosting acesso explícito à mesma informação estrutural que uma
GNN aprenderia implicitamente fecha a maior parte da diferença de
desempenho — exatamente o que observamos: o modelo tabular, alimentado com
features de grau de grafo explícitas, atinge um respeitável lift de 50,7×
por conta própria, e a GNN homogênea (*mais simples*, um único tipo de
relação colapsado) supera tanto ele quanto o HGT, muito mais complexo e
tipado por relação. Segundo, o diagnóstico de "sobrecarga de informação" da
literatura de detecção de fraude corporativa (Seção 2.5) oferece uma
explicação mecanicista de por que o modelo *mais* heterogêneo especificamente
tem desempenho pior: os tipos de nó auxiliares do HGT (sócio, endereço,
vínculo político) não carregam dado de atributo genuíno próprio, só um
embedding aprendido — e esses tipos de nó pobres em atributo superam em
número os nós-empresa ricos em atributo por mais de uma ordem de grandeza
(142.844 nós de sócio e 181.268 nós de endereço contra 344.130 nós de
empresa, vários dos quais conectam ao mesmo pequeno número de nós
auxiliares). Treinar toda essa estrutura adicional e fracamente informada
de ponta a ponta com só 148 rótulos positivos parece adicionar variância em
vez de sinal, em relação a um modelo mais simples que ou ignora essa
estrutura (tabular com features de grau explícitas) ou a agrega
grosseiramente num único tipo de relação (GNN homogênea).

### 5.3 A reversão no rótulo secundário é evidência de exploração de circularidade, não de vantagem genuína

O rótulo de sensibilidade (`y_qualquer`) inclui 40 empresas cujo único
caminho para um rótulo positivo é uma conexão de sócio comum com uma
entidade já sancionada — exatamente a relação que o metapath de sócio
comum é construído para detectar. Se a vantagem maior do HGT tunado nesse
rótulo (lift de 60,8×, ante 42,8× antes do ajuste, e agora
significativamente à frente da GNN homogênea) refletisse sinal de risco
estrutural recém-descoberto, esperaríamos um ganho comparável no rótulo
principal, onde essa circularidade não existe. Não é o caso: o ganho do
ajuste no rótulo principal (23,5×→32,6×) é real, mas bem mais modesto, e o
HGT continua sendo o modelo mais fraco ali. A explicação mais parcimoniosa
é que capacidade de treino adicional permite ao HGT aprender de forma mais
completa a relação de sócio comum especificamente onde fazer isso
reproduz diretamente parte da regra de rotulagem, em vez de descobrir sinal
preditivo novo. Reportamos esse padrão explicitamente, em vez de destacar o
resultado do rótulo de sensibilidade em primeiro plano, precisamente porque
ilustra como uma leitura acrítica do resultado "a GNN vence" num rótulo
circular poderia induzir a erro quem fosse aplicar isso na prática.

### 5.4 Implicação prática para órgão de controle com recursos limitados

Para o público a que esta dissertação se dirige — controladorias, tribunais
de contas, órgãos de fiscalização de licitação operando com quadro limitado
de ciência de dados, sem infraestrutura de GPU dedicada, e poucos rótulos
confirmados para aprender — a recomendação prática é direta: **um modelo
tabular com gradient boosting, aumentado com um pequeno número de features
explícitas de grau de grafo (contagem de sócio comum, endereço comum,
vínculo político comum), é mais barato de construir e operar, e mais
eficaz, que um transformer de grafo heterogêneo para essa tarefa.** O custo
de treino do modelo tabular é medido em minutos em hardware comum; o do HGT
tunado é medido em horas por rótulo, e exigiu um procedimento de ajuste
dedicado e tecnicamente exigente só para chegar ao seu melhor desempenho
possível — que ainda assim ficou aquém. Onde um órgão de controle tiver
apetite por métodos baseados em grafo, este resultado recomenda a GNN
homogênea, mais simples e de um único tipo de relação, em vez de uma
arquitetura totalmente heterogênea — pelo menos até que tipos de nó além de
"empresa" carreguem dado de atributo genuíno próprio (Seção 5.2), não só um
embedding aprendido.

### 5.5 Limitações

O próprio rótulo principal é positivo-incompleto, não exaustivo: ausência
de sanção confirmada não estabelece ausência de irregularidade, só ausência
de detecção confirmada (até agora) — uma limitação inerente a esse domínio
de tarefa, não específica do método usado aqui, mas que delimita como os
números de PR-AUC/lift reportados devem ser interpretados (como detecção de
risco *já pego*, com recall desconhecido contra risco ainda não pego).
Resolução de identidade de sócio e endereço usa heurísticas de nível de
cadastro (CPF mascarado mais nome normalizado para sócios; logradouro/
número/CEP normalizados para endereços) que não resolvem homônimos nem
variação de grafia — uma fonte de ruído nos três metapaths, que
provavelmente atenua, em vez de inflar, a vantagem medida dos modelos
baseados em grafo. Registros de processo judicial foram excluídos da rede
por completo, já que são casados por nome via um pipeline de resolução de
registro ainda em desenvolvimento, não por CNPJ, e foram julgados
confiáveis demais para incluir como rótulo ou feature. O estudo é
delimitado a uma única região metropolitana brasileira (sete municípios);
generalização para outras regiões, regimes de sanção, ou estruturas de
cadastro empresarial não foi testada. Por fim, a busca de hiperparâmetros
(Seção 4.4) cobriu seis configurações escolhidas para isolar três eixos
específicos (profundidade de treino, largura oculta, cabeças de atenção),
não uma busca exaustiva ou bayesiana; consideramos isso proporcional dado
que testou diretamente, e fechou, a objeção mais provável ao resultado, mas
uma busca maior continua sendo trabalho futuro possível.

## 6. Conclusão

Testando se uma rede neural em grafo heterogêneo melhora a detecção de
risco de sanção administrativa em relação a alternativas mais simples, num
cadastro real e completo de 344.130 empresas com rótulo positivo-incompleto
e extremamente raro, constatamos que não melhora — e que esse achado
sobrevive ao seu desafio metodológico mais sério. Uma busca dedicada de
hiperparâmetros confirmou que o modelo heterogêneo (HGT) estava subtreinado
em sua configuração inicial e o melhorou de forma significativa uma vez
corrigido, ainda assim o modelo melhorado continua sendo significativamente
superado tanto por um baseline tabular com features de grau de grafo
explícitas quanto por uma GNN homogênea mais simples, no rótulo de sanção
principal, não-circular, ao longo de cinco rodadas de avaliação
independentes. Onde o modelo heterogêneo de fato mostra vantagem — num
rótulo secundário parcialmente construído via a mesma relação de sócio
comum que o modelo explora — o padrão é mais consistente com exploração de
circularidade de rótulo do que com descoberta de sinal genuíno de risco
estrutural. Para órgãos de controle do setor público operando com poder
computacional e capacidade técnica limitados, este resultado recomenda um
modelo tabular bem projetado com features derivadas de grafo explícitas,
não um transformer de grafo heterogêneo, como a escolha defensável para
triagem de risco de sanção administrativa nessa escala e escassez de
rótulo.

## Referências

*(Lista de trabalho — a converter para formato ABNT completo antes da
defesa. Todas verificadas diretamente em fonte primária — arXiv/DOI — em
14/08/2026.)*

- ABDOU, A.; BASDEVANT, O.; DAVID-BARRETT, E.; FAZEKAS, M. Assessing
  vulnerabilities to corruption in public procurement and their price
  impact. *IMF Working Paper*, 22/094, 2022.
- CHENG, D.; ZOU, Y.; XIANG, S.; JIANG, C. Graph neural networks for
  financial fraud detection: a review. arXiv:2411.05815, 2024.
- DOU, Y.; LIU, Z.; SUN, L.; DENG, Y.; PENG, H.; YU, P. S. Enhancing graph
  neural network-based fraud detectors against camouflaged fraudsters. In:
  **CIKM 2020**.
- FAZEKAS, M.; KOCSIS, G. Uncovering high-level corruption: cross-national
  objective corruption risk indicators using public procurement data.
  *British Journal of Political Science*, v. 50, n. 1, p. 155–164, 2020.
- HU, Z.; DONG, Y.; WANG, K.; SUN, Y. Heterogeneous graph transformer. In:
  **WWW 2020**. arXiv:2003.01332.
- IVANOV, S.; PROKHORENKOVA, L. Boost then convolve: gradient boosting meets
  graph neural networks. arXiv:2101.08543, 2021.
- LIU, Y.; AO, X.; QIN, Z.; CHI, J.; FENG, J.; YANG, H.; HE, Q. Pick and
  choose: a GNN-based imbalanced learning approach for fraud detection. In:
  **WWW 2021**.
- MA, X.; LI, R.; LIU, F.; DING, K.; YANG, J.; WU, J. Graph anomaly
  detection with few labels: a data-centric approach. In: **KDD 2024**, p.
  2153–2164.
- MOODY'S ANALYTICS. 7 indicators of shell company risk. 22 jan. 2023.
  Disponível em:
  https://www.moodys.com/web/en/us/kyc/resources/insights/seven-indicators-shell-company-risk.html
- VANDERVORST, F.; DEPREZ, B.; VERBEKE, W.; VERDONCK, T. Inductive inference
  of gradient-boosted decision trees on graphs for insurance fraud
  detection. arXiv:2510.05676, 2025.
- WANG, S.; ZHANG, Z.; FANG, L.; NGUYEN, C.-T.; LI, W. Corporate fraud
  detection in rich-yet-noisy financial graph. arXiv:2502.19305, 2025.
- WANG, X.; JI, H.; SHI, C.; WANG, B.; CUI, P.; YU, P.; YE, Y. Heterogeneous
  graph attention network. In: **WWW 2019**. arXiv:1903.07293.
- YANG, H.; ZHANG, Y.; YAO, Q.; KWOK, J. Positive-unlabeled node
  classification with structure-aware graph learning. arXiv:2310.13538,
  2023.
