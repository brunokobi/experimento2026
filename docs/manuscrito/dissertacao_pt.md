# Título provisório

**Modelos Simples, Sinal de Grafo Explícito: Repensando Redes Neurais em
Grafos Heterogêneos para Triagem de Risco de Sanção Administrativa em Dados
de Governo Local**

*(provisório — não há template ABNT obrigatório da instituição informado até
o momento; estrutura livre de dissertação de mestrado, capítulos de
conteúdo. Capa/folha de rosto/ficha catalográfica/resumo formal ficam para
quando o formato institucional for confirmado.)*

> **Status deste rascunho**: Introdução, Referencial Teórico e Metodologia
> são rascunhos substantivos. Resultados, Discussão e Conclusão são
> placeholders — a preencher quando o experimento final de 30 folds
> (`docs/resultados/`) terminar. Todos os números e citações abaixo foram
> reconferidos direto em `docs/research_plan.md` e em fonte primária
> (arXiv/DOI), não recuperados de memória — ver nota de rigor em
> `docs/research_plan.md`, correção de 14/08/2026.

---

## Resumo

*(placeholder — escrever depois dos números finais)*

## Abstract

*(placeholder — versão em inglês do resumo, depois dos números finais)*

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
selecionada — ver Seção 4 *(a definir)* para o achado que motivou essa etapa.

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

*(A definir — pendente do experimento final de 30 folds com o HGT ajustado.
Estrutura planejada: 4.1 resultados do rótulo principal e testes pareados;
4.2 resultados do rótulo de sensibilidade; 4.3 efeito da engenharia de
features motivada por literatura ao longo das três iterações; 4.4 efeito do
ajuste de hiperparâmetros sobre o HGT.)*

## 5. Discussão

*(A definir. Ângulos planejados: (a) interpretação via as literaturas de
features-de-grafo-vs-GNN e sobrecarga de informação da Seção 2; (b)
implicação prática para órgão de controle com recursos limitados — o que um
tribunal de contas ou controladoria deveria de fato usar, dado o quadro de
pessoal, poder computacional e orçamento de rotulagem típicos do setor
público, não um cenário de pesquisa idealizado; (c) limitações — moldura de
rótulo incompleto/PU, heurísticas de resolução de identidade de
sócio/endereço, dado de processo judicial excluído por não-confiabilidade,
escopo de uma única região.)*

## 6. Conclusão

*(A definir.)*

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
