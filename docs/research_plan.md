# Plano de Pesquisa — Dissertação de Mestrado

> **Objetivo do documento**: registrar a pergunta de pesquisa, o escopo e as decisões
> metodológicas da dissertação, orientadas para produzir um artigo publicável em
> veículo de classificação (Qualis/CAPES) alta. Este documento é vivo — deve ser
> atualizado a cada decisão relevante tomada.
>
> **Nota sobre orientação**: pesquisa conduzida sem orientador formal — as decisões
> registradas aqui são de responsabilidade do próprio pesquisador (Bruno Kobi),
> com apoio técnico de IA. Onde antes se lia "validar com o orientador", leia-se
> "decisão travada pelo pesquisador".

**Status**: tarefa-fim decidida (seção 4) — foco atual: validar viabilidade real do
rótulo (CEIS/CNEP/TCU) antes de travar o schema completo da HIN.

---

## 1. Motivação e lacuna na literatura

Modelos tabulares (ex.: XGBoost sobre dados cadastrais da Receita Federal) e GNNs
homogêneas ignoram um sinal estrutural importante em fraude societária: padrões de
**sócio, endereço e contador compartilhados** entre empresas. Esse sinal é exatamente
o que um metapath em uma Rede Heterogênea de Informação (HIN) captura.

A literatura brasileira de detecção de fraude corporativa é majoritariamente tabular.
Trabalhos internacionais com HAN/HGT/R-GCN em grafos corporativos existem, mas
concentrados em previsão de insolvência/risco de crédito — um espaço já saturado.
**Detecção de fraude societária via metapaths, no contexto de dados públicos
brasileiros, é pouco explorada** — esse é o gap que esta dissertação ataca.

## 2. Pergunta de pesquisa

> Metapaths estruturais em uma HIN de empresas melhoram a detecção de indícios de
> fraude societária em relação a baselines tabulares e a GNNs homogêneas — e quais
> metapaths carregam mais sinal?

**Título provisório**: *Detecção de Indícios de Fraude Societária em Redes
Heterogêneas de Empresas Brasileiras via Metapaths e Graph Neural Networks*.

## 3. Contribuições reivindicadas

1. Uma HIN de empresas brasileiras com schema e metapaths desenhados especificamente
   para sinais de fraude societária (sócio comum, endereço comum, contador comum).
2. Comparação empírica rigorosa entre HAN/HGT, GNN homogênea, metapath2vec e baseline
   tabular (XGBoost/LightGBM), com ablation por metapath.
3. Análise interpretável de quais metapaths carregam mais sinal — não apenas "o
   modelo ganhou", mas "ganhou por causa de X".

## 4. Tarefa-fim: opções avaliadas

| Tarefa | Novidade | Risco de rótulo | Concorrência na literatura |
|---|---|---|---|
| **Fraude societária (empresa "de fachada")** — via metapaths de sócio/endereço/contador, rótulo fraco de CEIS/CNEP/TCU | Alta | Médio | Baixa |
| Grupo econômico oculto / beneficiário final (classificação de pares de empresas) | Alta | Alto (poucos rótulos públicos) | Muito baixa |
| Previsão de insolvência/falência | Baixa-média | Alto (dados dispersos) | **Alta** |
| Risco de crédito de PME | Baixa | Médio | Muito alta |

**Recomendação**: a opção de fraude societária — maior aproveitamento do que uma HIN
oferece que um modelo tabular não oferece, rótulo público real disponível (CEIS/CNEP),
e gap de literatura genuíno.

**Decisão final**: ✅ fechada — tarefa-fim é **detecção de fraude societária** (empresa
"de fachada" via metapaths de sócio/endereço/contador compartilhados, rótulo fraco de
CEIS/CNEP/TCU). Esta decisão trava o schema de nós/arcos da HIN e a lista de metapaths
das próximas seções.

## 5. Metodologia

- **Dados**: Receita Federal (empresas, sócios, CNAE) + rótulo fraco de proxy de
  fraude via CEIS/CNEP/TCU ("empresas inidôneas e suspensas").
- **Split temporal** (não aleatório) — evita vazamento de informação futura; é o
  primeiro ponto que um revisor de veículo de peso ataca em dados corporativos.
- **Extração de metapath via matriz esparsa** (produto de matrizes de adjacência),
  não busca em profundidade (DFS) — DFS não escala para o volume real de dados e
  reviewer de sistemas percebe essa limitação.
- **Baselines**: tabular (XGBoost/LightGBM), GNN homogênea, metapath2vec, HAN/HGT.
- **Rigor estatístico**: múltiplas seeds + teste estatístico (ex.: Wilcoxon) nas
  comparações entre modelos — sem isso a afirmação "nosso modelo é melhor" não
  sobrevive à revisão.

## 6. Ética e LGPD

- Sócios (pessoas físicas) exigem pseudonimização (hash de CPF) antes de qualquer
  dado tocar `data/raw/` — ver `.gitignore` e `src/config/settings.py`.
- Justificativa de uso de dado público a documentar formalmente.
- Verificar exigência de parecer de comitê de ética da instituição.

## 7. Riscos declarados

- **Rótulo de fraude é fraco/ruidoso** (proxy, não verdade absoluta) — mitigação:
  validação manual de uma amostra dos rótulos.
- **Escala do grafo real vs. tempo de mestrado** — mitigação: amostragem
  estratificada por UF/setor nos experimentos intermediários; grafo completo apenas
  no capítulo final.
- **Risco de novidade "comida"** por publicação concorrente — mitigação: publicar um
  resultado parcial em workshop/preprint antes do artigo principal.

## 8. Venues-alvo (realistas para o prazo de mestrado)

- **Rede de segurança nacional**: BRACIS, SBBD.
- **Alvo principal**: periódicos de bom impacto e ciclo de revisão compatível com o
  prazo (ex.: *Expert Systems with Applications*, *Knowledge-Based Systems*,
  *Decision Support Systems*).
- Evitar KDD/WWW/CIKM como alvo primário — ciclo e barreira de aceitação
  incompatíveis com o tempo disponível; considerar apenas como alvo secundário
  (workshop) se houver resultado forte e tempo de sobra.

## 9. Cronograma orientado a publicação

Os marcos seguem o ciclo de publicação, não fases de engenharia — os capítulos da
dissertação espelham estes marcos, não o inverso:

1. **Marco 1** — HIN + metapaths validados em amostra pequena, com resultado
   preliminar.
2. **Marco 2** — submissão de resultado parcial em workshop/BRACIS.
3. **Marco 3** — experimento completo com todos os baselines e ablation por metapath.
4. **Marco 4** — submissão do artigo principal ao periódico-alvo.

## 10. Próximo passo imediato

Com a tarefa-fim travada (seção 4), o próximo passo é **validar a viabilidade real do
rótulo** antes de investir no schema completo da HIN e na coleta de dados em escala:

- Checar formato, cobertura temporal e volume de empresas listadas em CEIS/CNEP/TCU.
- Confirmar que os registros trazem CNPJ (ou informação suficiente para cruzar com a
  base de empresas da Receita Federal) — sem isso o rótulo é inutilizável.
- Estimar a proporção de exemplos positivos (empresas sancionadas) vs. universo total
  — se for extremamente raro, isso já define a estratégia de avaliação (ex.:
  PR-AUC em vez de acurácia, técnicas de desbalanceamento).

Se o rótulo se mostrar inviável nessa checagem, a tarefa-fim recua para a segunda
opção da tabela da seção 4 (grupo econômico oculto) ou para uma reavaliação da fonte
de rótulo — antes de qualquer código de schema ser escrito.

---

*Ver também o scaffold de código em `src/` (config, loaders, HIN builder, extração
de metapaths, testes) — construído para suportar esta pesquisa, mas ainda
independente da tarefa-fim final.*
