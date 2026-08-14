# Working title

**Simple Models, Explicit Graph Signals: Rethinking Heterogeneous Graph Neural
Networks for Administrative Sanction Risk Screening in Local Government Data**

*(provisional — revisit once Results/Discussion are finalized; target venue:
Government Information Quarterly)*

> **Status of this draft**: Introduction, Related Work and Data/Methods are
> substantive drafts, ready for revision. Results, Discussion and Conclusion
> are placeholders — to be written once the final 30-fold experiment
> (`docs/resultados/`) is complete. Word budget target for GIQ: ~8,000–10,000
> words; this draft will need trimming once all sections exist.

---

## Abstract

*(placeholder — draft after final numbers are in; ~200 words for GIQ)*

Oversight bodies in local and regional government are increasingly asked to
screen large corporate registries for administrative-sanction risk — evasion
of debarment lists, shell-company fronting, and undisclosed political
connections — with limited data-science budgets and no dedicated
infrastructure. Graph neural networks (GNNs) that model corporate networks
explicitly (shared partners, shared addresses, political ties) are often
proposed as the state-of-the-art answer, on the assumption that structural
risk is only visible to a relational model. We test this assumption on a
real, complete registry of 344,130 companies from a Brazilian metropolitan
region, comparing a tabular gradient-boosting baseline, a homogeneous GNN,
and a heterogeneous graph transformer (HGT), under a positive-unlabeled
framing appropriate to the extreme rarity of confirmed sanctions (148–188
positives). [RESULT SUMMARY — TBD]. We discuss the practical implication for
resource-constrained oversight bodies: [TBD].

**Keywords**: administrative sanction risk; heterogeneous graph neural
networks; anomaly detection; public sector data science; corporate network
analysis; positive-unlabeled learning.

---

## 1. Introduction

Public administrations at every level maintain registries of the companies
they interact with — as taxpayers, as bidders in public procurement, as
recipients of tax benefits, as employers. A small fraction of these companies
are eventually found to have violated administrative rules seriously enough
to be placed on a debarment or sanctions list (in Brazil, the federal *CEIS*
and *CNEP* registries, state-level equivalents, and sector-specific lists such
as the *CEPIM* for irregular non-profits). For oversight bodies — courts of
accounts, comptroller offices, procurement agencies — the practical question
is not retrospective ("was this company sanctioned?") but prospective: *among
the companies not yet sanctioned, which ones carry enough risk to justify
scrutiny now?*

Two structural facts make this a hard screening problem. First, confirmed
sanctions are extremely rare relative to the size of any real registry — in
the dataset used here, 188 confirmed cases out of 344,130 companies (0.055%).
Any model trained on this label is learning from a positive-unlabeled (PU)
signal: absence of a sanction record means *not yet caught*, not *not at
risk*. Second, corporate risk is frequently relational rather than intrinsic
to any single company's declared attributes: a company with an unremarkable
tax filing may share a partner, a registered address, or an undisclosed
political connection with an already-sanctioned entity — signal that is
invisible to a model that treats each company as an independent row of
features.

This second fact is the premise behind a growing body of work applying
heterogeneous graph neural networks (HGNNs) — models that treat companies,
partners, addresses and other entities as distinct node types connected by
typed relations — to corporate risk detection, largely in credit/insolvency
scoring and financial-fraud contexts (see Section 2). The implicit promise of
this literature is that relational, end-to-end-trained models will recover
structural risk signal that simpler, tabular approaches cannot.

We test this promise directly, in a setting closer to what a real,
resource-constrained oversight body would actually have available: a
complete, real corporate registry for a Brazilian metropolitan region (the
*Grande Vitória* area, Espírito Santo state, seven municipalities), built into
a heterogeneous information network (HIN) with three metapaths chosen for
their direct relevance to administrative-risk oversight — **shared partner**,
**shared registered address**, and **shared political connection** (via
federal electoral records) — and evaluated under the extreme class imbalance
and PU-label conditions that are the norm in this domain, not an edge case to
be tuned away.

We make four contributions relevant to practitioners and researchers working
on public-sector risk screening:

1. **A real, reproducible empirical comparison** of a tabular gradient-boosting
   baseline, a homogeneous GNN, and a heterogeneous graph transformer (HGT),
   under a shared, statistically rigorous evaluation protocol (repeated
   stratified cross-validation, paired Wilcoxon tests across 30 folds),
   replicated across three successive feature-engineering iterations and a
   dedicated hyperparameter-tuning pass — a level of methodological scrutiny
   uncommon in prior graph-based fraud/risk papers, which typically report a
   single configuration.
2. **[RESULT — TBD]**: whether, and under what conditions, the heterogeneous
   model's implicit structural learning outperforms giving the *same*
   structural signal explicitly, as engineered features, to the tabular
   baseline — directly probing a claim from the graph-features-vs-GNN
   literature (Section 2.3) in a new domain.
3. **A feature set for administrative-sanction screening grounded in
   procurement-corruption and shell-company literature** (Section 2.4),
   adapted to what is realistically available in a Brazilian municipal-level
   registry: non-competitive contract awards, cost overruns, company age, and
   partner concentration across many firms.
4. **A practical discussion, aimed at oversight-body practitioners rather than
   at the graph-learning research community**, of what modeling approach is
   *actually* justified given the computational budget, technical capacity,
   and label scarcity that real oversight bodies operate under — not what is
   justified in an idealized, well-resourced benchmark setting.

The remainder of the paper is organized as follows. Section 2 situates this
work within four literatures: administrative/corporate risk detection in
Brazilian public data, heterogeneous GNNs for corporate networks, the
graph-features-vs-GNN debate in fraud detection, and imbalanced/few-label
graph learning. Section 3 describes the data, label construction, network
design, feature engineering, models and evaluation protocol. Section 4
reports results [TBD]. Section 5 discusses the findings from a public-sector
practitioner's perspective [TBD]. Section 6 concludes [TBD].

---

## 2. Related work

### 2.1 Administrative and corporate sanction risk in Brazilian public data

Prior Brazilian work on corporate risk and fraud detection using public
registry data is predominantly tabular, treating each company as an
independent observation described by registration attributes, tax regime,
debt records, and sector codes. This body of work does not, to our knowledge,
model the *network* of relationships between companies — shared partners,
shared addresses, shared political ties — as a first-class object, despite
these relationships being directly queryable in the same public data sources
(Federal Revenue registry, electoral court records). International work
applying relational models (HAN, HGT, R-GCN) to corporate networks exists,
but concentrates heavily on credit/insolvency risk — a comparatively
data-rich, well-labeled problem — leaving administrative-sanction risk
detection via ownership/address/political metapaths, in a Brazilian public-data
context, comparatively unexplored.

### 2.2 Heterogeneous graph neural networks for corporate and financial networks

Heterogeneous graph attention/transformer architectures — HAN (Wang et al.,
2019) and HGT (Hu et al., 2020) foremost among them — extend graph neural
networks beyond a single node/edge type, learning type-specific attention
weights across metapaths or relations. They have been applied to corporate
ownership networks for shell-company and beneficial-ownership risk (Moody's,
2023) and to financial fraud detection more broadly, where fraud rings are
argued to leave a structural signature — shared infrastructure, shared
counterparties — that is "invisible in feature space but detectable in graph
topology." A parallel literature on organized-crime shell-company networks in
public procurement documents concretely how ownership/management data,
combined with contracting data, reveals connected components indicative of
collusion risk (see Section 2.4).

### 2.3 Graph features versus end-to-end GNN training

A distinct and directly relevant literature asks a narrower, more skeptical
question: does an end-to-end-trained GNN outperform a much simpler baseline
that is *given the same structural information explicitly*, as engineered
graph features (degree, centrality, PageRank, neighborhood aggregates) fed to
a gradient-boosted tree model? Evidence from fraud-detection benchmarks
suggests the answer is frequently "no, or not by much." A recent benchmark in
insurance fraud detection explicitly compares gradient-boosted trees against
HinSAGE, HAN and HGT, noting that "gradient-boosted tree approaches on
tabular data still dominate the field" and that graph-based approaches
specifically struggle under the high class imbalance typical of fraud data
(Vandervorst et al., 2025) — the same combination of conditions (heterogeneous
graph, extreme imbalance) studied in this paper. A separate line of work
combining gradient boosting with graph structure directly (rather than
comparing them as rivals) further shows that GBDT-based models can match or
exceed pure GNN architectures once graph information is made available to
them in a suitable form (Ivanov & Prokhorenkova, 2021). This paper's
design — feeding the *same* three metapaths to the tabular baseline as
explicit degree features, and to the heterogeneous model as learned
structure — is a direct test of this claim in a new domain.

### 2.4 Corruption-risk indicators in public procurement and shell-company detection

Cross-national empirical work on public-procurement corruption risk (Fazekas
& Kocsis, 2020; Abdou et al., 2022) establishes objective, non-competitive-award and
cost-overrun indicators — sole-source or waived-competition contract
modality, and divergence between initial and final contract value — as
robust, auditable proxies for corruption risk, independent of any
prosecutorial outcome. Shell-company detection practice (Moody's, 2023;
organized-crime shell-network literature in procurement) further identifies
company age and partner/director concentration across many firms as
practical red flags. These four indicators are directly relevant to, and
feasible to construct from, the registry data used in this study (Section
3.4), and motivated a dedicated round of literature-grounded feature
engineering prior to the final experiment reported here.

### 2.5 Imbalanced and few-label graph learning

A body of work specific to graph-based fraud detection addresses class
imbalance and label scarcity at the architecture level rather than only at
the loss-function level. PC-GNN (Liu et al., 2021) and CARE-GNN (Dou et al.,
2020) resample or reinforcement-learning-select neighborhoods to counteract
both class imbalance and adversarial "camouflage" by fraudulent nodes,
reporting consistent gains over naive GNN aggregation on imbalanced fraud
benchmarks. A closely related diagnosis, directly applicable to this paper's
setting, comes from recent work on corporate fraud detection in "rich-yet-noisy"
financial graphs (Wang et al., 2025), which identifies **information overload** — the
numerical dominance of node types with no genuine attribute information
(here: shared-partner, shared-address and political-connection nodes, which
carry only a learned embedding, not real features) over the target node type —
as a specific mechanism by which naive heterogeneous message-passing can
*dilute* rather than enrich signal when labels are scarce. Separately, a small
but growing literature on positive-unlabeled (PU) learning on graphs
(structure-aware PU losses; Yang et al., 2023) addresses the label-incompleteness
problem this paper's own label construction shares, though not yet integrated
into a heterogeneous-GNN training objective in the corporate-risk setting. We
draw on this literature in Section 5 [TBD] to interpret our own findings on
why a more complex, end-to-end heterogeneous model may or may not out-perform
simpler alternatives under the exact conditions (extreme imbalance, few
labels, attribute-poor auxiliary node types) this literature was designed to
address.

*(References to be compiled in full BibTeX/APA form once the draft
stabilizes — see inline citations above for the working list.)*

---

## 3. Data and methods

### 3.1 Data source and study scope

The dataset is drawn from a maintained, publicly documented ETL pipeline
(`projeto_grande_vitoria_empresas`) that consolidates several official
Brazilian public-data sources into a single relational database, matched by
company tax ID (CNPJ): the Federal Revenue company registry, the federal
debarment/sanctions registries (CEIS, CNEP), the state accounts court
registry (TCEES), the non-profit irregularity registry (CEPIM), active-debt
records, state commercial-registry filings (JUCEES), environmental-infraction
records, government-contract records, tax-benefit records, and federal
electoral court (TSE) political-connection records. The study scope is the
*Grande Vitória* metropolitan region (Espírito Santo state, Brazil, seven
municipalities): 344,130 registered companies, a scale that is fully tractable
in memory on commodity hardware — a deliberate choice, since the target
audience for this work's practical conclusions is oversight bodies without
dedicated big-data infrastructure.

### 3.2 Label construction and the positive-unlabeled framing

Only 188 of 344,130 companies (0.055%) have a confirmed administrative
sanction on record. We treat this as a positive-unlabeled problem: absence of
a sanction record indicates the company has not (yet) been caught, not that
it is risk-free. Consequently we frame the task as anomaly detection /
risk ranking rather than balanced binary classification, and report
PR-AUC and Precision@k rather than accuracy or ROC-AUC, which are
uninformative or misleading at this base rate.

A second, more subtle labeling issue directly interacts with this paper's
central research question. Of the 188 confirmed positives, 148 are sanctioned
directly (the company itself appears on a debarment list) while 40 are
labeled positive only because they share a partner with an already-sanctioned
entity — the exact mechanism the shared-partner metapath is designed to
detect. Treating all 188 as a single label risks circularity: a model that
"discovers" shared-partner risk on this subset is not discovering new signal,
it is reproducing the labeling rule. We report our primary results on the
148 directly-sanctioned label (`y_direto`) and treat the full 188-company
label (`y_qualquer`, including the partner-inferred cases) as a declared
sensitivity analysis, never conflating the two without stating which is in
use.

### 3.3 Heterogeneous information network construction

We construct a heterogeneous information network (HIN) with five node types
— company, partner, address, municipality, and political connection — and
edges representing partnership (`participa_de`), co-location (`sediada_em`),
municipal membership (`localizada_em`), and political connection
(`tem_vinculo_politico`). Three metapaths, chosen for direct relevance to
administrative-risk oversight rather than for graph-theoretic convenience,
compose the network's structural hypothesis: company–partner–company (shared
partner), company–address–company (shared registered address), and
company–political-connection–company (shared political connection, via
electoral candidacy/donation records). A fourth candidate metapath via shared
municipality was excluded during development: with only seven municipality
nodes for 344,130 companies, its adjacency product is combinatorially
explosive and carries no meaningful discriminative signal (every company
shares a municipality with tens of thousands of others). Metapath extraction
uses sparse matrix products rather than depth-first search, a scalability
requirement at this network size (344,130 company nodes).

### 3.4 Feature engineering

Beyond standard registration attributes (capital, size category, tax regime,
sector, partner count, aggregated active debt, environmental infractions,
government contracts, tax-benefit status), we construct two literature-motivated
feature groups directly informed by Sections 2.3–2.4:

- **Explicit graph-degree features**: for each of the three metapaths, the
  company's degree in that metapath's adjacency (number of other companies
  reachable via shared partner / address / political connection), plus the
  connectivity of the company's most-connected partner — giving the tabular
  baseline direct access to the same structural information available to the
  graph models, as a test of the graph-features-vs-GNN claim (Section 2.3).
- **Procurement-corruption and shell-company indicators**: non-competitive
  contract-award flag and maximum cost-overrun ratio (Fazekas & Kocsis, 2020;
  IMF, 2022), and company age from commercial-registry incorporation date,
  with an explicit sentinel value for companies without a registry match
  (Moody's, 2023).

### 3.5 Models

We compare three models under an identical evaluation protocol:

1. **Tabular baseline**: gradient-boosted trees (XGBoost), with per-fold
   class-weight rebalancing (`scale_pos_weight`), using the full feature set
   from Section 3.4 including the explicit graph-degree features.
2. **Homogeneous GNN**: the three metapath adjacencies collapsed into a
   single company–company graph (with a degree cap on the shared-address
   metapath — a small number of large commercial-building addresses account
   for a disproportionate share of shared-address edges, a data artifact
   requiring pruning before it dominates the graph), trained with GraphSAGE
   over the same tabular features.
3. **Heterogeneous graph transformer (HGT)**: each node and relation type
   modeled distinctly (Hu et al., 2020); only the company node type carries
   real tabular features, the remaining node types (partner, address,
   political connection) carry a learned embedding, as they have no attribute
   data of their own in the source registry — the exact "information overload"
   condition discussed in Section 2.5.

All three models were re-evaluated across three successive iterations of the
feature set (107, 117 and 124 columns) as literature-motivated features were
added (Sections 2.3–2.4), and the HGT's hyperparameters (hidden dimension,
attention heads, training epochs) were subjected to a dedicated tuning pass
before the final reported configuration was selected — see Section 4 [TBD]
for the finding that motivated this step.

### 3.6 Evaluation protocol

We use repeated stratified k-fold cross-validation (5 folds × 6 repeats = 30
folds), the same folds and random seed shared across all three models to
permit paired statistical testing. We report PR-AUC and Precision@k (k = 10,
20, 50) per fold, and compare models pairwise with the Wilcoxon signed-rank
test on per-fold PR-AUC. All results are reported for both the primary
(`y_direto`, 148 positives) and sensitivity (`y_qualquer`, 188 positives)
labels (Section 3.2).

---

## 4. Results

*(TBD — pending the final 30-fold, tuned-HGT experiment. Structure planned:
4.1 primary label results and pairwise tests; 4.2 sensitivity label results;
4.3 effect of literature-motivated feature engineering across the three
iterations; 4.4 effect of hyperparameter tuning on the HGT.)*

## 5. Discussion

*(TBD. Planned angles: (a) interpretation through the graph-features-vs-GNN
and information-overload literatures from Section 2; (b) practical
implication for resource-constrained oversight bodies — what should a
comptroller's office or court of accounts actually deploy, given headcount,
compute and labeling-budget constraints typical of the public sector, not an
idealized research setting; (c) limitations — label incompleteness/PU
framing, address/partner identity-resolution heuristics, judicial-process
data excluded as unreliable, single-region scope.)*

## 6. Conclusion

*(TBD.)*

## References

*(Working list — every entry below has been verified directly against its
primary source — arXiv abstract page or publisher DOI record, fetched
directly, not taken from search-engine summaries — on 2026-08-14, following
an explicit accuracy check. One entry originally in this draft, an unverifiable
ResearchGate paper whose "author list" turned out to be a fabricated,
alphabetically-sequential set of names, was removed rather than fixed — see
`docs/research_plan.md` for the correction log. To be converted to full
APA/BibTeX formatting before submission.)*

- Abdou, A., Basdevant, O., David-Barrett, E., & Fazekas, M. (2022).
  Assessing vulnerabilities to corruption in public procurement and their
  price impact. *IMF Working Paper 22/094*.
- Cheng, D., Zou, Y., Xiang, S., & Jiang, C. (2024). Graph neural networks
  for financial fraud detection: A review. arXiv:2411.05815.
- Dou, Y., Liu, Z., Sun, L., Deng, Y., Peng, H., & Yu, P. S. (2020). Enhancing
  graph neural network-based fraud detectors against camouflaged fraudsters.
  *CIKM 2020*.
- Fazekas, M., & Kocsis, G. (2020). Uncovering high-level corruption:
  Cross-national objective corruption risk indicators using public
  procurement data. *British Journal of Political Science*, 50(1), 155–164.
- Hu, Z., Dong, Y., Wang, K., & Sun, Y. (2020). Heterogeneous graph
  transformer. *WWW 2020*. arXiv:2003.01332.
- Ivanov, S., & Prokhorenkova, L. (2021). Boost then convolve: Gradient
  boosting meets graph neural networks. arXiv:2101.08543.
- Liu, Y., Ao, X., Qin, Z., Chi, J., Feng, J., Yang, H., & He, Q. (2021). Pick
  and choose: A GNN-based imbalanced learning approach for fraud detection.
  *WWW 2021*.
- Ma, X., Li, R., Liu, F., Ding, K., Yang, J., & Wu, J. (2024). Graph anomaly
  detection with few labels: A data-centric approach. *KDD 2024*, 2153–2164.
- Moody's Analytics. (2023, January 22). 7 indicators of shell company risk.
  https://www.moodys.com/web/en/us/kyc/resources/insights/seven-indicators-shell-company-risk.html
- Vandervorst, F., Deprez, B., Verbeke, W., & Verdonck, T. (2025). Inductive
  inference of gradient-boosted decision trees on graphs for insurance fraud
  detection. arXiv:2510.05676.
- Wang, S., Zhang, Z., Fang, L., Nguyen, C.-T., & Li, W. (2025). Corporate
  fraud detection in rich-yet-noisy financial graph. arXiv:2502.19305.
- Wang, X., Ji, H., Shi, C., Wang, B., Cui, P., Yu, P., & Ye, Y. (2019).
  Heterogeneous graph attention network. *WWW 2019*. arXiv:1903.07293.
- Yang, H., Zhang, Y., Yao, Q., & Kwok, J. (2023). Positive-unlabeled node
  classification with structure-aware graph learning. arXiv:2310.13538.
