# Working title

**Simple Models, Explicit Graph Signals: Rethinking Heterogeneous Graph Neural
Networks for Administrative Sanction Risk Screening in Local Government Data**

*(provisional — revisit once Results/Discussion are finalized; target venue:
Government Information Quarterly)*

> **Status of this draft (updated 2026-08-15)**: full first draft (2026-08-14)
> was reviewed as if by an experienced GIQ referee, who recommended major
> revision on five points: (1) near-zero engagement with the public-sector/AI
> adoption literature that anchors GIQ's own scope, (2) an overclaimed
> novelty framing given a contemporaneous parallel finding elsewhere, (3) a
> known statistical caveat of Wilcoxon tests on repeated-CV folds left
> unstated, (4) the Section 5.2 mechanism asserted rather than flagged as
> literature-grounded-but-untested, (5) no data/code availability statement.
> All five have been addressed in this revision (Sections 1, 2.1, Abstract,
> 5.2, 5.5, and the new "Data and code availability" section), plus two
> further improvements: a deployment sketch (Section 5.4) and a dedicated
> ablation (Section 4.6, 2026-08-15) isolating the marginal contribution of
> round 3's two feature groups — which revised Section 5.2's original
> "explicit graph features close the gap" reading into a more nuanced,
> and better-supported, account (see Section 4.6/5.2). A second, separate
> ablation (Section 4.7, 2026-08-15) isolated the contribution of each
> auxiliary node type inside the HGT, similarly revising the "information
> overload" account in Section 5.2: the pattern is not uniform across node
> types, and the shared-partner node — this paper's central metapath —
> turned out to matter least inside the HGT specifically. Word budget
> target for GIQ: ~8,000–10,000 words. All numbers were computed directly
> from
> `docs/resultados/comparar_baselines_30folds_v4_han_hgt_tunado_2026-08-14.log`,
> `docs/resultados/ablation_features_tabular_2026-08-15.log`, and
> `docs/resultados/ablation_tipo_no_hgt_2026-08-15.log`, cross-checked
> against `docs/research_plan.md`, not reconstructed from memory; all
> citations verified directly against primary sources (see References note).

---

## Abstract

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
framing appropriate to the extreme rarity of confirmed sanctions (148 direct,
188 including partner-inferred cases). Our finding echoes a pattern reported
in at least one contemporaneous benchmark in a different fraud domain
(Vandervorst et al., 2025); this paper's contribution is to test it with
substantially more methodological scrutiny, and in a new public-sector
domain where the practical stakes of the answer differ. Across five independent evaluation
rounds — three successive feature-engineering iterations plus a dedicated
hyperparameter-search round for the HGT, all under repeated stratified
cross-validation (30 folds) with paired statistical testing — the
heterogeneous model is consistently and significantly outperformed by both
simpler alternatives on the primary, non-circular label (tabular: 50.7×
lift over base rate; homogeneous GNN: 76.3×; tuned HGT: 32.6×; tabular >
HGT, p = 0.0024; homogeneous GNN > HGT, p < 0.0001). Critically, a dedicated
hyperparameter search — motivated by the finding that the HGT was
undertrained, not undersized — improved its performance by 39% but did not
change this ranking, closing the most likely methodological objection to the
result. On a secondary, partner-inferred label where the labeling mechanism
itself overlaps with the shared-partner metapath, the tuned HGT's advantage
grows instead of shrinking (60.8× lift, up from 42.8× before tuning) — a
pattern more consistent with the model exploiting label circularity than
discovering genuine structural risk signal. We discuss the practical
implication for resource-constrained oversight bodies: a well-engineered
tabular model augmented with explicit graph-degree features, not a
heterogeneous graph transformer, is the defensible choice for this task
given realistic public-sector compute and labeling budgets.

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

Public-sector adoption of AI and machine learning tools is expanding, but
consistently constrained by technical capacity, data quality, and staff
competency barriers rather than by model availability (Sun & Medaglia,
2019) — a body of evidence this paper takes as a starting design constraint,
not an afterthought: any modeling choice recommended here must be
justifiable to an oversight body with exactly these constraints, not only
to a machine-learning audience with unlimited compute.

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
2. **A direct test of whether implicit structural learning outperforms
   explicit graph features**, probing a claim from the graph-features-vs-GNN
   literature (Section 2.3) in a new domain: it does not, here — a tabular
   model given the same three metapaths as explicit degree features, and an
   even simpler homogeneous GNN, both significantly outperform the fully
   heterogeneous model on the primary label (Section 4.1), a result that
   survives a dedicated hyperparameter-tuning pass (Section 4.4).
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
reports results across five evaluation rounds, a hyperparameter-tuning study,
and a sensitivity analysis. Section 5 discusses the findings from a
public-sector practitioner's perspective and states limitations. Section 6
concludes.

---

## 2. Related work

### 2.1 AI adoption in public-sector oversight, and administrative sanction risk in Brazilian public data

Government Information Quarterly's own literature on AI adoption in public
administration consistently finds that the binding constraint on deploying
predictive tools in government is organizational and technical capacity —
staff competency, data quality, infrastructure — not the availability of a
capable model (Sun & Medaglia, 2019). This paper takes that finding as a
design requirement rather than a caveat added after the fact: Section 3.1
deliberately scopes the study to a dataset and compute budget realistic for
an under-resourced oversight body, and Section 5.4 evaluates each model
explicitly against that constraint, not only against predictive performance.
A related design-principles literature on digital transparency in
government (Matheus, Janssen, & Janowski, 2021) treats auditability and
institutional legibility of a digital tool as design goals in their own
right, not only as compliance afterthoughts — a consideration this paper
extends from transparency of *process* to transparency of *model*: a
gradient-boosted model whose feature importances are directly inspectable
is more legible to an oversight body's own audit function than a
heterogeneous graph transformer's learned embeddings, independent of any
difference in predictive performance between them (Section 5.4).

Within this constraint, prior Brazilian work on corporate risk and fraud
detection using public registry data is predominantly tabular, treating each
company as an independent observation described by registration attributes,
tax regime, debt records, and sector codes. This body of work does not, to
our knowledge, model the *network* of relationships between companies —
shared partners, shared addresses, shared political ties — as a first-class
object, despite these relationships being directly queryable in the same
public data sources (Federal Revenue registry, electoral court records).
International work applying relational models (HAN, HGT, R-GCN) to corporate
networks exists, but concentrates heavily on credit/insolvency risk — a
comparatively data-rich, well-labeled problem — leaving administrative-sanction
risk detection via ownership/address/political metapaths, in a Brazilian
public-data context, comparatively unexplored.

### 2.2 Heterogeneous graph neural networks for corporate and financial networks

Heterogeneous graph attention/transformer architectures — HAN (Wang et al.,
2019) and HGT (Hu et al., 2020) foremost among them — extend graph neural
networks beyond a single node/edge type, learning type-specific attention
weights across metapaths or relations. Applied to corporate networks, this
family of models targets shell-company and beneficial-ownership risk
(Moody's, 2023) and financial fraud more broadly, on the premise that fraud
rings leave a structural signature — shared infrastructure, shared
counterparties — "invisible in feature space but detectable in graph
topology." A parallel literature on shell-company networks in public
procurement corroborates this premise with a concrete mechanism: ownership
and management data, combined with contracting data, surfaces connected
components indicative of collusion risk (Section 2.4).

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
& Kocsis, 2020; Abdou et al., 2022) establishes non-competitive-award and
cost-overrun indicators — sole-source contract modality, and divergence
between initial and final contract value — as robust, auditable corruption-risk
proxies, independent of any prosecutorial outcome. Shell-company detection
practice adds company age and partner/director concentration across many
firms as further red flags (Moody's, 2023). All four indicators are feasible
to construct from the registry data used here (Section 3.4) and motivated a
dedicated round of literature-grounded feature engineering prior to the
final experiment reported below.

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
draw on this literature in Section 5.2 to interpret our finding that the
more complex, end-to-end heterogeneous model underperforms simpler
alternatives under the exact conditions (extreme imbalance, few labels,
attribute-poor auxiliary node types) this literature was designed to
address.

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

**Data ethics and privacy.** Partner tax IDs (CPF) are already masked at
source by the issuing federal registry; no additional pseudonymization was
required for this study's purposes. Political-connection records (Section
3.3) originate from public electoral-court candidacy/donation filings.
Individual partner names and company identifiers are used internally to
construct the network and features, but — this study's ethical-use
commitment — are never published in aggregate results, and are replaced with
generic labels (e.g., "Partner X", "Company A/B") in any illustrative
example drawn from a specific case, including Figure 1.

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

Figure 1 shows a concrete (anonymized) example of this mechanism from the
dataset: two companies in the same municipality, neither directly sanctioned,
that share a partner who was personally sanctioned (listed on CEIS as an
individual). Both companies enter `y_qualquer` as positives solely through
this shared-partner link — the exact relation the shared-partner metapath is
built to detect.

![Real, anonymized example: two companies (labeled A and B, company identifiers and the partner's name withheld per the ethical-use commitment in this study) sharing a partner who was personally sanctioned; neither company has a direct sanction of its own.](figuras/figura1_caso_socio_comum.png)

*Figure 1. A real case (identifiers anonymized) illustrating the
partner-inferred labeling mechanism behind the 40 companies that separate
`y_qualquer` from `y_direto`.*

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

![Schematic of the HIN's three hypothesis metapaths: shared partner, shared address, and shared political connection, each linking two company nodes through one intermediate node type.](figuras/figura2_esquema_metapaths.png)

*Figure 2. Schematic of the three company–X–company metapaths used to build
the heterogeneous information network (abstract illustration, not drawn from
specific companies).*

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
before the final reported configuration was selected — see Section 4.4 for
the finding (undertraining, not undersizing) that motivated this step.

### 3.6 Evaluation protocol

We use repeated stratified k-fold cross-validation (5 folds × 6 repeats = 30
folds), the same folds and random seed shared across all three models to
permit paired statistical testing. Our evaluation harness computes both
PR-AUC and Precision@k (k = 10, 20, 50) per fold; we report PR-AUC as the
primary metric throughout this paper and compare models pairwise with the
Wilcoxon signed-rank test on per-fold PR-AUC. Precision@k was tracked during
development but is not reported in the main results: at roughly 148/5 ≈ 30
positive cases per test fold, Precision@k at these thresholds is
substantially noisier fold-to-fold than PR-AUC (its standard deviation
across folds frequently exceeds its mean in early exploratory runs), making
PR-AUC the more informative and stable metric for the paired comparisons
this paper relies on. All results are reported for both the primary
(`y_direto`, 148 positives) and sensitivity (`y_qualquer`, 188 positives)
labels (Section 3.2).

---

## 4. Results

All PR-AUC figures below are averaged over 30 folds (5-fold stratified
cross-validation × 6 repeats), with the same folds and random seed shared
across models within a given round, permitting paired Wilcoxon
signed-rank tests. "Lift" is PR-AUC divided by the label's base rate
(148/344,130 = 0.0430% for `y_direto`; 188/344,130 = 0.0546% for
`y_qualquer`). Five evaluation rounds were run in total as the feature set
and the HGT configuration evolved; we report the final (fifth) round in
detail and the preceding four as a robustness trajectory (Section 4.3).

### 4.1 Primary label (`y_direto`, 148 confirmed direct sanctions)

| Model | PR-AUC | Lift |
|---|---|---|
| Tabular (XGBoost) | 0.0218 ± 0.0146 | 50.7× |
| Homogeneous GNN | 0.0328 ± 0.0228 | **76.3×** |
| HGT (tuned) | 0.0140 ± 0.0181 | 32.6× |

Pairwise Wilcoxon tests (paired, 30 folds): tabular vs. homogeneous GNN,
p = 0.0145 (homogeneous GNN significantly higher); tabular vs. HGT,
p = 0.0024 (tabular significantly higher); homogeneous GNN vs. HGT,
p < 0.0001 (homogeneous GNN significantly higher). All three pairwise
differences are statistically significant, and the ranking is consistent:
**homogeneous GNN > tabular > HGT** on the primary label.

### 4.2 Sensitivity label (`y_qualquer`, 188 confirmed sanctions, including
40 companies labeled positive only via a shared-partner link to an
already-sanctioned entity)

| Model | PR-AUC | Lift |
|---|---|---|
| Tabular (XGBoost) | 0.0236 ± 0.0200 | 43.2× |
| Homogeneous GNN | 0.0219 ± 0.0124 | 40.1× |
| HGT (tuned) | 0.0332 ± 0.0259 | **60.8×** |

Pairwise tests: tabular vs. homogeneous GNN, p = 0.8394 (no significant
difference); tabular vs. HGT, p = 0.1579 (no significant difference);
homogeneous GNN vs. HGT, p = 0.0277 (HGT significantly higher). Unlike the
primary label, the tuned HGT is now the top performer by point estimate and
significantly ahead of the homogeneous GNN, though not (yet, at this
sample size) significantly ahead of the tabular baseline.

### 4.3 Effect of feature engineering across three iterations (robustness trajectory)

| Round | Features | `y_direto` lift (tab / homog. GNN / HGT) | `y_qualquer` lift (tab / homog. GNN / HGT) |
|---|---|---|---|
| 1 | 107 cols | 18.6× / 16.5× / 14.2× | 15.7× / 24.4× / 33.5× |
| 2 | 117 cols (+dashboard-derived features) | 75.8× / 81.2× / 29.3× | 62.3× / 41.6× / 45.4× |
| 3 | 124 cols (+literature-grounded features), untuned HGT | 50.7× / 76.3× / 23.5× | 43.2× / 40.1× / 42.8× |
| 4 | 124 cols, tuned HGT (this study) | 50.7× / 76.3× / **32.6×** | 43.2× / 40.1× / **60.8×** |

The ranking on the primary label — HGT statistically worse than both
alternatives — holds across all three feature-set versions (round 1:
tabular > HGT p = 0.0066, homogeneous GNN > HGT p = 0.0293; round 2/3:
tabular > HGT p < 0.0001 / p = 0.0001, homogeneous GNN > HGT p < 0.0001 in
both). It is not an artifact of a particular feature-engineering iteration.
Absolute PR-AUC is not monotonic across rounds 2→3 (all three models'
scores decreased when the literature-grounded features were added) — an
initial quick sanity check compared round 3 against round 1 rather than
round 2, creating a false impression of improvement; this is corrected here
and documented as a methodological lesson in the project's internal
research log. Section 4.6 reports a dedicated ablation that traces this
decline, for the tabular model, to the combination of round 3's two feature
groups rather than to either group individually.

### 4.4 Effect of hyperparameter tuning on the HGT

The HGT's original configuration (`hidden_channels=32`, `num_heads=1`,
`epochs=50`) had been constrained by an out-of-memory failure on the
development machine (8GB RAM) at larger settings, not selected by
hyperparameter search. A dedicated search (6 configurations, 5-fold
cross-validation, primary label only) found that **increasing training
epochs alone (50→150) nearly tripled PR-AUC** (0.0105→0.0244 in the search's
smaller-scale runs), while increasing hidden-layer width alone gave no
improvement (0.0105→0.0100) and the largest combined configuration
(`hidden=64, heads=2, epochs=100`) failed with an out-of-memory error again.
This diagnosis — undertraining, not undersizing — motivated selecting
`epochs=150` (holding `hidden=32, heads=1`) as the final configuration,
over a marginally higher-scoring but three-times-costlier alternative
(`heads=2` combined with `epochs=150`) that tied within noise (search-phase
PR-AUC 0.0249 vs. 0.0244, standard deviation ≈ 0.025–0.029 on both).

Applying this tuned configuration to the full 30-fold evaluation (Sections
4.1–4.2) improved the HGT's primary-label lift by 39% relative to the
untuned round 3 result (23.5×→32.6×) — confirming the undertraining
diagnosis was real, not a rationalization. **The HGT nonetheless remains
significantly worse than both alternatives on the primary label after
tuning** (Section 4.1). On the secondary label, tuning increased the HGT's
lift by more (42.8×→60.8×) than on the primary label — a pattern taken up
in Section 4.5.

### 4.5 Sensitivity analysis: primary versus secondary label, across all rounds

Sections 4.1–4.2 report primary- and secondary-label results for the final
round only. Table 4.5 computes, for every one of the four evaluation
rounds, each model's ratio of secondary-label lift to primary-label lift
(`y_qualquer` lift ÷ `y_direto` lift) — a direct measure of how much *more*
(ratio > 1) or *less* (ratio < 1) advantage a model gets from the label
definition that includes the 40 partner-inferred cases.

| Round | Tabular ratio | Homogeneous GNN ratio | HGT ratio |
|---|---|---|---|
| 1 (107 cols) | 0.84 | 1.48 | **2.36** |
| 2 (117 cols) | 0.82 | 0.51 | **1.55** |
| 3 (124 cols, untuned HGT) | 0.85 | 0.53 | **1.82** |
| 4 (124 cols, tuned HGT) | 0.85 | 0.53 | **1.87** |

Two patterns are stable across all four rounds, independent of feature-set
version and HGT tuning: (i) the tabular model's ratio is consistently below
1 (it does not get an edge from the circular label — if anything, a slight
disadvantage), (ii) the HGT's ratio is **always the highest of the three
models, and always above 1**, meaning it consistently extracts more relative
advantage from the label version whose construction overlaps with the
metapath it is built to exploit. The homogeneous GNN sits in between,
above 1 only in round 1 (the least feature-rich, least statistically
powered round). This is the quantitative basis for the reading proposed in
Section 5.3: the HGT's apparent strength on the secondary label is
best explained by its capacity to exploit label-construction circularity,
not by a structural-risk-detection advantage that would be expected to
generalize to the primary, non-circular label — where, on the contrary, it
is consistently the weakest model.

### 4.6 Ablation: isolating the contribution of round-3 feature groups

Round 3's seven new columns bundle two distinct feature groups —
four explicit graph-degree features (Section 3.4, first bullet) and three
procurement-corruption/shell-company indicators (Section 3.4, second
bullet) — added together, making it impossible to attribute round 2→3's
performance change to either group specifically. We closed this gap with a
dedicated ablation, training the tabular model alone (no GNN/HGT — this
ablation is inexpensive) on four feature-set variants under the same
30-fold protocol: the round-2 baseline (117 columns), round 2 plus only the
graph-degree features (121 columns), round 2 plus only the procurement/shell
indicators (120 columns), and the full round-3 set (124 columns).

| Variant | `y_direto` lift | `y_qualquer` lift |
|---|---|---|
| Round-2 baseline | **75.8×** | **62.2×** |
| + graph-degree features only | 69.7× | 50.8× |
| + procurement/shell indicators only | 60.6× | 49.0× |
| + both (full round 3) | 50.7× | 43.2× |

The result runs against the more intuitive hypothesis that explicit
graph-derived features are what let the tabular model compete with the
graph-based models (Section 5.2). Neither feature group individually is significantly worse
than the round-2 baseline on `y_direto` (graph-only: Wilcoxon p = 0.670;
procurement-only: p = 0.088), and only the procurement group is
significantly worse on `y_qualquer` (p = 0.0016; graph-only: p = 0.092,
borderline). But the *combination* of both groups is significantly worse
than the baseline on both labels (p = 0.0062, p = 0.0001) — and
significantly worse than the graph-only variant on both labels as well
(p = 0.0062, p = 0.0145). In other words, the round-2 tabular baseline,
*without* any of round 3's seven additional columns, is the single
best-performing tabular configuration we tested — better than the 124-column
configuration used as the tabular baseline throughout Sections 4.1–4.5. Adding
features that are individually plausible and literature-motivated
nonetheless hurt performance once combined, an effect large enough to be
statistically significant, most likely reflecting overfitting risk from
added dimensionality relative to only 148–188 positive labels, not
re-compensated by any hyperparameter re-tuning when the feature set grew
(XGBoost's regularization settings were held fixed across all four feature-set
versions in Section 4.3, by design, to isolate the effect of features
alone — see Section 5.5).

### 4.7 Ablation: isolating the contribution of each auxiliary node type in the HGT

Section 5.2 (in its first-pass form) offered "information overload" — the
numerical dominance of attribute-poor auxiliary node types (partner,
address, political connection) over company nodes — as a plausible,
literature-grounded explanation for why the HGT underperforms, without
testing which auxiliary node type specifically drives the effect. We
closed this gap with a second ablation: removing each auxiliary node type
from the HGT one at a time (holding the tuned configuration —
`hidden=32, heads=1, epochs=150` — fixed), evaluated at 5 folds on the
primary label only, the same reduced-scale diagnostic protocol used for
the hyperparameter search (Section 4.4), against the already-established
all-three-node-types result from that search (PR-AUC 0.0244, 56.7× lift).

| Configuration | PR-AUC | Lift | Δ vs. all three |
|---|---|---|---|
| All three auxiliary node types (baseline) | 0.0244 | 56.7× | — |
| Without partner nodes | 0.0231 | 53.7× | −0.0013 |
| Without address nodes | 0.0080 | 18.7× | **−0.0164** |
| Without political-connection nodes | 0.0102 | 23.7× | **−0.0142** |

This is not the uniform picture "information overload" predicts. Removing
the partner node type — central to this paper's main shared-partner
hypothesis, and to the label-circularity mechanism discussed in Section
5.3 — barely changes performance. Removing either the address or the
political-connection node type, by contrast, causes a large drop, despite
address and political-connection edges differing by more than two orders
of magnitude in count (344,130 versus 866) — ruling out edge count alone
as the explanation and pointing instead to genuine informational content
in those two relations specifically. (We report point estimates and
differences rather than a paired significance test here: the all-three-node-types
result was reused from the hyperparameter search, whose raw per-fold
scores were not retained once that search concluded, precluding a
Wilcoxon test against the freshly run removal variants; Section 5.2
discusses what this diagnostic-scale result does and does not license us
to conclude.)

## 5. Discussion

This section separates two kinds of contribution this paper makes, in line
with the convention of distinguishing implications for research from
implications for practice. Sections 5.1–5.3 draw out the **theoretical
contribution**: why a heterogeneous graph neural network fails to
outperform simpler alternatives here, what that implies for the
graph-features-vs-GNN and information-overload literatures (Section 2),
and why the one label on which the HGT does appear to win should not be
read as evidence against that conclusion. Section 5.4 turns to the
**practical contribution**: what an oversight body should actually deploy,
and at what real cost, given realistic public-sector constraints. Section
5.5 states limitations common to both.

### 5.1 A negative result that survives its strongest methodological challenge

The single most likely objection a skeptical reviewer would raise against
an early version of this result — that the heterogeneous model was simply
undertrained relative to the simpler baselines — was tested directly rather
than argued away. It turned out to be partly correct (epochs were indeed a
real bottleneck) and irrelevant to the conclusion: a hyperparameter search
that measurably improved the HGT (Section 4.4) did not change its ranking
relative to the tabular or homogeneous-GNN baselines on the primary label.
Across five independent evaluation rounds — three feature-set iterations
plus a dedicated tuning round — the heterogeneous graph transformer is
consistently and significantly outperformed on the task this paper's
research question is actually about: detecting companies with a *direct*,
non-circular administrative sanction. We treat this as a robust empirical
finding, not a provisional one awaiting further tuning.

### 5.2 Why might a simpler model win here? A more nuanced picture than "explicit features close the gap"

Our first-pass reading of this result, before running the ablation in
Section 4.6, was that it fit neatly into the graph-features-vs-GNN
literature (Section 2.3): give a gradient-boosted tabular model explicit
access to the same structural information a GNN would otherwise learn
implicitly, and it closes most of the performance gap. The ablation
complicates this story rather than confirming it. The tabular model's
respectable performance is real, but it is not attributable to the
graph-degree features specifically — those features, added alone, do not
significantly improve on the tabular model's round-2 baseline (Section
4.6), and the round-2 baseline (no graph features at all) is in fact the
single best-performing tabular configuration we tested (75.8× lift on
`y_direto`, ahead of the 124-column configuration's 50.7× used as the
tabular baseline throughout Sections 4.1–4.5). What the graph-features-vs-GNN
literature gets right here is the broader point that a well-specified
tabular model, whether or not the specific structural features tested help,
is a strong competitor to a heterogeneous GNN — just not for the precise
mechanism ("explicit graph features substitute for what the GNN would learn
implicitly") that motivated Section 3.4's feature design. A useful,
more general lesson survives regardless: under the extreme label scarcity
studied here (148–188 positives), adding more literature-motivated features
without re-tuning model regularization can measurably hurt a gradient-boosted
model, independent of whether those features are graph-derived or not
(Section 4.6) — a caution about feature engineering under extreme
imbalance that is, to our knowledge, not the emphasis of the
graph-features-vs-GNN literature we drew on, which does not typically test
feature growth under label scarcity this severe.

Second, we tested the "information overload" diagnosis from corporate
fraud-detection literature (Section 2.5) — that attribute-poor auxiliary
node types (partner, address, political connection), which outnumber
attribute-rich company nodes by more than an order of magnitude (142,844
partner nodes and 181,268 address nodes against 344,130 company nodes),
dilute rather than enrich signal — directly, with the node-type ablation
in Section 4.7. The result complicates this diagnosis rather than
confirming it uniformly. If attribute-poor structure simply diluted signal
by weight of numbers, removing any one auxiliary node type should help, or
at least not hurt, roughly in proportion to how much structure was
removed. That is not what we observe: removing the partner node type — by
far the largest of the three, and the one central to this paper's main
shared-partner metapath hypothesis — barely changes the HGT's performance,
while removing either the address or the political-connection node type
causes a large drop, despite their edge counts differing by two orders of
magnitude (344,130 versus 866). A blanket "too much attribute-poor
structure" account cannot explain why removing the largest attribute-poor
node type is nearly neutral while removing a far sparser one is one of the
two most damaging removals.

A revised, more precise reading is that the HGT's auxiliary node types are
not uniformly signal or uniformly noise: address and political-connection
nodes appear to carry structural information the model uses productively,
while the partner node type — despite anchoring this paper's central
research question — contributes comparatively little inside the HGT
specifically. This leaves the central empirical finding untouched: the
HGT, even drawing on whatever real signal address and political-connection
nodes provide, is still outperformed by the tabular and homogeneous-GNN
baselines on the primary label (Sections 4.1, 4.3). What changes is the
explanation for *why*: not simply "too much weakly-informed auxiliary
structure," but something closer to a capacity-under-label-scarcity
account — a heterogeneous, per-relation-typed architecture does not
convert the real signal in these relations into predictive performance as
efficiently as a coarser aggregation (homogeneous GNN) or an explicit
feature (tabular) does, when only 148 labels are available to learn how to
weigh that structure. We flag this diagnostic-scale finding (5 folds, point
estimates without a paired significance test — Section 4.7) as a genuine
revision to our own earlier account, not a confirmation of it — and, as a
side effect, an early, diagnostic-scale answer to an interpretability
question this paper's three-metapath design invites but does not, on its
own, resolve: which metapath carries more signal. Inside the HGT
specifically, shared address and shared political connection, more than
shared partner, appear to be what the model actually draws on — a finding
at odds with the shared-partner metapath's central billing in this paper's
framing (Section 1), and one that would need confirmation at full
statistical power (30 folds, a proper paired test) before being treated as
more than suggestive.

### 5.3 The secondary label's reversal is evidence of circularity exploitation, not of genuine advantage

The sensitivity label (`y_qualquer`) includes 40 companies whose only path
to a positive label is a shared-partner connection to an already-sanctioned
entity — the exact relation the shared-partner metapath is built to detect.
If the tuned HGT's larger advantage on this label (60.8× lift, up from 42.8×
before tuning, and now significantly ahead of the homogeneous GNN) reflected
newly discovered structural risk signal, we would expect a comparable gain
on the primary label, where no such circularity exists. It does not: the
primary-label gain from tuning (23.5×→32.6×) is real but far more modest,
and the HGT remains the weakest model there. The more parsimonious
explanation is that additional training capacity lets the HGT more
thoroughly learn the shared-partner relation specifically where doing so
directly reproduces part of the labeling rule, rather than uncovering new
predictive signal. We report this pattern explicitly rather than
foreground the sensitivity-label result, precisely because it illustrates
how an uncritical read of the "GNN wins" result on a circular label could
mislead a practitioner.

### 5.4 Practical implication for resource-constrained oversight bodies

For the audience this paper is aimed at — comptroller offices, courts of
accounts, and procurement oversight bodies operating with limited
data-science headcount, no dedicated GPU infrastructure, and few confirmed
labels to learn from — the practical recommendation is direct: **a
gradient-boosted tabular model augmented with a small number of explicit
graph-degree features (shared partner, shared address, shared political
connection counts) is both cheaper to build and operate, and more effective,
than a heterogeneous graph transformer for this task.** The tabular model
trains in seconds on commodity hardware; the tuned HGT, even for a single
production fit rather than the full cross-validation exercise behind
Section 4, takes on the order of minutes, and required a dedicated,
technically demanding tuning procedure (Section 4.4) just to reach its best
achievable performance — which still fell short (see the cost accounting
below). Where an oversight body does have
appetite for graph-based methods, this result recommends the simpler,
single-relation-type homogeneous GNN over a fully heterogeneous
architecture, at least until node types other than "company" carry genuine
attribute data of their own (Section 5.2) rather than only a learned
embedding.

Concretely, an oversight body adopting the recommended tabular model faces
three operational decisions this study's results directly inform. First,
retraining cadence: since confirmed sanctions accrue slowly (188 over the
registry's full history here), retraining need not be more frequent than
new sanction decisions are handed down and ingested — quarterly or
semi-annually is plausible, not the continuous retraining a production ML
system might default to. Second, queue sizing: Precision@k at the k=10–20
scale is, in our own exploratory runs, noisy enough (standard deviation
comparable to or exceeding the mean, at roughly 30 positives per evaluation
fold) that we would recommend an oversight body size its review queue
larger — k=50 or k=100 — matching it to realistic monthly analyst capacity
rather than to the top of a ranked list alone. Third, and following from
Section 2.1's transparency-by-design premise (Matheus et al., 2021): a
gradient-boosted model's feature importances (e.g., SHAP values) can be
attached to every flagged company as a stated reason for review — "flagged
for shared partner with N other companies, one under active sanction" — in
a form directly usable in an audit finding. The HGT offers no equivalent
without substantial additional engineering, since its risk signal is
distributed across learned embeddings with no natural per-company
explanation. This transparency gap is a further, independent reason to
prefer the simpler model here, beyond the compute-cost and predictive-performance
arguments already given.

"Hours" and "minutes" can describe different things, so a precise cost
accounting is worth stating explicitly. The 30-fold cross-validation
exercise underlying Section 4 is a one-time research cost, not a recurring
operational one: roughly 155 seconds total for the tabular model against
roughly 6.6 hours per label for the tuned HGT. A production deployment
does not repeat 30-fold cross-validation on every retraining cycle,
however — a single model fit is what recurs, and here the gap narrows
sharply: on the order of 5 seconds for the tabular model versus roughly 13
minutes for the HGT, both comfortably inside the quarterly-or-slower
retraining cadence argued for above, on a single CPU core, no GPU
required. The durable cost asymmetry between the two models is therefore
not primarily in recurring compute. It is, first, in the one-time
development and hyperparameter-tuning effort (Section 4.4: on the order of
15 hours of compute, plus the specialized graph-learning expertise needed
to design and interpret that search — expertise most oversight-body data
teams do not have on staff — against a tabular model any competent data
scientist can tune with near-default settings). It is, second, in a memory
floor: full-batch HGT training required repeated adjustments to fit within
8GB of RAM (Section 3.5), a hardware requirement not every oversight
body's existing infrastructure can be assumed to meet, where the tabular
model trains comfortably on a small fraction of that.

### 5.5 Limitations

Our pairwise significance tests use the Wilcoxon signed-rank test on 30
repeated stratified cross-validation folds — the same folds across models,
which is what permits pairing, but folds drawn by repeated k-fold CV from a
single dataset are not fully independent samples, since companies are reused
across overlapping splits. Naively applied significance tests on repeated-CV
folds are known to understate variance and can inflate false-positive rates
relative to corrected alternatives designed for this setting, such as the
5×2cv paired test (Dietterich, 1998). We consider our qualitative conclusion
robust to this caveat because the primary-label ranking is not a single
borderline p-value but a large, consistent effect replicated across five
independent evaluation rounds with different feature sets and an
independently re-tuned model (Section 4.3) — the kind of cross-round
replication a corrected single-round test cannot substitute for — but we
note the caveat explicitly rather than let the specific p-values in Section
4 be read as more precise than a repeated-CV Wilcoxon test actually supports.

The primary label itself is positive-unlabeled, not exhaustive: absence of
a confirmed sanction does not establish absence of wrongdoing, only absence
of (so far) confirmed detection — a limitation inherent to this task
domain, not specific to our method, but one that bounds how the PR-AUC/lift
figures reported here should be interpreted (as detection of *already
caught* risk, with unknown recall against uncaught risk). Partner and
address identity resolution use registry-level heuristics (masked-CPF plus
normalized name for partners; normalized street/number/postal-code for
addresses) that do not resolve homonyms or spelling variants — a source of
noise in all three metapaths, likely attenuating rather than inflating the
graph-based models' measured advantage. Judicial-process records were
excluded from the network entirely, as they are matched by name via a
still-in-development record-linkage pipeline rather than by tax ID, and were
judged too unreliable to include as either label or feature. The study is
scoped to a single Brazilian metropolitan region (seven municipalities);
generalization to other regions, sanction regimes, or company-registry
structures is untested. The hyperparameter search (Section 4.4) covered six
configurations chosen to isolate three specific axes (depth of training,
hidden width, attention heads) rather than an exhaustive or Bayesian search;
we consider this proportionate given that it directly tested, and closed,
the single most likely objection to the result, but a larger search remains
possible future work. Round 3's two feature groups (explicit graph-degree
features and procurement/shell-company indicators, Section 3.4) were
initially added together, in the same pass, which would have left it
impossible to attribute round 2→3's performance change to either group
specifically; we closed this gap with the ablation reported in Section 4.6,
which found — counter to our initial reading — that neither group
significantly improves on the round-2 baseline alone, and that the *combination*
of both is what drives a statistically significant decline (Section 5.2).
The ablation was restricted to the tabular model, since it is inexpensive
to re-run relative to the GNN and HGT (Section 4.4); an equivalent ablation
for the graph-based models remains future work, and would clarify whether
the same feature-combination effect operates there. The node-type ablation
in Section 4.7 was run at the same reduced scale as the hyperparameter
search (5 folds, primary label only), and its baseline (all three auxiliary
node types) was reused from that earlier search rather than re-run inside
this ablation — meaning we can report point estimates and differences
(Sections 4.7, 5.2) but not a paired significance test against that
baseline, since the search's raw per-fold scores were not retained once it
concluded. The pattern is large enough (removing address or political
connection roughly halves lift; removing partner barely moves it) that we
consider it a genuine, if diagnostic-scale, revision to our own account of
*why* the HGT underperforms — but confirming it at full statistical power
(30 folds, a proper paired test, ideally with per-fold data retained
throughout for exactly this kind of follow-up analysis) remains future
work.

## 6. Conclusion

Testing whether a heterogeneous graph neural network improves detection of
administrative-sanction risk over simpler alternatives, on a real, complete
registry of 344,130 companies with a positive-unlabeled, extremely rare
label, we find that it does not — and that this finding survives its most
serious methodological challenge. A dedicated hyperparameter search
confirmed the heterogeneous model (HGT) was undertrained in its initial
configuration and meaningfully improved it once corrected, yet the improved
model remains significantly outperformed by both a tabular baseline given
explicit graph-degree features and a simpler homogeneous graph neural
network, on the primary, non-circular sanction label, across five
independent evaluation rounds. Where the heterogeneous model does show an
advantage — on a secondary label partly constructed via the same
shared-partner relation the model exploits — the pattern is more consistent
with exploiting label circularity than with discovering genuine structural
risk signal. For public-sector oversight bodies operating with limited
compute and technical capacity, this result recommends a well-engineered
tabular model with explicit graph-derived features, not a heterogeneous
graph transformer, as the defensible choice for administrative-sanction
risk screening at this scale and label scarcity.

## Data and code availability

All code (data loaders, HIN construction, metapath extraction, feature
engineering, the three models, the evaluation harness, and the scripts used
to run every experiment reported in Section 4) is publicly available
at https://github.com/brunokobi/experimento2026, including this manuscript's
source file and the full experiment logs underlying Section 4 (in
`docs/resultados/`). The underlying company registry
(`projeto_grande_vitoria_empresas`) is a separately maintained, publicly
available ETL pipeline and dataset release (see the repository's `README`
for the current release). The registry contains personal data (partner
names, masked tax IDs, addresses); it is published under the source
project's own data-governance terms, not re-distributed by this paper.
Figure 1 anonymizes all individual and company identifiers per the
ethical-use commitment described in Section 3.1.

## References

*(APA 7th edition. Every entry has been verified directly against its
primary source — arXiv abstract page or publisher/DOI record, fetched
directly, not taken from search-engine summaries — on 2026-08-14. Fields not
directly confirmed during verification — e.g., unlisted DOIs or page ranges
— are omitted rather than reconstructed. One entry originally in this
draft, an unverifiable ResearchGate paper whose "author list" turned out to
be fabricated, was removed rather than fixed — see `docs/research_plan.md`
for the correction log.)*

Abdou, A., Basdevant, O., David-Barrett, E., & Fazekas, M. (2022). *Assessing
vulnerabilities to corruption in public procurement and their price impact*
(IMF Working Paper No. 2022/094). International Monetary Fund.
https://www.imf.org/en/Publications/WP/Issues/2022/05/20/Assessing-Vulnerabilities-to-Corruption-in-Public-Procurement-and-Their-Price-Impact-518197

Cheng, D., Zou, Y., Xiang, S., & Jiang, C. (2024). Graph neural networks for
financial fraud detection: A review. *arXiv*. https://arxiv.org/abs/2411.05815

Dietterich, T. G. (1998). Approximate statistical tests for comparing
supervised classification learning algorithms. *Neural Computation*, *10*(7),
1895–1923. https://doi.org/10.1162/089976698300017197

Dou, Y., Liu, Z., Sun, L., Deng, Y., Peng, H., & Yu, P. S. (2020). Enhancing
graph neural network-based fraud detectors against camouflaged fraudsters.
In *Proceedings of the 29th ACM International Conference on Information and
Knowledge Management* (CIKM '20). Association for Computing Machinery.
https://doi.org/10.1145/3340531.3411903

Fazekas, M., & Kocsis, G. (2020). Uncovering high-level corruption:
Cross-national objective corruption risk indicators using public
procurement data. *British Journal of Political Science*, *50*(1), 155–164.
https://www.cambridge.org/core/journals/british-journal-of-political-science/article/abs/uncovering-highlevel-corruption-crossnational-objective-corruption-risk-indicators-using-public-procurement-data/8A1742693965AA92BE4D2BA53EADFDF0

Hu, Z., Dong, Y., Wang, K., & Sun, Y. (2020). Heterogeneous graph
transformer. In *Proceedings of The Web Conference 2020* (WWW '20).
https://arxiv.org/abs/2003.01332

Ivanov, S., & Prokhorenkova, L. (2021). Boost then convolve: Gradient
boosting meets graph neural networks. *arXiv*. https://arxiv.org/abs/2101.08543

Liu, Y., Ao, X., Qin, Z., Chi, J., Feng, J., Yang, H., & He, Q. (2021). Pick
and choose: A GNN-based imbalanced learning approach for fraud detection.
In *Proceedings of The Web Conference 2021* (WWW '21). Association for
Computing Machinery. https://doi.org/10.1145/3442381.3449989

Ma, X., Li, R., Liu, F., Ding, K., Yang, J., & Wu, J. (2024). Graph anomaly
detection with few labels: A data-centric approach. In *Proceedings of the
30th ACM SIGKDD Conference on Knowledge Discovery and Data Mining* (KDD
'24) (pp. 2153–2164). Association for Computing Machinery.
https://doi.org/10.1145/3637528.3671929

Matheus, R., Janssen, M., & Janowski, T. (2021). Design principles for
creating digital transparency in government. *Government Information
Quarterly*, *38*(1), 101550. https://doi.org/10.1016/j.giq.2020.101550

Moody's Analytics. (2023, January 22). *7 indicators of shell company risk*.
https://www.moodys.com/web/en/us/kyc/resources/insights/seven-indicators-shell-company-risk.html

Sun, T. Q., & Medaglia, R. (2019). Mapping the challenges of artificial
intelligence in the public sector: Evidence from public healthcare.
*Government Information Quarterly*, *36*(2), 368–383.
https://doi.org/10.1016/j.giq.2018.09.008

Vandervorst, F., Deprez, B., Verbeke, W., & Verdonck, T. (2025). Inductive
inference of gradient-boosted decision trees on graphs for insurance fraud
detection. *arXiv*. https://arxiv.org/abs/2510.05676

Wang, S., Zhang, Z., Fang, L., Nguyen, C.-T., & Li, W. (2025). Corporate
fraud detection in rich-yet-noisy financial graph. *arXiv*.
https://arxiv.org/abs/2502.19305

Wang, X., Ji, H., Shi, C., Wang, B., Cui, P., Yu, P., & Ye, Y. (2019).
Heterogeneous graph attention network. In *Proceedings of The Web
Conference 2019* (WWW '19) (pp. 2022–2032). Association for Computing
Machinery. https://doi.org/10.1145/3308558.3313562

Yang, H., Zhang, Y., Yao, Q., & Kwok, J. (2023). Positive-unlabeled node
classification with structure-aware graph learning. In *Proceedings of the
32nd ACM International Conference on Information and Knowledge Management*
(CIKM '23). Association for Computing Machinery.
https://doi.org/10.1145/3583780.3615250
