# RAG Assurance - un système qui sait quand il a mal lu

**Démo en ligne : [rag-assurance.streamlit.app](https://rag-assurance.streamlit.app/)**

> Ce n'est pas un RAG qui lit des PDF.
> C'est un système qui **sait quand il a mal lu**, qui **le dit**, qui **raisonne contrat par contrat** pour ne pas confondre, et qui **mesure sa propre fiabilité**.

*[English version below](#english-version) · [Version française](#version-française)*

---

## English version

### The idea

In insurance, an AI answer only has value if it is **traceable, honest about its uncertainty, and reliable**. Giving the right answer is not enough - you have to show *what it rests on*, and admit when you cannot conclude.

This project is a document-QA assistant for insurance contracts. It goes from a business question all the way to a **usable decision**: which clauses it relies on, how confident it is, and - crucially - **when it should abstain** rather than answer on a shaky basis.

Injecting one raw contract into an LLM works for one document and collapses beyond that. This project is about *what you do instead*.

### The story, not the tech stack

The interesting part of this project is not "E5 + Chroma + an LLM". It is the sequence of problems found and decisions made.

**1. Retrieval is not the bottleneck.**
On an annotated gold set (14 questions), retrieval reaches **Recall@5 ≈ 0.97**. The errors that remain have *perfect* retrieval - the model has the right clauses in front of it and still reasons wrong. So the weak link is **generation and reasoning across documents**, not search.

**2. Real PDFs break naive chunking - and the system should know it.**
Section-based chunking works on clean Markdown; a real PDF (columns, tables, page breaks) produces cut or interleaved passages. Rather than chasing *perfect* extraction, each chunk carries an **explainable quality score** (good / medium / poor) *and the reason why*. On a mixed corpus of 5 documents (237 chunks), the breakdown is **64% good, 34% medium, 2% poor**.
Honest limits: the score catches cut sentences and fragments, but **not yet** column interleaving (Allianz: *"14 Morocco days"*) nor character-by-character spacing (Virgin Money: *"A m T ru s t"*). When text is too degraded, the LLM's **abstention** acts as a second safety net.

**3. Contract confusion - the core experiment.**
When several contracts share an attribute with *different values* (theft deductible: 750 CHF in one, 500 CHF in another), a RAG reasoning over a *mixed* context answers "750 CHF" without saying which contract - or abstains, unable to decide. Useless for a case handler.
So I changed the **unit of reasoning**: identify the relevant contracts, then retrieve and reason **inside each one separately**, never mixing their clauses. Each contract gets its own decision, evidence and uncertainty.
I built **6 trap questions** (attributes shared across contracts with different values) and measured, on the synthetic corpus where ground truth is fully known:

| | Global RAG | Per-contract RAG |
|---|---|---|
| Correctly attributed answers | **0 / 11** | **11 / 11** |
| LLM calls | 6 | 12 |
| Input tokens | 2 711 | 5 322 |

The global mode never produced a correctly attributed answer (incomplete answers or wrong abstentions); per-contract reasoning recovered the right value every time — at **~2× the cost**. A measured cost/reliability trade-off, not an assumption.

> Attribution is *guaranteed by construction* in per-contract mode (the search is filtered to one contract), so this measures value correctness, not attribution - the claim is "contracts are never mixed during retrieval and interpretation", not "the model can't confuse them".

### Measured trade-offs

Two trade-offs the demo lets you *measure* rather than guess:

- **Routing** (sequential vs batch strategy): **−39% tokens for −7 points of accuracy**. Routing is a trade-off, not a free win.
- **Per-contract reasoning**: eliminates cross-contract attribution errors, at **~2× the cost**.

### A note on trusting your own metrics

While measuring contract confusion, my automated metric reported the system *failing* on large numbers. The system was actually right: the LLM wrote "80 000" with a **non-breaking space** (`\u00a0`), while my ground truth compared against a regular space — invisible to the eye, a mismatch for the machine. I normalised separators before comparing.
The lesson: **test your evaluation the way you test your system.** A metric can lie because of one invisible character.

### Architecture

```
Business question
      │
      ▼
Relevant contracts (document-level routing)
      │
      ▼  (per contract, isolated)
PDF → extraction → quality scoring → chunking → retrieval → reasoning
      │
      ▼
Decision  +  cited clauses  +  extraction quality  +  uncertainty
      │
      ▼
Business-facing interface (technical view folded away)
```

### What I would do next

- **Structured-evidence comparison**: extract facts to a structured form before comparing across contracts, instead of asking the LLM to compare raw passages.
- **Layout-aware extraction** to catch column interleaving and spacing artefacts the current score misses.
- **A larger annotated gold set** (100–200 questions) to move from illustrative to statistically meaningful.
- **Calibrated abstention**: replace "the LLM says it doesn't know" with a statistically guaranteed abstention rate (conformal risk control - the subject of my thesis).

### Tech

Python · sentence-transformers (multilingual-e5-small) · Chroma · Groq · Streamlit · pdfplumber.
Retrieval and generation are isolated behind a single interface, which is why the same engine serves both the Markdown corpus and the PDF corpus, and why switching LLM provider touched a single file.

### Run it

```bash
pip install -r requirements.txt
# create a .env with your GROQ_API_KEY
python ingest.py          # build the Markdown index
python ingest_pdf.py      # build the PDF index (with quality scores)
streamlit run app.py
```

Public / synthetic data only. Documents containing personal data are excluded from the deployed corpus.

---

## Version française

### L'idée

En assurance, une réponse d'IA n'a de valeur que si elle est **traçable, honnête sur son incertitude et fiable**. Donner la bonne réponse ne suffit pas : il faut montrer *sur quoi elle repose*, et admettre quand on ne peut pas conclure.

Ce projet est un assistant de questions-réponses sur des contrats d'assurance. Il va de la question métier jusqu'à une **décision exploitable** : sur quelles clauses il s'appuie, à quel point il est sûr, et — surtout — **quand il vaut mieux s'abstenir** que répondre sur une base fragile.

Injecter un contrat brut dans un LLM fonctionne sur un document et s'écroule ensuite. Ce projet porte sur *ce qu'on fait à la place*.

### L'histoire, pas la liste d'outils

L'intéressant n'est pas « E5 + Chroma + un LLM », mais la suite de problèmes trouvés et de décisions prises.

**1. Le retrieval n'est pas le goulot.**
Sur un gold set annoté (14 questions), le retrieval atteint **Recall@5 ≈ 0,97**. Les erreurs restantes ont un retrieval *parfait* : le modèle a les bonnes clauses sous les yeux et se trompe quand même. Le maillon faible est donc la **génération et le raisonnement multi-documents**, pas la recherche.

**2. Les vrais PDF cassent le découpage naïf - et le système doit le savoir.**
Le découpage par section tient sur du Markdown propre ; un vrai PDF (colonnes, tableaux, sauts de page) produit des passages coupés ou entrelacés. Plutôt que de viser une extraction *parfaite*, chaque chunk porte un **score de qualité explicable** (bon / moyen / faible) *et la raison*. Sur un corpus mixte de 5 documents (237 chunks) : **64 % bon, 34 % moyen, 2 % faible**.
Limites assumées : le score attrape les phrases coupées et les fragments, mais **pas encore** l'entrelacement de colonnes (Allianz : *« 14 Maroc jours »*) ni l'espacement caractère par caractère (Virgin Money : *« A m T ru s t »*). Quand le texte est trop dégradé, l'**abstention** du LLM sert de second filet.

**3. La confusion entre contrats - l'expérience centrale.**
Quand plusieurs contrats partagent un attribut avec des *valeurs différentes* (franchise vol : 750 CHF dans l'un, 500 CHF dans l'autre), un RAG qui raisonne sur un contexte *mélangé* répond « 750 CHF » sans dire quel contrat — ou s'abstient, faute de trancher. Inexploitable pour un gestionnaire.
J'ai donc changé l'**unité de raisonnement** : identifier les contrats pertinents, puis faire le retrieval et le raisonnement **dans chacun séparément**, sans jamais mélanger leurs clauses. Chaque contrat reçoit sa propre décision, ses preuves, son incertitude.
J'ai construit **6 questions-pièges** (attributs partagés entre contrats avec des valeurs différentes) et mesuré, sur le corpus synthétique où la vérité terrain est entièrement connue :

| | RAG global | RAG par contrat |
|---|---|---|
| Réponses correctement attribuées | **0 / 11** | **11 / 11** |
| Appels au LLM | 6 | 12 |
| Tokens (entrée) | 2 711 | 5 322 |

Le mode global n'a produit aucune réponse correctement attribuée (réponses incomplètes ou abstentions à tort) ; le raisonnement par contrat retrouve la bonne valeur à chaque fois - au prix de **~2× le coût**. Un arbitrage coût/fiabilité mesuré, pas supposé.

> L'attribution est *garantie par construction* en mode par contrat (la recherche est filtrée sur un seul contrat) : on mesure donc la justesse de la valeur, pas l'attribution - l'affirmation est « les contrats ne sont jamais mélangés pendant le retrieval et l'interprétation », pas « le modèle ne peut pas les confondre ».

### Arbitrages mesurés

Deux compromis que la démo permet de *mesurer* plutôt que de supposer :

- **Routage** (stratégie séquentielle vs batch) : **−39 % de tokens pour −7 points d'exactitude**. Le routage est un arbitrage, pas un gain gratuit.
- **Raisonnement par contrat** : élimine les erreurs d'attribution entre contrats, au prix de **~2× le coût**.

### Ne pas croire aveuglément ses métriques

En mesurant la confusion entre contrats, ma métrique automatique signalait le système comme *défaillant* sur les grands nombres. Le système avait raison : le LLM écrivait « 80 000 » avec une **espace insécable** (`\u00a0`), là où ma vérité terrain comparait avec une espace normale - invisible à l'œil, un écart pour la machine. J'ai normalisé les séparateurs avant comparaison.
La leçon : **tester son évaluation comme on teste son système.** Une métrique peut mentir à cause d'un seul caractère invisible.

### Architecture

```
Question métier
      │
      ▼
Contrats pertinents (routage au niveau document)
      │
      ▼  (par contrat, isolé)
PDF → extraction → score qualité → découpage → retrieval → raisonnement
      │
      ▼
Décision  +  clauses citées  +  qualité d'extraction  +  incertitude
      │
      ▼
Interface métier (vue technique repliée)
```

### Ce que je ferais ensuite

- **Comparaison sur faits structurés** : extraire les faits dans une forme structurée avant de comparer entre contrats, au lieu de demander au LLM de comparer des passages bruts.
- **Extraction sensible à la mise en page** pour attraper l'entrelacement de colonnes et l'espacement que le score actuel rate.
- **Un gold set annoté plus large** (100–200 questions) pour passer de l'illustratif au statistiquement significatif.
- **Abstention calibrée** : remplacer « le LLM dit qu'il ne sait pas » par un taux d'abstention à garantie statistique (contrôle de risque conforme - le sujet de mon mémoire).

### Technique

Python · sentence-transformers (multilingual-e5-small) · Chroma · Groq · Streamlit · pdfplumber.
Le retrieval et la génération sont isolés derrière une interface unique : c'est pourquoi le même moteur sert le corpus Markdown et le corpus PDF, et pourquoi changer de fournisseur de LLM n'a touché qu'un seul fichier.

### Lancer le projet

```bash
pip install -r requirements.txt
# créer un .env avec votre GROQ_API_KEY
python ingest.py          # index Markdown
python ingest_pdf.py      # index PDF (avec scores de qualité)
streamlit run app.py
```

Données publiques ou synthétiques uniquement. Les documents contenant des données personnelles sont exclus du corpus déployé.

---

*Auteur : Jules Odjé - [github.com/OJules](https://github.com/OJules)*