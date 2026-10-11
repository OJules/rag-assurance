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

**1. On a clean synthetic corpus, retrieval is not the bottleneck.**
On an annotated gold set (14 questions, 7 synthetic policies), retrieval reaches **Recall@5 ≈ 0.97**. The errors that remain have *perfect* retrieval - the model has the right clauses in front of it and still reasons wrong. So on this corpus the weak link is **generation and reasoning across documents**, not search. Point 4 shows this conclusion does not survive real contracts.

**2. Real PDFs break naive chunking - and the system should know it.**
Section-based chunking works on clean Markdown; a real PDF (columns, tables, page breaks) produces cut or interleaved passages. Rather than chasing *perfect* extraction, each chunk carries an **explainable quality score** (good / medium / poor) *and the reason why*. On the current corpus of 7 documents (813 chunks), the breakdown is **45% good, 45% medium, 10% poor**. The two real home-insurance contracts are the hardest: 39% / 43% / 17% and 33% / 58% / 9%.
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

**4. On real contracts, search breaks first, then reasoning.**
I ran two coverage questions on two real home-insurance general conditions (about 250 and 325 chunks): *a bicycle stolen from an unlocked shared room of the building*, and *a candle left on a garden table that melts in the heat and damages the table*. Here is what broke, in order.

- **Search.** The question says "shared room", the contract says "common areas". With embeddings alone, the deciding clause came at rank 4 in one contract and was missing from the top 8 in the other. And my sequential loop gave up after two passages without an answer, so it never read rank 4.
  *What I changed:* coverage questions now read 8 passages in one call, and search is **hybrid**: embeddings plus a word-based search (BM25 with French stemming), merged round-robin. On the bicycle question, the "simple theft" clause went from absent to rank 2, and no useful clause left the top 8.
  *What I tried and removed:* asking the LLM to break the question into insurance "facets" before searching. Two prompt versions, no stable gain in rank, about 1 300 to 1 700 extra tokens per question, and it invented a fire for the candle question. Measured, then deleted.
- **Reasoning.** Once the right clauses were read, the model concluded from a passage that merely *resembled* the question: a guarantee for theft *with violence* applied to a theft without violence, a clause about mopeds applied to a bicycle.
  *What I changed:* a dedicated coverage decision. The model must list the facts, give each excerpt a role (guarantee, exclusion, definition, condition), say whether each condition is met by the facts, and only then give one of four verdicts: covered, not covered, undetermined, contract not concerned.
  *A guard in code, not in the prompt:* a decisive verdict is downgraded to "undetermined" unless its evidence is found as **one continuous quote inside a single passage**, and that passage is not of "poor" extraction quality. It fired on a real case: the model had stitched two fragments together with ellipses, and the stitched quote was rejected.
- **What is not solved.** Two-column PDF extraction still interleaves sentences, and it is the root cause behind both problems above. For the candle question, none of the search variants finds a deciding clause; the system answers "undetermined" and shows what it read. I do not yet have verified ground truth for these two questions on the real contracts, and **everything in this point is measured on two questions**.

### Measured trade-offs

- **Per-contract reasoning**: eliminates cross-contract attribution errors, at **~2× the cost**. It also does not scale to hundreds of contracts: one LLM call per contract, and the free tier's rate limit is reached at three.
- **Routing** (sequential vs batch strategy): on the last run, the routed mode used **10 822 input tokens against 14 263 for all-batch (−24%)**, at equal accuracy (10 / 14). The accuracy gap between strategies is within run-to-run noise on 14 questions, so I no longer quote one.

### A note on trusting your own metrics

Three times now, the measurement was the thing that was wrong.

- While measuring contract confusion, my metric reported the system *failing* on large numbers. The system was right: the LLM wrote "80 000" with a **non-breaking space** (` `), while my ground truth compared against a regular space.
- The same invisible character came back in the main evaluation: "5 days" written with a non-breaking space was counted as wrong. The judge now neutralises typography before comparing.
- The model is **not reproducible at temperature 0**: the same code run twice in a row gave 10 / 14 then 8 / 14. With 14 questions, one question is 7 points, so I report "10 of 14" rather than a percentage. On the last run, two of the four failures are answers that are correct in substance but that the keyword judge does not recognise.

The lesson: **test your evaluation the way you test your system.**

### Architecture

```
Business question
      │
      ▼
Relevant contracts (document-level routing)
      │
      ▼  (per contract, isolated)
PDF → extraction → quality scoring → chunking
      │
      ▼
Hybrid search: meaning (E5) + words (BM25), 8 passages
      │
      ▼
Coverage decision: facts → clause-by-clause check → verdict
      │
      ▼
Guard: no decisive verdict without one continuous, verifiable quote
      │
      ▼
Decision  +  cited clause  +  extraction quality  +  what is missing
```

### What I would do next

- **Structured extraction into a table** (deductibles, ceilings, guarantees per contract), queried directly for value questions. This is the answer to "hundreds of contracts" that per-contract reasoning cannot give.
- **Layout-aware extraction**, the root cause of most remaining errors on real PDFs.
- **A larger annotated gold set**, including coverage scenarios on real contracts, to tell real progress from tuning on two questions.
- **Calibrated abstention**: replace "the LLM says it doesn't know" with a statistically guaranteed abstention rate (conformal risk control - the subject of my thesis).

### Tech

Python · sentence-transformers (multilingual-e5-small) · Chroma · BM25 (written in-house) · snowballstemmer · Groq · Streamlit · pdfplumber.
Retrieval and generation are isolated behind a single interface, which is why the same engine serves both the Markdown corpus and the PDF corpus, and why switching LLM provider touched a single file.

### Run it

```bash
pip install -r requirements.txt
# create a .env with your GROQ_API_KEY
python ingest.py            # build the Markdown index
python ingest_pdf.py        # build the PDF index (with quality scores)
python build_eval_cache.py  # optional: recompute the evaluation, prints each failure
streamlit run app.py
```

Public / synthetic data only. Documents containing personal data are excluded from the deployed corpus. Real general conditions are published by their insurers; the app links to the insurer's site rather than offering them for download.

---

## Version française

### L'idée

En assurance, une réponse d'IA n'a de valeur que si elle est **traçable, honnête sur son incertitude et fiable**. Donner la bonne réponse ne suffit pas : il faut montrer *sur quoi elle repose*, et admettre quand on ne peut pas conclure.

Ce projet est un assistant de questions-réponses sur des contrats d'assurance. Il va de la question métier jusqu'à une **décision exploitable** : sur quelles clauses il s'appuie, à quel point il est sûr, et — surtout — **quand il vaut mieux s'abstenir** que répondre sur une base fragile.

Injecter un contrat brut dans un LLM fonctionne sur un document et s'écroule ensuite. Ce projet porte sur *ce qu'on fait à la place*.

### L'histoire, pas la liste d'outils

L'intéressant n'est pas « E5 + Chroma + un LLM », mais la suite de problèmes trouvés et de décisions prises.

**1. Sur un corpus synthétique propre, le retrieval n'est pas le goulot.**
Sur un gold set annoté (14 questions, 7 polices synthétiques), le retrieval atteint **Recall@5 ≈ 0,97**. Les erreurs restantes ont un retrieval *parfait* : le modèle a les bonnes clauses sous les yeux et se trompe quand même. Sur ce corpus, le maillon faible est donc la **génération et le raisonnement multi-documents**, pas la recherche. Le point 4 montre que cette conclusion ne tient pas sur de vrais contrats.

**2. Les vrais PDF cassent le découpage naïf - et le système doit le savoir.**
Le découpage par section tient sur du Markdown propre ; un vrai PDF (colonnes, tableaux, sauts de page) produit des passages coupés ou entrelacés. Plutôt que de viser une extraction *parfaite*, chaque chunk porte un **score de qualité explicable** (bon / moyen / faible) *et la raison*. Sur le corpus actuel de 7 documents (813 chunks) : **45 % bon, 45 % moyen, 10 % faible**. Les deux vrais contrats habitation sont les plus difficiles : 39 % / 43 % / 17 % et 33 % / 58 % / 9 %.
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

**4. Sur de vrais contrats, la recherche casse en premier, puis le raisonnement.**
J'ai passé deux questions de couverture sur deux vraies conditions générales habitation (environ 250 et 325 chunks) : *un vélo volé dans le local commun de l'immeuble, qui n'était pas fermé à clé*, et *une bougie laissée sur une table de jardin, qui fond avec la chaleur et abîme la table*. Voici ce qui a cassé, dans l'ordre.

- **La recherche.** La question dit « local commun », le contrat dit « parties communes ». Avec les embeddings seuls, la clause qui décide arrivait au rang 4 dans un contrat et manquait dans les 8 premiers de l'autre. Et ma boucle séquentielle abandonnait après deux passages sans réponse : elle ne lisait jamais le rang 4.
  *Ce que j'ai changé :* les questions de couverture lisent maintenant 8 passages en un appel, et la recherche est **hybride** : embeddings plus recherche par mots (BM25 avec racinisation française), fusionnées en tour de table. Sur la question du vélo, la clause « vol simple » est passée d'absente au rang 2, et aucune clause utile n'est sortie des 8 premiers.
  *Ce que j'ai essayé puis retiré :* demander au LLM de décomposer la question en « facettes » d'assurance avant de chercher. Deux versions du prompt, aucun gain stable de rang, environ 1 300 à 1 700 tokens de plus par question, et un feu inventé pour la question de la bougie. Mesuré, puis supprimé.
- **Le raisonnement.** Une fois les bonnes clauses lues, le modèle concluait sur un passage qui *ressemblait* à la question : une garantie contre le vol *avec violence* appliquée à un vol sans violence, une clause sur les vélomoteurs appliquée à un vélo.
  *Ce que j'ai changé :* une décision de couverture dédiée. Le modèle doit lister les faits, donner à chaque extrait un rôle (garantie, exclusion, définition, condition), dire si chaque condition est remplie par les faits, et seulement ensuite rendre un verdict parmi quatre : couvert, non couvert, indéterminé, contrat non concerné.
  *Un garde-fou en code, pas dans le prompt :* un verdict tranché est ramené à « indéterminé » si sa preuve n'est pas retrouvée **d'un seul tenant dans un seul passage**, ou si ce passage est d'une qualité d'extraction « faible ». Il s'est déclenché sur un cas réel : le modèle avait recollé deux fragments avec des points de suspension, et la citation recollée a été refusée.
- **Ce qui n'est pas résolu.** L'extraction des PDF en deux colonnes entrelace encore des phrases, et c'est la cause de fond des deux problèmes précédents. Pour la bougie, aucune variante de recherche ne trouve de clause qui tranche ; le système répond « indéterminé » et montre ce qu'il a lu. Je n'ai pas encore de vérité terrain vérifiée pour ces deux questions sur les vrais contrats, et **tout ce point est mesuré sur deux questions**.

### Arbitrages mesurés

- **Raisonnement par contrat** : élimine les erreurs d'attribution entre contrats, au prix de **~2× le coût**. Il ne passe pas non plus à l'échelle de centaines de contrats : un appel au LLM par contrat, et la limite de débit du palier gratuit est atteinte dès trois.
- **Routage** (stratégie séquentielle vs batch) : au dernier calcul, le mode routé a consommé **10 822 tokens en entrée contre 14 263 pour le tout-batch (−24 %)**, à exactitude égale (10 / 14). L'écart d'exactitude entre stratégies est dans le bruit d'un calcul à l'autre sur 14 questions : je n'en cite donc plus.

### Ne pas croire aveuglément ses métriques

Trois fois déjà, c'est la mesure qui avait tort.

- En mesurant la confusion entre contrats, ma métrique signalait le système comme *défaillant* sur les grands nombres. Le système avait raison : le LLM écrivait « 80 000 » avec une **espace insécable** (` `), là où ma vérité terrain comparait avec une espace normale.
- Le même caractère invisible est revenu dans l'évaluation principale : « 5 jours » écrit avec une espace insécable était compté faux. Le juge neutralise maintenant la typographie avant de comparer.
- Le modèle **n'est pas reproductible à température 0** : le même code lancé deux fois de suite a donné 10 / 14 puis 8 / 14. Avec 14 questions, une question pèse 7 points : j'écris donc « 10 sur 14 » plutôt qu'un pourcentage. Au dernier calcul, deux des quatre échecs sont des réponses correctes sur le fond que le juge par mots-clés ne reconnaît pas.

La leçon : **tester son évaluation comme on teste son système.**

### Architecture

```
Question métier
      │
      ▼
Contrats pertinents (routage au niveau document)
      │
      ▼  (par contrat, isolé)
PDF → extraction → score qualité → découpage
      │
      ▼
Recherche hybride : le sens (E5) + les mots (BM25), 8 passages
      │
      ▼
Décision de couverture : faits → confrontation clause par clause → verdict
      │
      ▼
Garde-fou : pas de verdict tranché sans citation continue et vérifiable
      │
      ▼
Décision  +  clause citée  +  qualité d'extraction  +  ce qu'il manque
```

### Ce que je ferais ensuite

- **Extraction structurée vers une table** (franchises, plafonds, garanties par contrat), interrogée directement pour les questions de valeur. C'est la réponse aux « centaines de contrats » que le raisonnement par contrat ne peut pas donner.
- **Extraction sensible à la mise en page**, cause de fond de la plupart des erreurs restantes sur les vrais PDF.
- **Un gold set annoté plus large**, avec des scénarios de couverture sur de vrais contrats, pour distinguer un vrai progrès d'un réglage sur deux questions.
- **Abstention calibrée** : remplacer « le LLM dit qu'il ne sait pas » par un taux d'abstention à garantie statistique (contrôle de risque conforme - le sujet de mon mémoire).

### Technique

Python · sentence-transformers (multilingual-e5-small) · Chroma · BM25 (écrit à la main) · snowballstemmer · Groq · Streamlit · pdfplumber.
Le retrieval et la génération sont isolés derrière une interface unique : c'est pourquoi le même moteur sert le corpus Markdown et le corpus PDF, et pourquoi changer de fournisseur de LLM n'a touché qu'un seul fichier.

### Lancer le projet

```bash
pip install -r requirements.txt
# créer un .env avec votre GROQ_API_KEY
python ingest.py            # index Markdown
python ingest_pdf.py        # index PDF (avec scores de qualité)
python build_eval_cache.py  # optionnel : recalcule l'évaluation et affiche chaque échec
streamlit run app.py
```

Données publiques ou synthétiques uniquement. Les documents contenant des données personnelles sont exclus du corpus déployé. Les vraies conditions générales sont publiées par leurs assureurs : l'app renvoie vers le site de l'assureur plutôt que de les proposer en téléchargement.

---

*Auteur : Jules Odjé - [github.com/OJules](https://github.com/OJules)*