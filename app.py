"""
Interface Streamlit - RAG documentaire d'assurance (démo).
Onglet 1 : chatbot sur le corpus MARKDOWN (7 polices synthétiques propres).
Onglet 2 : tableau de bord d'évaluation (résultats pré-calculés).
Onglet 3 : chatbot sur le corpus PDF (formats réels) + qualité des sources.

La même chaîne de génération sert les deux corpus (search_fn injectable).
Déploiement : clé API depuis st.secrets (cloud) ou .env (local).
Index Chroma reconstruits au 1er démarrage s'ils sont absents.
"""
import os
from pathlib import Path
import json

import streamlit as st

st.set_page_config(page_title="RAG Assurance — démo", page_icon="📄", layout="wide")

# --- Pont clé API : Streamlit Cloud fournit les secrets, pas un .env ---
try:
    for _k in ("GROQ_API_KEY", "MISTRAL_API_KEY"):
        if _k in st.secrets:
            os.environ[_k] = st.secrets[_k]
except Exception:
    pass


# --- Bootstrap moteur markdown (démo d'origine) ---
@st.cache_resource(show_spinner="Initialisation du moteur (1er démarrage)…")
def get_engine_md():
    if not Path("chroma_db").exists():
        import ingest
        ingest.main()
    from router import answer_routed
    return answer_routed


# --- Bootstrap moteur PDF : index PDF + search_fn dédiée ---
@st.cache_resource(show_spinner="Initialisation du moteur PDF…")
def get_engine_pdf():
    if not Path("chroma_db_pdf").exists():
        import ingest_pdf
        ingest_pdf.main()
    import rag_pdf
    from router import answer_routed
    return answer_routed, rag_pdf


answer_routed_md = get_engine_md()

FLAG = {"bon": "🟢", "moyen": "🟠", "faible": "🔴"}
LABEL = {"bon": "bonne", "moyen": "moyenne", "faible": "faible"}

# ===========================================================================
st.title("📄 Assistant documentaire - contrats d'assurance")
st.caption("Démo RAG. Retrieval E5 + Chroma, génération avec answer contract, routage déterministe. "
           "Le même moteur répond sur un corpus markdown propre (onglet 1) et sur de vrais PDF (onglet 3).")

tab_chat, tab_eval, tab_pdf = st.tabs(["💬 Chatbot (markdown)", "📊 Évaluation", "📄 Aide à la décision (PDF)"])

# ---------------------------------------------------------------- Onglet chatbot markdown
with tab_chat:
    st.markdown("Corpus : 7 polices synthétiques (markdown propre). Le système cite sa source "
                "et **s'abstient** si l'information n'est pas dans les documents.")
    exemples = [
        "Quelle est la date d'effet du contrat AUTO-2024-0137 ?",
        "Quelles sont toutes les exclusions applicables au vol du contrat AUTO-2024-0137 ?",
        "Suis-je couvert si je fais du VTC avec ma voiture ?",
    ]
    with st.expander("Exemples de questions"):
        for e in exemples:
            st.markdown(f"- {e}")

    q = st.text_input("Votre question", key="md_q",
                      placeholder="Ex. : Quelle est la franchise vol de ma moto ?")
    if st.button("Interroger", type="primary", key="md_btn") and q.strip():
        with st.spinner("Recherche et génération…"):
            r = answer_routed_md(q, k=5, verbose=False)

        if r.get("answer_found"):
            st.success(r["answer"])
            if r.get("evidence"):
                st.markdown("**Source citée :**")
                st.info(r["evidence"])
        else:
            st.warning("🤚 " + r["answer"] + "\n\n*(Le système s'abstient : l'information "
                       "n'est pas présente dans les documents.)*")

        rt = r.get("_routing", {})
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Type détecté", rt.get("type_detecte", "—"))
        c2.metric("Stratégie", rt.get("strategie", "—"))
        c3.metric("Tokens (entrée)", r.get("_total_input_tokens", "—"))
        c4.metric("Complète", "oui" if r.get("complete_answer_found") else "non")
        if rt.get("fallback"):
            st.caption("↩️ Fallback batch déclenché (le séquentiel n'a pas conclu).")

# ------------------------------------------------------------- Onglet évaluation
with tab_eval:
    st.markdown("### Comment ce système se comporte-t-il ?")
    st.caption("Résultats pré-calculés sur un gold set de 14 questions annotées "
               "(petit échantillon : illustration de mécanisme, pas benchmark statistique).")

    try:
        ev = json.load(open("eval_results.json", encoding="utf-8"))
    except FileNotFoundError:
        st.error("eval_results.json manquant — lancez `python build_eval_cache.py`.")
        st.stop()

    st.markdown("#### Comparaison des 3 stratégies")
    st.caption("À exactitude comparable, le routage réduit le coût — mais c'est un arbitrage, pas un gain gratuit.")
    strat_rows = [{"Stratégie": k, "Exactitude": f"{v['accuracy']:.0%}", "Tokens (entrée) totaux": v["tokens"]}
                  for k, v in ev["strategies"].items()]
    st.table(strat_rows)

    st.markdown("#### Par type de question (mode routé)")
    st.caption(f"Recall@{ev['k']} = le retrieval a-t-il ramené les bonnes clauses ; "
               "Exactitude = la réponse finale est-elle correcte.")
    type_rows = [{"Type": t,
                  "Recall@k": "—" if v["recall"] is None else f"{v['recall']:.2f}",
                  "Exactitude": "—" if v["accuracy"] is None else f"{v['accuracy']:.0%}"}
                 for t, v in ev["by_type"].items()]
    st.table(type_rows)

    st.markdown("#### Ce qui casse, et pourquoi")
    st.markdown(
        "- **Le retrieval n'est pas le goulot** : Recall@5 ≈ 0.97. Les erreurs ont un retrieval parfait.\n"
        "- **Erreurs de stratégie** (désambiguïsation, contradiction) : récupérables en affinant la classification.\n"
        "- **Erreur de raisonnement** (comparaison de montants entre documents) : le LLM se trompe malgré un contexte complet - "
        "limite de la génération, pas du pipeline.\n"
        "- **L'abstention fonctionne** : sur les questions sans réponse, le système se tait au lieu d'inventer."
    )

    with st.expander("Détail par question"):
        rows = [{"id": p["id"], "type": p["type"],
                 "recall": "—" if p["recall"] is None else f"{p['recall']:.2f}",
                 "ok": "✅" if p["ok"] else ("🤚" if p["ok"] is None else "❌"),
                 "question": p["question"]}
                for p in ev["per_question"]]
        st.table(rows)

# ---------------------------------------------------------------- Onglet chatbot PDF
with tab_pdf:
    st.markdown("### Aide à la décision sur documents réels")
    st.caption("Vue orientée gestionnaire : décision, justification, clauses citées, incertitude. "
               "En mode « tous les contrats », le système raisonne CONTRAT PAR CONTRAT (jamais de "
               "mélange) et présente une décision par contrat.")

    answer_routed_pdf, rag_pdf = get_engine_pdf()

    docs = ["Tous les contrats"] + rag_pdf.documents_disponibles()
    choix = st.selectbox("Contrat à interroger", docs, key="pdf_doc",
                         help="Un contrat précis = une décision. « Tous » = une décision par "
                              "contrat pertinent, sans jamais mélanger leurs clauses.")
    mode_tous = (choix == "Tous les contrats")
    if mode_tous:
        n_docs = st.slider("Nombre de contrats à analyser", 1, 5, 3, key="pdf_n",
                           help="Le système isole le raisonnement par contrat : chaque contrat "
                                "analysé = un appel séparé (plus juste, mais plus coûteux).")

    qp = st.text_input("Question du gestionnaire", key="pdf_q",
                       placeholder="Ex. : Quelle est la franchise en cas de bris de glace ?")

    def _search_pour(fichier):
        return lambda question, k=5: rag_pdf.search(question, k=k, filtre_fichier=fichier)

    def _etat(found, hits):
        n_deg = sum(1 for h in hits if h.get("quality") in ("moyen", "faible"))
        if not found:
            return "🔴", "Information insuffisante", n_deg
        if n_deg:
            return "🟠", "Réponse à vérifier", n_deg
        return "🟢", "Réponse trouvée", n_deg

    def _carte_contrat(fichier, r):
        """Affiche une carte de décision pour un contrat."""
        hits = r.get("_hits", [])
        found = r.get("answer_found", False)
        couleur, libelle, n_deg = _etat(found, hits)
        with st.container(border=True):
            st.markdown(f"### {couleur} {fichier}")
            st.caption(libelle)
            if found:
                st.markdown(f"**{r['answer']}**")
            else:
                st.markdown(f"*{r['answer']}*")
            if found and n_deg:
                raisons = {h.get("quality_raisons", "") for h in hits
                           if h.get("quality") in ("moyen", "faible") and h.get("quality_raisons")}
                st.warning("⚠️ À vérifier : " + " ; ".join(sorted(raisons)))
            # clauses citées
            with st.expander("Clauses citées"):
                for h in hits:
                    qy = h.get("quality", "?")
                    titre = f"Section {h['titre']}" if h.get("titre") else "Passage"
                    st.markdown(f"{FLAG.get(qy,'?')} **{h.get('source_file','?')}** — {titre}")
                    texte = h.get("text", "")
                    if texte.startswith("["):
                        c = texte.find("]\n")
                        if c != -1:
                            texte = texte[c + 2:]
                    st.write(texte[:400] + ("…" if len(texte) > 400 else ""))
                    if qy != "bon":
                        st.caption(f"⚠️ Extraction {LABEL.get(qy, qy)} : {h.get('quality_raisons','')}")
            st.caption(f"🔧 {r.get('_routing',{}).get('strategie','—')} · "
                       f"{r.get('_total_input_tokens','—')} tokens (entrée)")

    if st.button("Analyser", type="primary", key="pdf_btn") and qp.strip():
        if not mode_tous:
            # --- un seul contrat ---
            with st.spinner("Analyse…"):
                r = answer_routed_pdf(qp, k=5, verbose=False, search_fn=_search_pour(choix))
            st.markdown("---")
            _carte_contrat(choix, r)
        else:
            # --- per-contract : on isole le raisonnement par contrat ---
            with st.spinner("Identification des contrats pertinents…"):
                contrats = rag_pdf.documents_pertinents(qp, n_docs=n_docs)
            st.markdown("---")
            st.caption(f"Contrats analysés séparément : {', '.join(contrats)}")
            for fichier in contrats:
                with st.spinner(f"Analyse de {fichier}…"):
                    r = answer_routed_pdf(qp, k=5, verbose=False, search_fn=_search_pour(fichier))
                _carte_contrat(fichier, r)
            st.info("ℹ️ Chaque contrat est analysé indépendamment : leurs clauses ne sont jamais "
                    "mélangées. C'est ce qui évite d'attribuer une information au mauvais contrat.")