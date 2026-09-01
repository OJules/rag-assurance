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

tab_chat, tab_eval, tab_pdf = st.tabs(["💬 Chatbot (markdown)", "📊 Évaluation", "📄 Chatbot (PDF réels)"])

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
    st.markdown("### Le même système, sur de vrais PDF")
    st.caption("Corpus : 2 polices synthétiques converties en PDF + 3 vrais documents d'assureurs "
               "(Allianz, Virgin Money). La réponse est générée comme dans l'onglet 1, mais chaque "
               "**source affiche sa qualité d'extraction** — le système montre quand il s'appuie sur "
               "un texte mal reconstruit par le PDF, au lieu de le masquer.")

    with st.expander("Pourquoi ce contrôle qualité ?"):
        st.markdown(
            "Un PDF n'est pas du texte structuré : colonnes, tableaux et sauts de page produisent "
            "des passages coupés ou mélangés à l'extraction. Un score par passage (🟢 bon, 🟠 moyen, "
            "🔴 faible) signale ces dégradations **et pourquoi**. Limite connue : le score attrape les "
            "coupures et fragments, pas encore l'entrelacement de colonnes — dans ce cas, c'est "
            "l'abstention du LLM qui sert de second filet."
        )

    answer_routed_pdf, rag_pdf = get_engine_pdf()

    # --- sélecteur de document : on ne mélange pas les contrats ---
    docs = ["Tous les documents"] + rag_pdf.documents_disponibles()
    choix = st.selectbox("Document à interroger", docs, key="pdf_doc",
                         help="Filtrer la recherche sur un seul contrat évite de mélanger "
                              "les sources de plusieurs documents (clé du passage à l'échelle).")
    filtre = None if choix == "Tous les documents" else choix

    # search_fn pré-configurée avec le filtre choisi
    def pdf_search(question, k=5):
        return rag_pdf.search(question, k=k, filtre_fichier=filtre)

    qp = st.text_input("Votre question", key="pdf_q",
                       placeholder="Ex. : Que faire en cas de sinistre à l'étranger ?")
    if st.button("Interroger les PDF", type="primary", key="pdf_btn") and qp.strip():
        with st.spinner("Recherche et génération sur les PDF…"):
            r = answer_routed_pdf(qp, k=5, verbose=False, search_fn=pdf_search)

        # --- la réponse ---
        if r.get("answer_found"):
            st.success(r["answer"])
        else:
            st.warning("🤚 " + r["answer"] + "\n\n*(Le système s'abstient : l'information n'est pas "
                       "présente, ou les passages sont trop dégradés pour conclure.)*")

        # --- qualité des sources utilisées ---
        hits = r.get("_hits", [])
        n_degrade = sum(1 for h in hits if h.get("quality") in ("moyen", "faible"))
        if n_degrade:
            st.warning(f"⚠️ {n_degrade} des {len(hits)} sources utilisées proviennent d'une "
                       "extraction imparfaite. La réponse est à vérifier dans le document original.")

        st.markdown("#### Sources utilisées")
        for h in hits:
            qy = h.get("quality", "?")
            with st.container(border=True):
                st.markdown(f"{FLAG.get(qy, '?')} **{h.get('source_file', '?')}** · "
                            f"similarité {h.get('similarity', 0):.2f} · qualité {LABEL.get(qy, qy)}")
                if h.get("titre"):
                    st.caption(f"Section détectée : {h['titre']}")
                # retire le préfixe d'identité [.. — ..] pour l'affichage
                texte = h.get("text", "")
                if texte.startswith("["):
                    coupe = texte.find("]\n")
                    if coupe != -1:
                        texte = texte[coupe + 2:]
                st.write(texte[:600] + ("…" if len(texte) > 600 else ""))
                if qy != "bon":
                    st.caption(f"⚠️ Qualité {LABEL.get(qy, qy)} : {h.get('quality_raisons', '')}")
                pdf_path = Path("pdfs") / h.get("source_file", "")
                if pdf_path.exists():
                    with open(pdf_path, "rb") as f:
                        st.download_button("📄 Ouvrir le PDF source", f.read(),
                                           file_name=h["source_file"], mime="application/pdf",
                                           key=f"dl_{h.get('rank')}_{h.get('source_file')}")

        rt = r.get("_routing", {})
        c1, c2, c3 = st.columns(3)
        c1.metric("Type détecté", rt.get("type_detecte", "—"))
        c2.metric("Stratégie", rt.get("strategie", "—"))
        c3.metric("Tokens (entrée)", r.get("_total_input_tokens", "—"))