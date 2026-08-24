"""
Interface Streamlit - RAG documentaire d'assurance (démo).
Onglet 1 : chatbot en direct (routage + garde-fous).
Onglet 2 : tableau de bord d'évaluation (résultats pré-calculés).

Déploiement : la clé Mistral vient de st.secrets (cloud) ou de .env (local).
L'index Chroma est reconstruit au 1er démarrage s'il est absent (le serveur ne l'a pas).
"""
import os
from pathlib import Path
import json

import streamlit as st

st.set_page_config(page_title="RAG Assurance — démo", page_icon="📄", layout="wide")

# --- Pont clé API : Streamlit Cloud fournit les secrets, pas un .env ---
try:
    if "MISTRAL_API_KEY" in st.secrets:
        os.environ["MISTRAL_API_KEY"] = st.secrets["MISTRAL_API_KEY"]
except Exception:
    pass  # en local, llm.py lira le .env

# --- DEBUG TEMPORAIRE (à retirer avant l'envoi à Kezhan) ---
_k = os.environ.get("MISTRAL_API_KEY", "")
st.sidebar.write("DEBUG longueur:", len(_k))
st.sidebar.write("DEBUG début/fin:", (_k[:4] + "…" + _k[-4:]) if _k else "VIDE")
try:
    st.sidebar.write("DEBUG secret présent:", "MISTRAL_API_KEY" in st.secrets)
except Exception:
    st.sidebar.write("DEBUG secret présent: (pas de st.secrets)")

# --- Bootstrap : construit l'index si absent (1er démarrage sur le serveur) ---
@st.cache_resource(show_spinner="Initialisation du moteur (1er démarrage)…")
def get_engine():
    if not Path("chroma_db").exists():
        import ingest
        ingest.main()
    from router import answer_routed
    return answer_routed

answer_routed = get_engine()

# ===========================================================================
st.title("📄 Assistant documentaire — contrats d'assurance")
st.caption("Démo RAG sur un corpus synthétique de 7 polices. Retrieval E5 + Chroma, "
           "génération Mistral avec answer contract, routage déterministe.")

tab_chat, tab_eval = st.tabs(["💬 Chatbot", "📊 Évaluation"])

# ---------------------------------------------------------------- Onglet chatbot
with tab_chat:
    st.markdown("Posez une question sur les contrats. Le système cite sa source, "
                "et **s'abstient** si l'information n'est pas dans les documents.")
    exemples = [
        "Quelle est la date d'effet du contrat AUTO-2024-0137 ?",
        "Quelles sont toutes les exclusions applicables au vol du contrat AUTO-2024-0137 ?",
        "Suis-je couvert si je fais du VTC avec ma voiture ?",
    ]
    with st.expander("Exemples de questions"):
        for e in exemples:
            st.markdown(f"- {e}")

    q = st.text_input("Votre question", placeholder="Ex. : Quelle est la franchise vol de ma moto ?")
    if st.button("Interroger", type="primary") and q.strip():
        with st.spinner("Recherche et génération…"):
            r = answer_routed(q, k=5, verbose=False)

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
        "- **Erreur de raisonnement** (comparaison de montants entre documents) : le LLM se trompe malgré un contexte complet — "
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