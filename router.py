"""
Routage déterministe par règles + garde-fou d'exécution.

Pipeline : classify (règles) -> stratégie -> exécution -> validation -> fallback borné -> logging

Asymétrie du risque, mesurée sur le gold set :
  - comparaison/liste envoyée à tort en séquentiel -> CASSE (dangereux)
  - lookup envoyé à tort en batch -> juste un peu plus cher (bénin)
Donc : en cas de doute on penche batch, et le fallback ne va JAMAIS de batch vers séquentiel.

Type 'couverture' (« suis-je couvert ? », « est-ce pris en charge ? ») -> batch.
Une décision de couverture se lit sur PLUSIEURS clauses à la fois : la garantie,
ses conditions, et l'exclusion qui peut l'annuler. En séquentiel, le modèle peut
conclure sur la clause de garantie sans jamais lire l'exclusion placée plus loin.
C'est le même risque que pour une liste : s'arrêter trop tôt.
Quand la recherche est limitée à UN contrat (un_seul_contrat=True, onglet PDF), il
utilise aussi un contrat de réponse dédié (llm.generate_couverture) : faits,
confrontation clause par clause, verdict. Ce contrat suppose que tous les extraits
viennent du même contrat : sur un corpus où les contrats sont mélangés (onglet
markdown, jeu d'évaluation), on garde le contrat de réponse général.

SOURCE DE RECHERCHE INJECTABLE (search_fn) : transmise telle quelle à chatbot.
Par défaut = corpus markdown ; passer rag_pdf.search pour interroger le corpus PDF.
"""
import re

from chatbot import answer_batch, answer_sequential
from llm import generate_couverture

RE_LISTE       = re.compile(r"\b(toutes|tous|lister?|liste|l'ensemble|quelles sont)\b", re.I)
RE_COMPARAISON = re.compile(r"\b(le plus|la plus|lequel|laquelle|comparer?|entre|différence|versus|vs)\b", re.I)
# Question de couverture : on confronte une situation à un contrat.
# 'couverture' (le nom) n'est volontairement PAS capté : « quelle est la couverture
# bris de glace ? » reste une recherche de valeur.
# De même pour 'remboursement' : « quel est le plafond de remboursement optique ? » est
# une recherche de valeur. Seules les formes verbales (« serai-je remboursé ? ») comptent.
RE_COUVERTURE  = re.compile(
    r"\b(couverte?s?|prise?s? en charge|indemnisée?s?|remboursée?s?|dédommagée?s?)\b", re.I)


def classify(question: str) -> str:
    """Renvoie 'comparaison' | 'liste' | 'couverture' | 'lookup' (défaut)."""
    if RE_COMPARAISON.search(question):
        return "comparaison"
    if RE_LISTE.search(question):
        return "liste"
    if RE_COUVERTURE.search(question):
        return "couverture"
    return "lookup"


STRATEGY = {"lookup": "sequential", "liste": "batch", "comparaison": "batch",
            "couverture": "batch"}


def answer_routed(question: str, k: int = 5, verbose: bool = True, search_fn=None,
                  un_seul_contrat: bool = False) -> dict:
    qtype = classify(question)
    strategy = STRATEGY[qtype]
    log = {"question": question, "type_detecte": qtype, "strategie": strategy, "fallback": False}

    if strategy == "batch":
        # couverture : même lecture en batch, mais avec le contrat de réponse dédié
        gen = generate_couverture if (qtype == "couverture" and un_seul_contrat) else None
        result = answer_batch(question, k=k, search_fn=search_fn, generate_fn=gen)
    else:
        result = answer_sequential(question, k_max=k, search_fn=search_fn)
        if result.get("_stop_reason") == "épuisement" and result.get("answer_found"):
            log["fallback"] = True
            log["fallback_raison"] = "séquentiel épuisé sans complétude -> repli batch"
            result = answer_batch(question, k=k, search_fn=search_fn)

    result["_routing"] = log
    if verbose:
        fb = "  ->  FALLBACK batch" if log["fallback"] else ""
        print(f"[route] '{question[:55]}...'  type={qtype}  strat={strategy}{fb}")
    return result


if __name__ == "__main__":
    tests = [
        "Quelle est la date d'effet du contrat AUTO-2024-0137 ?",
        "Quelles sont toutes les exclusions applicables au vol de AUTO-2024-0137 ?",
        "Parmi les contrats auto et moto, lequel a la franchise collision la plus basse ?",
        "Suis-je couvert si je fais du VTC avec ma voiture ?",
    ]
    for q in tests:
        r = answer_routed(q)
        print(f"        found={r['answer_found']} complete={r['complete_answer_found']} "
              f"tokens={r['_total_input_tokens']} | {r['answer'][:70]}\n")