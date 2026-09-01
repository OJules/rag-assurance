"""
Routage déterministe par règles + garde-fou d'exécution.

Pipeline : classify (règles) -> stratégie -> exécution -> validation -> fallback borné -> logging

Asymétrie du risque, mesurée sur le gold set :
  - comparaison/liste envoyée à tort en séquentiel -> CASSE (dangereux)
  - lookup envoyé à tort en batch -> juste un peu plus cher (bénin)
Donc : en cas de doute on penche batch, et le fallback ne va JAMAIS de batch vers séquentiel.

SOURCE DE RECHERCHE INJECTABLE (search_fn) : transmise telle quelle à chatbot.
Par défaut = corpus markdown ; passer rag_pdf.search pour interroger le corpus PDF.
"""
import re

from chatbot import answer_batch, answer_sequential

RE_LISTE       = re.compile(r"\b(toutes|tous|lister?|liste|l'ensemble|quelles sont)\b", re.I)
RE_COMPARAISON = re.compile(r"\b(le plus|la plus|lequel|laquelle|comparer?|entre|différence|versus|vs)\b", re.I)


def classify(question: str) -> str:
    """Renvoie 'liste' | 'comparaison' | 'lookup' (défaut)."""
    if RE_COMPARAISON.search(question):
        return "comparaison"
    if RE_LISTE.search(question):
        return "liste"
    return "lookup"


STRATEGY = {"lookup": "sequential", "liste": "batch", "comparaison": "batch"}


def answer_routed(question: str, k: int = 5, verbose: bool = True, search_fn=None) -> dict:
    qtype = classify(question)
    strategy = STRATEGY[qtype]
    log = {"question": question, "type_detecte": qtype, "strategie": strategy, "fallback": False}

    if strategy == "batch":
        result = answer_batch(question, k=k, search_fn=search_fn)
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