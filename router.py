"""
Routage déterministe par règles + garde-fou d'exécution.

Pipeline (répond à la remarque de Kezhan : le parser n'est PAS une autorité) :
  classify (règles) -> stratégie -> exécution -> validation -> fallback borné -> logging

Asymétrie du risque, mesurée sur le gold set :
  - comparaison/liste envoyée à tort en séquentiel -> CASSE (dangereux)
  - lookup envoyé à tort en batch -> juste un peu plus cher (bénin)
Donc : en cas de doute on penche batch, et le fallback ne va JAMAIS de batch vers séquentiel.
"""
import re

from chatbot import answer_batch, answer_sequential

# --- 1. Classifieur par règles (déterministe, auditable) ---
RE_LISTE       = re.compile(r"\b(toutes|tous|lister?|liste|l'ensemble|quelles sont)\b", re.I)
RE_COMPARAISON = re.compile(r"\b(le plus|la plus|lequel|laquelle|comparer?|entre|différence|versus|vs)\b", re.I)

def classify(question: str) -> str:
    """Renvoie 'liste' | 'comparaison' | 'lookup' (défaut)."""
    if RE_COMPARAISON.search(question):
        return "comparaison"
    if RE_LISTE.search(question):
        return "liste"
    return "lookup"

# --- 2. Stratégie par type ---
#   lookup            -> séquentiel (économe : top-1 suffit souvent)
#   liste/comparaison -> batch (nécessaire : vue d'ensemble requise)
STRATEGY = {"lookup": "sequential", "liste": "batch", "comparaison": "batch"}

# --- 3. Orchestration avec garde-fou ---
def answer_routed(question: str, k: int = 5, verbose: bool = True) -> dict:
    qtype = classify(question)
    strategy = STRATEGY[qtype]
    log = {"question": question, "type_detecte": qtype, "strategie": strategy, "fallback": False}

    if strategy == "batch":
        result = answer_batch(question, k=k)
    else:
        result = answer_sequential(question, k_max=k)
        # GARDE-FOU : un séquentiel qui s'arrête par ÉPUISEMENT n'a pas su conclure.
        # On ne fait JAMAIS confiance aveuglément au routage -> repli batch (mode sûr), borné à 1.
        if result.get("_stop_reason") == "épuisement" and result.get("answer_found"):
            log["fallback"] = True
            log["fallback_raison"] = "séquentiel épuisé sans complétude -> repli batch"
            result = answer_batch(question, k=k)

    result["_routing"] = log
    if verbose:
        fb = "  ->  FALLBACK batch" if log["fallback"] else ""
        print(f"[route] '{question[:55]}...'  type={qtype}  strat={strategy}{fb}")
    return result


if __name__ == "__main__":
    tests = [
        "Quelle est la date d'effet du contrat AUTO-2024-0137 ?",              # -> lookup -> seq
        "Quelles sont toutes les exclusions applicables au vol de AUTO-2024-0137 ?",  # -> liste -> batch
        "Parmi les contrats auto et moto, lequel a la franchise collision la plus basse ?",  # -> comparaison -> batch
        "Suis-je couvert si je fais du VTC avec ma voiture ?",                 # -> lookup -> seq (absence)
    ]
    for q in tests:
        r = answer_routed(q)
        print(f"        found={r['answer_found']} complete={r['complete_answer_found']} "
              f"tokens={r['_total_input_tokens']} | {r['answer'][:70]}\n")