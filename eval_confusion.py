"""
Expérience : la confusion entre contrats — Global RAG vs Per-contract RAG.

Corpus : polices synthétiques markdown (vérité terrain connue -> mesure possible).
On compare deux façons de répondre aux questions-pièges de trap_questions.json :
  - GLOBAL      : une seule réponse sur le mélange de tous les contrats
  - PER-CONTRACT: une réponse ISOLÉE par contrat concerné, puis on agrège

Classification de chaque réponse (proxy heuristique, PAS une vérité absolue) :
  CORRECT          bonne valeur attribuée au bon contrat
  MISATTRIBUTED    bonne valeur mais au mauvais contrat (la vraie "confusion")
  INCORRECT_VALUE  contrat cité mais valeur fausse
  INCOMPLETE       valeur donnée sans préciser le contrat
  ABSTAINED        le système ne s'est pas prononcé

Matching sur l'id EXACT (conservateur) : si le LLM n'écrit pas l'id, on sous-estime
la bonne attribution. Assumé -> métrique = "contract attribution accuracy (heuristic)".

Usage : python eval_confusion.py
"""
import json
import re

from rag import search
from router import answer_routed


def _norme(txt):
    return (txt or "").lower()


# Séparateurs de milliers possibles : espace normale, espace insécable/fine,
# apostrophes droite et courbe (Suisse), point. On les SUPPRIME des deux côtés
# avant comparaison -> "80 000", "80\u00a0000", "80\u2019000" deviennent "80000".
_SEPARATEURS = [" ", "\u00a0", "\u202f", "'", "\u2019", ".", ","]

def _sans_separateurs(txt):
    for s in _SEPARATEURS:
        txt = txt.replace(s, "")
    return txt

def _valeur_presente(reponse, valeur_num):
    """La valeur numérique attendue apparaît-elle, quel que soit le séparateur de milliers ?"""
    r = _sans_separateurs(_norme(reponse))
    return str(valeur_num) in r


def _id_present(reponse, doc_id):
    return _norme(doc_id) in _norme(reponse)


def _proches(reponse, valeur_num, doc_id, fenetre=80):
    """La valeur et l'id sont-ils proches (même zone de texte) ? -> attribution."""
    r = _norme(reponse)
    did = _norme(doc_id)
    for m in re.finditer(re.escape(str(valeur_num)), r):
        deb = max(0, m.start() - fenetre)
        fin = min(len(r), m.end() + fenetre)
        if did in r[deb:fin]:
            return True
    return False


def classifier_global(reponse, found, attendus):
    """
    Mode GLOBAL : une seule réponse sur le mélange des contrats.
    Ici l'id DOIT apparaître dans la réponse : c'est la seule façon de savoir
    à quel contrat le LLM attribue la valeur. On mesure donc l'attribution.
    """
    if not found:
        return {a["document_id"]: "ABSTAINED" for a in attendus}
    resultats = {}
    for a in attendus:
        did, vnum = a["document_id"], a["valeur_num"]
        val_ok = _valeur_presente(reponse, vnum)
        id_ok = _id_present(reponse, did)
        proche = _proches(reponse, vnum, did) if (val_ok and id_ok) else False
        if val_ok and id_ok and proche:
            resultats[did] = "CORRECT"
        elif val_ok and id_ok and not proche:
            resultats[did] = "MISATTRIBUTED"      # valeur et id présents mais pas associés
        elif val_ok and not id_ok:
            resultats[did] = "INCOMPLETE"         # valeur donnée sans préciser le contrat
        elif id_ok and not val_ok:
            resultats[did] = "INCORRECT_VALUE"    # bon contrat cité, valeur absente/fausse
        else:
            resultats[did] = "INCORRECT_VALUE"
    return resultats


def classifier_per_contract(reponse, found, attendu):
    """
    Mode PER-CONTRACT : la recherche est filtrée sur UN contrat.
    L'attribution est ACQUISE PAR CONSTRUCTION (on sait de quel contrat vient la
    réponse) -> on ne mesure PAS l'attribution, seulement la VALEUR.
      bonne valeur -> CORRECT ; sinon INCORRECT_VALUE ; abstention -> ABSTAINED.
    Pas d'INCOMPLETE ni de MISATTRIBUTED possibles ici : l'id est implicite.
    """
    if not found:
        return "ABSTAINED"
    return "CORRECT" if _valeur_presente(reponse, attendu["valeur_num"]) else "INCORRECT_VALUE"


def run():
    data = json.load(open("trap_questions.json", encoding="utf-8"))
    questions = [q for q in data["questions"] if not q["ambigue"]]  # on écarte la Q6 ambiguë du calcul principal

    CATS = ["CORRECT", "MISATTRIBUTED", "INCORRECT_VALUE", "INCOMPLETE", "ABSTAINED"]
    compte = {"global": {c: 0 for c in CATS}, "per_contract": {c: 0 for c in CATS}}
    tokens = {"global": 0, "per_contract": 0}
    appels = {"global": 0, "per_contract": 0}

    for q in questions:
        print(f"\n### {q['id']} — {q['question']}")

        # --- GLOBAL : une réponse sur tout le corpus ---
        rg = answer_routed(q["question"], k=5, verbose=False)
        tokens["global"] += rg.get("_total_input_tokens", 0)
        appels["global"] += rg.get("_llm_calls", 1)
        cls_g = classifier_global(rg.get("answer", ""), rg.get("answer_found", False), q["attendus"])
        for did, cat in cls_g.items():
            compte["global"][cat] += 1
        print(f"  GLOBAL       : {cls_g}")
        print(f"    → {rg.get('answer','')[:120]}")

        # --- PER-CONTRACT : une réponse isolée par contrat attendu ---
        for a in q["attendus"]:
            did = a["document_id"]
            rp = answer_routed(q["question"], k=5, verbose=False,
                               search_fn=lambda ques, k=5, _d=did: search(ques, k=k, filtre_document=_d))
            tokens["per_contract"] += rp.get("_total_input_tokens", 0)
            appels["per_contract"] += rp.get("_llm_calls", 1)
            # per-contract : attribution acquise par le filtre -> on vérifie la valeur
            cat_p = classifier_per_contract(rp.get("answer", ""), rp.get("answer_found", False), a)
            compte["per_contract"][cat_p] += 1
            print(f"  PER [{did}] : {cat_p}  → {rp.get('answer','')[:80]}")

    # --- Bilan ---
    print("\n" + "=" * 60)
    print("BILAN — attribution par contrat (proxy heuristique)")
    print("=" * 60)
    total = sum(compte["global"].values())
    print(f"{'Catégorie':<18}{'Global':>10}{'Per-contract':>16}")
    for c in CATS:
        print(f"{c:<18}{compte['global'][c]:>10}{compte['per_contract'][c]:>16}")
    print(f"\n{'Appels LLM':<18}{appels['global']:>10}{appels['per_contract']:>16}")
    print(f"{'Tokens (entrée)':<18}{tokens['global']:>10}{tokens['per_contract']:>16}")

    conf_g = compte["global"]["MISATTRIBUTED"]
    conf_p = compte["per_contract"]["MISATTRIBUTED"]
    print(f"\nErreurs d'attribution (MISATTRIBUTED) : global={conf_g}  per-contract={conf_p}")
    print(f"Réponses incomplètes (INCOMPLETE)     : global={compte['global']['INCOMPLETE']}  "
          f"per-contract={compte['per_contract']['INCOMPLETE']}")
    print("\nNB : mesure approximative (matching heuristique id/valeur). "
          "Un id non repris par le LLM compte comme INCOMPLETE même si l'humain comprendrait.")


if __name__ == "__main__":
    run()