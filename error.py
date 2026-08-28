"""
Analyse erreur par erreur du mode ROUTÉ.
Pour chaque question ratée : type détecté, stratégie, recall@k, réponse réelle vs attendu,
+ un diagnostic auto de la catégorie d'erreur (retrieval vs génération vs routage).
Usage : python error_analysis.py
"""
import json
import unicodedata

from rag import search
from router import answer_routed

K = 5

def norm(s):
    s = unicodedata.normalize("NFD", s.lower())
    return "".join(c for c in s if unicodedata.category(c) != "Mn")

def retrieved_ids(q, k=K):
    return [f"{h['document_id']}::sec{h['metadata']['section_index']}" for h in search(q, k=k)]

def recall(gold, retr):
    if not gold:
        return None
    return sum(1 for g in gold if g in retr) / len(gold)

def answer_ok(item, c):
    if item["should_abstain"]:
        return c.get("answer_found") is False
    ans = norm(c.get("answer", ""))
    kws = item.get("expected_answer_contains")
    if kws:
        return all(norm(k) in ans for k in kws)
    groups = item.get("expected_answer_contains_any")
    if groups:
        return any(all(norm(k) in ans for k in g) for g in groups)
    return None

gt = json.load(open("ground_truth.json", encoding="utf-8"))["questions"]

errors = []
for item in gt:
    q = item["question"]
    retr = retrieved_ids(q)
    rec = recall(item["gold_sections"], retr)
    c = answer_routed(q, k=K, verbose=False)
    ok = answer_ok(item, c)
    if ok is False:
        errors.append((item, c, rec))

print(f"\n{len(errors)} erreur(s) en mode routé sur {len(gt)} questions.\n")
for item, c, rec in errors:
    r = "-" if rec is None else f"{rec:.2f}"
    routing = c.get("_routing", {})
    # diagnostic auto : où ça casse?
    if rec is not None and rec < 1.0:
        diag = "RETRIEVAL (bonne clause pas/partiellement récupérée)"
    else:
        diag = "GÉNÉRATION/RAISONNEMENT (retrieval OK, réponse fausse)"
    print("=" * 72)
    print(f"{item['id']} [{item['type']}]  {item['question']}")
    print(f"  routage : type_détecté={routing.get('type_detecte')} strat={routing.get('strategie')} fallback={routing.get('fallback')}")
    print(f"  recall@{K}={r}  ->  DIAGNOSTIC : {diag}")
    print(f"  attendu : {item.get('expected_answer_contains') or item.get('expected_answer_contains_any')}")
    print(f"  RÉPONSE : {c['answer'][:160]}")