"""
Calcule une fois les résultats d'évaluation et les fige dans eval_results.json.
L'app Streamlit LIT ce fichier au lieu de tout recalculer (sinon 40+ appels au LLM par visite).
À relancer seulement si le corpus, le pipeline ou le MODÈLE change.
Usage : python build_eval_cache.py

Le fichier garde maintenant, pour chaque question, la RÉPONSE du système et ce qui
était attendu : un échec doit pouvoir se lire (vraie erreur, ou réponse juste que le
juge par mots-clés n'a pas reconnue ?). Un rapport des échecs s'affiche en fin de calcul.
"""
import json
import unicodedata
from collections import defaultdict

from rag import search
from chatbot import answer_batch, answer_sequential
from router import answer_routed
from llm import MODEL

K = 5

def norm(s):
    """
    Minuscules, sans accents, et typographie neutralisée : le modèle écrit souvent
    « 5 jours » avec une espace insécable et « n’est » avec une apostrophe courbe,
    ce qui faisait échouer la comparaison avec « 5 jours » alors que la réponse était juste.
    """
    s = unicodedata.normalize("NFKD", s.lower())        # NFKD ramène les espaces insécables à l'espace
    s = "".join(c for c in s if unicodedata.category(c) != "Mn")
    s = s.replace("\u2019", "'").replace("\u2018", "'")
    return " ".join(s.split())                           # toute suite d'espaces -> une espace

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

def attendu(item):
    """Ce que le juge attend, en clair."""
    if item["should_abstain"]:
        return "abstention (answer_found = false)"
    if item.get("expected_answer_contains"):
        return "tous ces mots : " + ", ".join(item["expected_answer_contains"])
    if item.get("expected_answer_contains_any"):
        return "un de ces groupes : " + " | ".join(" + ".join(g) for g in item["expected_answer_contains_any"])
    return "rien de précis"

gt = json.load(open("ground_truth.json", encoding="utf-8"))["questions"]

# --- détail par question (mode routé, le mode de production) ---
per_q = []
by_type = defaultdict(lambda: {"recall": [], "ok": []})
for item in gt:
    q = item["question"]
    retr = retrieved_ids(q)
    rec = recall(item["gold_sections"], retr)
    c = answer_routed(q, k=K, verbose=False)
    ok = answer_ok(item, c)
    per_q.append({
        "id": item["id"], "type": item["type"], "question": q,
        "recall": rec, "ok": ok,
        "routing": c.get("_routing", {}),
        "answer": c.get("answer", ""),
        "answer_found": c.get("answer_found"),
        "attendu": attendu(item),
        "tokens": c.get("_total_input_tokens", 0),
    })
    t = by_type[item["type"]]
    if rec is not None:
        t["recall"].append(rec)
    if ok is not None:
        t["ok"].append(1 if ok else 0)

type_summary = {}
for typ, d in by_type.items():
    type_summary[typ] = {
        "recall": round(sum(d["recall"])/len(d["recall"]), 2) if d["recall"] else None,
        "accuracy": round(sum(d["ok"])/len(d["ok"]), 2) if d["ok"] else None,
    }

# --- comparaison des 3 stratégies ---
# Le mode routé n'est PAS recalculé : on reprend le passage ci-dessus. Deux passages
# du même code donnaient des chiffres différents (le modèle n'est pas parfaitement
# reproductible, même à température 0) : le tableau et le détail doivent se correspondre.
strategies = {"all_batch": lambda q: answer_batch(q, k=K),
              "all_seq": lambda q: answer_sequential(q, k_max=K)}
strat_cmp = {}
for name, fn in strategies.items():
    oks, toks = [], 0
    for item in gt:
        c = fn(item["question"])
        ok = answer_ok(item, c)
        if ok is not None:
            oks.append(1 if ok else 0)
        toks += c.get("_total_input_tokens", 0)
    strat_cmp[name] = {"accuracy": round(sum(oks)/len(oks), 2), "tokens": toks}
oks_r = [1 if p["ok"] else 0 for p in per_q if p["ok"] is not None]
strat_cmp["routed"] = {"accuracy": round(sum(oks_r)/len(oks_r), 2),
                       "tokens": sum(p["tokens"] for p in per_q)}

out = {"per_question": per_q, "by_type": type_summary, "strategies": strat_cmp, "k": K,
       "model": MODEL}
json.dump(out, open("eval_results.json", "w", encoding="utf-8"), ensure_ascii=False, indent=2)
print("eval_results.json écrit :", len(per_q), "questions,", len(type_summary), "types")

# --- rapport lisible ---
print(f"\nModèle : {MODEL}")
print("Exactitude par stratégie :")
for name, v in strat_cmp.items():
    print(f"  {name:10s} {v['accuracy']:.0%}   ({v['tokens']} tokens en entrée)")
rec_vals = [p["recall"] for p in per_q if p["recall"] is not None]
print(f"Recall@{K} moyen : {sum(rec_vals)/len(rec_vals):.2f}")

echecs = [p for p in per_q if p["ok"] is False]
print(f"\nMode routé : {len(per_q) - len(echecs)} bonnes réponses sur {len(per_q)}. "
      f"Une question pèse {100/len(per_q):.0f} points : préférer « n sur {len(per_q)} » à un pourcentage.")
print(f"Échecs en mode routé : {len(echecs)} sur {len(per_q)}")
for p in echecs:
    rt = p["routing"]
    print(f"\n  {p['id']} [{p['type']}]  type détecté = {rt.get('type_detecte')}, "
          f"stratégie = {rt.get('strategie')}, repli = {rt.get('fallback')}")
    print(f"    question : {p['question']}")
    print(f"    attendu  : {p['attendu']}")
    print(f"    trouvé   : answer_found = {p['answer_found']}")
    print(f"    réponse  : {p['answer'][:300]}")