"""
Calcule une fois les résultats d'évaluation et les fige dans eval_results.json.
L'app Streamlit LIT ce fichier au lieu de tout recalculer (sinon 40+ appels Mistral par visite).
À relancer seulement si le corpus ou le pipeline change.
Usage : python build_eval_cache.py
"""
import json
import unicodedata
from collections import defaultdict

from rag import search
from chatbot import answer_batch, answer_sequential
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
strategies = {"all_batch": lambda q: answer_batch(q, k=K),
              "all_seq": lambda q: answer_sequential(q, k_max=K),
              "routed": lambda q: answer_routed(q, k=K, verbose=False)}
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

out = {"per_question": per_q, "by_type": type_summary, "strategies": strat_cmp, "k": K}
json.dump(out, open("eval_results.json", "w", encoding="utf-8"), ensure_ascii=False, indent=2)
print("eval_results.json écrit :", len(per_q), "questions,", len(type_summary), "types")