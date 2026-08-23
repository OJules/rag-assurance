"""
Compare 3 stratégies sur tout le gold set, à exactitude ET tokens :
  - all_batch  : tout en batch
  - all_seq    : tout en séquentiel
  - routed     : dispatcher par règles + garde-fous (router.answer_routed)

But : montrer que le routage garde l'exactitude tout en réduisant les tokens.
Usage : python compare_strategies.py
"""
import json
import unicodedata

from chatbot import answer_batch, answer_sequential
from router import answer_routed

def norm(s):
    s = unicodedata.normalize("NFD", s.lower())
    return "".join(c for c in s if unicodedata.category(c) != "Mn")

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

strategies = {
    "all_batch": lambda q: answer_batch(q, k=5),
    "all_seq":   lambda q: answer_sequential(q, k_max=5),
    "routed":    lambda q: answer_routed(q, k=5, verbose=False),
}

print(f"{'stratégie':<12}{'exactitude':<12}{'tokens_in totaux':<18}{'appels LLM totaux'}")
for name, fn in strategies.items():
    oks, toks, calls = [], 0, 0
    for item in gt:
        c = fn(item["question"])
        ok = answer_ok(item, c)
        if ok is not None:
            oks.append(1 if ok else 0)
        toks += c.get("_total_input_tokens", c.get("_usage", {}).get("input_tokens", 0))
        calls += c.get("_llm_calls", 1)
    acc = f"{sum(oks)/len(oks):.0%}" if oks else "-"
    print(f"{name:<12}{acc:<12}{toks:<18}{calls}")