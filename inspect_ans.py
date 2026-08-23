"""
Affiche la réponse RÉELLE du système pour des questions données, à côté de la
métrique attendue. Sert à distinguer un vrai échec d'un faux négatif de mesure.

Usage : python inspect_answers.py            # les 4 questions "X" par défaut
        python inspect_answers.py q02 q05     # des ids précis
        python inspect_answers.py q06 seq     # ajouter 'seq' pour le mode séquentiel
"""
import json
import sys

from chatbot import answer_batch, answer_sequential

gt = {q["id"]: q for q in json.load(open("ground_truth.json", encoding="utf-8"))["questions"]}

args = [a for a in sys.argv[1:] if a != "seq"]
mode_seq = "seq" in sys.argv[1:]
ids = args or ["q02", "q05", "q06", "q07"]   # les "X" du run batch
answer_fn = answer_sequential if mode_seq else answer_batch

print(f"\n### Inspection ({'séquentiel' if mode_seq else 'batch'}) ###")
for qid in ids:
    item = gt[qid]
    c = answer_fn(item["question"])
    print("\n" + "=" * 70)
    print(f"{qid} [{item['type']}]  {item['question']}")
    print(f"  attendu (mots-clés) : {item['expected_answer_contains']}")
    if item.get("note"):
        print(f"  note gold           : {item['note']}")
    print(f"  answer_found={c['answer_found']}  complete={c['complete_answer_found']}  conf={c.get('confidence')}")
    print(f"  RÉPONSE réelle :\n    {c['answer']}")
    print(f"  preuve :\n    {c['evidence'][:200]}")