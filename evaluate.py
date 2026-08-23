"""
Mesure sur le gold set, à DEUX étages :
  - Étage 1 (retrieval)  : Recall@k par élément (proportion des gold_sections dans le top-k).
  - Étage 2 (réponse)    : exactitude — mots-clés attendus / abstention correcte.
Résultats agrégés ET par TYPE de question.

Deux formes de vérité pour l'exactitude :
  - expected_answer_contains       : TOUS ces mots-clés doivent être présents (ET).
  - expected_answer_contains_any   : liste de groupes ; OK si AU MOINS UN groupe
                                      est entièrement satisfait (OU de ET).

Usage : python evaluate.py          (batch, k=5)
        python evaluate.py seq      (séquentiel)
"""
import json
import sys
import unicodedata
from collections import defaultdict

from rag import search
from chatbot import answer_batch, answer_sequential

K = 5

def norm(s: str) -> str:
    s = unicodedata.normalize("NFD", s.lower())
    return "".join(c for c in s if unicodedata.category(c) != "Mn")

def retrieved_ids(question: str, k: int = K):
    return [f"{h['document_id']}::sec{h['metadata']['section_index']}" for h in search(question, k=k)]

def recall_at_k(gold_sections, retrieved):
    if not gold_sections:
        return None
    hits = sum(1 for g in gold_sections if g in retrieved)
    return hits / len(gold_sections)

def answer_ok(item, contract):
    """Étage 2 : réponse correcte ?"""
    if item["should_abstain"]:
        return contract.get("answer_found") is False
    ans = norm(contract.get("answer", ""))

    # forme ET : tous les mots-clés présents
    kws = item.get("expected_answer_contains")
    if kws:
        return all(norm(k) in ans for k in kws)

    # forme OU-de-ET : au moins un groupe entièrement satisfait
    groups = item.get("expected_answer_contains_any")
    if groups:
        return any(all(norm(k) in ans for k in group) for group in groups)

    return None  # rien d'attendu (ex. absence gérée plus haut)

def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "batch"
    answer_fn = answer_sequential if mode == "seq" else answer_batch

    gt = json.load(open("ground_truth.json", encoding="utf-8"))["questions"]

    rows = []
    by_type = defaultdict(lambda: {"recall": [], "answer_ok": [], "abstain_ok": []})

    for item in gt:
        q = item["question"]
        retrieved = retrieved_ids(q, K)
        rec = recall_at_k(item["gold_sections"], retrieved)

        contract = answer_fn(q)
        ok = answer_ok(item, contract)

        rows.append((item["id"], item["type"], rec, ok, contract.get("answer_found")))
        t = by_type[item["type"]]
        if rec is not None:
            t["recall"].append(rec)
        if ok is not None:
            (t["abstain_ok"] if item["should_abstain"] else t["answer_ok"]).append(1 if ok else 0)

    print(f"\n=== Détail par question (mode {mode}, k={K}) ===")
    print(f"{'id':<5}{'type':<16}{'recall@k':<10}{'réponse':<10}{'found'}")
    for qid, typ, rec, ok, found in rows:
        r = "-" if rec is None else f"{rec:.2f}"
        a = "-" if ok is None else ("OK" if ok else "X")
        print(f"{qid:<5}{typ:<16}{r:<10}{a:<10}{found}")

    print(f"\n=== Agrégé par type ===")
    print(f"{'type':<16}{'recall@k moy':<14}{'exactitude':<12}")
    for typ, d in by_type.items():
        rec = f"{sum(d['recall'])/len(d['recall']):.2f}" if d["recall"] else "-"
        allok = d["answer_ok"] + d["abstain_ok"]
        acc = f"{sum(allok)/len(allok):.0%}" if allok else "-"
        print(f"{typ:<16}{rec:<14}{acc:<12}")

    all_rec = [r for _, _, r, _, _ in rows if r is not None]
    all_ok = [1 if ok else 0 for _, _, _, ok, _ in rows if ok is not None]
    print(f"\n=== Global ===")
    print(f"  Recall@{K} moyen (hors absence) : {sum(all_rec)/len(all_rec):.2f}")
    print(f"  Exactitude globale             : {sum(all_ok)/len(all_ok):.0%}")

if __name__ == "__main__":
    main()