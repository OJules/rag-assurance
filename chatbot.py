"""
Orchestration : assemble retrieval (rag.search) + génération (llm.generate).
Deux stratégies comparables sur la même question :
  - answer_batch     : on donne les k chunks d'un coup (1 appel LLM)
  - answer_sequential: top-1, puis top-2, ... avec 3 sorties de boucle :
        * suffisance       -> answer_found ET complete
        * abstention bornée-> N tentatives sans answer_found (early-stop, coût maîtrisé)
        * épuisement       -> tous les chunks parcourus sans complétude

Règle de décision (posée avec Jules) :
  answer_found == False            -> CONTINUER (on ignore 'complete')
  answer_found == True + complete  -> STOP (suffisance)
  answer_found == True + !complete -> CONTINUER
Contexte CUMULATIF : à l'étape i on envoie chunk_1..chunk_i (pour recomposer).
"""
from rag import search
from llm import generate

ABSTAIN_AFTER = 2   # early-stop : N tentatives sans answer_found -> on borne et on s'abstient


def answer_batch(question: str, k: int = 5) -> dict:
    hits = search(question, k=k)
    contract = generate(question, hits)
    contract["_strategy"] = "batch"
    contract["_docs_used"] = len(hits)
    contract["_llm_calls"] = 1
    contract["_total_input_tokens"] = contract["_usage"]["input_tokens"]
    return contract


def answer_sequential(question: str, k_max: int = 5) -> dict:
    hits = search(question, k=k_max)

    total_input = 0
    llm_calls = 0
    consecutive_not_found = 0
    last = None

    for i in range(1, len(hits) + 1):
        context = hits[:i]                 # contexte CUMULATIF
        contract = generate(question, context)
        llm_calls += 1
        total_input += contract["_usage"]["input_tokens"]
        last = contract

        found = contract.get("answer_found", False)
        complete = contract.get("complete_answer_found", False)

        # 'complete' n'est lu QUE si answer_found est True (garde-fou)
        if found and complete:
            last["_stop_reason"] = "suffisance"
            last["_docs_used"] = i
            break

        # early-stop : après N tentatives sans rien trouver, on borne et on s'abstient
        consecutive_not_found = consecutive_not_found + 1 if not found else 0
        if consecutive_not_found >= ABSTAIN_AFTER:
            last["_stop_reason"] = "abstention_bornee"
            last["_docs_used"] = i
            break
    else:
        last["_stop_reason"] = "épuisement"
        last["_docs_used"] = len(hits)

    last["_strategy"] = "sequential"
    last["_llm_calls"] = llm_calls
    last["_total_input_tokens"] = total_input
    return last


def _show(tag, c):
    print(f"  [{tag}] found={c['answer_found']} complete={c['complete_answer_found']} "
          f"| docs={c['_docs_used']} appels={c['_llm_calls']} "
          f"tokens_in={c['_total_input_tokens']}"
          + (f" stop={c['_stop_reason']}" if '_stop_reason' in c else ""))
    print(f"        réponse: {c['answer'][:110]}")


if __name__ == "__main__":
    questions = [
        "Quelle est la date d'effet du contrat AUTO-2024-0137 ?",
        "Quelles sont toutes les exclusions applicables au vol du contrat AUTO-2024-0137 ?",
        "Suis-je couvert si je fais du VTC avec ma voiture ?",
    ]
    for q in questions:
        print(f"\n### {q}")
        _show("SEQ ", answer_sequential(q, k_max=5))
        _show("BATCH", answer_batch(q, k=5))