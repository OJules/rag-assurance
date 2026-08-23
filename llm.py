"""
Génération avec answer contract.
Le LLM ne renvoie PAS du texte libre : il renvoie un JSON typé qui dit
s'il a trouvé la réponse, si elle est complète, avec quelle preuve et quelle confiance.
C'est ce contrat qui rend l'abstention (mine 4) et l'incomplétude (mine 2) détectables.

Provider isolé dans une variable -> LLM interchangeable (on ne change qu'ici).
Clé API lue depuis l'environnement (.env) -> jamais en dur dans le code.
"""
import os
import json

from dotenv import load_dotenv
from mistralai.client import Mistral

load_dotenv()  # lit le fichier .env et peuple os.environ

PROVIDER = "mistral"
MODEL = "mistral-small-latest"
_client = Mistral(api_key=os.environ["MISTRAL_API_KEY"])

# --- L'answer contract : la forme EXACTE que le LLM doit renvoyer ---
SYSTEM_PROMPT = """Tu es un assistant qui répond UNIQUEMENT à partir des extraits de contrats d'assurance fournis.
Règles strictes :
- Tu ne réponds qu'avec ce qui est présent dans les extraits. Tu n'inventes rien.
- Si l'information n'est pas dans les extraits, answer_found = false et tu le dis clairement.
- Si tu ne trouves qu'une partie de la réponse (ex. certaines exclusions mais peut-être pas toutes), complete_answer_found = false.
- Tu cites systématiquement l'extrait qui justifie ta réponse dans "evidence".

Tu réponds STRICTEMENT en JSON, sans texte autour, avec ce schéma :
{
  "answer": "ta réponse en français, ou une phrase disant que l'information n'est pas disponible",
  "evidence": "l'extrait exact du contexte qui justifie la réponse (ou chaîne vide)",
  "answer_found": true | false,
  "complete_answer_found": true | false,
  "confidence": un nombre entre 0 et 1
}"""

def _format_context(hits):
    """Transforme les hits de search() en un bloc de contexte numéroté."""
    blocs = []
    for h in hits:
        blocs.append(f"[Extrait {h['rank']} — {h['document_id']} / {h['section']}]\n{h['text']}")
    return "\n\n".join(blocs)

def generate(question: str, hits: list) -> dict:
    """
    Prend la question + les passages récupérés, renvoie l'answer contract (dict).
    En cas de JSON malformé, on renvoie un contrat d'échec explicite plutôt que de planter.
    """
    context = _format_context(hits)
    user_msg = f"CONTEXTE :\n{context}\n\nQUESTION : {question}"

    resp = _client.chat.complete(
        model=MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ],
        response_format={"type": "json_object"},  # force une sortie JSON
        temperature=0,                             # déterministe : pas d'invention gratuite
    )
    raw = resp.choices[0].message.content

    # on compte les tokens consommés (utile pour la démo top-1 vs top-k plus tard)
    usage = {"input_tokens": resp.usage.prompt_tokens,
             "output_tokens": resp.usage.completion_tokens}

    try:
        contract = json.loads(raw)
    except json.JSONDecodeError:
        contract = {"answer": raw, "evidence": "", "answer_found": False,
                    "complete_answer_found": False, "confidence": 0.0,
                    "_parse_error": True}
    contract["_usage"] = usage
    return contract


# petit test manuel : python llm.py  (nécessite rag.py + une base ingérée + une clé)
if __name__ == "__main__":
    from rag import search
    for q in [
        "Quelle est la date d'effet du contrat AUTO-2024-0137 ?",
        "Suis-je couvert si je fais du VTC avec ma voiture ?",   # mine 4 : doit s'abstenir
    ]:
        hits = search(q, k=3)
        c = generate(q, hits)
        print(f"\n### {q}")
        print(f"  answer_found={c['answer_found']}  complete={c['complete_answer_found']}  conf={c['confidence']}")
        print(f"  réponse : {c['answer']}")
        print(f"  preuve  : {c['evidence'][:80]}")
        print(f"  tokens  : {c['_usage']}")