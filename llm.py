"""
Génération avec answer contract - fournisseur : Groq (gratuit, sans carte).
Le LLM renvoie un JSON typé (answer / evidence / answer_found / complete_answer_found / confidence).
C'est ce contrat qui rend l'abstention et l'incomplétude détectables.

Deux contrats de réponse :
  - generate(question, hits)            : le contrat général (recherche d'une information).
  - generate_couverture(question, hits) : pour « suis-je couvert ? ». Le modèle doit
    d'abord lister les faits, puis confronter CHAQUE clause à ces faits, et seulement
    ensuite rendre un verdict : couvert / non_couvert / indetermine / non_concerne.
    Garde-fou en code : pas de verdict tranché sans citation continue retrouvée dans un passage.

Client chargé PARESSEUSEMENT (au 1er appel) : la clé est déjà dans os.environ à ce moment.
"""
import os
import re
import json
import time
import unicodedata

from dotenv import load_dotenv
from groq import Groq, RateLimitError

load_dotenv()  # lit le .env en local ; sur le cloud la clé vient des secrets

PROVIDER = "groq"
# Modèle Groq. Si celui-ci est rejeté ("model not found"), va voir la liste
# à jour sur console.groq.com et remplace la chaîne ci-dessous.
MODEL = "openai/gpt-oss-120b"
_client = None


def _get_client():
    """Instancie le client Groq au premier appel seulement (chargement paresseux)."""
    global _client
    if _client is None:
        _client = Groq(api_key=os.environ["GROQ_API_KEY"])
    return _client


ESSAIS_MAX = 4          # 1 appel + 3 nouvelles tentatives si la limite de débit est atteinte
ATTENTE_MAX = 40        # secondes : au-delà, on renonce plutôt que de figer l'interface


def _attente_conseillee(erreur) -> float:
    """Lit dans l'erreur Groq le délai conseillé ("Please try again in 15.99s")."""
    m = re.search(r"try again in (?:(\d+)m)?([\d.]+)s", str(erreur))
    if not m:
        return 10.0
    return int(m.group(1) or 0) * 60 + float(m.group(2))


def _appel(messages):
    """
    Appelle le modèle en sortie JSON. Si la limite de débit du palier gratuit est
    atteinte (erreur 429), attend le délai conseillé puis réessaie, au lieu de
    faire planter l'application. Plusieurs contrats analysés à la suite dépassent
    vite les 8 000 tokens par minute.
    """
    for essai in range(1, ESSAIS_MAX + 1):
        try:
            return _get_client().chat.completions.create(
                model=MODEL,
                messages=messages,
                response_format={"type": "json_object"},  # force une sortie JSON
                temperature=0,                             # déterministe : pas d'invention gratuite
            )
        except RateLimitError as e:
            attente = _attente_conseillee(e) + 1
            if essai == ESSAIS_MAX or attente > ATTENTE_MAX:
                raise
            print(f"[llm] limite de débit atteinte, nouvelle tentative dans {attente:.0f}s "
                  f"(essai {essai}/{ESSAIS_MAX - 1})")
            time.sleep(attente)


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
        blocs.append(f"[Extrait {h['rank']} - {h['document_id']} / {h['section']}]\n{h['text']}")
    return "\n\n".join(blocs)


def generate(question: str, hits: list) -> dict:
    """
    Prend la question + les passages récupérés, renvoie l'answer contract (dict).
    En cas de JSON malformé, on renvoie un contrat d'échec explicite plutôt que de planter.
    """
    context = _format_context(hits)
    user_msg = f"CONTEXTE :\n{context}\n\nQUESTION : {question}"

    resp = _appel([
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_msg},
    ])
    raw = resp.choices[0].message.content

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


# --- Le contrat "couverture" : confronter un sinistre aux clauses d'UN contrat ---
# Pourquoi un contrat à part : avec le contrat général, le modèle s'accrochait à un
# passage qui RESSEMBLAIT à la question sans vérifier que ses conditions étaient
# remplies (une garantie exigeant une violence appliquée à un vol sans violence ;
# une garantie incendie appliquée à un dommage sans flamme). On l'oblige donc à
# écrire la confrontation fait par fait avant de conclure.
# Les exemples du prompt sont pris HORS des questions de test (vélo, bougie).
COUVERTURE_PROMPT = """Tu es gestionnaire de sinistres. Tu dois dire si un sinistre est couvert, UNIQUEMENT à partir des extraits fournis, qui viennent tous d'UN seul contrat d'assurance.

Méthode obligatoire, dans cet ordre :

1. FAITS. Liste les faits énoncés dans la question, sans rien ajouter ni requalifier. Ce que la question ne dit pas est ABSENT : pas de flamme, de violence, d'effraction ou de faute si elle n'en parle pas.

2. CLAUSES. Pour chaque extrait, donne son rôle : "garantie", "exclusion", "definition", "condition" ou "hors_sujet". Pour les extraits utiles, écris la condition qu'ils posent.
   Un mot qui ressemble ne suffit pas : "véhicule terrestre à moteur" ne désigne pas une trottinette sans moteur.

3. CONFRONTATION. Pour chaque clause utile, la condition est-elle remplie par les FAITS ? "oui", "non" ou "inconnu".
   Une garantie ne s'applique que si TOUTES ses conditions sont remplies : une garantie "rupture de canalisation" ne couvre pas une infiltration par la toiture ; une garantie "bris de glace" ne couvre pas une rayure.

   Attention au sens d'une liste : si un extrait s'ouvre sur "sont exclus", "restent exclus", "ne sont pas assurés" ou une formule équivalente, TOUT ce qui suit dans la liste est exclu, même si le mot "garantissons" apparaît ailleurs dans l'extrait.
   Les extraits viennent de PDF : certains mélangent deux colonnes (phrases interrompues, mots sans rapport collés). Un extrait visiblement mélangé ne peut pas servir à trancher.

4. VERDICT, une seule valeur :
   - "non_concerne" : ce contrat n'assure pas ce type de bien ou de risque (par exemple un contrat automobile pour un bien qui n'est pas un véhicule). À vérifier AVANT tout le reste.
   - "couvert" : une garantie s'applique (toutes ses conditions = oui) ET aucune exclusion lue ne s'applique.
   - "non_couvert" : une exclusion s'applique explicitement aux faits, OU une condition indispensable de la garantie concernée est explicitement non remplie.
   - "indetermine" : tous les autres cas (condition inconnue, garantie concernée absente des extraits, extrait tronqué, mélangé ou illisible).
   L'absence de mention n'est pas une exclusion : "le contrat ne parle pas de ce bien" donne "indetermine", jamais "non_couvert".
   Tu ne qualifies pas les faits à la place du gestionnaire : une porte non fermée à clé n'est une "négligence grave" que si le contrat le dit.

Tu réponds STRICTEMENT en JSON, sans texte autour, avec ce schéma :
{
  "faits": ["fait 1", "fait 2"],
  "clauses": [
    {"extrait": 3, "role": "garantie", "condition": "ce que la clause exige", "remplie": "oui" | "non" | "inconnu", "pourquoi": "une phrase"}
  ],
  "verdict": "couvert" | "non_couvert" | "indetermine" | "non_concerne",
  "answer": "deux phrases au plus, en français courant : la décision et sa raison principale, avec le numéro de l'extrait qui décide",
  "evidence": "UN passage continu de l'extrait qui décide, recopié MOT POUR MOT, 30 mots au plus, sans points de suspension et sans recoller deux morceaux (chaîne vide si indetermine ou non_concerne)",
  "ce_qui_manque": "ce qu'il faudrait vérifier pour trancher (chaîne vide si le verdict est tranché)",
  "confidence": un nombre entre 0 et 1
}"""

_VERDICTS = ("couvert", "non_couvert", "indetermine", "non_concerne")
_TRANCHES = ("couvert", "non_couvert")
_ROLES = ("garantie", "exclusion", "definition", "condition", "hors_sujet")


def _alnum(texte):
    """Ne garde que lettres et chiffres, sans accents : comparaison robuste aux espaces et césures."""
    t = unicodedata.normalize("NFKD", (texte or "").lower())
    return re.sub(r"[^a-z0-9]", "", "".join(c for c in t if not unicodedata.combining(c)))


def citation_retrouvee(evidence: str, hits: list, minimum: int = 30):
    """
    La citation donnée comme preuve figure-t-elle, D'UN SEUL TENANT, dans un passage lu ?
    Renvoie le rang du passage qui la contient, ou None.

    La comparaison ignore espaces, ponctuation, accents et césures, mais exige la
    continuité : une citation recollée avec des points de suspension (« Nous
    garantissons ... dans les parties communes ») est refusée, car recoller deux
    morceaux peut transformer une exclusion en garantie.
    """
    if "…" in (evidence or "") or "..." in (evidence or ""):
        return None
    cible = _alnum(evidence)
    if len(cible) < minimum:
        return None
    for h in hits:
        if cible in _alnum(h.get("text", "")):
            return h.get("rank")
    return None


def generate_couverture(question: str, hits: list) -> dict:
    """
    Décision de couverture pour UN contrat. Renvoie le contrat général (answer, evidence,
    answer_found, complete_answer_found, confidence) enrichi de :
      verdict (couvert / non_couvert / indetermine / non_concerne), faits, clauses,
      ce_qui_manque, _evidence_ok, _evidence_rank, _retrograde, _retrograde_raison.

    Règles appliquées EN CODE après le modèle :
      - answer_found = le verdict est tranché (couvert ou non_couvert) ;
      - un verdict tranché est rétrogradé en "indetermine" si sa citation n'est pas
        retrouvée d'un seul tenant dans un passage lu, ou si ce passage est de
        qualité d'extraction "faible" : pas de décision sans preuve vérifiable.
    """
    context = _format_context(hits)
    user_msg = f"EXTRAITS DU CONTRAT :\n{context}\n\nSINISTRE : {question}"

    resp = _appel([
        {"role": "system", "content": COUVERTURE_PROMPT},
        {"role": "user", "content": user_msg},
    ])
    raw = resp.choices[0].message.content
    usage = {"input_tokens": resp.usage.prompt_tokens,
             "output_tokens": resp.usage.completion_tokens}

    try:
        brut = json.loads(raw)
        if not isinstance(brut, dict):
            raise ValueError("JSON inattendu")
    except (json.JSONDecodeError, ValueError):
        brut = {"answer": raw, "_parse_error": True}

    verdict = brut.get("verdict")
    if verdict not in _VERDICTS:
        verdict = "indetermine"

    clauses = []
    for c in brut.get("clauses") or []:
        if not isinstance(c, dict):
            continue
        role = c.get("role") if c.get("role") in _ROLES else "hors_sujet"
        remplie = c.get("remplie") if c.get("remplie") in ("oui", "non", "inconnu") else "inconnu"
        try:
            extrait = int(c.get("extrait"))
        except (TypeError, ValueError):
            continue
        clauses.append({"extrait": extrait, "role": role,
                        "condition": str(c.get("condition") or ""),
                        "remplie": remplie, "pourquoi": str(c.get("pourquoi") or "")})

    evidence = str(brut.get("evidence") or "")
    rang_preuve = citation_retrouvee(evidence, hits)
    evidence_ok = rang_preuve is not None
    qualite_preuve = next((h.get("quality") for h in hits if h.get("rank") == rang_preuve), None)

    raison = ""
    if verdict in _TRANCHES and not evidence_ok:
        raison = ("la citation donnée comme preuve n'a pas été retrouvée d'un seul tenant "
                  "dans les passages lus")
    elif verdict in _TRANCHES and qualite_preuve == "faible":
        raison = "la clause qui décide vient d'un passage dont l'extraction est de qualité faible"
    retrograde = bool(raison)
    if retrograde:
        verdict = "indetermine"

    # l'answer ne doit pas commencer par le jeton du verdict ("non_couvert – ...")
    answer = re.sub(r"^\s*(non[_ ]couvert|non[_ ]concerne|indetermine|couvert)\s*[-–—:]\s*", "",
                    str(brut.get("answer") or ""), flags=re.I)
    answer = answer[:1].upper() + answer[1:]

    try:
        confidence = float(brut.get("confidence", 0.0))
    except (TypeError, ValueError):
        confidence = 0.0

    tranche = verdict in _TRANCHES
    contract = {
        "answer": answer,
        "evidence": evidence,
        "answer_found": tranche,
        "complete_answer_found": tranche,
        "confidence": confidence,
        "verdict": verdict,
        "faits": [str(f) for f in (brut.get("faits") or []) if f],
        "clauses": clauses,
        "ce_qui_manque": str(brut.get("ce_qui_manque") or ""),
        "_evidence_ok": evidence_ok,
        "_evidence_rank": rang_preuve,
        "_retrograde": retrograde,
        "_retrograde_raison": raison,
        "_usage": usage,
    }
    if brut.get("_parse_error"):
        contract["_parse_error"] = True
    return contract


# petit test manuel : python llm.py  (nécessite rag.py + une base ingérée + une clé Groq)
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