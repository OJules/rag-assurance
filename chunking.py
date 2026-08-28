"""
Découpage de texte brut (issu d'un PDF/Word) en chunks.
Python PUR — aucune dépendance externe.

Principe : préserver l'unité sémantique/juridique. Les titres/numéros sont des
INDICES (enrichissement), jamais une DÉPENDANCE. La taille max est un GARDE-FOU.

Ordre de traitement :
  1. découpe en blocs sur les lignes vides (unités sémantiques candidates)
  2. fusionne les TITRES ORPHELINS avec le bloc suivant (le titre appartient à sa clause)
  3. cascade de taille sur chaque bloc :
       - tient dans la limite            -> gardé entier
       - trop long                       -> subdivisé par PHRASES
       - phrase seule dépassant la limite-> découpage BRUTAL (dernier recours)
  4. fusionne les fragments trop courts avec le voisin
Chevauchement ajouté uniquement aux coupes forcées, pour ne pas perdre une info à la frontière.
"""
import re

MAX_CHARS = 1200      # garde-fou de taille, pas critère premier
MIN_CHARS = 60        # en dessous : fragment, à fusionner
OVERLAP   = 150       # caractères repris entre deux morceaux d'une coupe forcée

RE_TITRE = re.compile(r"^\s*((?:article|chapitre|section)\s+\d+|\d+(?:\.\d+)*\.?)\b", re.I)


def _detecte_titre(bloc: str):
    """Titre/numéro en tête de bloc, ou None. Indice, n'impose rien."""
    m = RE_TITRE.match(bloc)
    return m.group(1).strip() if m else None


def _est_titre_seul(bloc: str) -> bool:
    """Vrai si le bloc n'est qu'un titre (court + matche un motif de titre)."""
    return _detecte_titre(bloc) is not None and len(bloc) < MIN_CHARS


def _decoupe_par_phrases(texte: str):
    phrases = re.split(r"(?<=[.!?:])\s+", texte.strip())
    return [p for p in phrases if p]


def _coupe_brutale(texte: str, taille: int, overlap: int):
    morceaux, i, n = [], 0, len(texte)
    while i < n:
        morceaux.append(texte[i:i + taille])
        i += taille - overlap
    return morceaux


def _subdivise(bloc: str):
    """Bloc trop long : par phrases d'abord ; coupe brutale en dernier recours."""
    morceaux, courant = [], ""
    for phrase in _decoupe_par_phrases(bloc):
        if len(phrase) > MAX_CHARS:
            if courant:
                morceaux.append(courant.strip()); courant = ""
            morceaux.extend(_coupe_brutale(phrase, MAX_CHARS, OVERLAP))
        elif len(courant) + len(phrase) + 1 <= MAX_CHARS:
            courant = (courant + " " + phrase).strip()
        else:
            morceaux.append(courant.strip()); courant = phrase
    if courant.strip():
        morceaux.append(courant.strip())
    return morceaux


def chunk_text(texte: str):
    """
    Renvoie une liste de dicts : {"text": ..., "titre": <titre détecté ou None>}.
    Le découpage ne dépend jamais de la présence d'un titre.
    """
    blocs = [b.strip() for b in re.split(r"\n\s*\n", texte) if b.strip()]

    # --- Fusion des titres orphelins avec le bloc suivant ---
    fusionnes_titres = []
    i = 0
    while i < len(blocs):
        if _est_titre_seul(blocs[i]) and i + 1 < len(blocs):
            fusionnes_titres.append(blocs[i] + " " + blocs[i + 1])  # titre + sa clause
            i += 2
        else:
            fusionnes_titres.append(blocs[i])
            i += 1

    # --- Cascade de taille ---
    chunks = []
    for bloc in fusionnes_titres:
        titre = _detecte_titre(bloc)
        if len(bloc) <= MAX_CHARS:
            chunks.append({"text": bloc, "titre": titre})
        else:
            for j, m in enumerate(_subdivise(bloc)):
                chunks.append({"text": m, "titre": titre if j == 0 else None})

    # --- Fusion des fragments trop courts restants ---
    final = []
    for c in chunks:
        if final and len(c["text"]) < MIN_CHARS:
            final[-1]["text"] = (final[-1]["text"] + " " + c["text"]).strip()
        else:
            final.append(c)
    return final


if __name__ == "__main__":
    faux_pdf = """Article 1. Définitions

Au sens du présent contrat, on entend par sinistre tout événement de nature à mettre en jeu la garantie de l'assureur.

Article 5. Exclusions

La garantie ne s'applique pas dans les cas suivants : lorsque le conducteur n'était pas désigné au contrat ; lorsque le véhicule était utilisé à des fins de transport rémunéré de personnes ; lorsque l'assuré a volontairement causé le dommage ; lorsque le sinistre résulte d'un état d'ébriété caractérisé au moment des faits ; lorsque le véhicule n'avait pas passé le contrôle technique obligatoire ; lorsque les clés ont été laissées à l'intérieur du véhicule non verrouillé ; lorsque le sinistre survient hors de la zone géographique définie à l'article 6 ; lorsque l'antivol agréé n'était pas activé au moment du vol.

Article 6. Zone

Suisse et pays limitrophes."""
    for i, c in enumerate(chunk_text(faux_pdf)):
        titre = c["titre"] or "—"
        print(f"\n[{i}] titre={titre}  ({len(c['text'])} car.)")
        print(f"    {c['text'][:110]}{'...' if len(c['text']) > 110 else ''}")