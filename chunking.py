"""
Découpage de texte extrait d'un PDF/Word en chunks.
Python PUR — aucune dépendance externe.

Réalité du texte extrait d'un PDF (constatée sur pdfplumber) :
  - PAS de lignes vides entre les sections (juste des \n simples)
  - les phrases d'une même clause sont parfois éclatées sur plusieurs lignes
  - les débuts de section ("1.", "2.1", "8.", "Article N") restent le seul
    repère de frontière fiable

Stratégie :
  1. RECOLLE les lignes d'une même clause (une ligne qui ne débute pas une
     section rejoint la précédente) -> répare les phrases coupées par le PDF
  2. COUPE à chaque début de section détecté -> une clause = un chunk (+ son titre)
  3. CASCADE de taille (garde-fou) : une clause > MAX_CHARS est subdivisée par phrases
  4. REPLI : si AUCUN début de section n'est trouvé (structure trop abîmée),
     on découpe par taille -> le quality.py signalera la dégradation

Principe : le titre reste un INDICE privilégié de frontière, avec repli.
La taille n'est qu'un garde-fou secondaire.
"""
import re

MAX_CHARS = 1200
MIN_CHARS = 60
OVERLAP   = 150

# Début de SECTION : numéro (éventuellement à sous-niveaux) + point + espace,
# ou mot-clé (Article/Chapitre/Section) + numéro, en TOUT début de ligne.
# "24 premiers mois" n'est PAS un titre (pas de point après le nombre isolé exigé
#  par la forme "N. " ; la forme "N.N" exige au moins un sous-niveau).
RE_DEBUT_SECTION = re.compile(
    r"^\s*(?:"
    r"(?:article|chapitre|section)\s+\d{1,2}\b"        # Article 5, Chapitre 2...
    r"|\d{1,2}\.\d{1,2}(?:\.\d{1,2})?\s+[A-Za-zÀ-ÿ]"  # 2.1 Libellé (sous-niveaux courts)
    r"|\d{1,2}\.\s+[A-Za-zÀ-ÿ]"                        # 1. Libellé (petit num + point + lettre)
    r")",
    re.I,
)
# Note : un TITRE de section = petit numéro (1-2 chiffres) suivi d'un libellé (lettre).
# Cela EXCLUT volontairement les dates (01.01.2025), les numéros longs (0002.599.519)
# et les montants (CHF 5000) — découverts comme faux positifs sur un vrai document.


def _est_debut_section(ligne: str) -> bool:
    return bool(RE_DEBUT_SECTION.match(ligne))


def _titre_de(ligne: str):
    """Extrait le numéro de section en tête de ligne (ou None). Sans la lettre du libellé."""
    m = RE_DEBUT_SECTION.match(ligne)
    if not m:
        return None
    # on renvoie le numéro seul (ex. "2.1", "8.") sans la 1re lettre du libellé capturée
    num = re.match(r"^\s*(?:(?:article|chapitre|section)\s+\d{1,2}|\d{1,2}(?:\.\d{1,2}){0,2}\.?)", ligne, re.I)
    return num.group(0).strip() if num else m.group(0).strip()


def _fin_de_phrase(ligne: str) -> bool:
    return ligne.rstrip().endswith((".", "!", "?", ":", "»"))


def _regrouper_en_sections(texte: str):
    """
    Parcourt les lignes. Ouvre un nouveau bloc à chaque début de section ;
    sinon, rattache la ligne au bloc courant (recolle les phrases éclatées).
    Renvoie une liste de (titre, texte_du_bloc).
    """
    lignes = [l.rstrip() for l in texte.split("\n")]
    sections = []
    titre_courant = None
    buffer = []
    nb_sections_detectees = 0

    for ligne in lignes:
        if not ligne.strip():
            continue
        if _est_debut_section(ligne):
            if buffer:
                sections.append((titre_courant, " ".join(buffer).strip()))
            titre_courant = _titre_de(ligne)
            buffer = [ligne]
            nb_sections_detectees += 1
        else:
            buffer.append(ligne)
    if buffer:
        sections.append((titre_courant, " ".join(buffer).strip()))

    return sections, nb_sections_detectees


def _decoupe_par_phrases(texte: str):
    return [p for p in re.split(r"(?<=[.!?:])\s+", texte.strip()) if p]


def _coupe_brutale(texte: str, taille: int, overlap: int):
    morceaux, i, n = [], 0, len(texte)
    while i < n:
        morceaux.append(texte[i:i + taille])
        i += taille - overlap
    return morceaux


def _subdivise(bloc: str):
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


def _repli_par_taille(texte: str):
    """Aucune section détectée : la structure a disparu. Découpage par taille."""
    flux = " ".join(l.strip() for l in texte.split("\n") if l.strip())
    return [{"text": m, "titre": None} for m in _subdivise(flux)]


def chunk_text(texte: str):
    """
    Renvoie une liste de dicts : {"text": ..., "titre": <titre ou None>}.
    Adapté au texte extrait de PDF (sans lignes vides fiables).
    """
    sections, nb = _regrouper_en_sections(texte)

    # Repli si la structure est trop abîmée (aucune section trouvée)
    if nb == 0:
        return _repli_par_taille(texte)

    chunks = []
    for titre, bloc in sections:
        if len(bloc) <= MAX_CHARS:
            chunks.append({"text": bloc, "titre": titre})
        else:
            for j, m in enumerate(_subdivise(bloc)):
                chunks.append({"text": m, "titre": titre if j == 0 else None})

    # Fusion des fragments trop courts avec le voisin précédent
    final = []
    for c in chunks:
        if final and len(c["text"]) < MIN_CHARS:
            final[-1]["text"] = (final[-1]["text"] + " " + c["text"]).strip()
        else:
            final.append(c)
    return final


if __name__ == "__main__":
    extrait_pdf = (
        "POLICE D'ASSURANCE AUTOMOBILE\nContrat n° AUTO-2024-0137\n"
        "Assureur : Helvétia Synthétique SA\nFormule : Tous risques\n"
        "1. Définitions\n"
        "Véhicule assuré : le véhicule désigné aux conditions particulières.\n"
        "Tiers : toute personne autre que l'assuré, le conducteur et les membres de leur famille vivant\n"
        "sous le même toit.\n"
        "2. Garanties\n"
        "2.1 Responsabilité civile. Couvre les dommages corporels et matériels causés à un tiers du\n"
        "fait du véhicule assuré.\n"
        "Valeur à neuf : applicable pendant les 24 premiers mois suivant la mise en circulation.\n"
    )
    for i, c in enumerate(chunk_text(extrait_pdf)):
        titre = c["titre"] or "—"
        print(f"[{i}] titre={titre:<14} ({len(c['text'])} car.)  {c['text'][:70]}{'...' if len(c['text'])>70 else ''}")