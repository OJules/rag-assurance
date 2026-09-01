"""
Score de qualité d'un chunk extrait d'un document (PDF, Word...).
Python PUR — aucune dépendance externe.

Objectif : quand l'extraction/reconstruction est imparfaite, le SIGNALER
plutôt que de laisser le système répondre sur du texte charcuté.

Catégorie = conséquence de règles nommées, classées par gravité (pas un seuil opaque) :
    bon    = aucun signal
    moyen  = au moins un signal MINEUR, aucun grave
    faible = au moins un signal GRAVE

Signaux (aucun n'est juge unique — c'est leur CONJONCTION qui décide) :
  1. longueur            (fragment orphelin / agglomérat)
  2. debut_coupe / fin_coupe   (phrase tranchée par un saut de page/colonne)
  3. alpha_ratio         (part de lettres : bas = structure potentiellement dégradée)
  4. structure_tabulaire (séparateurs de colonnes répétés : tableau aplati)
"""
import re

LONGUEUR_FRAGMENT = 25
LONGUEUR_COURTE   = 60
LONGUEUR_LONGUE   = 3000
ALPHA_GRAVE       = 0.35
ALPHA_MINEUR      = 0.55
PONCTUATION_FIN   = ".!?:»)\"'"

SEP_TABULAIRE = re.compile(r"(\s\|\s|\s{3,}|•|·|\.{4,}|\t)")
MIN_SEPARATEURS = 2


def score_chunk(text: str) -> dict:
    graves = []
    mineurs = []
    t = (text or "").strip()
    n = len(t)
    signaux = {"longueur": n}

    # --- Signal 1 : longueur ---
    if n < LONGUEUR_FRAGMENT:
        graves.append(f"passage très court ({n} caractères) : fragment probable, pas une vraie clause")
    elif n < LONGUEUR_COURTE:
        mineurs.append(f"passage court ({n} caractères) : à prendre avec prudence")
    elif n > LONGUEUR_LONGUE:
        mineurs.append(f"passage anormalement long ({n} caractères) : découpage peut-être raté")

    # --- Signal 2 : phrase coupée aux bords ---
    lettres = [c for c in t if c.isalpha()]
    debut_coupe = bool(lettres) and lettres[0].islower()
    fin_coupe = bool(t) and t[-1] not in PONCTUATION_FIN
    signaux["debut_coupe"] = debut_coupe
    signaux["fin_coupe"] = fin_coupe
    if debut_coupe and fin_coupe:
        graves.append("phrase coupée au début ET à la fin : passage extrait au milieu d'un flux")
    elif debut_coupe:
        mineurs.append("commence au milieu d'une phrase : début peut-être manquant")
    elif fin_coupe:
        mineurs.append("ne se termine pas par une ponctuation : fin peut-être tronquée")

    # --- Signal 3 : proportion de texte ---
    non_espace = [c for c in t if not c.isspace()]
    alpha_ratio = (sum(1 for c in non_espace if c.isalpha()) / len(non_espace)) if non_espace else 0.0
    signaux["alpha_ratio"] = round(alpha_ratio, 2)
    part_non_texte = round((1 - alpha_ratio) * 100)

    # --- Signal 4 : motif tabulaire ---
    nb_sep = len(SEP_TABULAIRE.findall(t))
    tabulaire = nb_sep >= MIN_SEPARATEURS
    signaux["nb_separateurs"] = nb_sep

    # --- Décision par CONJONCTION (alpha_ratio n'est jamais juge unique) ---
    if alpha_ratio < ALPHA_MINEUR and tabulaire:
        graves.append(f"tableau aplati probable : {part_non_texte}% de caractères non alphabétiques et {nb_sep} séparateurs de colonnes")
    elif alpha_ratio < ALPHA_GRAVE:
        graves.append(f"très forte proportion de caractères non alphabétiques ({part_non_texte}%) : structure dégradée probable")
    elif alpha_ratio < ALPHA_MINEUR:
        mineurs.append(f"proportion élevée de caractères non alphabétiques ({part_non_texte}%)")

    if graves:
        categorie = "faible"
    elif mineurs:
        categorie = "moyen"
    else:
        categorie = "bon"

    return {"categorie": categorie, "raisons": graves + mineurs, "signaux": signaux}


if __name__ == "__main__":
    exemples = {
        "clause propre":
            "La garantie Vol couvre le vol par effraction du véhicule assuré, "
            "sous réserve que l'antivol agréé ait été activé au moment des faits.",
        "phrase coupée (milieu de flux)":
            "notamment lorsque le conducteur n'était pas désigné au contrat et que le sinistre",
        "fragment orphelin":
            "Article 5",
        "fin tronquée":
            "L'assuré doit déclarer tout sinistre dans un délai de cinq jours ouvrables à compter",
        "tableau APLATI":
            "Franchise  |  Vol 750  |  Bris 150  |  Collision 500  |  RC 0  |  ....  p. 4/12",
        "tableau BIEN STRUCTURE (phrase à chiffres)":
            "La franchise vol s'élève à 750 CHF, la franchise bris de glace à 150 CHF, "
            "et la franchise collision à 500 CHF selon les conditions du contrat.",
    }
    for nom, txt in exemples.items():
        r = score_chunk(txt)
        print(f"\n### {nom}  ->  [{r['categorie'].upper()}]")
        print(f"    signaux : {r['signaux']}")
        for raison in r["raisons"]:
            print(f"    - {raison}")
        if not r["raisons"]:
            print("    (aucun signal : passage bien formé)")