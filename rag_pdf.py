"""
Brique de retrieval pour le corpus PDF (collection 'polices_pdf').
Calquée sur rag.py, mais :
  - pointe sur la base chroma_db_pdf / collection polices_pdf
  - remonte la QUALITÉ de chaque chunk (quality, quality_raisons)
    -> c'est ce qui permet d'afficher la fiabilité de la source dans l'app.
  - propose une recherche HYBRIDE (search_hybride) : le sens de la question
    (embeddings E5) ET ses mots (BM25), fusionnés en tour de table.

Pourquoi l'hybride : sur de vrais contrats, la recherche par le sens seule rate
des clauses dont le mot-clé est pourtant dans la question ("chaleur", "jardin").
La recherche par les mots est déterministe, gratuite, et explicable : on peut
dire quel mot a fait remonter quel passage.

(Une étape « analyse du sinistre » par le LLM a été essayée puis RETIRÉE : mesurée
sur les deux questions de test, elle paraphrasait la question au lieu de la
traduire, requalifiait les faits, et coûtait ~1500 tokens sans gain de rang.)

Le modèle E5 et Chroma sont chargés UNE FOIS à l'import (comme rag.py).
"""
import math
import re
import unicodedata
from collections import Counter
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer

try:                                    # racinisation française : "voler" et "vol" -> "vol"
    import snowballstemmer
    _stem = snowballstemmer.stemmer("french").stemWord
except ImportError:                     # repli minimal si le paquet manque : pluriels seulement
    def _stem(mot):
        return mot[:-1] if len(mot) > 4 and mot[-1] in "sx" else mot

CHROMA_DIR = Path("chroma_db_pdf")
COLLECTION = "polices_pdf"
MODEL_NAME = "intfloat/multilingual-e5-small"

print(f"[rag_pdf] chargement de {MODEL_NAME} ...")
_model = SentenceTransformer(MODEL_NAME)
_client = chromadb.PersistentClient(path=str(CHROMA_DIR))
_collection = _client.get_collection(COLLECTION)
print(f"[rag_pdf] prêt - {_collection.count()} chunks.")


# ------------------------------------------------------------------ recherche par le SENS
def embed_query(text: str):
    """Encode une question. Seul endroit où vit le préfixe 'query:' (asymétrie E5)."""
    return _model.encode([f"query: {text}"], normalize_embeddings=True).tolist()


def _hit(rank, doc, meta, dist=None):
    """Met un résultat au format attendu par llm.py et app.py."""
    return {
        "rank": rank,
        "similarity": None if dist is None else round(1 - dist, 4),
        "document_id": meta.get("document_id", "??"),
        "source_file": meta.get("source_file", ""),
        "titre": meta.get("titre", ""),
        "section": meta.get("titre", ""),                       # alias : llm.py attend 'section'
        "quality": meta.get("quality", "?"),                    # bon / moyen / faible
        "quality_raisons": meta.get("quality_raisons", ""),     # le pourquoi, en clair
        "text": doc,
        "metadata": meta,
    }


def search(query: str, k: int = 5, filtre_fichier: str = None):
    """
    Retourne les k passages les plus proches PAR LE SENS, avec leur QUALITÉ :
    {rank, similarity, document_id, source_file, titre, quality, quality_raisons, text, metadata}
    """
    # filtre optionnel : ne chercher que dans UN document (par nom de fichier).
    # C'est la clé du passage à l'échelle : on ne mélange pas les contrats.
    where = {"source_file": filtre_fichier} if filtre_fichier else None
    res = _collection.query(query_embeddings=embed_query(query), n_results=k, where=where)
    return [
        _hit(rank, doc, meta, dist)
        for rank, (doc, meta, dist) in enumerate(
            zip(res["documents"][0], res["metadatas"][0], res["distances"][0]), start=1
        )
    ]


# ------------------------------------------------------------------ recherche par les MOTS (BM25)
_MOTS_VIDES = """au aux avec ce ces cet cette dans de des du elle en et eux il ils je la le les
leur leurs lui ma mais me mes moi mon ne nos notre nous on ou par pas pour que qui sa se ses son
sur ta te tes toi ton tu un une vos votre vous est sont ont suis ete etre avoir fait si pendant
qui que quoi dont lors ainsi tout tous toute toutes plus moins comme entre sans sous
fais faire etait etais ete avait ont cela ceci donc alors aussi tres bien""".split()
_MOTS_VIDES = set(_MOTS_VIDES)

K1, B = 1.5, 0.75           # réglages BM25 usuels
_INDEX_LEX = {}             # un index par contrat, construit au premier besoin


def _sans_accents(texte):
    t = unicodedata.normalize("NFKD", texte.lower())
    return "".join(c for c in t if not unicodedata.combining(c))


def _tokens(texte):
    """Découpe en mots utiles : minuscules, sans accents, sans mots vides, racinisés."""
    mots = re.findall(r"[a-z0-9]+", _sans_accents(texte))
    return [_stem(m) for m in mots if len(m) > 2 and m not in _MOTS_VIDES]


def _sans_entete(doc):
    """Retire la ligne d'en-tête '[fichier — Section : ...]' ajoutée à l'ingestion."""
    if doc.startswith("["):
        c = doc.find("]\n")
        if c != -1:
            return doc[c + 2:]
    return doc


def _index_lexical(filtre_fichier=None):
    """Construit (une fois) l'index BM25 d'un contrat, ou de tout le corpus si filtre vide."""
    cle = filtre_fichier or "*"
    if cle not in _INDEX_LEX:
        where = {"source_file": filtre_fichier} if filtre_fichier else None
        data = _collection.get(where=where, include=["documents", "metadatas"])
        sacs = [Counter(_tokens(_sans_entete(d))) for d in data["documents"]]
        n = len(sacs)
        df = Counter(t for sac in sacs for t in sac)
        longueurs = [sum(sac.values()) for sac in sacs]
        _INDEX_LEX[cle] = {
            "ids": data["ids"], "documents": data["documents"], "metadatas": data["metadatas"],
            "sacs": sacs, "longueurs": longueurs,
            "moyenne": (sum(longueurs) / n) if n else 0.0,
            # un mot rare dans le contrat pèse lourd ; un mot présent partout ne pèse rien
            "idf": {t: math.log(1 + (n - d + 0.5) / (d + 0.5)) for t, d in df.items()},
        }
    return _INDEX_LEX[cle]


def search_lexical(query: str, k: int = 5, filtre_fichier: str = None):
    """
    Retourne les k passages qui partagent le plus de mots RARES avec la question (BM25).
    Chaque hit porte 'mots_trouves' : les mots de la question présents dans le passage.
    """
    ix = _index_lexical(filtre_fichier)
    q = list(dict.fromkeys(_tokens(query)))
    notes = []
    for i, sac in enumerate(ix["sacs"]):
        score, trouves = 0.0, []
        for t in q:
            tf = sac.get(t, 0)
            if not tf:
                continue
            norme = 1 - B + B * ix["longueurs"][i] / (ix["moyenne"] or 1)
            score += ix["idf"][t] * tf * (K1 + 1) / (tf + K1 * norme)
            trouves.append(t)
        if score > 0:
            notes.append((score, i, trouves))
    notes.sort(key=lambda x: -x[0])
    hits = []
    for rank, (score, i, trouves) in enumerate(notes[:k], start=1):
        h = _hit(rank, ix["documents"][i], ix["metadatas"][i])
        h["_id"] = ix["ids"][i]
        h["bm25"] = round(score, 3)
        h["mots_trouves"] = trouves
        hits.append(h)
    return hits


# ------------------------------------------------------------------ recherche HYBRIDE
def search_hybride(query: str, k: int = 8, filtre_fichier: str = None):
    """
    Fusionne la recherche par le sens et la recherche par les mots en tour de table :
    1er passage par le sens, 1er par les mots, 2e par le sens, 2e par les mots...
    sans doublon, jusqu'à k passages.

    Le tour de table garantit que le meilleur passage de CHAQUE méthode est lu,
    sans avoir à comparer deux scores qui n'ont pas la même échelle.
    Chaque hit porte 'trouve_par' : ["sens"], ["mots"] ou les deux.
    """
    where = {"source_file": filtre_fichier} if filtre_fichier else None
    res = _collection.query(query_embeddings=embed_query(query), n_results=k, where=where)
    par_sens = [
        dict(_hit(0, doc, meta, dist), _id=cid)
        for cid, doc, meta, dist in zip(res["ids"][0], res["documents"][0],
                                        res["metadatas"][0], res["distances"][0])
    ]
    par_mots = search_lexical(query, k=k, filtre_fichier=filtre_fichier)
    ids_sens = {h["_id"] for h in par_sens}
    ids_mots = {h["_id"]: h for h in par_mots}

    hits, pris = [], set()
    for rang in range(k):
        for methode, liste in (("sens", par_sens), ("mots", par_mots)):
            if rang >= len(liste) or liste[rang]["_id"] in pris:
                continue
            h = dict(liste[rang])
            pris.add(h["_id"])
            autre = "mots" if methode == "sens" else "sens"
            dans_autre = h["_id"] in (ids_mots if methode == "sens" else ids_sens)
            h["trouve_par"] = [methode] + ([autre] if dans_autre else [])
            h["mots_trouves"] = ids_mots[h["_id"]]["mots_trouves"] if h["_id"] in ids_mots else []
            h["rank"] = len(hits) + 1
            hits.append(h)
            if len(hits) >= k:
                return hits
    return hits


# ------------------------------------------------------------------ choix des contrats
def documents_pertinents(query: str, n_docs: int = 3, k_scan: int = 15):
    """
    Identifie les contrats les plus pertinents pour une question.

    On fait un retrieval global large (k_scan), puis on donne un score à chaque
    FICHIER : la somme de 1 / (10 + rang) sur ses 3 meilleurs passages. Un contrat
    qui a plusieurs passages proches passe ainsi devant un contrat qui n'en a qu'un.
    Renvoie une liste de source_file, du plus pertinent au moins pertinent.
    """
    res = _collection.query(query_embeddings=embed_query(query), n_results=k_scan)
    scores, compte = {}, {}
    for rang, meta in enumerate(res["metadatas"][0], start=1):
        f = meta.get("source_file", "?")
        if compte.get(f, 0) >= 3:
            continue
        compte[f] = compte.get(f, 0) + 1
        scores[f] = scores.get(f, 0.0) + 1.0 / (10 + rang)
    return sorted(scores, key=lambda f: -scores[f])[:n_docs]


def documents_disponibles():
    """Liste les noms de fichiers présents dans la collection (pour un menu de filtre)."""
    data = _collection.get(include=["metadatas"])
    fichiers = sorted({m.get("source_file", "?") for m in data["metadatas"]})
    return fichiers


if __name__ == "__main__":
    flag = {"bon": "🟢", "moyen": "🟠", "faible": "🔴"}
    for q in [
        "Quelle est la franchise en cas de bris de glace ?",
        "Suis-je couvert si je fais du VTC avec ma voiture ?",
        "Quelles exclusions s'appliquent à la garantie vol ?",
    ]:
        print(f"\n### {q}")
        for h in search(q, k=3):
            f = flag.get(h["quality"], "?")
            print(f"  {h['rank']}. sim={h['similarity']}  {f} {h['quality'].upper():<6} "
                  f"[{h['document_id']} / {h['source_file']}]")
            if h["quality"] != "bon":
                print(f"       qualité : {h['quality_raisons']}")