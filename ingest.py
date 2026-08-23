"""
Ingestion du corpus d'assurance dans Chroma.
Chaîne : polices .md -> découpage par section (+ métadonnées)
         -> multilingual-e5-small (dense, normalisé) -> Chroma (cosinus).

Lancer une fois :  python ingest.py
Premier lancement : télécharge le modèle (~470 Mo). Ensuite c'est instantané.

Note E5 : le modèle est ASYMÉTRIQUE. Les documents doivent être préfixés
par "passage: " et les questions par "query: " avant l'encodage.
"""
import re
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer

# --- Chemins (adapte si besoin) ---
CORPUS_DIR = Path("corpus")
CHROMA_DIR = Path("chroma_db")
COLLECTION = "polices_assurance"
MODEL_NAME = "intfloat/multilingual-e5-small"

# ---------------------------------------------------------------------------
# 1. Découpage + métadonnées
# ---------------------------------------------------------------------------
MOIS = {"janvier": 1, "février": 2, "mars": 3, "avril": 4, "mai": 5, "juin": 6,
        "juillet": 7, "août": 8, "septembre": 9, "octobre": 10, "novembre": 11, "décembre": 12}

def clean(s: str) -> str:
    return s.replace("*", "").strip()

def fr_date_to_iso(s: str) -> str:
    """'1er mars 2024' -> '2024-03-01'.  Sert aux comparaisons temporelles."""
    if not s:
        return ""
    s = s.strip().lower().replace("1er", "1")
    m = re.search(r"(\d{1,2})\s+([a-zûé]+)\s+(\d{4})", s)
    if not m:
        return ""
    d, mois, y = int(m.group(1)), m.group(2), int(m.group(3))
    return f"{y:04d}-{MOIS.get(mois, 0):02d}-{d:02d}" if mois in MOIS else ""

def extract_doc_metadata(text: str) -> dict:
    md = {}
    m = re.search(r"#\s*POLICE D'ASSURANCE\s+(.+)", text)
    md["doc_type"] = clean(m.group(1)).title() if m else "Inconnu"
    m = re.search(r"Contrat n°\s*([A-Z0-9\-]+)", text)
    md["document_id"] = m.group(1) if m else "??"
    m = re.search(r"Formule\s*:\s*(.+)", text)
    md["formule"] = clean(m.group(1)) if m else ""
    m = re.search(r"Date d'effet\s*:\s*([^\n*]+)", text)
    md["date_effet"] = clean(m.group(1)) if m else ""
    md["date_effet_iso"] = fr_date_to_iso(md["date_effet"])
    m = re.search(r"Date d'échéance\s*:\s*([^\n*]+)", text)
    md["date_echeance"] = clean(m.group(1)) if m else ""
    md["date_echeance_iso"] = fr_date_to_iso(md["date_echeance"])
    return md

def split_sections(text: str):
    parts = re.split(r"\n##\s+", text)
    sections = [("En-tête (identité et dates)", parts[0])]
    for p in parts[1:]:
        line, _, body = p.partition("\n")
        sections.append((line.strip(), body.strip()))
    return sections

def build_chunks(path: Path):
    text = path.read_text(encoding="utf-8")
    doc_md = extract_doc_metadata(text)
    chunks = []
    for i, (title, body) in enumerate(split_sections(text)):
        # préfixe d'identité : sans lui, un chunk "Franchise : 500 CHF" ne sait
        # pas de quelle police il vient -> impossible de désambiguïser R1/R2.
        prefix = f"[{doc_md['document_id']} — {doc_md['doc_type']} {doc_md['formule']} — Section : {title}]"
        chunks.append({
            "id": f"{doc_md['document_id']}::sec{i}",
            "text": f"{prefix}\n{body}".strip(),
            "metadata": {
                **doc_md,
                "section_title": title,
                "section_index": i,
                "is_avenant": "avenant" in title.lower(),
                "source_file": path.name,
            },
        })
    return chunks

# ---------------------------------------------------------------------------
# 2. Embedding + stockage
# ---------------------------------------------------------------------------
def main():
    files = sorted(CORPUS_DIR.glob("police_*.md"))
    all_chunks = [c for f in files for c in build_chunks(f)]
    print(f"{len(all_chunks)} chunks issus de {len(files)} polices.")

    print(f"Chargement de {MODEL_NAME} ...")
    model = SentenceTransformer(MODEL_NAME)

    texts = [c["text"] for c in all_chunks]
    # E5 est asymétrique : on préfixe les documents par "passage: " pour l'encodage.
    # Le préfixe ne sert QU'À l'encodage : on stocke le texte brut (documents=texts).
    passages = [f"passage: {t}" for t in texts]
    # normalize_embeddings=True -> vecteurs de norme 1 -> produit scalaire = cosinus
    embeddings = model.encode(passages, normalize_embeddings=True,
                              show_progress_bar=True).tolist()

    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    # on repart propre à chaque ingestion
    try:
        client.delete_collection(COLLECTION)
    except Exception:
        pass
    # hnsw:space = cosine : on FORCE la métrique, sinon Chroma prend L2 par défaut
    collection = client.create_collection(COLLECTION, metadata={"hnsw:space": "cosine"})

    collection.add(
        ids=[c["id"] for c in all_chunks],
        embeddings=embeddings,
        documents=texts,            # texte brut, sans le préfixe "passage: "
        metadatas=[c["metadata"] for c in all_chunks],
    )
    print(f"Collection '{COLLECTION}' : {collection.count()} chunks stockés dans {CHROMA_DIR}/")

    # -------------------------------------------------------------------
    # 3. Test de bon sens : une requête, on regarde ce qui remonte
    # -------------------------------------------------------------------
    q = "Quelle est la date d'effet de mon assurance auto tous risques ?"
    # côté requête : préfixe "query: " (asymétrie E5)
    q_emb = model.encode([f"query: {q}"], normalize_embeddings=True).tolist()
    res = collection.query(query_embeddings=q_emb, n_results=3)
    print(f"\nRequête test : {q}")
    for doc, meta, dist in zip(res["documents"][0], res["metadatas"][0], res["distances"][0]):
        sim = 1 - dist  # cosine distance -> similarité
        print(f"  sim={sim:.3f}  [{meta['document_id']} / {meta['section_title'][:40]}]")

if __name__ == "__main__":
    main()