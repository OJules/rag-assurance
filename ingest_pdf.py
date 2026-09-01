"""
Ingestion du corpus PDF dans Chroma, AVEC score de qualité par chunk.

Chaîne : PDF -> pdfplumber (texte) -> chunk_text -> score_chunk
         -> multilingual-e5-small (dense, normalisé) -> Chroma (cosinus).

Collection SÉPARÉE (polices_pdf) : ne touche pas à la démo markdown existante.

Lancer :  python ingest_pdf.py
"""
import re
from pathlib import Path

import pdfplumber
import chromadb
from sentence_transformers import SentenceTransformer

from chunking import chunk_text
from quality import score_chunk

PDF_DIR    = Path("pdfs")
CHROMA_DIR = Path("chroma_db_pdf")          # base séparée de la démo markdown
COLLECTION = "polices_pdf"
MODEL_NAME = "intfloat/multilingual-e5-small"

# Liste noire : documents contenant des données personnelles -> JAMAIS ingérés.
# (garde-fou en dur : ne pas se reposer sur un oubli manuel)
EXCLUS = {"policy_H.pdf", "policy_h.pdf"}


def extract_text(pdf_path: Path) -> str:
    morceaux = []
    with pdfplumber.open(str(pdf_path)) as pdf:
        for page in pdf.pages:
            morceaux.append(page.extract_text() or "")
    return "\n\n".join(morceaux)


def document_id_de(texte: str, nom_fichier: str) -> str:
    """
    Identité du document : d'abord un 'Contrat n° ...' si présent dans le texte,
    sinon le nom de fichier nettoyé. Sert à ne pas mélanger les polices au retrieval.
    """
    m = re.search(r"Contrat n[°o]\s*([A-Z0-9\-]+)", texte)
    if m:
        return m.group(1)
    return Path(nom_fichier).stem.replace("_", "-").lower()


def build_chunks(pdf_path: Path):
    texte = extract_text(pdf_path)
    doc_id = document_id_de(texte, pdf_path.name)
    chunks = []
    for i, c in enumerate(chunk_text(texte)):
        q = score_chunk(c["text"])
        titre = c["titre"] or ""
        # préfixe d'identité : un chunk isolé doit savoir d'où il vient
        prefix = f"[{doc_id} — {pdf_path.stem} — Section : {titre or '?'}]"
        chunks.append({
            # id basé sur le NOM DE FICHIER (unique) : deux fichiers peuvent partager
            # le même 'Contrat n°' (ex. police_01 et sa variante tableau).
            "id": f"{pdf_path.stem}::chunk{i}",
            "text": f"{prefix}\n{c['text']}".strip(),
            "metadata": {
                "document_id": doc_id,
                "source_file": pdf_path.name,
                "titre": titre,
                "chunk_index": i,
                "quality": q["categorie"],                       # bon / moyen / faible
                "quality_raisons": " ; ".join(q["raisons"]) or "aucun signal",
            },
        })
    return chunks


def main():
    fichiers = [p for p in sorted(PDF_DIR.glob("*.pdf")) if p.name not in EXCLUS]
    exclus = [p.name for p in sorted(PDF_DIR.glob("*.pdf")) if p.name in EXCLUS]
    if exclus:
        print(f"Exclus (données personnelles) : {', '.join(exclus)}")
    print(f"{len(fichiers)} PDF à ingérer : {', '.join(p.name for p in fichiers)}")

    all_chunks = []
    for f in fichiers:
        cs = build_chunks(f)
        all_chunks.extend(cs)
        # petit récap qualité par document
        cats = {"bon": 0, "moyen": 0, "faible": 0}
        for c in cs:
            cats[c["metadata"]["quality"]] += 1
        print(f"  {f.name:<40} {len(cs):>3} chunks  "
              f"(bon {cats['bon']}, moyen {cats['moyen']}, faible {cats['faible']})")

    print(f"\nTotal : {len(all_chunks)} chunks. Chargement de {MODEL_NAME} ...")
    model = SentenceTransformer(MODEL_NAME)

    texts = [c["text"] for c in all_chunks]
    passages = [f"passage: {t}" for t in texts]     # E5 asymétrique : préfixe documents
    embeddings = model.encode(passages, normalize_embeddings=True,
                              show_progress_bar=True).tolist()

    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    try:
        client.delete_collection(COLLECTION)
    except Exception:
        pass
    collection = client.create_collection(COLLECTION, metadata={"hnsw:space": "cosine"})
    collection.add(
        ids=[c["id"] for c in all_chunks],
        embeddings=embeddings,
        documents=texts,
        metadatas=[c["metadata"] for c in all_chunks],
    )
    print(f"\nCollection '{COLLECTION}' : {collection.count()} chunks dans {CHROMA_DIR}/")

    # bilan qualité global
    cats = {"bon": 0, "moyen": 0, "faible": 0}
    for c in all_chunks:
        cats[c["metadata"]["quality"]] += 1
    tot = len(all_chunks)
    print("Bilan qualité global :")
    for k in ("bon", "moyen", "faible"):
        print(f"  {k:<6} {cats[k]:>3} ({cats[k]/tot*100:.0f}%)")


if __name__ == "__main__":
    main()