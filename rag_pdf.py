"""
Brique de retrieval pour le corpus PDF (collection 'polices_pdf').
Calquée sur rag.py, mais :
  - pointe sur la base chroma_db_pdf / collection polices_pdf
  - remonte la QUALITÉ de chaque chunk (quality, quality_raisons)
    -> c'est ce qui permet d'afficher la fiabilité de la source dans l'app.

Le modèle E5 et Chroma sont chargés UNE FOIS à l'import (comme rag.py).
"""
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer

CHROMA_DIR = Path("chroma_db_pdf")
COLLECTION = "polices_pdf"
MODEL_NAME = "intfloat/multilingual-e5-small"

print(f"[rag_pdf] chargement de {MODEL_NAME} ...")
_model = SentenceTransformer(MODEL_NAME)
_client = chromadb.PersistentClient(path=str(CHROMA_DIR))
_collection = _client.get_collection(COLLECTION)
print(f"[rag_pdf] prêt - {_collection.count()} chunks.")


def embed_query(text: str):
    """Encode une question. Seul endroit où vit le préfixe 'query:' (asymétrie E5)."""
    return _model.encode([f"query: {text}"], normalize_embeddings=True).tolist()


def search(query: str, k: int = 5):
    """
    Retourne les k passages les plus proches, avec leur QUALITÉ :
    {rank, similarity, document_id, source_file, titre, quality, quality_raisons, text, metadata}
    """
    q_emb = embed_query(query)
    res = _collection.query(query_embeddings=q_emb, n_results=k)
    hits = []
    for rank, (doc, meta, dist) in enumerate(
        zip(res["documents"][0], res["metadatas"][0], res["distances"][0]), start=1
    ):
        hits.append({
            "rank": rank,
            "similarity": round(1 - dist, 4),
            "document_id": meta.get("document_id", "??"),
            "source_file": meta.get("source_file", ""),
            "titre": meta.get("titre", ""),
            "quality": meta.get("quality", "?"),                    # bon / moyen / faible
            "quality_raisons": meta.get("quality_raisons", ""),     # le pourquoi, en clair
            "text": doc,
            "metadata": meta,
        })
    return hits


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