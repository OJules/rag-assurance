"""
Brique de retrieval réutilisable.
Le modèle E5 et la connexion Chroma sont chargés UNE SEULE FOIS, au niveau module
(un import ne s'exécute qu'une fois par processus). Tout le reste du système
importe search() sans se soucier du modèle ni du préfixe "query:".
"""
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer

CHROMA_DIR = Path("chroma_db")
COLLECTION = "polices_assurance"
MODEL_NAME = "intfloat/multilingual-e5-small"

# --- Chargés une fois, à l'import du module ---
print(f"[rag] chargement de {MODEL_NAME} ...")
_model = SentenceTransformer(MODEL_NAME)
_client = chromadb.PersistentClient(path=str(CHROMA_DIR))
_collection = _client.get_collection(COLLECTION)
print(f"[rag] prêt — {_collection.count()} chunks.")


def embed_query(text: str):
    """Encode une question. SEUL endroit où vit le préfixe 'query:' (asymétrie E5)."""
    return _model.encode([f"query: {text}"], normalize_embeddings=True).tolist()


def search(query: str, k: int = 5):
    """
    Retourne les k passages les plus proches, sous forme de liste de dicts :
    {rank, similarity, document_id, section, is_avenant, date_effet_iso, text, metadata}
    """
    q_emb = embed_query(query)
    res = _collection.query(query_embeddings=q_emb, n_results=k)

    hits = []
    for rank, (doc, meta, dist) in enumerate(
        zip(res["documents"][0], res["metadatas"][0], res["distances"][0]), start=1
    ):
        hits.append({
            "rank": rank,
            "similarity": round(1 - dist, 4),   # distance cosinus -> similarité
            "document_id": meta.get("document_id", "??"),
            "section": meta.get("section_title", ""),
            "is_avenant": meta.get("is_avenant", False),
            "date_effet_iso": meta.get("date_effet_iso", ""),
            "text": doc,
            "metadata": meta,
        })
    return hits


# petit test manuel : python rag.py
if __name__ == "__main__":
    for q in [
        "Quelle est la date d'effet de mon assurance auto tous risques ?",
        "Quelles sont toutes les exclusions applicables au vol du contrat AUTO-2024-0137 ?",
        "Suis-je couvert si je fais du VTC avec ma voiture ?",
    ]:
        print(f"\n### {q}")
        for h in search(q, k=3):
            flag = " [AVENANT]" if h["is_avenant"] else ""
            print(f"  {h['rank']}. sim={h['similarity']}  [{h['document_id']} / {h['section'][:40]}]{flag}")