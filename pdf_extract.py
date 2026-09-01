"""
Extraction d'un PDF -> texte brut -> chunking -> score qualité.
Premier test de bout en bout de la chaîne PDF.

Usage : python pdf_extract.py pdfs/police_01.pdf
"""
import sys

import pdfplumber

from chunking import chunk_text
from quality import score_chunk


def extract_text(pdf_path: str) -> str:
    """Extrait le texte de toutes les pages, séparées par un saut de ligne."""
    morceaux = []
    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages, start=1):
            txt = page.extract_text() or ""
            morceaux.append(txt)
            print(f"[extract] page {i} : {len(txt)} caractères")
    return "\n\n".join(morceaux)


def main(pdf_path: str):
    print(f"\n=== Extraction : {pdf_path} ===")
    texte = extract_text(pdf_path)
    print(f"[extract] total : {len(texte)} caractères\n")

    chunks = chunk_text(texte)
    print(f"=== Chunking : {len(chunks)} chunks ===\n")

    # récap qualité
    compteur = {"bon": 0, "moyen": 0, "faible": 0}
    for i, c in enumerate(chunks):
        r = score_chunk(c["text"])
        compteur[r["categorie"]] += 1
        titre = c["titre"] or "—"
        flag = {"bon": "🟢", "moyen": "🟠", "faible": "🔴"}[r["categorie"]]
        print(f"[{i:>2}] {flag} {r['categorie'].upper():<6} titre={titre:<12} ({len(c['text'])} car.)")
        print(f"     {c['text'][:90]}{'...' if len(c['text']) > 90 else ''}")
        for raison in r["raisons"]:
            print(f"       · {raison}")

    print(f"\n=== Bilan qualité ===")
    total = len(chunks)
    for cat in ("bon", "moyen", "faible"):
        n = compteur[cat]
        pct = (n / total * 100) if total else 0
        print(f"  {cat:<6} : {n:>2} chunks ({pct:.0f}%)")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage : python pdf_extract.py <fichier.pdf>")
        sys.exit(1)
    main(sys.argv[1])