"""
Convertit une police markdown (corpus/*.md) en PDF.
Étape 1 : PDF PROPRE (une colonne, texte linéaire) pour valider la chaîne d'extraction.

Usage : python make_pdf.py corpus/police_01_auto_tousrisques.md pdfs/police_01.pdf
"""
import sys
import re
from pathlib import Path

from fpdf import FPDF


def md_to_pdf(md_path: str, pdf_path: str):
    text = Path(md_path).read_text(encoding="utf-8")

    pdf = FPDF()
    pdf.set_margins(left=20, top=20, right=20)          # marges explicites
    pdf.set_auto_page_break(auto=True, margin=20)
    pdf.add_page()
    pdf.set_font("Helvetica", "", 11)                   # police posée AVANT toute écriture

    epw = pdf.w - pdf.l_margin - pdf.r_margin           # largeur utile explicite

    for ligne in text.split("\n"):
        ligne = ligne.rstrip()

        if ligne.startswith("## "):
            pdf.set_font("Helvetica", "B", 13)
            contenu = ligne[3:]
        elif ligne.startswith("# "):
            pdf.set_font("Helvetica", "B", 15)
            contenu = ligne[2:]
        elif ligne.strip() == "":
            pdf.ln(4)                                   # ligne vide -> frontière de paragraphe
            continue
        else:
            pdf.set_font("Helvetica", "", 11)
            contenu = ligne

        contenu = re.sub(r"\*\*(.+?)\*\*", r"\1", contenu)          # retire le gras markdown
        contenu = contenu.encode("latin-1", "replace").decode("latin-1")  # évite un crash d'encodage

        if contenu.strip() == "":
            pdf.ln(4)
            continue

        pdf.multi_cell(epw, 6, contenu)                 # largeur explicite, plus de 0

    Path(pdf_path).parent.mkdir(parents=True, exist_ok=True)
    pdf.output(pdf_path)
    print(f"PDF écrit : {pdf_path}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage : python make_pdf.py <entrée.md> <sortie.pdf>")
        sys.exit(1)
    md_to_pdf(sys.argv[1], sys.argv[2])