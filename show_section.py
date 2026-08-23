"""Imprime, pour chaque police, l'index de section et son titre.
Sert à vérifier les gold_sections du ground_truth AVANT de mesurer."""
import re
from pathlib import Path

CORPUS_DIR = Path("corpus")

def split_sections(text):
    parts = re.split(r"\n##\s+", text)
    sections = [("En-tête (identité et dates)", parts[0])]
    for p in parts[1:]:
        line, _, _ = p.partition("\n")
        sections.append((line.strip(), ""))
    return sections

for f in sorted(CORPUS_DIR.glob("police_*.md")):
    text = f.read_text(encoding="utf-8")
    m = re.search(r"Contrat n°\s*([A-Z0-9\-]+)", text)
    doc_id = m.group(1) if m else "??"
    print(f"\n=== {doc_id}  ({f.name}) ===")
    for i, (title, _) in enumerate(split_sections(text)):
        print(f"  sec{i:>2} : {title}")