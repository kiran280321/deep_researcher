import os
import re
from typing import List
from PyPDF2 import PdfReader
def clean_text(s: str) -> str:
    s = s.replace("\n", " ").strip()
    s = re.sub(r"\s+", " ", s)
    return s
def read_pdf(path: str) -> str:
    text_parts = []
    reader = PdfReader(path)
    for p in reader.pages:
        try:
            text_parts.append(p.extract_text() or "")
        except Exception:
            continue
    return "\n".join(text_parts)
def read_text_file(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()
def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
    tokens = text.split()
    chunks = []
    i = 0
    while i < len(tokens):
        chunk_tokens = tokens[i:i+chunk_size]
        chunks.append(" ".join(chunk_tokens))
        i += chunk_size - overlap
    return chunks
def list_documents(data_dir: str):
    docs = []
    for root, _, files in os.walk(data_dir):
        for f in files:
            if f.lower().endswith((".pdf", ".txt", ".md")):
                docs.append(os.path.join(root, f))
    return docs
