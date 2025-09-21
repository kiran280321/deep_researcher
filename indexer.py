import os
import argparse
import pickle
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
from tqdm import tqdm
from utils import read_pdf, read_text_file, chunk_text, clean_text, list_documents
INDEX_PATH = "faiss.index"
META_PATH = "meta.pkl"
EMB_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
def build_index(data_dir: str, index_path=INDEX_PATH, meta_path=META_PATH, emb_model_name=EMB_MODEL_NAME):
    model = SentenceTransformer(emb_model_name)
    docs = list_documents(data_dir)
    all_chunks = []
    metadata = []
    for doc_path in tqdm(docs, desc="Reading docs"):
        if doc_path.lower().endswith(".pdf"):
            text = read_pdf(doc_path)
        else:
            text = read_text_file(doc_path)
        text = clean_text(text)
        chunks = chunk_text(text, chunk_size=300, overlap=50)
        for i, ch in enumerate(chunks):
            all_chunks.append(ch)
            metadata.append({"source": doc_path, "chunk_id": i, "text": ch})
    print(f"[indexer] Total chunks: {len(all_chunks)}")
    if len(all_chunks) == 0:
        raise ValueError("No documents or chunks found in data directory.")
    embeddings = model.encode(all_chunks, show_progress_bar=True, convert_to_numpy=True)
    d = embeddings.shape[1]
    index = faiss.IndexFlatIP(d) 
    faiss.normalize_L2(embeddings)
    index.add(embeddings)
    faiss.write_index(index, index_path)
    with open(meta_path, "wb") as f:
        pickle.dump(metadata, f)
    print(f"[indexer] Saved FAISS index to {index_path} and metadata to {meta_path}")
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data/docs")
    parser.add_argument("--index_path", type=str, default=INDEX_PATH)
    parser.add_argument("--meta_path", type=str, default=META_PATH)
    args = parser.parse_args()
    build_index(args.data_dir, args.index_path, args.meta_path)
