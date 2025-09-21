import faiss
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
INDEX_PATH = "faiss.index"
META_PATH = "meta.pkl"
EMB_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
class Retriever:
    def __init__(self, index_path=INDEX_PATH, meta_path=META_PATH, emb_model_name=EMB_MODEL_NAME):
        self.index = faiss.read_index(index_path)
        with open(meta_path, "rb") as f:
            self.metadata = pickle.load(f)
        self.model = SentenceTransformer(emb_model_name)
    def query(self, text: str, top_k: int = 5):
        q_emb = self.model.encode([text], convert_to_numpy=True)
        faiss.normalize_L2(q_emb)
        D, I = self.index.search(q_emb, top_k)
        results = []
        for idx, score in zip(I[0], D[0]):
            if idx < 0 or idx >= len(self.metadata):
                continue
            meta = self.metadata[idx]
            results.append({"score": float(score), "source": meta["source"], "chunk_id": meta["chunk_id"], "text": meta["text"]})
        return results
