import os
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from sklearn.neighbors import NearestNeighbors
try:
    import faiss
    _FAISS = True
except Exception:
    _FAISS = False

class DenseIndex:
    def __init__(self, product_df, model_name="sentence-transformers/all-MiniLM-L6-v2", normalize=True):
        self.df = product_df.reset_index(drop=True)
        self.model_name = model_name
        self.normalize = normalize
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = SentenceTransformer("./wands_search/models/all-MiniLM-L6-v2", device=self.device)
        self.emb = None
        self.index = None

    def fit(self, batch_size=256):
        texts = (self.df['product_name'].fillna('') + ' ' + self.df['product_description'].fillna('')).tolist()
        self.emb = self.model.encode(texts, batch_size=batch_size, show_progress_bar=True,
                                     normalize_embeddings=self.normalize).astype("float32")
        np.save("embeddings.npy", self.emb)

        if _FAISS:
            d = self.emb.shape[1]
            d = self.emb.shape[1]  # obtiene dimensión real de los embeddings
            self.index = faiss.IndexFlatIP(d)

            self.index.add(self.emb)
        else:
            self.index = NearestNeighbors(n_neighbors=50, metric="cosine").fit(self.emb)
        return self

    def load_cached(self, emb_path="embeddings.npy"):
        self.emb = np.load(emb_path)
        if _FAISS:
            d = self.emb.shape[1]
            self.index = faiss.IndexFlatIP(d)
            self.index.add(self.emb)
        else:
            self.index = NearestNeighbors(n_neighbors=50, metric="cosine").fit(self.emb)
        return self

    def search(self, query: str, k=10):
        q = self.model.encode([query], normalize_embeddings=self.normalize).astype("float32")
        if _FAISS:
            sims, idx = self.index.search(q, k)
            ids = self.df.iloc[idx[0]]["product_id"].tolist()
            scores = sims[0].tolist()
        else:
            dist, idx = self.index.kneighbors(q, n_neighbors=k, return_distance=True)
            ids = self.df.iloc[idx[0]]["product_id"].tolist()
            scores = (1.0 - dist[0]).tolist()
        return ids, scores
