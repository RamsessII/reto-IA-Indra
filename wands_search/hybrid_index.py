import pandas as pd
from .index import MultiFieldIndex
from .dense_index import DenseIndex
from .fusion import rrf_fuse

class HybridIndex:
    """
    Ejecuta búsqueda léxica (MultiFieldIndex) + densa (DenseIndex) y fusiona con RRF.
    """
    def __init__(self, product_df: pd.DataFrame,
                 lexical_index: MultiFieldIndex | None = None,
                 dense_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
                 k_lex: int = 50, k_dense: int = 50):
        self.df = product_df.reset_index(drop=True)
        self.k_lex = k_lex
        self.k_dense = k_dense
        self.lex = lexical_index or MultiFieldIndex(self.df).fit()
        self.dense = DenseIndex(self.df, model_name=dense_model_name).fit()

    def search(self, query: str, k: int = 10):
        lex_ids, _ = self.lex.search(query, k=self.k_lex)
        den_ids, _ = self.dense.search(query, k=self.k_dense)
        fused_ids, fused_scores = rrf_fuse([lex_ids, den_ids], k=k, K=60)
        return fused_ids, fused_scores
