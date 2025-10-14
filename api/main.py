import os, logging
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field
from typing import List, Optional
from wands_search.index import MultiFieldIndex
from wands_search.config import VectorizerParams, FieldWeights, LabelGains
from wands_search.evaluate import evaluate_queries
from wands_search.dense_index import DenseIndex
from wands_search.hybrid_index import HybridIndex

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("wands-api")

DATA_DIR = os.getenv("DATA_DIR", "WANDS/dataset")
K_DEFAULT = int(os.getenv("TOP_K", "10"))

app = FastAPI(title="WANDS Search API", version="1.0.0")

class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1)
    k: int = Field(K_DEFAULT, ge=1, le=50)

class SearchHit(BaseModel):
    product_id: int
    score: float

class SearchResponse(BaseModel):
    hits: List[SearchHit]

product_df = query_df = label_df = None
index_lex = index_dense = index_hybrid = None

gains = LabelGains()

@app.on_event("startup")
def startup_event():
    global product_df, query_df, label_df, index_lex, index_dense, index_hybrid
    try:
        product_df = pd.read_csv(f"{DATA_DIR}/product.csv", sep="\t")
        query_df   = pd.read_csv(f"{DATA_DIR}/query.csv", sep="\t")
        label_df   = pd.read_csv(f"{DATA_DIR}/label.csv",  sep="\t")

        index_lex = MultiFieldIndex(product_df).fit()

        if os.path.exists("embeddings.npy"):
            index_dense = DenseIndex(product_df).load_cached()
        else:
            index_dense = DenseIndex(product_df).fit()

        index_hybrid = HybridIndex(product_df, lexical_index=index_lex)

        logger.info("Indices construidos, Productos=%d", len(product_df))
    except Exception as e:
        logger.exception("Error inicializando el servicio")
        raise


@app.get("/health")
def health():
    ok = all([
        product_df is not None,
        index_lex is not None,
        index_dense is not None,
        index_hybrid is not None
    ])
    return {"status": "ok" if ok else "no ok"}


@app.post("/search", response_model=SearchResponse)
def search(req: SearchRequest, mode: str = Query("hybrid", regex="^(lexical|dense|hybrid)$")):
    """
    mode puede ser:
      - lexical: usa TF-IDF (MultiFieldIndex)
      - dense: usa embeddings (DenseIndex)
      - hybrid: combina ambos con RRF
    """
    try:
        if mode == "lexical":
            ids, scores = index_lex.search(req.query, k=req.k)
        elif mode == "dense":
            ids, scores = index_dense.search(req.query, k=req.k)
        else: #"hybrid"
            ids, scores = index_hybrid.search(req.query, k=req.k)

        hits = [SearchHit(product_id=int(i), score=float(s)) for i, s in zip(ids, scores)]
        return SearchResponse(hits=hits)
    except Exception as e:
        logger.exception("Fallo en el end /search")
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/metrics")
def metrics(k: int = K_DEFAULT, mode: str = Query("hybrid", regex="^(lexical|dense|hybrid)$")):
    """
    Calcula MAP/Soft-MAP/nDCG@K para el modo seleccionado:
      - lexical: usa TF-IDF clásico
      - dense: usa embeddings
      - hybrid: fusión RRF
    """
    try:
        if mode == "lexical":
            idx = index_lex
        elif mode == "dense":
            idx = index_dense
        else: # "hybrid"
            idx = index_hybrid

        res = evaluate_queries(query_df, label_df, idx, k=k,
                               gains_soft=gains.soft, gains_ndcg=gains.ndcg)
        return {
            "map": res[f"MAP@{k}"],
            "soft_map": res[f"Soft-MAP@{k}"],
            "ndcg": res[f"nDCG@{k}"]
        }
    except Exception as e:
        logger.exception("Fallo en el end /metrics")
        raise HTTPException(status_code=500, detail=str(e))


