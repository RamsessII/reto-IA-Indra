from .config import VectorizerParams, FieldWeights, LabelGains
from .index import MultiFieldIndex
from .metrics import map_at_k, graded_rel_for_query, soft_ap_at_k, ndcg_at_k
from .evaluate import evaluate_queries

try:
    from .dense_index import DenseIndex
except ImportError:
    DenseIndex = None

try:
    from .hybrid_index import HybridRetriever
except ImportError:
    HybridRetriever = None

__all__ = [
    "VectorizerParams",
    "FieldWeights",
    "LabelGains",
    "MultiFieldIndex",
    "map_at_k",
    "graded_rel_for_query",
    "soft_ap_at_k",
    "ndcg_at_k",
    "evaluate_queries",
    "DenseIndex",
    "HybridRetriever",
]

__version__ = "0.1.0"
