import math
from typing import List, Dict

def map_at_k(true_ids: List[int], predicted_ids: List[int], k:int=10) -> float:
    if not true_ids or not predicted_ids:
        return 0.0
    score = 0.0; hits = 0.0
    for i, pid in enumerate(predicted_ids[:k]):
        if pid in true_ids and pid not in predicted_ids[:i]:
            hits += 1.0
            score += hits/(i+1.0)
    return score / min(len(true_ids), k)

def graded_rel_for_query(label_rows, gains: Dict[str, float]) -> Dict[int, float]:
    rel = {}
    for _, r in label_rows.iterrows():
        pid = int(r["product_id"]); lab = str(r["label"]).strip().lower()
        rel[pid] = max(rel.get(pid, 0.0), gains.get(lab, 0.0))
    return rel

def soft_ap_at_k(graded_rel: Dict[int,float], predicted_ids: List[int], k:int=10)->float:
    """
Soft-AP@K con relev gradudada:
    Exact = 1.0, Partial = 0.5 (se puede cambiar).
    idea: si un 'partial' igual sirve, no deberia ser 0, asi mide mejor la utilidad.
    peero ojo: si el dataset marca 'partial' muy facil, la metrica se infla.
"""
    if not predicted_ids:
        return 0.0
    ideal = sorted(graded_rel.values(), reverse=True)[:k]
    denom = sum(ideal)
    if denom == 0: return 0.0
    cum = 0.0; score = 0.0
    for i, pid in enumerate(predicted_ids[:k], start=1):
        g = graded_rel.get(pid, 0.0)
        cum += g
        score += (cum/i)*g
    return score/denom

def dcg_at_k(graded_rel: Dict[int,float], predicted_ids: List[int], k:int=10)->float:
    dcg = 0.0
    for i, pid in enumerate(predicted_ids[:k], start=1):
        g = graded_rel.get(pid, 0.0)
        dcg += g / math.log2(i+1)
    return dcg

def ndcg_at_k(graded_rel: Dict[int,float], predicted_ids: List[int], k:int=10)->float:
    dcg = dcg_at_k(graded_rel, predicted_ids, k)
    ideal = sorted(graded_rel.values(), reverse=True)[:k]
    idcg = sum(g/math.log2(i+1) for i,g in enumerate(ideal, start=1))
    return 0.0 if idcg==0 else dcg/idcg
