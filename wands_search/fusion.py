from collections import defaultdict
from typing import List, Tuple

def rrf_fuse(rankings: List[List[int]], k: int = 10, K: int = 60):
    """
        Reciprocal Rank Fusion (RRF).
        Basicamente suma 1/(K+pos) para cada id qe aparezca en los rankings.
        K=60 es como "constante de suavizado", evita que lo ultimo pese mucho.

        Bueno porq mezcla dense+lexical sin tener que calibrar scores q son de escalas distintas.
        Malo: pierde magnitud real del score (usa solo orden).
    """

    scores = defaultdict(float)
    for r in rankings:
        for rank, pid in enumerate(r, start=1):
            scores[pid] += 1.0 / (K + rank)
    items = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:k]
    ids = [pid for pid, _ in items]
    scs = [s for _, s in items]
    return ids, scs
