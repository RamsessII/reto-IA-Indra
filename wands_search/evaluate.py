import pandas as pd
from .metrics import map_at_k, graded_rel_for_query, soft_ap_at_k, ndcg_at_k

def evaluate_queries(query_df, label_df, index, k=10,
                     gains_soft=None, gains_ndcg=None):
    grouped = label_df.groupby("query_id")

    query_df = query_df.copy()
    query_df["pred_ids"] = query_df["query"].apply(lambda q: index.search(q, k=k)[0])
    
    def exact(qid):
        g = grouped.get_group(qid)
        return g.loc[g["label"]=="Exact","product_id"].tolist()
    query_df["relevant_ids"] = query_df["query_id"].apply(exact)
   
    query_df["map@k"] = query_df.apply(lambda x: map_at_k(x["relevant_ids"], x["pred_ids"], k), axis=1)

    def _graded(qid, gains):
        return graded_rel_for_query(grouped.get_group(qid), gains)
    query_df["soft_map@k"] = query_df.apply(lambda x: soft_ap_at_k(_graded(x["query_id"], gains_soft), x["pred_ids"], k), axis=1)
    query_df["ndcg@k"]      = query_df.apply(lambda x: ndcg_at_k(_graded(x["query_id"], gains_ndcg), x["pred_ids"], k), axis=1)

    return {
        "MAP@{}".format(k): query_df["map@k"].mean(),
        "Soft-MAP@{}".format(k): query_df["soft_map@k"].mean(),
        "nDCG@{}".format(k): query_df["ndcg@k"].mean(),
        "per_query": query_df
    }
