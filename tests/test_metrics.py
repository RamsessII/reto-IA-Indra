from wands_search.metrics import map_at_k, soft_ap_at_k, ndcg_at_k

def test_map_simple():
    assert map_at_k([1,2], [1,3,2], 3) > 0

def test_soft_and_ndcg_zero_on_empty():
    assert soft_ap_at_k({}, [], 10) == 0.0
    assert ndcg_at_k({}, [], 10) == 0.0
