import pandas as pd
from wands_search.index import MultiFieldIndex

def test_build_index_and_search():
    df = pd.DataFrame({
        "product_id":[1,2],
        "product_name":["wood chair","office armchair"],
        "product_description":["solid wood","comfortable office"]
    })
    idx = MultiFieldIndex(df).fit()
    ids, scores = idx.search("armchair", k=1)
    assert len(ids)==1
