import pandas as pd
import re

def safe(s: pd.Series) -> pd.Series:
    return s.fillna("").astype(str)

def clean_category(s: pd.Series) -> pd.Series:
    return safe(s).str.replace(r"[>/|]", " ", regex=True)

def guess_col(df: pd.DataFrame, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None
