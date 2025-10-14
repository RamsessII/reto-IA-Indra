import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
from .preprocess import safe, clean_category, guess_col
from .config import VectorizerParams, FieldWeights

class MultiFieldIndex:
    """
        Index TF-IDF que mezcla name/desc y si hay brand/cat tb.
        La idea es darle +peso al name y algo menos al desc, porq el name casi siempre
        tiene lo clave. Brand y cat ayudan si existen, sino no molesta.

        Cosas que hace:
        - usa ngramas en name (1,2) para encontra frases ("arm chair")
        - desc solo unigramas porq mete ruido
        - normalizo scores de cada campo (mm scaler) asi ninguno pisa al otro
        - pesos default: name=0.65, desc=0.35, brand=0.1, cat=0.15 (pero si no hay se ignora)

        Limit: no lematiza ni nada fancy, o sea "chairs" y "chair" son distintos.

        Uso rapido:
        idx = MultiFieldIndex(df).fit()
        ids, scores = idx.search("armchair")
    """
    def __init__(self, product_df: pd.DataFrame,
                 vec_name: VectorizerParams = VectorizerParams(ngram_range=(1,2)),
                 vec_desc: VectorizerParams = VectorizerParams(ngram_range=(1,1)),
                 vec_brand: VectorizerParams = VectorizerParams(ngram_range=(1,1)),
                 vec_cat: VectorizerParams = VectorizerParams(ngram_range=(1,2)),
                 weights: FieldWeights = FieldWeights()):
        self.df = product_df.reset_index(drop=True)
        self.vec_name_params = vec_name
        self.vec_desc_params = vec_desc
        self.vec_brand_params = vec_brand
        self.vec_cat_params = vec_cat
        self.weights = weights
        
        self.brand_col = guess_col(self.df, ["brand","brand_name","manufacturer","maker","vendor"])
        self.cat_col   = guess_col(self.df, ["category","categories","category_name","category_path","taxonomy","class","class_name","product_type"])
        
        self.vec_name = self.vec_desc = self.vec_brand = self.vec_cat = None
        self.X_name = self.X_desc = self.X_brand = self.X_cat = None

    @staticmethod
    def _mk_vec(params: VectorizerParams) -> TfidfVectorizer:
        return TfidfVectorizer(
            lowercase=params.lowercase,
            strip_accents=params.strip_accents,
            stop_words=params.stop_words,
            ngram_range=params.ngram_range,
            min_df=params.min_df,
            max_df=params.max_df,
            sublinear_tf=params.sublinear_tf
        )

    def fit(self):
        name = safe(self.df.get("product_name", ""))
        desc = safe(self.df.get("product_description", ""))

        self.vec_name = self._mk_vec(self.vec_name_params)
        self.vec_desc = self._mk_vec(self.vec_desc_params)
        self.X_name = self.vec_name.fit_transform(name.astype("U"))
        self.X_desc = self.vec_desc.fit_transform(desc.astype("U"))

        if self.brand_col is not None and safe(self.df[self.brand_col]).str.strip().str.len().gt(0).any():
            self.vec_brand = self._mk_vec(self.vec_brand_params)
            self.X_brand = self.vec_brand.fit_transform(safe(self.df[self.brand_col]).astype("U"))
        if self.cat_col is not None and safe(self.df[self.cat_col]).str.strip().str.len().gt(0).any():
            self.vec_cat = self._mk_vec(self.vec_cat_params)
            self.X_cat = self.vec_cat.fit_transform(clean_category(self.df[self.cat_col]).astype("U"))
        return self

    @staticmethod
    def _mm(x: np.ndarray) -> np.ndarray:
        x = x.reshape(-1,1)
        return MinMaxScaler().fit_transform(x).ravel()

    def search(self, query: str, k: int = 10):
        s_name = cosine_similarity(self.vec_name.transform([query]), self.X_name).ravel()
        s_desc = cosine_similarity(self.vec_desc.transform([query]), self.X_desc).ravel()

        fused = self.weights.name*self._mm(s_name) + self.weights.desc*self._mm(s_desc)

        if self.vec_brand is not None:
            s_brand = cosine_similarity(self.vec_brand.transform([query]), self.X_brand).ravel()
            fused += self.weights.brand*self._mm(s_brand)
        if self.vec_cat is not None:
            s_cat = cosine_similarity(self.vec_cat.transform([query]), self.X_cat).ravel()
            fused += self.weights.cat*self._mm(s_cat)

        idx = np.argsort(fused)[-k:][::-1]
        ids = self.df.iloc[idx]["product_id"].tolist()
        scores = fused[idx].tolist()
        return ids, scores
