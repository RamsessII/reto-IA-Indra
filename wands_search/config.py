from dataclasses import dataclass
from typing import Tuple, Optional, Dict

@dataclass
class VectorizerParams:
    lowercase: bool = True
    strip_accents: str = "unicode"
    stop_words: Optional[str] = None  # o "english"
    ngram_range: Tuple[int, int] = (1, 2)
    min_df: int = 1
    max_df: float = 0.95
    sublinear_tf: bool = True

@dataclass
class FieldWeights:
    name: float = 0.65
    desc: float = 0.35
    brand: float = 0.10
    cat: float = 0.15

@dataclass
class LabelGains:
    soft: Dict[str, float] = None
    ndcg: Dict[str, float] = None

    def __post_init__(self):
        if self.soft is None:
            self.soft = {"exact": 1.0, "partial": 0.5, "irrelevant": 0.0}
        if self.ndcg is None:
            self.ndcg = {"exact": 2.0, "partial": 1.0, "irrelevant": 0.0}
