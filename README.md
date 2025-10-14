# WANDS Search – TF-IDF, Dense y Híbrido (FastAPI)

## 1) Objetivo
Mejorar MAP@10 sobre el baseline TF-IDF del dataset WANDS y exponer un microservicio de búsqueda listo para demo.

## 2) Dataset
- WANDS (Wayfair) – queries, products, labels (Exact/Partial/Irrelevant).
- **No se incluye** en el repo. Se clona automáticamente desde GitHub.

## 3) Enfoques de recuperación
- **Lexical (TF-IDF multi-campo)**: name/desc (+ brand/cat si existen) con fusión de scores normalizados.
- **Dense**: Sentence-Transformers (`all-MiniLM-L6-v2`) + FAISS (o NearestNeighbors como fallback).
- **Hybrid (RRF)**: fusión Reciprocal Rank Fusion sobre top-k lexical y dense.

### Ventajas y desventajas (resumen)
- TF-IDF: +rápido, interpretable, barato; −sin semántica.
- Dense: +semántica y sinónimos; −latencias/costos y dependencia del modelo.
- Híbrido: +robusto y estable; −complejidad y tuning (k_lex/k_dense, K de RRF).

## 4) Métricas
- **MAP@10 clásico** (Exact=1).
- **Soft-MAP@10** (Exact=1, Partial=0.5) – relevancia graduada; evita penalizar parciales.
- **nDCG@10** (Exact=2, Partial=1).

**Trade-offs:** Soft-MAP y nDCG premian parciales; pueden inflar métricas si hay muchos “partial” amplios.

## 5) Resultados (k=10)
| Modo   | MAP@10 | Soft-MAP@10 | nDCG@10 |
|--------|--------|-------------|---------|
| Baseline TF-IDF | 0.2932 | – | – |
| Lexical (multi-campo) | 0.3199 | 0.4548 | 0.6622 |
| Dense | 0.3353 | 0.4822 | 0.6914 |
| **Hybrid (RRF)** | **0.3747** | **0.5169** | **0.7286** |

Lift baseline → hybrid: **+27.8%** en MAP@10.

## 6) Arquitectura OOP
- `wands_search/index.py` → `MultiFieldIndex`
- `wands_search/dense_index.py` → `DenseIndex`
- `wands_search/hybrid_index.py` → `HybridIndex` (RRF)
- `wands_search/metrics.py` → MAP, Soft-AP, nDCG
- `wands_search/preprocess.py` → normalización y detección de columnas
- `wands_search/evaluate.py` → evaluación por queries

## 7) API (FastAPI)
- `GET /health` – estado.
- `POST /search?mode=(lexical|dense|hybrid)` – búsqueda.
- `GET /metrics?k=10&mode=...` – MAP/Soft-MAP/nDCG.

## 8) Setup local
```bash
python -m venv .venv && source .venv/bin/activate  
pip install -r requirements.txt
git clone https://github.com/wayfair/WANDS.git
export DATA_DIR=WANDS/dataset
uvicorn api.main:app --reload --port 8000
