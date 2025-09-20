from __future__ import annotations
import re
from typing import List, Tuple
from langchain.schema import Document
from langchain_chroma import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever

def build_hybrid_retriever(
    vectorstore: Chroma, corpus_chunks: List[Document], k: int = 6, alpha: float = 0.7
) -> EnsembleRetriever:
    """Hybrid = semantic (Chroma) + keyword (BM25). alpha is semantic weight."""
    bm25 = BM25Retriever.from_documents(corpus_chunks)
    bm25.k = k
    vret = vectorstore.as_retriever(search_kwargs={"k": k})
    return EnsembleRetriever(retrievers=[vret, bm25], weights=[alpha, 1 - alpha])

# --- simple rule-based router ---
_PATTERNS = {
    "definition": [r"\b(what is|define|meaning|explain|describe)\b"],
    "example":    [r"\b(example|examples|case study|show me|for example|such as)\b"],
    "howto":      [r"\b(how to|how can|steps to|ways to|methods to|improve|develop|build|increase|enhance)\b"],
}

def classify_query(q: str) -> str:
    ql = q.lower()
    scores = {lbl: sum(1 for pat in pats if re.search(pat, ql)) for lbl, pats in _PATTERNS.items()}
    return max(scores, key=scores.get) if max(scores.values()) > 0 else "general"

def routed_weights(label: str) -> Tuple[float, float]:
    """Return (semantic_weight, keyword_weight)."""
    if label == "definition": return 0.8, 0.2
    if label == "example":    return 0.6, 0.4
    if label == "howto":      return 0.7, 0.3
    return 0.7, 0.3
