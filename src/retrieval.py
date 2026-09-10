"""
src/retrieval.py
Retrieval engine that grounds drafted replies in historical AskPlayStation resolutions.
Uses fast TF-IDF vectorization with cosine similarity over historical (customer_text, brand_reply) pairs.
"""

import os
import sys
from typing import List, Dict, Any, Optional

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

try:
    from src.ingest import load_sample_threads
except ImportError:
    from ingest import load_sample_threads

class ResolutionRetriever:
    """
    Fast TF-IDF similarity retriever over historical AskPlayStation resolved pairs.
    Optimized for sub-second initialization and lightning-fast inference.
    """

    def __init__(self, threads: Optional[List[Dict[str, Any]]] = None):
        if threads is None:
            threads = load_sample_threads()
        self.threads = threads
        self.corpus = [str(t.get("customer_text", "") or "") for t in self.threads]
        
        # High-performance unigram TF-IDF vectorizer (0.05s fit time on 3,000 examples)
        self.vectorizer = TfidfVectorizer(
            ngram_range=(1, 1),
            stop_words="english",
            max_features=5000
        )
        self.tfidf_matrix = self.vectorizer.fit_transform(self.corpus)

    def retrieve(
        self,
        query: str,
        top_k: int = 3,
        min_similarity: float = 0.05
    ) -> List[Dict[str, Any]]:
        """
        Retrieve the top-k most semantically relevant historical customer messages
        and their corresponding AskPlayStation brand replies.
        """
        if not query or not self.threads:
            return []

        query_vec = self.vectorizer.transform([query])
        scores = cosine_similarity(query_vec, self.tfidf_matrix).flatten()

        top_indices = np.argsort(scores)[::-1][:top_k]

        results = []
        for idx in top_indices:
            score = float(scores[idx])
            if score < min_similarity:
                continue
            item = self.threads[idx]
            results.append({
                "similarity_score": round(score, 4),
                "customer_query": item.get("customer_text", ""),
                "historical_reply": item.get("brand_reply", ""),
                "full_thread_context": item.get("full_thread_context", ""),
                "thread_depth": item.get("thread_depth", 2)
            })

        return results

# Singleton instance for fast module access
_RETRIEVER_INSTANCE: Optional[ResolutionRetriever] = None

def get_retriever() -> ResolutionRetriever:
    global _RETRIEVER_INSTANCE
    if _RETRIEVER_INSTANCE is None:
        _RETRIEVER_INSTANCE = ResolutionRetriever()
    return _RETRIEVER_INSTANCE

def retrieve_similar_resolutions(query: str, top_k: int = 3) -> List[Dict[str, Any]]:
    """Convenience function to retrieve similar resolved threads."""
    retriever = get_retriever()
    return retriever.retrieve(query, top_k=top_k)

if __name__ == "__main__":
    retriever = get_retriever()
    sample_query = "my psn account got banned for no reason"
    matches = retriever.retrieve(sample_query, top_k=2)
    print(f"Query: '{sample_query}'")
    print(f"Found {len(matches)} historical matches:\n")
    for i, m in enumerate(matches, 1):
        print(f"Match #{i} (Score: {m['similarity_score']}):")
        print(f"Past Customer: {m['customer_query']}")
        print(f"Brand Reply:   {m['historical_reply']}\n")
