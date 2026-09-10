"""
src/retrieval.py
High-performance TF-IDF vector space retriever with inverted index and cosine similarity.
Optimized for ultra-fast startup (<150ms) and sub-5ms query inference over historical AskPlayStation resolutions.
"""

import os
import sys
import re
import math
from collections import Counter, defaultdict
from typing import List, Dict, Any, Optional

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.ingest import load_sample_threads

def tokenize(text: str) -> List[str]:
    """Extract alphanumeric words in lowercase."""
    if not isinstance(text, str):
        return []
    return re.findall(r"\b[a-z0-9]+\b", text.lower())

class ResolutionRetriever:
    """
    Inverted-Index TF-IDF Retriever with exact Cosine Similarity.
    Builds in ~100ms over 3,000 threads and evaluates queries in <2ms.
    """

    def __init__(self, threads: Optional[List[Dict[str, Any]]] = None):
        if threads is None:
            threads = load_sample_threads()
        self.threads = threads
        self.num_docs = len(self.threads)

        # 1. Tokenize corpus
        doc_tokens = [tokenize(t.get("customer_text", "") or "") for t in self.threads]

        # 2. Document frequency (DF)
        df = Counter()
        for tokens in doc_tokens:
            for word in set(tokens):
                df[word] += 1

        # 3. Smooth Inverse Document Frequency (IDF)
        self.idf = {
            word: math.log((self.num_docs + 1.0) / (count + 1.0)) + 1.0
            for word, count in df.items()
        }

        # 4. Inverted Index with L2-normalized TF-IDF weights: word -> [(doc_id, norm_weight)]
        self.inverted_index = defaultdict(list)
        for doc_id, tokens in enumerate(doc_tokens):
            if not tokens:
                continue
            tf = Counter(tokens)
            weights = {w: tf[w] * self.idf[w] for w in tf}
            l2_norm = math.sqrt(sum(v * v for v in weights.values())) or 1.0
            for w, weight in weights.items():
                self.inverted_index[w].append((doc_id, weight / l2_norm))

    def retrieve(
        self,
        query: str,
        top_k: int = 3,
        min_similarity: float = 0.05
    ) -> List[Dict[str, Any]]:
        """
        Compute TF-IDF cosine similarity between the incoming query and historical threads.
        Returns top_k most relevant resolution pairs.
        """
        q_tokens = tokenize(query)
        if not q_tokens:
            return []

        q_tf = Counter(q_tokens)
        q_weights = {w: q_tf[w] * self.idf[w] for w in q_tf if w in self.idf}
        if not q_weights:
            return []

        q_norm = math.sqrt(sum(v * v for v in q_weights.values())) or 1.0

        # Accumulate dot products across inverted postings
        scores = defaultdict(float)
        for w, weight in q_weights.items():
            query_component = weight / q_norm
            for doc_id, doc_weight in self.inverted_index[w]:
                scores[doc_id] += query_component * doc_weight

        # Rank and select top_k
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]

        results = []
        for doc_id, score in ranked:
            if score < min_similarity:
                continue
            item = self.threads[doc_id]
            results.append({
                "similarity_score": round(score, 4),
                "customer_query": item.get("customer_text", ""),
                "historical_reply": item.get("brand_reply", ""),
                "full_thread_context": item.get("full_thread_context", ""),
                "thread_depth": item.get("thread_depth", 2)
            })

        return results

# Singleton instance
_RETRIEVER_INSTANCE: Optional[ResolutionRetriever] = None

def get_retriever() -> ResolutionRetriever:
    global _RETRIEVER_INSTANCE
    if _RETRIEVER_INSTANCE is None:
        _RETRIEVER_INSTANCE = ResolutionRetriever()
    return _RETRIEVER_INSTANCE

def retrieve_similar_resolutions(query: str, top_k: int = 3) -> List[Dict[str, Any]]:
    """Convenience functional interface."""
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
