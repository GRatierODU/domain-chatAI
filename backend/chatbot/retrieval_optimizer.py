import numpy as np
from typing import List, Dict, Tuple, Optional
from sentence_transformers import SentenceTransformer
import torch
from rank_bm25 import BM25Okapi
from sklearn.feature_extraction.text import TfidfVectorizer
import re
from dataclasses import dataclass
import logging

# Import the centralized ChromaDB manager
try:
    from ..core.chromadb_manager import chroma_manager
except:
    # Fallback if import path is different
    import sys
    import os

    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from core.chromadb_manager import chroma_manager

logger = logging.getLogger(__name__)


@dataclass
class RetrievalResult:
    content: str
    metadata: Dict
    score: float
    source_type: str
    relevance_explanation: str


class OptimizedRetriever:
    """
    Production-ready retrieval system with hybrid search and re-ranking
    """

    def __init__(self, collection_name: str):
        self.collection_name = collection_name

        # Use centralized ChromaDB manager to get collection
        try:
            self.collection = chroma_manager.get_collection(collection_name)
            logger.info(f"Successfully got collection {collection_name}")
        except Exception as e:
            logger.error(f"Failed to get collection {collection_name}: {e}")
            raise Exception(f"Collection {collection_name} not found in ChromaDB")

        # Get embedding function from manager
        self.embedding_function = chroma_manager.get_embedding_function()

        # Initialize other components with error handling
        self._initialize_components()

    def _initialize_components(self):
        """Initialize retrieval components with error handling"""
        # Cross-encoder for re-ranking
        try:
            self.reranker = SentenceTransformer("cross-encoder/ms-marco-MiniLM-L-6-v2")
        except Exception as e:
            logger.warning(f"Failed to load reranker model: {e}")
            self.reranker = None

        # BM25 for keyword search
        try:
            self._initialize_bm25()
        except Exception as e:
            logger.warning(f"Failed to initialize BM25: {e}")
            self.bm25 = None

        # Query expansion model
        self.query_expander = None  # Simplified for now

    def _initialize_bm25(self):
        """
        Initialize BM25 for hybrid search
        """
        try:
            # Get all documents from collection
            all_docs = self.collection.get()

            if not all_docs["documents"]:
                logger.warning("No documents found in collection for BM25")
                self.bm25 = None
                self.doc_ids = []
                self.doc_contents = []
                self.doc_metadatas = []
                return

            # Tokenize for BM25
            tokenized_docs = [doc.lower().split() for doc in all_docs["documents"]]
            self.bm25 = BM25Okapi(tokenized_docs)
            self.doc_ids = all_docs["ids"]
            self.doc_contents = all_docs["documents"]
            self.doc_metadatas = (
                all_docs["metadatas"]
                if "metadatas" in all_docs
                else [{} for _ in all_docs["ids"]]
            )

        except Exception as e:
            logger.error(f"Error initializing BM25: {e}")
            self.bm25 = None
            self.doc_ids = []
            self.doc_contents = []
            self.doc_metadatas = []

    async def retrieve(
        self, query: str, context: Dict = {}, top_k: int = 10, rerank: bool = True
    ) -> List[RetrievalResult]:
        """
        Advanced retrieval with multiple strategies
        """
        try:
            # 1. Query expansion
            expanded_queries = self._expand_query(query, context)

            # 2. Hybrid search (semantic + keyword)
            semantic_results = await self._semantic_search(expanded_queries, top_k * 2)
            keyword_results = (
                self._keyword_search(query, top_k * 2) if self.bm25 else []
            )

            # 3. Merge results
            merged_results = self._merge_results(semantic_results, keyword_results)

            # 4. Re-rank if enabled and reranker available
            if rerank and self.reranker and merged_results:
                reranked_results = self._rerank_results(query, merged_results)
            else:
                reranked_results = merged_results

            # 5. Post-process and format
            final_results = self._format_results(reranked_results[:top_k])

            return final_results

        except Exception as e:
            logger.error(f"Retrieval error: {e}")
            return []

    def _expand_query(self, query: str, context: Dict) -> List[str]:
        """
        Expand query with synonyms and related terms
        """
        expanded = [query]

        # Add context-based expansions
        if context.get("page_type"):
            expanded.append(f"{query} {context['page_type']}")

        # Common expansions based on patterns
        expansions = {
            r"\bhours?\b": ["hours", "open", "closed", "schedule", "times"],
            r"\bpric(e|ing)\b": ["price", "cost", "fee", "charge", "pricing"],
            r"\breturn": ["return", "refund", "exchange", "policy"],
            r"\bship": ["ship", "shipping", "delivery", "send"],
            r"\bcontact\b": ["contact", "phone", "email", "address", "reach"],
            r"\bservice": ["service", "services", "offer", "provide", "offerings"],
            r"\bmenu\b": ["menu", "food", "dishes", "cuisine", "items"],
        }

        query_lower = query.lower()
        for pattern, terms in expansions.items():
            if re.search(pattern, query_lower):
                for term in terms:
                    expanded.append(f"{query} {term}")

        return list(set(expanded))[:5]  # Limit to 5 variations

    async def _semantic_search(
        self, queries: List[str], top_k: int
    ) -> List[Tuple[str, Dict, float]]:
        """
        Semantic search using embeddings
        """
        all_results = []

        try:
            for query in queries:
                results = self.collection.query(
                    query_texts=[query],
                    n_results=min(top_k, self.collection.count()),
                    include=["documents", "metadatas", "distances"],
                )

                if results and results["ids"] and results["ids"][0]:
                    for i in range(len(results["ids"][0])):
                        content = (
                            results["documents"][0][i] if results["documents"] else ""
                        )
                        metadata = (
                            results["metadatas"][0][i] if results["metadatas"] else {}
                        )
                        distance = (
                            results["distances"][0][i] if results["distances"] else 1.0
                        )

                        all_results.append(
                            (
                                content,
                                metadata,
                                1.0 - distance,  # Convert distance to similarity
                            )
                        )

        except Exception as e:
            logger.error(f"Semantic search error: {e}")

        # Deduplicate and sort by score
        seen = set()
        unique_results = []
        for result in sorted(all_results, key=lambda x: x[2], reverse=True):
            if result[0] and result[0] not in seen:
                seen.add(result[0])
                unique_results.append(result)

        return unique_results

    def _keyword_search(self, query: str, top_k: int) -> List[Tuple[str, Dict, float]]:
        """
        BM25 keyword search
        """
        if not self.bm25 or not self.doc_contents:
            return []

        try:
            query_tokens = query.lower().split()
            scores = self.bm25.get_scores(query_tokens)

            # Get top k indices
            top_indices = np.argsort(scores)[::-1][:top_k]

            results = []
            for idx in top_indices:
                if idx < len(self.doc_contents) and scores[idx] > 0:
                    content = self.doc_contents[idx]
                    metadata = (
                        self.doc_metadatas[idx] if idx < len(self.doc_metadatas) else {}
                    )
                    results.append(
                        (
                            content,
                            metadata,
                            float(scores[idx]) / 10,  # Normalize BM25 scores
                        )
                    )

            return results

        except Exception as e:
            logger.error(f"Keyword search error: {e}")
            return []

    def _merge_results(
        self, semantic_results: List[Tuple], keyword_results: List[Tuple]
    ) -> List[Tuple]:
        """
        Merge semantic and keyword results with weighted scoring
        """
        # Create a unified scoring system
        result_scores = {}

        # Weight semantic results higher (0.7)
        for content, metadata, score in semantic_results:
            if content:
                key = content[:200]  # Use first 200 chars as key
                if key not in result_scores:
                    result_scores[key] = {
                        "content": content,
                        "metadata": metadata,
                        "score": 0,
                    }
                result_scores[key]["score"] += score * 0.7

        # Add keyword results (0.3)
        for content, metadata, score in keyword_results:
            if content:
                key = content[:200]
                if key not in result_scores:
                    result_scores[key] = {
                        "content": content,
                        "metadata": metadata,
                        "score": 0,
                    }
                result_scores[key]["score"] += score * 0.3

        # Sort by combined score
        sorted_results = sorted(
            result_scores.values(), key=lambda x: x["score"], reverse=True
        )

        return [(r["content"], r["metadata"], r["score"]) for r in sorted_results]

    def _rerank_results(self, query: str, results: List[Tuple]) -> List[Tuple]:
        """
        Re-rank using cross-encoder
        """
        if not results or not self.reranker:
            return results

        try:
            # Prepare pairs for cross-encoder
            pairs = [[query, result[0]] for result in results if result[0]]

            if not pairs:
                return results

            # Get similarity scores
            with torch.no_grad():
                scores = self.reranker.predict(pairs)

            # Combine with original scores (0.5 weight each)
            reranked = []
            score_idx = 0
            for i, (content, metadata, orig_score) in enumerate(results):
                if content:
                    combined_score = (orig_score + float(scores[score_idx])) / 2
                    reranked.append((content, metadata, combined_score))
                    score_idx += 1

            # Sort by new scores
            reranked.sort(key=lambda x: x[2], reverse=True)

            return reranked

        except Exception as e:
            logger.error(f"Reranking error: {e}")
            return results

    def _format_results(self, results: List[Tuple]) -> List[RetrievalResult]:
        """Format results into RetrievalResult objects"""
        formatted = []

        for content, metadata, score in results:
            if content:  # Only include non-empty content
                formatted.append(
                    RetrievalResult(
                        content=content,
                        metadata=metadata or {},
                        score=score,
                        source_type=(
                            metadata.get("chunk_type", "text") if metadata else "text"
                        ),
                        relevance_explanation=f"Score: {score:.2f}",
                    )
                )

        return formatted

    def _load_query_expansion_model(self):
        """Load query expansion model (placeholder)"""
        return None
