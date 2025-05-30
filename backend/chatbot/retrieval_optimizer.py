import numpy as np
from typing import List, Dict, Tuple, Optional
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.utils import embedding_functions
import torch
from rank_bm25 import BM25Okapi
from sklearn.feature_extraction.text import TfidfVectorizer
import re
from dataclasses import dataclass
import logging

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
        
        # Initialize ChromaDB with powerful embedding model
        self.embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="BAAI/bge-large-en-v1.5"
        )
        
        self.chroma_client = chromadb.PersistentClient(path="./chroma_db")
        self.collection = self.chroma_client.get_collection(
            name=collection_name,
            embedding_function=self.embedding_function
        )
        
        # Cross-encoder for re-ranking
        self.reranker = SentenceTransformer('cross-encoder/ms-marco-MiniLM-L-6-v2')
        
        # BM25 for keyword search
        self._initialize_bm25()
        
        # Query expansion model
        self.query_expander = self._load_query_expansion_model()
        
    def _initialize_bm25(self):
        """
        Initialize BM25 for hybrid search
        """
        # Get all documents
        all_docs = self.collection.get()
        
        # Tokenize for BM25
        tokenized_docs = [doc.lower().split() for doc in all_docs['documents']]
        self.bm25 = BM25Okapi(tokenized_docs)
        self.doc_ids = all_docs['ids']
        self.doc_contents = all_docs['documents']
        self.doc_metadatas = all_docs['metadatas']
        
    async def retrieve(
        self,
        query: str,
        context: Dict = {},
        top_k: int = 10,
        rerank: bool = True
    ) -> List[RetrievalResult]:
        """
        Advanced retrieval with multiple strategies
        """
        # 1. Query expansion
        expanded_queries = self._expand_query(query, context)
        
        # 2. Hybrid search (semantic + keyword)
        semantic_results = await self._semantic_search(expanded_queries, top_k * 2)
        keyword_results = self._keyword_search(query, top_k * 2)
        
        # 3. Merge results
        merged_results = self._merge_results(semantic_results, keyword_results)
        
        # 4. Re-rank if enabled
        if rerank:
            reranked_results = self._rerank_results(query, merged_results)
        else:
            reranked_results = merged_results
            
        # 5. Post-process and format
        final_results = self._format_results(reranked_results[:top_k])
        
        return final_results
        
    def _expand_query(self, query: str, context: Dict) -> List[str]:
        """
        Expand query with synonyms and related terms
        """
        expanded = [query]
        
        # Add context-based expansions
        if context.get('page_type'):
            expanded.append(f"{query} {context['page_type']}")
            
        # Common expansions based on patterns
        expansions = {
            r'\bhours?\b': ['hours', 'open', 'closed', 'schedule', 'times'],
            r'\bpric(e|ing)\b': ['price', 'cost', 'fee', 'charge', 'pricing'],
            r'\breturn': ['return', 'refund', 'exchange', 'policy'],
            r'\bship': ['ship', 'shipping', 'delivery', 'send'],
            r'\bcontact\b': ['contact', 'phone', 'email', 'address', 'reach']
        }
        
        query_lower = query.lower()
        for pattern, terms in expansions.items():
            if re.search(pattern, query_lower):
                for term in terms:
                    expanded.append(f"{query} {term}")
                    
        return list(set(expanded))[:5]  # Limit to 5 variations
        
    async def _semantic_search(
        self,
        queries: List[str],
        top_k: int
    ) -> List[Tuple[str, Dict, float]]:
        """
        Semantic search using embeddings
        """
        all_results = []
        
        for query in queries:
            results = self.collection.query(
                query_texts=[query],
                n_results=top_k,
                include=['documents', 'metadatas', 'distances']
            )
            
            for i in range(len(results['ids'][0])):
                all_results.append((
                    results['documents'][0][i],
                    results['metadatas'][0][i],
                    1.0 - results['distances'][0][i]  # Convert distance to similarity
                ))
                
        # Deduplicate and sort by score
        seen = set()
        unique_results = []
        for result in sorted(all_results, key=lambda x: x[2], reverse=True):
            if result[0] not in seen:
                seen.add(result[0])
                unique_results.append(result)
                
        return unique_results
        
    def _keyword_search(self, query: str, top_k: int) -> List[Tuple[str, Dict, float]]:
        """
        BM25 keyword search
        """
        query_tokens = query.lower().split()
        scores = self.bm25.get_scores(query_tokens)
        
        # Get top k indices
        top_indices = np.argsort(scores)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            if scores[idx] > 0:
                results.append((
                    self.doc_contents[idx],
                    self.doc_metadatas[idx],
                    float(scores[idx]) / 10  # Normalize BM25 scores
                ))
                
        return results
        
    def _merge_results(
        self,
        semantic_results: List[Tuple],
        keyword_results: List[Tuple]
    ) -> List[Tuple]:
        """
        Merge semantic and keyword results with weighted scoring
        """
        # Create a unified scoring system
        result_scores = {}
        
        # Weight semantic results higher (0.7)
        for content, metadata, score in semantic_results:
            key = content[:100]  # Use first 100 chars as key
            if key not in result_scores:
                result_scores[key] = {
                    'content': content,
                    'metadata': metadata,
                    'score': 0
                }
            result_scores[key]['score'] += score * 0.7
            
        # Add keyword results (0.3)
        for content, metadata, score in keyword_results:
            key = content[:100]
            if key not in result_scores:
                result_scores[key] = {
                    'content': content,
                    'metadata': metadata,
                    'score': 0
                }
            result_scores[key]['score'] += score * 0.3
            
        # Sort by combined score
        sorted_results = sorted(
            result_scores.values(),
            key=lambda x: x['score'],
            reverse=True
        )
        
        return [(r['content'], r['metadata'], r['score']) for r in sorted_results]
        
    def _rerank_results(
        self,
        query: str,
        results: List[Tuple]
    ) -> List[Tuple]:
        """
        Re-rank using cross-encoder
        """
        if not results:
            return results
            
        # Prepare pairs for cross-encoder
        pairs = [[query, result[0]] for result in results]
        
        # Get similarity scores
        with torch.no_grad():
            scores = self.reranker.predict(pairs)
            
        # Combine with original scores (0.5 weight each)
        reranked = []
        for i, (content, metadata, orig_score) in enumerate(results):
            combined_score = (orig_score + float(scores[i])) / 2
            reranked.append((content, metadata, combined_score))
            
        # Sort by new scores
        reranked.sort(key=lambda x: x[2], reverse=True)
        
        return reranked