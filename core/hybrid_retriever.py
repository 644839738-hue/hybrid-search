"""
Hybrid Retriever Module

Combines BM25 keyword search with semantic vector search for improved retrieval.
Uses Reciprocal Rank Fusion (RRF) to merge results from both retrieval methods.
"""

import logging
from typing import List, Optional, Dict, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStoreRetriever

from .bm25_retriever import BM25Retriever, BM25Result
from .reranker import BaseReranker, RerankResult, RerankerFactory

logger = logging.getLogger(__name__)


class RetrievalMethod(Enum):
    """Available retrieval methods."""
    SEMANTIC = "semantic"
    BM25 = "bm25"
    HYBRID = "hybrid"


@dataclass
class HybridResult:
    """
    Result from hybrid retrieval with combined scoring.

    Attributes:
        document: The retrieved document
        final_score: Combined relevance score after fusion/reranking
        semantic_score: Score from semantic search (if applicable)
        bm25_score: Score from BM25 search (if applicable)
        rerank_score: Score from re-ranker (if applicable)
        retrieval_method: Method(s) used to retrieve this document
        metadata: Additional retrieval metadata
    """
    document: Document
    final_score: float
    semantic_score: Optional[float] = None
    bm25_score: Optional[float] = None
    rerank_score: Optional[float] = None
    retrieval_method: str = "hybrid"
    metadata: Dict[str, Any] = field(default_factory=dict)


class HybridRetriever:
    """
    Hybrid retriever combining BM25 and semantic search.

    This retriever implements a two-stage retrieval process:
    1. First stage: Retrieve candidates using BM25 and/or semantic search
    2. Second stage (optional): Re-rank candidates using a cross-encoder

    The results are combined using Reciprocal Rank Fusion (RRF) which
    effectively merges ranked lists from different retrieval methods.

    Attributes:
        semantic_retriever: Vector store retriever for semantic search
        bm25_retriever: BM25 retriever for keyword search
        reranker: Optional re-ranker for second-stage ranking
        alpha: Weight for semantic search in fusion (0-1)
        rrf_k: RRF constant (typically 60)

    Example:
        >>> hybrid = HybridRetriever(
        ...     semantic_retriever=vector_store.get_retriever(),
        ...     documents=chunks,
        ...     alpha=0.5
        ... )
        >>> results = hybrid.retrieve("machine learning", k=5)
    """

    def __init__(
        self,
        semantic_retriever: VectorStoreRetriever,
        documents: List[Document] = None,
        reranker: BaseReranker = None,
        alpha: float = 0.5,
        rrf_k: int = 60,
        bm25_k1: float = 1.5,
        bm25_b: float = 0.75
    ):
        """
        Initialize hybrid retriever.

        Args:
            semantic_retriever: LangChain vector store retriever
            documents: Documents to index for BM25 (optional, can add later)
            reranker: Optional re-ranker for second-stage ranking
            alpha: Weight for semantic search in fusion (0 = BM25 only, 1 = semantic only)
            rrf_k: RRF constant (higher = more weight to lower ranks)
            bm25_k1: BM25 term frequency saturation parameter
            bm25_b: BM25 length normalization parameter
        """
        self.semantic_retriever = semantic_retriever
        self.alpha = alpha
        self.rrf_k = rrf_k
        self.reranker = reranker

        # Initialize BM25 retriever
        self.bm25_retriever = BM25Retriever(k1=bm25_k1, b=bm25_b)

        # Index documents if provided
        if documents:
            self.index_documents(documents)

        logger.info(
            f"HybridRetriever initialized: alpha={alpha}, rrf_k={rrf_k}, "
            f"reranker={'enabled' if reranker else 'disabled'}"
        )

    def index_documents(self, documents: List[Document]) -> int:
        """
        Index documents for BM25 retrieval.

        Args:
            documents: List of documents to index

        Returns:
            Number of documents indexed
        """
        return self.bm25_retriever.index_documents(documents)

    def set_reranker(self, reranker: BaseReranker) -> None:
        """
        Set or update the re-ranker.

        Args:
            reranker: Re-ranker instance to use
        """
        self.reranker = reranker
        logger.info(f"Reranker updated: {type(reranker).__name__}")

    def _reciprocal_rank_fusion(
        self,
        semantic_results: List[Tuple[Document, float]],
        bm25_results: List[BM25Result],
        alpha: float = None
    ) -> List[HybridResult]:
        """
        Combine results using Reciprocal Rank Fusion (RRF).

        RRF Formula: score = alpha * (1/(rrf_k + semantic_rank)) +
                            (1-alpha) * (1/(rrf_k + bm25_rank))

        Args:
            semantic_results: Results from semantic search with scores
            bm25_results: Results from BM25 search
            alpha: Override weight for semantic search

        Returns:
            Combined results sorted by RRF score
        """
        alpha = alpha if alpha is not None else self.alpha
        doc_scores: Dict[str, HybridResult] = {}

        # Process semantic results
        for rank, (doc, score) in enumerate(semantic_results, start=1):
            doc_id = self._get_doc_id(doc)
            rrf_score = alpha * (1.0 / (self.rrf_k + rank))

            if doc_id in doc_scores:
                doc_scores[doc_id].final_score += rrf_score
                doc_scores[doc_id].semantic_score = score
            else:
                doc_scores[doc_id] = HybridResult(
                    document=doc,
                    final_score=rrf_score,
                    semantic_score=score,
                    retrieval_method="semantic"
                )

        # Process BM25 results
        for result in bm25_results:
            doc_id = self._get_doc_id(result.document)
            rrf_score = (1 - alpha) * (1.0 / (self.rrf_k + result.rank))

            if doc_id in doc_scores:
                doc_scores[doc_id].final_score += rrf_score
                doc_scores[doc_id].bm25_score = result.score
                doc_scores[doc_id].retrieval_method = "hybrid"
            else:
                doc_scores[doc_id] = HybridResult(
                    document=result.document,
                    final_score=rrf_score,
                    bm25_score=result.score,
                    retrieval_method="bm25"
                )

        # Sort by final score
        results = sorted(
            doc_scores.values(),
            key=lambda x: x.final_score,
            reverse=True
        )

        return results

    def _get_doc_id(self, doc: Document) -> str:
        """
        Generate a unique identifier for a document.

        Uses content hash for deduplication.

        Args:
            doc: Document to identify

        Returns:
            Unique document identifier
        """
        return str(hash(doc.page_content))

    def retrieve(
        self,
        query: str,
        k: int = 5,
        method: RetrievalMethod = RetrievalMethod.HYBRID,
        alpha: float = None,
        use_reranker: bool = True,
        fetch_k: int = None
    ) -> List[HybridResult]:
        """
        Retrieve relevant documents using specified method.

        Args:
            query: Search query text
            k: Number of final results to return
            method: Retrieval method (SEMANTIC, BM25, or HYBRID)
            alpha: Override weight for semantic search in hybrid mode
            use_reranker: Whether to apply re-ranking if available
            fetch_k: Number of candidates to fetch before reranking (default: 3*k)

        Returns:
            List of HybridResult sorted by relevance

        Example:
            >>> results = retriever.retrieve(
            ...     "machine learning",
            ...     k=5,
            ...     method=RetrievalMethod.HYBRID
            ... )
        """
        fetch_k = fetch_k or k * 3

        logger.info(
            f"Retrieving with method={method.value}, k={k}, "
            f"alpha={alpha or self.alpha}, use_reranker={use_reranker}"
        )

        # Get results based on method
        if method == RetrievalMethod.SEMANTIC:
            results = self._semantic_retrieve(query, fetch_k)
        elif method == RetrievalMethod.BM25:
            results = self._bm25_retrieve(query, fetch_k)
        else:  # HYBRID
            results = self._hybrid_retrieve(query, fetch_k, alpha)

        # Apply re-ranking if available and requested
        if use_reranker and self.reranker and self.reranker.is_available():
            results = self._apply_reranking(query, results)

        # Limit to k results
        results = results[:k]

        logger.info(f"Retrieved {len(results)} documents")
        return results

    def _semantic_retrieve(
        self,
        query: str,
        k: int
    ) -> List[HybridResult]:
        """Retrieve using semantic search only."""
        docs = self.semantic_retriever.invoke(query)[:k]

        results = []
        for i, doc in enumerate(docs):
            results.append(HybridResult(
                document=doc,
                final_score=1.0 / (i + 1),  # Rank-based score
                semantic_score=1.0 / (i + 1),
                retrieval_method="semantic"
            ))

        return results

    def _bm25_retrieve(
        self,
        query: str,
        k: int
    ) -> List[HybridResult]:
        """Retrieve using BM25 only."""
        if not self.bm25_retriever.is_indexed():
            logger.warning("BM25 index not built. Falling back to semantic search.")
            return self._semantic_retrieve(query, k)

        bm25_results = self.bm25_retriever.retrieve(query, k)

        results = []
        for r in bm25_results:
            results.append(HybridResult(
                document=r.document,
                final_score=r.score,
                bm25_score=r.score,
                retrieval_method="bm25"
            ))

        return results

    def _hybrid_retrieve(
        self,
        query: str,
        k: int,
        alpha: float = None
    ) -> List[HybridResult]:
        """Retrieve using hybrid BM25 + semantic search."""
        # Get semantic results
        semantic_docs = self.semantic_retriever.invoke(query)[:k]
        semantic_results = [
            (doc, 1.0 / (i + 1))  # Rank-based score
            for i, doc in enumerate(semantic_docs)
        ]

        # Get BM25 results if indexed
        if self.bm25_retriever.is_indexed():
            bm25_results = self.bm25_retriever.retrieve(query, k)
        else:
            logger.warning("BM25 index not built. Using semantic-only results.")
            bm25_results = []

        # Fuse results using RRF
        return self._reciprocal_rank_fusion(
            semantic_results,
            bm25_results,
            alpha
        )

    def _apply_reranking(
        self,
        query: str,
        results: List[HybridResult]
    ) -> List[HybridResult]:
        """Apply re-ranking to results."""
        if not results:
            return results

        try:
            documents = [r.document for r in results]
            rerank_results = self.reranker.rerank(query, documents)

            # Update results with rerank scores
            updated_results = []
            for rr in rerank_results:
                # Find original result
                original = next(
                    (r for r in results if r.document.page_content == rr.document.page_content),
                    None
                )
                if original:
                    updated_results.append(HybridResult(
                        document=rr.document,
                        final_score=rr.score,
                        semantic_score=original.semantic_score,
                        bm25_score=original.bm25_score,
                        rerank_score=rr.score,
                        retrieval_method=original.retrieval_method,
                        metadata={"original_rank": rr.original_rank, "new_rank": rr.new_rank}
                    ))

            logger.info(f"Re-ranking applied to {len(updated_results)} documents")
            return updated_results

        except Exception as e:
            logger.error(f"Re-ranking failed, using original results: {e}")
            return results

    def get_retrieval_stats(self) -> Dict[str, Any]:
        """
        Get retrieval configuration statistics.

        Returns:
            Dictionary with retriever configuration and status
        """
        return {
            "alpha": self.alpha,
            "rrf_k": self.rrf_k,
            "bm25_indexed": self.bm25_retriever.is_indexed(),
            "bm25_doc_count": self.bm25_retriever.get_document_count(),
            "reranker_available": self.reranker.is_available() if self.reranker else False,
            "reranker_type": type(self.reranker).__name__ if self.reranker else None
        }


def create_hybrid_retriever(
    semantic_retriever: VectorStoreRetriever,
    documents: List[Document] = None,
    enable_reranker: bool = True,
    reranker_provider: str = "auto",
    alpha: float = 0.5,
    **kwargs
) -> HybridRetriever:
    """
    Factory function to create a configured hybrid retriever.

    Args:
        semantic_retriever: LangChain vector store retriever
        documents: Documents to index for BM25
        enable_reranker: Whether to enable re-ranking
        reranker_provider: Re-ranker provider ("cohere", "jina", or "auto")
        alpha: Weight for semantic search in fusion
        **kwargs: Additional arguments for HybridRetriever

    Returns:
        Configured HybridRetriever instance
    """
    reranker = None

    if enable_reranker:
        if reranker_provider == "auto":
            reranker = RerankerFactory.get_available_reranker()
        else:
            try:
                reranker = RerankerFactory.create(reranker_provider)
                if not reranker.is_available():
                    logger.warning(f"{reranker_provider} reranker not available")
                    reranker = None
            except ValueError as e:
                logger.error(f"Failed to create reranker: {e}")

    return HybridRetriever(
        semantic_retriever=semantic_retriever,
        documents=documents,
        reranker=reranker,
        alpha=alpha,
        **kwargs
    )
