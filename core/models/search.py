"""
Search request and response models.

These models define the interface for search operations,
including scoping, retrieval methods, and result formatting.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, List, Dict, Any


class RetrievalMethod(str, Enum):
    """
    Available retrieval methods for search.

    Attributes:
        SEMANTIC: Pure vector-based semantic search
        BM25: Pure keyword-based BM25 search
        HYBRID: Combined semantic + BM25 with RRF fusion
        HYBRID_RERANK: Hybrid search with cross-encoder reranking
    """
    SEMANTIC = "semantic"
    BM25 = "bm25"
    HYBRID = "hybrid"
    HYBRID_RERANK = "hybrid_rerank"


@dataclass
class SearchScores:
    """
    Detailed score breakdown for a search result.

    Provides transparency into how each result was ranked.

    Attributes:
        semantic_score: Cosine similarity from vector search (0-1)
        bm25_score: BM25 relevance score (varies)
        combined_score: RRF combined score (if hybrid)
        rerank_score: Cross-encoder score (if reranking enabled)
        final_score: The score used for final ranking

    Example:
        >>> scores = SearchScores(
        ...     semantic_score=0.85,
        ...     bm25_score=12.5,
        ...     combined_score=0.72,
        ...     final_score=0.72
        ... )
    """
    semantic_score: float = 0.0
    bm25_score: float = 0.0
    combined_score: float = 0.0
    rerank_score: Optional[float] = None
    final_score: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        data = {
            "semantic_score": round(self.semantic_score, 4),
            "bm25_score": round(self.bm25_score, 4),
            "combined_score": round(self.combined_score, 4),
            "final_score": round(self.final_score, 4),
        }
        if self.rerank_score is not None:
            data["rerank_score"] = round(self.rerank_score, 4)
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SearchScores":
        """Create from dictionary."""
        return cls(
            semantic_score=data.get("semantic_score", 0.0),
            bm25_score=data.get("bm25_score", 0.0),
            combined_score=data.get("combined_score", 0.0),
            rerank_score=data.get("rerank_score"),
            final_score=data.get("final_score", 0.0),
        )


@dataclass
class SearchRequest:
    """
    Request parameters for a search operation.

    Supports scoping to specific collections and/or documents.

    Attributes:
        query: The search query text
        collection_id: Scope to specific collection (None = all)
        document_ids: Scope to specific documents (None = all in scope)
        k: Number of results to return
        method: Retrieval method to use
        alpha: Balance between semantic (1.0) and BM25 (0.0)
        use_reranker: Whether to apply reranking

    Example:
        >>> request = SearchRequest(
        ...     query="machine learning algorithms",
        ...     collection_id="abc123",
        ...     k=5,
        ...     method=RetrievalMethod.HYBRID
        ... )
    """
    query: str
    collection_id: Optional[str] = None
    document_ids: Optional[List[str]] = None
    k: int = 5
    method: RetrievalMethod = RetrievalMethod.HYBRID
    alpha: float = 0.5
    use_reranker: bool = True

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "query": self.query,
            "collection_id": self.collection_id,
            "document_ids": self.document_ids,
            "k": self.k,
            "method": self.method.value,
            "alpha": self.alpha,
            "use_reranker": self.use_reranker,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SearchRequest":
        """Create from dictionary."""
        method = data.get("method", "hybrid")
        if isinstance(method, str):
            method = RetrievalMethod(method)

        return cls(
            query=data["query"],
            collection_id=data.get("collection_id"),
            document_ids=data.get("document_ids"),
            k=data.get("k", 5),
            method=method,
            alpha=data.get("alpha", 0.5),
            use_reranker=data.get("use_reranker", True),
        )

    def get_filter(self) -> Optional[Dict[str, Any]]:
        """
        Build ChromaDB filter from scoping parameters.

        Returns:
            Filter dict for ChromaDB or None if no scoping
        """
        conditions = []

        # ChromaDB requires explicit operators ($eq, $in) for all filter conditions
        # Direct equality like {"field": "value"} may silently fail
        if self.collection_id:
            conditions.append({"collection_id": {"$eq": self.collection_id}})

        if self.document_ids:
            if len(self.document_ids) == 1:
                conditions.append({"document_id": {"$eq": self.document_ids[0]}})
            else:
                conditions.append({"document_id": {"$in": self.document_ids}})

        if not conditions:
            return None
        elif len(conditions) == 1:
            return conditions[0]
        else:
            return {"$and": conditions}


@dataclass
class SearchResult:
    """
    A single search result with context and scores.

    Attributes:
        content: The retrieved text chunk
        scores: Detailed score breakdown
        source: Source document filename
        metadata: Additional chunk metadata (includes collection_id, document_id, etc.)
        page: Source page number (if available)
        chunk_index: Position of chunk in document

    Example:
        >>> result = SearchResult(
        ...     content="Machine learning is...",
        ...     source="paper.pdf",
        ...     scores=SearchScores(final_score=0.85)
        ... )
    """
    content: str
    scores: SearchScores = field(default_factory=SearchScores)
    source: str = "unknown"
    metadata: Dict[str, Any] = field(default_factory=dict)
    page: Optional[int] = None
    chunk_index: Optional[int] = None

    @property
    def document_id(self) -> Optional[str]:
        """Get document ID from metadata if available."""
        return self.metadata.get("document_id")

    @property
    def collection_id(self) -> Optional[str]:
        """Get collection ID from metadata if available."""
        return self.metadata.get("collection_id")

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "content": self.content,
            "source": self.source,
            "page": self.page,
            "chunk_index": self.chunk_index,
            "scores": self.scores.to_dict(),
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SearchResult":
        """Create from dictionary."""
        scores_data = data.get("scores", {})
        scores = SearchScores.from_dict(scores_data) if scores_data else SearchScores()

        return cls(
            content=data["content"],
            source=data.get("source", "unknown"),
            page=data.get("page"),
            chunk_index=data.get("chunk_index"),
            scores=scores,
            metadata=data.get("metadata", {}),
        )


@dataclass
class SearchResponse:
    """
    Response from a search operation.

    Attributes:
        results: List of search results, ordered by relevance
        query: Original query text
        method: Retrieval method used
        total_results: Number of results returned
        search_time_ms: Search latency in milliseconds

    Example:
        >>> response = SearchResponse(
        ...     results=[...],
        ...     query="machine learning",
        ...     method=RetrievalMethod.HYBRID,
        ...     total_results=5,
        ...     search_time_ms=150
        ... )
    """
    results: List[SearchResult]
    query: str
    method: RetrievalMethod
    total_results: int = 0
    search_time_ms: float = 0.0

    def __post_init__(self):
        if self.total_results == 0:
            self.total_results = len(self.results)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "results": [r.to_dict() for r in self.results],
            "query": self.query,
            "method": self.method.value,
            "total_results": self.total_results,
            "search_time_ms": round(self.search_time_ms, 2),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SearchResponse":
        """Create from dictionary."""
        method = data.get("method", "hybrid")
        if isinstance(method, str):
            method = RetrievalMethod(method)

        results = [SearchResult.from_dict(r) for r in data.get("results", [])]

        return cls(
            results=results,
            query=data["query"],
            method=method,
            total_results=data.get("total_results", len(results)),
            search_time_ms=data.get("search_time_ms", 0.0),
        )
