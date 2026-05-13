"""
Search Manager API.

Stripe-like API for unified search across collections and documents.
Provides scoped search, score breakdown, and multiple retrieval methods.
"""

import logging
from typing import Optional, List, Dict, Any
from datetime import datetime

from core.models.search import (
    SearchRequest,
    SearchResponse,
    SearchResult,
    SearchScores,
    RetrievalMethod,
)
from core.models.errors import ValidationError, NotFoundError
from core.models.document import Document as DocumentModel
from core.storage import JSONStorage, COLLECTIONS_FILE, DOCUMENTS_FILE

logger = logging.getLogger(__name__)


class SearchManager:
    """
    Stripe-like API for unified search operations.

    Provides a unified interface for all search scenarios:
    - Global search (all collections)
    - Collection-scoped search
    - Document-scoped search
    - Multiple retrieval methods (semantic, BM25, hybrid)

    Attributes:
        storage: JSON storage for metadata
        vector_store: VectorStoreManager for semantic search
        hybrid_retriever: HybridRetriever for combined search

    Example:
        >>> manager = SearchManager(vector_store=vs_manager)
        >>> response = manager.search(SearchRequest(
        ...     query="machine learning",
        ...     collection_id="abc123",
        ...     method=RetrievalMethod.HYBRID,
        ...     k=5
        ... ))
        >>> for result in response.results:
        ...     print(f"{result.content[:50]}... (score: {result.scores.final:.3f})")
    """

    def __init__(
        self,
        storage: Optional[JSONStorage] = None,
        vector_store=None,
        hybrid_retriever_factory=None,
        data_dir: str = "./data",
    ):
        """
        Initialize the search manager.

        Args:
            storage: Optional JSONStorage instance
            vector_store: VectorStoreManager for semantic search
            hybrid_retriever_factory: Optional factory function to create HybridRetriever
            data_dir: Directory for JSON storage
        """
        self.storage = storage or JSONStorage(data_dir=data_dir)
        self.vector_store = vector_store
        self.hybrid_retriever_factory = hybrid_retriever_factory
        self._hybrid_retriever = None
        logger.info("SearchManager initialized")

    def search(self, request: SearchRequest) -> SearchResponse:
        """
        Execute search with optional scoping.

        Supports multiple search scenarios:
        - collection_id=None, document_ids=None: Search all collections
        - collection_id set: Search within specific collection
        - document_ids set: Search within specific documents
        - Both set: Search specific docs in specific collection

        Args:
            request: SearchRequest with query and optional scoping

        Returns:
            SearchResponse with results, metadata, and timing

        Raises:
            ValidationError: If request parameters are invalid
            NotFoundError: If specified collection/documents don't exist

        Example:
            >>> # Search within a collection
            >>> response = manager.search(SearchRequest(
            ...     query="neural networks",
            ...     collection_id="abc123",
            ...     k=5
            ... ))

            >>> # Search specific documents
            >>> response = manager.search(SearchRequest(
            ...     query="transformers",
            ...     document_ids=["doc1", "doc2"],
            ...     method=RetrievalMethod.HYBRID
            ... ))
        """
        start_time = datetime.now()

        # Validate request
        self._validate_request(request)

        # Build ChromaDB filter from request
        filter_dict = request.get_filter()

        # Execute search based on method
        if request.method == RetrievalMethod.SEMANTIC:
            results = self._semantic_search(request, filter_dict)
        elif request.method == RetrievalMethod.BM25:
            results = self._bm25_search(request, filter_dict)
        else:  # HYBRID
            results = self._hybrid_search(request, filter_dict)

        # Calculate timing
        end_time = datetime.now()
        search_time_ms = (end_time - start_time).total_seconds() * 1000

        # Build response
        response = SearchResponse(
            query=request.query,
            results=results,
            total_results=len(results),
            method=request.method,
            search_time_ms=search_time_ms,
        )

        logger.info(
            f"Search completed: query='{request.query[:30]}...', "
            f"results={len(results)}, time={search_time_ms:.0f}ms"
        )

        return response

    def _validate_request(self, request: SearchRequest) -> None:
        """
        Validate search request parameters.

        Args:
            request: SearchRequest to validate

        Raises:
            ValidationError: If parameters are invalid
            NotFoundError: If collection/documents don't exist
        """
        # Validate query
        if not request.query or not request.query.strip():
            raise ValidationError(
                message="Search query cannot be empty",
                param="query"
            )

        # Validate k
        if request.k < 1:
            raise ValidationError(
                message="k must be at least 1",
                param="k"
            )

        if request.k > 50:
            raise ValidationError(
                message="k cannot exceed 50",
                param="k"
            )

        # Validate collection exists if specified
        if request.collection_id:
            collection = self.storage.get_by_id(
                COLLECTIONS_FILE, request.collection_id
            )
            if not collection:
                raise NotFoundError(
                    message=f"Collection '{request.collection_id}' not found",
                    param="collection_id",
                    resource_type="collection",
                    resource_id=request.collection_id
                )

        # Validate documents exist if specified
        if request.document_ids:
            for doc_id in request.document_ids:
                doc = self.storage.get_by_id(DOCUMENTS_FILE, doc_id)
                if not doc:
                    raise NotFoundError(
                        message=f"Document '{doc_id}' not found",
                        param="document_ids",
                        resource_type="document",
                        resource_id=doc_id
                    )

    def _semantic_search(
        self,
        request: SearchRequest,
        filter_dict: Optional[Dict[str, Any]]
    ) -> List[SearchResult]:
        """
        Execute semantic-only search.

        Args:
            request: Search request
            filter_dict: Optional ChromaDB filter

        Returns:
            List of SearchResult objects
        """
        if not self.vector_store:
            logger.warning("No vector store configured, returning empty results")
            return []

        # Use vector store's search_similar with filter
        docs = self.vector_store.search_similar(
            query=request.query,
            k=request.k,
            filter=filter_dict
        )

        # Convert to SearchResults
        results = []
        for i, doc in enumerate(docs):
            # Calculate semantic score based on rank
            semantic_score = 1.0 / (i + 1)

            scores = SearchScores(
                semantic_score=semantic_score,
                final_score=semantic_score
            )

            results.append(SearchResult(
                content=doc.page_content,
                metadata=doc.metadata,
                scores=scores,
                source=doc.metadata.get("source", "unknown"),
                page=doc.metadata.get("page"),
                chunk_index=doc.metadata.get("start_index"),
            ))

        return results

    def _bm25_search(
        self,
        request: SearchRequest,
        filter_dict: Optional[Dict[str, Any]]
    ) -> List[SearchResult]:
        """
        Execute BM25-only search.

        Note: BM25 filtering is done post-retrieval since BM25 doesn't
        support native filtering like ChromaDB.

        Args:
            request: Search request
            filter_dict: Optional filter to apply post-retrieval

        Returns:
            List of SearchResult objects
        """
        # Check if we have a hybrid retriever
        if not self._hybrid_retriever and not self.hybrid_retriever_factory:
            logger.warning("No hybrid retriever configured, falling back to semantic")
            return self._semantic_search(request, filter_dict)

        # Create hybrid retriever if needed
        if not self._hybrid_retriever and self.hybrid_retriever_factory:
            self._hybrid_retriever = self.hybrid_retriever_factory()

        if not self._hybrid_retriever:
            return self._semantic_search(request, filter_dict)

        from core.hybrid_retriever import RetrievalMethod as HRMethod

        # Execute BM25 search via hybrid retriever
        hybrid_results = self._hybrid_retriever.retrieve(
            query=request.query,
            k=request.k * 2,  # Over-fetch for filtering
            method=HRMethod.BM25,
            use_reranker=request.use_reranker
        )

        # Filter results if needed
        results = []
        for hr in hybrid_results:
            # Apply filter (post-retrieval for BM25)
            if filter_dict and not self._matches_filter(hr.document, filter_dict):
                continue

            scores = SearchScores(
                bm25_score=hr.bm25_score or 0.0,
                final_score=hr.final_score
            )

            results.append(SearchResult(
                content=hr.document.page_content,
                metadata=hr.document.metadata,
                scores=scores,
                source=hr.document.metadata.get("source", "unknown"),
                page=hr.document.metadata.get("page"),
                chunk_index=hr.document.metadata.get("start_index"),
            ))

            if len(results) >= request.k:
                break

        return results

    def _hybrid_search(
        self,
        request: SearchRequest,
        filter_dict: Optional[Dict[str, Any]]
    ) -> List[SearchResult]:
        """
        Execute hybrid (BM25 + semantic) search.

        Args:
            request: Search request
            filter_dict: Optional ChromaDB filter

        Returns:
            List of SearchResult objects
        """
        # For hybrid, we need the hybrid retriever
        if not self._hybrid_retriever and self.hybrid_retriever_factory:
            # Create retriever with filtered semantic retriever
            retriever = None
            if self.vector_store:
                retriever = self.vector_store.get_retriever(
                    search_k=request.k,
                    filter=filter_dict
                )
            self._hybrid_retriever = self.hybrid_retriever_factory(
                semantic_retriever=retriever
            )

        # If no hybrid retriever available, fall back to semantic
        if not self._hybrid_retriever:
            logger.info("No hybrid retriever, using semantic search only")
            return self._semantic_search(request, filter_dict)

        from core.hybrid_retriever import RetrievalMethod as HRMethod

        # Execute hybrid search
        hybrid_results = self._hybrid_retriever.retrieve(
            query=request.query,
            k=request.k,
            method=HRMethod.HYBRID,
            alpha=request.alpha,
            use_reranker=request.use_reranker
        )

        # Convert to SearchResults
        results = []
        for hr in hybrid_results:
            scores = SearchScores(
                semantic_score=hr.semantic_score or 0.0,
                bm25_score=hr.bm25_score or 0.0,
                rerank_score=hr.rerank_score,
                final_score=hr.final_score
            )

            results.append(SearchResult(
                content=hr.document.page_content,
                metadata=hr.document.metadata,
                scores=scores,
                source=hr.document.metadata.get("source", "unknown"),
                page=hr.document.metadata.get("page"),
                chunk_index=hr.document.metadata.get("start_index"),
            ))

        return results

    def _matches_filter(
        self,
        doc,
        filter_dict: Dict[str, Any]
    ) -> bool:
        """
        Check if a document matches a filter.

        Used for post-retrieval filtering (e.g., BM25 results).

        Args:
            doc: Document to check
            filter_dict: Filter criteria

        Returns:
            True if document matches filter
        """
        metadata = doc.metadata

        for key, value in filter_dict.items():
            if key == "$and":
                # All conditions must match
                return all(
                    self._matches_filter(doc, condition)
                    for condition in value
                )
            elif key == "$or":
                # At least one condition must match
                return any(
                    self._matches_filter(doc, condition)
                    for condition in value
                )
            elif isinstance(value, dict):
                # Complex operator
                if "$in" in value:
                    if metadata.get(key) not in value["$in"]:
                        return False
                elif "$eq" in value:
                    if metadata.get(key) != value["$eq"]:
                        return False
            else:
                # Simple equality
                if metadata.get(key) != value:
                    return False

        return True

    def get_search_stats(self) -> Dict[str, Any]:
        """
        Get search statistics and configuration.

        Returns:
            Dictionary with search configuration and stats
        """
        stats = {
            "vector_store_available": self.vector_store is not None,
            "hybrid_retriever_available": self._hybrid_retriever is not None,
        }

        if self.vector_store:
            stats["indexed_documents"] = self.vector_store.get_collection_count()

        if self._hybrid_retriever:
            stats["hybrid_config"] = self._hybrid_retriever.get_retrieval_stats()

        return stats

    def set_hybrid_retriever(self, retriever) -> None:
        """
        Set the hybrid retriever instance.

        Args:
            retriever: HybridRetriever instance
        """
        self._hybrid_retriever = retriever
        logger.info("Hybrid retriever set")
