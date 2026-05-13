"""
Tests for Hybrid Retriever module.
"""

import pytest
from unittest.mock import MagicMock
from core.hybrid_retriever import (
    HybridRetriever,
    HybridResult,
    RetrievalMethod,
    create_hybrid_retriever
)
from core.bm25_retriever import BM25Retriever


class TestHybridRetriever:
    """Test cases for HybridRetriever class."""

    def test_initialization(self, mock_semantic_retriever):
        """Test hybrid retriever initialization."""
        retriever = HybridRetriever(
            semantic_retriever=mock_semantic_retriever,
            alpha=0.5,
            rrf_k=60
        )

        assert retriever.alpha == 0.5
        assert retriever.rrf_k == 60
        assert retriever.semantic_retriever == mock_semantic_retriever
        assert retriever.reranker is None

    def test_initialization_with_documents(self, mock_semantic_retriever, sample_documents):
        """Test initialization with document indexing."""
        retriever = HybridRetriever(
            semantic_retriever=mock_semantic_retriever,
            documents=sample_documents,
            alpha=0.7
        )

        assert retriever.bm25_retriever.is_indexed()
        assert retriever.bm25_retriever.get_document_count() == len(sample_documents)

    def test_index_documents(self, mock_semantic_retriever, sample_documents):
        """Test document indexing after initialization."""
        retriever = HybridRetriever(semantic_retriever=mock_semantic_retriever)
        count = retriever.index_documents(sample_documents)

        assert count == len(sample_documents)
        assert retriever.bm25_retriever.is_indexed()

    def test_semantic_retrieve(self, mock_semantic_retriever, sample_documents):
        """Test semantic-only retrieval."""
        retriever = HybridRetriever(
            semantic_retriever=mock_semantic_retriever,
            documents=sample_documents
        )

        results = retriever.retrieve(
            "machine learning",
            k=3,
            method=RetrievalMethod.SEMANTIC
        )

        assert len(results) <= 3
        assert all(isinstance(r, HybridResult) for r in results)
        assert all(r.retrieval_method == "semantic" for r in results)

    def test_bm25_retrieve(self, mock_semantic_retriever, sample_documents):
        """Test BM25-only retrieval."""
        retriever = HybridRetriever(
            semantic_retriever=mock_semantic_retriever,
            documents=sample_documents
        )

        results = retriever.retrieve(
            "machine learning",
            k=3,
            method=RetrievalMethod.BM25
        )

        assert len(results) <= 3
        assert all(isinstance(r, HybridResult) for r in results)
        # BM25 results should have bm25_score set
        for r in results:
            assert r.bm25_score is not None

    def test_hybrid_retrieve(self, mock_semantic_retriever, sample_documents_large):
        """Test hybrid retrieval combining BM25 and semantic."""
        retriever = HybridRetriever(
            semantic_retriever=mock_semantic_retriever,
            documents=sample_documents_large,
            alpha=0.5
        )

        results = retriever.retrieve(
            "neural networks deep learning",
            k=5,
            method=RetrievalMethod.HYBRID
        )

        assert len(results) <= 5
        assert all(isinstance(r, HybridResult) for r in results)

    def test_hybrid_retrieve_alpha_zero(self, mock_semantic_retriever, sample_documents):
        """Test hybrid with alpha=0 (BM25 only)."""
        retriever = HybridRetriever(
            semantic_retriever=mock_semantic_retriever,
            documents=sample_documents,
            alpha=0.0
        )

        results = retriever.retrieve(
            "machine learning",
            k=3,
            method=RetrievalMethod.HYBRID,
            alpha=0.0
        )

        # With alpha=0, only BM25 contributes to score
        assert len(results) <= 3

    def test_hybrid_retrieve_alpha_one(self, mock_semantic_retriever, sample_documents):
        """Test hybrid with alpha=1 (semantic only)."""
        retriever = HybridRetriever(
            semantic_retriever=mock_semantic_retriever,
            documents=sample_documents,
            alpha=1.0
        )

        results = retriever.retrieve(
            "machine learning",
            k=3,
            method=RetrievalMethod.HYBRID,
            alpha=1.0
        )

        # With alpha=1, only semantic contributes to score
        assert len(results) <= 3

    def test_set_reranker(self, mock_semantic_retriever):
        """Test setting reranker."""
        retriever = HybridRetriever(semantic_retriever=mock_semantic_retriever)
        mock_reranker = MagicMock()

        retriever.set_reranker(mock_reranker)

        assert retriever.reranker == mock_reranker

    def test_get_retrieval_stats(self, mock_semantic_retriever, sample_documents):
        """Test getting retrieval statistics."""
        retriever = HybridRetriever(
            semantic_retriever=mock_semantic_retriever,
            documents=sample_documents,
            alpha=0.6,
            rrf_k=50
        )

        stats = retriever.get_retrieval_stats()

        assert stats["alpha"] == 0.6
        assert stats["rrf_k"] == 50
        assert stats["bm25_indexed"] is True
        assert stats["bm25_doc_count"] == len(sample_documents)
        assert stats["reranker_available"] is False

    def test_bm25_fallback_when_not_indexed(self, mock_semantic_retriever):
        """Test fallback to semantic when BM25 not indexed."""
        retriever = HybridRetriever(semantic_retriever=mock_semantic_retriever)
        # Don't index documents

        results = retriever.retrieve(
            "test query",
            k=3,
            method=RetrievalMethod.BM25
        )

        # Should fall back to semantic search
        assert len(results) <= 3


class TestHybridResult:
    """Test cases for HybridResult dataclass."""

    def test_hybrid_result_creation(self, sample_documents):
        """Test creating HybridResult."""
        result = HybridResult(
            document=sample_documents[0],
            final_score=0.85,
            semantic_score=0.9,
            bm25_score=0.8,
            retrieval_method="hybrid"
        )

        assert result.document == sample_documents[0]
        assert result.final_score == 0.85
        assert result.semantic_score == 0.9
        assert result.bm25_score == 0.8
        assert result.retrieval_method == "hybrid"
        assert result.rerank_score is None

    def test_hybrid_result_with_rerank_score(self, sample_documents):
        """Test HybridResult with rerank score."""
        result = HybridResult(
            document=sample_documents[0],
            final_score=0.95,
            rerank_score=0.95,
            retrieval_method="hybrid"
        )

        assert result.rerank_score == 0.95


class TestRetrievalMethod:
    """Test cases for RetrievalMethod enum."""

    def test_retrieval_method_values(self):
        """Test retrieval method enum values."""
        assert RetrievalMethod.SEMANTIC.value == "semantic"
        assert RetrievalMethod.BM25.value == "bm25"
        assert RetrievalMethod.HYBRID.value == "hybrid"


class TestCreateHybridRetriever:
    """Test cases for create_hybrid_retriever factory function."""

    def test_create_without_reranker(self, mock_semantic_retriever, sample_documents):
        """Test factory without reranker."""
        retriever = create_hybrid_retriever(
            semantic_retriever=mock_semantic_retriever,
            documents=sample_documents,
            enable_reranker=False,
            alpha=0.4
        )

        assert isinstance(retriever, HybridRetriever)
        assert retriever.alpha == 0.4
        assert retriever.reranker is None

    def test_create_with_auto_reranker(self, mock_semantic_retriever, sample_documents):
        """Test factory with auto reranker selection."""
        retriever = create_hybrid_retriever(
            semantic_retriever=mock_semantic_retriever,
            documents=sample_documents,
            enable_reranker=True,
            reranker_provider="auto"
        )

        assert isinstance(retriever, HybridRetriever)
        # Reranker may or may not be available depending on dependencies
