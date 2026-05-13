"""
Tests for BM25 Retriever module.
"""

import pytest
from core.bm25_retriever import BM25Retriever, BM25Result


class TestBM25Retriever:
    """Test cases for BM25Retriever class."""

    def test_initialization_default_params(self):
        """Test BM25 retriever initialization with default parameters."""
        retriever = BM25Retriever()
        assert retriever.k1 == 1.5
        assert retriever.b == 0.75
        assert retriever.bm25 is None
        assert len(retriever.documents) == 0

    def test_initialization_custom_params(self):
        """Test BM25 retriever initialization with custom parameters."""
        retriever = BM25Retriever(k1=2.0, b=0.5)
        assert retriever.k1 == 2.0
        assert retriever.b == 0.5

    def test_index_documents(self, sample_documents):
        """Test document indexing."""
        retriever = BM25Retriever()
        count = retriever.index_documents(sample_documents)

        assert count == len(sample_documents)
        assert retriever.is_indexed()
        assert retriever.get_document_count() == len(sample_documents)

    def test_index_empty_documents_raises_error(self):
        """Test that indexing empty documents raises ValueError."""
        retriever = BM25Retriever()
        with pytest.raises(ValueError, match="Cannot index empty document list"):
            retriever.index_documents([])

    def test_retrieve_before_indexing_raises_error(self):
        """Test that retrieving before indexing raises ValueError."""
        retriever = BM25Retriever()
        with pytest.raises(ValueError, match="Index not built"):
            retriever.retrieve("test query")

    def test_retrieve_returns_results(self, sample_documents):
        """Test that retrieve returns BM25Result objects."""
        retriever = BM25Retriever()
        retriever.index_documents(sample_documents)

        results = retriever.retrieve("machine learning", k=2)

        assert len(results) <= 2
        assert all(isinstance(r, BM25Result) for r in results)
        assert all(r.score >= 0 for r in results)
        assert all(r.rank >= 1 for r in results)

    def test_retrieve_respects_k_parameter(self, sample_documents_large):
        """Test that retrieve respects k parameter."""
        retriever = BM25Retriever()
        retriever.index_documents(sample_documents_large)

        results_k3 = retriever.retrieve("neural networks", k=3)
        results_k5 = retriever.retrieve("neural networks", k=5)

        assert len(results_k3) <= 3
        assert len(results_k5) <= 5

    def test_retrieve_sorted_by_score(self, sample_documents_large):
        """Test that results are sorted by score descending."""
        retriever = BM25Retriever()
        retriever.index_documents(sample_documents_large)

        results = retriever.retrieve("learning", k=5)

        scores = [r.score for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_retrieve_with_threshold(self, sample_documents):
        """Test retrieve with score threshold."""
        retriever = BM25Retriever()
        retriever.index_documents(sample_documents)

        results = retriever.retrieve("machine learning", k=10, score_threshold=0.1)

        assert all(r.score >= 0.1 for r in results)

    def test_retrieve_with_scores(self, sample_documents):
        """Test retrieve_with_scores returns tuples."""
        retriever = BM25Retriever()
        retriever.index_documents(sample_documents)

        results = retriever.retrieve_with_scores("artificial intelligence", k=2)

        assert all(isinstance(r, tuple) for r in results)
        assert all(len(r) == 2 for r in results)

    def test_get_top_documents(self, sample_documents):
        """Test get_top_documents returns Document objects."""
        retriever = BM25Retriever()
        retriever.index_documents(sample_documents)

        docs = retriever.get_top_documents("learning", k=2)

        assert len(docs) <= 2
        assert all(hasattr(d, 'page_content') for d in docs)

    def test_clear_index(self, sample_documents):
        """Test clearing the index."""
        retriever = BM25Retriever()
        retriever.index_documents(sample_documents)
        assert retriever.is_indexed()

        retriever.clear_index()

        assert not retriever.is_indexed()
        assert retriever.get_document_count() == 0

    def test_tokenization(self):
        """Test tokenization produces lowercase tokens."""
        retriever = BM25Retriever()
        tokens = retriever._tokenize("Machine Learning AND Deep Learning")

        assert "machine" in tokens
        assert "learning" in tokens
        assert "deep" in tokens
        # Special characters should be removed
        assert "AND" not in tokens

    def test_empty_query_returns_empty_results(self, sample_documents):
        """Test that empty tokenized query returns empty results."""
        retriever = BM25Retriever()
        retriever.index_documents(sample_documents)

        # Query with only special characters
        results = retriever.retrieve("!@#$%", k=3)
        assert results == []
