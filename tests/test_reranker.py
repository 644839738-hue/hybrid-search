"""
Tests for Re-ranker module.
"""

import pytest
from unittest.mock import MagicMock, patch
from core.reranker import (
    CohereReranker,
    JinaReranker,
    RerankerFactory,
    RerankResult
)


class TestCohereReranker:
    """Test cases for CohereReranker class."""

    def test_initialization_without_api_key(self, monkeypatch):
        """Test initialization without COHERE_API_KEY."""
        monkeypatch.delenv("COHERE_API_KEY", raising=False)
        reranker = CohereReranker()
        assert not reranker.is_available()

    def test_initialization_with_api_key(self, monkeypatch):
        """Test initialization with COHERE_API_KEY set."""
        monkeypatch.setenv("COHERE_API_KEY", "test-key")

        with patch('cohere.Client') as mock_client:
            reranker = CohereReranker()
            # Should attempt to create client
            mock_client.assert_called_once_with("test-key")

    def test_rerank_not_available_raises_error(self, monkeypatch, sample_documents):
        """Test that rerank raises error when not available."""
        monkeypatch.delenv("COHERE_API_KEY", raising=False)
        reranker = CohereReranker()

        with pytest.raises(RuntimeError, match="not available"):
            reranker.rerank("test query", sample_documents)

    def test_rerank_empty_documents(self, monkeypatch):
        """Test rerank with empty documents returns empty list."""
        monkeypatch.setenv("COHERE_API_KEY", "test-key")

        with patch('cohere.Client'):
            reranker = CohereReranker()
            reranker._available = True
            result = reranker.rerank("query", [])
            assert result == []


class TestJinaReranker:
    """Test cases for JinaReranker class."""

    def test_initialization(self):
        """Test Jina reranker initialization."""
        # Note: This may fail if sentence-transformers is not installed
        # In that case, the reranker will be unavailable
        reranker = JinaReranker()
        # Just verify it doesn't raise an exception
        assert hasattr(reranker, 'model_name')

    def test_rerank_not_available_raises_error(self, sample_documents):
        """Test that rerank raises error when not available."""
        reranker = JinaReranker()
        reranker._available = False

        with pytest.raises(RuntimeError, match="not available"):
            reranker.rerank("test query", sample_documents)

    def test_rerank_empty_documents(self):
        """Test rerank with empty documents returns empty list."""
        reranker = JinaReranker()
        reranker._available = True

        # Mock the model
        reranker._model = MagicMock()

        result = reranker.rerank("query", [])
        assert result == []

    def test_rerank_with_mock_model(self, sample_documents):
        """Test rerank with mocked model."""
        reranker = JinaReranker()
        reranker._available = True

        # Mock the model to return scores
        mock_model = MagicMock()
        mock_model.predict.return_value = [0.9, 0.7, 0.5]
        reranker._model = mock_model

        results = reranker.rerank("test query", sample_documents)

        assert len(results) == len(sample_documents)
        assert all(isinstance(r, RerankResult) for r in results)
        # Results should be sorted by score
        scores = [r.score for r in results]
        assert scores == sorted(scores, reverse=True)


class TestRerankerFactory:
    """Test cases for RerankerFactory class."""

    def test_create_cohere_reranker(self, monkeypatch):
        """Test creating Cohere reranker via factory."""
        monkeypatch.delenv("COHERE_API_KEY", raising=False)
        reranker = RerankerFactory.create("cohere")
        assert isinstance(reranker, CohereReranker)

    def test_create_jina_reranker(self):
        """Test creating Jina reranker via factory."""
        reranker = RerankerFactory.create("jina")
        assert isinstance(reranker, JinaReranker)

    def test_create_unknown_provider_raises_error(self):
        """Test that unknown provider raises ValueError."""
        with pytest.raises(ValueError, match="Unknown reranker provider"):
            RerankerFactory.create("unknown_provider")

    def test_get_available_reranker(self, monkeypatch):
        """Test getting first available reranker."""
        monkeypatch.delenv("COHERE_API_KEY", raising=False)
        # This should fall back to Jina if available
        reranker = RerankerFactory.get_available_reranker()
        # Result depends on what's installed
        # Just verify it doesn't crash
        assert reranker is None or hasattr(reranker, 'rerank')


class TestRerankResult:
    """Test cases for RerankResult dataclass."""

    def test_rerank_result_creation(self, sample_documents):
        """Test creating RerankResult."""
        result = RerankResult(
            document=sample_documents[0],
            score=0.95,
            original_rank=3,
            new_rank=1
        )

        assert result.document == sample_documents[0]
        assert result.score == 0.95
        assert result.original_rank == 3
        assert result.new_rank == 1
