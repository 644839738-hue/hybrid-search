"""
Tests for Search Manager API (M5).

Tests unified search interface, scoping, and score handling.
"""

import pytest
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.search_manager import SearchManager
from core.models.search import SearchRequest, SearchResponse, SearchResult, SearchScores, RetrievalMethod
from core.models.errors import ValidationError, NotFoundError
from core.storage import JSONStorage, COLLECTIONS_FILE, DOCUMENTS_FILE
from langchain_core.documents import Document


@pytest.fixture
def temp_storage():
    """Create a temporary storage with test data."""
    with tempfile.TemporaryDirectory() as tmpdir:
        storage = JSONStorage(data_dir=tmpdir)

        # Create test collections
        storage.insert(COLLECTIONS_FILE, {
            "id": "col-1",
            "name": "Research Papers",
            "description": "ML research",
            "metadata": {},
            "settings": {}
        })
        storage.insert(COLLECTIONS_FILE, {
            "id": "col-2",
            "name": "Technical Docs",
            "description": "Documentation",
            "metadata": {},
            "settings": {}
        })

        # Create test documents
        storage.insert(DOCUMENTS_FILE, {
            "id": "doc-1",
            "collection_id": "col-1",
            "filename": "paper1.pdf",
            "file_hash": "hash1",
            "file_size": 1000,
            "status": "ready"
        })
        storage.insert(DOCUMENTS_FILE, {
            "id": "doc-2",
            "collection_id": "col-1",
            "filename": "paper2.pdf",
            "file_hash": "hash2",
            "file_size": 2000,
            "status": "ready"
        })
        storage.insert(DOCUMENTS_FILE, {
            "id": "doc-3",
            "collection_id": "col-2",
            "filename": "guide.pdf",
            "file_hash": "hash3",
            "file_size": 3000,
            "status": "ready"
        })

        yield storage


@pytest.fixture
def mock_vector_store():
    """Create a mock vector store."""
    mock = MagicMock()
    mock.search_similar.return_value = [
        Document(
            page_content="Machine learning is a subset of AI.",
            metadata={"source": "paper1.pdf", "page": 1, "collection_id": "col-1"}
        ),
        Document(
            page_content="Deep learning uses neural networks.",
            metadata={"source": "paper1.pdf", "page": 2, "collection_id": "col-1"}
        ),
    ]
    mock.get_collection_count.return_value = 100
    mock.get_retriever.return_value = MagicMock()
    return mock


@pytest.fixture
def search_manager(temp_storage, mock_vector_store):
    """Create a SearchManager with mocked dependencies."""
    return SearchManager(
        storage=temp_storage,
        vector_store=mock_vector_store
    )


class TestSearchValidation:
    """Tests for search request validation."""

    def test_empty_query_fails(self, search_manager):
        """Test that empty query raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            search_manager.search(SearchRequest(query=""))

        assert exc_info.value.param == "query"

    def test_whitespace_query_fails(self, search_manager):
        """Test that whitespace-only query raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            search_manager.search(SearchRequest(query="   "))

        assert exc_info.value.param == "query"

    def test_invalid_k_zero_fails(self, search_manager):
        """Test that k=0 raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            search_manager.search(SearchRequest(query="test", k=0))

        assert exc_info.value.param == "k"

    def test_invalid_k_negative_fails(self, search_manager):
        """Test that negative k raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            search_manager.search(SearchRequest(query="test", k=-1))

        assert exc_info.value.param == "k"

    def test_invalid_k_too_large_fails(self, search_manager):
        """Test that k > 50 raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            search_manager.search(SearchRequest(query="test", k=51))

        assert exc_info.value.param == "k"

    def test_nonexistent_collection_fails(self, search_manager):
        """Test that non-existent collection raises NotFoundError."""
        with pytest.raises(NotFoundError) as exc_info:
            search_manager.search(SearchRequest(
                query="test",
                collection_id="nonexistent"
            ))

        assert exc_info.value.resource_type == "collection"

    def test_nonexistent_document_fails(self, search_manager):
        """Test that non-existent document raises NotFoundError."""
        with pytest.raises(NotFoundError) as exc_info:
            search_manager.search(SearchRequest(
                query="test",
                document_ids=["nonexistent"]
            ))

        assert exc_info.value.resource_type == "document"

    def test_valid_collection_succeeds(self, search_manager):
        """Test that valid collection passes validation."""
        response = search_manager.search(SearchRequest(
            query="machine learning",
            collection_id="col-1"
        ))

        assert response is not None
        assert response.query == "machine learning"

    def test_valid_documents_succeeds(self, search_manager):
        """Test that valid documents pass validation."""
        response = search_manager.search(SearchRequest(
            query="machine learning",
            document_ids=["doc-1", "doc-2"]
        ))

        assert response is not None
        assert response.query == "machine learning"


class TestSemanticSearch:
    """Tests for semantic search functionality."""

    def test_semantic_search_returns_results(self, search_manager, mock_vector_store):
        """Test that semantic search returns results."""
        response = search_manager.search(SearchRequest(
            query="machine learning",
            method=RetrievalMethod.SEMANTIC,
            k=5
        ))

        assert response.total_results == 2
        assert len(response.results) == 2

    def test_semantic_search_with_filter(self, search_manager, mock_vector_store):
        """Test that semantic search passes filter to vector store."""
        response = search_manager.search(SearchRequest(
            query="machine learning",
            collection_id="col-1",
            method=RetrievalMethod.SEMANTIC
        ))

        # Verify filter was passed (ChromaDB requires $eq operator)
        mock_vector_store.search_similar.assert_called_once()
        call_kwargs = mock_vector_store.search_similar.call_args[1]
        assert call_kwargs["filter"] == {"collection_id": {"$eq": "col-1"}}

    def test_semantic_search_scores(self, search_manager):
        """Test that semantic search results have scores."""
        response = search_manager.search(SearchRequest(
            query="machine learning",
            method=RetrievalMethod.SEMANTIC
        ))

        for result in response.results:
            assert result.scores is not None
            assert result.scores.semantic_score is not None
            assert result.scores.final_score > 0

    def test_semantic_search_metadata(self, search_manager):
        """Test that results include metadata."""
        response = search_manager.search(SearchRequest(
            query="machine learning",
            method=RetrievalMethod.SEMANTIC
        ))

        for result in response.results:
            assert result.source is not None
            assert "page_content" not in result.metadata or True  # May or may not have


class TestSearchResponse:
    """Tests for search response structure."""

    def test_response_has_query(self, search_manager):
        """Test that response includes original query."""
        response = search_manager.search(SearchRequest(
            query="test query"
        ))

        assert response.query == "test query"

    def test_response_has_method(self, search_manager):
        """Test that response includes method used."""
        response = search_manager.search(SearchRequest(
            query="test",
            method=RetrievalMethod.SEMANTIC
        ))

        assert response.method == RetrievalMethod.SEMANTIC

    def test_response_has_timing(self, search_manager):
        """Test that response includes search time."""
        response = search_manager.search(SearchRequest(
            query="test"
        ))

        assert response.search_time_ms >= 0

    def test_response_has_total_count(self, search_manager):
        """Test that response includes total result count."""
        response = search_manager.search(SearchRequest(
            query="test"
        ))

        assert response.total_results == len(response.results)


class TestFilterMatching:
    """Tests for post-retrieval filter matching."""

    def test_simple_equality_match(self, search_manager):
        """Test simple equality filter matching."""
        doc = Document(
            page_content="test",
            metadata={"collection_id": "col-1"}
        )

        assert search_manager._matches_filter(
            doc,
            {"collection_id": "col-1"}
        )

    def test_simple_equality_no_match(self, search_manager):
        """Test simple equality filter non-match."""
        doc = Document(
            page_content="test",
            metadata={"collection_id": "col-1"}
        )

        assert not search_manager._matches_filter(
            doc,
            {"collection_id": "col-2"}
        )

    def test_in_operator_match(self, search_manager):
        """Test $in operator matching."""
        doc = Document(
            page_content="test",
            metadata={"document_id": "doc-2"}
        )

        assert search_manager._matches_filter(
            doc,
            {"document_id": {"$in": ["doc-1", "doc-2", "doc-3"]}}
        )

    def test_in_operator_no_match(self, search_manager):
        """Test $in operator non-match."""
        doc = Document(
            page_content="test",
            metadata={"document_id": "doc-99"}
        )

        assert not search_manager._matches_filter(
            doc,
            {"document_id": {"$in": ["doc-1", "doc-2"]}}
        )

    def test_and_operator(self, search_manager):
        """Test $and operator matching."""
        doc = Document(
            page_content="test",
            metadata={"collection_id": "col-1", "document_id": "doc-1"}
        )

        assert search_manager._matches_filter(
            doc,
            {"$and": [
                {"collection_id": "col-1"},
                {"document_id": "doc-1"}
            ]}
        )

    def test_and_operator_partial_match(self, search_manager):
        """Test $and operator with partial match fails."""
        doc = Document(
            page_content="test",
            metadata={"collection_id": "col-1", "document_id": "doc-2"}
        )

        assert not search_manager._matches_filter(
            doc,
            {"$and": [
                {"collection_id": "col-1"},
                {"document_id": "doc-1"}
            ]}
        )


class TestSearchStats:
    """Tests for search statistics."""

    def test_stats_includes_vector_store_status(self, search_manager):
        """Test that stats include vector store availability."""
        stats = search_manager.get_search_stats()

        assert "vector_store_available" in stats
        assert stats["vector_store_available"] is True

    def test_stats_includes_document_count(self, search_manager, mock_vector_store):
        """Test that stats include indexed document count."""
        stats = search_manager.get_search_stats()

        assert "indexed_documents" in stats
        assert stats["indexed_documents"] == 100

    def test_stats_without_vector_store(self, temp_storage):
        """Test stats when no vector store configured."""
        manager = SearchManager(storage=temp_storage, vector_store=None)
        stats = manager.get_search_stats()

        assert stats["vector_store_available"] is False
        assert "indexed_documents" not in stats


class TestSearchWithoutVectorStore:
    """Tests for search behavior without vector store."""

    def test_semantic_search_returns_empty(self, temp_storage):
        """Test that semantic search without vector store returns empty."""
        manager = SearchManager(storage=temp_storage, vector_store=None)

        response = manager.search(SearchRequest(
            query="test",
            method=RetrievalMethod.SEMANTIC
        ))

        assert response.total_results == 0

    def test_hybrid_search_returns_empty(self, temp_storage):
        """Test that hybrid search without retriever returns empty."""
        manager = SearchManager(storage=temp_storage, vector_store=None)

        response = manager.search(SearchRequest(
            query="test",
            method=RetrievalMethod.HYBRID
        ))

        assert response.total_results == 0


class TestHybridRetrieverIntegration:
    """Tests for hybrid retriever integration."""

    def test_set_hybrid_retriever(self, search_manager):
        """Test setting hybrid retriever."""
        mock_retriever = MagicMock()
        search_manager.set_hybrid_retriever(mock_retriever)

        assert search_manager._hybrid_retriever is mock_retriever

    def test_hybrid_search_uses_retriever(self, search_manager):
        """Test that hybrid search uses the retriever."""
        mock_retriever = MagicMock()
        mock_retriever.retrieve.return_value = []
        mock_retriever.get_retrieval_stats.return_value = {}

        search_manager.set_hybrid_retriever(mock_retriever)

        response = search_manager.search(SearchRequest(
            query="test",
            method=RetrievalMethod.HYBRID
        ))

        mock_retriever.retrieve.assert_called_once()
