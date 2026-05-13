"""
Tests for data models (M1).

Tests collection, document, search models and error classes.
"""

import pytest
import sys
from pathlib import Path
from datetime import datetime

# Add project root to path for direct imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.models.collection import Collection, CollectionSettings
from core.models.document import Document, DocumentStatus
from core.models.search import (
    SearchRequest, SearchResult, SearchScores, SearchResponse, RetrievalMethod
)
from core.models.errors import (
    APIError, ValidationError, NotFoundError, DuplicateError, LimitExceededError
)
from core.models.responses import ListResponse, DeletedResponse, OperationResult


class TestCollectionSettings:
    """Tests for CollectionSettings dataclass."""

    def test_default_values(self):
        """Test default settings are applied."""
        settings = CollectionSettings()
        assert settings.chunk_size == 1000
        assert settings.chunk_overlap == 200
        assert settings.embedding_model == "text-embedding-3-large"

    def test_custom_values(self):
        """Test custom settings are preserved."""
        settings = CollectionSettings(
            chunk_size=500,
            chunk_overlap=100,
            embedding_model="text-embedding-ada-002"
        )
        assert settings.chunk_size == 500
        assert settings.chunk_overlap == 100
        assert settings.embedding_model == "text-embedding-ada-002"

    def test_to_dict(self):
        """Test serialization to dictionary."""
        settings = CollectionSettings(chunk_size=800)
        data = settings.to_dict()
        assert data["chunk_size"] == 800
        assert "chunk_overlap" in data
        assert "embedding_model" in data

    def test_from_dict(self):
        """Test deserialization from dictionary."""
        data = {"chunk_size": 600, "chunk_overlap": 150}
        settings = CollectionSettings.from_dict(data)
        assert settings.chunk_size == 600
        assert settings.chunk_overlap == 150
        assert settings.embedding_model == "text-embedding-3-large"  # default


class TestCollection:
    """Tests for Collection dataclass."""

    def test_create_factory(self):
        """Test Collection.create factory method."""
        collection = Collection.create(
            name="Test Collection",
            description="A test collection"
        )
        assert collection.id is not None
        assert len(collection.id) == 36  # UUID format
        assert collection.name == "Test Collection"
        assert collection.description == "A test collection"
        assert isinstance(collection.created_at, datetime)
        assert isinstance(collection.settings, CollectionSettings)

    def test_to_dict_without_computed(self):
        """Test serialization excludes computed fields by default."""
        collection = Collection.create(name="Test")
        collection.document_count = 5
        collection.chunk_count = 100

        data = collection.to_dict(include_computed=False)
        assert "document_count" not in data
        assert "chunk_count" not in data

    def test_to_dict_with_computed(self):
        """Test serialization includes computed fields when requested."""
        collection = Collection.create(name="Test")
        collection.document_count = 5
        collection.chunk_count = 100

        data = collection.to_dict(include_computed=True)
        assert data["document_count"] == 5
        assert data["chunk_count"] == 100

    def test_from_dict(self):
        """Test deserialization from dictionary."""
        data = {
            "id": "test-id-123",
            "name": "Restored Collection",
            "description": "From dict",
            "metadata": {"key": "value"},
            "created_at": "2024-01-01T00:00:00",
            "updated_at": "2024-01-02T00:00:00",
            "settings": {"chunk_size": 500}
        }
        collection = Collection.from_dict(data)
        assert collection.id == "test-id-123"
        assert collection.name == "Restored Collection"
        assert collection.metadata == {"key": "value"}
        assert collection.settings.chunk_size == 500

    def test_update_method(self):
        """Test update creates new instance with changes."""
        original = Collection.create(name="Original", description="First")
        updated = original.update(name="Updated", description="Second")

        assert original.name == "Original"  # Original unchanged
        assert updated.name == "Updated"
        assert updated.description == "Second"
        assert updated.id == original.id  # Same ID
        assert updated.updated_at > original.created_at


class TestDocument:
    """Tests for Document dataclass."""

    def test_create_factory(self):
        """Test Document.create factory method."""
        doc = Document.create(
            collection_id="col-123",
            filename="test.pdf",
            file_hash="abc123hash",
            file_size=1024
        )
        assert doc.id is not None
        assert doc.collection_id == "col-123"
        assert doc.filename == "test.pdf"
        assert doc.status == DocumentStatus.PROCESSING

    def test_mark_ready(self):
        """Test marking document as ready."""
        doc = Document.create(
            collection_id="col-123",
            filename="test.pdf",
            file_hash="abc123",
            file_size=1024
        )
        ready_doc = doc.mark_ready(page_count=10, chunk_count=50)

        assert ready_doc.status == DocumentStatus.READY
        assert ready_doc.page_count == 10
        assert ready_doc.chunk_count == 50
        assert ready_doc.is_ready

    def test_mark_failed(self):
        """Test marking document as failed."""
        doc = Document.create(
            collection_id="col-123",
            filename="test.pdf",
            file_hash="abc123",
            file_size=1024
        )
        failed_doc = doc.mark_failed("PDF parsing error")

        assert failed_doc.status == DocumentStatus.FAILED
        assert failed_doc.error_message == "PDF parsing error"
        assert failed_doc.is_failed

    def test_format_size(self):
        """Test human-readable file size formatting."""
        doc = Document.create(
            collection_id="col-123",
            filename="test.pdf",
            file_hash="abc123",
            file_size=1536  # 1.5 KB
        )
        assert "KB" in doc.format_size()

    def test_to_dict_and_from_dict(self):
        """Test round-trip serialization."""
        original = Document.create(
            collection_id="col-123",
            filename="test.pdf",
            file_hash="abc123",
            file_size=2048
        )
        original = original.mark_ready(page_count=5, chunk_count=25)

        data = original.to_dict()
        restored = Document.from_dict(data)

        assert restored.id == original.id
        assert restored.status == DocumentStatus.READY
        assert restored.page_count == 5


class TestSearchModels:
    """Tests for search-related models."""

    def test_search_scores(self):
        """Test SearchScores dataclass."""
        scores = SearchScores(
            semantic_score=0.85,
            bm25_score=12.5,
            combined_score=0.72,
            rerank_score=0.90,
            final_score=0.90
        )
        data = scores.to_dict()
        assert data["semantic_score"] == 0.85
        assert data["rerank_score"] == 0.90

    def test_search_request_filter_collection(self):
        """Test filter generation for collection scope.

        Note: ChromaDB requires explicit $eq operator for equality filters.
        Direct equality like {"field": "value"} may silently fail.
        """
        request = SearchRequest(
            query="test query",
            collection_id="col-123"
        )
        filter_dict = request.get_filter()
        # ChromaDB requires explicit $eq operator
        assert filter_dict == {"collection_id": {"$eq": "col-123"}}

    def test_search_request_filter_documents(self):
        """Test filter generation for document scope with multiple docs."""
        request = SearchRequest(
            query="test query",
            document_ids=["doc-1", "doc-2"]
        )
        filter_dict = request.get_filter()
        assert filter_dict == {"document_id": {"$in": ["doc-1", "doc-2"]}}

    def test_search_request_filter_single_document(self):
        """Test filter generation for single document scope uses $eq."""
        request = SearchRequest(
            query="test query",
            document_ids=["doc-1"]
        )
        filter_dict = request.get_filter()
        # Single document should use $eq, not $in
        assert filter_dict == {"document_id": {"$eq": "doc-1"}}

    def test_search_request_filter_combined(self):
        """Test filter generation for combined scope."""
        request = SearchRequest(
            query="test query",
            collection_id="col-123",
            document_ids=["doc-1"]
        )
        filter_dict = request.get_filter()
        assert "$and" in filter_dict

    def test_search_result(self):
        """Test SearchResult dataclass."""
        result = SearchResult(
            content="This is the content...",
            source="paper.pdf",
            metadata={"document_id": "doc-123", "collection_id": "col-456"},
            scores=SearchScores(final_score=0.85)
        )
        # Test properties that read from metadata
        assert result.document_id == "doc-123"
        assert result.collection_id == "col-456"
        data = result.to_dict()
        assert data["source"] == "paper.pdf"
        assert data["scores"]["final_score"] == 0.85

    def test_search_response(self):
        """Test SearchResponse dataclass."""
        results = [
            SearchResult(
                content="Content A",
                source="a.pdf",
                metadata={"document_id": "doc-1", "collection_id": "col-1"}
            ),
            SearchResult(
                content="Content B",
                source="b.pdf",
                metadata={"document_id": "doc-2", "collection_id": "col-1"}
            )
        ]
        response = SearchResponse(
            results=results,
            query="test",
            method=RetrievalMethod.HYBRID,
            search_time_ms=150.5
        )
        assert response.total_results == 2
        data = response.to_dict()
        assert len(data["results"]) == 2


class TestErrors:
    """Tests for error classes."""

    def test_api_error_to_dict(self):
        """Test APIError serialization."""
        error = APIError(
            code="test_error",
            message="Something went wrong",
            param="test_param",
            error_type="test_type"
        )
        data = error.to_dict()
        assert data["error"]["code"] == "test_error"
        assert data["error"]["message"] == "Something went wrong"
        assert data["error"]["param"] == "test_param"

    def test_validation_error(self):
        """Test ValidationError defaults."""
        error = ValidationError(message="Invalid input", param="name")
        assert error.code == "validation_error"
        assert error.error_type == "validation_error"

    def test_not_found_error(self):
        """Test NotFoundError with resource details."""
        error = NotFoundError(
            message="Collection not found",
            resource_type="collection",
            resource_id="abc123"
        )
        data = error.to_dict()
        assert data["error"]["details"]["resource_type"] == "collection"
        assert data["error"]["details"]["resource_id"] == "abc123"

    def test_duplicate_error(self):
        """Test DuplicateError with existing ID."""
        error = DuplicateError(
            message="Collection already exists",
            existing_id="existing-123"
        )
        data = error.to_dict()
        assert data["error"]["details"]["existing_id"] == "existing-123"

    def test_limit_exceeded_error(self):
        """Test LimitExceededError with limit details."""
        error = LimitExceededError(
            message="Too many collections",
            limit=3,
            current=3
        )
        data = error.to_dict()
        assert data["error"]["details"]["limit"] == 3
        assert data["error"]["details"]["current"] == 3


class TestResponses:
    """Tests for response models."""

    def test_list_response(self):
        """Test ListResponse with pagination."""
        items = [{"id": "1"}, {"id": "2"}]
        response = ListResponse(
            data=items,
            has_more=True,
            total_count=5,
            next_cursor="cursor-123"
        )
        assert len(response) == 2
        assert response.has_more
        data = response.to_dict()
        assert data["object"] == "list"
        assert data["total_count"] == 5

    def test_list_response_iteration(self):
        """Test ListResponse can be iterated."""
        items = [{"id": "1"}, {"id": "2"}, {"id": "3"}]
        response = ListResponse(data=items)
        ids = [item["id"] for item in response]
        assert ids == ["1", "2", "3"]

    def test_deleted_response(self):
        """Test DeletedResponse."""
        response = DeletedResponse(id="del-123", object="collection")
        data = response.to_dict()
        assert data["id"] == "del-123"
        assert data["deleted"] is True

    def test_operation_result_with_warnings(self):
        """Test OperationResult with warnings."""
        result = OperationResult(
            success=True,
            data={"id": "new-123"},
            warnings=["Approaching limit"]
        )
        assert result.has_warnings
        data = result.to_dict()
        assert data["warnings"] == ["Approaching limit"]
