"""
Tests for Document Manager API (M3).

Tests document CRUD, deduplication, and cascade deletion.
"""

import pytest
import sys
import tempfile
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.document_manager import DocumentManager
from core.models.document import Document, DocumentStatus
from core.models.errors import ValidationError, NotFoundError, DuplicateError
from core.storage import JSONStorage, COLLECTIONS_FILE, DOCUMENTS_FILE


@pytest.fixture
def temp_manager():
    """Create a manager with temporary storage."""
    with tempfile.TemporaryDirectory() as tmpdir:
        storage = JSONStorage(data_dir=tmpdir)

        # Create a test collection
        storage.insert(COLLECTIONS_FILE, {
            "id": "test-collection-1",
            "name": "Test Collection",
            "description": "For testing",
            "metadata": {},
            "settings": {}
        })

        manager = DocumentManager(storage=storage)
        yield manager, storage


@pytest.fixture
def sample_pdf_content():
    """Sample PDF-like content for testing."""
    # Just some bytes - not a real PDF but enough for testing
    return b"%PDF-1.4 sample content for testing hash calculation"


@pytest.fixture
def another_pdf_content():
    """Different content for duplicate testing."""
    return b"%PDF-1.4 different content to produce different hash"


class TestDocumentAdd:
    """Tests for document creation."""

    def test_add_basic(self, temp_manager, sample_pdf_content):
        """Test basic document addition."""
        manager, _ = temp_manager
        result = manager.add(
            collection_id="test-collection-1",
            filename="test.pdf",
            file_content=sample_pdf_content
        )

        assert result.success
        assert result.data is not None
        assert result.data.filename == "test.pdf"
        assert result.data.status == DocumentStatus.PROCESSING

    def test_add_with_metadata(self, temp_manager, sample_pdf_content):
        """Test adding document with metadata."""
        manager, _ = temp_manager
        result = manager.add(
            collection_id="test-collection-1",
            filename="test.pdf",
            file_content=sample_pdf_content,
            metadata={"author": "Test Author", "year": 2024}
        )

        assert result.success
        assert result.data.metadata["author"] == "Test Author"

    def test_add_calculates_hash(self, temp_manager, sample_pdf_content):
        """Test that file hash is calculated."""
        manager, _ = temp_manager
        result = manager.add(
            collection_id="test-collection-1",
            filename="test.pdf",
            file_content=sample_pdf_content
        )

        assert result.data.file_hash is not None
        assert len(result.data.file_hash) == 64  # SHA256 hex length

    def test_add_records_file_size(self, temp_manager, sample_pdf_content):
        """Test that file size is recorded."""
        manager, _ = temp_manager
        result = manager.add(
            collection_id="test-collection-1",
            filename="test.pdf",
            file_content=sample_pdf_content
        )

        assert result.data.file_size == len(sample_pdf_content)

    def test_add_empty_filename_fails(self, temp_manager, sample_pdf_content):
        """Test that empty filename raises ValidationError."""
        manager, _ = temp_manager
        with pytest.raises(ValidationError) as exc_info:
            manager.add(
                collection_id="test-collection-1",
                filename="",
                file_content=sample_pdf_content
            )

        assert exc_info.value.param == "filename"

    def test_add_non_pdf_fails(self, temp_manager):
        """Test that non-PDF files raise ValidationError."""
        manager, _ = temp_manager
        with pytest.raises(ValidationError) as exc_info:
            manager.add(
                collection_id="test-collection-1",
                filename="test.txt",
                file_content=b"text content"
            )

        assert "PDF" in exc_info.value.message

    def test_add_nonexistent_collection_fails(self, temp_manager, sample_pdf_content):
        """Test that adding to non-existent collection fails."""
        manager, _ = temp_manager
        with pytest.raises(NotFoundError) as exc_info:
            manager.add(
                collection_id="nonexistent",
                filename="test.pdf",
                file_content=sample_pdf_content
            )

        assert exc_info.value.resource_type == "collection"

    def test_add_duplicate_fails(self, temp_manager, sample_pdf_content):
        """Test that duplicate content raises DuplicateError."""
        manager, _ = temp_manager

        # Add first document
        manager.add(
            collection_id="test-collection-1",
            filename="first.pdf",
            file_content=sample_pdf_content
        )

        # Try to add same content with different name
        with pytest.raises(DuplicateError) as exc_info:
            manager.add(
                collection_id="test-collection-1",
                filename="second.pdf",
                file_content=sample_pdf_content
            )

        assert exc_info.value.existing_id is not None

    def test_add_same_content_different_collection_ok(
        self, temp_manager, sample_pdf_content
    ):
        """Test that same content in different collection is OK."""
        manager, storage = temp_manager

        # Create second collection
        storage.insert(COLLECTIONS_FILE, {
            "id": "test-collection-2",
            "name": "Second Collection",
            "metadata": {},
            "settings": {}
        })

        # Add to first collection
        manager.add(
            collection_id="test-collection-1",
            filename="test.pdf",
            file_content=sample_pdf_content
        )

        # Add same to second collection - should succeed
        result = manager.add(
            collection_id="test-collection-2",
            filename="test.pdf",
            file_content=sample_pdf_content
        )

        assert result.success

    def test_add_soft_limit_warning(self, temp_manager, sample_pdf_content):
        """Test that soft limit triggers warning."""
        manager, _ = temp_manager

        # Add documents up to soft limit
        for i in range(DocumentManager.SOFT_LIMIT_DOCUMENTS):
            content = sample_pdf_content + str(i).encode()
            manager.add(
                collection_id="test-collection-1",
                filename=f"doc{i}.pdf",
                file_content=content
            )

        # Next addition should warn
        extra_content = sample_pdf_content + b"extra"
        result = manager.add(
            collection_id="test-collection-1",
            filename="over_limit.pdf",
            file_content=extra_content
        )

        assert result.success
        assert result.has_warnings
        assert "limit" in result.warnings[0].lower()


class TestDocumentGet:
    """Tests for document retrieval."""

    def test_get_existing(self, temp_manager, sample_pdf_content):
        """Test getting an existing document."""
        manager, _ = temp_manager
        result = manager.add(
            collection_id="test-collection-1",
            filename="test.pdf",
            file_content=sample_pdf_content
        )

        doc = manager.get(result.data.id)

        assert doc.id == result.data.id
        assert doc.filename == "test.pdf"

    def test_get_nonexistent(self, temp_manager):
        """Test that getting non-existent raises NotFoundError."""
        manager, _ = temp_manager
        with pytest.raises(NotFoundError) as exc_info:
            manager.get("nonexistent-id")

        assert exc_info.value.resource_type == "document"

    def test_get_by_hash(self, temp_manager, sample_pdf_content):
        """Test finding document by hash."""
        manager, _ = temp_manager
        result = manager.add(
            collection_id="test-collection-1",
            filename="test.pdf",
            file_content=sample_pdf_content
        )

        doc = manager.get_by_hash(
            "test-collection-1",
            result.data.file_hash
        )

        assert doc is not None
        assert doc.id == result.data.id

    def test_get_by_hash_not_found(self, temp_manager):
        """Test get_by_hash returns None when not found."""
        manager, _ = temp_manager
        doc = manager.get_by_hash("test-collection-1", "nonexistent-hash")
        assert doc is None


class TestDocumentList:
    """Tests for listing documents."""

    def test_list_empty(self, temp_manager):
        """Test listing when no documents exist."""
        manager, _ = temp_manager
        response = manager.list(collection_id="test-collection-1")

        assert len(response) == 0
        assert response.total_count == 0

    def test_list_all(self, temp_manager, sample_pdf_content):
        """Test listing all documents."""
        manager, _ = temp_manager

        # Add some documents
        for i in range(3):
            content = sample_pdf_content + str(i).encode()
            manager.add(
                collection_id="test-collection-1",
                filename=f"doc{i}.pdf",
                file_content=content
            )

        response = manager.list(collection_id="test-collection-1")

        assert len(response) == 3
        assert response.total_count == 3

    def test_list_with_limit(self, temp_manager, sample_pdf_content):
        """Test listing with limit."""
        manager, _ = temp_manager

        # Add 5 documents
        for i in range(5):
            content = sample_pdf_content + str(i).encode()
            manager.add(
                collection_id="test-collection-1",
                filename=f"doc{i}.pdf",
                file_content=content
            )

        response = manager.list(collection_id="test-collection-1", limit=2)

        assert len(response) == 2
        assert response.has_more
        assert response.total_count == 5

    def test_list_pagination(self, temp_manager, sample_pdf_content):
        """Test pagination."""
        manager, _ = temp_manager

        # Add 5 documents
        for i in range(5):
            content = sample_pdf_content + str(i).encode()
            manager.add(
                collection_id="test-collection-1",
                filename=f"doc{i}.pdf",
                file_content=content
            )

        # Get pages
        page1 = manager.list(collection_id="test-collection-1", limit=2)
        page2 = manager.list(
            collection_id="test-collection-1",
            limit=2,
            starting_after=page1.next_cursor
        )

        assert len(page1) == 2
        assert len(page2) == 2
        assert page1[0].id != page2[0].id

    def test_list_by_status(self, temp_manager, sample_pdf_content):
        """Test filtering by status."""
        manager, _ = temp_manager

        # Add documents
        result1 = manager.add(
            collection_id="test-collection-1",
            filename="doc1.pdf",
            file_content=sample_pdf_content
        )
        result2 = manager.add(
            collection_id="test-collection-1",
            filename="doc2.pdf",
            file_content=sample_pdf_content + b"2"
        )

        # Mark one as ready
        manager.update_status(result1.data.id, DocumentStatus.READY, 10, 50)

        # List only ready
        response = manager.list(
            collection_id="test-collection-1",
            status=DocumentStatus.READY
        )

        assert len(response) == 1
        assert response[0].status == DocumentStatus.READY


class TestDocumentUpdateStatus:
    """Tests for status updates."""

    def test_update_to_ready(self, temp_manager, sample_pdf_content):
        """Test marking document as ready."""
        manager, _ = temp_manager
        result = manager.add(
            collection_id="test-collection-1",
            filename="test.pdf",
            file_content=sample_pdf_content
        )

        updated = manager.update_status(
            result.data.id,
            DocumentStatus.READY,
            page_count=10,
            chunk_count=50
        )

        assert updated.status == DocumentStatus.READY
        assert updated.page_count == 10
        assert updated.chunk_count == 50

    def test_update_to_failed(self, temp_manager, sample_pdf_content):
        """Test marking document as failed."""
        manager, _ = temp_manager
        result = manager.add(
            collection_id="test-collection-1",
            filename="test.pdf",
            file_content=sample_pdf_content
        )

        updated = manager.update_status(
            result.data.id,
            DocumentStatus.FAILED,
            error_message="PDF parsing error"
        )

        assert updated.status == DocumentStatus.FAILED
        assert updated.error_message == "PDF parsing error"


class TestDocumentDelete:
    """Tests for document deletion."""

    def test_delete_existing(self, temp_manager, sample_pdf_content):
        """Test deleting an existing document."""
        manager, _ = temp_manager
        result = manager.add(
            collection_id="test-collection-1",
            filename="test.pdf",
            file_content=sample_pdf_content
        )

        response = manager.delete(result.data.id)

        assert response.deleted
        assert response.object == "document"

    def test_delete_removes_from_storage(self, temp_manager, sample_pdf_content):
        """Test that delete removes from storage."""
        manager, _ = temp_manager
        result = manager.add(
            collection_id="test-collection-1",
            filename="test.pdf",
            file_content=sample_pdf_content
        )

        manager.delete(result.data.id)

        assert not manager.exists(result.data.id)

    def test_delete_nonexistent(self, temp_manager):
        """Test that deleting non-existent raises NotFoundError."""
        manager, _ = temp_manager
        with pytest.raises(NotFoundError):
            manager.delete("nonexistent")

    def test_delete_by_collection(self, temp_manager, sample_pdf_content):
        """Test deleting all documents in a collection."""
        manager, _ = temp_manager

        # Add documents
        for i in range(3):
            content = sample_pdf_content + str(i).encode()
            manager.add(
                collection_id="test-collection-1",
                filename=f"doc{i}.pdf",
                file_content=content
            )

        deleted_count = manager.delete_by_collection("test-collection-1")

        assert deleted_count == 3
        assert manager.count("test-collection-1") == 0


class TestDocumentUtilities:
    """Tests for utility methods."""

    def test_exists_true(self, temp_manager, sample_pdf_content):
        """Test exists returns True for existing document."""
        manager, _ = temp_manager
        result = manager.add(
            collection_id="test-collection-1",
            filename="test.pdf",
            file_content=sample_pdf_content
        )

        assert manager.exists(result.data.id)

    def test_exists_false(self, temp_manager):
        """Test exists returns False for non-existing."""
        manager, _ = temp_manager
        assert not manager.exists("nonexistent")

    def test_count_all(self, temp_manager, sample_pdf_content):
        """Test counting all documents."""
        manager, storage = temp_manager

        # Create second collection
        storage.insert(COLLECTIONS_FILE, {
            "id": "test-collection-2",
            "name": "Second",
            "metadata": {},
            "settings": {}
        })

        # Add to both collections
        manager.add(
            collection_id="test-collection-1",
            filename="doc1.pdf",
            file_content=sample_pdf_content
        )
        manager.add(
            collection_id="test-collection-2",
            filename="doc2.pdf",
            file_content=sample_pdf_content
        )

        assert manager.count() == 2

    def test_count_by_collection(self, temp_manager, sample_pdf_content):
        """Test counting documents in a collection."""
        manager, storage = temp_manager

        # Create second collection
        storage.insert(COLLECTIONS_FILE, {
            "id": "test-collection-2",
            "name": "Second",
            "metadata": {},
            "settings": {}
        })

        # Add to collections
        manager.add(
            collection_id="test-collection-1",
            filename="doc1.pdf",
            file_content=sample_pdf_content
        )
        manager.add(
            collection_id="test-collection-1",
            filename="doc2.pdf",
            file_content=sample_pdf_content + b"2"
        )
        manager.add(
            collection_id="test-collection-2",
            filename="doc3.pdf",
            file_content=sample_pdf_content + b"3"
        )

        assert manager.count("test-collection-1") == 2
        assert manager.count("test-collection-2") == 1
