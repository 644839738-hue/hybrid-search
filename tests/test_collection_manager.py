"""
Tests for Collection Manager API (M2).

Tests CRUD operations, soft limits, validation, and error handling.
"""

import pytest
import sys
import tempfile
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.collection_manager import CollectionManager
from core.models.collection import Collection, CollectionSettings
from core.models.errors import ValidationError, NotFoundError, DuplicateError
from core.storage import JSONStorage, COLLECTIONS_FILE, DOCUMENTS_FILE


@pytest.fixture
def temp_manager():
    """Create a manager with temporary storage."""
    with tempfile.TemporaryDirectory() as tmpdir:
        storage = JSONStorage(data_dir=tmpdir)
        manager = CollectionManager(storage=storage)
        yield manager


@pytest.fixture
def manager_with_collections(temp_manager):
    """Create a manager with some pre-existing collections."""
    temp_manager.create(name="Collection A", description="First collection")
    temp_manager.create(name="Collection B", description="Second collection")
    return temp_manager


class TestCollectionCreate:
    """Tests for collection creation."""

    def test_create_basic(self, temp_manager):
        """Test basic collection creation."""
        result = temp_manager.create(name="Test Collection")

        assert result.success
        assert result.data is not None
        assert result.data.name == "Test Collection"
        assert result.data.id is not None

    def test_create_with_all_fields(self, temp_manager):
        """Test creation with all optional fields."""
        settings = CollectionSettings(chunk_size=500)
        result = temp_manager.create(
            name="Full Collection",
            description="A detailed description",
            metadata={"project": "research", "year": 2024},
            settings=settings
        )

        assert result.success
        collection = result.data
        assert collection.description == "A detailed description"
        assert collection.metadata["project"] == "research"
        assert collection.settings.chunk_size == 500

    def test_create_empty_name_fails(self, temp_manager):
        """Test that empty name raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            temp_manager.create(name="")

        assert exc_info.value.param == "name"

    def test_create_whitespace_name_fails(self, temp_manager):
        """Test that whitespace-only name raises ValidationError."""
        with pytest.raises(ValidationError):
            temp_manager.create(name="   ")

    def test_create_duplicate_name_fails(self, temp_manager):
        """Test that duplicate name raises DuplicateError."""
        temp_manager.create(name="Unique Name")

        with pytest.raises(DuplicateError) as exc_info:
            temp_manager.create(name="Unique Name")

        assert exc_info.value.existing_id is not None

    def test_create_name_trimmed(self, temp_manager):
        """Test that name whitespace is trimmed."""
        result = temp_manager.create(name="  Trimmed Name  ")

        assert result.data.name == "Trimmed Name"

    def test_create_soft_limit_warning(self, temp_manager):
        """Test that soft limit triggers warning but allows creation."""
        # Create collections up to soft limit
        for i in range(CollectionManager.SOFT_LIMIT_COLLECTIONS):
            temp_manager.create(name=f"Collection {i}")

        # Next creation should succeed with warning
        result = temp_manager.create(name="Over Limit")

        assert result.success
        assert result.has_warnings
        assert "limit" in result.warnings[0].lower()


class TestCollectionGet:
    """Tests for collection retrieval."""

    def test_get_existing(self, manager_with_collections):
        """Test getting an existing collection."""
        # First create and get the ID
        result = manager_with_collections.create(name="To Get")
        collection_id = result.data.id

        # Now retrieve it
        collection = manager_with_collections.get(collection_id)

        assert collection.id == collection_id
        assert collection.name == "To Get"

    def test_get_nonexistent(self, temp_manager):
        """Test that getting non-existent collection raises NotFoundError."""
        with pytest.raises(NotFoundError) as exc_info:
            temp_manager.get("nonexistent-id")

        assert exc_info.value.resource_type == "collection"

    def test_get_by_name(self, manager_with_collections):
        """Test getting collection by name."""
        collection = manager_with_collections.get_by_name("Collection A")

        assert collection is not None
        assert collection.name == "Collection A"

    def test_get_by_name_nonexistent(self, temp_manager):
        """Test that get_by_name returns None for non-existent."""
        result = temp_manager.get_by_name("Nonexistent")
        assert result is None

    def test_get_includes_stats(self, temp_manager):
        """Test that get includes document/chunk counts."""
        result = temp_manager.create(name="With Stats")
        collection = temp_manager.get(result.data.id)

        # New collection has 0 documents
        assert collection.document_count == 0
        assert collection.chunk_count == 0


class TestCollectionList:
    """Tests for listing collections."""

    def test_list_empty(self, temp_manager):
        """Test listing when no collections exist."""
        response = temp_manager.list()

        assert len(response) == 0
        assert response.total_count == 0
        assert not response.has_more

    def test_list_all(self, manager_with_collections):
        """Test listing all collections."""
        response = manager_with_collections.list()

        assert len(response) >= 2
        assert response.total_count >= 2

    def test_list_with_limit(self, manager_with_collections):
        """Test listing with limit."""
        # Create more collections
        for i in range(5):
            manager_with_collections.create(name=f"Extra {i}")

        response = manager_with_collections.list(limit=3)

        assert len(response) == 3
        assert response.has_more
        assert response.next_cursor is not None

    def test_list_pagination(self, temp_manager):
        """Test pagination through collections."""
        # Create 5 collections
        for i in range(5):
            temp_manager.create(name=f"Page Test {i}")

        # Get first page
        page1 = temp_manager.list(limit=2)
        assert len(page1) == 2
        assert page1.has_more

        # Get second page
        page2 = temp_manager.list(limit=2, starting_after=page1.next_cursor)
        assert len(page2) == 2
        assert page2.has_more

        # Get last page
        page3 = temp_manager.list(limit=2, starting_after=page2.next_cursor)
        assert len(page3) == 1
        assert not page3.has_more

    def test_list_sorted_by_date(self, temp_manager):
        """Test that list returns newest first."""
        temp_manager.create(name="First")
        temp_manager.create(name="Second")
        temp_manager.create(name="Third")

        response = temp_manager.list()

        # Newest should be first
        assert response[0].name == "Third"


class TestCollectionUpdate:
    """Tests for collection updates."""

    def test_update_name(self, temp_manager):
        """Test updating collection name."""
        result = temp_manager.create(name="Original")
        collection_id = result.data.id

        updated = temp_manager.update(collection_id, name="Updated")

        assert updated.name == "Updated"
        assert updated.id == collection_id

    def test_update_description(self, temp_manager):
        """Test updating description."""
        result = temp_manager.create(name="Test", description="Old")
        updated = temp_manager.update(result.data.id, description="New")

        assert updated.description == "New"
        assert updated.name == "Test"  # Unchanged

    def test_update_metadata(self, temp_manager):
        """Test updating metadata."""
        result = temp_manager.create(name="Test", metadata={"key": "old"})
        updated = temp_manager.update(
            result.data.id,
            metadata={"key": "new", "extra": "value"}
        )

        assert updated.metadata["key"] == "new"
        assert updated.metadata["extra"] == "value"

    def test_update_nonexistent(self, temp_manager):
        """Test that updating non-existent raises NotFoundError."""
        with pytest.raises(NotFoundError):
            temp_manager.update("nonexistent", name="New")

    def test_update_duplicate_name(self, manager_with_collections):
        """Test that updating to duplicate name fails."""
        result = manager_with_collections.create(name="Unique")

        with pytest.raises(DuplicateError):
            manager_with_collections.update(result.data.id, name="Collection A")

    def test_update_same_name_ok(self, temp_manager):
        """Test that updating with same name is OK."""
        result = temp_manager.create(name="Keep This")
        updated = temp_manager.update(result.data.id, name="Keep This")

        assert updated.name == "Keep This"

    def test_update_changes_updated_at(self, temp_manager):
        """Test that update changes updated_at timestamp."""
        result = temp_manager.create(name="Test")
        original_updated = result.data.updated_at

        import time
        time.sleep(0.01)  # Small delay

        updated = temp_manager.update(result.data.id, description="Changed")

        assert updated.updated_at > original_updated


class TestCollectionDelete:
    """Tests for collection deletion."""

    def test_delete_existing(self, temp_manager):
        """Test deleting an existing collection."""
        result = temp_manager.create(name="To Delete")
        collection_id = result.data.id

        response = temp_manager.delete(collection_id)

        assert response.deleted
        assert response.id == collection_id
        assert response.object == "collection"

    def test_delete_removes_from_storage(self, temp_manager):
        """Test that delete removes from storage."""
        result = temp_manager.create(name="To Delete")
        collection_id = result.data.id

        temp_manager.delete(collection_id)

        assert not temp_manager.exists(collection_id)

    def test_delete_nonexistent(self, temp_manager):
        """Test that deleting non-existent raises NotFoundError."""
        with pytest.raises(NotFoundError):
            temp_manager.delete("nonexistent")

    def test_delete_idempotent_after_deletion(self, temp_manager):
        """Test that deleting already deleted raises NotFoundError."""
        result = temp_manager.create(name="To Delete")
        temp_manager.delete(result.data.id)

        with pytest.raises(NotFoundError):
            temp_manager.delete(result.data.id)

    def test_delete_with_documents_requires_force(self, temp_manager):
        """Test that deleting collection with documents needs force=True."""
        result = temp_manager.create(name="Has Docs")

        # Manually add a document to storage
        temp_manager.storage.insert(DOCUMENTS_FILE, {
            "id": "doc-1",
            "collection_id": result.data.id,
            "filename": "test.pdf",
            "file_hash": "abc",
            "file_size": 1000
        })

        with pytest.raises(ValidationError) as exc_info:
            temp_manager.delete(result.data.id)

        assert "force" in exc_info.value.message.lower()

    def test_delete_with_force(self, temp_manager):
        """Test force delete removes documents too."""
        result = temp_manager.create(name="Has Docs")

        # Add documents
        for i in range(3):
            temp_manager.storage.insert(DOCUMENTS_FILE, {
                "id": f"doc-{i}",
                "collection_id": result.data.id,
                "filename": f"test{i}.pdf",
                "file_hash": f"hash{i}",
                "file_size": 1000
            })

        response = temp_manager.delete(result.data.id, force=True)

        assert response.deleted

        # Verify documents are gone
        remaining_docs = temp_manager.storage.find_by_field(
            DOCUMENTS_FILE, "collection_id", result.data.id
        )
        assert len(remaining_docs) == 0


class TestCollectionUtilities:
    """Tests for utility methods."""

    def test_exists_true(self, temp_manager):
        """Test exists returns True for existing collection."""
        result = temp_manager.create(name="Exists")
        assert temp_manager.exists(result.data.id)

    def test_exists_false(self, temp_manager):
        """Test exists returns False for non-existing."""
        assert not temp_manager.exists("nonexistent")

    def test_count_empty(self, temp_manager):
        """Test count when no collections."""
        assert temp_manager.count() == 0

    def test_count_with_collections(self, manager_with_collections):
        """Test count with collections."""
        assert manager_with_collections.count() >= 2
