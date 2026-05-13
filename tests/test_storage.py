"""
Tests for JSON storage layer (M1).

Tests CRUD operations, atomic writes, and thread safety.
"""

import pytest
import tempfile
import threading
import time
from pathlib import Path
from core.storage import JSONStorage, COLLECTIONS_FILE, DOCUMENTS_FILE


@pytest.fixture
def temp_storage():
    """Create a temporary storage directory for tests."""
    with tempfile.TemporaryDirectory() as tmpdir:
        storage = JSONStorage(data_dir=tmpdir)
        yield storage


class TestJSONStorageBasics:
    """Basic storage operations."""

    def test_init_creates_directory(self):
        """Test that init creates data directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            storage_path = Path(tmpdir) / "new_data_dir"
            storage = JSONStorage(data_dir=str(storage_path))
            assert storage_path.exists()

    def test_load_empty_file(self, temp_storage):
        """Test loading from non-existent file returns empty list."""
        data = temp_storage.load("nonexistent.json")
        assert data == []

    def test_save_and_load(self, temp_storage):
        """Test basic save and load cycle."""
        items = [
            {"id": "1", "name": "First"},
            {"id": "2", "name": "Second"}
        ]
        temp_storage.save("test.json", items)
        loaded = temp_storage.load("test.json")
        assert loaded == items

    def test_save_with_datetime(self, temp_storage):
        """Test saving items with datetime values."""
        from datetime import datetime
        items = [
            {"id": "1", "created_at": datetime(2024, 1, 15, 10, 30, 0)}
        ]
        temp_storage.save("test.json", items)
        loaded = temp_storage.load("test.json")
        assert loaded[0]["created_at"] == "2024-01-15T10:30:00"


class TestJSONStorageCRUD:
    """CRUD operation tests."""

    def test_insert(self, temp_storage):
        """Test inserting a new item."""
        item = {"id": "new-1", "name": "New Item"}
        temp_storage.insert("test.json", item)
        loaded = temp_storage.load("test.json")
        assert len(loaded) == 1
        assert loaded[0]["id"] == "new-1"

    def test_get_by_id_found(self, temp_storage):
        """Test getting an item by ID when it exists."""
        items = [
            {"id": "1", "name": "First"},
            {"id": "2", "name": "Second"}
        ]
        temp_storage.save("test.json", items)

        result = temp_storage.get_by_id("test.json", "2")
        assert result is not None
        assert result["name"] == "Second"

    def test_get_by_id_not_found(self, temp_storage):
        """Test getting an item by ID when it doesn't exist."""
        items = [{"id": "1", "name": "First"}]
        temp_storage.save("test.json", items)

        result = temp_storage.get_by_id("test.json", "nonexistent")
        assert result is None

    def test_get_by_field(self, temp_storage):
        """Test getting an item by arbitrary field."""
        items = [
            {"id": "1", "status": "active"},
            {"id": "2", "status": "inactive"}
        ]
        temp_storage.save("test.json", items)

        result = temp_storage.get_by_field("test.json", "status", "inactive")
        assert result is not None
        assert result["id"] == "2"

    def test_find_by_field(self, temp_storage):
        """Test finding all items matching a field."""
        items = [
            {"id": "1", "category": "A"},
            {"id": "2", "category": "B"},
            {"id": "3", "category": "A"}
        ]
        temp_storage.save("test.json", items)

        results = temp_storage.find_by_field("test.json", "category", "A")
        assert len(results) == 2
        assert all(r["category"] == "A" for r in results)

    def test_update(self, temp_storage):
        """Test updating an existing item."""
        items = [{"id": "1", "name": "Original", "count": 0}]
        temp_storage.save("test.json", items)

        updated = temp_storage.update("test.json", "1", {"name": "Updated", "count": 5})
        assert updated is not None
        assert updated["name"] == "Updated"
        assert updated["count"] == 5

        # Verify persistence
        loaded = temp_storage.load("test.json")
        assert loaded[0]["name"] == "Updated"

    def test_update_nonexistent(self, temp_storage):
        """Test updating a non-existent item returns None."""
        items = [{"id": "1", "name": "Exists"}]
        temp_storage.save("test.json", items)

        result = temp_storage.update("test.json", "nonexistent", {"name": "New"})
        assert result is None

    def test_replace(self, temp_storage):
        """Test replacing an item entirely."""
        items = [{"id": "1", "name": "Original", "extra": "data"}]
        temp_storage.save("test.json", items)

        new_item = {"id": "1", "name": "Replaced"}
        result = temp_storage.replace("test.json", "1", new_item)
        assert result is True

        loaded = temp_storage.load("test.json")
        assert loaded[0] == new_item
        assert "extra" not in loaded[0]

    def test_delete(self, temp_storage):
        """Test deleting an item."""
        items = [
            {"id": "1", "name": "First"},
            {"id": "2", "name": "Second"}
        ]
        temp_storage.save("test.json", items)

        result = temp_storage.delete("test.json", "1")
        assert result is True

        loaded = temp_storage.load("test.json")
        assert len(loaded) == 1
        assert loaded[0]["id"] == "2"

    def test_delete_nonexistent(self, temp_storage):
        """Test deleting a non-existent item returns False."""
        items = [{"id": "1", "name": "First"}]
        temp_storage.save("test.json", items)

        result = temp_storage.delete("test.json", "nonexistent")
        assert result is False

    def test_delete_by_field(self, temp_storage):
        """Test deleting all items matching a field."""
        items = [
            {"id": "1", "collection_id": "col-A"},
            {"id": "2", "collection_id": "col-B"},
            {"id": "3", "collection_id": "col-A"}
        ]
        temp_storage.save("test.json", items)

        deleted_count = temp_storage.delete_by_field("test.json", "collection_id", "col-A")
        assert deleted_count == 2

        loaded = temp_storage.load("test.json")
        assert len(loaded) == 1
        assert loaded[0]["collection_id"] == "col-B"


class TestJSONStorageUtilities:
    """Utility method tests."""

    def test_count(self, temp_storage):
        """Test counting items."""
        items = [{"id": "1"}, {"id": "2"}, {"id": "3"}]
        temp_storage.save("test.json", items)

        assert temp_storage.count("test.json") == 3

    def test_count_empty(self, temp_storage):
        """Test counting items in non-existent file."""
        assert temp_storage.count("nonexistent.json") == 0

    def test_count_by_field(self, temp_storage):
        """Test counting items matching a field."""
        items = [
            {"id": "1", "status": "active"},
            {"id": "2", "status": "inactive"},
            {"id": "3", "status": "active"}
        ]
        temp_storage.save("test.json", items)

        assert temp_storage.count_by_field("test.json", "status", "active") == 2

    def test_exists_true(self, temp_storage):
        """Test exists returns True for existing items."""
        items = [{"id": "exists-123"}]
        temp_storage.save("test.json", items)

        assert temp_storage.exists("test.json", "exists-123") is True

    def test_exists_false(self, temp_storage):
        """Test exists returns False for non-existing items."""
        items = [{"id": "exists-123"}]
        temp_storage.save("test.json", items)

        assert temp_storage.exists("test.json", "not-exists") is False

    def test_clear(self, temp_storage):
        """Test clearing all items."""
        items = [{"id": "1"}, {"id": "2"}]
        temp_storage.save("test.json", items)

        result = temp_storage.clear("test.json")
        assert result is True

        loaded = temp_storage.load("test.json")
        assert loaded == []


class TestJSONStorageThreadSafety:
    """Thread safety tests."""

    def test_concurrent_writes(self, temp_storage):
        """Test that concurrent writes don't corrupt data."""
        errors = []

        def writer(storage, item_id):
            try:
                for i in range(10):
                    storage.insert("concurrent.json", {"id": f"{item_id}-{i}", "data": "x" * 100})
                    time.sleep(0.001)
            except Exception as e:
                errors.append(e)

        threads = [
            threading.Thread(target=writer, args=(temp_storage, f"thread-{i}"))
            for i in range(5)
        ]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0

        # Verify data integrity
        loaded = temp_storage.load("concurrent.json")
        assert len(loaded) == 50  # 5 threads * 10 items each

    def test_concurrent_read_write(self, temp_storage):
        """Test concurrent reads and writes."""
        # Initialize with some data
        temp_storage.save("test.json", [{"id": "initial", "count": 0}])

        read_results = []
        errors = []

        def reader(storage):
            try:
                for _ in range(20):
                    data = storage.load("test.json")
                    read_results.append(len(data))
                    time.sleep(0.001)
            except Exception as e:
                errors.append(e)

        def writer(storage):
            try:
                for i in range(10):
                    storage.insert("test.json", {"id": f"new-{i}"})
                    time.sleep(0.002)
            except Exception as e:
                errors.append(e)

        reader_thread = threading.Thread(target=reader, args=(temp_storage,))
        writer_thread = threading.Thread(target=writer, args=(temp_storage,))

        reader_thread.start()
        writer_thread.start()

        reader_thread.join()
        writer_thread.join()

        assert len(errors) == 0
        # All reads should return valid data
        assert all(r >= 1 for r in read_results)


class TestFileConstants:
    """Test file name constants."""

    def test_collections_file_constant(self):
        """Test COLLECTIONS_FILE constant."""
        assert COLLECTIONS_FILE == "collections.json"

    def test_documents_file_constant(self):
        """Test DOCUMENTS_FILE constant."""
        assert DOCUMENTS_FILE == "documents.json"
