"""
JSON file storage layer with atomic writes.

Provides simple persistent storage for collections and documents
using JSON files. Designed to be replaced with database storage
in Stage 2.
"""

import json
import threading
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, TypeVar, Callable
from datetime import datetime

logger = logging.getLogger(__name__)

T = TypeVar("T")


class JSONStorage:
    """
    Simple JSON file storage with atomic writes and thread safety.

    Provides CRUD operations for JSON-serializable data with
    file locking to prevent corruption from concurrent access.

    Attributes:
        data_dir: Directory for storing JSON files

    Example:
        >>> storage = JSONStorage("./data")
        >>> storage.save("collections.json", [{"id": "1", "name": "Test"}])
        >>> data = storage.load("collections.json")
    """

    def __init__(self, data_dir: str = "./data"):
        """
        Initialize storage with data directory.

        Args:
            data_dir: Path to directory for JSON files.
                     Created if it doesn't exist.
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        logger.info(f"JSONStorage initialized at {self.data_dir.absolute()}")

    def _get_path(self, filename: str) -> Path:
        """Get full path for a filename."""
        return self.data_dir / filename

    def load(self, filename: str) -> List[Dict[str, Any]]:
        """
        Load data from JSON file.

        Args:
            filename: Name of JSON file (e.g., "collections.json")

        Returns:
            List of dictionaries, or empty list if file doesn't exist
        """
        path = self._get_path(filename)
        if not path.exists():
            logger.debug(f"File {filename} not found, returning empty list")
            return []

        try:
            with self._lock:
                with open(path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    logger.debug(f"Loaded {len(data)} items from {filename}")
                    return data
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse {filename}: {e}")
            return []
        except Exception as e:
            logger.error(f"Failed to load {filename}: {e}")
            return []

    def save(self, filename: str, data: List[Dict[str, Any]]) -> bool:
        """
        Save data to JSON file with atomic write.

        Uses a temporary file and rename for atomic writes,
        preventing corruption if write is interrupted.

        Args:
            filename: Name of JSON file
            data: List of dictionaries to save

        Returns:
            True if save succeeded, False otherwise
        """
        path = self._get_path(filename)
        temp_path = path.with_suffix('.tmp')

        try:
            with self._lock:
                with open(temp_path, 'w', encoding='utf-8') as f:
                    json.dump(data, f, indent=2, default=self._json_serializer)
                temp_path.replace(path)
                logger.debug(f"Saved {len(data)} items to {filename}")
                return True
        except Exception as e:
            logger.error(f"Failed to save {filename}: {e}")
            if temp_path.exists():
                temp_path.unlink()
            return False

    def _json_serializer(self, obj: Any) -> Any:
        """Custom JSON serializer for complex types."""
        if isinstance(obj, datetime):
            return obj.isoformat()
        raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

    # CRUD Operations

    def get_by_id(
        self,
        filename: str,
        item_id: str,
        id_field: str = "id"
    ) -> Optional[Dict[str, Any]]:
        """
        Get a single item by ID.

        Args:
            filename: JSON file to search
            item_id: ID to find
            id_field: Name of ID field (default: "id")

        Returns:
            Item dict if found, None otherwise
        """
        data = self.load(filename)
        for item in data:
            if item.get(id_field) == item_id:
                return item
        return None

    def get_by_field(
        self,
        filename: str,
        field: str,
        value: Any
    ) -> Optional[Dict[str, Any]]:
        """
        Get first item matching a field value.

        Args:
            filename: JSON file to search
            field: Field name to match
            value: Value to match

        Returns:
            First matching item or None
        """
        data = self.load(filename)
        for item in data:
            if item.get(field) == value:
                return item
        return None

    def find_by_field(
        self,
        filename: str,
        field: str,
        value: Any
    ) -> List[Dict[str, Any]]:
        """
        Find all items matching a field value.

        Args:
            filename: JSON file to search
            field: Field name to match
            value: Value to match

        Returns:
            List of matching items
        """
        data = self.load(filename)
        return [item for item in data if item.get(field) == value]

    def insert(
        self,
        filename: str,
        item: Dict[str, Any]
    ) -> bool:
        """
        Insert a new item (thread-safe).

        Args:
            filename: JSON file to update
            item: Item to insert

        Returns:
            True if successful
        """
        path = self._get_path(filename)
        temp_path = path.with_suffix('.tmp')

        try:
            with self._lock:
                # Load existing data inside lock
                if path.exists():
                    with open(path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                else:
                    data = []

                # Append new item
                data.append(item)

                # Write atomically
                with open(temp_path, 'w', encoding='utf-8') as f:
                    json.dump(data, f, indent=2, default=self._json_serializer)
                temp_path.replace(path)

                logger.debug(f"Inserted item into {filename}")
                return True
        except Exception as e:
            logger.error(f"Failed to insert into {filename}: {e}")
            if temp_path.exists():
                temp_path.unlink()
            return False

    def update(
        self,
        filename: str,
        item_id: str,
        updates: Dict[str, Any],
        id_field: str = "id"
    ) -> Optional[Dict[str, Any]]:
        """
        Update an existing item.

        Args:
            filename: JSON file to update
            item_id: ID of item to update
            updates: Dictionary of field updates
            id_field: Name of ID field

        Returns:
            Updated item if found, None otherwise
        """
        data = self.load(filename)
        updated_item = None

        for i, item in enumerate(data):
            if item.get(id_field) == item_id:
                data[i] = {**item, **updates}
                updated_item = data[i]
                break

        if updated_item:
            self.save(filename, data)

        return updated_item

    def replace(
        self,
        filename: str,
        item_id: str,
        new_item: Dict[str, Any],
        id_field: str = "id"
    ) -> bool:
        """
        Replace an item entirely.

        Args:
            filename: JSON file to update
            item_id: ID of item to replace
            new_item: New item data
            id_field: Name of ID field

        Returns:
            True if item was found and replaced
        """
        data = self.load(filename)
        found = False

        for i, item in enumerate(data):
            if item.get(id_field) == item_id:
                data[i] = new_item
                found = True
                break

        if found:
            self.save(filename, data)

        return found

    def delete(
        self,
        filename: str,
        item_id: str,
        id_field: str = "id"
    ) -> bool:
        """
        Delete an item by ID.

        Args:
            filename: JSON file to update
            item_id: ID of item to delete
            id_field: Name of ID field

        Returns:
            True if item was found and deleted
        """
        data = self.load(filename)
        original_length = len(data)
        data = [item for item in data if item.get(id_field) != item_id]

        if len(data) < original_length:
            self.save(filename, data)
            return True

        return False

    def delete_by_field(
        self,
        filename: str,
        field: str,
        value: Any
    ) -> int:
        """
        Delete all items matching a field value.

        Args:
            filename: JSON file to update
            field: Field name to match
            value: Value to match

        Returns:
            Number of items deleted
        """
        data = self.load(filename)
        original_length = len(data)
        data = [item for item in data if item.get(field) != value]
        deleted_count = original_length - len(data)

        if deleted_count > 0:
            self.save(filename, data)

        return deleted_count

    def count(self, filename: str) -> int:
        """
        Count items in a file.

        Args:
            filename: JSON file to count

        Returns:
            Number of items
        """
        return len(self.load(filename))

    def count_by_field(
        self,
        filename: str,
        field: str,
        value: Any
    ) -> int:
        """
        Count items matching a field value.

        Args:
            filename: JSON file to search
            field: Field name to match
            value: Value to match

        Returns:
            Number of matching items
        """
        return len(self.find_by_field(filename, field, value))

    def exists(
        self,
        filename: str,
        item_id: str,
        id_field: str = "id"
    ) -> bool:
        """
        Check if an item exists.

        Args:
            filename: JSON file to check
            item_id: ID to find
            id_field: Name of ID field

        Returns:
            True if item exists
        """
        return self.get_by_id(filename, item_id, id_field) is not None

    def clear(self, filename: str) -> bool:
        """
        Remove all items from a file.

        Args:
            filename: JSON file to clear

        Returns:
            True if successful
        """
        return self.save(filename, [])


# File constants
COLLECTIONS_FILE = "collections.json"
DOCUMENTS_FILE = "documents.json"
