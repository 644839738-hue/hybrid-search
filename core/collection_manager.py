"""
Collection Manager API.

Stripe-like API for managing document collections. Provides CRUD operations
with soft limits, validation, and integration with ChromaDB for vector storage.
"""

import logging
from typing import Optional, List, Dict, Any
from datetime import datetime

from core.models.collection import Collection, CollectionSettings
from core.models.errors import (
    ValidationError,
    NotFoundError,
    DuplicateError,
    LimitExceededError,
)
from core.models.responses import ListResponse, DeletedResponse, OperationResult
from core.storage import JSONStorage, COLLECTIONS_FILE, DOCUMENTS_FILE

logger = logging.getLogger(__name__)


class CollectionManager:
    """
    Stripe-like API for collection management.

    Provides CRUD operations for collections with:
    - Validation and error handling
    - Soft limits with warnings
    - ChromaDB integration (optional)
    - Pagination support

    Attributes:
        SOFT_LIMIT_COLLECTIONS: Maximum recommended collections (warning only)

    Example:
        >>> manager = CollectionManager()
        >>> collection = manager.create(name="Research Papers")
        >>> all_collections = manager.list()
        >>> manager.delete(collection.id)
    """

    SOFT_LIMIT_COLLECTIONS = 3

    def __init__(
        self,
        storage: Optional[JSONStorage] = None,
        data_dir: str = "./data",
        chroma_client=None,
        vector_store=None,
    ):
        """
        Initialize the collection manager.

        Args:
            storage: Optional JSONStorage instance (created if not provided)
            data_dir: Directory for JSON storage
            chroma_client: Optional ChromaDB client for vector store integration (deprecated)
            vector_store: Optional EnhancedVectorStore for chunk management (preferred)

        Note:
            The vector_store parameter is preferred over chroma_client for proper
            chunk cleanup during collection deletion. If only chroma_client is provided,
            chunk deletion will use metadata filters directly on the underlying collection.
        """
        self.storage = storage or JSONStorage(data_dir=data_dir)
        self.chroma_client = chroma_client
        self.vector_store = vector_store
        logger.info("CollectionManager initialized")

    def create(
        self,
        name: str,
        description: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        settings: Optional[CollectionSettings] = None,
    ) -> OperationResult:
        """
        Create a new collection.

        Args:
            name: Collection name (required, must be unique)
            description: Optional description
            metadata: Optional custom metadata
            settings: Optional collection settings

        Returns:
            OperationResult with created Collection and any warnings

        Raises:
            ValidationError: If name is empty or invalid
            DuplicateError: If name already exists

        Example:
            >>> result = manager.create(
            ...     name="ML Papers",
            ...     description="Machine learning research"
            ... )
            >>> if result.has_warnings:
            ...     print(result.warnings)
            >>> collection = result.data
        """
        warnings = []

        # Validate name
        if not name or not name.strip():
            raise ValidationError(
                message="Collection name cannot be empty",
                param="name"
            )

        name = name.strip()

        # Check for duplicate name
        existing = self.storage.get_by_field(COLLECTIONS_FILE, "name", name)
        if existing:
            raise DuplicateError(
                message=f"Collection with name '{name}' already exists",
                param="name",
                existing_id=existing["id"]
            )

        # Check soft limit
        current_count = self.storage.count(COLLECTIONS_FILE)
        if current_count >= self.SOFT_LIMIT_COLLECTIONS:
            warnings.append(
                f"Collection limit ({self.SOFT_LIMIT_COLLECTIONS}) exceeded. "
                f"Current: {current_count + 1}. Consider deleting unused collections."
            )
            logger.warning(f"Soft limit exceeded: {current_count + 1} collections")

        # Create collection
        collection = Collection.create(
            name=name,
            description=description,
            metadata=metadata or {},
            settings=settings or CollectionSettings(),
        )

        # Save to storage
        self.storage.insert(COLLECTIONS_FILE, collection.to_dict())
        logger.info(f"Created collection: {collection.id} ({name})")

        # Create ChromaDB collection if client available
        if self.chroma_client:
            try:
                self._create_chroma_collection(collection)
            except Exception as e:
                logger.error(f"Failed to create ChromaDB collection: {e}")
                warnings.append(f"ChromaDB collection creation failed: {str(e)}")

        return OperationResult(
            success=True,
            data=collection,
            warnings=warnings,
            message=f"Collection '{name}' created successfully"
        )

    def get(
        self,
        collection_id: str,
        expand: Optional[List[str]] = None,
    ) -> Collection:
        """
        Retrieve a collection by ID.

        Args:
            collection_id: UUID of the collection
            expand: Optional list of fields to expand (e.g., ["documents", "stats"])

        Returns:
            Collection object with computed fields populated

        Raises:
            NotFoundError: If collection doesn't exist

        Example:
            >>> collection = manager.get("abc123", expand=["stats"])
            >>> print(f"Documents: {collection.document_count}")
        """
        expand = expand or []

        data = self.storage.get_by_id(COLLECTIONS_FILE, collection_id)
        if not data:
            raise NotFoundError(
                message=f"Collection '{collection_id}' not found",
                param="collection_id",
                resource_type="collection",
                resource_id=collection_id
            )

        collection = Collection.from_dict(data)

        # Compute stats if requested or by default
        if "stats" in expand or not expand:
            collection = self._add_stats(collection)

        return collection

    def get_by_name(self, name: str) -> Optional[Collection]:
        """
        Retrieve a collection by name.

        Args:
            name: Collection name

        Returns:
            Collection if found, None otherwise
        """
        data = self.storage.get_by_field(COLLECTIONS_FILE, "name", name)
        if not data:
            return None

        collection = Collection.from_dict(data)
        return self._add_stats(collection)

    def list(
        self,
        limit: int = 10,
        starting_after: Optional[str] = None,
        include_stats: bool = True,
    ) -> ListResponse[Collection]:
        """
        List all collections with pagination.

        Args:
            limit: Maximum collections to return (default: 10)
            starting_after: Cursor for pagination (collection ID)
            include_stats: Whether to include document/chunk counts

        Returns:
            ListResponse with collections and pagination info

        Example:
            >>> response = manager.list(limit=5)
            >>> for collection in response:
            ...     print(collection.name)
            >>> if response.has_more:
            ...     next_page = manager.list(starting_after=response.next_cursor)
        """
        all_data = self.storage.load(COLLECTIONS_FILE)

        # Sort by created_at descending (newest first)
        all_data.sort(key=lambda x: x.get("created_at", ""), reverse=True)

        # Handle pagination cursor
        start_index = 0
        if starting_after:
            for i, item in enumerate(all_data):
                if item.get("id") == starting_after:
                    start_index = i + 1
                    break

        # Slice for pagination
        end_index = start_index + limit
        page_data = all_data[start_index:end_index]
        has_more = end_index < len(all_data)

        # Convert to Collection objects
        collections = []
        for data in page_data:
            collection = Collection.from_dict(data)
            if include_stats:
                collection = self._add_stats(collection)
            collections.append(collection)

        # Set next cursor
        next_cursor = None
        if has_more and collections:
            next_cursor = collections[-1].id

        return ListResponse(
            data=collections,
            has_more=has_more,
            total_count=len(all_data),
            next_cursor=next_cursor,
        )

    def update(
        self,
        collection_id: str,
        name: Optional[str] = None,
        description: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        settings: Optional[CollectionSettings] = None,
    ) -> Collection:
        """
        Update collection properties.

        Only provided fields are updated. Omitted fields remain unchanged.

        Args:
            collection_id: UUID of the collection
            name: New name (must be unique if changed)
            description: New description
            metadata: New metadata (replaces existing)
            settings: New settings

        Returns:
            Updated Collection object

        Raises:
            NotFoundError: If collection doesn't exist
            DuplicateError: If new name already exists

        Example:
            >>> updated = manager.update(
            ...     "abc123",
            ...     name="New Name",
            ...     description="Updated description"
            ... )
        """
        # Get existing collection
        collection = self.get(collection_id)

        # Validate new name if provided
        if name and name.strip() != collection.name:
            name = name.strip()
            existing = self.storage.get_by_field(COLLECTIONS_FILE, "name", name)
            if existing and existing["id"] != collection_id:
                raise DuplicateError(
                    message=f"Collection with name '{name}' already exists",
                    param="name",
                    existing_id=existing["id"]
                )

        # Create updated collection
        updated = collection.update(
            name=name,
            description=description,
            metadata=metadata,
            settings=settings,
        )

        # Save to storage
        self.storage.replace(COLLECTIONS_FILE, collection_id, updated.to_dict())
        logger.info(f"Updated collection: {collection_id}")

        return self._add_stats(updated)

    def delete(
        self,
        collection_id: str,
        force: bool = False,
    ) -> DeletedResponse:
        """
        Delete a collection.

        By default, fails if collection has documents. Use force=True to
        delete collection and all its documents.

        Args:
            collection_id: UUID of the collection
            force: If True, delete even if collection has documents

        Returns:
            DeletedResponse confirming deletion

        Raises:
            NotFoundError: If collection doesn't exist
            ValidationError: If collection has documents and force=False

        Example:
            >>> response = manager.delete("abc123", force=True)
            >>> print(f"Deleted: {response.deleted}")
        """
        # Verify collection exists
        collection = self.get(collection_id)

        # Check for documents
        doc_count = self.storage.count_by_field(
            DOCUMENTS_FILE, "collection_id", collection_id
        )

        if doc_count > 0 and not force:
            raise ValidationError(
                message=f"Collection has {doc_count} documents. Use force=True to delete.",
                param="force",
            )

        # CRITICAL: Delete ChromaDB chunks FIRST before deleting JSON records
        # This ensures no orphan chunks remain if the process is interrupted
        # The old approach tried to delete a named collection that doesn't exist -
        # all chunks are stored in a shared collection with collection_id in metadata
        chunks_deleted = 0
        if self.vector_store or self.chroma_client:
            try:
                chunks_deleted = self._delete_collection_chunks(collection_id)
                logger.info(f"Deleted {chunks_deleted} chunks from ChromaDB for collection {collection_id}")
            except Exception as e:
                logger.error(f"Failed to delete ChromaDB chunks for collection {collection_id}: {e}")
                # Continue with deletion - chunks may be orphaned but collection should still be removed

        # Delete associated documents from JSON storage
        if doc_count > 0:
            deleted_docs = self.storage.delete_by_field(
                DOCUMENTS_FILE, "collection_id", collection_id
            )
            logger.info(f"Deleted {deleted_docs} documents from collection {collection_id}")

        # Delete collection record
        self.storage.delete(COLLECTIONS_FILE, collection_id)
        logger.info(f"Deleted collection: {collection_id}")

        return DeletedResponse(id=collection_id, object="collection")

    def exists(self, collection_id: str) -> bool:
        """
        Check if a collection exists.

        Args:
            collection_id: UUID to check

        Returns:
            True if collection exists
        """
        return self.storage.exists(COLLECTIONS_FILE, collection_id)

    def count(self) -> int:
        """
        Get total number of collections.

        Returns:
            Number of collections
        """
        return self.storage.count(COLLECTIONS_FILE)

    def _add_stats(self, collection: Collection) -> Collection:
        """
        Add computed statistics to a collection.

        Args:
            collection: Collection to add stats to

        Returns:
            Collection with populated document_count and chunk_count
        """
        # Count documents
        documents = self.storage.find_by_field(
            DOCUMENTS_FILE, "collection_id", collection.id
        )

        collection.document_count = len(documents)
        collection.chunk_count = sum(doc.get("chunk_count", 0) for doc in documents)

        return collection

    def _create_chroma_collection(self, collection: Collection) -> None:
        """Create a ChromaDB collection for vector storage."""
        if not self.chroma_client:
            return

        # ChromaDB collection name must be valid
        chroma_name = f"collection_{collection.id.replace('-', '_')}"
        self.chroma_client.get_or_create_collection(
            name=chroma_name,
            metadata={"collection_id": collection.id}
        )
        logger.info(f"Created ChromaDB collection: {chroma_name}")

    def _delete_collection_chunks(self, collection_id: str) -> int:
        """
        Delete all chunks for a collection from the shared vector store.

        All document chunks are stored in a single shared ChromaDB collection
        (semantic_search_docs_streamlit) with collection_id in their metadata - NOT in
        separate per-collection ChromaDB collections.

        Args:
            collection_id: ID of the collection whose chunks should be deleted

        Returns:
            Number of chunks deleted

        Note:
            This method prefers using vector_store.delete_by_collection_id() if
            available, falling back to direct ChromaDB operations if only
            chroma_client is provided.
        """
        # Prefer using EnhancedVectorStore if available (cleaner abstraction)
        if self.vector_store:
            return self.vector_store.delete_by_collection_id(collection_id)

        # Fallback: Direct ChromaDB operations if only chroma_client is available
        if not self.chroma_client:
            logger.warning("No vector store or chroma_client available for chunk deletion")
            return 0

        try:
            # Get the shared collection where all chunks are stored
            collection = self.chroma_client.get_or_create_collection(
                name="semantic_search_docs_streamlit"
            )

            # Find all chunks with this collection_id in metadata
            # ChromaDB requires explicit $eq operator for equality filters
            results = collection.get(
                where={"collection_id": {"$eq": collection_id}},
                include=[]
            )

            chunk_ids = results.get("ids", [])
            if not chunk_ids:
                logger.info(f"No chunks found for collection {collection_id}")
                return 0

            # Delete all matching chunks
            collection.delete(ids=chunk_ids)
            logger.info(f"Deleted {len(chunk_ids)} chunks for collection {collection_id}")
            return len(chunk_ids)

        except Exception as e:
            logger.error(f"Error deleting chunks for collection {collection_id}: {e}")
            raise

    def _delete_chroma_collection(self, collection_id: str) -> None:
        """
        DEPRECATED: Use _delete_collection_chunks instead.

        This method incorrectly tried to delete a named ChromaDB collection
        that doesn't exist. All chunks are stored in a shared collection.
        Kept for backwards compatibility but logs a deprecation warning.
        """
        logger.warning(
            f"_delete_chroma_collection is deprecated - use _delete_collection_chunks. "
            f"Collection {collection_id}"
        )
        self._delete_collection_chunks(collection_id)
