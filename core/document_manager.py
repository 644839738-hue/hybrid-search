"""
Document Manager API.

Stripe-like API for managing documents within collections. Handles file
processing, deduplication via hash, and cascade deletion of chunks.
"""

import hashlib
import logging
from typing import Optional, List, Dict, Any, Callable
from datetime import datetime

from core.models.document import Document, DocumentStatus
from core.models.errors import (
    ValidationError,
    NotFoundError,
    DuplicateError,
    LimitExceededError,
)
from core.models.responses import ListResponse, DeletedResponse, OperationResult
from core.storage import JSONStorage, COLLECTIONS_FILE, DOCUMENTS_FILE

logger = logging.getLogger(__name__)

# Type alias for BM25 cache invalidation callback
# Callback receives collection_id to invalidate the BM25 index for that collection
BM25InvalidationCallback = Callable[[str], None]


class DocumentManager:
    """
    Stripe-like API for document management.

    Provides CRUD operations for documents with:
    - File hash deduplication
    - Collection validation
    - Soft limits per collection
    - Cascade deletion of chunks

    Attributes:
        SOFT_LIMIT_DOCUMENTS: Maximum recommended documents per collection

    Example:
        >>> manager = DocumentManager()
        >>> doc = manager.add(collection_id="abc123", file=uploaded_file)
        >>> docs = manager.list(collection_id="abc123")
        >>> manager.delete(doc.id)
    """

    SOFT_LIMIT_DOCUMENTS = 5

    def __init__(
        self,
        storage: Optional[JSONStorage] = None,
        data_dir: str = "./data",
        vector_store=None,
        on_document_change: Optional[BM25InvalidationCallback] = None,
    ):
        """
        Initialize the document manager.

        Args:
            storage: Optional JSONStorage instance
            data_dir: Directory for JSON storage
            vector_store: Optional VectorStoreManager for chunk operations
            on_document_change: Optional callback for BM25 cache invalidation.
                Called with collection_id whenever a document is added or deleted.
                This allows the hybrid retriever to invalidate stale BM25 indexes.

        Example:
            >>> def invalidate_bm25(collection_id: str):
            ...     hybrid_retriever.clear_bm25_cache(collection_id)
            >>> manager = DocumentManager(on_document_change=invalidate_bm25)
        """
        self.storage = storage or JSONStorage(data_dir=data_dir)
        self.vector_store = vector_store
        self._on_document_change = on_document_change
        logger.info("DocumentManager initialized")

    def add(
        self,
        collection_id: str,
        filename: str,
        file_content: bytes,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> OperationResult:
        """
        Add a document to a collection.

        Calculates file hash for deduplication and creates document record.
        Note: Actual PDF processing and chunking should be done separately.

        Args:
            collection_id: UUID of the target collection
            filename: Original filename
            file_content: Raw file bytes
            metadata: Optional custom metadata

        Returns:
            OperationResult with created Document and any warnings

        Raises:
            NotFoundError: If collection doesn't exist
            DuplicateError: If file hash already exists in collection
            ValidationError: If file is invalid

        Example:
            >>> with open("paper.pdf", "rb") as f:
            ...     result = manager.add(
            ...         collection_id="abc123",
            ...         filename="paper.pdf",
            ...         file_content=f.read()
            ...     )
        """
        import time
        start_time = time.time()
        logger.info(f"[DM.add] START - filename: {filename}, size: {len(file_content)} bytes, collection: {collection_id}")

        warnings = []

        # Validate filename
        logger.debug(f"[DM.add] Validating filename...")
        if not filename or not filename.strip():
            raise ValidationError(
                message="Filename cannot be empty",
                param="filename"
            )

        filename = filename.strip()

        # Validate file extension
        if not filename.lower().endswith('.pdf'):
            raise ValidationError(
                message="Only PDF files are supported",
                param="filename"
            )
        logger.debug(f"[DM.add] Filename validation passed")

        # Validate collection exists
        logger.debug(f"[DM.add] Checking collection exists...")
        collection = self.storage.get_by_id(COLLECTIONS_FILE, collection_id)
        if not collection:
            raise NotFoundError(
                message=f"Collection '{collection_id}' not found",
                param="collection_id",
                resource_type="collection",
                resource_id=collection_id
            )
        logger.debug(f"[DM.add] Collection found: {collection.get('name')}")

        # Calculate file hash
        logger.debug(f"[DM.add] Calculating file hash...")
        hash_start = time.time()
        file_hash = self._calculate_hash(file_content)
        hash_elapsed = time.time() - hash_start
        logger.debug(f"[DM.add] Hash calculated in {hash_elapsed:.3f}s: {file_hash[:16]}...")

        # Check for duplicate in collection
        logger.debug(f"[DM.add] Checking for duplicates...")
        dup_start = time.time()
        existing = self._find_duplicate(collection_id, file_hash)
        dup_elapsed = time.time() - dup_start
        logger.debug(f"[DM.add] Duplicate check in {dup_elapsed:.3f}s")
        if existing:
            raise DuplicateError(
                message=f"Document with same content already exists: {existing['filename']}",
                param="file_content",
                existing_id=existing["id"]
            )

        # Check soft limit
        logger.debug(f"[DM.add] Checking soft limit...")
        current_count = self.storage.count_by_field(
            DOCUMENTS_FILE, "collection_id", collection_id
        )
        if current_count >= self.SOFT_LIMIT_DOCUMENTS:
            warnings.append(
                f"Document limit ({self.SOFT_LIMIT_DOCUMENTS}) per collection exceeded. "
                f"Current: {current_count + 1}. Consider creating a new collection."
            )
            logger.warning(
                f"Soft limit exceeded for collection {collection_id}: "
                f"{current_count + 1} documents"
            )

        # Create document record
        logger.debug(f"[DM.add] Creating document record...")
        document = Document.create(
            collection_id=collection_id,
            filename=filename,
            file_hash=file_hash,
            file_size=len(file_content),
            metadata=metadata or {},
        )
        logger.debug(f"[DM.add] Document created with ID: {document.id}")

        # Save to storage
        logger.debug(f"[DM.add] Saving to storage...")
        storage_start = time.time()
        self.storage.insert(DOCUMENTS_FILE, document.to_dict())
        storage_elapsed = time.time() - storage_start
        logger.debug(f"[DM.add] Storage insert in {storage_elapsed:.3f}s")

        total_elapsed = time.time() - start_time
        logger.info(f"[DM.add] COMPLETE - document {document.id} ({filename}) in {total_elapsed:.3f}s")

        # Invoke BM25 cache invalidation callback
        # This ensures the BM25 index is rebuilt on next search to include this document
        self._notify_document_change(collection_id)

        return OperationResult(
            success=True,
            data=document,
            warnings=warnings,
            message=f"Document '{filename}' added successfully"
        )

    def get(self, document_id: str) -> Document:
        """
        Retrieve a document by ID.

        Args:
            document_id: UUID of the document

        Returns:
            Document object

        Raises:
            NotFoundError: If document doesn't exist

        Example:
            >>> doc = manager.get("xyz789")
            >>> print(f"File: {doc.filename}, Status: {doc.status}")
        """
        data = self.storage.get_by_id(DOCUMENTS_FILE, document_id)
        if not data:
            raise NotFoundError(
                message=f"Document '{document_id}' not found",
                param="document_id",
                resource_type="document",
                resource_id=document_id
            )

        return Document.from_dict(data)

    def get_by_hash(
        self,
        collection_id: str,
        file_hash: str
    ) -> Optional[Document]:
        """
        Find a document by its file hash within a collection.

        Args:
            collection_id: Collection to search in
            file_hash: SHA256 hash of file content

        Returns:
            Document if found, None otherwise
        """
        data = self._find_duplicate(collection_id, file_hash)
        if data:
            return Document.from_dict(data)
        return None

    def list(
        self,
        collection_id: str,
        limit: int = 10,
        starting_after: Optional[str] = None,
        status: Optional[DocumentStatus] = None,
    ) -> ListResponse[Document]:
        """
        List documents in a collection.

        Args:
            collection_id: UUID of the collection
            limit: Maximum documents to return
            starting_after: Cursor for pagination (document ID)
            status: Filter by document status

        Returns:
            ListResponse with documents and pagination info

        Example:
            >>> response = manager.list(collection_id="abc123")
            >>> for doc in response:
            ...     print(f"{doc.filename}: {doc.status}")
        """
        # Get all documents for collection
        all_data = self.storage.find_by_field(
            DOCUMENTS_FILE, "collection_id", collection_id
        )

        # Filter by status if specified
        if status:
            all_data = [d for d in all_data if d.get("status") == status.value]

        # Sort by uploaded_at descending (newest first)
        all_data.sort(key=lambda x: x.get("uploaded_at", ""), reverse=True)

        # Handle pagination
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

        # Convert to Document objects
        documents = [Document.from_dict(data) for data in page_data]

        # Set next cursor
        next_cursor = None
        if has_more and documents:
            next_cursor = documents[-1].id

        return ListResponse(
            data=documents,
            has_more=has_more,
            total_count=len(all_data),
            next_cursor=next_cursor,
        )

    def update_status(
        self,
        document_id: str,
        status: DocumentStatus,
        page_count: int = 0,
        chunk_count: int = 0,
        error_message: Optional[str] = None,
    ) -> Document:
        """
        Update document processing status.

        Args:
            document_id: UUID of the document
            status: New status
            page_count: Number of pages (for READY status)
            chunk_count: Number of chunks (for READY status)
            error_message: Error details (for FAILED status)

        Returns:
            Updated Document

        Raises:
            NotFoundError: If document doesn't exist
        """
        document = self.get(document_id)

        if status == DocumentStatus.READY:
            updated = document.mark_ready(page_count, chunk_count)
        elif status == DocumentStatus.FAILED:
            updated = document.mark_failed(error_message or "Processing failed")
        else:
            # Just update status
            updated = Document(
                id=document.id,
                collection_id=document.collection_id,
                filename=document.filename,
                file_hash=document.file_hash,
                file_size=document.file_size,
                page_count=page_count or document.page_count,
                chunk_count=chunk_count or document.chunk_count,
                metadata=document.metadata,
                uploaded_at=document.uploaded_at,
                status=status,
                error_message=error_message,
            )

        self.storage.replace(DOCUMENTS_FILE, document_id, updated.to_dict())
        logger.info(f"Updated document {document_id} status to {status.value}")

        return updated

    def delete(self, document_id: str) -> DeletedResponse:
        """
        Delete a document and its chunks.

        Removes document from storage and deletes all associated chunks
        from the vector store (if available).

        Args:
            document_id: UUID of the document

        Returns:
            DeletedResponse confirming deletion

        Raises:
            NotFoundError: If document doesn't exist

        Example:
            >>> response = manager.delete("xyz789")
            >>> print(f"Deleted: {response.deleted}")
        """
        document = self.get(document_id)
        collection_id = document.collection_id

        # Delete chunks from vector store if available
        if self.vector_store:
            try:
                self._delete_chunks(document)
            except Exception as e:
                logger.error(f"Failed to delete chunks for document {document_id}: {e}")

        # Delete document record
        self.storage.delete(DOCUMENTS_FILE, document_id)
        logger.info(f"Deleted document: {document_id} ({document.filename})")

        # Invoke BM25 cache invalidation callback
        # This ensures deleted document is excluded from next BM25 search
        self._notify_document_change(collection_id)

        return DeletedResponse(id=document_id, object="document")

    def delete_by_collection(self, collection_id: str) -> int:
        """
        Delete all documents in a collection.

        Args:
            collection_id: UUID of the collection

        Returns:
            Number of documents deleted
        """
        # Get all documents first (to delete chunks)
        documents = self.storage.find_by_field(
            DOCUMENTS_FILE, "collection_id", collection_id
        )

        # Delete chunks from vector store
        if self.vector_store:
            for doc_data in documents:
                try:
                    doc = Document.from_dict(doc_data)
                    self._delete_chunks(doc)
                except Exception as e:
                    logger.error(f"Failed to delete chunks for document {doc_data.get('id')}: {e}")

        # Delete document records
        deleted_count = self.storage.delete_by_field(
            DOCUMENTS_FILE, "collection_id", collection_id
        )

        logger.info(f"Deleted {deleted_count} documents from collection {collection_id}")
        return deleted_count

    def exists(self, document_id: str) -> bool:
        """
        Check if a document exists.

        Args:
            document_id: UUID to check

        Returns:
            True if document exists
        """
        return self.storage.exists(DOCUMENTS_FILE, document_id)

    def count(self, collection_id: Optional[str] = None) -> int:
        """
        Count documents.

        Args:
            collection_id: Optional filter by collection

        Returns:
            Number of documents
        """
        if collection_id:
            return self.storage.count_by_field(
                DOCUMENTS_FILE, "collection_id", collection_id
            )
        return self.storage.count(DOCUMENTS_FILE)

    def _calculate_hash(self, content: bytes) -> str:
        """
        Calculate SHA256 hash of file content.

        Args:
            content: File bytes

        Returns:
            Hex-encoded hash string
        """
        return hashlib.sha256(content).hexdigest()

    def _find_duplicate(
        self,
        collection_id: str,
        file_hash: str
    ) -> Optional[Dict[str, Any]]:
        """
        Find a document with the same hash in the collection.

        Args:
            collection_id: Collection to search
            file_hash: Hash to find

        Returns:
            Document dict if found, None otherwise
        """
        documents = self.storage.find_by_field(
            DOCUMENTS_FILE, "collection_id", collection_id
        )

        for doc in documents:
            if doc.get("file_hash") == file_hash:
                return doc

        return None

    def _delete_chunks(self, document: Document) -> int:
        """
        Delete all chunks for a document from the vector store.

        Args:
            document: Document whose chunks to delete

        Returns:
            Number of chunks deleted
        """
        if not self.vector_store:
            return 0

        try:
            deleted = self.vector_store.delete_by_document_id(document.id)
            logger.info(f"Deleted {deleted} chunks for document {document.id}")
            return deleted
        except Exception as e:
            logger.error(f"Error deleting chunks for document {document.id}: {e}")
            return 0

    def _notify_document_change(self, collection_id: str) -> None:
        """
        Notify that a document has been added or deleted.

        This triggers BM25 cache invalidation so the index is rebuilt
        on the next search to reflect the current document state.

        Args:
            collection_id: ID of the collection that was modified

        Note:
            This is a no-op if no callback was registered during initialization.
            The callback is expected to handle its own error handling.
        """
        if self._on_document_change is None:
            return

        try:
            logger.debug(f"Invoking BM25 cache invalidation for collection {collection_id}")
            self._on_document_change(collection_id)
            logger.info(f"BM25 cache invalidated for collection {collection_id}")
        except Exception as e:
            # Log but don't fail - cache invalidation is not critical to the operation
            logger.error(f"BM25 cache invalidation failed for collection {collection_id}: {e}")
