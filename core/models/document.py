"""
Document data model.

A Document represents a single PDF file that has been uploaded and
processed into searchable chunks within a collection.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional, Dict, Any
import uuid


class DocumentStatus(str, Enum):
    """
    Processing status of a document.

    Attributes:
        PROCESSING: Document is being processed (chunking, embedding)
        READY: Document is fully indexed and searchable
        FAILED: Processing failed, document not searchable
    """
    PROCESSING = "processing"
    READY = "ready"
    FAILED = "failed"


@dataclass
class Document:
    """
    A document within a collection.

    Documents track metadata about uploaded PDF files, including
    their processing status and statistics.

    Attributes:
        id: Unique identifier (UUID)
        collection_id: Parent collection ID (FK)
        filename: Original filename
        file_hash: SHA256 hash for deduplication
        file_size: File size in bytes
        page_count: Number of pages in PDF
        chunk_count: Number of text chunks created
        metadata: Custom key-value metadata
        uploaded_at: Upload timestamp
        status: Processing status
        error_message: Error details if status is FAILED

    Example:
        >>> doc = Document.create(
        ...     collection_id="abc123",
        ...     filename="paper.pdf",
        ...     file_hash="sha256...",
        ...     file_size=1048576
        ... )
    """
    id: str
    collection_id: str
    filename: str
    file_hash: str
    file_size: int
    page_count: int = 0
    chunk_count: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)
    uploaded_at: datetime = field(default_factory=datetime.utcnow)
    status: DocumentStatus = DocumentStatus.PROCESSING
    error_message: Optional[str] = None

    @classmethod
    def create(
        cls,
        collection_id: str,
        filename: str,
        file_hash: str,
        file_size: int,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> "Document":
        """
        Factory method to create a new document with generated ID.

        Args:
            collection_id: Parent collection ID
            filename: Original filename
            file_hash: SHA256 hash of file contents
            file_size: File size in bytes
            metadata: Optional custom metadata

        Returns:
            New Document instance with generated UUID
        """
        return cls(
            id=str(uuid.uuid4()),
            collection_id=collection_id,
            filename=filename,
            file_hash=file_hash,
            file_size=file_size,
            metadata=metadata or {},
            uploaded_at=datetime.utcnow(),
            status=DocumentStatus.PROCESSING,
        )

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert to dictionary for serialization.

        Returns:
            Dictionary representation
        """
        return {
            "id": self.id,
            "collection_id": self.collection_id,
            "filename": self.filename,
            "file_hash": self.file_hash,
            "file_size": self.file_size,
            "page_count": self.page_count,
            "chunk_count": self.chunk_count,
            "metadata": self.metadata,
            "uploaded_at": self.uploaded_at.isoformat(),
            "status": self.status.value,
            "error_message": self.error_message,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Document":
        """
        Create from dictionary.

        Args:
            data: Dictionary with document data

        Returns:
            Document instance
        """
        uploaded_at = data.get("uploaded_at")
        if isinstance(uploaded_at, str):
            uploaded_at = datetime.fromisoformat(uploaded_at)
        elif uploaded_at is None:
            uploaded_at = datetime.utcnow()

        status = data.get("status", "processing")
        if isinstance(status, str):
            status = DocumentStatus(status)

        return cls(
            id=data["id"],
            collection_id=data["collection_id"],
            filename=data["filename"],
            file_hash=data["file_hash"],
            file_size=data["file_size"],
            page_count=data.get("page_count", 0),
            chunk_count=data.get("chunk_count", 0),
            metadata=data.get("metadata", {}),
            uploaded_at=uploaded_at,
            status=status,
            error_message=data.get("error_message"),
        )

    def mark_ready(self, page_count: int, chunk_count: int) -> "Document":
        """
        Mark document as ready after successful processing.

        Args:
            page_count: Number of pages processed
            chunk_count: Number of chunks created

        Returns:
            Updated Document instance
        """
        return Document(
            id=self.id,
            collection_id=self.collection_id,
            filename=self.filename,
            file_hash=self.file_hash,
            file_size=self.file_size,
            page_count=page_count,
            chunk_count=chunk_count,
            metadata=self.metadata,
            uploaded_at=self.uploaded_at,
            status=DocumentStatus.READY,
            error_message=None,
        )

    def mark_failed(self, error_message: str) -> "Document":
        """
        Mark document as failed after processing error.

        Args:
            error_message: Description of what went wrong

        Returns:
            Updated Document instance
        """
        return Document(
            id=self.id,
            collection_id=self.collection_id,
            filename=self.filename,
            file_hash=self.file_hash,
            file_size=self.file_size,
            page_count=0,
            chunk_count=0,
            metadata=self.metadata,
            uploaded_at=self.uploaded_at,
            status=DocumentStatus.FAILED,
            error_message=error_message,
        )

    @property
    def is_ready(self) -> bool:
        """Check if document is ready for search."""
        return self.status == DocumentStatus.READY

    @property
    def is_failed(self) -> bool:
        """Check if document processing failed."""
        return self.status == DocumentStatus.FAILED

    @property
    def is_processing(self) -> bool:
        """Check if document is still being processed."""
        return self.status == DocumentStatus.PROCESSING

    def format_size(self) -> str:
        """
        Format file size in human-readable format.

        Returns:
            Formatted size string (e.g., "1.5 MB")
        """
        size = self.file_size
        for unit in ["B", "KB", "MB", "GB"]:
            if size < 1024:
                return f"{size:.1f} {unit}"
            size /= 1024
        return f"{size:.1f} TB"
