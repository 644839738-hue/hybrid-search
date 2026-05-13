"""
Collection data model.

A Collection represents a logical grouping of documents that can be
searched together. Each collection has its own settings for chunking
and embedding.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, Dict, Any
import uuid


@dataclass
class CollectionSettings:
    """
    Configuration settings for a collection.

    These settings control how documents in the collection are processed.

    Attributes:
        chunk_size: Number of characters per text chunk
        chunk_overlap: Overlap between consecutive chunks
        embedding_model: OpenAI embedding model to use

    Example:
        >>> settings = CollectionSettings(chunk_size=500, chunk_overlap=100)
    """
    chunk_size: int = 1000
    chunk_overlap: int = 200
    embedding_model: str = "text-embedding-3-large"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
            "embedding_model": self.embedding_model,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CollectionSettings":
        """Create from dictionary."""
        return cls(
            chunk_size=data.get("chunk_size", 1000),
            chunk_overlap=data.get("chunk_overlap", 200),
            embedding_model=data.get("embedding_model", "text-embedding-3-large"),
        )


@dataclass
class Collection:
    """
    A collection of documents for semantic search.

    Collections provide logical separation of documents, allowing users
    to organize their content and scope searches appropriately.

    Attributes:
        id: Unique identifier (UUID)
        name: User-facing name (must be unique)
        description: Optional description
        metadata: Custom key-value metadata
        created_at: Creation timestamp
        updated_at: Last update timestamp
        settings: Collection-specific settings
        document_count: Number of documents (computed, not stored)
        chunk_count: Total chunks across all documents (computed, not stored)

    Example:
        >>> collection = Collection.create(
        ...     name="Research Papers",
        ...     description="Academic papers on ML"
        ... )
    """
    id: str
    name: str
    description: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    settings: CollectionSettings = field(default_factory=CollectionSettings)

    # Computed fields (not stored, populated on read)
    document_count: int = 0
    chunk_count: int = 0

    @classmethod
    def create(
        cls,
        name: str,
        description: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        settings: Optional[CollectionSettings] = None,
    ) -> "Collection":
        """
        Factory method to create a new collection with generated ID.

        Args:
            name: Collection name (required)
            description: Optional description
            metadata: Optional custom metadata
            settings: Optional collection settings

        Returns:
            New Collection instance with generated UUID
        """
        now = datetime.utcnow()
        return cls(
            id=str(uuid.uuid4()),
            name=name,
            description=description,
            metadata=metadata or {},
            created_at=now,
            updated_at=now,
            settings=settings or CollectionSettings(),
        )

    def to_dict(self, include_computed: bool = False) -> Dict[str, Any]:
        """
        Convert to dictionary for serialization.

        Args:
            include_computed: Whether to include computed fields

        Returns:
            Dictionary representation
        """
        data = {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "settings": self.settings.to_dict(),
        }
        if include_computed:
            data["document_count"] = self.document_count
            data["chunk_count"] = self.chunk_count
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Collection":
        """
        Create from dictionary.

        Args:
            data: Dictionary with collection data

        Returns:
            Collection instance
        """
        settings_data = data.get("settings", {})
        settings = CollectionSettings.from_dict(settings_data) if settings_data else CollectionSettings()

        created_at = data.get("created_at")
        if isinstance(created_at, str):
            created_at = datetime.fromisoformat(created_at)
        elif created_at is None:
            created_at = datetime.utcnow()

        updated_at = data.get("updated_at")
        if isinstance(updated_at, str):
            updated_at = datetime.fromisoformat(updated_at)
        elif updated_at is None:
            updated_at = datetime.utcnow()

        return cls(
            id=data["id"],
            name=data["name"],
            description=data.get("description"),
            metadata=data.get("metadata", {}),
            created_at=created_at,
            updated_at=updated_at,
            settings=settings,
            document_count=data.get("document_count", 0),
            chunk_count=data.get("chunk_count", 0),
        )

    def update(
        self,
        name: Optional[str] = None,
        description: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        settings: Optional[CollectionSettings] = None,
    ) -> "Collection":
        """
        Create updated copy of collection.

        Only provided fields are updated. Returns a new instance.

        Args:
            name: New name (if provided)
            description: New description (if provided)
            metadata: New metadata (if provided, replaces existing)
            settings: New settings (if provided)

        Returns:
            Updated Collection instance
        """
        return Collection(
            id=self.id,
            name=name if name is not None else self.name,
            description=description if description is not None else self.description,
            metadata=metadata if metadata is not None else self.metadata,
            created_at=self.created_at,
            updated_at=datetime.utcnow(),
            settings=settings if settings is not None else self.settings,
            document_count=self.document_count,
            chunk_count=self.chunk_count,
        )
