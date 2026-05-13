"""
Generic response models following Stripe-like API design.

These models provide consistent response structures for
list and delete operations.
"""

from dataclasses import dataclass, field
from typing import TypeVar, Generic, List, Optional, Dict, Any

T = TypeVar("T")


@dataclass
class ListResponse(Generic[T]):
    """
    Paginated list response.

    Follows Stripe's list response pattern with cursor-based pagination.

    Attributes:
        data: List of items
        has_more: Whether more items exist beyond this page
        total_count: Total number of items (if available)
        next_cursor: Cursor for fetching next page

    Example:
        >>> response = ListResponse(
        ...     data=[collection1, collection2],
        ...     has_more=True,
        ...     total_count=5,
        ...     next_cursor="col_abc123"
        ... )
    """
    data: List[T]
    has_more: bool = False
    total_count: Optional[int] = None
    next_cursor: Optional[str] = None

    def to_dict(self, item_serializer=None) -> Dict[str, Any]:
        """
        Convert to dictionary.

        Args:
            item_serializer: Optional function to serialize items.
                            If not provided, items must have to_dict method.

        Returns:
            Dictionary representation
        """
        if item_serializer:
            data = [item_serializer(item) for item in self.data]
        else:
            data = [item.to_dict() if hasattr(item, 'to_dict') else item for item in self.data]

        result = {
            "object": "list",
            "data": data,
            "has_more": self.has_more,
        }
        if self.total_count is not None:
            result["total_count"] = self.total_count
        if self.next_cursor:
            result["next_cursor"] = self.next_cursor
        return result

    def __len__(self) -> int:
        """Return number of items in this page."""
        return len(self.data)

    def __iter__(self):
        """Iterate over items."""
        return iter(self.data)

    def __getitem__(self, index: int) -> T:
        """Get item by index."""
        return self.data[index]


@dataclass
class DeletedResponse:
    """
    Response for delete operations.

    Follows Stripe's deleted response pattern.

    Attributes:
        id: ID of the deleted resource
        object: Type of the deleted resource
        deleted: Always True for successful deletions

    Example:
        >>> response = DeletedResponse(
        ...     id="col_abc123",
        ...     object="collection"
        ... )
    """
    id: str
    object: str = "resource"
    deleted: bool = True

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "object": self.object,
            "deleted": self.deleted,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DeletedResponse":
        """Create from dictionary."""
        return cls(
            id=data["id"],
            object=data.get("object", "resource"),
            deleted=data.get("deleted", True),
        )


@dataclass
class OperationResult:
    """
    Generic result for operations that may include warnings.

    Useful for soft limit scenarios where operation succeeds
    but user should be notified.

    Attributes:
        success: Whether operation completed successfully
        data: The result data (if successful)
        warnings: List of warning messages
        message: Success or error message

    Example:
        >>> result = OperationResult(
        ...     success=True,
        ...     data=new_collection,
        ...     warnings=["Approaching collection limit (3/3)"]
        ... )
    """
    success: bool
    data: Optional[Any] = None
    warnings: List[str] = field(default_factory=list)
    message: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        result = {
            "success": self.success,
        }
        if self.data is not None:
            if hasattr(self.data, 'to_dict'):
                result["data"] = self.data.to_dict()
            else:
                result["data"] = self.data
        if self.warnings:
            result["warnings"] = self.warnings
        if self.message:
            result["message"] = self.message
        return result

    @property
    def has_warnings(self) -> bool:
        """Check if there are any warnings."""
        return len(self.warnings) > 0
