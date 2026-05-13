"""
Custom exception classes following Stripe-like error design.

All errors inherit from APIError and provide structured error responses
with code, message, param, and type fields.
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any


@dataclass
class APIError(Exception):
    """
    Base exception for all API errors.

    Follows Stripe's error response format for consistency.

    Attributes:
        code: Machine-readable error code (e.g., "not_found", "limit_exceeded")
        message: Human-readable error message
        param: The parameter that caused the error (if applicable)
        error_type: Category of error (e.g., "validation_error", "invalid_request")
        details: Additional error details

    Example:
        >>> raise APIError(
        ...     code="not_found",
        ...     message="Collection 'abc123' not found",
        ...     param="collection_id",
        ...     error_type="invalid_request"
        ... )
    """
    code: str
    message: str
    param: Optional[str] = None
    error_type: str = "api_error"
    details: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        super().__init__(self.message)

    def to_dict(self) -> Dict[str, Any]:
        """Convert error to Stripe-like response format."""
        error_dict = {
            "error": {
                "code": self.code,
                "message": self.message,
                "type": self.error_type,
            }
        }
        if self.param:
            error_dict["error"]["param"] = self.param
        if self.details:
            error_dict["error"]["details"] = self.details
        return error_dict


@dataclass
class ValidationError(APIError):
    """
    Raised when input validation fails.

    Example:
        >>> raise ValidationError(
        ...     message="Collection name cannot be empty",
        ...     param="name"
        ... )
    """
    code: str = "validation_error"
    message: str = "Validation failed"
    param: Optional[str] = None
    error_type: str = "validation_error"
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class NotFoundError(APIError):
    """
    Raised when a requested resource doesn't exist.

    Example:
        >>> raise NotFoundError(
        ...     message="Collection 'abc123' not found",
        ...     param="collection_id",
        ...     resource_type="collection",
        ...     resource_id="abc123"
        ... )
    """
    code: str = "not_found"
    message: str = "Resource not found"
    param: Optional[str] = None
    error_type: str = "invalid_request"
    details: Dict[str, Any] = field(default_factory=dict)
    resource_type: Optional[str] = None
    resource_id: Optional[str] = None

    def __post_init__(self):
        if self.resource_type and self.resource_id:
            self.details["resource_type"] = self.resource_type
            self.details["resource_id"] = self.resource_id
        super().__post_init__()


@dataclass
class DuplicateError(APIError):
    """
    Raised when attempting to create a duplicate resource.

    Example:
        >>> raise DuplicateError(
        ...     message="Collection with name 'Research' already exists",
        ...     param="name",
        ...     existing_id="abc123"
        ... )
    """
    code: str = "duplicate"
    message: str = "Resource already exists"
    param: Optional[str] = None
    error_type: str = "invalid_request"
    details: Dict[str, Any] = field(default_factory=dict)
    existing_id: Optional[str] = None

    def __post_init__(self):
        if self.existing_id:
            self.details["existing_id"] = self.existing_id
        super().__post_init__()


@dataclass
class LimitExceededError(APIError):
    """
    Raised when a soft limit is exceeded.

    This is a warning-level error - operations may still proceed
    but the user should be notified.

    Example:
        >>> raise LimitExceededError(
        ...     message="Maximum 3 collections allowed",
        ...     param="collection_count",
        ...     limit=3,
        ...     current=3
        ... )
    """
    code: str = "limit_exceeded"
    message: str = "Limit exceeded"
    param: Optional[str] = None
    error_type: str = "invalid_request"
    details: Dict[str, Any] = field(default_factory=dict)
    limit: Optional[int] = None
    current: Optional[int] = None

    def __post_init__(self):
        if self.limit is not None:
            self.details["limit"] = self.limit
        if self.current is not None:
            self.details["current"] = self.current
        super().__post_init__()
