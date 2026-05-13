"""
Data models for collection management.

This module provides type-safe dataclasses for collections, documents,
and search operations following Stripe-like API design principles.
"""

from core.models.errors import (
    APIError,
    ValidationError,
    NotFoundError,
    DuplicateError,
    LimitExceededError,
)
from core.models.collection import Collection, CollectionSettings
from core.models.document import Document, DocumentStatus
from core.models.search import (
    SearchRequest,
    SearchResult,
    SearchScores,
    SearchResponse,
    RetrievalMethod,
)
from core.models.responses import ListResponse, DeletedResponse

__all__ = [
    # Errors
    "APIError",
    "ValidationError",
    "NotFoundError",
    "DuplicateError",
    "LimitExceededError",
    # Collection
    "Collection",
    "CollectionSettings",
    # Document
    "Document",
    "DocumentStatus",
    # Search
    "SearchRequest",
    "SearchResult",
    "SearchScores",
    "SearchResponse",
    "RetrievalMethod",
    # Responses
    "ListResponse",
    "DeletedResponse",
]
