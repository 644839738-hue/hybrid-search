"""
Tests for Vector Store Filter Functionality (M4).

Tests collection-scoped search, document-scoped search, and cascade deletion.
Uses mocks to avoid heavy dependency requirements.
"""

import pytest
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch, PropertyMock
from typing import List

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from langchain_core.documents import Document


class TestDocumentProcessorMetadata:
    """Tests for DocumentProcessor metadata handling."""

    def test_metadata_added_to_chunks(self):
        """Test that collection_id and document_id are added to chunks."""
        from core.document_processor import DocumentProcessor

        processor = DocumentProcessor(chunk_size=500, chunk_overlap=50)

        # Create a mock uploaded file
        mock_file = MagicMock()
        mock_file.name = "test.pdf"
        mock_file.size = 1000
        mock_file.getbuffer.return_value = b"%PDF-1.4 test content"

        # Mock the PDF loading to return test documents
        mock_docs = [
            Document(page_content="This is page one content.", metadata={"page": 0}),
            Document(page_content="This is page two content.", metadata={"page": 1}),
        ]

        with patch.object(processor, '_load_pdf', return_value=mock_docs):
            with patch.object(processor, '_cleanup_temp_file'):
                chunks = processor.process_uploaded_file(
                    mock_file,
                    collection_id="test-collection-123",
                    document_id="test-document-456",
                    extra_metadata={"custom_field": "custom_value"}
                )

        # Verify metadata on all chunks
        assert len(chunks) > 0
        for chunk in chunks:
            assert chunk.metadata.get("collection_id") == "test-collection-123"
            assert chunk.metadata.get("document_id") == "test-document-456"
            assert chunk.metadata.get("source") == "test.pdf"
            assert chunk.metadata.get("custom_field") == "custom_value"

    def test_metadata_optional(self):
        """Test that metadata parameters are optional."""
        from core.document_processor import DocumentProcessor

        processor = DocumentProcessor(chunk_size=500, chunk_overlap=50)

        # Create a mock uploaded file
        mock_file = MagicMock()
        mock_file.name = "test.pdf"
        mock_file.size = 1000
        mock_file.getbuffer.return_value = b"%PDF-1.4 test content"

        # Mock the PDF loading
        mock_docs = [
            Document(page_content="Test content.", metadata={"page": 0}),
        ]

        with patch.object(processor, '_load_pdf', return_value=mock_docs):
            with patch.object(processor, '_cleanup_temp_file'):
                # Process without collection/document IDs
                chunks = processor.process_uploaded_file(mock_file)

        # Verify no collection/document IDs are set
        assert len(chunks) > 0
        for chunk in chunks:
            assert "collection_id" not in chunk.metadata
            assert "document_id" not in chunk.metadata
            assert chunk.metadata.get("source") == "test.pdf"


class TestSearchRequestFilters:
    """Tests for SearchRequest filter generation."""

    def test_collection_filter(self):
        """Test filter generation for collection scope."""
        from core.models.search import SearchRequest

        request = SearchRequest(
            query="test query",
            collection_id="col-123"
        )

        filter_dict = request.get_filter()

        assert filter_dict is not None
        # ChromaDB requires explicit $eq operator for equality filters
        assert filter_dict.get("collection_id") == {"$eq": "col-123"}

    def test_single_document_filter(self):
        """Test filter generation for single document scope."""
        from core.models.search import SearchRequest

        request = SearchRequest(
            query="test query",
            document_ids=["doc-456"]
        )

        filter_dict = request.get_filter()

        assert filter_dict is not None
        # ChromaDB requires explicit $eq operator for single document filter
        assert filter_dict.get("document_id") == {"$eq": "doc-456"}

    def test_multiple_documents_filter(self):
        """Test filter generation for multiple documents scope."""
        from core.models.search import SearchRequest

        request = SearchRequest(
            query="test query",
            document_ids=["doc-1", "doc-2", "doc-3"]
        )

        filter_dict = request.get_filter()

        assert filter_dict is not None
        assert "document_id" in filter_dict
        assert filter_dict["document_id"] == {"$in": ["doc-1", "doc-2", "doc-3"]}

    def test_combined_filter(self):
        """Test filter generation with both collection and documents."""
        from core.models.search import SearchRequest

        request = SearchRequest(
            query="test query",
            collection_id="col-123",
            document_ids=["doc-1", "doc-2"]
        )

        filter_dict = request.get_filter()

        assert filter_dict is not None
        # Should have $and with both conditions
        assert "$and" in filter_dict

    def test_no_filter(self):
        """Test that no filter is generated when no scope specified."""
        from core.models.search import SearchRequest

        request = SearchRequest(query="test query")
        filter_dict = request.get_filter()

        assert filter_dict is None


# Check if langchain_chroma is available for VectorStoreManager tests
try:
    import langchain_chroma
    HAS_CHROMA = True
except ImportError:
    HAS_CHROMA = False


@pytest.mark.skipif(not HAS_CHROMA, reason="langchain_chroma not installed")
class TestVectorStoreManagerMethods:
    """Tests for VectorStoreManager new methods using mocks."""

    @pytest.fixture
    def mock_vector_store(self):
        """Create a mock VectorStoreManager with mocked internals."""
        # Create mock Chroma collection
        mock_collection = MagicMock()
        mock_collection.count.return_value = 6
        mock_collection.get.return_value = {
            "ids": ["chunk-1", "chunk-2"],
            "documents": ["Content 1", "Content 2"],
            "metadatas": [
                {"document_id": "doc-1", "collection_id": "col-1"},
                {"document_id": "doc-1", "collection_id": "col-1"}
            ]
        }
        mock_collection.delete.return_value = None

        # Create mock Chroma vector store
        mock_chroma = MagicMock()
        mock_chroma._collection = mock_collection
        mock_chroma.similarity_search.return_value = [
            Document(
                page_content="Test content",
                metadata={"collection_id": "col-1", "document_id": "doc-1"}
            )
        ]
        mock_chroma.as_retriever.return_value = MagicMock()

        return mock_chroma, mock_collection

    def test_search_similar_with_filter(self, mock_vector_store):
        """Test search_similar passes filter to Chroma."""
        mock_chroma, mock_collection = mock_vector_store

        # Create a partial mock of VectorStoreManager
        from core.vector_store import VectorStoreManager

        with patch.object(VectorStoreManager, '__init__', lambda x, **kwargs: None):
            manager = VectorStoreManager()
            manager.vector_store = mock_chroma

            # Call with filter
            results = manager.search_similar(
                query="test query",
                k=5,
                filter={"collection_id": "col-1"}
            )

            # Verify filter was passed
            mock_chroma.similarity_search.assert_called_once_with(
                "test query",
                k=5,
                filter={"collection_id": "col-1"}
            )

    def test_search_by_collection(self, mock_vector_store):
        """Test search_by_collection creates correct filter."""
        mock_chroma, mock_collection = mock_vector_store

        from core.vector_store import VectorStoreManager

        with patch.object(VectorStoreManager, '__init__', lambda x, **kwargs: None):
            manager = VectorStoreManager()
            manager.vector_store = mock_chroma

            results = manager.search_by_collection(
                query="test query",
                collection_id="col-123",
                k=5
            )

            # Verify correct filter was used (ChromaDB requires $eq operator)
            mock_chroma.similarity_search.assert_called_once_with(
                "test query",
                k=5,
                filter={"collection_id": {"$eq": "col-123"}}
            )

    def test_search_by_documents_single(self, mock_vector_store):
        """Test search_by_documents with single document."""
        mock_chroma, mock_collection = mock_vector_store

        from core.vector_store import VectorStoreManager

        with patch.object(VectorStoreManager, '__init__', lambda x, **kwargs: None):
            manager = VectorStoreManager()
            manager.vector_store = mock_chroma

            results = manager.search_by_documents(
                query="test query",
                document_ids=["doc-1"],
                k=5
            )

            # Verify single document filter (uses $eq, not $in)
            mock_chroma.similarity_search.assert_called_once_with(
                "test query",
                k=5,
                filter={"document_id": {"$eq": "doc-1"}}
            )

    def test_search_by_documents_multiple(self, mock_vector_store):
        """Test search_by_documents with multiple documents."""
        mock_chroma, mock_collection = mock_vector_store

        from core.vector_store import VectorStoreManager

        with patch.object(VectorStoreManager, '__init__', lambda x, **kwargs: None):
            manager = VectorStoreManager()
            manager.vector_store = mock_chroma

            results = manager.search_by_documents(
                query="test query",
                document_ids=["doc-1", "doc-2", "doc-3"],
                k=5
            )

            # Verify $in filter for multiple documents
            mock_chroma.similarity_search.assert_called_once_with(
                "test query",
                k=5,
                filter={"document_id": {"$in": ["doc-1", "doc-2", "doc-3"]}}
            )

    def test_delete_by_document_id(self, mock_vector_store):
        """Test delete_by_document_id removes chunks."""
        mock_chroma, mock_collection = mock_vector_store

        from core.vector_store import VectorStoreManager

        with patch.object(VectorStoreManager, '__init__', lambda x, **kwargs: None):
            manager = VectorStoreManager()
            manager.vector_store = mock_chroma

            deleted = manager.delete_by_document_id("doc-1")

            # Verify get was called with correct filter (ChromaDB requires $eq operator)
            mock_collection.get.assert_called_once_with(
                where={"document_id": {"$eq": "doc-1"}},
                include=[]
            )

            # Verify delete was called with chunk IDs
            mock_collection.delete.assert_called_once_with(ids=["chunk-1", "chunk-2"])

            # Verify return count
            assert deleted == 2

    def test_delete_by_document_id_no_chunks(self, mock_vector_store):
        """Test delete_by_document_id when no chunks exist."""
        mock_chroma, mock_collection = mock_vector_store
        mock_collection.get.return_value = {"ids": []}

        from core.vector_store import VectorStoreManager

        with patch.object(VectorStoreManager, '__init__', lambda x, **kwargs: None):
            manager = VectorStoreManager()
            manager.vector_store = mock_chroma

            deleted = manager.delete_by_document_id("nonexistent")

            # Verify delete was NOT called
            mock_collection.delete.assert_not_called()

            # Verify return count is 0
            assert deleted == 0

    def test_delete_by_collection_id(self, mock_vector_store):
        """Test delete_by_collection_id removes chunks."""
        mock_chroma, mock_collection = mock_vector_store
        mock_collection.get.return_value = {"ids": ["c1", "c2", "c3", "c4"]}

        from core.vector_store import VectorStoreManager

        with patch.object(VectorStoreManager, '__init__', lambda x, **kwargs: None):
            manager = VectorStoreManager()
            manager.vector_store = mock_chroma

            deleted = manager.delete_by_collection_id("col-1")

            # Verify get was called with correct filter (ChromaDB requires $eq operator)
            mock_collection.get.assert_called_once_with(
                where={"collection_id": {"$eq": "col-1"}},
                include=[]
            )

            # Verify delete was called
            mock_collection.delete.assert_called_once_with(ids=["c1", "c2", "c3", "c4"])

            assert deleted == 4

    def test_get_chunks_by_document(self, mock_vector_store):
        """Test get_chunks_by_document returns Document objects."""
        mock_chroma, mock_collection = mock_vector_store

        from core.vector_store import VectorStoreManager

        with patch.object(VectorStoreManager, '__init__', lambda x, **kwargs: None):
            manager = VectorStoreManager()
            manager.vector_store = mock_chroma

            chunks = manager.get_chunks_by_document("doc-1")

            # Verify get was called with includes (ChromaDB requires $eq operator)
            mock_collection.get.assert_called_once_with(
                where={"document_id": {"$eq": "doc-1"}},
                include=["documents", "metadatas"]
            )

            # Verify return type
            assert len(chunks) == 2
            assert all(isinstance(c, Document) for c in chunks)
            assert chunks[0].page_content == "Content 1"
            assert chunks[0].metadata.get("document_id") == "doc-1"

    def test_get_retriever_with_filter(self, mock_vector_store):
        """Test get_retriever passes filter to search_kwargs."""
        mock_chroma, mock_collection = mock_vector_store

        from core.vector_store import VectorStoreManager

        with patch.object(VectorStoreManager, '__init__', lambda x, **kwargs: None):
            manager = VectorStoreManager()
            manager.vector_store = mock_chroma

            retriever = manager.get_retriever(
                search_k=5,
                filter={"collection_id": "col-1"}
            )

            # Verify as_retriever was called with filter in search_kwargs
            mock_chroma.as_retriever.assert_called_once()
            call_kwargs = mock_chroma.as_retriever.call_args[1]
            assert call_kwargs["search_kwargs"]["filter"] == {"collection_id": "col-1"}
            assert call_kwargs["search_kwargs"]["k"] == 5


class TestDocumentManagerCascadeDelete:
    """Tests for DocumentManager cascade deletion integration."""

    def test_delete_calls_vector_store(self):
        """Test that document deletion calls vector store delete."""
        from core.document_manager import DocumentManager
        from core.models.document import Document, DocumentStatus

        # Create mock storage
        mock_storage = MagicMock()
        mock_storage.get_by_id.return_value = {
            "id": "doc-123",
            "collection_id": "col-1",
            "filename": "test.pdf",
            "file_hash": "abc123",
            "file_size": 1000,
            "status": "ready"
        }
        mock_storage.delete.return_value = True

        # Create mock vector store
        mock_vector_store = MagicMock()
        mock_vector_store.delete_by_document_id.return_value = 5

        manager = DocumentManager(
            storage=mock_storage,
            vector_store=mock_vector_store
        )

        response = manager.delete("doc-123")

        # Verify vector store was called
        mock_vector_store.delete_by_document_id.assert_called_once_with("doc-123")

        # Verify storage delete was called
        mock_storage.delete.assert_called_once()

        assert response.deleted

    def test_delete_without_vector_store(self):
        """Test that document deletion works without vector store."""
        from core.document_manager import DocumentManager

        # Create mock storage
        mock_storage = MagicMock()
        mock_storage.get_by_id.return_value = {
            "id": "doc-123",
            "collection_id": "col-1",
            "filename": "test.pdf",
            "file_hash": "abc123",
            "file_size": 1000,
            "status": "ready"
        }
        mock_storage.delete.return_value = True

        # No vector store
        manager = DocumentManager(storage=mock_storage, vector_store=None)

        response = manager.delete("doc-123")

        # Verify storage delete was called
        mock_storage.delete.assert_called_once()

        assert response.deleted
