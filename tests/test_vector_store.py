"""
Unit tests for VectorStoreManager - ChromaDB vector store operations.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock

# Check if langchain_chroma is available
try:
    import langchain_chroma
    HAS_CHROMA = True
except ImportError:
    HAS_CHROMA = False

# Skip entire module if langchain_chroma not installed
pytestmark = pytest.mark.skipif(not HAS_CHROMA, reason="langchain_chroma not installed")

if HAS_CHROMA:
    from core.vector_store import VectorStoreManager
    from langchain_core.documents import Document


@pytest.mark.unit
class TestVectorStoreManager:
    """Test suite for VectorStoreManager class."""

    @patch('core.vector_store.OpenAIEmbeddings')
    @patch('core.vector_store.Chroma')
    def test_initialization(self, mock_chroma, mock_embeddings, mock_env_vars):
        """Test VectorStoreManager initialization."""
        manager = VectorStoreManager(
            embedding_model_name="text-embedding-3-small",
            collection_name="test_collection",
            persist_directory="./test_db"
        )

        assert manager.embedding_model_name == "text-embedding-3-small"
        assert manager.collection_name == "test_collection"
        assert manager.persist_directory == "./test_db"
        mock_embeddings.assert_called_once()
        mock_chroma.assert_called_once()

    @patch('core.vector_store.OpenAIEmbeddings')
    @patch('core.vector_store.Chroma')
    def test_add_documents(self, mock_chroma, mock_embeddings, sample_documents, mock_env_vars):
        """Test adding documents to vector store."""
        # Setup mock
        mock_vs = MagicMock()
        mock_vs.add_documents.return_value = ["id1", "id2", "id3"]
        mock_chroma.return_value = mock_vs

        manager = VectorStoreManager()
        ids = manager.add_documents(sample_documents)

        assert len(ids) == 3
        assert ids == ["id1", "id2", "id3"]
        mock_vs.add_documents.assert_called_once_with(documents=sample_documents)

    @patch('core.vector_store.OpenAIEmbeddings')
    @patch('core.vector_store.Chroma')
    def test_get_retriever(self, mock_chroma, mock_embeddings, mock_env_vars):
        """Test creating a retriever from vector store."""
        mock_vs = MagicMock()
        mock_retriever = Mock()
        mock_vs.as_retriever.return_value = mock_retriever
        mock_chroma.return_value = mock_vs

        manager = VectorStoreManager()
        retriever = manager.get_retriever(search_type="similarity", search_k=5)

        mock_vs.as_retriever.assert_called_once_with(
            search_type="similarity",
            search_kwargs={"k": 5}
        )
        assert retriever == mock_retriever

    @patch('core.vector_store.OpenAIEmbeddings')
    @patch('core.vector_store.Chroma')
    def test_get_collection_count(self, mock_chroma, mock_embeddings, mock_env_vars):
        """Test getting document count from collection."""
        mock_vs = MagicMock()
        mock_vs._collection.count.return_value = 42
        mock_chroma.return_value = mock_vs

        manager = VectorStoreManager()
        count = manager.get_collection_count()

        assert count == 42
        mock_vs._collection.count.assert_called_once()

    @patch('core.vector_store.OpenAIEmbeddings')
    @patch('core.vector_store.Chroma')
    def test_get_collection_count_error_handling(self, mock_chroma, mock_embeddings, mock_env_vars):
        """Test that collection count errors are handled gracefully."""
        mock_vs = MagicMock()
        mock_vs._collection.count.side_effect = Exception("Connection error")
        mock_chroma.return_value = mock_vs

        manager = VectorStoreManager()
        count = manager.get_collection_count()

        assert count == 0  # Should return 0 on error

    @patch('core.vector_store.OpenAIEmbeddings')
    @patch('core.vector_store.Chroma')
    def test_clear_collection(self, mock_chroma, mock_embeddings, mock_env_vars):
        """Test clearing the vector store collection."""
        mock_vs = MagicMock()
        mock_chroma.return_value = mock_vs

        manager = VectorStoreManager()
        manager.clear_collection()

        # Should delete and recreate collection
        mock_vs.delete_collection.assert_called_once()
        # Chroma should be called twice: initial + recreate
        assert mock_chroma.call_count == 2

    @patch('core.vector_store.OpenAIEmbeddings')
    @patch('core.vector_store.Chroma')
    def test_search_similar(self, mock_chroma, mock_embeddings, sample_documents, mock_env_vars):
        """Test similarity search functionality."""
        mock_vs = MagicMock()
        mock_vs.similarity_search.return_value = sample_documents[:2]
        mock_chroma.return_value = mock_vs

        manager = VectorStoreManager()
        results = manager.search_similar("machine learning", k=2)

        assert len(results) == 2
        mock_vs.similarity_search.assert_called_once_with("machine learning", k=2)

    @patch('core.vector_store.OpenAIEmbeddings')
    @patch('core.vector_store.Chroma')
    def test_default_parameters(self, mock_chroma, mock_embeddings, mock_env_vars):
        """Test that default parameters are set correctly."""
        manager = VectorStoreManager()

        assert manager.embedding_model_name == "text-embedding-3-large"
        assert manager.collection_name == "semantic_search_docs_streamlit"
        assert manager.persist_directory == "./chroma/db"

    @patch('core.vector_store.OpenAIEmbeddings')
    @patch('core.vector_store.Chroma')
    def test_get_all_documents(self, mock_chroma, mock_embeddings, mock_env_vars):
        """Test retrieving all documents from vector store."""
        mock_vs = MagicMock()
        mock_collection = MagicMock()
        mock_collection.get.return_value = {
            "documents": ["content1", "content2", "content3"],
            "metadatas": [
                {"source": "doc1.pdf"},  # No collection_id
                {"source": "doc2.pdf", "collection_id": "col-1"},  # In collection
                {"source": "doc3.pdf"}  # No collection_id
            ]
        }
        mock_vs._collection = mock_collection
        mock_chroma.return_value = mock_vs

        manager = VectorStoreManager()

        # Get non-collection documents (default)
        docs = manager.get_all_documents()
        assert len(docs) == 2  # Only docs without collection_id
        assert docs[0].page_content == "content1"
        assert docs[1].page_content == "content3"

    @patch('core.vector_store.OpenAIEmbeddings')
    @patch('core.vector_store.Chroma')
    def test_get_all_documents_by_collection(self, mock_chroma, mock_embeddings, mock_env_vars):
        """Test retrieving documents filtered by collection_id."""
        mock_vs = MagicMock()
        mock_collection = MagicMock()
        mock_collection.get.return_value = {
            "documents": ["content1", "content2", "content3"],
            "metadatas": [
                {"source": "doc1.pdf"},
                {"source": "doc2.pdf", "collection_id": "col-1"},
                {"source": "doc3.pdf", "collection_id": "col-1"}
            ]
        }
        mock_vs._collection = mock_collection
        mock_chroma.return_value = mock_vs

        manager = VectorStoreManager()

        # Get documents by collection
        docs = manager.get_all_documents(collection_id="col-1")
        assert len(docs) == 2
        assert docs[0].metadata.get("collection_id") == "col-1"

    @patch('core.vector_store.OpenAIEmbeddings')
    @patch('core.vector_store.Chroma')
    def test_get_non_collection_count(self, mock_chroma, mock_embeddings, mock_env_vars):
        """Test counting non-collection documents."""
        mock_vs = MagicMock()
        mock_collection = MagicMock()
        mock_collection.get.return_value = {
            "metadatas": [
                {"source": "doc1.pdf"},
                {"source": "doc2.pdf", "collection_id": "col-1"},
                {"source": "doc3.pdf"},
                None  # Handle None metadata
            ]
        }
        mock_vs._collection = mock_collection
        mock_chroma.return_value = mock_vs

        manager = VectorStoreManager()
        count = manager.get_non_collection_count()

        assert count == 3  # 2 without collection_id + 1 None
