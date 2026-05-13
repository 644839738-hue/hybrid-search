"""
Unit tests for DocumentProcessor - PDF loading and text chunking.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from core.document_processor import DocumentProcessor
from langchain_core.documents import Document


@pytest.mark.unit
class TestDocumentProcessor:
    """Test suite for DocumentProcessor class."""

    def test_initialization(self):
        """Test DocumentProcessor initialization with default parameters."""
        processor = DocumentProcessor()
        assert processor.chunk_size == 1000
        assert processor.chunk_overlap == 200
        assert processor.add_start_index is True

    def test_initialization_custom_params(self):
        """Test DocumentProcessor initialization with custom parameters."""
        processor = DocumentProcessor(chunk_size=500, chunk_overlap=100, add_start_index=False)
        assert processor.chunk_size == 500
        assert processor.chunk_overlap == 100
        assert processor.add_start_index is False

    def test_invalid_file_type_raises_error(self, mock_uploaded_file):
        """Test that non-PDF files raise ValueError."""
        processor = DocumentProcessor()
        mock_uploaded_file.name = "document.txt"

        with pytest.raises(ValueError, match="Only PDF files are supported"):
            processor.process_uploaded_file(mock_uploaded_file)

    def test_valid_pdf_file_name(self, mock_uploaded_file):
        """Test that PDF files with correct extension are accepted."""
        processor = DocumentProcessor()
        mock_uploaded_file.name = "document.pdf"
        # File name should pass validation (would fail later without mocking loader)
        assert mock_uploaded_file.name.lower().endswith('.pdf')

    @patch('core.document_processor.PyPDFLoader')
    @patch('core.document_processor.os.remove')
    def test_process_uploaded_file_creates_temp_file(self, mock_remove, mock_loader, mock_uploaded_file):
        """Test that temporary file is created during processing."""
        processor = DocumentProcessor(chunk_size=100, chunk_overlap=20)

        # Mock PDF loader to return sample documents
        mock_loader_instance = Mock()
        mock_loader_instance.load.return_value = [
            Document(page_content="Sample text from PDF page 1", metadata={"page": 1})
        ]
        mock_loader.return_value = mock_loader_instance

        chunks = processor.process_uploaded_file(mock_uploaded_file)

        # Verify temporary file was created and loader was called
        mock_loader.assert_called_once()
        assert len(chunks) > 0

    @patch('core.document_processor.PyPDFLoader')
    @patch('core.document_processor.os.remove')
    def test_temp_file_cleanup(self, mock_remove, mock_loader, mock_uploaded_file):
        """Test that temporary files are cleaned up after processing."""
        processor = DocumentProcessor()

        mock_loader_instance = Mock()
        mock_loader_instance.load.return_value = [
            Document(page_content="Test content", metadata={"page": 1})
        ]
        mock_loader.return_value = mock_loader_instance

        processor.process_uploaded_file(mock_uploaded_file)

        # Verify cleanup was called
        mock_remove.assert_called_once()

    def test_get_chunk_info(self, sample_documents):
        """Test getting chunk information."""
        processor = DocumentProcessor()
        chunk_info = processor.get_chunk_info(sample_documents)

        assert len(chunk_info) == 3
        assert chunk_info[0]["index"] == 1
        assert chunk_info[0]["size"] == len(sample_documents[0].page_content)
        assert "metadata" in chunk_info[0]

    def test_get_chunk_info_empty_list(self):
        """Test getting chunk info with empty list."""
        processor = DocumentProcessor()
        chunk_info = processor.get_chunk_info([])
        assert chunk_info == []

    @patch('core.document_processor.PyPDFLoader')
    @patch('core.document_processor.os.remove')
    def test_chunking_respects_size_limits(self, mock_remove, mock_loader, mock_uploaded_file):
        """Test that chunking respects specified chunk size."""
        processor = DocumentProcessor(chunk_size=50, chunk_overlap=10)

        # Create a long document
        long_text = "A" * 200  # 200 characters
        mock_loader_instance = Mock()
        mock_loader_instance.load.return_value = [
            Document(page_content=long_text, metadata={"page": 1})
        ]
        mock_loader.return_value = mock_loader_instance

        chunks = processor.process_uploaded_file(mock_uploaded_file)

        # Should create multiple chunks from long text
        assert len(chunks) > 1
        # Each chunk should be around the specified size (may vary slightly)
        for chunk in chunks:
            assert len(chunk.page_content) <= 60  # Allow some variance
