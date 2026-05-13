"""
Integration Tests for Document Lifecycle.

These tests run against REAL ChromaDB and verify:
1. Documents are indexed in the correct collection
2. Documents are searchable
3. Deletes properly clean up ChromaDB (no orphan chunks)
4. JSON storage stays in sync

Requirements:
- ChromaDB Docker running: docker run -d -p 8000:8000 chromadb/chroma
- OpenAI API key set

Run with: pytest tests/test_integration.py -v -s
"""

import pytest
import sys
import os
import tempfile
import time
from pathlib import Path
from io import BytesIO
from unittest.mock import MagicMock
from dotenv import load_dotenv

# Load .env file
load_dotenv()

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import chromadb
from core.vector_store import VectorStoreManager
from core.document_processor import DocumentProcessor
from core.collection_manager import CollectionManager
from core.document_manager import DocumentManager
from core.search_manager import SearchManager
from core.storage import JSONStorage, COLLECTIONS_FILE, DOCUMENTS_FILE
from core.models.search import SearchRequest, RetrievalMethod
from config_loader import ConfigLoader


# Test collection name - isolated from production
TEST_COLLECTION_NAME = "test_integration_collection"


def chromadb_available():
    """Check if ChromaDB is running."""
    try:
        client = chromadb.HttpClient(host='localhost', port=8000)
        client.heartbeat()
        return True
    except Exception:
        return False


def openai_available():
    """Check if OpenAI API key is set."""
    return bool(os.getenv("OPENAI_API_KEY"))


# Skip all tests if dependencies not available
pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(not chromadb_available(), reason="ChromaDB not running on localhost:8000"),
    pytest.mark.skipif(not openai_available(), reason="OPENAI_API_KEY not set"),
]


@pytest.fixture
def test_pdf_content():
    """Create a minimal valid PDF for testing."""
    # Minimal PDF with text content
    pdf_content = b"""%PDF-1.4
1 0 obj
<< /Type /Catalog /Pages 2 0 R >>
endobj
2 0 obj
<< /Type /Pages /Kids [3 0 R] /Count 1 >>
endobj
3 0 obj
<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792]
   /Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >>
endobj
4 0 obj
<< /Length 178 >>
stream
BT
/F1 12 Tf
100 700 Td
(This is a test document about machine learning and artificial intelligence.) Tj
0 -20 Td
(Neural networks are a key component of deep learning systems.) Tj
0 -20 Td
(Python is commonly used for AI development.) Tj
ET
endstream
endobj
5 0 obj
<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>
endobj
xref
0 6
0000000000 65535 f
0000000009 00000 n
0000000058 00000 n
0000000115 00000 n
0000000266 00000 n
0000000497 00000 n
trailer
<< /Size 6 /Root 1 0 R >>
startxref
576
%%EOF"""
    return pdf_content


@pytest.fixture
def temp_storage():
    """Create temporary JSON storage."""
    with tempfile.TemporaryDirectory() as tmpdir:
        storage = JSONStorage(data_dir=tmpdir)
        yield storage


@pytest.fixture
def vector_store():
    """Create VectorStoreManager with test collection."""
    # Use test collection name
    manager = VectorStoreManager(
        collection_name=TEST_COLLECTION_NAME,
        use_docker=True
    )

    yield manager

    # Cleanup: delete all chunks in test collection
    try:
        collection = manager.vector_store._collection
        all_ids = collection.get()['ids']
        if all_ids:
            collection.delete(ids=all_ids)
    except Exception:
        pass


@pytest.fixture
def chroma_client():
    """Direct ChromaDB client for verification."""
    return chromadb.HttpClient(host='localhost', port=8000)


class TestDocumentLifecycleStandalone:
    """Test document lifecycle - single document delete."""

    def test_full_lifecycle(self, temp_storage, vector_store, test_pdf_content, chroma_client):
        """
        Test document delete lifecycle:
        1. Create collection
        2. Upload document
        3. Verify chunks in ChromaDB
        4. Verify searchable
        5. Delete document (not collection)
        6. Verify no orphan chunks for that document
        7. Verify JSON clean for document
        """
        print("\n=== Test: Single Document Delete Lifecycle ===")

        # Setup managers
        collection_manager = CollectionManager(storage=temp_storage, vector_store=vector_store)
        doc_manager = DocumentManager(storage=temp_storage, vector_store=vector_store)
        doc_processor = DocumentProcessor(chunk_size=200, chunk_overlap=20)

        # Step 1: Create collection first (required by this project)
        print("Step 1: Creating collection...")
        col_result = collection_manager.create(name="Standalone Test Collection")
        collection_id = col_result.data.id
        print(f"  Collection ID: {collection_id}")

        # Create mock uploaded file
        mock_file = MagicMock()
        mock_file.name = "test_ml_document.pdf"
        mock_file.size = len(test_pdf_content)
        mock_file.getvalue.return_value = test_pdf_content
        mock_file.getbuffer.return_value = test_pdf_content
        mock_file.read.return_value = test_pdf_content
        mock_file.seek = MagicMock()

        # Step 2: Process and upload document
        print("\nStep 2: Processing document...")
        chunks = doc_processor.process_uploaded_file(mock_file)
        print(f"  Created {len(chunks)} chunks")

        doc_result = doc_manager.add(
            collection_id=collection_id,
            filename=mock_file.name,
            file_content=test_pdf_content
        )
        document = doc_result.data
        document_id = document.id
        print(f"  Document ID: {document_id}")

        for chunk in chunks:
            chunk.metadata["document_id"] = document_id
            chunk.metadata["collection_id"] = collection_id

        # Index chunks
        chroma_ids = vector_store.add_documents(chunks)
        print(f"  Indexed {len(chroma_ids)} chunks in ChromaDB")

        # Step 3: Verify chunks exist in correct collection
        print("\nStep 3: Verifying chunks in ChromaDB...")
        collection = chroma_client.get_collection(TEST_COLLECTION_NAME)
        chunk_count = collection.count()
        print(f"  Collection '{TEST_COLLECTION_NAME}' has {chunk_count} chunks")
        assert chunk_count == len(chunks), f"Expected {len(chunks)} chunks, got {chunk_count}"

        # Verify document_id in metadata
        results = collection.get(include=["metadatas"])
        doc_ids_in_chroma = set(m.get("document_id") for m in results["metadatas"] if m)
        print(f"  Document IDs in ChromaDB: {doc_ids_in_chroma}")
        assert document_id in doc_ids_in_chroma, "Document ID not found in chunk metadata"

        # Step 4: Verify searchable
        print("\nStep 4: Verifying document is searchable...")
        search_results = vector_store.search_similar("machine learning", k=3)
        print(f"  Search returned {len(search_results)} results")
        assert len(search_results) > 0, "Search returned no results"

        # Verify our document is in results
        result_sources = [r.metadata.get("source") for r in search_results]
        print(f"  Result sources: {result_sources}")
        assert mock_file.name in result_sources, "Our document not found in search results"

        # Step 5: Delete document (not collection)
        print("\nStep 5: Deleting document...")
        deleted_chunks = vector_store.delete_by_document_id(document_id)
        print(f"  Deleted {deleted_chunks} chunks from ChromaDB")

        delete_result = doc_manager.delete(document_id)
        print(f"  Document deleted from storage: {delete_result.deleted}")

        # Step 6: Verify no orphan chunks for that document
        print("\nStep 6: Verifying no orphan chunks...")
        chunk_count_after = collection.count()
        print(f"  Collection now has {chunk_count_after} chunks")
        assert chunk_count_after == 0, f"Orphan chunks found: {chunk_count_after}"

        # Step 7: Verify document JSON clean (collection still exists)
        print("\nStep 7: Verifying JSON storage...")
        docs_in_storage = temp_storage.load(DOCUMENTS_FILE)
        collections_in_storage = temp_storage.load(COLLECTIONS_FILE)
        print(f"  Documents in JSON: {len(docs_in_storage)}")
        print(f"  Collections in JSON: {len(collections_in_storage)}")
        assert len(docs_in_storage) == 0, f"Orphan documents in JSON: {docs_in_storage}"
        assert len(collections_in_storage) == 1, "Collection should still exist"

        # Cleanup: delete the collection
        collection_manager.delete(collection_id, force=True)

        print("\n✅ Test PASSED: Single document delete lifecycle")


class TestDocumentLifecycleWithCollection:
    """Test document lifecycle with collection."""

    def test_full_lifecycle(self, temp_storage, vector_store, test_pdf_content, chroma_client):
        """
        Test complete document lifecycle with collection:
        1. Create collection
        2. Upload document to collection
        3. Verify chunks have collection_id
        4. Verify searchable with collection filter
        5. Delete collection (cascade)
        6. Verify no orphan chunks
        7. Verify JSON clean
        """
        print("\n=== Test: Document Lifecycle With Collection ===")

        # Setup managers
        collection_manager = CollectionManager(storage=temp_storage, vector_store=vector_store)
        doc_manager = DocumentManager(storage=temp_storage, vector_store=vector_store)
        search_manager = SearchManager(storage=temp_storage, vector_store=vector_store)
        doc_processor = DocumentProcessor(chunk_size=200, chunk_overlap=20)

        # Step 1: Create collection
        print("Step 1: Creating collection...")
        col_result = collection_manager.create(
            name="Test ML Collection",
            description="Test collection for ML documents"
        )
        collection_obj = col_result.data
        collection_id = collection_obj.id
        print(f"  Collection ID: {collection_id}")

        # Create mock uploaded file
        mock_file = MagicMock()
        mock_file.name = "test_ai_paper.pdf"
        mock_file.size = len(test_pdf_content)
        mock_file.getvalue.return_value = test_pdf_content
        mock_file.getbuffer.return_value = test_pdf_content
        mock_file.read.return_value = test_pdf_content
        mock_file.seek = MagicMock()

        # Step 2: Upload document to collection
        print("\nStep 2: Uploading document to collection...")
        chunks = doc_processor.process_uploaded_file(mock_file)
        print(f"  Created {len(chunks)} chunks")

        doc_result = doc_manager.add(
            collection_id=collection_id,
            filename=mock_file.name,
            file_content=test_pdf_content
        )
        document = doc_result.data
        document_id = document.id
        print(f"  Document ID: {document_id}")

        # Add collection_id and document_id to chunks
        for chunk in chunks:
            chunk.metadata["collection_id"] = collection_id
            chunk.metadata["document_id"] = document_id

        # Index chunks
        chroma_ids = vector_store.add_documents(chunks)
        print(f"  Indexed {len(chroma_ids)} chunks")

        # Step 3: Verify chunks have collection_id
        print("\nStep 3: Verifying chunks have collection_id...")
        collection = chroma_client.get_collection(TEST_COLLECTION_NAME)
        results = collection.get(include=["metadatas"])

        chunks_with_col_id = sum(1 for m in results["metadatas"] if m and m.get("collection_id") == collection_id)
        print(f"  Chunks with collection_id '{collection_id}': {chunks_with_col_id}/{len(chunks)}")
        assert chunks_with_col_id == len(chunks), "Not all chunks have collection_id"

        # Step 4: Verify searchable with collection filter
        print("\nStep 4: Verifying searchable with collection filter...")
        search_results = vector_store.search_by_collection(
            query="artificial intelligence",
            collection_id=collection_id,
            k=3
        )
        print(f"  Search returned {len(search_results)} results")
        assert len(search_results) > 0, "Search returned no results"

        # Verify results are from our collection
        for r in search_results:
            assert r.metadata.get("collection_id") == collection_id, \
                f"Result from wrong collection: {r.metadata.get('collection_id')}"
        print("  All results have correct collection_id")

        # Step 5: Delete collection (cascade)
        print("\nStep 5: Deleting collection (cascade)...")
        delete_result = collection_manager.delete(collection_id, force=True)
        print(f"  Collection deleted: {delete_result.deleted}")

        # Step 6: Verify no orphan chunks
        print("\nStep 6: Verifying no orphan chunks...")
        chunk_count_after = collection.count()
        print(f"  Collection now has {chunk_count_after} chunks")
        assert chunk_count_after == 0, f"Orphan chunks found: {chunk_count_after}"

        # Step 7: Verify JSON clean
        print("\nStep 7: Verifying JSON storage is clean...")
        collections_in_storage = temp_storage.load(COLLECTIONS_FILE)
        docs_in_storage = temp_storage.load(DOCUMENTS_FILE)
        print(f"  Collections in JSON: {len(collections_in_storage)}")
        print(f"  Documents in JSON: {len(docs_in_storage)}")
        assert len(collections_in_storage) == 0, f"Orphan collections: {collections_in_storage}"
        assert len(docs_in_storage) == 0, f"Orphan documents: {docs_in_storage}"

        print("\n✅ Test PASSED: Document lifecycle with collection")


class TestChromaDBCollectionIsolation:
    """Test that data goes to correct ChromaDB collection."""

    def test_correct_collection_name(self, vector_store, chroma_client):
        """Verify VectorStoreManager uses correct collection name."""
        print("\n=== Test: ChromaDB Collection Isolation ===")

        # Get the collection name from vector store
        collection_name = vector_store.vector_store._collection.name
        print(f"  VectorStoreManager collection: {collection_name}")
        assert collection_name == TEST_COLLECTION_NAME, \
            f"Wrong collection! Expected {TEST_COLLECTION_NAME}, got {collection_name}"

        # List all collections
        all_collections = chroma_client.list_collections()
        collection_names = [c.name for c in all_collections]
        print(f"  All ChromaDB collections: {collection_names}")

        print("\n✅ Test PASSED: Collection isolation verified")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
