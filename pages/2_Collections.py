"""
Collections Management Page

Manage document collections and their contents.
Provides CRUD operations for collections and documents.
Now includes search functionality within collections.
"""

import time
import streamlit as st
import logging
from datetime import datetime
from typing import Optional

# Must be first Streamlit command
st.set_page_config(
    page_title="Collections - Semantic Search",
    page_icon="📁",
    layout="wide"
)

from dotenv import load_dotenv
load_dotenv()

from config_loader import load_config
from core.collection_manager import CollectionManager
from core.document_manager import DocumentManager
from core.models.collection import CollectionSettings
from core.models.document import DocumentStatus
from core.models.errors import ValidationError, NotFoundError, DuplicateError
from core import (
    DocumentProcessor,
    VectorStoreManager,
    QAChain,
    HybridRetriever,
    RetrievalMethod,
    create_hybrid_retriever
)
from core.reranker import RerankerFactory
from utils import add_documents_with_retry, stream_llm_with_retry
from ui import (
    apply_page_styles,
    render_sidebar_header,
    render_retrieval_settings,
    render_configuration_display,
)

# Configure logging - ensure we see all INFO and above
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Also set document manager logging to INFO
logging.getLogger('core.document_manager').setLevel(logging.INFO)
logging.getLogger('core.storage').setLevel(logging.INFO)

# Load configuration
config = load_config()

# Apply shared page styles (hide nav + base styles)
apply_page_styles()

# Additional page-specific CSS for collection cards
st.markdown("""
    <style>
    /* Collection cards */
    .collection-card {
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid #e0e0e0;
        margin-bottom: 0.5rem;
    }
    .collection-card:hover {
        border-color: #1f77b4;
    }

    /* Document list */
    .doc-row {
        padding: 0.5rem;
        border-bottom: 1px solid #f0f0f0;
    }
    </style>
""", unsafe_allow_html=True)


def initialize_managers():
    """Initialize collection, document, and search managers."""
    if 'collection_manager' not in st.session_state:
        st.session_state.collection_manager = CollectionManager()

    if 'document_manager' not in st.session_state:
        st.session_state.document_manager = DocumentManager()

    if 'selected_collection_id' not in st.session_state:
        st.session_state.selected_collection_id = None

    # Initialize vector store for search
    if 'vector_store_manager' not in st.session_state:
        st.session_state.vector_store_manager = VectorStoreManager(
            embedding_model_name=config.get_embedding_model(),
            collection_name=config.get_collection_name(),
            persist_directory=config.get_persist_directory(),
            use_docker=config.use_chroma_docker(),
            chroma_host=config.get_chroma_host(),
            chroma_port=config.get_chroma_port()
        )

    # Initialize QA chain for search
    if 'qa_chain' not in st.session_state:
        retriever = st.session_state.vector_store_manager.get_retriever(
            search_type=config.get_search_type(),
            search_k=config.get_search_k()
        )
        st.session_state.qa_chain = QAChain(
            model_name=config.get_chat_model(),
            temperature=config.get_chat_temperature(),
            retriever=retriever,
            system_prompt=config.get_qa_system_prompt()
        )

    # Initialize retrieval settings
    if 'current_retrieval_method' not in st.session_state:
        st.session_state.current_retrieval_method = config.get_default_retrieval_method()
    if 'current_preset' not in st.session_state:
        st.session_state.current_preset = config.get_default_preset()
    if 'search_k' not in st.session_state:
        st.session_state.search_k = config.get_search_k()
    if 'hybrid_alpha' not in st.session_state:
        st.session_state.hybrid_alpha = config.get_hybrid_alpha()
    if 'use_reranking' not in st.session_state:
        st.session_state.use_reranking = config.is_reranking_enabled()


def render_sidebar():
    """Render sidebar with retrieval settings."""
    # Branding and navigation (shared component)
    render_sidebar_header()

    # Retrieval settings (shared component)
    render_retrieval_settings(config)

    # Configuration display (shared component)
    render_configuration_display(config)

    # Database Management for Collection Documents (page-specific)
    render_sidebar_database_management()


@st.dialog("Clear All Collection Documents")
def confirm_clear_collection_documents():
    """Confirmation dialog for clearing all collection documents.

    This clears all document chunks that belong to any collection,
    leaving non-collection documents (uploaded on home page) intact.
    """
    collection_count = st.session_state.vector_store_manager.get_collection_documents_count()

    if collection_count == 0:
        st.info("No collection documents to clear.")
        if st.button("Close", use_container_width=True):
            st.rerun()
        return

    st.warning(
        f"⚠️ This will permanently delete **{collection_count}** document chunks "
        "from all collections."
    )
    st.caption("Note: This removes chunks from the vector store. Collection metadata remains in place.")
    st.markdown("Are you sure you want to continue?")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Yes, Clear All", type="primary", use_container_width=True):
            try:
                deleted = st.session_state.vector_store_manager.clear_all_collection_documents()
                logger.info(f"Cleared {deleted} collection document chunks by user")
                st.success(f"Cleared {deleted} document chunks from collections")
                st.rerun()
            except Exception as e:
                st.error(f"Error clearing database: {str(e)}")
                logger.error(f"Error clearing collection documents: {e}", exc_info=True)
    with col2:
        if st.button("Cancel", use_container_width=True):
            st.rerun()


def render_sidebar_database_management():
    """Render database management section for collection documents."""
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Collection Database")

    try:
        # Get stats from collection manager (more reliable than vector store metadata)
        cm = st.session_state.collection_manager
        collections = cm.list(include_stats=True)
        total_chunks = sum(c.chunk_count for c in collections)
        total_docs = sum(c.document_count for c in collections)

        if total_chunks > 0:
            st.sidebar.success(f"**{total_chunks}** chunks indexed ({total_docs} docs)")
        else:
            st.sidebar.info("No collection documents indexed yet.")
    except Exception as e:
        st.sidebar.warning(f"Could not check database status: {str(e)}")
        logger.error(f"Error checking collection database status: {e}")

    if st.sidebar.button("Clear All Collection Documents"):
        confirm_clear_collection_documents()


def format_file_size(size_bytes: int) -> str:
    """Format file size for display."""
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.1f} KB"
    else:
        return f"{size_bytes / (1024 * 1024):.1f} MB"


def format_date(dt: datetime) -> str:
    """Format datetime for display."""
    if isinstance(dt, str):
        dt = datetime.fromisoformat(dt.replace('Z', '+00:00'))
    return dt.strftime("%b %d, %Y %H:%M")


def render_collection_list():
    """Render the list of collections."""
    cm = st.session_state.collection_manager

    st.subheader("Your Collections")

    # Get collections
    response = cm.list(limit=20, include_stats=True)
    collections = response.data

    # Show count with soft limit indicator
    soft_limit = CollectionManager.SOFT_LIMIT_COLLECTIONS
    count_text = f"{len(collections)}/{soft_limit}"
    if len(collections) >= soft_limit:
        st.warning(f"Collection limit reached ({count_text}). Consider consolidating collections.")
    else:
        st.caption(f"Collections: {count_text}")

    if not collections:
        st.info("No collections yet. Create your first collection below!")
        return

    # Render each collection
    for col in collections:
        with st.container():
            col1, col2, col3, col4 = st.columns([3, 1, 1, 1])

            with col1:
                # Collection name and description
                if st.session_state.selected_collection_id == col.id:
                    st.markdown(f"**{col.name}** (selected)")
                else:
                    if st.button(f"{col.name}", key=f"select_{col.id}"):
                        st.session_state.selected_collection_id = col.id
                        st.rerun()

                if col.description:
                    st.caption(col.description)

            with col2:
                st.caption(f"{col.document_count} docs")

            with col3:
                st.caption(f"{col.chunk_count} chunks")

            with col4:
                if st.button("Delete", key=f"del_{col.id}", type="secondary"):
                    st.session_state[f"confirm_delete_{col.id}"] = True

            # Confirmation dialog - outside narrow column for better layout
            if st.session_state.get(f"confirm_delete_{col.id}"):
                st.warning(f"Delete '{col.name}' and all its documents?")
                col_a, col_b, col_c = st.columns([1, 1, 2])
                with col_a:
                    if st.button("Yes", key=f"confirm_yes_{col.id}", type="primary", use_container_width=True):
                        try:
                            cm.delete(col.id, force=True)
                            st.session_state[f"confirm_delete_{col.id}"] = False
                            if st.session_state.selected_collection_id == col.id:
                                st.session_state.selected_collection_id = None
                            st.success(f"Deleted '{col.name}'")
                            st.rerun()
                        except Exception as e:
                            st.error(f"Error: {e}")
                with col_b:
                    if st.button("Cancel", key=f"confirm_no_{col.id}", use_container_width=True):
                        st.session_state[f"confirm_delete_{col.id}"] = False
                        st.rerun()

            st.divider()


def render_create_collection_form():
    """Render form to create a new collection."""
    st.subheader("Create New Collection")

    with st.form("create_collection_form"):
        name = st.text_input(
            "Collection Name",
            placeholder="e.g., Research Papers",
            help="Must be unique"
        )
        description = st.text_area(
            "Description (optional)",
            placeholder="What documents will this collection contain?",
            height=100
        )

        # Advanced settings in expander
        with st.expander("Advanced Settings"):
            chunk_size = st.number_input(
                "Chunk Size",
                min_value=100,
                max_value=4000,
                value=1000,
                help="Number of characters per chunk"
            )
            chunk_overlap = st.number_input(
                "Chunk Overlap",
                min_value=0,
                max_value=500,
                value=200,
                help="Overlap between chunks"
            )

        submitted = st.form_submit_button("Create Collection", type="primary")

        if submitted:
            if not name:
                st.error("Collection name is required")
            else:
                try:
                    cm = st.session_state.collection_manager
                    settings = CollectionSettings(
                        chunk_size=chunk_size,
                        chunk_overlap=chunk_overlap
                    )
                    result = cm.create(
                        name=name.strip(),
                        description=description.strip() if description else None,
                        settings=settings
                    )

                    if result.warnings:
                        for warning in result.warnings:
                            st.warning(warning)

                    st.success(f"Created collection '{result.data.name}'")
                    st.session_state.selected_collection_id = result.data.id
                    st.rerun()

                except DuplicateError as e:
                    st.error(f"A collection with this name already exists")
                except ValidationError as e:
                    st.error(f"Validation error: {e.message}")
                except Exception as e:
                    st.error(f"Error creating collection: {e}")


def render_document_list(collection_id: str):
    """Render documents in the selected collection."""
    dm = st.session_state.document_manager
    cm = st.session_state.collection_manager

    # Get collection info
    try:
        collection = cm.get(collection_id)
    except NotFoundError:
        st.error("Collection not found")
        st.session_state.selected_collection_id = None
        st.rerun()
        return

    st.subheader(f"Documents in: {collection.name}")

    # Get documents
    response = dm.list(collection_id=collection_id, limit=20)
    documents = response.data

    # Show count with soft limit
    soft_limit = DocumentManager.SOFT_LIMIT_DOCUMENTS
    count_text = f"{len(documents)}/{soft_limit}"
    if len(documents) >= soft_limit:
        st.warning(f"Document limit reached ({count_text}). Consider splitting into multiple collections.")
    else:
        st.caption(f"Documents: {count_text}")

    if not documents:
        st.info("No documents in this collection. Upload your first document!")
    else:
        # Document table
        for doc in documents:
            with st.container():
                col1, col2, col3, col4, col5 = st.columns([3, 1, 1, 1, 1])

                with col1:
                    status_icon = {
                        DocumentStatus.PROCESSING: "⏳",
                        DocumentStatus.READY: "✅",
                        DocumentStatus.FAILED: "❌"
                    }.get(doc.status, "📄")
                    st.markdown(f"{status_icon} **{doc.filename}**")

                with col2:
                    st.caption(format_file_size(doc.file_size))

                with col3:
                    st.caption(f"{doc.chunk_count} chunks")

                with col4:
                    st.caption(format_date(doc.uploaded_at))

                with col5:
                    if st.button("Delete", key=f"del_doc_{doc.id}", type="secondary"):
                        try:
                            dm.delete(doc.id)
                            st.success(f"Deleted '{doc.filename}'")
                            st.rerun()
                        except Exception as e:
                            st.error(f"Error: {e}")

                if doc.status == DocumentStatus.FAILED and doc.error_message:
                    st.error(f"Error: {doc.error_message}")

                st.divider()


def render_upload_document(collection_id: str):
    """Render document upload form with clear UX feedback."""
    st.subheader("Upload Documents")

    # Initialize session state for upload tracking
    if 'upload_key' not in st.session_state:
        st.session_state.upload_key = 0
    if 'last_upload_result' not in st.session_state:
        st.session_state.last_upload_result = None

    # Show last upload result if exists
    if st.session_state.last_upload_result:
        result = st.session_state.last_upload_result
        if result['success'] > 0:
            st.success(f"✅ Successfully uploaded {result['success']} file(s)!")
            with st.expander("View uploaded files", expanded=False):
                for name in result['files']:
                    st.write(f"  📄 {name}")
        if result['errors'] > 0:
            st.error(f"❌ {result['errors']} file(s) failed to upload")
        # Clear the result after showing
        if st.button("Clear & Upload More", type="secondary"):
            st.session_state.last_upload_result = None
            st.session_state.upload_key += 1
            st.rerun()
        return  # Don't show uploader while showing results

    # File uploader with unique key to allow reset
    uploaded_files = st.file_uploader(
        "Choose PDF files",
        type=["pdf"],
        help="Select one or more PDF files to upload",
        accept_multiple_files=True,
        key=f"pdf_uploader_{st.session_state.upload_key}"
    )

    if not uploaded_files:
        st.info("📎 Drag and drop PDF files above, or click 'Browse files'")
        return

    # Show selected files
    st.write(f"**{len(uploaded_files)} file(s) selected:**")
    for f in uploaded_files:
        col1, col2 = st.columns([4, 1])
        with col1:
            st.write(f"📄 {f.name}")
        with col2:
            st.caption(format_file_size(f.size))

    st.divider()

    # Upload button
    if st.button(f"⬆️ Upload {len(uploaded_files)} File(s)", type="primary", use_container_width=True):
        dm = st.session_state.document_manager
        cm = st.session_state.collection_manager

        # Get collection settings for chunk size/overlap
        try:
            collection = cm.get(collection_id)
            chunk_size = collection.settings.chunk_size
            chunk_overlap = collection.settings.chunk_overlap
        except Exception:
            chunk_size = config.get_chunk_size()
            chunk_overlap = config.get_chunk_overlap()

        success_count = 0
        error_count = 0
        uploaded_names = []

        # Create a container for progress
        progress_container = st.container()

        with progress_container:
            st.write("**Uploading and processing...**")
            progress_bar = st.progress(0)
            status_text = st.empty()

            for i, uploaded_file in enumerate(uploaded_files):
                # Update progress
                progress_bar.progress((i) / len(uploaded_files))
                status_text.info(f"📤 Processing ({i+1}/{len(uploaded_files)}): {uploaded_file.name}")

                try:
                    file_content = uploaded_file.getvalue()

                    # Step 1: Create document record
                    result = dm.add(
                        collection_id=collection_id,
                        filename=uploaded_file.name,
                        file_content=file_content
                    )
                    document = result.data
                    logger.info(f"Document record created: {document.id}")

                    # Step 2: Process PDF and create chunks
                    status_text.info(f"📄 Extracting text from {uploaded_file.name}...")
                    doc_processor = DocumentProcessor(
                        chunk_size=chunk_size,
                        chunk_overlap=chunk_overlap,
                        add_start_index=config.get_add_start_index(),
                        enable_ocr=config.is_ocr_enabled(),
                        ocr_languages=config.get_ocr_languages(),
                        ocr_dpi=config.get_ocr_dpi(),
                        tesseract_cmd=config.get_tesseract_cmd(),
                        chunking_strategy=config.get_chunking_strategy(),
                        min_chunk_size=config.get_min_chunk_size(),
                        max_chunk_size=config.get_max_chunk_size(),
                        nlp_config=config.get_nlp_chunking_config()
                    )

                    # Reset file position for processing
                    uploaded_file.seek(0)
                    chunks = doc_processor.process_uploaded_file(uploaded_file)
                    logger.info(f"Created {len(chunks)} chunks from {uploaded_file.name}")

                    # Step 3: Add collection_id and document_id to chunk metadata
                    for chunk in chunks:
                        chunk.metadata["collection_id"] = collection_id
                        chunk.metadata["document_id"] = document.id

                    # Step 4: Index chunks in vector store
                    status_text.info(f"🔍 Indexing {len(chunks)} chunks...")
                    chroma_ids = add_documents_with_retry(
                        st.session_state.vector_store_manager.vector_store,
                        chunks
                    )
                    logger.info(f"Indexed {len(chroma_ids)} chunks for document {document.id}")

                    # Step 5: Update document status with chunk count
                    # Get page count from metadata if available
                    page_count = 0
                    if chunks and chunks[0].metadata.get("page"):
                        page_count = max(c.metadata.get("page", 0) for c in chunks)

                    dm.update_status(
                        document_id=document.id,
                        status=DocumentStatus.READY,
                        page_count=page_count,
                        chunk_count=len(chunks)
                    )

                    success_count += 1
                    uploaded_names.append(uploaded_file.name)
                    logger.info(f"Completed: {uploaded_file.name} -> {document.id} ({len(chunks)} chunks)")

                except DuplicateError:
                    status_text.warning(f"⚠️ '{uploaded_file.name}' already exists - skipped")
                    error_count += 1
                except ValidationError as e:
                    status_text.error(f"❌ '{uploaded_file.name}': {e.message}")
                    error_count += 1
                except Exception as e:
                    logger.exception(f"Upload error: {e}")
                    status_text.error(f"❌ '{uploaded_file.name}': {str(e)}")
                    error_count += 1
                    # Mark document as failed if it was created
                    if 'document' in locals() and document:
                        try:
                            dm.update_status(
                                document_id=document.id,
                                status=DocumentStatus.FAILED,
                                error_message=str(e)
                            )
                        except Exception:
                            pass

            # Complete progress
            progress_bar.progress(1.0)
            status_text.empty()

        # Store result and refresh
        st.session_state.last_upload_result = {
            'success': success_count,
            'errors': error_count,
            'files': uploaded_names
        }
        st.session_state.upload_key += 1  # Reset uploader
        st.rerun()


def render_search_tab(collection_id: str, collection):
    """Render search interface for the collection."""
    st.subheader("Search Documents")

    # Check if collection has documents
    if collection.document_count == 0:
        st.info("📄 Upload documents first to enable search.")
        return

    # Initialize search history storage
    if 'collection_search_history' not in st.session_state:
        st.session_state.collection_search_history = {}

    # Get previous search for this collection
    previous_search = st.session_state.collection_search_history.get(collection_id)

    # Question input with clear button
    col1, col2, col3 = st.columns([5, 1, 1])
    with col1:
        question = st.text_input(
            "Your question",
            placeholder="Ask a question about your documents...",
            label_visibility="collapsed",
            key=f"collection_search_input_{collection_id}"
        )
    with col2:
        search_button = st.button("🔍 Search", type="primary", use_container_width=True)
    with col3:
        clear_button = st.button("🗑️ Clear", type="secondary", use_container_width=True,
                                  disabled=previous_search is None)

    # Handle clear button
    if clear_button and previous_search:
        del st.session_state.collection_search_history[collection_id]
        st.rerun()

    # Handle new search
    if search_button and question:
        handle_collection_search(question, collection_id, collection)
    elif search_button and not question:
        st.warning("Please enter a question.")
    # Display previous search results if no new search
    elif previous_search and not search_button:
        display_search_results(previous_search, collection_id)


def handle_collection_search(query: str, collection_id: str, collection):
    """Handle search within a specific collection and store results."""
    st.markdown(f"**Question:** {query}")

    with st.spinner("Searching..."):
        try:
            start_time = time.perf_counter()

            method_map = {
                "semantic": RetrievalMethod.SEMANTIC,
                "bm25": RetrievalMethod.BM25,
                "hybrid": RetrievalMethod.HYBRID
            }
            method = method_map.get(
                st.session_state.current_retrieval_method,
                RetrievalMethod.SEMANTIC
            )

            # Use vector store similarity search with collection filter
            results = st.session_state.vector_store_manager.search_similar(
                query,
                k=st.session_state.search_k,
                filter={"collection_id": collection_id}
            )

            # Apply reranking if enabled
            rerank_scores = {}
            if st.session_state.use_reranking and results:
                reranker = RerankerFactory.get_available_reranker()
                if reranker:
                    logger.info(f"Applying reranking to {len(results)} results")
                    rerank_results = reranker.rerank(query, results, top_k=st.session_state.search_k)
                    # Reorder results based on reranking
                    results = [r.document for r in rerank_results]
                    rerank_scores = {id(r.document): r.score for r in rerank_results}
                    logger.info(f"Reranking complete, top score: {rerank_results[0].score:.3f}")
                else:
                    logger.warning("Reranking enabled but no reranker available")

            retrieval_time = (time.perf_counter() - start_time) * 1000

            if not results:
                st.warning("No relevant information found. Try rephrasing your question or upload more documents.")
                return

            # Store context for display
            context_chunks = []
            for doc in results:
                chunk_data = {
                    "content": doc.page_content,
                    "source": doc.metadata.get("source", "")
                }
                # Add rerank score if available
                if id(doc) in rerank_scores:
                    chunk_data["rerank_score"] = rerank_scores[id(doc)]
                context_chunks.append(chunk_data)

            # Display search results
            rerank_status = "✅ Reranked" if rerank_scores else "No reranking"
            with st.expander(f"Context found ({len(results)} chunks) - {rerank_status}", expanded=False):
                preset_info = f"Profile: **{st.session_state.current_preset}**"
                st.caption(f"{preset_info} | Method: **{method.value}** | Time: {retrieval_time:.0f}ms")

                for i, doc in enumerate(results):
                    # Show rerank score if available
                    score_info = ""
                    if id(doc) in rerank_scores:
                        score_info = f" (Rerank score: {rerank_scores[id(doc)]:.3f})"
                    st.markdown(f"**Chunk {i+1}**{score_info}")
                    if doc.metadata.get("source"):
                        st.caption(f"Source: {doc.metadata.get('source')}")
                    st.code(doc.page_content[:500] + "..." if len(doc.page_content) > 500 else doc.page_content, language=None)

            # Generate answer using QA chain
            context = "\n\n".join([doc.page_content for doc in results])

            st.subheader("Answer:")
            answer_placeholder = st.empty()

            full_answer = ""
            for chunk in stream_llm_with_retry(
                st.session_state.qa_chain.llm_model,
                st.session_state.qa_chain.prompt_template.invoke({
                    "question": query,
                    "document": context
                })
            ):
                full_answer += chunk.content
                answer_placeholder.write(full_answer)

            # Store search results in session state for persistence
            st.session_state.collection_search_history[collection_id] = {
                "query": query,
                "answer": full_answer,
                "context_chunks": context_chunks,
                "method": method.value,
                "retrieval_time": retrieval_time,
                "preset": st.session_state.current_preset,
                "timestamp": time.strftime("%H:%M:%S")
            }

            logger.info(f"Search completed in collection {collection_id}: {len(results)} results")

        except Exception as e:
            st.error(f"Error searching: {str(e)}")
            logger.error(f"Search error in collection {collection_id}: {e}", exc_info=True)


def display_search_results(search_data: dict, collection_id: str):
    """Display previously stored search results.

    Args:
        search_data: Dictionary containing query, answer, context_chunks, etc.
        collection_id: The collection ID (for logging)
    """
    st.caption(f"📝 Previous search from {search_data.get('timestamp', 'earlier')}")

    # Display the question
    st.markdown(f"**Question:** {search_data['query']}")

    # Display context chunks
    context_chunks = search_data.get('context_chunks', [])
    if context_chunks:
        with st.expander(f"Context found ({len(context_chunks)} chunks)", expanded=False):
            preset_info = f"Profile: **{search_data.get('preset', 'unknown')}**"
            st.caption(f"{preset_info} | Method: **{search_data.get('method', 'unknown')}** | Time: {search_data.get('retrieval_time', 0):.0f}ms")

            for i, chunk in enumerate(context_chunks):
                st.markdown(f"**Chunk {i+1}**")
                if chunk.get("source"):
                    st.caption(f"Source: {chunk['source']}")
                content = chunk.get("content", "")
                st.code(content[:500] + "..." if len(content) > 500 else content, language=None)

    # Display the answer
    st.subheader("Answer:")
    st.write(search_data.get('answer', 'No answer stored'))


def render_collection_details(collection_id: str):
    """Render details for selected collection."""
    cm = st.session_state.collection_manager

    try:
        collection = cm.get(collection_id, expand=["stats"])
    except NotFoundError:
        st.error("Collection not found")
        st.session_state.selected_collection_id = None
        return

    # Header with back button
    col1, col2 = st.columns([4, 1])
    with col1:
        st.title(f"📁 {collection.name}")
    with col2:
        if st.button("← Back to list"):
            st.session_state.selected_collection_id = None
            st.rerun()

    if collection.description:
        st.caption(collection.description)

    # Stats
    stat_cols = st.columns(4)
    with stat_cols[0]:
        st.metric("Documents", collection.document_count)
    with stat_cols[1]:
        st.metric("Chunks", collection.chunk_count)
    with stat_cols[2]:
        st.metric("Chunk Size", collection.settings.chunk_size)
    with stat_cols[3]:
        st.metric("Overlap", collection.settings.chunk_overlap)

    st.divider()

    # Tabs for search, documents, and upload
    tab1, tab2, tab3 = st.tabs(["🔍 Search", "📄 Documents", "⬆️ Upload"])

    with tab1:
        render_search_tab(collection_id, collection)

    with tab2:
        render_document_list(collection_id)

    with tab3:
        render_upload_document(collection_id)


def main():
    """Main page function."""
    initialize_managers()

    # Render sidebar with retrieval settings
    render_sidebar()

    # Main content
    st.title("Search: Document Collections")
    st.caption("Organize your documents into searchable collections")

    # If a collection is selected, show its details
    if st.session_state.selected_collection_id:
        render_collection_details(st.session_state.selected_collection_id)
    else:
        # Show collection list and create form
        col1, col2 = st.columns([2, 1])

        with col1:
            render_collection_list()

        with col2:
            render_create_collection_form()


if __name__ == "__main__":
    main()
