"""
Core Module for Semantic Search

This module contains the core functionality for document processing,
vector storage, question-answering, hybrid retrieval, re-ranking,
conversation history, and A/B testing.

Imports are lazy to avoid issues during testing when dependencies
may not be available.
"""


def __getattr__(name):
    """Lazy import pattern to avoid loading heavy dependencies at module import."""

    # Document processing
    if name == 'DocumentProcessor':
        from .document_processor import DocumentProcessor
        return DocumentProcessor

    # Vector store
    if name == 'VectorStoreManager':
        from .vector_store import VectorStoreManager
        return VectorStoreManager

    # QA Chain
    if name == 'QAChain':
        from .qa_chain import QAChain
        return QAChain

    # BM25 Retriever
    if name == 'BM25Retriever':
        from .bm25_retriever import BM25Retriever
        return BM25Retriever
    if name == 'BM25Result':
        from .bm25_retriever import BM25Result
        return BM25Result

    # Rerankers
    if name in ('BaseReranker', 'CohereReranker', 'JinaReranker', 'RerankerFactory', 'RerankResult'):
        from .reranker import BaseReranker, CohereReranker, JinaReranker, RerankerFactory, RerankResult
        return locals()[name]

    # Hybrid Retriever
    if name in ('HybridRetriever', 'HybridResult', 'RetrievalMethod', 'create_hybrid_retriever'):
        from .hybrid_retriever import HybridRetriever, HybridResult, RetrievalMethod, create_hybrid_retriever
        return locals()[name]

    # Conversation
    if name in ('ConversationManager', 'ConversationSession', 'ConversationMessage', 'QueryRecord'):
        from .conversation import ConversationManager, ConversationSession, ConversationMessage, QueryRecord
        return locals()[name]

    # A/B Testing
    if name in ('ABTestingManager', 'ABTestExperiment', 'ABTestResult', 'TestVariant', 'RetrievalMetrics'):
        from .ab_testing import ABTestingManager, ABTestExperiment, ABTestResult, TestVariant, RetrievalMetrics
        return locals()[name]

    # Storage (new in Stage 1)
    if name == 'JSONStorage':
        from .storage import JSONStorage
        return JSONStorage

    raise AttributeError(f"module 'core' has no attribute '{name}'")

__all__ = [
    # Original exports
    'DocumentProcessor',
    'VectorStoreManager',
    'QAChain',
    # BM25 Retriever
    'BM25Retriever',
    'BM25Result',
    # Re-rankers
    'BaseReranker',
    'CohereReranker',
    'JinaReranker',
    'RerankerFactory',
    'RerankResult',
    # Hybrid Retriever
    'HybridRetriever',
    'HybridResult',
    'RetrievalMethod',
    'create_hybrid_retriever',
    # Conversation History
    'ConversationManager',
    'ConversationSession',
    'ConversationMessage',
    'QueryRecord',
    # A/B Testing
    'ABTestingManager',
    'ABTestExperiment',
    'ABTestResult',
    'TestVariant',
    'RetrievalMetrics',
    # Storage (Stage 1)
    'JSONStorage',
]
