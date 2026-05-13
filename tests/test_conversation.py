"""
Tests for Conversation History module.
"""

import pytest
import os
import json
from core.conversation import (
    ConversationManager,
    ConversationSession,
    ConversationMessage,
    QueryRecord
)


class TestQueryRecord:
    """Test cases for QueryRecord dataclass."""

    def test_query_record_creation(self):
        """Test creating a query record."""
        record = QueryRecord(
            query_id="test-123",
            query="What is machine learning?",
            answer="Machine learning is..."
        )

        assert record.query_id == "test-123"
        assert record.query == "What is machine learning?"
        assert record.answer == "Machine learning is..."
        assert record.retrieval_method == "semantic"  # default

    def test_query_record_with_all_fields(self):
        """Test creating a query record with all fields."""
        record = QueryRecord(
            query_id="test-456",
            query="What is AI?",
            answer="AI is...",
            retrieved_docs=["doc1", "doc2"],
            retrieval_method="hybrid",
            scores=[0.9, 0.8],
            follow_up_context="Previous context..."
        )

        assert len(record.retrieved_docs) == 2
        assert record.retrieval_method == "hybrid"
        assert len(record.scores) == 2


class TestConversationSession:
    """Test cases for ConversationSession dataclass."""

    def test_session_creation(self):
        """Test creating a conversation session."""
        session = ConversationSession(
            session_id="session-123",
            created_at="2024-01-01T00:00:00",
            updated_at="2024-01-01T00:00:00"
        )

        assert session.session_id == "session-123"
        assert len(session.queries) == 0

    def test_add_query(self):
        """Test adding a query to session."""
        session = ConversationSession(
            session_id="session-123",
            created_at="2024-01-01T00:00:00",
            updated_at="2024-01-01T00:00:00"
        )

        record = QueryRecord(
            query_id="q1",
            query="Test?",
            answer="Test answer"
        )

        session.add_query(record)

        assert len(session.queries) == 1
        assert session.queries[0].query_id == "q1"

    def test_get_recent_context(self):
        """Test getting recent context from session."""
        session = ConversationSession(
            session_id="session-123",
            created_at="2024-01-01T00:00:00",
            updated_at="2024-01-01T00:00:00"
        )

        # Add multiple queries
        for i in range(5):
            session.add_query(QueryRecord(
                query_id=f"q{i}",
                query=f"Question {i}?",
                answer=f"Answer {i}"
            ))

        context = session.get_recent_context(n=2)

        assert "Question 3?" in context
        assert "Question 4?" in context
        assert "Question 0?" not in context


class TestConversationManager:
    """Test cases for ConversationManager class."""

    def test_initialization(self, temp_conversation_dir):
        """Test conversation manager initialization."""
        manager = ConversationManager(storage_dir=temp_conversation_dir)

        assert manager.storage_dir.exists()
        assert manager.current_session is None

    def test_start_session(self, temp_conversation_dir):
        """Test starting a new session."""
        manager = ConversationManager(storage_dir=temp_conversation_dir)
        session = manager.start_session(document_name="test.pdf")

        assert session is not None
        assert session.document_name == "test.pdf"
        assert manager.current_session == session

    def test_start_session_with_id(self, temp_conversation_dir):
        """Test starting session with specific ID."""
        manager = ConversationManager(storage_dir=temp_conversation_dir)
        session = manager.start_session(session_id="custom-id-123")

        assert session.session_id == "custom-id-123"

    def test_add_query(self, temp_conversation_dir, sample_documents):
        """Test adding a query to session."""
        manager = ConversationManager(storage_dir=temp_conversation_dir)
        manager.start_session()

        record = manager.add_query(
            query="What is AI?",
            answer="AI is artificial intelligence.",
            retrieved_docs=sample_documents[:2],
            scores=[0.9, 0.8],
            retrieval_method="hybrid"
        )

        assert record is not None
        assert record.query == "What is AI?"
        assert len(manager.current_session.queries) == 1

    def test_add_query_auto_starts_session(self, temp_conversation_dir):
        """Test that add_query auto-starts session if none exists."""
        manager = ConversationManager(storage_dir=temp_conversation_dir)

        # Don't call start_session
        record = manager.add_query(
            query="Test?",
            answer="Test answer"
        )

        assert manager.current_session is not None
        assert record is not None

    def test_save_and_load_session(self, temp_conversation_dir):
        """Test saving and loading session."""
        manager = ConversationManager(storage_dir=temp_conversation_dir)
        session = manager.start_session(document_name="test.pdf")
        session_id = session.session_id

        manager.add_query("Q1?", "A1")
        manager.add_query("Q2?", "A2")
        manager.save_session()

        # Create new manager and load session
        manager2 = ConversationManager(storage_dir=temp_conversation_dir)
        loaded = manager2.load_session(session_id)

        assert loaded is not None
        assert loaded.session_id == session_id
        assert len(loaded.queries) == 2

    def test_load_nonexistent_session(self, temp_conversation_dir):
        """Test loading non-existent session returns None."""
        manager = ConversationManager(storage_dir=temp_conversation_dir)
        result = manager.load_session("nonexistent-id")

        assert result is None

    def test_get_follow_up_context(self, temp_conversation_dir):
        """Test getting follow-up context."""
        manager = ConversationManager(storage_dir=temp_conversation_dir)
        manager.start_session()

        manager.add_query("What is ML?", "ML is machine learning.")
        manager.add_query("How does it work?", "It learns from data.")

        context = manager.get_follow_up_context(n=2)

        assert context is not None
        assert "What is ML?" in context
        assert "How does it work?" in context

    def test_get_follow_up_context_empty_session(self, temp_conversation_dir):
        """Test getting context from empty session returns None."""
        manager = ConversationManager(storage_dir=temp_conversation_dir)
        manager.start_session()

        context = manager.get_follow_up_context()
        assert context is None

    def test_optimize_follow_up_query(self, temp_conversation_dir):
        """Test query optimization for follow-ups."""
        manager = ConversationManager(storage_dir=temp_conversation_dir)
        manager.start_session()
        manager.add_query("What is neural networks?", "Neural networks are...")

        # Short query with pronoun - should be optimized
        optimized = manager.optimize_follow_up_query("How do they work?")

        assert "they" in optimized.lower() or "neural" in optimized.lower()

    def test_optimize_regular_query(self, temp_conversation_dir):
        """Test that regular queries are not overly modified."""
        manager = ConversationManager(storage_dir=temp_conversation_dir)
        manager.start_session()

        # Long, specific query - should be returned mostly as-is
        query = "Explain the architecture of convolutional neural networks used in image classification"
        optimized = manager.optimize_follow_up_query(query, include_context=False)

        assert optimized == query

    def test_get_query_history(self, temp_conversation_dir):
        """Test getting query history."""
        manager = ConversationManager(storage_dir=temp_conversation_dir)
        manager.start_session()

        manager.add_query("Q1?", "A1", retrieval_method="semantic")
        manager.add_query("Q2?", "A2", retrieval_method="hybrid")

        history = manager.get_query_history(n=5)

        assert len(history) == 2
        assert history[0]["query"] == "Q1?"
        assert history[1]["retrieval_method"] == "hybrid"

    def test_list_sessions(self, temp_conversation_dir):
        """Test listing all sessions."""
        manager = ConversationManager(storage_dir=temp_conversation_dir)

        # Create multiple sessions
        manager.start_session(document_name="doc1.pdf")
        manager.add_query("Q1?", "A1")

        manager.start_session(document_name="doc2.pdf")
        manager.add_query("Q2?", "A2")

        sessions = manager.list_sessions()

        assert len(sessions) == 2

    def test_delete_session(self, temp_conversation_dir):
        """Test deleting a session."""
        manager = ConversationManager(storage_dir=temp_conversation_dir)
        session = manager.start_session()
        session_id = session.session_id
        manager.add_query("Q?", "A")
        manager.save_session()

        # Delete the session
        result = manager.delete_session(session_id)

        assert result is True
        assert manager.current_session is None

        # Verify file is deleted
        sessions = manager.list_sessions()
        assert len(sessions) == 0

    def test_delete_nonexistent_session(self, temp_conversation_dir):
        """Test deleting non-existent session returns False."""
        manager = ConversationManager(storage_dir=temp_conversation_dir)
        result = manager.delete_session("nonexistent-id")

        assert result is False

    def test_clear_current_session(self, temp_conversation_dir):
        """Test clearing current session."""
        manager = ConversationManager(storage_dir=temp_conversation_dir)
        manager.start_session()

        manager.clear_current_session()

        assert manager.current_session is None

    def test_export_session(self, temp_conversation_dir):
        """Test exporting session as dictionary."""
        manager = ConversationManager(storage_dir=temp_conversation_dir)
        manager.start_session(document_name="test.pdf")
        manager.add_query("Q?", "A")

        exported = manager.export_session()

        assert exported is not None
        assert "session_id" in exported
        assert "queries" in exported
        assert exported["document_name"] == "test.pdf"

    def test_max_history_trimming(self, temp_conversation_dir):
        """Test that history is trimmed to max_history."""
        manager = ConversationManager(
            storage_dir=temp_conversation_dir,
            max_history=5
        )
        manager.start_session()

        # Add more queries than max_history
        for i in range(10):
            manager.add_query(f"Q{i}?", f"A{i}")

        assert len(manager.current_session.queries) == 5
