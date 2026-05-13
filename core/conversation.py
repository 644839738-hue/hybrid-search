"""
Conversation History Module

Manages session-based conversation storage with local JSON persistence.
Enables context maintenance across queries and follow-up question optimization.
"""

import json
import logging
import os
import uuid
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Dict, Any
from dataclasses import dataclass, field, asdict
from langchain_core.documents import Document

logger = logging.getLogger(__name__)


@dataclass
class ConversationMessage:
    """
    A single message in a conversation.

    Attributes:
        role: Message role ("user" or "assistant")
        content: Message content text
        timestamp: When the message was created
        metadata: Additional message metadata
    """
    role: str  # "user" or "assistant"
    content: str
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class QueryRecord:
    """
    Record of a single query and its response.

    Attributes:
        query_id: Unique identifier for this query
        query: Original user question
        answer: Generated answer
        retrieved_docs: Documents used for context
        retrieval_method: Method used for retrieval
        scores: Relevance scores for retrieved documents
        timestamp: When the query was made
        follow_up_context: Context from previous queries used
    """
    query_id: str
    query: str
    answer: str
    retrieved_docs: List[str] = field(default_factory=list)
    retrieval_method: str = "semantic"
    scores: List[float] = field(default_factory=list)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    follow_up_context: Optional[str] = None


@dataclass
class ConversationSession:
    """
    A conversation session containing multiple queries.

    Attributes:
        session_id: Unique session identifier
        created_at: Session creation timestamp
        updated_at: Last update timestamp
        queries: List of query records in this session
        document_name: Name of the uploaded document
        metadata: Session metadata
    """
    session_id: str
    created_at: str
    updated_at: str
    queries: List[QueryRecord] = field(default_factory=list)
    document_name: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def add_query(self, query_record: QueryRecord) -> None:
        """Add a query record to the session."""
        self.queries.append(query_record)
        self.updated_at = datetime.now().isoformat()

    def get_recent_context(self, n: int = 3) -> str:
        """
        Get context from recent queries for follow-up optimization.

        Args:
            n: Number of recent queries to include

        Returns:
            Formatted context string from recent Q&A pairs
        """
        recent = self.queries[-n:] if len(self.queries) >= n else self.queries
        context_parts = []

        for q in recent:
            context_parts.append(f"Q: {q.query}")
            context_parts.append(f"A: {q.answer[:500]}...")  # Truncate long answers

        return "\n".join(context_parts)


class ConversationManager:
    """
    Manages conversation sessions with JSON persistence.

    Provides session management, query tracking, and context
    maintenance for follow-up question optimization.

    Attributes:
        storage_dir: Directory for JSON storage
        current_session: Active conversation session
        max_history: Maximum queries to keep per session

    Example:
        >>> manager = ConversationManager()
        >>> manager.start_session("document.pdf")
        >>> manager.add_query("What is AI?", "AI is...", docs, scores)
        >>> context = manager.get_follow_up_context()
    """

    def __init__(
        self,
        storage_dir: str = "./conversation_history",
        max_history: int = 50
    ):
        """
        Initialize conversation manager.

        Args:
            storage_dir: Directory path for JSON storage
            max_history: Maximum queries to retain per session
        """
        self.storage_dir = Path(storage_dir)
        self.max_history = max_history
        self.current_session: Optional[ConversationSession] = None

        # Ensure storage directory exists
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"ConversationManager initialized: storage={storage_dir}")

    def start_session(
        self,
        document_name: Optional[str] = None,
        session_id: Optional[str] = None
    ) -> ConversationSession:
        """
        Start a new conversation session.

        Args:
            document_name: Name of the uploaded document
            session_id: Optional specific session ID

        Returns:
            New ConversationSession instance
        """
        now = datetime.now().isoformat()
        self.current_session = ConversationSession(
            session_id=session_id or str(uuid.uuid4()),
            created_at=now,
            updated_at=now,
            document_name=document_name
        )

        logger.info(f"Started new session: {self.current_session.session_id}")
        return self.current_session

    def load_session(self, session_id: str) -> Optional[ConversationSession]:
        """
        Load an existing session from storage.

        Args:
            session_id: Session ID to load

        Returns:
            Loaded ConversationSession or None if not found
        """
        file_path = self.storage_dir / f"{session_id}.json"

        if not file_path.exists():
            logger.warning(f"Session not found: {session_id}")
            return None

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # Reconstruct session object
            queries = [QueryRecord(**q) for q in data.pop('queries', [])]
            session = ConversationSession(**data, queries=queries)
            self.current_session = session

            logger.info(f"Loaded session: {session_id} with {len(queries)} queries")
            return session

        except Exception as e:
            logger.error(f"Failed to load session {session_id}: {e}")
            return None

    def save_session(self) -> bool:
        """
        Save current session to JSON storage.

        Returns:
            True if saved successfully, False otherwise
        """
        if not self.current_session:
            logger.warning("No active session to save")
            return False

        file_path = self.storage_dir / f"{self.current_session.session_id}.json"

        try:
            # Convert to JSON-serializable dict
            session_dict = self._session_to_dict(self.current_session)

            with open(file_path, 'w') as f:
                json.dump(session_dict, f, indent=2)

            logger.info(f"Session saved: {self.current_session.session_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to save session: {e}")
            return False

    def _session_to_dict(self, session: ConversationSession) -> Dict[str, Any]:
        """Convert ConversationSession to JSON-serializable dict."""
        return {
            "session_id": session.session_id,
            "created_at": session.created_at,
            "updated_at": session.updated_at,
            "document_name": session.document_name,
            "metadata": session.metadata,
            "queries": [asdict(q) for q in session.queries]
        }

    def add_query(
        self,
        query: str,
        answer: str,
        retrieved_docs: List[Document] = None,
        scores: List[float] = None,
        retrieval_method: str = "semantic"
    ) -> QueryRecord:
        """
        Add a query record to the current session.

        Args:
            query: User's question
            answer: Generated answer
            retrieved_docs: Documents used for context
            scores: Relevance scores for documents
            retrieval_method: Retrieval method used

        Returns:
            Created QueryRecord

        Raises:
            ValueError: If no active session
        """
        if not self.current_session:
            # Auto-start session if none exists
            self.start_session()

        # Get follow-up context if available
        follow_up_context = None
        if len(self.current_session.queries) > 0:
            follow_up_context = self.current_session.get_recent_context(n=2)

        record = QueryRecord(
            query_id=str(uuid.uuid4()),
            query=query,
            answer=answer,
            retrieved_docs=[doc.page_content[:200] for doc in (retrieved_docs or [])],
            retrieval_method=retrieval_method,
            scores=scores or [],
            follow_up_context=follow_up_context
        )

        self.current_session.add_query(record)

        # Trim history if exceeds max
        if len(self.current_session.queries) > self.max_history:
            self.current_session.queries = self.current_session.queries[-self.max_history:]

        # Auto-save after each query
        self.save_session()

        logger.info(f"Added query to session: {record.query_id}")
        return record

    def get_follow_up_context(self, n: int = 3) -> Optional[str]:
        """
        Get context from recent queries for follow-up questions.

        Args:
            n: Number of recent queries to include

        Returns:
            Formatted context string or None if no history
        """
        if not self.current_session or not self.current_session.queries:
            return None

        return self.current_session.get_recent_context(n)

    def optimize_follow_up_query(
        self,
        query: str,
        include_context: bool = True
    ) -> str:
        """
        Optimize a follow-up query with conversation context.

        Adds context from previous Q&A to help resolve pronouns
        and maintain topic continuity.

        Args:
            query: User's follow-up question
            include_context: Whether to include conversation context

        Returns:
            Optimized query string with context
        """
        if not include_context:
            return query

        context = self.get_follow_up_context(n=2)
        if not context:
            return query

        # Check if query seems like a follow-up (short, uses pronouns)
        follow_up_indicators = ["it", "this", "that", "they", "them", "what about", "how about", "and", "also"]
        query_lower = query.lower()

        is_follow_up = (
            len(query.split()) < 10 or
            any(indicator in query_lower for indicator in follow_up_indicators)
        )

        if is_follow_up:
            optimized = f"""Given this conversation context:
{context}

Current question: {query}

Please answer the current question, using the context to understand any references."""
            logger.info(f"Optimized follow-up query with {len(context)} chars of context")
            return optimized

        return query

    def get_query_history(self, n: int = None) -> List[Dict[str, str]]:
        """
        Get query history for display.

        Args:
            n: Number of recent queries (None = all)

        Returns:
            List of dicts with query/answer pairs
        """
        if not self.current_session:
            return []

        queries = self.current_session.queries
        if n:
            queries = queries[-n:]

        return [
            {
                "query": q.query,
                "answer": q.answer,
                "timestamp": q.timestamp,
                "retrieval_method": q.retrieval_method
            }
            for q in queries
        ]

    def list_sessions(self) -> List[Dict[str, Any]]:
        """
        List all saved sessions.

        Returns:
            List of session summaries
        """
        sessions = []

        for file_path in self.storage_dir.glob("*.json"):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                sessions.append({
                    "session_id": data.get("session_id"),
                    "document_name": data.get("document_name"),
                    "created_at": data.get("created_at"),
                    "updated_at": data.get("updated_at"),
                    "query_count": len(data.get("queries", []))
                })
            except Exception as e:
                logger.warning(f"Failed to read session file {file_path}: {e}")

        # Sort by updated_at descending
        sessions.sort(key=lambda x: x.get("updated_at", ""), reverse=True)
        return sessions

    def delete_session(self, session_id: str) -> bool:
        """
        Delete a session from storage.

        Args:
            session_id: Session ID to delete

        Returns:
            True if deleted, False otherwise
        """
        file_path = self.storage_dir / f"{session_id}.json"

        if not file_path.exists():
            logger.warning(f"Session not found: {session_id}")
            return False

        try:
            file_path.unlink()
            logger.info(f"Deleted session: {session_id}")

            # Clear current session if it was the deleted one
            if self.current_session and self.current_session.session_id == session_id:
                self.current_session = None

            return True

        except Exception as e:
            logger.error(f"Failed to delete session {session_id}: {e}")
            return False

    def clear_current_session(self) -> None:
        """Clear the current session without deleting from storage."""
        if self.current_session:
            logger.info(f"Cleared current session: {self.current_session.session_id}")
        self.current_session = None

    def export_session(self, session_id: str = None) -> Optional[Dict[str, Any]]:
        """
        Export a session as a dictionary for external use.

        Args:
            session_id: Session to export (default: current session)

        Returns:
            Session data as dictionary or None
        """
        if session_id:
            session = self.load_session(session_id)
        else:
            session = self.current_session

        if not session:
            return None

        return self._session_to_dict(session)
