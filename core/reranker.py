"""
Re-ranker Module

Implements document re-ranking using Cohere and Jina models.
Re-ranking improves retrieval quality by scoring document-query pairs
with cross-encoder models that consider both inputs simultaneously.
"""

import os
import logging
from abc import ABC, abstractmethod
from typing import List, Optional, Tuple
from dataclasses import dataclass
from langchain_core.documents import Document

logger = logging.getLogger(__name__)


@dataclass
class RerankResult:
    """
    Result from re-ranking with relevance score.

    Attributes:
        document: The re-ranked document
        score: Relevance score (0-1, higher is more relevant)
        original_rank: Position before re-ranking
        new_rank: Position after re-ranking
    """
    document: Document
    score: float
    original_rank: int
    new_rank: int


class BaseReranker(ABC):
    """
    Abstract base class for document re-rankers.

    Re-rankers use cross-encoder models to score document-query pairs,
    providing more accurate relevance assessment than bi-encoder similarity.
    """

    @abstractmethod
    def rerank(
        self,
        query: str,
        documents: List[Document],
        top_k: int = None
    ) -> List[RerankResult]:
        """
        Re-rank documents based on relevance to query.

        Args:
            query: Search query text
            documents: List of documents to re-rank
            top_k: Number of top documents to return (None = all)

        Returns:
            List of RerankResult sorted by relevance score descending
        """
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """
        Check if the re-ranker is available and configured.

        Returns:
            True if re-ranker can be used, False otherwise
        """
        pass


class CohereReranker(BaseReranker):
    """
    Cohere-based document re-ranker.

    Uses Cohere's rerank API for high-quality cross-encoder re-ranking.
    Requires COHERE_API_KEY environment variable.

    Attributes:
        model: Cohere rerank model name
        client: Cohere client instance

    Example:
        >>> reranker = CohereReranker()
        >>> if reranker.is_available():
        ...     results = reranker.rerank("What is AI?", documents)
    """

    def __init__(self, model: str = "rerank-english-v3.0"):
        """
        Initialize Cohere re-ranker.

        Args:
            model: Cohere rerank model name.
                Options: rerank-english-v3.0, rerank-multilingual-v3.0
        """
        self.model = model
        self._client = None
        self._available = False

        self._initialize_client()

    def _initialize_client(self) -> None:
        """Initialize Cohere client if API key is available."""
        api_key = os.getenv("COHERE_API_KEY")
        if not api_key:
            logger.warning("COHERE_API_KEY not found. Cohere reranker unavailable.")
            return

        try:
            import cohere
            self._client = cohere.Client(api_key)
            self._available = True
            logger.info(f"Cohere reranker initialized with model: {self.model}")
        except ImportError:
            logger.warning("cohere package not installed. Run: pip install cohere")
        except Exception as e:
            logger.error(f"Failed to initialize Cohere client: {e}")

    def is_available(self) -> bool:
        """Check if Cohere re-ranker is available."""
        return self._available and self._client is not None

    def rerank(
        self,
        query: str,
        documents: List[Document],
        top_k: int = None
    ) -> List[RerankResult]:
        """
        Re-rank documents using Cohere's rerank API.

        Args:
            query: Search query text
            documents: List of documents to re-rank
            top_k: Number of top documents to return (default: all)

        Returns:
            List of RerankResult sorted by relevance score descending

        Raises:
            RuntimeError: If Cohere client is not available
        """
        if not self.is_available():
            raise RuntimeError("Cohere reranker not available. Check COHERE_API_KEY.")

        if not documents:
            return []

        # Prepare document texts
        doc_texts = [doc.page_content for doc in documents]

        try:
            # Call Cohere rerank API
            response = self._client.rerank(
                model=self.model,
                query=query,
                documents=doc_texts,
                top_n=top_k or len(documents)
            )

            # Build results with original indices
            results = []
            for new_rank, item in enumerate(response.results, start=1):
                original_idx = item.index
                results.append(RerankResult(
                    document=documents[original_idx],
                    score=item.relevance_score,
                    original_rank=original_idx + 1,
                    new_rank=new_rank
                ))

            logger.info(f"Cohere reranked {len(results)} documents for query: '{query[:50]}...'")
            return results

        except Exception as e:
            logger.error(f"Cohere rerank failed: {e}")
            raise


class JinaReranker(BaseReranker):
    """
    Jina-based document re-ranker using sentence-transformers.

    Uses the Jina reranker model locally via sentence-transformers.
    No API key required but requires model download on first use.

    Attributes:
        model_name: Jina reranker model name
        model: CrossEncoder model instance

    Example:
        >>> reranker = JinaReranker()
        >>> if reranker.is_available():
        ...     results = reranker.rerank("What is AI?", documents)
    """

    def __init__(self, model_name: str = "jinaai/jina-reranker-v1-tiny-en"):
        """
        Initialize Jina re-ranker.

        Args:
            model_name: Jina reranker model name.
                Options:
                - jinaai/jina-reranker-v1-tiny-en (fastest, English only)
                - jinaai/jina-reranker-v1-turbo-en (balanced)
                - jinaai/jina-reranker-v2-base-multilingual (multilingual)
        """
        self.model_name = model_name
        self._model = None
        self._available = False

        self._initialize_model()

    def _initialize_model(self) -> None:
        """Initialize Jina reranker model."""
        try:
            from sentence_transformers import CrossEncoder
            # Load model (downloads on first use)
            self._model = CrossEncoder(
                self.model_name,
                max_length=512,
                trust_remote_code=True
            )
            self._available = True
            logger.info(f"Jina reranker initialized with model: {self.model_name}")
        except ImportError:
            logger.warning("sentence-transformers not installed. Run: pip install sentence-transformers")
        except Exception as e:
            logger.error(f"Failed to load Jina reranker model: {e}")

    def is_available(self) -> bool:
        """Check if Jina re-ranker is available."""
        return self._available and self._model is not None

    def rerank(
        self,
        query: str,
        documents: List[Document],
        top_k: int = None
    ) -> List[RerankResult]:
        """
        Re-rank documents using Jina's cross-encoder model.

        Args:
            query: Search query text
            documents: List of documents to re-rank
            top_k: Number of top documents to return (default: all)

        Returns:
            List of RerankResult sorted by relevance score descending

        Raises:
            RuntimeError: If Jina model is not available
        """
        if not self.is_available():
            raise RuntimeError("Jina reranker not available.")

        if not documents:
            return []

        # Prepare query-document pairs
        pairs = [(query, doc.page_content) for doc in documents]

        try:
            # Get scores from cross-encoder
            scores = self._model.predict(pairs)

            # Create (score, original_index) pairs
            scored_indices = [(float(score), idx) for idx, score in enumerate(scores)]

            # Sort by score descending
            scored_indices.sort(key=lambda x: x[0], reverse=True)

            # Take top_k if specified
            if top_k is not None:
                scored_indices = scored_indices[:top_k]

            # Build results
            results = []
            for new_rank, (score, original_idx) in enumerate(scored_indices, start=1):
                results.append(RerankResult(
                    document=documents[original_idx],
                    score=score,
                    original_rank=original_idx + 1,
                    new_rank=new_rank
                ))

            logger.info(f"Jina reranked {len(results)} documents for query: '{query[:50]}...'")
            return results

        except Exception as e:
            logger.error(f"Jina rerank failed: {e}")
            raise


class RerankerFactory:
    """
    Factory for creating re-ranker instances.

    Provides a unified interface for creating re-rankers based on
    provider name and configuration.

    Example:
        >>> reranker = RerankerFactory.create("cohere")
        >>> if reranker.is_available():
        ...     results = reranker.rerank(query, docs)
    """

    @staticmethod
    def create(
        provider: str = "cohere",
        **kwargs
    ) -> BaseReranker:
        """
        Create a re-ranker instance.

        Args:
            provider: Re-ranker provider ("cohere" or "jina")
            **kwargs: Additional arguments passed to re-ranker constructor

        Returns:
            Configured re-ranker instance

        Raises:
            ValueError: If provider is not supported
        """
        providers = {
            "cohere": CohereReranker,
            "jina": JinaReranker
        }

        if provider.lower() not in providers:
            raise ValueError(
                f"Unknown reranker provider: {provider}. "
                f"Supported: {list(providers.keys())}"
            )

        return providers[provider.lower()](**kwargs)

    @staticmethod
    def get_available_reranker() -> Optional[BaseReranker]:
        """
        Get first available re-ranker.

        Tries Jina first (local, no API cost), then Cohere (cloud).

        Returns:
            Available re-ranker instance or None if none available
        """
        logger.info("🔍 Auto-selecting reranker...")

        # Try Jina first (local, no API cost, no latency)
        jina_reranker = JinaReranker()
        if jina_reranker.is_available():
            logger.info("✅ RERANKER SELECTED: Jina (local)")
            return jina_reranker
        else:
            logger.info("❌ Jina not available (missing sentence-transformers)")

        # Fall back to Cohere (cloud API)
        cohere_reranker = CohereReranker()
        if cohere_reranker.is_available():
            logger.info("✅ RERANKER SELECTED: Cohere (cloud)")
            return cohere_reranker
        else:
            logger.info("❌ Cohere not available (missing COHERE_API_KEY)")

        logger.warning("⚠️ NO RERANKER AVAILABLE. Install sentence-transformers or set COHERE_API_KEY.")
        return None
