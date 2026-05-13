"""
BM25 Retriever Module

Implements BM25 (Best Matching 25) algorithm for keyword-based document retrieval.
BM25 is a bag-of-words retrieval function that ranks documents based on term frequency
and inverse document frequency.
"""

import logging
import re
from typing import List, Optional, Tuple
from dataclasses import dataclass
from rank_bm25 import BM25Okapi
from langchain_core.documents import Document

logger = logging.getLogger(__name__)


@dataclass
class BM25Result:
    """
    Result from BM25 retrieval with score.

    Attributes:
        document: The retrieved document
        score: BM25 relevance score (higher is more relevant)
        rank: Position in the result list (1-indexed)
    """
    document: Document
    score: float
    rank: int


class BM25Retriever:
    """
    BM25-based document retriever for keyword matching.

    This retriever uses the BM25 algorithm to find documents that match
    query terms based on term frequency and inverse document frequency.
    It complements semantic search by capturing exact keyword matches
    that embedding-based search might miss.

    Attributes:
        documents: List of indexed documents
        bm25: BM25Okapi instance for scoring
        tokenized_corpus: Tokenized version of all documents

    Example:
        >>> retriever = BM25Retriever()
        >>> retriever.index_documents(chunks)
        >>> results = retriever.retrieve("machine learning", k=5)
        >>> for result in results:
        ...     print(f"Score: {result.score:.3f} - {result.document.page_content[:50]}")
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        """
        Initialize BM25 retriever with tuning parameters.

        Args:
            k1: Term frequency saturation parameter (default 1.5).
                Higher values increase the impact of term frequency.
            b: Length normalization parameter (default 0.75).
                0 = no length normalization, 1 = full normalization.
        """
        self.k1 = k1
        self.b = b
        self.documents: List[Document] = []
        self.bm25: Optional[BM25Okapi] = None
        self.tokenized_corpus: List[List[str]] = []

        logger.info(f"BM25Retriever initialized with k1={k1}, b={b}")

    def _tokenize(self, text: str) -> List[str]:
        """
        Tokenize text into lowercase words.

        Performs simple whitespace tokenization with lowercasing
        and removal of non-alphanumeric characters.

        Args:
            text: Input text to tokenize

        Returns:
            List of lowercase tokens
        """
        # Convert to lowercase and extract alphanumeric tokens
        text = text.lower()
        tokens = re.findall(r'\b[a-z0-9]+\b', text)
        return tokens

    def index_documents(self, documents: List[Document]) -> int:
        """
        Index documents for BM25 retrieval.

        Tokenizes all documents and builds the BM25 index.
        This must be called before retrieval.

        Args:
            documents: List of LangChain Document objects to index

        Returns:
            Number of documents indexed

        Raises:
            ValueError: If documents list is empty

        Example:
            >>> retriever.index_documents(chunks)
            >>> print(f"Indexed {len(chunks)} documents")
        """
        if not documents:
            raise ValueError("Cannot index empty document list")

        self.documents = documents
        self.tokenized_corpus = [
            self._tokenize(doc.page_content)
            for doc in documents
        ]

        # Build BM25 index with custom parameters
        self.bm25 = BM25Okapi(
            self.tokenized_corpus,
            k1=self.k1,
            b=self.b
        )

        logger.info(f"Indexed {len(documents)} documents for BM25 retrieval")
        return len(documents)

    def retrieve(
        self,
        query: str,
        k: int = 3,
        score_threshold: float = 0.0
    ) -> List[BM25Result]:
        """
        Retrieve relevant documents using BM25 scoring.

        Args:
            query: Search query text
            k: Maximum number of documents to return
            score_threshold: Minimum score for inclusion (default 0.0)

        Returns:
            List of BM25Result objects sorted by relevance score

        Raises:
            ValueError: If index is not built (call index_documents first)

        Example:
            >>> results = retriever.retrieve("neural networks", k=5)
            >>> for r in results:
            ...     print(f"[{r.rank}] Score: {r.score:.3f}")
        """
        if self.bm25 is None:
            raise ValueError("Index not built. Call index_documents() first.")

        # Tokenize query
        query_tokens = self._tokenize(query)

        if not query_tokens:
            logger.warning(f"Query tokenization produced no tokens: '{query}'")
            return []

        # Get BM25 scores for all documents
        raw_scores = self.bm25.get_scores(query_tokens)

        # Normalize BM25 scores to 0-1 range using min-max scaling
        # This is critical for combining with semantic scores (already 0-1)
        # and for meaningful percentage display in the UI
        scores = self._normalize_scores(raw_scores)

        # Create (score, index) pairs and sort by score descending
        scored_indices = [(score, idx) for idx, score in enumerate(scores)]
        scored_indices.sort(key=lambda x: x[0], reverse=True)

        # Filter by threshold and take top k
        results = []
        for rank, (score, idx) in enumerate(scored_indices[:k], start=1):
            if score >= score_threshold:
                results.append(BM25Result(
                    document=self.documents[idx],
                    score=score,  # Now normalized to 0-1
                    rank=rank
                ))

        logger.info(f"BM25 retrieved {len(results)} documents for query: '{query[:50]}...'")
        return results

    def retrieve_with_scores(
        self,
        query: str,
        k: int = 3
    ) -> List[Tuple[Document, float]]:
        """
        Retrieve documents with scores in LangChain-compatible format.

        Args:
            query: Search query text
            k: Maximum number of documents to return

        Returns:
            List of (Document, score) tuples

        Example:
            >>> docs_with_scores = retriever.retrieve_with_scores("AI", k=3)
            >>> for doc, score in docs_with_scores:
            ...     print(f"Score: {score:.3f}")
        """
        results = self.retrieve(query, k)
        return [(r.document, r.score) for r in results]

    def get_top_documents(self, query: str, k: int = 3) -> List[Document]:
        """
        Get top k documents without scores.

        Convenience method that returns just the documents
        without score information.

        Args:
            query: Search query text
            k: Maximum number of documents to return

        Returns:
            List of Document objects
        """
        results = self.retrieve(query, k)
        return [r.document for r in results]

    def is_indexed(self) -> bool:
        """
        Check if documents have been indexed.

        Returns:
            True if index is built, False otherwise
        """
        return self.bm25 is not None and len(self.documents) > 0

    def get_document_count(self) -> int:
        """
        Get number of indexed documents.

        Returns:
            Number of documents in the index
        """
        return len(self.documents)

    def clear_index(self) -> None:
        """
        Clear the BM25 index.

        Removes all indexed documents and resets the retriever.
        """
        self.documents = []
        self.tokenized_corpus = []
        self.bm25 = None
        logger.info("BM25 index cleared")

    def _normalize_scores(self, scores) -> List[float]:
        """
        Normalize BM25 scores to 0-1 range using min-max scaling.

        BM25 raw scores are unbounded (can be 0 to 20+) which makes them
        incompatible with semantic scores (0-1 cosine similarity) and
        unusable as percentages in the UI.

        Min-max normalization: normalized = (score - min) / (max - min)

        Args:
            scores: Array or list of raw BM25 scores (typically numpy array from BM25Okapi)

        Returns:
            List of normalized scores in range [0, 1]

        Example:
            >>> raw = [0.0, 5.0, 10.0, 15.0]
            >>> normalized = self._normalize_scores(raw)
            >>> # Returns [0.0, 0.333, 0.666, 1.0]
        """
        # Handle empty array - use len() instead of truthiness check
        # because NumPy arrays raise ValueError on `if not array:`
        if len(scores) == 0:
            return []

        # Convert to Python floats for reliable min/max operations
        # (handles both NumPy arrays and Python lists)
        scores_list = [float(s) for s in scores]

        min_score = min(scores_list)
        max_score = max(scores_list)

        # Handle edge case where all scores are the same
        if max_score == min_score:
            # If all scores are 0, return all 0s
            # Otherwise, return all 1s (they're all equally good)
            return [0.0 if max_score == 0 else 1.0 for _ in scores_list]

        # Apply min-max normalization
        score_range = max_score - min_score
        normalized = [(s - min_score) / score_range for s in scores_list]

        logger.debug(
            f"BM25 score normalization: raw range [{min_score:.3f}, {max_score:.3f}] "
            f"-> normalized [0.0, 1.0]"
        )

        return normalized
