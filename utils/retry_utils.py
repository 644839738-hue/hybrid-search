"""
Retry Utilities Module

Provides retry decorators for API calls with exponential backoff.
"""

import logging
from typing import List, Generator, Any
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from openai import RateLimitError, APIConnectionError, APIError
from langchain_core.documents import Document

logger = logging.getLogger(__name__)


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    retry=retry_if_exception_type((RateLimitError, APIConnectionError, APIError)),
    reraise=True
)
def add_documents_with_retry(vector_store: Any, documents: List[Document]) -> List[str]:
    """
    Add documents to vector store with automatic retry logic.

    Uses exponential backoff strategy to handle temporary API issues:
    - Attempt 1: Immediate
    - Attempt 2: Wait 2 seconds
    - Attempt 3: Wait 4 seconds
    - Maximum wait: 10 seconds

    Args:
        vector_store: ChromaDB vector store instance
        documents: List of document chunks to embed and index

    Returns:
        List of document IDs

    Raises:
        RateLimitError: If API rate limit exceeded after all retries
        APIConnectionError: If connection fails after all retries
        APIError: If API error occurs after all retries

    Example:
        >>> from core.vector_store import VectorStoreManager
        >>> manager = VectorStoreManager()
        >>> ids = add_documents_with_retry(manager.vector_store, chunks)
    """
    logger.info("Adding documents to vector store with retry protection...")
    try:
        ids = vector_store.add_documents(documents=documents)
        logger.info(f"Successfully added {len(ids)} documents")
        return ids
    except (RateLimitError, APIConnectionError, APIError) as e:
        logger.warning(f"API error occurred, retrying: {str(e)}")
        raise


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    retry=retry_if_exception_type((RateLimitError, APIConnectionError, APIError)),
    reraise=True
)
def stream_llm_with_retry(llm_model: Any, prompt: Any) -> Generator[Any, None, None]:
    """
    Stream LLM completion with automatic retry logic.

    Uses exponential backoff strategy to handle temporary API issues.
    Retries are automatic for transient errors like rate limits and connection issues.

    Args:
        llm_model: ChatOpenAI model instance
        prompt: Formatted prompt for the model

    Yields:
        Stream chunks from the LLM

    Raises:
        RateLimitError: If API rate limit exceeded after all retries
        APIConnectionError: If connection fails after all retries
        APIError: If API error occurs after all retries

    Example:
        >>> from langchain_openai import ChatOpenAI
        >>> llm = ChatOpenAI(model="gpt-4o-mini")
        >>> for chunk in stream_llm_with_retry(llm, prompt):
        ...     print(chunk.content, end="")
    """
    logger.info("Streaming LLM response with retry protection...")
    try:
        for chunk in llm_model.stream(prompt):
            yield chunk
    except (RateLimitError, APIConnectionError, APIError) as e:
        logger.warning(f"API error during streaming, retrying: {str(e)}")
        raise
