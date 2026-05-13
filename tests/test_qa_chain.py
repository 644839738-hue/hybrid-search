"""
Unit tests for QAChain - Question answering with RAG pipeline.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from core.qa_chain import QAChain
from langchain_core.documents import Document


@pytest.mark.unit
class TestQAChain:
    """Test suite for QAChain class."""

    @patch('core.qa_chain.ChatOpenAI')
    def test_initialization(self, mock_chat_openai, mock_env_vars):
        """Test QAChain initialization with default parameters."""
        qa = QAChain()

        assert qa.model_name == "gpt-4o-mini"
        assert qa.temperature == 0.0
        assert qa.retriever is None
        mock_chat_openai.assert_called_once_with(model="gpt-4o-mini", temperature=0.0)

    @patch('core.qa_chain.ChatOpenAI')
    def test_initialization_custom_params(self, mock_chat_openai, mock_env_vars):
        """Test QAChain initialization with custom parameters."""
        mock_retriever = Mock()
        custom_prompt = "Custom prompt with {question} and {document}"

        qa = QAChain(
            model_name="gpt-4",
            temperature=0.5,
            retriever=mock_retriever,
            system_prompt=custom_prompt
        )

        assert qa.model_name == "gpt-4"
        assert qa.temperature == 0.5
        assert qa.retriever == mock_retriever
        assert qa.system_prompt == custom_prompt

    @patch('core.qa_chain.ChatOpenAI')
    def test_retrieve_context_no_retriever(self, mock_chat_openai, mock_env_vars):
        """Test that retrieving context without retriever raises error."""
        qa = QAChain()

        with pytest.raises(ValueError, match="Retriever not configured"):
            qa.retrieve_context("What is AI?")

    @patch('core.qa_chain.ChatOpenAI')
    def test_retrieve_context_success(self, mock_chat_openai, sample_documents, mock_env_vars):
        """Test successful context retrieval."""
        mock_retriever = Mock()
        mock_retriever.invoke.return_value = sample_documents[:2]

        qa = QAChain(retriever=mock_retriever)
        docs = qa.retrieve_context("What is machine learning?")

        assert len(docs) == 2
        mock_retriever.invoke.assert_called_once_with("What is machine learning?")

    @patch('core.qa_chain.ChatOpenAI')
    def test_format_context(self, mock_chat_openai, sample_documents, mock_env_vars):
        """Test formatting multiple documents into single context."""
        qa = QAChain()
        context = qa.format_context(sample_documents)

        assert isinstance(context, str)
        assert len(context) > 0
        # Should contain content from all documents
        assert "first chunk" in context
        assert "second chunk" in context
        assert "third chunk" in context

    @patch('core.qa_chain.ChatOpenAI')
    def test_generate_answer(self, mock_chat_openai, mock_env_vars):
        """Test non-streaming answer generation."""
        mock_llm = Mock()
        mock_response = Mock()
        mock_response.content = "This is the generated answer."
        mock_llm.invoke.return_value = mock_response
        mock_chat_openai.return_value = mock_llm

        qa = QAChain()
        answer = qa.generate_answer("What is AI?", "AI is artificial intelligence.")

        assert answer == "This is the generated answer."
        mock_llm.invoke.assert_called_once()

    @patch('core.qa_chain.ChatOpenAI')
    def test_stream_answer(self, mock_chat_openai, mock_env_vars):
        """Test streaming answer generation."""
        mock_llm = Mock()

        # Create mock chunks
        chunk1 = Mock()
        chunk1.content = "This is "
        chunk2 = Mock()
        chunk2.content = "a streaming "
        chunk3 = Mock()
        chunk3.content = "answer."

        mock_llm.stream.return_value = [chunk1, chunk2, chunk3]
        mock_chat_openai.return_value = mock_llm

        qa = QAChain()
        chunks = list(qa.stream_answer("What is AI?", "AI is artificial intelligence."))

        assert len(chunks) == 3
        assert "".join(chunks) == "This is a streaming answer."
        mock_llm.stream.assert_called_once()

    @patch('core.qa_chain.ChatOpenAI')
    def test_answer_question_no_documents(self, mock_chat_openai, mock_env_vars):
        """Test that answering with no retrieved documents raises error."""
        mock_retriever = Mock()
        mock_retriever.invoke.return_value = []  # No documents found

        qa = QAChain(retriever=mock_retriever)

        with pytest.raises(ValueError, match="No relevant information found"):
            list(qa.answer_question("What is AI?"))

    @patch('core.qa_chain.ChatOpenAI')
    def test_answer_question_streaming(self, mock_chat_openai, sample_documents, mock_env_vars):
        """Test complete RAG pipeline with streaming."""
        mock_retriever = Mock()
        mock_retriever.invoke.return_value = sample_documents[:2]

        mock_llm = Mock()
        chunk1 = Mock()
        chunk1.content = "Answer "
        chunk2 = Mock()
        chunk2.content = "chunk"
        mock_llm.stream.return_value = [chunk1, chunk2]
        mock_chat_openai.return_value = mock_llm

        qa = QAChain(retriever=mock_retriever)
        chunks = list(qa.answer_question("What is machine learning?", stream=True))

        assert len(chunks) == 2
        assert "".join(chunks) == "Answer chunk"
        mock_retriever.invoke.assert_called_once()

    @patch('core.qa_chain.ChatOpenAI')
    def test_answer_question_non_streaming(self, mock_chat_openai, sample_documents, mock_env_vars):
        """Test complete RAG pipeline without streaming."""
        mock_retriever = Mock()
        mock_retriever.invoke.return_value = sample_documents[:2]

        mock_llm = Mock()
        mock_response = Mock()
        mock_response.content = "Complete answer text"
        mock_llm.invoke.return_value = mock_response
        mock_chat_openai.return_value = mock_llm

        qa = QAChain(retriever=mock_retriever)
        answer = qa.answer_question("What is AI?", stream=False)

        assert answer == "Complete answer text"
        mock_retriever.invoke.assert_called_once()

    @patch('core.qa_chain.ChatOpenAI')
    def test_update_retriever(self, mock_chat_openai, mock_env_vars):
        """Test updating the retriever instance."""
        qa = QAChain()
        assert qa.retriever is None

        new_retriever = Mock()
        qa.update_retriever(new_retriever)

        assert qa.retriever == new_retriever
