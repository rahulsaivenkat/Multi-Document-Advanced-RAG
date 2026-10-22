"""Tests for query transformation and fallback behavior."""
from unittest.mock import MagicMock, patch
from app.schemas import ChatMessage
from app.services.query_transform import QueryTransformer


def test_query_transform_no_history_returns_original():
    transformer = QueryTransformer()
    orig_q = "What is the capital of France?"
    result = transformer.transform(orig_q, conversation_history=None)
    assert result == orig_q

    result_empty_list = transformer.transform(orig_q, conversation_history=[])
    assert result_empty_list == orig_q


def test_query_transform_with_history():
    transformer = QueryTransformer(api_key="valid_mock_key")
    
    history = [
        ChatMessage(role="user", content="Tell me about PaperMind AI's vector database."),
        ChatMessage(role="assistant", content="PaperMind AI uses ChromaDB for persistent vector storage.")
    ]
    follow_up = "What are its limitations?"

    # Mock the Groq client
    with patch("groq.Groq") as mock_groq_cls:
        mock_client = MagicMock()
        mock_groq_cls.return_value = mock_client
        mock_choice = MagicMock()
        mock_choice.message.content = "What are the limitations of ChromaDB in PaperMind AI?"
        mock_client.chat.completions.create.return_value.choices = [mock_choice]

        transformed = transformer.transform(follow_up, conversation_history=history)
        assert transformed == "What are the limitations of ChromaDB in PaperMind AI?"


def test_query_transform_failure_falls_back_to_original():
    transformer = QueryTransformer(api_key="valid_mock_key")
    history = [ChatMessage(role="user", content="Hello")]
    follow_up = "How does it work?"

    with patch("groq.Groq") as mock_groq_cls:
        mock_client = MagicMock()
        mock_groq_cls.return_value = mock_client
        mock_client.chat.completions.create.side_effect = ConnectionError("Groq timeout")

        # Must fall back to original question safely
        result = transformer.transform(follow_up, conversation_history=history)
        assert result == follow_up
