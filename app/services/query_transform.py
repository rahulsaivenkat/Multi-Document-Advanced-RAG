"""Query transformation service rewriting context-dependent questions for optimal retrieval."""
import logging
from typing import List, Optional
from app.config import settings
from app.schemas import ChatMessage

logger = logging.getLogger(__name__)


class QueryTransformer:
    """Uses Groq LLM to rewrite ambiguous or conversational questions into standalone retrieval queries."""

    TRANSFORM_PROMPT = """You are an expert search query reformulation engine for a multi-format RAG system.
Your job is to transform a user's follow-up or conversational question into a single, specific, self-contained search query.

Guidelines:
1. If the question is already clear, specific, and self-contained, return it unchanged.
2. If the question relies on previous conversation turns (e.g., pronouns like 'it', 'they', 'that function', 'the second file'), resolve those references using the provided conversation history into explicit names, terms, or concepts.
3. NEVER answer the question itself.
4. NEVER invent document contents, facts, or filenames not mentioned in the conversation.
5. Return ONLY the rewritten query text. Do not add explanations, conversational filler, or quotes.

Conversation History:
{history}

Current Question: {question}

Transformed Retrieval Query:"""

    def __init__(self, api_key: Optional[str] = settings.groq_api_key, model: str = settings.groq_model):
        self.api_key = api_key
        self.model = model

    def transform(
        self,
        question: str,
        conversation_history: Optional[List[ChatMessage]] = None
    ) -> str:
        """Transform question into standalone query if history is provided and API is available.
        
        Returns the transformed query, or the original question as safe fallback.
        """
        # If no history provided or query is trivial, fallback to original question
        if not conversation_history or len(conversation_history) == 0:
            return question.strip()

        # If Groq API key is not configured, fallback safely
        if not self.api_key or self.api_key == "your_groq_api_key_here":
            logger.debug("Groq API key not configured. Using original query.")
            return question.strip()

        try:
            from groq import Groq

            client = Groq(api_key=self.api_key, timeout=10.0)
            
            # Format history (take up to last 4 turns for context)
            history_text = "\n".join(
                f"{msg.role.capitalize()}: {msg.content}" for msg in conversation_history[-4:]
            )

            prompt = self.TRANSFORM_PROMPT.format(
                history=history_text,
                question=question.strip()
            )

            response = client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a precise query reformulation tool."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.0,
                max_tokens=150
            )

            transformed = response.choices[0].message.content.strip()
            # Sanity check: do not accept empty or bizarre responses
            if transformed and len(transformed) > 3 and not transformed.lower().startswith("error"):
                logger.info(f"Transformed query: '{question}' -> '{transformed}'")
                return transformed
            return question.strip()

        except Exception as e:
            logger.warning(f"Query transformation failed ({e}). Falling back to original question.")
            return question.strip()
