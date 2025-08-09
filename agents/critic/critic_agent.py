

import logging
from typing import Dict, Any, List, Optional
import asyncio
import json

from agents.base_agent import BaseAgent, Message, MessageType
from langchain_ollama import OllamaLLM as Ollama

logger = logging.getLogger(__name__)

class CriticAgent(BaseAgent):
    """Evaluates the quality of a response using Ollama."""

    def __init__(self, agent_id: str = "critic_agent", config: Dict[str, Any] = None):
        super().__init__(agent_id, config)
        self.llm_model = self.system_config.get("llm.models.critic", "llama3.1:8b")
        self.ollama_client = Ollama(base_url=self.system_config.get("llm.ollama.base_url"), model=self.llm_model)

    def get_capabilities(self) -> List[str]:
        """Returns the capabilities of the critic agent."""
        return ["response_evaluation", "quality_assessment", "feedback_generation", "llm_evaluation"]

    async def process_message(self, message: Message) -> Optional[Message]:
        """
        Processes a response from the RetrievalAgent and evaluates its quality.

        Args:
            message: The incoming message from the RetrievalAgent.

        Returns:
            A final response message for the user.
        """
        if message.type != MessageType.RESPONSE:
            return None

        try:
            # 1. Evaluate the response using Ollama (non-blocking option could be added later)
            evaluation = await self._evaluate_with_ollama(message)
            if not evaluation:
                return self._create_error_response("Failed to evaluate the response.", message)

            # 2. Forward the responder's content to user with evaluation attached
            final_response = Message(
                type=MessageType.RESPONSE,
                content=message.content,  # Pass the synthesized content through
                metadata={
                    "original_query": message.metadata.get("original_query"),
                    "retrieved_docs": message.metadata.get("retrieved_docs", []),
                    "evaluation": evaluation,
                    "final_answer_by": self.agent_id,
                },
                sender=self.agent_id,
                recipient="user",  # Final destination is the user
                language=message.language
            )

            self._log_action("response_evaluation", {"evaluation_score": evaluation.get("overall_score", 0)})

            return final_response

        except Exception as e:
            logger.error(f"Error in CriticAgent: {e}")
            return self._create_error_response(f"An unexpected error occurred during evaluation: {e}", message)

    async def _evaluate_with_ollama(self, message: Message) -> Optional[Dict[str, Any]]:
        """
        Uses Ollama to evaluate the response based on the original query and retrieved context.

        Args:
            message: The message from the RetrievalAgent.

        Returns:
            A dictionary containing the evaluation scores and feedback.
        """
        if not self.ollama_client:
            logger.error("Ollama client is not available.")
            return None

        query = message.metadata.get("original_query", "")
        retrieved_docs = message.metadata.get("retrieved_docs", [])
        response_text = message.content

        context = "\n\n".join([f"**Source:** {doc['chunk']['source_file']}\n**Content:** {doc['chunk']['content']}" for doc in retrieved_docs])

        prompt = f"""
You are a strict evaluator. Score the answer ONLY against the provided context. Penalize anything not supported by the context.

User query: {query}

Context:
{context}

Answer:
{response_text}

Return a compact JSON with keys:
relevance_score, accuracy_score, completeness_score, clarity_score, overall_score, feedback.
"""

        try:
            raw_response = await asyncio.to_thread(self.ollama_client.invoke, prompt)
            cleaned = raw_response.strip()
            # If model replied with plain text or non-JSON, attempt to extract JSON substring
            if not cleaned.startswith('{'):
                start = cleaned.find('{')
                end = cleaned.rfind('}')
                if start != -1 and end != -1 and end > start:
                    cleaned = cleaned[start:end+1]
            # Remove code fences if present
            cleaned = cleaned.replace("```json", "").replace("```", "").strip()
            evaluation = json.loads(cleaned)
            logger.info(f"Ollama ({self.llm_model}) evaluated response with score: {evaluation.get('overall_score')}")
            return evaluation
        except Exception as e:
            logger.error(f"Ollama evaluation failed: {e}.")
            return None

    def _create_error_response(self, error_message: str, original_message: Message) -> Message:
        """Creates a standardized error message."""
        return Message(
            type=MessageType.ERROR,
            content=error_message,
            sender=self.agent_id,
            recipient="user",
            metadata={"original_message_id": original_message.id}
        )

