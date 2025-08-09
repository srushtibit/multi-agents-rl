
import logging
from typing import Dict, Any, List, Optional
import asyncio

from agents.base_agent import BaseAgent, Message, MessageType
from langchain_ollama import OllamaLLM as Ollama
from utils.language_utils import detect_language, translate_to_english

logger = logging.getLogger(__name__)

class CommunicationAgent(BaseAgent):
    """Handles initial user interaction, query analysis, and rephrasing using Ollama."""

    def __init__(self, agent_id: str = "communication_agent", config: Dict[str, Any] = None):
        super().__init__(agent_id, config)
        self.llm_model = self.system_config.get("llm.models.communication", "llama3.1:8b")
        self.ollama_client = Ollama(base_url=self.system_config.get("llm.ollama.base_url"), model=self.llm_model)

    def get_capabilities(self) -> List[str]:
        """Returns the capabilities of the communication agent."""
        return [
            "query_analysis",
            "multilingual_understanding",
            "intent_clarification",
            "llm_interaction"
        ]

    async def process_message(self, message: Message) -> Optional[Message]:
        """
        Processes the user's query by analyzing and rephrasing it using Ollama.

        Args:
            message: The incoming message from the user.

        Returns:
            A new message for the RetrievalAgent with the analyzed query.
        """
        # Handle response from RetrievalAgent: forward to user and send to Critic for scoring
        if message.type == MessageType.RESPONSE and message.sender == "retrieval_agent":
            try:
                # Forward to critic for evaluation (async via coordinator)
                self.send_message(
                    recipient="critic_agent",
                    content=message.content,
                    message_type=MessageType.RESPONSE,
                    metadata={
                        "original_query": message.metadata.get("original_query", ""),
                        "retrieved_docs": message.metadata.get("retrieved_docs", [])
                    }
                )

                # Forward final answer to user
                return Message(
                    type=MessageType.RESPONSE,
                    content=message.content,
                    metadata={
                        "original_query": message.metadata.get("original_query", ""),
                        "retrieved_docs": message.metadata.get("retrieved_docs", []),
                        "final_answer_by": self.agent_id
                    },
                    sender=self.agent_id,
                    recipient="user",
                    language=message.language
                )
            except Exception as e:
                logger.error(f"Error forwarding retrieval response: {e}")
                return None

        if message.type != MessageType.QUERY:
            return None

        # Enhanced greeting / non-technical small-talk detection
        if self._is_non_technical_query(message.content):
            return Message(
                type=MessageType.RESPONSE,
                content=self._generate_friendly_response(message.content),
                sender=self.agent_id,
                recipient="user",
            )

        try:
            # 1. Detect language and translate if necessary
            lang_result = detect_language(message.content)
            query_text = message.content
            if lang_result.language != 'en':
                translation = translate_to_english(query_text, lang_result.language)
                if translation.confidence > 0.7:
                    query_text = translation.translated_text
                    logger.info(f"Translated query from {lang_result.language} to English.")

            # 2. Lightweight query normalization via LLM only when helpful
            # If the query is short and technical keywords present, skip LLM rewrite to reduce latency
            analyzed_query = query_text
            tech_keywords = ["zoom", "vpn", "email", "outlook", "account", "login", "password", "firewall", "network", "screen share"]
            if not any(k in query_text.lower() for k in tech_keywords) or len(query_text.split()) > 4:
                llm_rewrite = await self._analyze_with_ollama(query_text)
                if llm_rewrite:
                    analyzed_query = llm_rewrite

            # 3. Route to RetrievalAgent
            retrieval_message = Message(
                type=MessageType.QUERY,
                content=analyzed_query,
                metadata={
                    "original_query": message.content,
                    "original_language": lang_result.language,
                    "analysis_by": self.agent_id,
                },
                sender=self.agent_id,
                recipient="retrieval_agent",
                language='en'
            )

            self._log_action("query_analysis", {
                "original_query": message.content,
                "analyzed_query": analyzed_query,
                "model": self.llm_model
            })

            return retrieval_message

        except Exception as e:
            logger.error(f"Error in CommunicationAgent: {e}")
            return self._create_error_response(f"An unexpected error occurred: {e}", message)

    async def _analyze_with_ollama(self, query: str) -> Optional[str]:
        """
        Uses the Ollama model to analyze, rephrase, and enhance the user's query.

        Args:
            query: The user query to analyze.

        Returns:
            The enhanced query, or None if an error occurs.
        """
        if not self.ollama_client:
            logger.error("Ollama client is not available.")
            return None

        prompt = f"""
Rewrite the user's message into a concise, precise KB search query.

User message: "{query}"

Rules:
- One line, 3-12 words.
- Include product/app names and error cues.
- No filler; return ONLY the rewritten query.
"""

        try:
            response = await asyncio.to_thread(self.ollama_client.invoke, prompt)
            analyzed_query = response.strip()
            logger.info(f"Ollama ({self.llm_model}) analyzed query: '{query}' -> '{analyzed_query}'")
            return analyzed_query
        except Exception as e:
            logger.error(f"Ollama invocation failed: {e}")
            return None

    def _is_non_technical_query(self, query: str) -> bool:
        """
        Detect if a query is non-technical (greetings, casual conversation, etc.)

        Args:
            query: The user's query text

        Returns:
            True if the query is non-technical and should be handled directly
        """
        query_lower = query.lower().strip()

        # Remove common punctuation and extra words
        cleaned_query = query_lower.replace("?", "").replace("!", "").replace(",", "")

        # Greeting patterns
        greeting_patterns = [
            "hi", "hello", "hey", "good morning", "good afternoon", "good evening",
            "how are you", "what's up", "sup", "howdy", "greetings"
        ]

        # Casual conversation patterns
        casual_patterns = [
            "how are you doing", "what's going on", "how's it going", "nice to meet you",
            "thanks", "thank you", "bye", "goodbye", "see you", "have a good day",
            "good day", "good night", "take care"
        ]

        # Check for exact matches or patterns within the query
        all_patterns = greeting_patterns + casual_patterns

        # Direct match
        if cleaned_query in all_patterns:
            return True

        # Check if query contains greeting/casual words but also check it's not technical
        contains_greeting = any(pattern in cleaned_query for pattern in all_patterns)

        # Technical keywords that indicate a real support query
        technical_keywords = [
            "password", "login", "access", "error", "problem", "issue", "help", "support",
            "account", "email", "system", "application", "software", "hardware", "network",
            "vpn", "firewall", "database", "server", "website", "portal", "dashboard",
            "reset", "install", "update", "configure", "setup", "troubleshoot", "fix",
            "not working", "can't", "cannot", "unable", "failed", "broken", "down"
        ]

        contains_technical = any(keyword in cleaned_query for keyword in technical_keywords)

        # If it contains greeting words but no technical keywords, treat as non-technical
        if contains_greeting and not contains_technical:
            return True

        # Special case: very short queries that are likely greetings
        if len(cleaned_query.split()) <= 3 and contains_greeting:
            return True

        return False

    def _generate_friendly_response(self, query: str) -> str:
        """
        Generate an appropriate friendly response based on the query type

        Args:
            query: The user's query text

        Returns:
            A friendly, contextual response
        """
        query_lower = query.lower().strip()

        if any(greeting in query_lower for greeting in ["hi", "hello", "hey", "good morning", "good afternoon", "good evening"]):
            return "Hello! 👋 I'm your AI support assistant for NexaCorp. I'm here to help you with any technical issues, account problems, or questions you might have. How can I assist you today?"

        elif any(phrase in query_lower for phrase in ["how are you", "what's up", "how's it going"]):
            return "I'm doing great, thank you for asking! 😊 I'm here and ready to help you with any technical support needs. What can I assist you with today?"

        elif any(phrase in query_lower for phrase in ["thanks", "thank you"]):
            return "You're very welcome! 😊 If you have any other questions or need further assistance, please don't hesitate to ask. I'm here to help!"

        elif any(phrase in query_lower for phrase in ["bye", "goodbye", "see you", "good day", "good night"]):
            return "Goodbye! 👋 Have a wonderful day, and feel free to reach out anytime if you need technical support. Take care!"

        else:
            return "Hello! 👋 I'm your AI support assistant. I'm here to help you with any technical issues, account problems, or questions you might have. How can I assist you today?"

    def _create_error_response(self, error_message: str, original_message: Message) -> Message:
        """Creates a standardized error message."""
        return Message(
            type=MessageType.ERROR,
            content=error_message,
            sender=self.agent_id,
            recipient=original_message.sender,
            metadata={"original_message_id": original_message.id}
        )
