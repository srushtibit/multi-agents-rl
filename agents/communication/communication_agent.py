
import logging
from typing import Dict, Any, List, Optional
import asyncio
import numpy as np

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

        # RL-related attributes for training
        self.rl_enabled = False
        self.current_episode_data = []
        self.prompt_strategies = [
            "direct_rewrite",
            "context_enhanced",
            "keyword_focused",
            "intent_based"
        ]
        self.current_strategy = "direct_rewrite"
        self.strategy_performance = {strategy: [] for strategy in self.prompt_strategies}

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
        Uses different strategies based on RL training.

        Args:
            query: The user query to analyze.

        Returns:
            The enhanced query, or None if an error occurs.
        """
        if not self.ollama_client:
            logger.error("Ollama client is not available.")
            return None

        # Select prompt based on current RL strategy
        prompt = self._get_strategy_prompt(query, self.current_strategy)

        try:
            response = await asyncio.to_thread(self.ollama_client.invoke, prompt)
            analyzed_query = response.strip()

            # Record strategy usage for RL
            if self.rl_enabled:
                self.current_episode_data.append({
                    'original_query': query,
                    'strategy_used': self.current_strategy,
                    'analyzed_query': analyzed_query,
                    'timestamp': asyncio.get_event_loop().time()
                })

            logger.info(f"Ollama ({self.llm_model}) analyzed query using {self.current_strategy}: '{query}' -> '{analyzed_query}'")
            return analyzed_query
        except Exception as e:
            logger.error(f"Ollama invocation failed: {e}")
            return None

    def _get_strategy_prompt(self, query: str, strategy: str) -> str:
        """Get prompt based on the selected strategy."""
        base_query = f'User message: "{query}"'

        if strategy == "direct_rewrite":
            return f"""
Rewrite the user's message into a concise, precise KB search query.

{base_query}

Rules:
- One line, 3-12 words.
- Include product/app names and error cues.
- No filler; return ONLY the rewritten query.
"""

        elif strategy == "context_enhanced":
            return f"""
Analyze the user's message and create an enhanced search query with context.

{base_query}

Rules:
- Add relevant technical context and synonyms.
- Include related terms that might appear in documentation.
- Keep it focused but comprehensive.
- Return ONLY the enhanced query.
"""

        elif strategy == "keyword_focused":
            return f"""
Extract the most important keywords from the user's message for knowledge base search.

{base_query}

Rules:
- Focus on technical terms, product names, and action words.
- Remove filler words and casual language.
- Prioritize searchable terms.
- Return ONLY the keyword-focused query.
"""

        elif strategy == "intent_based":
            return f"""
Identify the user's intent and create a search query that captures their goal.

{base_query}

Rules:
- Focus on what the user wants to achieve.
- Include the problem type and desired outcome.
- Make it specific to support documentation.
- Return ONLY the intent-based query.
"""

        else:
            # Fallback to direct rewrite
            return self._get_strategy_prompt(query, "direct_rewrite")

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

    # RL Training Methods
    def start_episode(self):
        """Start a new RL training episode."""
        self.rl_enabled = True
        self.current_episode_data = []
        logger.debug(f"Started RL episode for {self.agent_id}")

    def end_episode(self):
        """End the current RL training episode."""
        self.rl_enabled = False
        logger.debug(f"Ended RL episode for {self.agent_id}")

    def update_from_reward(self, reward: float):
        """Update agent behavior based on received reward."""
        if self.rl_enabled:
            # Record performance of current strategy
            self.strategy_performance[self.current_strategy].append(reward)

            # Adaptive strategy selection based on performance
            if len(self.strategy_performance[self.current_strategy]) >= 5:
                avg_performance = np.mean(self.strategy_performance[self.current_strategy][-5:])

                # If current strategy is underperforming, try a different one
                if avg_performance < 0.5:
                    best_strategy = max(self.prompt_strategies,
                                      key=lambda s: np.mean(self.strategy_performance[s][-5:]) if self.strategy_performance[s] else 0)
                    if best_strategy != self.current_strategy:
                        self.current_strategy = best_strategy
                        logger.info(f"Switched to strategy: {self.current_strategy}")

    def get_rl_state(self, query: str) -> Dict[str, Any]:
        """Get current state representation for RL."""
        return {
            'query_length': len(query.split()),
            'has_technical_keywords': any(kw in query.lower() for kw in ["password", "login", "access", "error", "problem"]),
            'current_strategy': self.current_strategy,
            'strategy_performance': {k: np.mean(v[-5:]) if v else 0.0 for k, v in self.strategy_performance.items()},
            'query_complexity': self._assess_query_complexity(query)
        }

    def _assess_query_complexity(self, query: str) -> float:
        """Assess the complexity of a query (0-1 scale)."""
        factors = 0
        query_lower = query.lower()

        # Length factor
        if len(query.split()) > 10:
            factors += 0.3

        # Technical terms
        tech_terms = ["configuration", "authentication", "synchronization", "troubleshoot", "diagnostic"]
        if any(term in query_lower for term in tech_terms):
            factors += 0.4

        # Multiple issues mentioned
        if any(word in query_lower for word in ["and", "also", "additionally", "furthermore"]):
            factors += 0.3

        return min(1.0, factors)

    def _create_error_response(self, error_message: str, original_message: Message) -> Message:
        """Creates a standardized error message."""
        return Message(
            type=MessageType.ERROR,
            content=error_message,
            sender=self.agent_id,
            recipient=original_message.sender,
            metadata={"original_message_id": original_message.id}
        )
