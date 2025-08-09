

import logging
import re
from typing import Dict, Any, List, Optional
import asyncio

from agents.base_agent import BaseAgent, Message, MessageType
from langchain_ollama import OllamaLLM as Ollama
from kb.unified_knowledge_base import get_knowledge_base, SearchResult

logger = logging.getLogger(__name__)

class RetrievalAgent(BaseAgent):
    """Retrieves information and uses Ollama to synthesize answers (RAG)."""

    def __init__(self, agent_id: str = "retrieval_agent", config: Dict[str, Any] = None):
        super().__init__(agent_id, config)
        self.knowledge_base = get_knowledge_base()
        self.llm_model = self.system_config.get("llm.models.retrieval", "llama3.1:8b")
        self.ollama_client = Ollama(base_url=self.system_config.get("llm.ollama.base_url"), model=self.llm_model)
        self.max_results = self.system_config.get("agents.retrieval.max_documents", 5)
        self.min_similarity_score = self.system_config.get("knowledge_base.similarity_threshold", 0.7)

    def get_capabilities(self) -> List[str]:
        """Returns the capabilities of the retrieval agent."""
        return [
            "knowledge_retrieval",
            "semantic_search",
            "response_synthesis",
            "rag_with_llm"
        ]

    async def process_message(self, message: Message) -> Optional[Message]:
        """
        Processes a query, retrieves relevant documents, and synthesizes a response.

        Args:
            message: The incoming message from the CommunicationAgent.

        Returns:
            A response message for the CriticAgent to evaluate.
        """
        if message.type != MessageType.QUERY:
            return None

        try:
            # 1. Search the knowledge base (two-pass: strict then lenient)
            search_results = self.knowledge_base.search(
                query=message.content,
                max_results=self.max_results,
                min_score=self.min_similarity_score
            )
            if not search_results:
                # Fallback: lower threshold to get top-k regardless of score
                search_results = self.knowledge_base.search(
                    query=message.content,
                    max_results=self.max_results,
                    min_score=0.0
                )

            # 2. Synthesize a response using Ollama (RAG)
            synthesized_response = await self._synthesize_with_ollama(message.content, search_results)
            if not synthesized_response:
                return self._create_error_response("Failed to synthesize a response.", message)

            # 3. Create a response message for the CommunicationAgent (who will forward to user)
            response_message = Message(
                type=MessageType.RESPONSE,
                content=synthesized_response,
                metadata={
                    "original_query": message.metadata.get("original_query", message.content),
                    "retrieved_docs": [res.to_dict() for res in search_results],
                    "synthesis_by": self.agent_id,
                    "model": self.llm_model
                },
                sender=self.agent_id,
                recipient="communication_agent",
                language=message.language
            )

            self._log_action("response_synthesis", {
                "query": message.content,
                "num_retrieved": len(search_results),
                "response_length": len(synthesized_response)
            })

            return response_message

        except Exception as e:
            logger.error(f"Error in RetrievalAgent: {e}")
            return self._create_error_response(f"An unexpected error occurred: {e}", message)

    async def _synthesize_with_ollama(self, query: str, search_results: List[SearchResult]) -> Optional[str]:
        """
        Uses Ollama to synthesize a helpful response from the retrieved documents.

        Args:
            query: The user's original query.
            search_results: A list of relevant documents from the knowledge base.

        Returns:
            A synthesized, human-readable response.
        """
        if not self.ollama_client:
            logger.error("Ollama client is not available.")
            return "I am currently unable to process this request."

        if not search_results:
            return "I couldn't find any relevant information in the knowledge base to answer your question. Please try rephrasing your query."

        # Prepare the context from retrieved documents
        def _truncate(text: str, limit: int = 800) -> str:
            return text if len(text) <= limit else text[:limit] + "..."

        # Remove HTML-like tags from context to avoid LLM echoing them
        def _strip_html(s: str) -> str:
            return re.sub(r"<[^>]+>", "", s)

        context = "\n\n---\n\n".join([
            f"Source: {res.chunk.source_file}\nContent: {_strip_html(_truncate(res.chunk.content))}"
            for res in search_results[:self.max_results]
        ])

        prompt = f"""
You must answer using ONLY the context below. If the answer is not in the context, reply exactly:
"I couldn't find this in the knowledge base."

User query:
{query}

Context:
{context}

Answering rules (STRICT):
- Output ONLY the final answer; DO NOT include analysis or thinking.
- 3-6 concise bullet points tailored to the user's issue.
- Prefer steps that appeared in the context (e.g., lines starting with Resolution/Answer).
- No preface, no headings, no meta commentary.
"""

        try:
            response = await asyncio.to_thread(self.ollama_client.invoke, prompt)
            synthesized_answer = (response or "").strip()
            cleaned = self._postprocess_answer(synthesized_answer)
            if not cleaned:
                cleaned = self._build_extractive_answer(search_results)
            logger.info(f"Ollama ({self.llm_model}) synthesized response for query: '{query}'")
            return cleaned
        except Exception as e:
            logger.error(f"Ollama invocation for synthesis failed: {e}")
            return self._build_extractive_answer(search_results)

    def _postprocess_answer(self, text: str) -> str:
        """Remove chain-of-thought; keep only concise bullet points (max 6 lines)."""
        if not text:
            return ""
        # Strip HTML tags and known markers
        t = text.replace("```json", "").replace("```", "").strip()
        t = re.sub(r"<[^>]+>", "", t)
        # If think tags present, strip content between them
        if "<think>" in t:
            end = t.rfind("</think>")
            if end != -1:
                t = t[end + len("</think>"):].strip()
            else:
                # Drop everything up to first bullet
                pass
        lines = [ln.strip() for ln in t.splitlines() if ln.strip()]
        # Keep from first bullet-like line
        start_idx = 0
        for i, ln in enumerate(lines):
            if ln.startswith("-") or ln.startswith("*") or (ln[:2].isdigit() and ln.strip().find('.') == 1) or ln[:3].strip().isdigit():
                start_idx = i
                break
        kept = lines[start_idx:]
        # Filter out ticket metadata lines
        noise_patterns = [
            r"^ticket\b", r"^complaint id\b", r"^employee name\b", r"^domain\b", r"^priority\b",
            r"^queue\b", r"^business type\b", r"^tag\b", r"^language\b", r"^source:\b"
        ]
        filtered = []
        for ln in kept:
            low = ln.lower()
            if any(re.search(pat, low) for pat in noise_patterns):
                continue
            # Remove trailing periods-only lines or very long metadata-like lines
            if len(ln) > 300:
                continue
            filtered.append(ln)
        kept = filtered
        # If no bullets at all, trigger extractive fallback by returning empty string
        if not any(ln.startswith(('-', '*')) or (ln[:2].isdigit() and '.' in ln[:4]) for ln in kept):
            return ""
        else:
            # Keep max 6 bullet lines
            kept = [ln for ln in kept if ln][:6]
        return "\n".join(kept).strip()

    def _build_extractive_answer(self, search_results: List[SearchResult]) -> str:
        """Create a short answer by extracting Resolution/Answer lines from retrieved docs."""
        bullets: List[str] = []
        for res in search_results[: min(6, self.max_results)]:
            content = (res.chunk.content or "").strip()
            # Try to find explicit Resolution/Answer cues
            candidates: List[str] = []
            for key in ["Resolution:", "Answer:", "Solution:", "Steps:"]:
                idx = content.lower().find(key.lower())
                if idx != -1:
                    # Take from key to the end of line/next sentence
                    snippet = content[idx: idx + 220].split('\n')[0]
                    candidates.append(snippet)
            if not candidates:
                # fallback: first sentence
                candidates.append(content.split('\n')[0][:180])
            # Add first candidate as a bullet
            if candidates:
                bullets.append(f"- {candidates[0].strip()}")
            if len(bullets) >= 4:
                break
        if not bullets:
            return "I couldn't find this in the knowledge base."
        return "\n".join(bullets)

    def _create_error_response(self, error_message: str, original_message: Message) -> Message:
        """Creates a standardized error message."""
        return Message(
            type=MessageType.RESPONSE,
            content=error_message,
            sender=self.agent_id,
            recipient=original_message.sender,
            metadata={"original_message_id": original_message.id}
        )

