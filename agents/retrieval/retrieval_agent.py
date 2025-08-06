"""
Retrieval Agent for the multilingual multi-agent support system.
Handles knowledge base queries and retrieves relevant information.
"""

import logging
from typing import Dict, Any, List, Optional, Tuple
import asyncio
from dataclasses import asdict

from agents.base_agent import BaseAgent, Message, MessageType
from kb.unified_knowledge_base import get_knowledge_base, SearchResult
from utils.language_utils import detect_language, translate_to_english

logger = logging.getLogger(__name__)

class RetrievalAgent(BaseAgent):
    """Agent responsible for retrieving relevant information from the knowledge base."""
    
    def __init__(self, agent_id: str = "retrieval_agent", config: Dict[str, Any] = None):
        super().__init__(agent_id, config)
        
        # Configuration
        self.max_results = self.system_config.get('agents.retrieval.max_documents', 20)
        self.rerank_threshold = self.system_config.get('agents.retrieval.rerank_threshold', 0.8)
        self.context_window = self.system_config.get('agents.retrieval.context_window', 2048)
        self.min_similarity_score = self.system_config.get('knowledge_base.similarity_threshold', 0.75)
        
        # Knowledge base
        self.knowledge_base = get_knowledge_base()
        
        # Query processing
        self.query_cache: Dict[str, List[SearchResult]] = {}
        self.symbolic_query_cache: Dict[str, str] = {}  # Maps symbolic encodings to text queries
        
        # Retrieval statistics
        self.retrieval_stats = {
            'total_queries': 0,
            'cache_hits': 0,
            'successful_retrievals': 0,
            'failed_retrievals': 0,
            'average_results_per_query': 0.0,
            'average_retrieval_time': 0.0
        }
    
    def get_capabilities(self) -> List[str]:
        """Get agent capabilities."""
        return [
            "knowledge_retrieval",
            "semantic_search",
            "multilingual_query_processing",
            "symbolic_message_interpretation",
            "result_ranking",
            "context_synthesis"
        ]
    
    async def process_message(self, message: Message) -> Optional[Message]:
        """
        Process retrieval request and return relevant information.
        
        Args:
            message: Message containing query or symbolic encoding
            
        Returns:
            Message with retrieved information
        """
        try:
            import time
            start_time = time.time()
            
            # Extract query from message
            query_text = self._extract_query(message)
            if not query_text:
                return self._create_error_response("Could not extract query from message", message)
            
            # Check cache first
            cache_key = self._generate_cache_key(query_text, message.language)
            if cache_key in self.query_cache:
                search_results = self.query_cache[cache_key]
                self.retrieval_stats['cache_hits'] += 1
                logger.debug(f"Cache hit for query: {query_text[:50]}...")
            else:
                # Perform knowledge base search
                search_results = await self._search_knowledge_base(query_text, message)
                
                # Cache results
                self.query_cache[cache_key] = search_results
            
            # Process and rank results
            processed_results = self._process_search_results(search_results, query_text)
            
            # Create response message
            response = self._create_retrieval_response(processed_results, query_text, message)
            
            # Update statistics
            retrieval_time = time.time() - start_time
            self._update_retrieval_stats(len(search_results), retrieval_time, success=True)
            
            self._log_action("knowledge_retrieval", {
                "query_length": len(query_text),
                "results_count": len(search_results),
                "processing_time": retrieval_time,
                "language": message.language
            })
            
            return response
            
        except Exception as e:
            logger.error(f"Error in retrieval agent: {e}")
            self._update_retrieval_stats(0, 0, success=False)
            return self._create_error_response(f"Retrieval error: {str(e)}", message)
    
    def _extract_query(self, message: Message) -> Optional[str]:
        """
        Extract query text from message, handling symbolic encodings.
        
        Args:
            message: Input message
            
        Returns:
            Extracted query text
        """
        # If message has symbolic encoding, try to decode it
        if message.symbolic_encoding and message.type == MessageType.SYMBOLIC:
            # Check if we have a cached mapping
            symbolic_key = str(message.symbolic_encoding)
            if symbolic_key in self.symbolic_query_cache:
                return self.symbolic_query_cache[symbolic_key]
            
            # Try to interpret symbolic encoding
            interpreted_query = self._interpret_symbolic_encoding(message.symbolic_encoding, message.content)
            if interpreted_query:
                self.symbolic_query_cache[symbolic_key] = interpreted_query
                return interpreted_query
        
        # Fall back to direct content
        return message.content if message.content.strip() else None
    
    def _interpret_symbolic_encoding(self, encoding: List[int], fallback_text: str) -> Optional[str]:
        """
        Interpret symbolic encoding to extract meaning.
        
        Args:
            encoding: Symbolic encoding from communication agent
            fallback_text: Fallback text content
            
        Returns:
            Interpreted query text
        """
        try:
            # For now, use the fallback text
            # In a more advanced implementation, we would have learned mappings
            # between symbolic encodings and semantic meanings
            
            # Simple heuristic: if encoding seems meaningful, enhance the fallback text
            if len(encoding) > 0 and len(set(encoding)) > 1:  # Non-trivial encoding
                # Add symbolic context to the query
                enhanced_query = f"{fallback_text} [symbolic_context: {len(encoding)} tokens, diversity: {len(set(encoding))}]"
                return enhanced_query
            
            return fallback_text
            
        except Exception as e:
            logger.warning(f"Error interpreting symbolic encoding: {e}")
            return fallback_text
    
    async def _search_knowledge_base(self, query: str, message: Message) -> List[SearchResult]:
        """
        Search the knowledge base for relevant information.
        
        Args:
            query: Search query
            message: Original message for context
            
        Returns:
            List of search results
        """
        try:
            # Detect query language
            language_result = detect_language(query)
            
            # Translate query to English if needed for better search
            search_query = query
            if language_result.language != 'en':
                translation = translate_to_english(query, language_result.language)
                if translation.confidence > 0.7:
                    search_query = translation.translated_text
            
            # Perform search
            search_results = self.knowledge_base.search(
                query=search_query,
                max_results=self.max_results,
                language=None,  # Search all languages
                min_score=self.min_similarity_score
            )
            
            logger.debug(f"Knowledge base search returned {len(search_results)} results")
            return search_results
            
        except Exception as e:
            logger.error(f"Error searching knowledge base: {e}")
            return []
    
    def _process_search_results(self, results: List[SearchResult], query: str) -> List[Dict[str, Any]]:
        """
        Process and enhance search results.
        
        Args:
            results: Raw search results
            query: Original query
            
        Returns:
            Processed results with additional metadata
        """
        processed_results = []
        
        for result in results:
            try:
                # Extract relevant information
                chunk = result.chunk
                
                # Calculate additional relevance scores
                query_terms = set(query.lower().split())
                content_terms = set(chunk.content.lower().split())
                term_overlap = len(query_terms.intersection(content_terms)) / max(len(query_terms), 1)
                
                # Create processed result
                processed_result = {
                    'id': chunk.id,
                    'content': chunk.content,
                    'source_file': chunk.source_file,
                    'chunk_index': chunk.chunk_index,
                    'language': chunk.language,
                    'similarity_score': result.score,
                    'rank': result.rank,
                    'term_overlap_score': term_overlap,
                    'content_length': len(chunk.content),
                    'metadata': chunk.metadata
                }
                
                # Add contextual information
                processed_result['relevance_indicators'] = self._extract_relevance_indicators(
                    chunk.content, query
                )
                
                processed_results.append(processed_result)
                
            except Exception as e:
                logger.warning(f"Error processing search result: {e}")
                continue
        
        # Re-rank results if needed
        if len(processed_results) > 1:
            processed_results = self._rerank_results(processed_results, query)
        
        return processed_results
    
    def _extract_relevance_indicators(self, content: str, query: str) -> Dict[str, Any]:
        """
        Extract indicators of relevance between content and query.
        
        Args:
            content: Content text
            query: Query text
            
        Returns:
            Dictionary of relevance indicators
        """
        indicators = {}
        
        try:
            # Basic keyword matching
            query_words = set(word.lower() for word in query.split())
            content_words = set(word.lower() for word in content.split())
            
            matching_words = query_words.intersection(content_words)
            indicators['matching_keywords'] = list(matching_words)
            indicators['keyword_coverage'] = len(matching_words) / max(len(query_words), 1)
            
            # Positional information of matches
            match_positions = []
            content_lower = content.lower()
            for word in matching_words:
                pos = content_lower.find(word)
                if pos != -1:
                    match_positions.append(pos / len(content))
            
            indicators['match_positions'] = match_positions
            indicators['early_match'] = min(match_positions) if match_positions else 1.0
            
            # Content type indicators
            indicators['is_question'] = '?' in content
            indicators['is_procedural'] = any(word in content.lower() for word in ['step', 'procedure', 'how to', 'guide'])
            indicators['has_examples'] = any(word in content.lower() for word in ['example', 'for instance', 'such as'])
            
        except Exception as e:
            logger.debug(f"Error extracting relevance indicators: {e}")
        
        return indicators
    
    def _rerank_results(self, results: List[Dict[str, Any]], query: str) -> List[Dict[str, Any]]:
        """
        Re-rank results based on multiple criteria.
        
        Args:
            results: List of search results
            query: Original query
            
        Returns:
            Re-ranked results
        """
        try:
            # Calculate composite scores
            for result in results:
                # Combine different relevance signals
                similarity_score = result.get('similarity_score', 0.0)
                term_overlap_score = result.get('term_overlap_score', 0.0)
                early_match_bonus = 1.0 - result.get('relevance_indicators', {}).get('early_match', 1.0)
                
                # Composite score
                composite_score = (
                    0.5 * similarity_score +
                    0.3 * term_overlap_score +
                    0.2 * early_match_bonus
                )
                
                result['composite_score'] = composite_score
            
            # Sort by composite score
            results.sort(key=lambda x: x.get('composite_score', 0.0), reverse=True)
            
            # Update ranks
            for i, result in enumerate(results):
                result['reranked_position'] = i + 1
            
        except Exception as e:
            logger.warning(f"Error re-ranking results: {e}")
        
        return results
    
    def _create_retrieval_response(self, 
                                 results: List[Dict[str, Any]], 
                                 query: str, 
                                 original_message: Message) -> Message:
        """
        Create response message with retrieval results.
        
        Args:
            results: Processed search results
            query: Search query
            original_message: Original request message
            
        Returns:
            Response message
        """
        # Prepare response content
        if not results:
            response_content = f"No relevant information found for query: {query}"
            response_metadata = {
                'query': query,
                'results_count': 0,
                'search_status': 'no_results'
            }
        else:
            # Create structured response
            response_parts = [f"Found {len(results)} relevant results for: {query}\n"]
            
            for i, result in enumerate(results[:5]):  # Limit to top 5 results
                response_parts.append(f"\n{i+1}. {result['source_file']} (Score: {result['similarity_score']:.3f})")
                response_parts.append(f"   {result['content'][:200]}...")
            
            response_content = "\n".join(response_parts)
            
            response_metadata = {
                'query': query,
                'results_count': len(results),
                'search_status': 'success',
                'top_results': results[:3],  # Include top 3 full results
                'retrieval_context': {
                    'max_similarity': max(r['similarity_score'] for r in results),
                    'avg_similarity': sum(r['similarity_score'] for r in results) / len(results),
                    'languages_found': list(set(r['language'] for r in results)),
                    'source_files': list(set(r['source_file'] for r in results))
                }
            }
        
        # Create response message
        response = Message(
            type=MessageType.RESPONSE,
            content=response_content,
            metadata=response_metadata,
            sender=self.agent_id,
            recipient="critic_agent",  # Send to critic for evaluation
            language=original_message.language
        )
        
        return response
    
    def _create_error_response(self, error_message: str, original_message: Message) -> Message:
        """Create error response message."""
        return Message(
            type=MessageType.ERROR,
            content=f"Retrieval error: {error_message}",
            metadata={
                'error': error_message,
                'original_message_id': original_message.id
            },
            sender=self.agent_id,
            recipient=original_message.sender,
            language=original_message.language
        )
    
    def _generate_cache_key(self, query: str, language: str) -> str:
        """Generate cache key for query."""
        return f"{language}:{hash(query.lower())}"
    
    def _update_retrieval_stats(self, results_count: int, retrieval_time: float, success: bool):
        """Update retrieval statistics."""
        self.retrieval_stats['total_queries'] += 1
        
        if success:
            self.retrieval_stats['successful_retrievals'] += 1
            
            # Update running averages
            total_successful = self.retrieval_stats['successful_retrievals']
            current_avg_results = self.retrieval_stats['average_results_per_query']
            current_avg_time = self.retrieval_stats['average_retrieval_time']
            
            self.retrieval_stats['average_results_per_query'] = (
                (current_avg_results * (total_successful - 1) + results_count) / total_successful
            )
            
            self.retrieval_stats['average_retrieval_time'] = (
                (current_avg_time * (total_successful - 1) + retrieval_time) / total_successful
            )
        else:
            self.retrieval_stats['failed_retrievals'] += 1
    
    def get_retrieval_stats(self) -> Dict[str, Any]:
        """Get retrieval statistics."""
        stats = self.retrieval_stats.copy()
        stats.update({
            'cache_size': len(self.query_cache),
            'symbolic_cache_size': len(self.symbolic_query_cache),
            'success_rate': (
                self.retrieval_stats['successful_retrievals'] / 
                max(self.retrieval_stats['total_queries'], 1)
            ),
            'cache_hit_rate': (
                self.retrieval_stats['cache_hits'] / 
                max(self.retrieval_stats['total_queries'], 1)
            )
        })
        return stats
    
    def clear_cache(self):
        """Clear query caches."""
        self.query_cache.clear()
        self.symbolic_query_cache.clear()
        self._log_action("clear_cache")
    
    def warm_up_knowledge_base(self):
        """Warm up the knowledge base by loading common queries."""
        try:
            # Load sample queries to warm up embeddings
            sample_queries = [
                "How to reset password",
                "Email configuration issues", 
                "VPN connection problems",
                "Payroll questions",
                "Benefits enrollment",
                "IT support contact",
                "System maintenance",
                "Account access problems"
            ]
            
            for query in sample_queries:
                # Perform a lightweight search to warm up the system
                self.knowledge_base.search(query, max_results=1)
            
            self._log_action("knowledge_base_warmup", {"queries_processed": len(sample_queries)})
            logger.info("Knowledge base warmed up successfully")
            
        except Exception as e:
            logger.warning(f"Knowledge base warmup failed: {e}")
    
    async def process_batch_queries(self, queries: List[str]) -> List[List[SearchResult]]:
        """
        Process multiple queries in batch for efficiency.
        
        Args:
            queries: List of query strings
            
        Returns:
            List of search results for each query
        """
        results = []
        
        # Process queries concurrently
        tasks = []
        for query in queries:
            task = asyncio.create_task(self._search_knowledge_base(query, Message(content=query)))
            tasks.append(task)
        
        batch_results = await asyncio.gather(*tasks, return_exceptions=True)
        
        for i, result in enumerate(batch_results):
            if isinstance(result, Exception):
                logger.error(f"Error processing batch query {i}: {result}")
                results.append([])
            else:
                results.append(result)
        
        self._log_action("batch_query_processing", {"batch_size": len(queries)})
        return results