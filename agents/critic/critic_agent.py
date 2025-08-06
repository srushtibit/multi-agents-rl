"""
Critic Agent for the multilingual multi-agent support system.
Evaluates agent responses and provides reward signals for reinforcement learning.
"""

import logging
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
import re
from dataclasses import dataclass
import asyncio

from agents.base_agent import BaseAgent, Message, MessageType
from utils.language_utils import detect_language

logger = logging.getLogger(__name__)

@dataclass
class EvaluationCriteria:
    """Criteria for evaluating agent responses."""
    relevance_weight: float = 0.4
    accuracy_weight: float = 0.3
    completeness_weight: float = 0.2
    language_quality_weight: float = 0.1

@dataclass
class EvaluationResult:
    """Result of response evaluation."""
    overall_score: float
    relevance_score: float
    accuracy_score: float
    completeness_score: float
    language_quality_score: float
    feedback: str
    detailed_analysis: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'overall_score': self.overall_score,
            'relevance_score': self.relevance_score,
            'accuracy_score': self.accuracy_score,
            'completeness_score': self.completeness_score,
            'language_quality_score': self.language_quality_score,
            'feedback': self.feedback,
            'detailed_analysis': self.detailed_analysis
        }

class CriticAgent(BaseAgent):
    """Agent that evaluates responses and provides feedback for learning."""
    
    def __init__(self, agent_id: str = "critic_agent", config: Dict[str, Any] = None):
        super().__init__(agent_id, config)
        
        # Evaluation criteria
        criteria_config = self.system_config.get('agents.critic.reward_components', {})
        self.criteria = EvaluationCriteria(
            relevance_weight=criteria_config.get('relevance', 0.4),
            accuracy_weight=criteria_config.get('accuracy', 0.3),
            completeness_weight=criteria_config.get('completeness', 0.2),
            language_quality_weight=criteria_config.get('language_quality', 0.1)
        )
        
        self.score_threshold = self.system_config.get('agents.critic.score_threshold', 0.7)
        
        # Evaluation history
        self.evaluation_history: List[EvaluationResult] = []
        self.response_quality_trends: Dict[str, List[float]] = {}
        
        # Quality indicators
        self.quality_indicators = {
            'positive_keywords': [
                'solution', 'resolved', 'fixed', 'working', 'successful', 
                'complete', 'finished', 'done', 'ready', 'available'
            ],
            'negative_keywords': [
                'error', 'failed', 'broken', 'issue', 'problem', 'unable',
                'cannot', 'impossible', 'unavailable', 'missing'
            ],
            'uncertainty_keywords': [
                'maybe', 'perhaps', 'possibly', 'might', 'could', 'unclear',
                'uncertain', 'unsure', 'unknown', 'not sure'
            ],
            'action_keywords': [
                'follow', 'steps', 'instructions', 'procedure', 'guide',
                'contact', 'call', 'email', 'submit', 'request'
            ]
        }
        
        # Language quality patterns
        self.language_patterns = {
            'professional_phrases': [
                'please', 'thank you', 'i apologize', 'i understand',
                'let me help', 'i will assist', 'best regards'
            ],
            'technical_accuracy': [
                'configure', 'settings', 'parameters', 'options',
                'system', 'network', 'database', 'server'
            ]
        }
    
    def get_capabilities(self) -> List[str]:
        """Get agent capabilities."""
        return [
            "response_evaluation",
            "quality_assessment", 
            "reward_generation",
            "feedback_provision",
            "learning_optimization",
            "performance_tracking"
        ]
    
    async def process_message(self, message: Message) -> Optional[Message]:
        """
        Evaluate a response message and provide feedback.
        
        Args:
            message: Response message to evaluate
            
        Returns:
            Feedback message with evaluation results
        """
        try:
            # Extract evaluation context
            evaluation_context = self._extract_evaluation_context(message)
            
            # Perform comprehensive evaluation
            evaluation_result = await self._evaluate_response(message, evaluation_context)
            
            # Send reward signal to communication agent if this was from a symbolic message
            await self._send_reward_signal(evaluation_result, message)
            
            # Update learning trends
            self._update_quality_trends(message.sender, evaluation_result)
            
            # Create feedback response
            feedback_message = self._create_feedback_message(evaluation_result, message)
            
            # Log evaluation
            self._log_action("response_evaluation", {
                "overall_score": evaluation_result.overall_score,
                "sender": message.sender,
                "content_length": len(message.content),
                "language": message.language
            })
            
            # Store evaluation
            self.evaluation_history.append(evaluation_result)
            
            return feedback_message
            
        except Exception as e:
            logger.error(f"Error in critic agent evaluation: {e}")
            self._log_action("evaluation_error", {"error": str(e)}, success=False, error_message=str(e))
            return None
    
    def _extract_evaluation_context(self, message: Message) -> Dict[str, Any]:
        """
        Extract context needed for evaluation.
        
        Args:
            message: Message to evaluate
            
        Returns:
            Evaluation context
        """
        context = {
            'message_type': message.type.value,
            'sender': message.sender,
            'language': message.language,
            'has_symbolic_encoding': message.symbolic_encoding is not None,
            'metadata': message.metadata or {}
        }
        
        # Extract query context if available
        if 'query' in message.metadata:
            context['original_query'] = message.metadata['query']
        
        # Extract retrieval context if this is from retrieval agent
        if message.sender == 'retrieval_agent' and 'retrieval_context' in message.metadata:
            context['retrieval_context'] = message.metadata['retrieval_context']
            context['results_count'] = message.metadata.get('results_count', 0)
        
        return context
    
    async def _evaluate_response(self, message: Message, context: Dict[str, Any]) -> EvaluationResult:
        """
        Perform comprehensive evaluation of the response.
        
        Args:
            message: Message to evaluate
            context: Evaluation context
            
        Returns:
            Detailed evaluation result
        """
        content = message.content
        
        # Evaluate different dimensions
        relevance_score = self._evaluate_relevance(content, context)
        accuracy_score = self._evaluate_accuracy(content, context)
        completeness_score = self._evaluate_completeness(content, context)
        language_quality_score = self._evaluate_language_quality(content, message.language)
        
        # Calculate weighted overall score
        overall_score = (
            self.criteria.relevance_weight * relevance_score +
            self.criteria.accuracy_weight * accuracy_score +
            self.criteria.completeness_weight * completeness_score +
            self.criteria.language_quality_weight * language_quality_score
        )
        
        # Generate detailed feedback
        feedback, detailed_analysis = self._generate_feedback(
            content, relevance_score, accuracy_score, 
            completeness_score, language_quality_score, context
        )
        
        return EvaluationResult(
            overall_score=overall_score,
            relevance_score=relevance_score,
            accuracy_score=accuracy_score,
            completeness_score=completeness_score,
            language_quality_score=language_quality_score,
            feedback=feedback,
            detailed_analysis=detailed_analysis
        )
    
    def _evaluate_relevance(self, content: str, context: Dict[str, Any]) -> float:
        """
        Evaluate the relevance of the response.
        
        Args:
            content: Response content
            context: Evaluation context
            
        Returns:
            Relevance score (0.0 to 1.0)
        """
        try:
            base_score = 0.5
            
            # Check if response addresses the original query
            original_query = context.get('original_query', '')
            if original_query:
                query_words = set(original_query.lower().split())
                content_words = set(content.lower().split())
                word_overlap = len(query_words.intersection(content_words))
                overlap_ratio = word_overlap / max(len(query_words), 1)
                base_score += 0.3 * overlap_ratio
            
            # Check for presence of positive indicators
            positive_indicators = sum(1 for kw in self.quality_indicators['positive_keywords'] 
                                    if kw in content.lower())
            base_score += min(0.2, positive_indicators * 0.05)
            
            # Penalize for negative indicators
            negative_indicators = sum(1 for kw in self.quality_indicators['negative_keywords'] 
                                    if kw in content.lower())
            base_score -= min(0.2, negative_indicators * 0.05)
            
            # Bonus for actionable content
            action_indicators = sum(1 for kw in self.quality_indicators['action_keywords'] 
                                  if kw in content.lower())
            base_score += min(0.1, action_indicators * 0.02)
            
            # Check retrieval quality if applicable
            if context.get('retrieval_context'):
                retrieval_context = context['retrieval_context']
                max_similarity = retrieval_context.get('max_similarity', 0)
                base_score += 0.2 * max_similarity
            
            return max(0.0, min(1.0, base_score))
            
        except Exception as e:
            logger.warning(f"Error evaluating relevance: {e}")
            return 0.5
    
    def _evaluate_accuracy(self, content: str, context: Dict[str, Any]) -> float:
        """
        Evaluate the accuracy of the response.
        
        Args:
            content: Response content
            context: Evaluation context
            
        Returns:
            Accuracy score (0.0 to 1.0)
        """
        try:
            base_score = 0.6
            
            # Check for technical accuracy indicators
            technical_terms = sum(1 for term in self.language_patterns['technical_accuracy'] 
                                if term in content.lower())
            base_score += min(0.2, technical_terms * 0.05)
            
            # Check for uncertainty indicators (reduce accuracy)
            uncertainty_indicators = sum(1 for kw in self.quality_indicators['uncertainty_keywords'] 
                                       if kw in content.lower())
            base_score -= min(0.3, uncertainty_indicators * 0.1)
            
            # Check for specific information (URLs, file paths, specific procedures)
            specificity_bonus = 0
            if re.search(r'https?://', content):
                specificity_bonus += 0.1
            if re.search(r'[A-Za-z]:\\|/[a-z]', content):  # File paths
                specificity_bonus += 0.05
            if re.search(r'\d+\.\s+', content):  # Numbered steps
                specificity_bonus += 0.1
            
            base_score += min(0.2, specificity_bonus)
            
            # Penalize for contradictory information
            if 'but' in content.lower() and 'however' in content.lower():
                base_score -= 0.1
            
            return max(0.0, min(1.0, base_score))
            
        except Exception as e:
            logger.warning(f"Error evaluating accuracy: {e}")
            return 0.6
    
    def _evaluate_completeness(self, content: str, context: Dict[str, Any]) -> float:
        """
        Evaluate the completeness of the response.
        
        Args:
            content: Response content
            context: Evaluation context
            
        Returns:
            Completeness score (0.0 to 1.0)
        """
        try:
            base_score = 0.4
            
            # Length bonus (but not too long)
            content_length = len(content)
            if content_length > 50:
                length_bonus = min(0.3, (content_length - 50) / 500)
                base_score += length_bonus
            
            # Check for complete procedure/solution
            if any(pattern in content.lower() for pattern in ['step 1', 'first,', 'then,', 'finally']):
                base_score += 0.2
            
            # Check for multiple information sources
            results_count = context.get('results_count', 0)
            if results_count > 1:
                base_score += min(0.2, (results_count - 1) * 0.05)
            
            # Check for additional resources/contacts
            if any(pattern in content.lower() for pattern in ['contact', 'support', 'help', 'assistance']):
                base_score += 0.1
            
            # Penalize for incomplete responses
            if content.endswith('...') or 'incomplete' in content.lower():
                base_score -= 0.2
            
            return max(0.0, min(1.0, base_score))
            
        except Exception as e:
            logger.warning(f"Error evaluating completeness: {e}")
            return 0.4
    
    def _evaluate_language_quality(self, content: str, language: str) -> float:
        """
        Evaluate the language quality of the response.
        
        Args:
            content: Response content
            language: Language code
            
        Returns:
            Language quality score (0.0 to 1.0)
        """
        try:
            base_score = 0.6
            
            # Check for professional language
            professional_terms = sum(1 for phrase in self.language_patterns['professional_phrases'] 
                                   if phrase in content.lower())
            base_score += min(0.2, professional_terms * 0.1)
            
            # Check grammar and structure (simplified)
            sentences = content.split('.')
            avg_sentence_length = np.mean([len(s.split()) for s in sentences if s.strip()])
            
            # Optimal sentence length bonus
            if 8 <= avg_sentence_length <= 20:
                base_score += 0.1
            
            # Check for proper capitalization
            if content and content[0].isupper():
                base_score += 0.05
            
            # Penalize for excessive caps or poor formatting
            caps_ratio = sum(1 for c in content if c.isupper()) / max(len(content), 1)
            if caps_ratio > 0.3:
                base_score -= 0.2
            
            # Language-specific checks
            if language != 'en':
                # For non-English, we're more lenient but check for basic structure
                if len(content.split()) > 3:  # Has multiple words
                    base_score += 0.1
            
            return max(0.0, min(1.0, base_score))
            
        except Exception as e:
            logger.warning(f"Error evaluating language quality: {e}")
            return 0.6
    
    def _generate_feedback(self, 
                          content: str, 
                          relevance: float, 
                          accuracy: float, 
                          completeness: float, 
                          language_quality: float, 
                          context: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
        """
        Generate detailed feedback and analysis.
        
        Args:
            content: Response content
            relevance: Relevance score
            accuracy: Accuracy score
            completeness: Completeness score
            language_quality: Language quality score
            context: Evaluation context
            
        Returns:
            Tuple of (feedback_text, detailed_analysis)
        """
        feedback_parts = []
        
        # Overall assessment
        overall_score = (
            self.criteria.relevance_weight * relevance +
            self.criteria.accuracy_weight * accuracy +
            self.criteria.completeness_weight * completeness +
            self.criteria.language_quality_weight * language_quality
        )
        
        if overall_score >= 0.8:
            feedback_parts.append("Excellent response quality.")
        elif overall_score >= 0.6:
            feedback_parts.append("Good response with room for improvement.")
        else:
            feedback_parts.append("Response needs significant improvement.")
        
        # Specific feedback
        if relevance < 0.6:
            feedback_parts.append("Consider addressing the user's query more directly.")
        
        if accuracy < 0.6:
            feedback_parts.append("Verify information accuracy and reduce uncertainty.")
        
        if completeness < 0.6:
            feedback_parts.append("Provide more comprehensive information or additional steps.")
        
        if language_quality < 0.6:
            feedback_parts.append("Improve language clarity and professionalism.")
        
        # Positive feedback
        if relevance > 0.8:
            feedback_parts.append("Response is highly relevant to the query.")
        
        if accuracy > 0.8:
            feedback_parts.append("Information appears accurate and specific.")
        
        feedback_text = " ".join(feedback_parts)
        
        # Detailed analysis
        detailed_analysis = {
            'scores': {
                'relevance': relevance,
                'accuracy': accuracy,
                'completeness': completeness,
                'language_quality': language_quality,
                'overall': overall_score
            },
            'content_analysis': {
                'word_count': len(content.split()),
                'character_count': len(content),
                'sentence_count': len([s for s in content.split('.') if s.strip()]),
                'has_action_items': any(kw in content.lower() for kw in self.quality_indicators['action_keywords']),
                'language': context.get('language', 'unknown')
            },
            'improvement_suggestions': self._generate_improvement_suggestions(relevance, accuracy, completeness, language_quality)
        }
        
        return feedback_text, detailed_analysis
    
    def _generate_improvement_suggestions(self, relevance: float, accuracy: float, completeness: float, language_quality: float) -> List[str]:
        """Generate specific improvement suggestions."""
        suggestions = []
        
        if relevance < 0.7:
            suggestions.append("Focus more on addressing the specific user query")
            suggestions.append("Include relevant keywords from the original question")
        
        if accuracy < 0.7:
            suggestions.append("Provide more specific and verifiable information")
            suggestions.append("Reduce uncertain language and be more definitive")
        
        if completeness < 0.7:
            suggestions.append("Include step-by-step instructions where applicable")
            suggestions.append("Provide additional context or alternative solutions")
        
        if language_quality < 0.7:
            suggestions.append("Use more professional and clear language")
            suggestions.append("Check grammar and sentence structure")
        
        return suggestions
    
    async def _send_reward_signal(self, evaluation: EvaluationResult, message: Message):
        """
        Send reward signal to communication agent for RL training.
        
        Args:
            evaluation: Evaluation result
            message: Original message
        """
        try:
            # Only send reward if this was a response to a symbolic message
            if (message.metadata and 
                message.metadata.get('query') and 
                message.sender == 'retrieval_agent'):
                
                # Create reward message for communication agent
                reward_message = Message(
                    type=MessageType.FEEDBACK,
                    content=f"Reward: {evaluation.overall_score:.3f}",
                    metadata={
                        'reward_value': evaluation.overall_score,
                        'evaluation_details': evaluation.to_dict(),
                        'target_episode': 'current'
                    },
                    sender=self.agent_id,
                    recipient="communication_agent"
                )
                
                # Add to outbound queue
                self.outbound_queue.append(reward_message)
                
                logger.debug(f"Sent reward signal: {evaluation.overall_score:.3f}")
                
        except Exception as e:
            logger.warning(f"Error sending reward signal: {e}")
    
    def _create_feedback_message(self, evaluation: EvaluationResult, original_message: Message) -> Message:
        """
        Create feedback message with evaluation results.
        
        Args:
            evaluation: Evaluation result
            original_message: Original message that was evaluated
            
        Returns:
            Feedback message
        """
        feedback_content = f"Evaluation: {evaluation.overall_score:.3f}/1.0\n{evaluation.feedback}"
        
        return Message(
            type=MessageType.FEEDBACK,
            content=feedback_content,
            metadata={
                'evaluation_result': evaluation.to_dict(),
                'original_message_id': original_message.id,
                'evaluated_sender': original_message.sender
            },
            sender=self.agent_id,
            recipient=original_message.sender,
            language=original_message.language
        )
    
    def _update_quality_trends(self, agent_id: str, evaluation: EvaluationResult):
        """Update quality trend tracking for an agent."""
        if agent_id not in self.response_quality_trends:
            self.response_quality_trends[agent_id] = []
        
        self.response_quality_trends[agent_id].append(evaluation.overall_score)
        
        # Keep only recent evaluations (last 100)
        if len(self.response_quality_trends[agent_id]) > 100:
            self.response_quality_trends[agent_id] = self.response_quality_trends[agent_id][-100:]
    
    def get_evaluation_stats(self) -> Dict[str, Any]:
        """Get evaluation statistics."""
        if not self.evaluation_history:
            return {}
        
        recent_evaluations = self.evaluation_history[-50:]  # Last 50 evaluations
        
        stats = {
            'total_evaluations': len(self.evaluation_history),
            'recent_average_score': np.mean([e.overall_score for e in recent_evaluations]),
            'recent_scores': {
                'relevance': np.mean([e.relevance_score for e in recent_evaluations]),
                'accuracy': np.mean([e.accuracy_score for e in recent_evaluations]), 
                'completeness': np.mean([e.completeness_score for e in recent_evaluations]),
                'language_quality': np.mean([e.language_quality_score for e in recent_evaluations])
            },
            'quality_trends': {
                agent_id: {
                    'current_average': np.mean(scores[-10:]) if len(scores) >= 10 else np.mean(scores),
                    'trend': 'improving' if len(scores) >= 5 and np.mean(scores[-5:]) > np.mean(scores[:5]) else 'stable',
                    'total_responses': len(scores)
                }
                for agent_id, scores in self.response_quality_trends.items()
            },
            'score_distribution': {
                'excellent': sum(1 for e in recent_evaluations if e.overall_score >= 0.8),
                'good': sum(1 for e in recent_evaluations if 0.6 <= e.overall_score < 0.8),
                'poor': sum(1 for e in recent_evaluations if e.overall_score < 0.6)
            }
        }
        
        return stats
    
    def reset_evaluation_history(self):
        """Reset evaluation history."""
        self.evaluation_history.clear()
        self.response_quality_trends.clear()
        self._log_action("reset_evaluation_history")