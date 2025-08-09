"""
Ollama-Compatible Support Environment for Reinforcement Learning

This environment simulates a customer support scenario where an agent
learns to provide better responses through interaction and feedback.
Adapted to work with Ollama models instead of traditional neural networks.
"""

import logging
import numpy as np
import random
from typing import Dict, Any, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum
import asyncio
from datetime import datetime

from agents.base_agent import Message, MessageType
from utils.config_loader import get_config

logger = logging.getLogger(__name__)

class RewardType(Enum):
    """Types of rewards in the support environment."""
    RESPONSE_QUALITY = "response_quality"
    USER_SATISFACTION = "user_satisfaction"
    RESOLUTION_SUCCESS = "resolution_success"
    EFFICIENCY = "efficiency"
    STRATEGY_EFFECTIVENESS = "strategy_effectiveness"

@dataclass
class EnvironmentState:
    """Current state of the support environment."""
    current_query: str
    query_complexity: float
    user_satisfaction: float
    conversation_length: int
    resolved: bool
    agent_confidence: float
    strategy_used: str
    query_category: str
    
@dataclass
class OllamaEpisodeResult:
    """Result of an Ollama RL episode."""
    total_reward: float
    episode_length: int
    queries_processed: int
    strategies_used: Dict[str, int]
    average_response_quality: float
    resolution_rate: float
    user_satisfaction: float
    
class OllamaSupportEnvironment:
    """
    Ollama-compatible Reinforcement Learning environment for customer support scenarios.
    
    This environment provides:
    - Realistic customer support queries
    - Multi-dimensional reward signals based on response quality
    - State representation compatible with Ollama models
    - Episode management for strategy training
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the Ollama support environment.
        
        Args:
            config: Environment configuration
        """
        self.config = config or {}
        self.system_config = get_config()
        
        # Environment parameters
        self.max_episode_length = self.config.get('max_episode_length', 20)
        self.reward_weights = self.config.get('reward_weights', {
            'response_quality': 0.35,
            'user_satisfaction': 0.25,
            'resolution_success': 0.20,
            'efficiency': 0.10,
            'strategy_effectiveness': 0.10
        })
        
        # Current episode state
        self.current_state = None
        self.episode_step = 0
        self.episode_history = []
        self.total_episodes = 0
        self.current_episode_start = None
        
        # Query templates for training
        self.query_templates = self._load_query_templates()
        
        # Metrics tracking
        self.episode_rewards = []
        self.episode_lengths = []
        self.success_rate = []
        self.strategy_performance = {}
        
        # Ollama-specific tracking
        self.response_quality_history = []
        self.strategy_usage_history = []
        
    def _load_query_templates(self) -> List[Dict[str, Any]]:
        """Load query templates for training scenarios."""
        return [
            {
                'template': "I can't log into my account, it says my password is wrong",
                'complexity': 0.3,
                'category': 'authentication',
                'expected_keywords': ['password', 'reset', 'login', 'account'],
                'optimal_strategy': 'keyword_focused'
            },
            {
                'template': "The application is running very slowly and sometimes crashes",
                'complexity': 0.6,
                'category': 'performance',
                'expected_keywords': ['performance', 'slow', 'crash', 'optimization'],
                'optimal_strategy': 'context_enhanced'
            },
            {
                'template': "I need help setting up VPN access for remote work",
                'complexity': 0.7,
                'category': 'network',
                'expected_keywords': ['VPN', 'remote', 'network', 'connection'],
                'optimal_strategy': 'intent_based'
            },
            {
                'template': "My email notifications stopped working yesterday",
                'complexity': 0.4,
                'category': 'email',
                'expected_keywords': ['email', 'notification', 'SMTP', 'configuration'],
                'optimal_strategy': 'keyword_focused'
            },
            {
                'template': "I can't upload files to the portal, getting error 500",
                'complexity': 0.5,
                'category': 'upload',
                'expected_keywords': ['upload', 'file', 'error', 'portal'],
                'optimal_strategy': 'direct_rewrite'
            },
            {
                'template': "How do I reset my two-factor authentication?",
                'complexity': 0.4,
                'category': 'security',
                'expected_keywords': ['2FA', 'authentication', 'reset', 'security'],
                'optimal_strategy': 'intent_based'
            },
            {
                'template': "The dashboard shows incorrect data for last month's reports",
                'complexity': 0.8,
                'category': 'data',
                'expected_keywords': ['dashboard', 'data', 'report', 'incorrect'],
                'optimal_strategy': 'context_enhanced'
            },
            {
                'template': "I need access to the admin panel but getting permission denied",
                'complexity': 0.6,
                'category': 'permissions',
                'expected_keywords': ['admin', 'permission', 'access', 'denied'],
                'optimal_strategy': 'keyword_focused'
            },
            {
                'template': "Hi, I'm having trouble with my leave application not showing up",
                'complexity': 0.5,
                'category': 'hr',
                'expected_keywords': ['leave', 'application', 'HR', 'status'],
                'optimal_strategy': 'intent_based'
            },
            {
                'template': "The system keeps logging me out every few minutes",
                'complexity': 0.4,
                'category': 'session',
                'expected_keywords': ['logout', 'session', 'timeout', 'authentication'],
                'optimal_strategy': 'direct_rewrite'
            }
        ]
    
    def reset(self) -> EnvironmentState:
        """
        Reset the environment for a new episode.
        
        Returns:
            Initial environment state
        """
        # Select a random query template
        template = random.choice(self.query_templates)
        
        # Add some variation to the query
        query = self._add_query_variation(template['template'])
        
        # Initialize state
        self.current_state = EnvironmentState(
            current_query=query,
            query_complexity=template['complexity'],
            user_satisfaction=0.5,  # Neutral starting point
            conversation_length=0,
            resolved=False,
            agent_confidence=0.5,
            strategy_used="",
            query_category=template['category']
        )
        
        self.episode_step = 0
        self.episode_history = []
        self.current_episode_start = datetime.now()
        
        logger.debug(f"Environment reset with query: {query}")
        return self.current_state
    
    def _add_query_variation(self, template: str) -> str:
        """Add natural variation to query templates."""
        variations = [
            template,
            template.replace("I can't", "I cannot"),
            template.replace("I need", "I require"),
            template.replace("help", "assistance"),
            template.replace("problem", "issue"),
            template.replace("not working", "broken"),
        ]
        
        # Add casual variations
        casual_prefixes = ["", "Hi, ", "Hello, ", "Hey, ", "Excuse me, "]
        casual_suffixes = ["", ". Thanks!", ". Please help.", ". Any ideas?", ". What should I do?"]
        
        base_query = random.choice(variations)
        prefix = random.choice(casual_prefixes)
        suffix = random.choice(casual_suffixes)
        
        return f"{prefix}{base_query}{suffix}".strip()
    
    def step(self, action: Dict[str, Any]) -> Tuple[EnvironmentState, float, bool, Dict[str, Any]]:
        """
        Execute one step in the environment.
        
        Args:
            action: Action taken by the agent (contains strategy, response, etc.)
            
        Returns:
            Tuple of (next_state, reward, done, info)
        """
        self.episode_step += 1
        
        # Extract action components
        strategy_used = action.get('strategy', 'direct_rewrite')
        processed_query = action.get('processed_query', self.current_state.current_query)
        response_quality = action.get('response_quality', 0.5)
        
        # Calculate reward
        reward = self._calculate_reward(strategy_used, processed_query, response_quality)
        
        # Update state
        self.current_state.strategy_used = strategy_used
        self.current_state.conversation_length += 1
        self.current_state.agent_confidence = response_quality
        
        # Update user satisfaction based on response quality
        satisfaction_change = (response_quality - 0.5) * 0.3
        self.current_state.user_satisfaction = np.clip(
            self.current_state.user_satisfaction + satisfaction_change, 0.0, 1.0
        )
        
        # Check if resolved (simplified logic)
        if response_quality > 0.7 and self.current_state.user_satisfaction > 0.7:
            self.current_state.resolved = True
        
        # Episode termination conditions
        done = (
            self.current_state.resolved or
            self.episode_step >= self.max_episode_length or
            self.current_state.user_satisfaction < 0.2
        )
        
        # Record step
        step_info = {
            'step': self.episode_step,
            'strategy': strategy_used,
            'reward': reward,
            'response_quality': response_quality,
            'user_satisfaction': self.current_state.user_satisfaction,
            'resolved': self.current_state.resolved
        }
        self.episode_history.append(step_info)
        
        # Additional info
        info = {
            'episode_step': self.episode_step,
            'strategy_used': strategy_used,
            'response_quality': response_quality,
            'optimal_strategy': self._get_optimal_strategy(),
            'strategy_match': strategy_used == self._get_optimal_strategy()
        }
        
        return self.current_state, reward, done, info
    
    def _calculate_reward(self, strategy: str, processed_query: str, response_quality: float) -> float:
        """
        Calculate reward based on multiple factors.
        
        Args:
            strategy: Strategy used by the agent
            processed_query: Processed query
            response_quality: Quality of the response (0-1)
            
        Returns:
            Calculated reward
        """
        rewards = {}
        
        # Response quality reward
        rewards[RewardType.RESPONSE_QUALITY] = response_quality
        
        # User satisfaction reward
        satisfaction_reward = self.current_state.user_satisfaction
        rewards[RewardType.USER_SATISFACTION] = satisfaction_reward
        
        # Resolution success reward
        resolution_reward = 1.0 if self.current_state.resolved else 0.0
        rewards[RewardType.RESOLUTION_SUCCESS] = resolution_reward
        
        # Efficiency reward (fewer steps is better)
        efficiency_reward = max(0.0, 1.0 - (self.episode_step / self.max_episode_length))
        rewards[RewardType.EFFICIENCY] = efficiency_reward
        
        # Strategy effectiveness reward
        optimal_strategy = self._get_optimal_strategy()
        strategy_reward = 1.0 if strategy == optimal_strategy else 0.5
        rewards[RewardType.STRATEGY_EFFECTIVENESS] = strategy_reward
        
        # Weighted total reward
        total_reward = sum(
            rewards[reward_type] * self.reward_weights.get(reward_type.value, 0.0)
            for reward_type in rewards
        )
        
        return total_reward
    
    def _get_optimal_strategy(self) -> str:
        """Get the optimal strategy for the current query."""
        for template in self.query_templates:
            if template['category'] == self.current_state.query_category:
                return template.get('optimal_strategy', 'direct_rewrite')
        return 'direct_rewrite'
    
    def get_episode_result(self) -> OllamaEpisodeResult:
        """Get results for the completed episode."""
        if not self.episode_history:
            return OllamaEpisodeResult(
                total_reward=0.0,
                episode_length=0,
                queries_processed=0,
                strategies_used={},
                average_response_quality=0.0,
                resolution_rate=0.0,
                user_satisfaction=0.5
            )
        
        total_reward = sum(step['reward'] for step in self.episode_history)
        strategies_used = {}
        response_qualities = []
        
        for step in self.episode_history:
            strategy = step['strategy']
            strategies_used[strategy] = strategies_used.get(strategy, 0) + 1
            response_qualities.append(step['response_quality'])
        
        return OllamaEpisodeResult(
            total_reward=total_reward,
            episode_length=len(self.episode_history),
            queries_processed=1,  # One query per episode in this setup
            strategies_used=strategies_used,
            average_response_quality=np.mean(response_qualities) if response_qualities else 0.0,
            resolution_rate=1.0 if self.current_state.resolved else 0.0,
            user_satisfaction=self.current_state.user_satisfaction
        )
    
    def get_state_representation(self) -> Dict[str, Any]:
        """Get current state as a dictionary for RL agents."""
        return {
            'query': self.current_state.current_query,
            'query_complexity': self.current_state.query_complexity,
            'query_category': self.current_state.query_category,
            'conversation_length': self.current_state.conversation_length,
            'user_satisfaction': self.current_state.user_satisfaction,
            'agent_confidence': self.current_state.agent_confidence,
            'episode_step': self.episode_step,
            'resolved': self.current_state.resolved,
            'optimal_strategy': self._get_optimal_strategy()
        }
    
    def render(self, mode: str = 'human') -> Optional[str]:
        """Render the current environment state."""
        if mode == 'human':
            state_info = f"""
Episode Step: {self.episode_step}
Query: {self.current_state.current_query}
Category: {self.current_state.query_category}
Complexity: {self.current_state.query_complexity:.2f}
Strategy Used: {self.current_state.strategy_used}
User Satisfaction: {self.current_state.user_satisfaction:.2f}
Resolved: {self.current_state.resolved}
            """.strip()
            print(state_info)
            return state_info
        return None
