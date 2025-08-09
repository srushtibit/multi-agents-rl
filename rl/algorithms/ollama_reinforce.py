"""
Ollama-compatible REINFORCE implementation for training communication strategies.
Instead of training neural network weights, this trains prompt strategies and decision-making.
"""

import logging
import numpy as np
import json
import os
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from collections import deque, defaultdict
import asyncio
from datetime import datetime

from utils.config_loader import get_config

logger = logging.getLogger(__name__)

@dataclass
class StrategyPerformance:
    """Performance metrics for a strategy."""
    strategy_name: str
    total_uses: int
    total_reward: float
    average_reward: float
    success_rate: float
    last_updated: str

@dataclass
class OllamaTrainingStats:
    """Training statistics for Ollama-based RL."""
    episode: int
    total_episodes: int
    current_strategy: str
    strategy_rewards: Dict[str, float]
    episode_reward: float
    average_reward: float
    exploration_rate: float
    episode_length: int

class OllamaREINFORCETrainer:
    """REINFORCE-style trainer for Ollama-based communication strategies."""
    
    def __init__(self, communication_agent, config: Dict[str, Any] = None):
        """
        Initialize Ollama REINFORCE trainer.
        
        Args:
            communication_agent: The communication agent to train
            config: Training configuration
        """
        self.communication_agent = communication_agent
        self.config = config or {}
        self.system_config = get_config()
        
        # Training hyperparameters
        self.learning_rate = self.config.get('learning_rate', 0.1)
        self.gamma = self.config.get('gamma', 0.95)
        self.epsilon = self.config.get('epsilon', 0.3)  # Exploration rate
        self.epsilon_decay = self.config.get('epsilon_decay', 0.995)
        self.min_epsilon = self.config.get('min_epsilon', 0.05)
        
        # Strategy management
        self.strategies = communication_agent.prompt_strategies
        self.strategy_performance = {
            strategy: StrategyPerformance(
                strategy_name=strategy,
                total_uses=0,
                total_reward=0.0,
                average_reward=0.0,
                success_rate=0.0,
                last_updated=datetime.now().isoformat()
            ) for strategy in self.strategies
        }
        
        # Training state
        self.episode_count = 0
        self.total_steps = 0
        self.current_episode_rewards = []
        self.current_episode_strategies = []
        
        # Performance tracking
        self.reward_history = deque(maxlen=100)
        self.training_stats = []
        self.best_average_reward = float('-inf')
        
        # Load previous training if exists
        self.checkpoint_path = os.path.join("checkpoints", "ollama_rl_checkpoint.json")
        self.load_checkpoint()
    
    def select_strategy(self, query_context: Dict[str, Any]) -> str:
        """
        Select strategy using epsilon-greedy with contextual bandits.
        
        Args:
            query_context: Context about the current query
            
        Returns:
            Selected strategy name
        """
        # Epsilon-greedy exploration
        if np.random.random() < self.epsilon:
            # Explore: random strategy
            selected_strategy = np.random.choice(self.strategies)
            logger.debug(f"Exploring with strategy: {selected_strategy}")
        else:
            # Exploit: best performing strategy for this context
            selected_strategy = self._get_best_strategy_for_context(query_context)
            logger.debug(f"Exploiting with strategy: {selected_strategy}")
        
        return selected_strategy
    
    def _get_best_strategy_for_context(self, context: Dict[str, Any]) -> str:
        """Get the best performing strategy for the given context."""
        # Simple heuristic-based selection for now
        # In a more advanced implementation, this could use contextual bandits
        
        query_complexity = context.get('query_complexity', 0.5)
        has_technical = context.get('has_technical_keywords', False)
        
        if query_complexity > 0.7:
            # Complex queries benefit from context enhancement
            return "context_enhanced"
        elif has_technical:
            # Technical queries benefit from keyword focus
            return "keyword_focused"
        elif query_complexity < 0.3:
            # Simple queries work well with direct rewrite
            return "direct_rewrite"
        else:
            # Medium complexity queries benefit from intent analysis
            return "intent_based"
    
    def record_step(self, strategy: str, reward: float, context: Dict[str, Any]):
        """Record a training step."""
        self.current_episode_rewards.append(reward)
        self.current_episode_strategies.append(strategy)
        
        # Update strategy performance
        perf = self.strategy_performance[strategy]
        perf.total_uses += 1
        perf.total_reward += reward
        perf.average_reward = perf.total_reward / perf.total_uses
        perf.last_updated = datetime.now().isoformat()
        
        # Calculate success rate (reward > 0.5 considered success)
        strategy_rewards = [r for r, s in zip(self.current_episode_rewards, self.current_episode_strategies) if s == strategy]
        if strategy_rewards:
            successes = sum(1 for r in strategy_rewards if r > 0.5)
            perf.success_rate = successes / len(strategy_rewards)
        
        self.total_steps += 1
    
    def end_episode(self) -> OllamaTrainingStats:
        """End the current episode and update strategy preferences."""
        if not self.current_episode_rewards:
            logger.warning("Ending episode with no recorded steps")
            return self._create_empty_stats()
        
        # Calculate episode metrics
        episode_reward = sum(self.current_episode_rewards)
        average_reward = np.mean(self.current_episode_rewards)
        
        # Update reward history
        self.reward_history.append(episode_reward)
        
        # Update exploration rate
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)
        
        # Create training stats
        stats = OllamaTrainingStats(
            episode=self.episode_count + 1,
            total_episodes=self.episode_count + 1,
            current_strategy=self.communication_agent.current_strategy,
            strategy_rewards={s: self.strategy_performance[s].average_reward for s in self.strategies},
            episode_reward=episode_reward,
            average_reward=np.mean(self.reward_history) if self.reward_history else 0.0,
            exploration_rate=self.epsilon,
            episode_length=len(self.current_episode_rewards)
        )
        
        self.training_stats.append(stats)
        self.episode_count += 1
        
        # Update best performance
        current_avg = np.mean(self.reward_history) if self.reward_history else 0.0
        if current_avg > self.best_average_reward:
            self.best_average_reward = current_avg
            logger.info(f"New best average reward: {current_avg:.4f}")
        
        # Reset episode data
        self.current_episode_rewards = []
        self.current_episode_strategies = []
        
        # Save checkpoint periodically
        if self.episode_count % 10 == 0:
            self.save_checkpoint()
        
        logger.info(f"Episode {self.episode_count} completed: reward={episode_reward:.4f}, avg={current_avg:.4f}, ε={self.epsilon:.3f}")
        
        return stats
    
    def _create_empty_stats(self) -> OllamaTrainingStats:
        """Create empty stats for episodes with no data."""
        return OllamaTrainingStats(
            episode=self.episode_count,
            total_episodes=self.episode_count,
            current_strategy=self.communication_agent.current_strategy,
            strategy_rewards={s: 0.0 for s in self.strategies},
            episode_reward=0.0,
            average_reward=0.0,
            exploration_rate=self.epsilon,
            episode_length=0
        )
    
    def get_training_stats(self) -> Dict[str, Any]:
        """Get comprehensive training statistics."""
        if not self.training_stats:
            return {
                'total_episodes': 0,
                'total_steps': 0,
                'best_average_reward': 0.0,
                'current_exploration_rate': self.epsilon,
                'strategy_performance': {},
                'recent_performance': {},
                'training_progress': []
            }
        
        recent_stats = self.training_stats[-50:]  # Last 50 episodes
        
        return {
            'total_episodes': self.episode_count,
            'total_steps': self.total_steps,
            'best_average_reward': self.best_average_reward,
            'current_exploration_rate': self.epsilon,
            'strategy_performance': {
                strategy: {
                    'average_reward': perf.average_reward,
                    'total_uses': perf.total_uses,
                    'success_rate': perf.success_rate
                } for strategy, perf in self.strategy_performance.items()
            },
            'recent_performance': {
                'average_reward': np.mean([s.average_reward for s in recent_stats]) if recent_stats else 0.0,
                'average_episode_length': np.mean([s.episode_length for s in recent_stats]) if recent_stats else 0.0,
                'exploration_rate': self.epsilon
            },
            'training_progress': [
                {
                    'episode': s.episode,
                    'average_reward': s.average_reward,
                    'episode_reward': s.episode_reward,
                    'exploration_rate': s.exploration_rate,
                    'episode_length': s.episode_length,
                    'current_strategy': s.current_strategy
                }
                for s in self.training_stats[-20:]  # Last 20 episodes
            ]
        }
    
    def save_checkpoint(self):
        """Save training checkpoint."""
        checkpoint_data = {
            'episode_count': self.episode_count,
            'total_steps': self.total_steps,
            'epsilon': self.epsilon,
            'best_average_reward': self.best_average_reward,
            'strategy_performance': {
                strategy: {
                    'total_uses': perf.total_uses,
                    'total_reward': perf.total_reward,
                    'average_reward': perf.average_reward,
                    'success_rate': perf.success_rate,
                    'last_updated': perf.last_updated
                } for strategy, perf in self.strategy_performance.items()
            },
            'reward_history': list(self.reward_history),
            'training_stats': [
                {
                    'episode': s.episode,
                    'episode_reward': s.episode_reward,
                    'average_reward': s.average_reward,
                    'exploration_rate': s.exploration_rate,
                    'episode_length': s.episode_length,
                    'current_strategy': s.current_strategy
                } for s in self.training_stats[-100:]  # Save last 100 episodes
            ],
            'config': self.config,
            'timestamp': datetime.now().isoformat()
        }

        # Ensure directory exists
        checkpoint_dir = os.path.dirname(self.checkpoint_path)
        if checkpoint_dir:  # Only create if there's a directory part
            os.makedirs(checkpoint_dir, exist_ok=True)

        with open(self.checkpoint_path, 'w') as f:
            json.dump(checkpoint_data, f, indent=2)

        logger.info(f"Checkpoint saved to {self.checkpoint_path}")
    
    def load_checkpoint(self):
        """Load training checkpoint if it exists."""
        if not os.path.exists(self.checkpoint_path):
            logger.info("No checkpoint found, starting fresh training")
            return
        
        try:
            with open(self.checkpoint_path, 'r') as f:
                checkpoint_data = json.load(f)
            
            self.episode_count = checkpoint_data.get('episode_count', 0)
            self.total_steps = checkpoint_data.get('total_steps', 0)
            self.epsilon = checkpoint_data.get('epsilon', self.epsilon)
            self.best_average_reward = checkpoint_data.get('best_average_reward', float('-inf'))
            
            # Restore strategy performance
            for strategy, perf_data in checkpoint_data.get('strategy_performance', {}).items():
                if strategy in self.strategy_performance:
                    perf = self.strategy_performance[strategy]
                    perf.total_uses = perf_data.get('total_uses', 0)
                    perf.total_reward = perf_data.get('total_reward', 0.0)
                    perf.average_reward = perf_data.get('average_reward', 0.0)
                    perf.success_rate = perf_data.get('success_rate', 0.0)
                    perf.last_updated = perf_data.get('last_updated', datetime.now().isoformat())
            
            # Restore history
            self.reward_history = deque(checkpoint_data.get('reward_history', []), maxlen=100)
            
            # Restore training stats
            stats_data = checkpoint_data.get('training_stats', [])
            self.training_stats = []
            for stat in stats_data:
                self.training_stats.append(OllamaTrainingStats(
                    episode=stat['episode'],
                    total_episodes=stat['episode'],
                    current_strategy=stat['current_strategy'],
                    strategy_rewards={},  # Will be recalculated
                    episode_reward=stat['episode_reward'],
                    average_reward=stat['average_reward'],
                    exploration_rate=stat['exploration_rate'],
                    episode_length=stat['episode_length']
                ))
            
            logger.info(f"Checkpoint loaded from {self.checkpoint_path}")
            logger.info(f"Resumed training at episode {self.episode_count}, ε={self.epsilon:.3f}")
            
        except Exception as e:
            logger.error(f"Error loading checkpoint: {e}")
            logger.info("Starting fresh training")

class OllamaREINFORCEAgent:
    """Ollama-compatible REINFORCE agent for training communication strategies."""
    
    def __init__(self, communication_agent):
        """
        Initialize Ollama REINFORCE agent.
        
        Args:
            communication_agent: Communication agent to train
        """
        self.communication_agent = communication_agent
        self.trainer = OllamaREINFORCETrainer(communication_agent)
        
        # Training state
        self.training_active = False
        self.current_query_context = None
        self.current_strategy = None
        self.episode_start_time = None
    
    def start_training_episode(self):
        """Start a new training episode."""
        self.training_active = True
        self.episode_start_time = datetime.now()
        self.communication_agent.start_episode()
        logger.debug("Started Ollama RL training episode")
    
    def process_query_with_rl(self, query: str) -> Tuple[str, str]:
        """
        Process a query using RL-selected strategy.
        
        Args:
            query: Input query
            
        Returns:
            Tuple of (processed_query, strategy_used)
        """
        if not self.training_active:
            # Use current best strategy when not training
            strategy = self._get_best_strategy()
        else:
            # Get query context for strategy selection
            context = self.communication_agent.get_rl_state(query)
            strategy = self.trainer.select_strategy(context)
            
            # Store for reward attribution
            self.current_query_context = context
            self.current_strategy = strategy
        
        # Update communication agent strategy
        self.communication_agent.current_strategy = strategy
        
        return query, strategy
    
    def receive_reward(self, reward: float, query_context: Dict[str, Any] = None):
        """
        Receive reward and update training.
        
        Args:
            reward: Reward value (0-1 scale)
            query_context: Optional context about the query
        """
        if self.training_active and self.current_strategy:
            # Use stored context or provided context
            context = query_context or self.current_query_context or {}
            
            # Record the step
            self.trainer.record_step(self.current_strategy, reward, context)
            
            # Update communication agent
            self.communication_agent.update_from_reward(reward)
            
            logger.debug(f"Received reward {reward:.3f} for strategy {self.current_strategy}")
    
    def end_training_episode(self) -> OllamaTrainingStats:
        """
        End the current training episode.
        
        Returns:
            Training statistics
        """
        if not self.training_active:
            return None
        
        stats = self.trainer.end_episode()
        self.communication_agent.end_episode()
        
        self.training_active = False
        self.current_query_context = None
        self.current_strategy = None
        
        episode_duration = (datetime.now() - self.episode_start_time).total_seconds()
        logger.info(f"Training episode completed in {episode_duration:.2f}s")
        
        return stats
    
    def _get_best_strategy(self) -> str:
        """Get the currently best performing strategy."""
        best_strategy = max(
            self.trainer.strategy_performance.keys(),
            key=lambda s: self.trainer.strategy_performance[s].average_reward
        )
        return best_strategy
    
    def get_training_stats(self) -> Dict[str, Any]:
        """Get training statistics."""
        return self.trainer.get_training_stats()
    
    def save_model(self, filepath: str = None):
        """Save trained model."""
        if filepath:
            # Save to custom path
            old_path = self.trainer.checkpoint_path
            self.trainer.checkpoint_path = filepath

            # Ensure directory exists for custom path
            custom_dir = os.path.dirname(filepath)
            if custom_dir:
                os.makedirs(custom_dir, exist_ok=True)

            self.trainer.save_checkpoint()
            self.trainer.checkpoint_path = old_path
        else:
            self.trainer.save_checkpoint()
    
    def load_model(self, filepath: str):
        """Load trained model."""
        old_path = self.trainer.checkpoint_path
        self.trainer.checkpoint_path = filepath
        self.trainer.load_checkpoint()
        self.trainer.checkpoint_path = old_path
    
    def reset_training(self):
        """Reset training state."""
        self.trainer.episode_count = 0
        self.trainer.total_steps = 0
        self.trainer.epsilon = self.trainer.config.get('epsilon', 0.3)
        self.trainer.reward_history.clear()
        self.trainer.training_stats.clear()
        self.trainer.current_episode_rewards = []
        self.trainer.current_episode_strategies = []
        
        # Reset strategy performance
        for strategy in self.trainer.strategies:
            perf = self.trainer.strategy_performance[strategy]
            perf.total_uses = 0
            perf.total_reward = 0.0
            perf.average_reward = 0.0
            perf.success_rate = 0.0
            perf.last_updated = datetime.now().isoformat()
        
        logger.info("Ollama RL training state reset")
