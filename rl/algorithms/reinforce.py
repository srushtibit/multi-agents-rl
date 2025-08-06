"""
REINFORCE algorithm implementation for training the Communication Agent.
Policy gradient method for learning symbolic communication protocols.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import logging
from typing import Dict, Any, List, Tuple, Optional
from dataclasses import dataclass
from collections import deque
import pickle
import os

from utils.config_loader import get_config

logger = logging.getLogger(__name__)

@dataclass
class Episode:
    """Represents a single episode of interaction."""
    states: List[torch.Tensor]
    actions: List[List[int]]
    rewards: List[float]
    log_probs: List[torch.Tensor]
    
    def __len__(self):
        return len(self.states)
    
    def total_reward(self) -> float:
        return sum(self.rewards)
    
    def average_reward(self) -> float:
        return sum(self.rewards) / len(self.rewards) if self.rewards else 0.0

@dataclass
class TrainingStats:
    """Training statistics."""
    episode: int
    total_episodes: int
    average_reward: float
    episode_reward: float
    policy_loss: float
    entropy_loss: float
    learning_rate: float
    episode_length: int

class REINFORCETrainer:
    """REINFORCE algorithm trainer for the Communication Agent."""
    
    def __init__(self, 
                 policy_network: nn.Module,
                 config: Dict[str, Any] = None):
        """
        Initialize REINFORCE trainer.
        
        Args:
            policy_network: The policy network to train
            config: Training configuration
        """
        self.policy_network = policy_network
        self.config = config or {}
        self.system_config = get_config()
        
        # Training hyperparameters
        self.learning_rate = self.config.get('learning_rate', 
                                           self.system_config.get('reinforcement_learning.training.learning_rate', 0.0003))
        self.gamma = self.config.get('gamma',
                                   self.system_config.get('reinforcement_learning.training.gamma', 0.99))
        self.entropy_coef = self.config.get('entropy_coef',
                                          self.system_config.get('reinforcement_learning.training.entropy_coef', 0.01))
        
        # Optimizer
        self.optimizer = torch.optim.Adam(self.policy_network.parameters(), lr=self.learning_rate)
        
        # Learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=100, gamma=0.95)
        
        # Training state
        self.episode_count = 0
        self.total_steps = 0
        self.best_average_reward = float('-inf')
        
        # Episode buffer
        self.current_episode = Episode([], [], [], [])
        self.episode_buffer = deque(maxlen=1000)  # Keep last 1000 episodes
        
        # Training statistics
        self.training_stats = []
        self.reward_history = deque(maxlen=100)  # For running average
        
        # Baseline for variance reduction
        self.baseline_value = 0.0
        self.baseline_momentum = 0.9
        
    def start_episode(self):
        """Start a new episode."""
        self.current_episode = Episode([], [], [], [])
        logger.debug(f"Started episode {self.episode_count + 1}")
    
    def add_step(self, 
                state: torch.Tensor, 
                action: List[int], 
                reward: float,
                log_prob: torch.Tensor):
        """
        Add a step to the current episode.
        
        Args:
            state: State representation (text embedding)
            action: Action taken (symbolic encoding)
            reward: Reward received
            log_prob: Log probability of the action
        """
        self.current_episode.states.append(state)
        self.current_episode.actions.append(action)
        self.current_episode.rewards.append(reward)
        self.current_episode.log_probs.append(log_prob)
        self.total_steps += 1
    
    def end_episode(self) -> TrainingStats:
        """
        End the current episode and perform REINFORCE update.
        
        Returns:
            Training statistics for this episode
        """
        if len(self.current_episode) == 0:
            logger.warning("Attempting to end episode with no steps")
            return TrainingStats(
                episode=self.episode_count,
                total_episodes=self.episode_count,
                average_reward=0.0,
                episode_reward=0.0,
                policy_loss=0.0,
                entropy_loss=0.0,
                learning_rate=self.learning_rate,
                episode_length=0
            )
        
        # Calculate discounted returns
        returns = self._calculate_returns(self.current_episode.rewards)
        
        # Perform policy update
        policy_loss, entropy_loss = self._update_policy(returns, self.current_episode.log_probs)
        
        # Update statistics
        episode_reward = self.current_episode.total_reward()
        self.reward_history.append(episode_reward)
        average_reward = np.mean(self.reward_history)
        
        # Update baseline
        self.baseline_value = (self.baseline_momentum * self.baseline_value + 
                             (1 - self.baseline_momentum) * episode_reward)
        
        # Store episode
        self.episode_buffer.append(self.current_episode)
        self.episode_count += 1
        
        # Update learning rate
        self.scheduler.step()
        
        # Create training stats
        stats = TrainingStats(
            episode=self.episode_count,
            total_episodes=self.episode_count,
            average_reward=average_reward,
            episode_reward=episode_reward,
            policy_loss=policy_loss,
            entropy_loss=entropy_loss,
            learning_rate=self.optimizer.param_groups[0]['lr'],
            episode_length=len(self.current_episode)
        )
        
        self.training_stats.append(stats)
        
        # Check for best performance
        if average_reward > self.best_average_reward:
            self.best_average_reward = average_reward
            logger.info(f"New best average reward: {average_reward:.4f}")
        
        logger.debug(f"Episode {self.episode_count} completed: "
                    f"reward={episode_reward:.4f}, avg_reward={average_reward:.4f}")
        
        return stats
    
    def _calculate_returns(self, rewards: List[float]) -> torch.Tensor:
        """
        Calculate discounted returns for an episode.
        
        Args:
            rewards: List of rewards for each step
            
        Returns:
            Tensor of discounted returns
        """
        returns = []
        discounted_sum = 0.0
        
        # Calculate returns backwards
        for reward in reversed(rewards):
            discounted_sum = reward + self.gamma * discounted_sum
            returns.insert(0, discounted_sum)
        
        returns_tensor = torch.tensor(returns, dtype=torch.float32)
        
        # Normalize returns for stability
        if len(returns_tensor) > 1:
            returns_tensor = (returns_tensor - returns_tensor.mean()) / (returns_tensor.std() + 1e-8)
        
        return returns_tensor
    
    def _update_policy(self, returns: torch.Tensor, log_probs: List[torch.Tensor]) -> Tuple[float, float]:
        """
        Update the policy using REINFORCE.
        
        Args:
            returns: Discounted returns
            log_probs: Log probabilities of actions taken
            
        Returns:
            Tuple of (policy_loss, entropy_loss)
        """
        # Stack log probabilities
        log_probs_tensor = torch.stack(log_probs)
        
        # Apply baseline for variance reduction
        advantages = returns - self.baseline_value
        
        # Calculate policy loss (negative because we want to maximize)
        policy_loss = -torch.sum(log_probs_tensor * advantages)
        
        # Calculate entropy loss for exploration
        # We need to recalculate the action probabilities for entropy
        entropy_loss = 0.0
        if hasattr(self.policy_network, 'last_action_probs'):
            # If we stored action probabilities, calculate entropy
            for probs in self.policy_network.last_action_probs:
                entropy = -torch.sum(probs * torch.log(probs + 1e-8))
                entropy_loss += entropy
            entropy_loss = -self.entropy_coef * entropy_loss  # Negative to encourage exploration
        
        # Total loss
        total_loss = policy_loss + entropy_loss
        
        # Backward pass
        self.optimizer.zero_grad()
        total_loss.backward()
        
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(self.policy_network.parameters(), max_norm=1.0)
        
        self.optimizer.step()
        
        return policy_loss.item(), entropy_loss.item() if isinstance(entropy_loss, torch.Tensor) else entropy_loss
    
    def get_training_stats(self) -> Dict[str, Any]:
        """Get comprehensive training statistics."""
        if not self.training_stats:
            return {}
        
        recent_stats = self.training_stats[-50:]  # Last 50 episodes
        
        return {
            'total_episodes': self.episode_count,
            'total_steps': self.total_steps,
            'current_learning_rate': self.optimizer.param_groups[0]['lr'],
            'best_average_reward': self.best_average_reward,
            'current_baseline': self.baseline_value,
            'recent_performance': {
                'average_reward': np.mean([s.average_reward for s in recent_stats]),
                'average_episode_length': np.mean([s.episode_length for s in recent_stats]),
                'average_policy_loss': np.mean([s.policy_loss for s in recent_stats]),
                'reward_std': np.std([s.episode_reward for s in recent_stats])
            },
            'training_progress': [
                {
                    'episode': s.episode,
                    'average_reward': s.average_reward,
                    'episode_reward': s.episode_reward,
                    'policy_loss': s.policy_loss,
                    'episode_length': s.episode_length
                }
                for s in self.training_stats[-20:]  # Last 20 episodes
            ]
        }
    
    def save_checkpoint(self, filepath: str):
        """
        Save training checkpoint.
        
        Args:
            filepath: Path to save checkpoint
        """
        checkpoint = {
            'policy_state_dict': self.policy_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'episode_count': self.episode_count,
            'total_steps': self.total_steps,
            'best_average_reward': self.best_average_reward,
            'baseline_value': self.baseline_value,
            'reward_history': list(self.reward_history),
            'training_stats': self.training_stats[-100:],  # Save last 100 episodes
            'config': self.config
        }
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        torch.save(checkpoint, filepath)
        logger.info(f"Checkpoint saved to {filepath}")
    
    def load_checkpoint(self, filepath: str):
        """
        Load training checkpoint.
        
        Args:
            filepath: Path to load checkpoint from
        """
        if not os.path.exists(filepath):
            logger.warning(f"Checkpoint file not found: {filepath}")
            return
        
        checkpoint = torch.load(filepath, map_location='cpu')
        
        self.policy_network.load_state_dict(checkpoint['policy_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.episode_count = checkpoint['episode_count']
        self.total_steps = checkpoint['total_steps']
        self.best_average_reward = checkpoint['best_average_reward']
        self.baseline_value = checkpoint['baseline_value']
        
        # Restore history
        self.reward_history = deque(checkpoint['reward_history'], maxlen=100)
        self.training_stats = checkpoint['training_stats']
        
        logger.info(f"Checkpoint loaded from {filepath}")
        logger.info(f"Resumed training at episode {self.episode_count}")
    
    def evaluate_policy(self, num_episodes: int = 10) -> Dict[str, Any]:
        """
        Evaluate the current policy.
        
        Args:
            num_episodes: Number of episodes to evaluate
            
        Returns:
            Evaluation results
        """
        self.policy_network.eval()
        
        episode_rewards = []
        episode_lengths = []
        
        for _ in range(num_episodes):
            # For evaluation, we would run the policy without training
            # This is a simplified version - in practice, you'd run the full environment
            episode_reward = np.mean(self.reward_history) if self.reward_history else 0.0
            episode_length = np.mean([len(ep) for ep in list(self.episode_buffer)[-10:] if ep]) if self.episode_buffer else 0
            
            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)
        
        self.policy_network.train()
        
        return {
            'mean_reward': np.mean(episode_rewards),
            'std_reward': np.std(episode_rewards),
            'mean_episode_length': np.mean(episode_lengths),
            'std_episode_length': np.std(episode_lengths),
            'num_episodes': num_episodes
        }
    
    def reset_training(self):
        """Reset training state."""
        self.episode_count = 0
        self.total_steps = 0
        self.best_average_reward = float('-inf')
        self.baseline_value = 0.0
        
        self.current_episode = Episode([], [], [], [])
        self.episode_buffer.clear()
        self.training_stats.clear()
        self.reward_history.clear()
        
        # Reset optimizer
        self.optimizer = torch.optim.Adam(self.policy_network.parameters(), lr=self.learning_rate)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=100, gamma=0.95)
        
        logger.info("Training state reset")

class REINFORCEAgent:
    """Wrapper for using REINFORCE with the Communication Agent."""
    
    def __init__(self, communication_agent):
        """
        Initialize REINFORCE agent wrapper.
        
        Args:
            communication_agent: Communication agent to train
        """
        self.communication_agent = communication_agent
        self.trainer = REINFORCETrainer(communication_agent.encoder)
        
        # Training state
        self.training_active = False
        self.current_state = None
        self.current_action = None
        self.current_log_prob = None
    
    def start_training_episode(self):
        """Start a new training episode."""
        self.trainer.start_episode()
        self.training_active = True
        self.communication_agent.start_episode()
    
    def process_query(self, query: str, text_embedding: torch.Tensor) -> List[int]:
        """
        Process a query and return symbolic encoding.
        
        Args:
            query: Input query
            text_embedding: Text embedding for the query
            
        Returns:
            Symbolic encoding
        """
        # Store state for training
        if self.training_active:
            self.current_state = text_embedding
        
        # Get action from communication agent
        symbolic_encoding = self.communication_agent._encode_message(query)
        
        # Store action for training
        if self.training_active:
            self.current_action = symbolic_encoding.encoding
            
            # Calculate log probability (simplified)
            with torch.no_grad():
                logits = self.communication_agent.encoder.forward(text_embedding.unsqueeze(0))
                log_probs = F.log_softmax(logits, dim=-1)
                
                # Get log prob for actual actions taken
                action_tensor = torch.tensor(symbolic_encoding.encoding[:self.communication_agent.encoder.message_length])
                selected_log_probs = log_probs[0, range(len(action_tensor)), action_tensor]
                self.current_log_prob = selected_log_probs.sum()
        
        return symbolic_encoding.encoding
    
    def receive_reward(self, reward: float):
        """
        Receive reward and update training.
        
        Args:
            reward: Reward value
        """
        if self.training_active and self.current_state is not None:
            self.trainer.add_step(
                state=self.current_state,
                action=self.current_action,
                reward=reward,
                log_prob=self.current_log_prob
            )
            
            # Also update the communication agent
            self.communication_agent.update_from_reward(reward)
    
    def end_training_episode(self) -> TrainingStats:
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
        self.current_state = None
        self.current_action = None
        self.current_log_prob = None
        
        return stats
    
    def get_training_stats(self) -> Dict[str, Any]:
        """Get training statistics."""
        return self.trainer.get_training_stats()
    
    def save_model(self, filepath: str):
        """Save trained model."""
        self.trainer.save_checkpoint(filepath)
    
    def load_model(self, filepath: str):
        """Load trained model."""
        self.trainer.load_checkpoint(filepath)