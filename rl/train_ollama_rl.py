"""
Training script for Ollama-based Reinforcement Learning

This script trains the communication agent using reinforcement learning
with Ollama models instead of traditional neural networks.
"""

import asyncio
import logging
import json
import os
from datetime import datetime
from typing import Dict, Any, List
import argparse

from agents.communication.communication_agent import CommunicationAgent
from agents.retrieval.retrieval_agent import RetrievalAgent
from agents.critic.critic_agent import CriticAgent
from rl.algorithms.ollama_reinforce import OllamaREINFORCEAgent
from rl.environments.ollama_support_environment import OllamaSupportEnvironment
from utils.config_loader import get_config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class OllamaRLTrainer:
    """Main trainer for Ollama-based RL."""
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the Ollama RL trainer.
        
        Args:
            config: Training configuration
        """
        self.config = config or {}
        self.system_config = get_config()
        
        # Training parameters
        self.num_episodes = self.config.get('num_episodes', 100)
        self.eval_frequency = self.config.get('eval_frequency', 10)
        self.save_frequency = self.config.get('save_frequency', 20)
        
        # Initialize components
        self.communication_agent = CommunicationAgent()
        self.retrieval_agent = RetrievalAgent()
        self.critic_agent = CriticAgent()
        
        # Initialize RL components
        self.rl_agent = OllamaREINFORCEAgent(self.communication_agent)
        self.environment = OllamaSupportEnvironment(self.config.get('environment', {}))
        
        # Training state
        self.training_stats = []
        self.best_performance = 0.0
        
        # Results directory
        self.results_dir = self.config.get('results_dir', 'results/ollama_rl')
        os.makedirs(self.results_dir, exist_ok=True)
        
    async def train(self):
        """Main training loop."""
        logger.info(f"Starting Ollama RL training for {self.num_episodes} episodes")
        
        for episode in range(self.num_episodes):
            logger.info(f"Episode {episode + 1}/{self.num_episodes}")
            
            # Run training episode
            episode_stats = await self._run_training_episode(episode)
            self.training_stats.append(episode_stats)
            
            # Evaluation
            if (episode + 1) % self.eval_frequency == 0:
                eval_stats = await self._run_evaluation()
                logger.info(f"Evaluation - Average Reward: {eval_stats['average_reward']:.4f}")
                
                # Save best model
                if eval_stats['average_reward'] > self.best_performance:
                    self.best_performance = eval_stats['average_reward']
                    self._save_best_model()
            
            # Save checkpoint
            if (episode + 1) % self.save_frequency == 0:
                self._save_checkpoint(episode)
        
        # Final evaluation and save
        final_stats = await self._run_evaluation()
        self._save_final_results(final_stats)
        
        logger.info("Training completed!")
        logger.info(f"Best performance: {self.best_performance:.4f}")
        
    async def _run_training_episode(self, episode_num: int) -> Dict[str, Any]:
        """
        Run a single training episode.
        
        Args:
            episode_num: Episode number
            
        Returns:
            Episode statistics
        """
        # Start RL episode
        self.rl_agent.start_training_episode()
        
        # Reset environment
        state = self.environment.reset()
        
        total_reward = 0.0
        steps = 0
        
        while True:
            # Get current query
            query = state.current_query
            
            # Process query with RL strategy selection
            processed_query, strategy_used = self.rl_agent.process_query_with_rl(query)
            
            # Simulate full pipeline processing
            response_quality = await self._simulate_pipeline(query, processed_query, strategy_used)
            
            # Create action for environment
            action = {
                'strategy': strategy_used,
                'processed_query': processed_query,
                'response_quality': response_quality
            }
            
            # Step environment
            next_state, reward, done, info = self.environment.step(action)
            
            # Provide reward to RL agent
            self.rl_agent.receive_reward(reward, self.environment.get_state_representation())
            
            total_reward += reward
            steps += 1
            state = next_state
            
            if done:
                break
        
        # End RL episode
        rl_stats = self.rl_agent.end_training_episode()
        
        # Get environment results
        env_result = self.environment.get_episode_result()
        
        episode_stats = {
            'episode': episode_num + 1,
            'total_reward': total_reward,
            'steps': steps,
            'resolution_rate': env_result.resolution_rate,
            'user_satisfaction': env_result.user_satisfaction,
            'average_response_quality': env_result.average_response_quality,
            'strategies_used': env_result.strategies_used,
            'rl_stats': rl_stats.__dict__ if rl_stats else {},
            'timestamp': datetime.now().isoformat()
        }
        
        logger.debug(f"Episode {episode_num + 1} completed: reward={total_reward:.4f}, steps={steps}")
        
        return episode_stats
    
    async def _simulate_pipeline(self, original_query: str, processed_query: str, strategy: str) -> float:
        """
        Simulate the full pipeline processing to get response quality.
        
        Args:
            original_query: Original user query
            processed_query: Processed query from communication agent
            strategy: Strategy used
            
        Returns:
            Response quality score (0-1)
        """
        try:
            # Simulate retrieval
            retrieval_results = await self._simulate_retrieval(processed_query)
            
            # Simulate critic evaluation
            quality_score = await self._simulate_critic_evaluation(
                original_query, processed_query, retrieval_results, strategy
            )
            
            return quality_score
            
        except Exception as e:
            logger.error(f"Error in pipeline simulation: {e}")
            return 0.3  # Low score for errors
    
    async def _simulate_retrieval(self, query: str) -> Dict[str, Any]:
        """Simulate retrieval agent processing."""
        # Simple simulation - in practice, this would call the actual retrieval agent
        # For now, we'll use a heuristic based on query characteristics
        
        query_lower = query.lower()
        
        # Simulate retrieval quality based on query characteristics
        if any(keyword in query_lower for keyword in ['password', 'login', 'access']):
            relevance = 0.8
        elif any(keyword in query_lower for keyword in ['slow', 'performance', 'crash']):
            relevance = 0.7
        elif any(keyword in query_lower for keyword in ['email', 'notification']):
            relevance = 0.75
        else:
            relevance = 0.6
        
        return {
            'relevance_score': relevance,
            'num_results': 5,
            'query_processed': query
        }
    
    async def _simulate_critic_evaluation(self, original_query: str, processed_query: str, 
                                        retrieval_results: Dict[str, Any], strategy: str) -> float:
        """Simulate critic agent evaluation."""
        # Simulate evaluation based on multiple factors
        
        # Query processing quality
        processing_quality = self._evaluate_query_processing(original_query, processed_query, strategy)
        
        # Retrieval relevance
        retrieval_quality = retrieval_results.get('relevance_score', 0.5)
        
        # Strategy appropriateness
        strategy_score = self._evaluate_strategy_appropriateness(original_query, strategy)
        
        # Weighted combination
        overall_quality = (
            processing_quality * 0.4 +
            retrieval_quality * 0.4 +
            strategy_score * 0.2
        )
        
        return min(1.0, max(0.0, overall_quality))
    
    def _evaluate_query_processing(self, original: str, processed: str, strategy: str) -> float:
        """Evaluate how well the query was processed."""
        # Simple heuristics for evaluation
        score = 0.5  # Base score
        
        # Length appropriateness
        if 3 <= len(processed.split()) <= 12:
            score += 0.2
        
        # Keyword preservation
        original_words = set(original.lower().split())
        processed_words = set(processed.lower().split())
        
        # Important keywords should be preserved
        important_keywords = {'password', 'login', 'email', 'access', 'error', 'slow', 'crash'}
        original_important = original_words & important_keywords
        processed_important = processed_words & important_keywords
        
        if original_important and processed_important:
            preservation_rate = len(processed_important) / len(original_important)
            score += preservation_rate * 0.3
        
        return min(1.0, score)
    
    def _evaluate_strategy_appropriateness(self, query: str, strategy: str) -> float:
        """Evaluate if the strategy is appropriate for the query."""
        query_lower = query.lower()
        
        # Simple heuristics for strategy evaluation
        if strategy == 'keyword_focused':
            if any(kw in query_lower for kw in ['password', 'login', 'access', 'permission']):
                return 0.9
            return 0.6
        
        elif strategy == 'context_enhanced':
            if any(kw in query_lower for kw in ['slow', 'performance', 'complex', 'multiple']):
                return 0.9
            return 0.6
        
        elif strategy == 'intent_based':
            if any(kw in query_lower for kw in ['how', 'what', 'why', 'help', 'need']):
                return 0.9
            return 0.6
        
        elif strategy == 'direct_rewrite':
            if len(query.split()) <= 8:  # Simple queries
                return 0.9
            return 0.6
        
        return 0.5  # Default score
    
    async def _run_evaluation(self, num_episodes: int = 10) -> Dict[str, Any]:
        """
        Run evaluation episodes.
        
        Args:
            num_episodes: Number of evaluation episodes
            
        Returns:
            Evaluation statistics
        """
        logger.info(f"Running evaluation for {num_episodes} episodes")
        
        eval_rewards = []
        eval_resolutions = []
        eval_satisfactions = []
        
        for _ in range(num_episodes):
            # Reset environment
            state = self.environment.reset()
            
            total_reward = 0.0
            
            while True:
                query = state.current_query
                
                # Use best strategy (no exploration)
                processed_query, strategy_used = self.rl_agent.process_query_with_rl(query)
                
                # Simulate pipeline
                response_quality = await self._simulate_pipeline(query, processed_query, strategy_used)
                
                action = {
                    'strategy': strategy_used,
                    'processed_query': processed_query,
                    'response_quality': response_quality
                }
                
                next_state, reward, done, info = self.environment.step(action)
                total_reward += reward
                state = next_state
                
                if done:
                    break
            
            # Record results
            env_result = self.environment.get_episode_result()
            eval_rewards.append(total_reward)
            eval_resolutions.append(env_result.resolution_rate)
            eval_satisfactions.append(env_result.user_satisfaction)
        
        return {
            'average_reward': sum(eval_rewards) / len(eval_rewards),
            'average_resolution_rate': sum(eval_resolutions) / len(eval_resolutions),
            'average_satisfaction': sum(eval_satisfactions) / len(eval_satisfactions),
            'num_episodes': num_episodes
        }
    
    def _save_best_model(self):
        """Save the best performing model."""
        best_model_path = os.path.join(self.results_dir, 'best_model.json')
        self.rl_agent.save_model(best_model_path)
        logger.info(f"Best model saved to {best_model_path}")
    
    def _save_checkpoint(self, episode: int):
        """Save training checkpoint."""
        checkpoint_path = os.path.join(self.results_dir, f'checkpoint_episode_{episode + 1}.json')
        
        checkpoint_data = {
            'episode': episode + 1,
            'training_stats': self.training_stats,
            'best_performance': self.best_performance,
            'rl_stats': self.rl_agent.get_training_stats(),
            'config': self.config,
            'timestamp': datetime.now().isoformat()
        }
        
        with open(checkpoint_path, 'w') as f:
            json.dump(checkpoint_data, f, indent=2)
        
        logger.info(f"Checkpoint saved to {checkpoint_path}")
    
    def _save_final_results(self, final_stats: Dict[str, Any]):
        """Save final training results."""
        results_path = os.path.join(self.results_dir, 'final_results.json')
        
        final_results = {
            'training_config': self.config,
            'num_episodes': self.num_episodes,
            'best_performance': self.best_performance,
            'final_evaluation': final_stats,
            'training_stats': self.training_stats,
            'rl_stats': self.rl_agent.get_training_stats(),
            'timestamp': datetime.now().isoformat()
        }
        
        with open(results_path, 'w') as f:
            json.dump(final_results, f, indent=2)
        
        logger.info(f"Final results saved to {results_path}")

async def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description='Train Ollama RL Agent')
    parser.add_argument('--episodes', type=int, default=100, help='Number of training episodes')
    parser.add_argument('--eval-freq', type=int, default=10, help='Evaluation frequency')
    parser.add_argument('--save-freq', type=int, default=20, help='Save frequency')
    parser.add_argument('--results-dir', type=str, default='results/ollama_rl', help='Results directory')
    
    args = parser.parse_args()
    
    # Training configuration
    config = {
        'num_episodes': args.episodes,
        'eval_frequency': args.eval_freq,
        'save_frequency': args.save_freq,
        'results_dir': args.results_dir,
        'environment': {
            'max_episode_length': 15,
            'reward_weights': {
                'response_quality': 0.35,
                'user_satisfaction': 0.25,
                'resolution_success': 0.20,
                'efficiency': 0.10,
                'strategy_effectiveness': 0.10
            }
        }
    }
    
    # Initialize and run trainer
    trainer = OllamaRLTrainer(config)
    await trainer.train()

if __name__ == "__main__":
    asyncio.run(main())
