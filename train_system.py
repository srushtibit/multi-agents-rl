"""
Comprehensive Training Script for the Multilingual Multi-Agent Support System.
Handles reinforcement learning training, knowledge base building, and system evaluation.
"""

import asyncio
import argparse
import logging
import os
import json
import time
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
import yaml
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm

# Import system components
from agents.base_agent import AgentCoordinator
from agents.communication.communication_agent import CommunicationAgent
from agents.retrieval.retrieval_agent import RetrievalAgent
from agents.critic.critic_agent import CriticAgent
from agents.escalation.escalation_agent import EscalationAgent
from kb.unified_knowledge_base import get_knowledge_base
from rl.environments.support_environment import SupportEnvironment, SupportTaskGenerator, TaskType
from rl.algorithms.reinforce import REINFORCEAgent
from utils.config_loader import get_config
from utils.language_utils import detect_language

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class SystemTrainer:
    """Main trainer class for the multi-agent support system."""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the system trainer.
        
        Args:
            config_path: Path to configuration file
        """
        self.config = get_config()
        
        # Training configuration
        self.training_config = self.config.get('reinforcement_learning.training', {})
        self.max_episodes = self.training_config.get('episodes', 1000)
        self.batch_size = self.training_config.get('batch_size', 32)
        self.evaluation_interval = 50  # Evaluate every 50 episodes
        self.save_interval = 100  # Save model every 100 episodes
        
        # Initialize components
        self.coordinator = None
        self.agents = {}
        self.rl_agent = None
        self.environment = None
        self.task_generator = None
        self.knowledge_base = None
        
        # Training state
        self.current_episode = 0
        self.training_start_time = None
        self.training_metrics = []
        self.evaluation_results = []
        
        # Model paths
        self.model_save_dir = "models"
        self.checkpoint_dir = "checkpoints"
        self.results_dir = "results"
        
        # Create directories
        for directory in [self.model_save_dir, self.checkpoint_dir, self.results_dir, "logs"]:
            os.makedirs(directory, exist_ok=True)
    
    def setup(self):
        """Setup the training environment and agents."""
        logger.info("Setting up training environment...")
        
        try:
            # Initialize knowledge base
            self.knowledge_base = get_knowledge_base()
            logger.info("Knowledge base initialized")
            
            # Build knowledge base from dataset
            self._build_knowledge_base()
            
            # Initialize agents
            self._initialize_agents()
            
            # Initialize RL components
            self._initialize_rl_components()
            
            logger.info("✅ Training setup completed successfully")
            
        except Exception as e:
            logger.error(f"❌ Setup failed: {e}")
            raise
    
    def _build_knowledge_base(self):
        """Build the knowledge base from available datasets."""
        logger.info("Building knowledge base from datasets...")
        
        dataset_dir = "dataset"
        if os.path.exists(dataset_dir):
            try:
                successful, total = self.knowledge_base.add_documents_from_directory(
                    dataset_dir, recursive=True
                )
                logger.info(f"Added {successful}/{total} documents to knowledge base")
                
                # Save knowledge base
                self.knowledge_base.save_index()
                
                # Get statistics
                stats = self.knowledge_base.get_stats()
                logger.info(f"Knowledge base stats: {stats.total_documents} docs, "
                          f"{stats.total_chunks} chunks, {len(stats.languages)} languages")
                
            except Exception as e:
                logger.warning(f"Error building knowledge base: {e}")
        else:
            logger.warning(f"Dataset directory '{dataset_dir}' not found")
    
    def _initialize_agents(self):
        """Initialize all agents."""
        logger.info("Initializing agents...")
        
        # Create coordinator
        self.coordinator = AgentCoordinator()
        
        # Create agents
        self.agents = {
            'communication': CommunicationAgent(),
            'retrieval': RetrievalAgent(),
            'critic': CriticAgent(),
            'escalation': EscalationAgent()
        }
        
        # Register agents with coordinator
        for agent in self.agents.values():
            self.coordinator.register_agent(agent)
        
        # Start agents
        self.coordinator.start_all_agents()
        
        logger.info(f"Initialized and started {len(self.agents)} agents")
    
    def _initialize_rl_components(self):
        """Initialize reinforcement learning components."""
        logger.info("Initializing RL components...")
        
        # Create RL agent wrapper
        self.rl_agent = REINFORCEAgent(self.agents['communication'])
        
        # Create environment
        self.environment = SupportEnvironment()
        
        # Create task generator
        self.task_generator = SupportTaskGenerator()
        
        logger.info("RL components initialized")
    
    async def train(self, num_episodes: Optional[int] = None) -> Dict[str, Any]:
        """
        Train the system using reinforcement learning.
        
        Args:
            num_episodes: Number of episodes to train (uses config default if None)
            
        Returns:
            Training results summary
        """
        if num_episodes is None:
            num_episodes = self.max_episodes
        
        logger.info(f"Starting training for {num_episodes} episodes...")
        self.training_start_time = time.time()
        
        try:
            # Training loop
            with tqdm(total=num_episodes, desc="Training Progress") as pbar:
                for episode in range(num_episodes):
                    self.current_episode = episode + 1
                    
                    # Run training episode
                    episode_metrics = await self._run_training_episode()
                    self.training_metrics.append(episode_metrics)
                    
                    # Update progress bar
                    pbar.set_postfix({
                        'Reward': f"{episode_metrics['total_reward']:.3f}",
                        'Steps': episode_metrics['steps'],
                        'Avg Reward': f"{np.mean([m['total_reward'] for m in self.training_metrics[-10:]]):.3f}"
                    })
                    pbar.update(1)
                    
                    # Periodic evaluation
                    if episode % self.evaluation_interval == 0:
                        await self._evaluate_system()
                    
                    # Periodic saving
                    if episode % self.save_interval == 0:
                        self._save_checkpoint()
                    
                    # Log progress
                    if episode % 50 == 0:
                        self._log_training_progress()
            
            # Final evaluation and save
            await self._evaluate_system()
            self._save_final_model()
            
            # Generate training summary
            summary = self._generate_training_summary()
            
            logger.info("🎉 Training completed successfully!")
            return summary
            
        except Exception as e:
            logger.error(f"❌ Training failed: {e}")
            raise
    
    async def _run_training_episode(self) -> Dict[str, Any]:
        """
        Run a single training episode.
        
        Returns:
            Episode metrics
        """
        # Generate random task
        task = self.task_generator.generate_task()
        
        # Reset environment
        self.environment.reset(task)
        
        # Start RL episode
        self.rl_agent.start_training_episode()
        
        episode_start_time = time.time()
        total_reward = 0.0
        steps = 0
        done = False
        
        # Episode loop
        while not done and steps < 50:  # Max 50 steps per episode
            # Step environment
            observation, reward, done, info = await self.environment.step()
            
            # Update RL agent
            if reward != 0:  # Only update when there's meaningful reward
                self.rl_agent.receive_reward(reward)
            
            total_reward += reward
            steps += 1
        
        # End RL episode
        rl_stats = self.rl_agent.end_training_episode()
        
        # Calculate episode time
        episode_time = time.time() - episode_start_time
        
        # Create episode metrics
        metrics = {
            'episode': self.current_episode,
            'task_type': task.task_type.value,
            'task_difficulty': task.difficulty_level,
            'task_urgency': task.urgency_level,
            'language': task.language,
            'total_reward': total_reward,
            'steps': steps,
            'episode_time': episode_time,
            'task_completed': info.get('episode_metrics', {}).get('task_completed', False),
            'escalation_triggered': info.get('episode_metrics', {}).get('escalation_triggered', False),
            'rl_episode_reward': rl_stats.episode_reward if rl_stats else 0.0,
            'rl_policy_loss': rl_stats.policy_loss if rl_stats else 0.0
        }
        
        return metrics
    
    async def _evaluate_system(self) -> Dict[str, Any]:
        """
        Evaluate the current system performance.
        
        Returns:
            Evaluation results
        """
        logger.info(f"Evaluating system at episode {self.current_episode}...")
        
        evaluation_start_time = time.time()
        
        # Test on different task types
        task_types = list(TaskType)
        evaluation_metrics = {
            'episode': self.current_episode,
            'timestamp': datetime.now().isoformat(),
            'task_performance': {},
            'overall_metrics': {}
        }
        
        total_episodes = 0
        total_reward = 0.0
        total_steps = 0
        completed_tasks = 0
        escalations = 0
        
        # Test each task type
        for task_type in task_types:
            task_metrics = []
            
            # Run 5 test episodes for each task type
            for _ in range(5):
                # Generate task
                task = self.task_generator.generate_task(task_type)
                
                # Run episode without training
                self.environment.reset(task)
                
                episode_reward = 0.0
                episode_steps = 0
                done = False
                
                while not done and episode_steps < 30:  # Shorter episodes for evaluation
                    observation, reward, done, info = await self.environment.step()
                    episode_reward += reward
                    episode_steps += 1
                
                # Record metrics
                episode_metrics = info.get('episode_metrics', {})
                task_metrics.append({
                    'reward': episode_reward,
                    'steps': episode_steps,
                    'completed': episode_metrics.get('task_completed', False),
                    'escalated': episode_metrics.get('escalation_triggered', False)
                })
                
                # Update totals
                total_episodes += 1
                total_reward += episode_reward
                total_steps += episode_steps
                if episode_metrics.get('task_completed', False):
                    completed_tasks += 1
                if episode_metrics.get('escalation_triggered', False):
                    escalations += 1
            
            # Calculate task type performance
            evaluation_metrics['task_performance'][task_type.value] = {
                'average_reward': np.mean([m['reward'] for m in task_metrics]),
                'average_steps': np.mean([m['steps'] for m in task_metrics]),
                'completion_rate': np.mean([m['completed'] for m in task_metrics]),
                'escalation_rate': np.mean([m['escalated'] for m in task_metrics])
            }
        
        # Calculate overall metrics
        evaluation_metrics['overall_metrics'] = {
            'average_reward': total_reward / total_episodes,
            'average_steps': total_steps / total_episodes,
            'completion_rate': completed_tasks / total_episodes,
            'escalation_rate': escalations / total_episodes,
            'evaluation_time': time.time() - evaluation_start_time
        }
        
        # Store evaluation results
        self.evaluation_results.append(evaluation_metrics)
        
        # Log results
        overall = evaluation_metrics['overall_metrics']
        logger.info(f"Evaluation results - Avg Reward: {overall['average_reward']:.3f}, "
                   f"Completion Rate: {overall['completion_rate']:.2%}, "
                   f"Escalation Rate: {overall['escalation_rate']:.2%}")
        
        return evaluation_metrics
    
    def _save_checkpoint(self):
        """Save training checkpoint."""
        checkpoint_path = os.path.join(
            self.checkpoint_dir, 
            f"checkpoint_episode_{self.current_episode}.pt"
        )
        
        try:
            # Save RL model
            self.rl_agent.save_model(checkpoint_path)
            
            # Save training state
            state_path = checkpoint_path.replace('.pt', '_state.pkl')
            training_state = {
                'current_episode': self.current_episode,
                'training_metrics': self.training_metrics,
                'evaluation_results': self.evaluation_results,
                'training_start_time': self.training_start_time
            }
            
            with open(state_path, 'wb') as f:
                pickle.dump(training_state, f)
            
            logger.info(f"Checkpoint saved at episode {self.current_episode}")
            
        except Exception as e:
            logger.error(f"Error saving checkpoint: {e}")
    
    def _save_final_model(self):
        """Save the final trained model."""
        try:
            # Save RL model
            model_path = os.path.join(self.model_save_dir, "final_communication_model.pt")
            self.rl_agent.save_model(model_path)
            
            # Save agent models
            for agent_name, agent in self.agents.items():
                if hasattr(agent, 'save_model'):
                    agent_path = os.path.join(self.model_save_dir, f"{agent_name}_agent_model.pt")
                    agent.save_model(agent_path)
            
            # Save training results
            results_path = os.path.join(self.results_dir, f"training_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
            results_data = {
                'training_metrics': self.training_metrics,
                'evaluation_results': self.evaluation_results,
                'final_summary': self._generate_training_summary()
            }
            
            with open(results_path, 'w') as f:
                json.dump(results_data, f, indent=2)
            
            logger.info(f"Final model and results saved")
            
        except Exception as e:
            logger.error(f"Error saving final model: {e}")
    
    def _log_training_progress(self):
        """Log current training progress."""
        if not self.training_metrics:
            return
        
        # Calculate recent performance
        recent_metrics = self.training_metrics[-50:]  # Last 50 episodes
        avg_reward = np.mean([m['total_reward'] for m in recent_metrics])
        avg_steps = np.mean([m['steps'] for m in recent_metrics])
        completion_rate = np.mean([m['task_completed'] for m in recent_metrics])
        
        # Calculate training time
        elapsed_time = time.time() - self.training_start_time
        episodes_per_hour = self.current_episode / (elapsed_time / 3600)
        
        logger.info(f"Episode {self.current_episode}/{self.max_episodes} - "
                   f"Avg Reward: {avg_reward:.3f}, "
                   f"Avg Steps: {avg_steps:.1f}, "
                   f"Completion Rate: {completion_rate:.2%}, "
                   f"Speed: {episodes_per_hour:.1f} episodes/hour")
    
    def _generate_training_summary(self) -> Dict[str, Any]:
        """Generate comprehensive training summary."""
        if not self.training_metrics:
            return {}
        
        # Calculate overall statistics
        total_training_time = time.time() - self.training_start_time if self.training_start_time else 0
        
        rewards = [m['total_reward'] for m in self.training_metrics]
        steps = [m['steps'] for m in self.training_metrics]
        completions = [m['task_completed'] for m in self.training_metrics]
        
        summary = {
            'training_overview': {
                'total_episodes': len(self.training_metrics),
                'total_training_time': total_training_time,
                'episodes_per_hour': len(self.training_metrics) / (total_training_time / 3600) if total_training_time > 0 else 0,
                'training_start': datetime.fromtimestamp(self.training_start_time).isoformat() if self.training_start_time else None,
                'training_end': datetime.now().isoformat()
            },
            'performance_metrics': {
                'average_reward': np.mean(rewards),
                'best_reward': np.max(rewards),
                'reward_std': np.std(rewards),
                'average_steps': np.mean(steps),
                'completion_rate': np.mean(completions),
                'final_100_avg_reward': np.mean(rewards[-100:]) if len(rewards) >= 100 else np.mean(rewards)
            },
            'learning_progress': {
                'reward_improvement': (np.mean(rewards[-100:]) - np.mean(rewards[:100])) if len(rewards) >= 200 else 0,
                'convergence_episode': self._find_convergence_episode(),
                'learning_stability': self._calculate_learning_stability()
            },
            'task_analysis': self._analyze_task_performance(),
            'language_analysis': self._analyze_language_performance(),
            'agent_performance': self._get_agent_performance_summary()
        }
        
        return summary
    
    def _find_convergence_episode(self) -> Optional[int]:
        """Find the episode where learning converged."""
        if len(self.training_metrics) < 100:
            return None
        
        rewards = [m['total_reward'] for m in self.training_metrics]
        
        # Look for stable performance (low variance) in recent episodes
        window_size = 50
        for i in range(window_size, len(rewards)):
            window = rewards[i-window_size:i]
            if np.std(window) < 0.1 and np.mean(window) > 0.6:  # Stable and good performance
                return i
        
        return None
    
    def _calculate_learning_stability(self) -> float:
        """Calculate learning stability metric."""
        if len(self.training_metrics) < 50:
            return 0.0
        
        rewards = [m['total_reward'] for m in self.training_metrics[-50:]]
        return 1.0 / (1.0 + np.std(rewards))  # Higher values = more stable
    
    def _analyze_task_performance(self) -> Dict[str, Any]:
        """Analyze performance by task type."""
        task_analysis = {}
        
        for task_type in TaskType:
            task_metrics = [m for m in self.training_metrics if m['task_type'] == task_type.value]
            
            if task_metrics:
                task_analysis[task_type.value] = {
                    'episodes': len(task_metrics),
                    'average_reward': np.mean([m['total_reward'] for m in task_metrics]),
                    'completion_rate': np.mean([m['task_completed'] for m in task_metrics]),
                    'average_steps': np.mean([m['steps'] for m in task_metrics]),
                    'escalation_rate': np.mean([m['escalation_triggered'] for m in task_metrics])
                }
        
        return task_analysis
    
    def _analyze_language_performance(self) -> Dict[str, Any]:
        """Analyze performance by language."""
        language_analysis = {}
        
        languages = set(m['language'] for m in self.training_metrics)
        
        for language in languages:
            lang_metrics = [m for m in self.training_metrics if m['language'] == language]
            
            if lang_metrics:
                language_analysis[language] = {
                    'episodes': len(lang_metrics),
                    'average_reward': np.mean([m['total_reward'] for m in lang_metrics]),
                    'completion_rate': np.mean([m['task_completed'] for m in lang_metrics]),
                    'average_steps': np.mean([m['steps'] for m in lang_metrics])
                }
        
        return language_analysis
    
    def _get_agent_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary for all agents."""
        agent_summary = {}
        
        for agent_name, agent in self.agents.items():
            stats = agent.get_stats()
            
            agent_summary[agent_name] = {
                'messages_processed': stats.get('messages_processed', 0),
                'messages_sent': stats.get('messages_sent', 0),
                'errors': stats.get('errors', 0),
                'uptime': stats.get('uptime', 0),
                'success_rate': (stats.get('messages_processed', 0) - stats.get('errors', 0)) / max(stats.get('messages_processed', 0), 1)
            }
            
            # Agent-specific metrics
            if agent_name == 'retrieval' and hasattr(agent, 'get_retrieval_stats'):
                agent_summary[agent_name].update(agent.get_retrieval_stats())
            elif agent_name == 'critic' and hasattr(agent, 'get_evaluation_stats'):
                agent_summary[agent_name].update(agent.get_evaluation_stats())
            elif agent_name == 'escalation' and hasattr(agent, 'get_escalation_stats'):
                agent_summary[agent_name].update(agent.get_escalation_stats())
        
        return agent_summary
    
    async def evaluate_pretrained_model(self, model_path: str, num_episodes: int = 100) -> Dict[str, Any]:
        """
        Evaluate a pre-trained model.
        
        Args:
            model_path: Path to the trained model
            num_episodes: Number of evaluation episodes
            
        Returns:
            Evaluation results
        """
        logger.info(f"Evaluating pre-trained model from {model_path}")
        
        try:
            # Load model
            self.rl_agent.load_model(model_path)
            
            # Set to evaluation mode
            self.agents['communication'].set_training_mode(False)
            
            # Run evaluation episodes
            evaluation_metrics = []
            
            with tqdm(total=num_episodes, desc="Evaluation Progress") as pbar:
                for episode in range(num_episodes):
                    # Generate task
                    task = self.task_generator.generate_task()
                    
                    # Run episode
                    self.environment.reset(task)
                    
                    episode_reward = 0.0
                    episode_steps = 0
                    done = False
                    
                    while not done and episode_steps < 30:
                        observation, reward, done, info = await self.environment.step()
                        episode_reward += reward
                        episode_steps += 1
                    
                    # Record metrics
                    episode_metrics = info.get('episode_metrics', {})
                    evaluation_metrics.append({
                        'episode': episode + 1,
                        'task_type': task.task_type.value,
                        'language': task.language,
                        'reward': episode_reward,
                        'steps': episode_steps,
                        'completed': episode_metrics.get('task_completed', False),
                        'escalated': episode_metrics.get('escalation_triggered', False)
                    })
                    
                    pbar.update(1)
            
            # Calculate summary statistics
            rewards = [m['reward'] for m in evaluation_metrics]
            steps = [m['steps'] for m in evaluation_metrics]
            completions = [m['completed'] for m in evaluation_metrics]
            escalations = [m['escalated'] for m in evaluation_metrics]
            
            evaluation_summary = {
                'model_path': model_path,
                'num_episodes': num_episodes,
                'average_reward': np.mean(rewards),
                'reward_std': np.std(rewards),
                'average_steps': np.mean(steps),
                'completion_rate': np.mean(completions),
                'escalation_rate': np.mean(escalations),
                'detailed_metrics': evaluation_metrics
            }
            
            logger.info(f"Evaluation completed - Avg Reward: {evaluation_summary['average_reward']:.3f}, "
                       f"Completion Rate: {evaluation_summary['completion_rate']:.2%}")
            
            return evaluation_summary
            
        except Exception as e:
            logger.error(f"Error evaluating model: {e}")
            raise
    
    def cleanup(self):
        """Cleanup resources."""
        try:
            if self.coordinator:
                self.coordinator.stop_all_agents()
            
            # Save knowledge base
            if self.knowledge_base:
                self.knowledge_base.save_index()
            
            logger.info("Cleanup completed")
            
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")

async def main():
    """Main training script entry point."""
    parser = argparse.ArgumentParser(description="Train the Multilingual Multi-Agent Support System")
    
    parser.add_argument("--config", type=str, help="Configuration file path")
    parser.add_argument("--episodes", type=int, default=1000, help="Number of training episodes")
    parser.add_argument("--evaluate", type=str, help="Evaluate pre-trained model (provide model path)")
    parser.add_argument("--eval-episodes", type=int, default=100, help="Number of evaluation episodes")
    parser.add_argument("--setup-only", action="store_true", help="Only setup system without training")
    parser.add_argument("--build-kb", action="store_true", help="Build knowledge base and exit")
    
    args = parser.parse_args()
    
    # Initialize trainer
    trainer = SystemTrainer(args.config)
    
    try:
        # Setup system
        trainer.setup()
        
        if args.build_kb:
            logger.info("Knowledge base built successfully!")
            return
        
        if args.setup_only:
            logger.info("System setup completed!")
            return
        
        if args.evaluate:
            # Evaluate pre-trained model
            results = await trainer.evaluate_pretrained_model(args.evaluate, args.eval_episodes)
            
            # Save evaluation results
            eval_results_path = os.path.join(
                trainer.results_dir, 
                f"evaluation_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            )
            
            with open(eval_results_path, 'w') as f:
                json.dump(results, f, indent=2)
            
            logger.info(f"Evaluation results saved to {eval_results_path}")
        
        else:
            # Train the system
            training_summary = await trainer.train(args.episodes)
            
            # Print summary
            logger.info("🎉 Training Summary:")
            logger.info(f"Total Episodes: {training_summary['training_overview']['total_episodes']}")
            logger.info(f"Training Time: {training_summary['training_overview']['total_training_time']:.2f} seconds")
            logger.info(f"Average Reward: {training_summary['performance_metrics']['average_reward']:.3f}")
            logger.info(f"Completion Rate: {training_summary['performance_metrics']['completion_rate']:.2%}")
            
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
    
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise
    
    finally:
        # Cleanup
        trainer.cleanup()

if __name__ == "__main__":
    asyncio.run(main())