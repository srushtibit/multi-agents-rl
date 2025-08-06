"""
Support System Environment for Reinforcement Learning.
Simulates the multi-agent support system for training and evaluation.
"""

import asyncio
import logging
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
import random
from enum import Enum

from agents.base_agent import AgentCoordinator, Message, MessageType
from agents.communication.communication_agent import CommunicationAgent
from agents.retrieval.retrieval_agent import RetrievalAgent
from agents.critic.critic_agent import CriticAgent
from agents.escalation.escalation_agent import EscalationAgent
from kb.unified_knowledge_base import get_knowledge_base
from utils.config_loader import get_config

logger = logging.getLogger(__name__)

class TaskType(Enum):
    """Types of support tasks."""
    IT_SUPPORT = "it_support"
    HR_INQUIRY = "hr_inquiry"
    TECHNICAL_ISSUE = "technical_issue"
    ACCOUNT_ACCESS = "account_access"
    GENERAL_QUESTION = "general_question"

@dataclass
class SupportTask:
    """Represents a support task for the environment."""
    task_id: str
    task_type: TaskType
    user_query: str
    language: str
    expected_solution_type: str
    difficulty_level: int  # 1-5 scale
    urgency_level: int  # 1-5 scale
    context: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'task_id': self.task_id,
            'task_type': self.task_type.value,
            'user_query': self.user_query,
            'language': self.language,
            'expected_solution_type': self.expected_solution_type,
            'difficulty_level': self.difficulty_level,
            'urgency_level': self.urgency_level,
            'context': self.context
        }

class SupportEnvironment:
    """Multi-agent support system environment for RL training."""
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the support environment.
        
        Args:
            config: Environment configuration
        """
        self.config = config or {}
        self.system_config = get_config()
        
        # Environment parameters
        self.max_steps = self.config.get('max_steps', 
                                       self.system_config.get('reinforcement_learning.environment.max_steps', 50))
        self.reward_shaping = self.config.get('reward_shaping', True)
        
        # Initialize agents
        self.coordinator = AgentCoordinator()
        self.communication_agent = CommunicationAgent()
        self.retrieval_agent = RetrievalAgent()
        self.critic_agent = CriticAgent()
        self.escalation_agent = EscalationAgent()
        
        # Register agents
        self.coordinator.register_agent(self.communication_agent)
        self.coordinator.register_agent(self.retrieval_agent)
        self.coordinator.register_agent(self.critic_agent)
        self.coordinator.register_agent(self.escalation_agent)
        
        # Knowledge base
        self.knowledge_base = get_knowledge_base()
        
        # Environment state
        self.current_task: Optional[SupportTask] = None
        self.step_count = 0
        self.episode_count = 0
        self.conversation_history: List[Message] = []
        
        # Task generator
        self.task_generator = SupportTaskGenerator()
        
        # Evaluation metrics
        self.episode_metrics = {
            'total_reward': 0.0,
            'steps_taken': 0,
            'task_completed': False,
            'escalation_triggered': False,
            'communication_efficiency': 0.0,
            'retrieval_quality': 0.0,
            'response_time': 0.0
        }
    
    def reset(self, task: Optional[SupportTask] = None) -> Dict[str, Any]:
        """
        Reset the environment for a new episode.
        
        Args:
            task: Optional specific task to use
            
        Returns:
            Initial observation
        """
        # Generate or use provided task
        if task is None:
            self.current_task = self.task_generator.generate_task()
        else:
            self.current_task = task
        
        # Reset environment state
        self.step_count = 0
        self.episode_count += 1
        self.conversation_history.clear()
        
        # Reset agents
        for agent in self.coordinator.agents.values():
            agent.reset()
        
        # Start agents
        self.coordinator.start_all_agents()
        
        # Reset metrics
        self.episode_metrics = {
            'total_reward': 0.0,
            'steps_taken': 0,
            'task_completed': False,
            'escalation_triggered': False,
            'communication_efficiency': 0.0,
            'retrieval_quality': 0.0,
            'response_time': 0.0
        }
        
        logger.info(f"Environment reset for episode {self.episode_count}")
        logger.info(f"Task: {self.current_task.task_type.value} - {self.current_task.user_query[:50]}...")
        
        return self._get_observation()
    
    async def step(self, action: Optional[Dict[str, Any]] = None) -> Tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        """
        Execute one step in the environment.
        
        Args:
            action: Optional action to take (for manual control)
            
        Returns:
            Tuple of (observation, reward, done, info)
        """
        if self.current_task is None:
            raise RuntimeError("Environment not reset. Call reset() first.")
        
        self.step_count += 1
        
        # If this is the first step, inject the user query
        if self.step_count == 1:
            await self._inject_user_query()
        
        # Run one coordination cycle
        await self.coordinator.run_cycle()
        
        # Calculate reward
        reward = self._calculate_reward()
        self.episode_metrics['total_reward'] += reward
        
        # Check if episode is done
        done = self._is_episode_done()
        
        # Get updated observation
        observation = self._get_observation()
        
        # Create info dictionary
        info = self._get_step_info()
        
        self.episode_metrics['steps_taken'] = self.step_count
        
        return observation, reward, done, info
    
    async def _inject_user_query(self):
        """Inject the initial user query into the system."""
        user_message = Message(
            type=MessageType.QUERY,
            content=self.current_task.user_query,
            metadata=self.current_task.context,
            sender="user",
            recipient="communication_agent",
            language=self.current_task.language
        )
        
        # Send to communication agent
        self.communication_agent.receive_message(user_message)
        self.conversation_history.append(user_message)
        
        logger.debug(f"Injected user query: {self.current_task.user_query}")
    
    def _get_observation(self) -> Dict[str, Any]:
        """Get current environment observation."""
        # Get recent messages
        recent_messages = []
        for message in self.conversation_history[-5:]:  # Last 5 messages
            recent_messages.append({
                'type': message.type.value,
                'sender': message.sender,
                'recipient': message.recipient,
                'content_length': len(message.content),
                'has_symbolic_encoding': message.symbolic_encoding is not None,
                'language': message.language
            })
        
        # Get agent states
        agent_states = {}
        for agent_id, agent in self.coordinator.agents.items():
            agent_states[agent_id] = {
                'is_active': agent.is_active,
                'message_queue_size': len(agent.message_queue),
                'messages_processed': agent.stats['messages_processed'],
                'messages_sent': agent.stats['messages_sent'],
                'errors': agent.stats['errors']
            }
        
        observation = {
            'task': self.current_task.to_dict() if self.current_task else {},
            'step_count': self.step_count,
            'recent_messages': recent_messages,
            'agent_states': agent_states,
            'conversation_length': len(self.conversation_history),
            'total_reward': self.episode_metrics['total_reward']
        }
        
        return observation
    
    def _calculate_reward(self) -> float:
        """Calculate reward for the current step."""
        base_reward = 0.0
        
        # Get recent messages for analysis
        recent_messages = self.conversation_history[-3:] if self.conversation_history else []
        
        for message in recent_messages:
            if message.sender in self.coordinator.agents:
                # Reward for successful communication
                if message.type == MessageType.SYMBOLIC:
                    base_reward += 0.1  # Reward for symbolic encoding
                
                if message.type == MessageType.RESPONSE:
                    # Check if response has retrieval context
                    if 'retrieval_context' in message.metadata:
                        retrieval_context = message.metadata['retrieval_context']
                        max_similarity = retrieval_context.get('max_similarity', 0)
                        base_reward += 0.2 * max_similarity  # Reward for relevant retrieval
                
                if message.type == MessageType.FEEDBACK:
                    # Use critic evaluation as reward signal
                    if 'evaluation_result' in message.metadata:
                        evaluation = message.metadata['evaluation_result']
                        critic_score = evaluation.get('overall_score', 0)
                        base_reward += critic_score  # Direct critic feedback
        
        # Penalty for excessive steps
        if self.step_count > self.max_steps * 0.8:
            base_reward -= 0.05
        
        # Bonus for task completion
        if self._is_task_completed():
            base_reward += 1.0
            self.episode_metrics['task_completed'] = True
        
        # Penalty for escalation (indicates system couldn't handle the task)
        if self._check_escalation_triggered():
            base_reward -= 0.3
            self.episode_metrics['escalation_triggered'] = True
        
        # Reward shaping based on communication efficiency
        if self.reward_shaping:
            comm_efficiency = self._calculate_communication_efficiency()
            base_reward += 0.1 * comm_efficiency
            self.episode_metrics['communication_efficiency'] = comm_efficiency
        
        return base_reward
    
    def _is_episode_done(self) -> bool:
        """Check if the episode is finished."""
        # Episode ends if:
        # 1. Maximum steps reached
        if self.step_count >= self.max_steps:
            return True
        
        # 2. Task is completed (received satisfactory response)
        if self._is_task_completed():
            return True
        
        # 3. Escalation triggered for critical issues
        if self._check_escalation_triggered() and self.current_task.urgency_level >= 4:
            return True
        
        # 4. System error or no progress
        if self.step_count > 10 and len(self.conversation_history) <= 2:
            return True  # No meaningful conversation happening
        
        return False
    
    def _is_task_completed(self) -> bool:
        """Check if the current task has been completed satisfactorily."""
        if not self.conversation_history:
            return False
        
        # Look for responses with high critic scores
        for message in reversed(self.conversation_history[-5:]):
            if (message.type == MessageType.RESPONSE and 
                'retrieval_context' in message.metadata):
                
                retrieval_context = message.metadata['retrieval_context']
                max_similarity = retrieval_context.get('max_similarity', 0)
                
                # Consider task completed if we have a high-quality response
                if max_similarity > 0.8:
                    return True
        
        # Check for critic feedback indicating completion
        for message in reversed(self.conversation_history[-3:]):
            if (message.type == MessageType.FEEDBACK and
                'evaluation_result' in message.metadata):
                
                evaluation = message.metadata['evaluation_result']
                overall_score = evaluation.get('overall_score', 0)
                
                if overall_score > 0.85:
                    return True
        
        return False
    
    def _check_escalation_triggered(self) -> bool:
        """Check if escalation has been triggered."""
        for message in self.conversation_history:
            if message.type == MessageType.ESCALATION:
                return True
        return False
    
    def _calculate_communication_efficiency(self) -> float:
        """Calculate communication efficiency metric."""
        if len(self.conversation_history) == 0:
            return 0.0
        
        # Count meaningful exchanges
        meaningful_exchanges = 0
        total_messages = len(self.conversation_history)
        
        for message in self.conversation_history:
            if message.type in [MessageType.SYMBOLIC, MessageType.RESPONSE, MessageType.FEEDBACK]:
                meaningful_exchanges += 1
        
        # Efficiency is ratio of meaningful messages to total messages
        efficiency = meaningful_exchanges / max(total_messages, 1)
        
        # Bonus for concise communication
        if total_messages > 0 and total_messages <= 10:
            efficiency += 0.1
        
        return min(1.0, efficiency)
    
    def _get_step_info(self) -> Dict[str, Any]:
        """Get additional information about the current step."""
        info = {
            'step_count': self.step_count,
            'episode_count': self.episode_count,
            'task_type': self.current_task.task_type.value,
            'task_difficulty': self.current_task.difficulty_level,
            'task_urgency': self.current_task.urgency_level,
            'conversation_length': len(self.conversation_history),
            'agents_active': sum(1 for agent in self.coordinator.agents.values() if agent.is_active),
            'total_system_messages': sum(len(agent.message_history) for agent in self.coordinator.agents.values()),
            'episode_metrics': self.episode_metrics.copy()
        }
        
        # Add recent agent statistics
        info['agent_stats'] = {}
        for agent_id, agent in self.coordinator.agents.items():
            info['agent_stats'][agent_id] = agent.get_stats()
        
        return info
    
    def get_episode_summary(self) -> Dict[str, Any]:
        """Get summary of the completed episode."""
        return {
            'episode_count': self.episode_count,
            'task': self.current_task.to_dict() if self.current_task else {},
            'metrics': self.episode_metrics.copy(),
            'conversation_summary': {
                'total_messages': len(self.conversation_history),
                'message_types': {
                    msg_type.value: sum(1 for msg in self.conversation_history if msg.type == msg_type)
                    for msg_type in MessageType
                },
                'participants': list(set(msg.sender for msg in self.conversation_history))
            },
            'agent_performance': {
                agent_id: agent.get_stats()
                for agent_id, agent in self.coordinator.agents.items()
            }
        }
    
    def close(self):
        """Clean up environment resources."""
        self.coordinator.stop_all_agents()
        logger.info("Environment closed")

class SupportTaskGenerator:
    """Generates diverse support tasks for training."""
    
    def __init__(self):
        self.task_templates = {
            TaskType.IT_SUPPORT: [
                "My email is not syncing with the server. Can you help me fix this?",
                "I cannot connect to the VPN. It keeps showing an error message.",
                "The system is running very slowly and I need to complete urgent work.",
                "I forgot my password and cannot access my account.",
                "My computer won't start and I have an important presentation today."
            ],
            TaskType.HR_INQUIRY: [
                "I need information about my health insurance benefits.",
                "How do I request time off for vacation?",
                "I want to know about the company's retirement plan.",
                "What is the policy for working from home?",
                "I need to update my emergency contact information."
            ],
            TaskType.TECHNICAL_ISSUE: [
                "The application crashes every time I try to save my work.",
                "I'm getting a database connection error in the system.",
                "The website is not loading properly on my browser.",
                "I cannot install the required software on my machine.",
                "The printer is not responding to print commands."
            ],
            TaskType.ACCOUNT_ACCESS: [
                "I cannot log into my account - it says my credentials are invalid.",
                "My account seems to be locked and I need access immediately.",
                "I need permission to access the shared folder for my project.",
                "My user privileges were removed and I need them restored.",
                "I cannot access the customer database for my daily work."
            ],
            TaskType.GENERAL_QUESTION: [
                "Where can I find the employee handbook?",
                "What are the office hours for the IT support desk?",
                "How do I contact the facilities management team?",
                "What is the procedure for reporting security incidents?",
                "Where can I get training on the new software system?"
            ]
        }
        
        self.languages = ['en', 'es', 'de', 'fr']
        self.urgency_keywords = {
            1: [],
            2: ['soon', 'when possible'],
            3: ['important', 'priority'],
            4: ['urgent', 'asap', 'needed today'],
            5: ['critical', 'emergency', 'immediate']
        }
    
    def generate_task(self, task_type: Optional[TaskType] = None) -> SupportTask:
        """Generate a random support task."""
        if task_type is None:
            task_type = random.choice(list(TaskType))
        
        # Select template
        template = random.choice(self.task_templates[task_type])
        
        # Add urgency modifications
        urgency = random.randint(1, 5)
        if urgency > 2 and random.random() < 0.3:  # 30% chance to add urgency keywords
            urgency_words = self.urgency_keywords[urgency]
            if urgency_words:
                template = f"{random.choice(urgency_words).title()}! {template}"
        
        # Random properties
        difficulty = random.randint(1, 5)
        language = random.choice(self.languages)
        
        task = SupportTask(
            task_id=f"task_{random.randint(10000, 99999)}",
            task_type=task_type,
            user_query=template,
            language=language,
            expected_solution_type=self._get_expected_solution_type(task_type),
            difficulty_level=difficulty,
            urgency_level=urgency,
            context={
                'generated': True,
                'template_used': template,
                'time_of_day': random.choice(['morning', 'afternoon', 'evening']),
                'user_experience_level': random.choice(['beginner', 'intermediate', 'expert'])
            }
        )
        
        return task
    
    def _get_expected_solution_type(self, task_type: TaskType) -> str:
        """Get expected solution type for a task."""
        solution_types = {
            TaskType.IT_SUPPORT: 'technical_procedure',
            TaskType.HR_INQUIRY: 'policy_information',
            TaskType.TECHNICAL_ISSUE: 'troubleshooting_steps',
            TaskType.ACCOUNT_ACCESS: 'access_restoration',
            TaskType.GENERAL_QUESTION: 'information_reference'
        }
        return solution_types.get(task_type, 'general_assistance')