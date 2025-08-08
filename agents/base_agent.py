"""
Base agent class for the multilingual multi-agent support system.
Defines the common interface and functionality for all agents.
"""

import logging
import uuid
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

from utils.config_loader import get_config

logger = logging.getLogger(__name__)

class MessageType(Enum):
    """Types of messages in the multi-agent system."""
    QUERY = "query"
    RESPONSE = "response"
    SYMBOLIC = "symbolic"
    ERROR = "error"
    ESCALATION = "escalation"
    FEEDBACK = "feedback"

@dataclass
class Message:
    """Message structure for inter-agent communication."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    type: MessageType = MessageType.QUERY
    content: str = ""
    symbolic_encoding: Optional[List[int]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    sender: str = ""
    recipient: str = ""
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    language: str = "en"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert message to dictionary."""
        return {
            'id': self.id,
            'type': self.type.value,
            'content': self.content,
            'symbolic_encoding': self.symbolic_encoding,
            'metadata': self.metadata,
            'sender': self.sender,
            'recipient': self.recipient,
            'timestamp': self.timestamp,
            'language': self.language
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Message':
        """Create message from dictionary."""
        data['type'] = MessageType(data['type'])
        return cls(**data)

@dataclass
class AgentAction:
    """Represents an action taken by an agent."""
    action_type: str
    parameters: Dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    agent_id: str = ""
    success: bool = True
    error_message: Optional[str] = None
    
class BaseAgent(ABC):
    """Base class for all agents in the multi-agent system."""
    
    def __init__(self, agent_id: str, config: Dict[str, Any] = None):
        """
        Initialize the base agent.
        
        Args:
            agent_id: Unique identifier for the agent
            config: Agent-specific configuration
        """
        self.agent_id = agent_id
        self.config = config or {}
        self.system_config = get_config()
        
        # Agent state
        self.is_active = False
        self.message_history: List[Message] = []
        self.action_history: List[AgentAction] = []
        
        # Communication
        self.message_queue: List[Message] = []
        self.outbound_queue: List[Message] = []
        
        # Metrics
        self.stats = {
            'messages_processed': 0,
            'messages_sent': 0,
            'actions_taken': 0,
            'errors': 0,
            'start_time': datetime.now().isoformat()
        }
        
        logger.info(f"Initialized agent: {self.agent_id}")
    
    @abstractmethod
    async def process_message(self, message: Message) -> Optional[Message]:
        """
        Process an incoming message.
        
        Args:
            message: The message to process
            
        Returns:
            Optional response message
        """
        pass
    
    @abstractmethod
    def get_capabilities(self) -> List[str]:
        """Get list of agent capabilities."""
        pass
    
    def send_message(self, 
                    recipient: str, 
                    content: str, 
                    message_type: MessageType = MessageType.RESPONSE,
                    symbolic_encoding: Optional[List[int]] = None,
                    metadata: Dict[str, Any] = None) -> Message:
        """
        Send a message to another agent.
        
        Args:
            recipient: ID of the recipient agent
            content: Message content
            message_type: Type of message
            symbolic_encoding: Optional symbolic encoding
            metadata: Additional metadata
            
        Returns:
            The sent message
        """
        message = Message(
            type=message_type,
            content=content,
            symbolic_encoding=symbolic_encoding,
            metadata=metadata or {},
            sender=self.agent_id,
            recipient=recipient
        )
        
        self.outbound_queue.append(message)
        self._log_action("send_message", {"recipient": recipient, "type": message_type.value})
        self.stats['messages_sent'] += 1
        
        logger.debug(f"Agent {self.agent_id} sent message to {recipient}")
        return message
    
    def receive_message(self, message: Message):
        """
        Receive a message from another agent.
        
        Args:
            message: The received message
        """
        self.message_queue.append(message)
        self.message_history.append(message)
        logger.debug(f"Agent {self.agent_id} received message from {message.sender}")
    
    async def run_cycle(self) -> List[Message]:
        """
        Run one processing cycle of the agent.
        
        Returns:
            List of outbound messages generated during this cycle
        """
        if not self.is_active:
            return []
        
        responses = []
        
        # Process all messages in queue
        while self.message_queue:
            message = self.message_queue.pop(0)
            
            try:
                response = await self.process_message(message)
                if response:
                    responses.append(response)
                    self.outbound_queue.append(response)
                
                self.stats['messages_processed'] += 1
                
            except Exception as e:
                logger.error(f"Error processing message in {self.agent_id}: {e}")
                self._log_action("error", {"error": str(e)}, success=False, error_message=str(e))
                self.stats['errors'] += 1
        
        # Return and clear outbound queue
        outbound_messages = self.outbound_queue.copy()
        self.outbound_queue.clear()
        
        return outbound_messages
    
    def start(self):
        """Start the agent."""
        self.is_active = True
        self._log_action("start")
        logger.info(f"Started agent: {self.agent_id}")
    
    def stop(self):
        """Stop the agent."""
        self.is_active = False
        self._log_action("stop")
        logger.info(f"Stopped agent: {self.agent_id}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get agent statistics."""
        return {
            **self.stats,
            'uptime': self._calculate_uptime(),
            'queue_size': len(self.message_queue),
            'history_size': len(self.message_history)
        }
    
    def reset(self):
        """Reset agent state."""
        self.message_queue.clear()
        self.outbound_queue.clear()
        self.message_history.clear()
        self.action_history.clear()
        
        # Reset stats except start time
        start_time = self.stats['start_time']
        self.stats = {
            'messages_processed': 0,
            'messages_sent': 0,
            'actions_taken': 0,
            'errors': 0,
            'start_time': start_time
        }
        
        self._log_action("reset")
        logger.info(f"Reset agent: {self.agent_id}")
    
    def _log_action(self, 
                   action_type: str, 
                   parameters: Dict[str, Any] = None,
                   success: bool = True,
                   error_message: Optional[str] = None):
        """Log an action taken by the agent."""
        action = AgentAction(
            action_type=action_type,
            parameters=parameters or {},
            agent_id=self.agent_id,
            success=success,
            error_message=error_message
        )
        
        self.action_history.append(action)
        self.stats['actions_taken'] += 1
    
    def _calculate_uptime(self) -> float:
        """Calculate agent uptime in seconds."""
        start_time = datetime.fromisoformat(self.stats['start_time'])
        return (datetime.now() - start_time).total_seconds()
    
    def get_recent_messages(self, count: int = 10) -> List[Message]:
        """Get recent messages from history."""
        return self.message_history[-count:]
    
    def get_recent_actions(self, count: int = 10) -> List[AgentAction]:
        """Get recent actions from history."""
        return self.action_history[-count:]

class AgentCoordinator:
    """Coordinates communication between agents."""
    
    def __init__(self):
        self.agents: Dict[str, BaseAgent] = {}
        self.message_log: List[Message] = []
        self.is_running = False
        
    def register_agent(self, agent: BaseAgent):
        """Register an agent with the coordinator."""
        self.agents[agent.agent_id] = agent
        logger.info(f"Registered agent: {agent.agent_id}")
    
    def unregister_agent(self, agent_id: str):
        """Unregister an agent."""
        if agent_id in self.agents:
            del self.agents[agent_id]
            logger.info(f"Unregistered agent: {agent_id}")
    
    async def run_cycle(self):
        """Run one coordination cycle."""
        if not self.is_running:
            return []
        
        # Collect all outbound messages from agents
        all_messages = []
        for agent in self.agents.values():
            messages = await agent.run_cycle()
            all_messages.extend(messages)
        
        # Route messages to recipients
        for message in all_messages:
            await self._route_message(message)
        
        # Return all messages generated this cycle
        return all_messages
    
    async def _route_message(self, message: Message):
        """Route a message to its recipient."""
        recipient_id = message.recipient
        
        if recipient_id == "broadcast":
            # Broadcast to all agents except sender
            for agent_id, agent in self.agents.items():
                if agent_id != message.sender:
                    agent.receive_message(message)
        elif recipient_id in self.agents:
            # Send to specific agent
            self.agents[recipient_id].receive_message(message)
        else:
            logger.warning(f"Unknown recipient: {recipient_id}")
        
        # Log the message
        self.message_log.append(message)
    
    def start_all_agents(self):
        """Start all registered agents."""
        for agent in self.agents.values():
            agent.start()
        self.is_running = True
        logger.info("Started all agents")
    
    def stop_all_agents(self):
        """Stop all registered agents."""
        for agent in self.agents.values():
            agent.stop()
        self.is_running = False
        logger.info("Stopped all agents")
    
    def get_system_stats(self) -> Dict[str, Any]:
        """Get system-wide statistics."""
        agent_stats = {agent_id: agent.get_stats() for agent_id, agent in self.agents.items()}
        
        return {
            'total_agents': len(self.agents),
            'active_agents': sum(1 for agent in self.agents.values() if agent.is_active),
            'total_messages': len(self.message_log),
            'agent_stats': agent_stats
        }