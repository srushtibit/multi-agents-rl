"""
Communication Agent with Reinforcement Learning for Symbolic Message Encoding.
This agent learns to encode queries into symbolic messages for efficient inter-agent communication.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import logging
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass

from agents.base_agent import BaseAgent, Message, MessageType
from utils.language_utils import detect_language, translate_to_english, preprocess_multilingual_text

logger = logging.getLogger(__name__)

@dataclass
class SymbolicMessage:
    """Symbolic message representation."""
    encoding: List[int]
    original_text: str
    language: str
    confidence: float
    metadata: Dict[str, Any]

class SymbolicEncoder(nn.Module):
    """Neural network for encoding text into symbolic representations."""
    
    def __init__(self, 
                 vocab_size: int = 1000,
                 embedding_dim: int = 256,
                 hidden_dim: int = 512,
                 message_length: int = 64,
                 num_layers: int = 2):
        """
        Initialize the symbolic encoder.
        
        Args:
            vocab_size: Size of symbolic vocabulary
            embedding_dim: Dimension of input embeddings
            hidden_dim: Hidden layer dimension
            message_length: Length of symbolic messages
            num_layers: Number of transformer layers
        """
        super().__init__()
        
        self.vocab_size = vocab_size
        self.message_length = message_length
        self.hidden_dim = hidden_dim
        
        # Text encoder (simple MLP for now, could be replaced with transformer)
        self.text_encoder = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # Symbolic message generator
        self.message_generator = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, message_length * vocab_size)
        )
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize network weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, text_embedding: torch.Tensor) -> torch.Tensor:
        """
        Forward pass to generate symbolic message.
        
        Args:
            text_embedding: Input text embedding [batch_size, embedding_dim]
            
        Returns:
            Symbolic message logits [batch_size, message_length, vocab_size]
        """
        # Encode text
        encoded = self.text_encoder(text_embedding)
        
        # Generate symbolic message
        message_logits = self.message_generator(encoded)
        
        # Reshape to [batch_size, message_length, vocab_size]
        batch_size = text_embedding.size(0)
        message_logits = message_logits.view(batch_size, self.message_length, self.vocab_size)
        
        return message_logits
    
    def encode_text(self, text_embedding: torch.Tensor, use_sampling: bool = True) -> List[int]:
        """
        Encode text into symbolic message.
        
        Args:
            text_embedding: Input text embedding
            use_sampling: Whether to use sampling or greedy decoding
            
        Returns:
            Symbolic message as list of integers
        """
        self.eval()
        with torch.no_grad():
            if text_embedding.dim() == 1:
                text_embedding = text_embedding.unsqueeze(0)
            
            logits = self.forward(text_embedding)
            
            if use_sampling:
                # Sample from the distribution
                probs = F.softmax(logits, dim=-1)
                symbolic_message = torch.multinomial(probs.view(-1, self.vocab_size), 1)
                symbolic_message = symbolic_message.view(self.message_length).tolist()
            else:
                # Greedy decoding
                symbolic_message = torch.argmax(logits, dim=-1).squeeze(0).tolist()
            
            return symbolic_message

class CommunicationAgent(BaseAgent):
    """Communication agent with symbolic encoding capabilities."""
    
    def __init__(self, agent_id: str = "communication_agent", config: Dict[str, Any] = None):
        super().__init__(agent_id, config)
        
        # Configuration
        self.vocab_size = self.system_config.get('agents.communication.symbolic_vocab_size', 1000)
        self.message_length = self.system_config.get('agents.communication.message_length', 64)
        self.learning_rate = self.system_config.get('agents.communication.learning_rate', 0.001)
        self.hidden_dim = self.system_config.get('agents.communication.hidden_dim', 256)
        self.num_layers = self.system_config.get('agents.communication.num_layers', 2)
        
        # Initialize symbolic encoder
        self.encoder = SymbolicEncoder(
            vocab_size=self.vocab_size,
            embedding_dim=384,  # Assuming sentence transformer embeddings
            hidden_dim=self.hidden_dim,
            message_length=self.message_length,
            num_layers=self.num_layers
        )
        
        # Optimizer for RL training
        self.optimizer = torch.optim.Adam(self.encoder.parameters(), lr=self.learning_rate)
        
        # Training state
        self.training_mode = True
        self.episode_rewards: List[float] = []
        self.episode_messages: List[Tuple[str, List[int], float]] = []
        
        # Symbolic vocabulary (learned mappings)
        self.symbolic_vocab: Dict[int, str] = {}
        self.reverse_vocab: Dict[str, int] = {}
        
        # Message embeddings cache
        self.embedding_cache: Dict[str, torch.Tensor] = {}
        
        # Initialize embedding model for text encoding
        try:
            from sentence_transformers import SentenceTransformer
            model_name = self.system_config.get('languages.embedding_model', 
                                               'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
            self.embedding_model = SentenceTransformer(model_name)
            logger.info(f"Loaded embedding model: {model_name}")
        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}")
            self.embedding_model = None
    
    def get_capabilities(self) -> List[str]:
        """Get agent capabilities."""
        return [
            "symbolic_encoding",
            "multilingual_processing", 
            "message_translation",
            "emergent_communication",
            "reinforcement_learning"
        ]
    
    async def process_message(self, message: Message) -> Optional[Message]:
        """
        Process incoming message and generate symbolic encoding.
        
        Args:
            message: Input message to process
            
        Returns:
            Message with symbolic encoding
        """
        try:
            # Detect language
            language_result = detect_language(message.content)
            
            # Translate to English if needed for processing
            processed_text = message.content
            if language_result.language != 'en':
                translation_result = translate_to_english(message.content, language_result.language)
                if translation_result.confidence > 0.7:
                    processed_text = translation_result.translated_text
            
            # Preprocess text
            clean_text = preprocess_multilingual_text(processed_text, language_result.language)
            
            # Generate symbolic encoding
            symbolic_encoding = self._encode_message(clean_text)
            
            # Create response with symbolic encoding
            response = Message(
                type=MessageType.SYMBOLIC,
                content=processed_text,
                symbolic_encoding=symbolic_encoding.encoding,
                metadata={
                    'original_language': language_result.language,
                    'confidence': symbolic_encoding.confidence,
                    'processed': True,
                    'encoding_metadata': symbolic_encoding.metadata
                },
                sender=self.agent_id,
                recipient="retrieval_agent",  # Default recipient
                language=language_result.language
            )
            
            # Log for training
            if self.training_mode:
                self.episode_messages.append((clean_text, symbolic_encoding.encoding, 0.0))  # Reward will be updated later
            
            self._log_action("symbolic_encoding", {
                "original_length": len(message.content),
                "processed_length": len(processed_text),
                "symbolic_length": len(symbolic_encoding.encoding),
                "language": language_result.language
            })
            
            return response
            
        except Exception as e:
            logger.error(f"Error processing message in communication agent: {e}")
            self._log_action("error", {"error": str(e)}, success=False, error_message=str(e))
            return None
    
    def _encode_message(self, text: str) -> SymbolicMessage:
        """
        Encode text message into symbolic representation.
        
        Args:
            text: Input text to encode
            
        Returns:
            SymbolicMessage with encoding and metadata
        """
        if not self.embedding_model:
            # Fallback encoding
            return SymbolicMessage(
                encoding=list(range(min(len(text.split()), self.message_length))),
                original_text=text,
                language='en',
                confidence=0.5,
                metadata={'method': 'fallback'}
            )
        
        try:
            # Get text embedding
            if text in self.embedding_cache:
                embedding = self.embedding_cache[text]
            else:
                embedding = torch.tensor(self.embedding_model.encode(text), dtype=torch.float32)
                self.embedding_cache[text] = embedding
            
            # Generate symbolic encoding
            symbolic_encoding = self.encoder.encode_text(embedding, use_sampling=self.training_mode)
            
            # Calculate confidence based on model uncertainty
            with torch.no_grad():
                logits = self.encoder.forward(embedding.unsqueeze(0))
                probs = F.softmax(logits, dim=-1)
                entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=-1)
                confidence = 1.0 - (entropy.mean().item() / np.log(self.vocab_size))
            
            return SymbolicMessage(
                encoding=symbolic_encoding,
                original_text=text,
                language='en',
                confidence=confidence,
                metadata={
                    'method': 'neural_encoding',
                    'embedding_dim': embedding.shape[0],
                    'vocab_size': self.vocab_size,
                    'message_length': self.message_length
                }
            )
            
        except Exception as e:
            logger.error(f"Error in symbolic encoding: {e}")
            # Fallback to simple encoding
            words = text.split()[:self.message_length]
            encoding = [hash(word) % self.vocab_size for word in words]
            
            return SymbolicMessage(
                encoding=encoding,
                original_text=text,
                language='en',
                confidence=0.3,
                metadata={'method': 'hash_fallback', 'error': str(e)}
            )
    
    def update_from_reward(self, reward: float, message_index: int = -1):
        """
        Update the model based on reward feedback.
        
        Args:
            reward: Reward signal from critic agent
            message_index: Index of message to update (-1 for latest)
        """
        if not self.training_mode or not self.episode_messages:
            return
        
        try:
            # Update reward for the specified message
            if message_index == -1:
                message_index = len(self.episode_messages) - 1
            
            if 0 <= message_index < len(self.episode_messages):
                text, encoding, _ = self.episode_messages[message_index]
                self.episode_messages[message_index] = (text, encoding, reward)
                
                # Perform REINFORCE update
                self._perform_reinforce_update(text, encoding, reward)
                
                self.episode_rewards.append(reward)
                
                self._log_action("reward_update", {
                    "reward": reward,
                    "message_index": message_index,
                    "episode_length": len(self.episode_messages)
                })
                
        except Exception as e:
            logger.error(f"Error updating from reward: {e}")
    
    def _perform_reinforce_update(self, text: str, encoding: List[int], reward: float):
        """
        Perform REINFORCE policy gradient update.
        
        Args:
            text: Original text
            encoding: Symbolic encoding that was produced
            reward: Reward received for this encoding
        """
        if not self.embedding_model:
            return
        
        try:
            # Get text embedding
            if text in self.embedding_cache:
                embedding = self.embedding_cache[text]
            else:
                embedding = torch.tensor(self.embedding_model.encode(text), dtype=torch.float32)
                self.embedding_cache[text] = embedding
            
            # Forward pass to get logits
            logits = self.encoder.forward(embedding.unsqueeze(0))
            
            # Calculate log probabilities for the actions that were taken
            log_probs = F.log_softmax(logits, dim=-1)
            
            # Get log probabilities for the actual encoding
            encoding_tensor = torch.tensor(encoding[:self.message_length], dtype=torch.long)
            selected_log_probs = log_probs[0, range(len(encoding_tensor)), encoding_tensor]
            
            # Calculate policy gradient loss
            # REINFORCE: loss = -log_prob * reward
            policy_loss = -selected_log_probs.sum() * reward
            
            # Backward pass
            self.optimizer.zero_grad()
            policy_loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.encoder.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            logger.debug(f"REINFORCE update: reward={reward:.3f}, loss={policy_loss.item():.3f}")
            
        except Exception as e:
            logger.error(f"Error in REINFORCE update: {e}")
    
    def start_episode(self):
        """Start a new training episode."""
        self.episode_messages.clear()
        self.episode_rewards.clear()
        self._log_action("start_episode")
    
    def end_episode(self) -> Dict[str, Any]:
        """
        End the current training episode and return statistics.
        
        Returns:
            Episode statistics
        """
        episode_stats = {
            'num_messages': len(self.episode_messages),
            'total_reward': sum(self.episode_rewards),
            'average_reward': np.mean(self.episode_rewards) if self.episode_rewards else 0.0,
            'max_reward': max(self.episode_rewards) if self.episode_rewards else 0.0,
            'min_reward': min(self.episode_rewards) if self.episode_rewards else 0.0
        }
        
        self._log_action("end_episode", episode_stats)
        
        # Clear episode data
        self.episode_messages.clear()
        self.episode_rewards.clear()
        
        return episode_stats
    
    def set_training_mode(self, training: bool):
        """Set training mode."""
        self.training_mode = training
        if training:
            self.encoder.train()
        else:
            self.encoder.eval()
        
        self._log_action("set_training_mode", {"training": training})
    
    def save_model(self, path: str):
        """Save the encoder model."""
        torch.save({
            'model_state_dict': self.encoder.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'vocab_size': self.vocab_size,
            'message_length': self.message_length,
            'hidden_dim': self.hidden_dim,
            'symbolic_vocab': self.symbolic_vocab
        }, path)
        
        logger.info(f"Saved communication model to {path}")
    
    def load_model(self, path: str):
        """Load the encoder model."""
        checkpoint = torch.load(path, map_location='cpu')
        
        self.encoder.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.symbolic_vocab = checkpoint.get('symbolic_vocab', {})
        
        logger.info(f"Loaded communication model from {path}")
    
    def get_symbolic_vocab(self) -> Dict[int, str]:
        """Get the current symbolic vocabulary."""
        return self.symbolic_vocab.copy()
    
    def decode_symbolic_message(self, encoding: List[int]) -> str:
        """
        Attempt to decode a symbolic message back to text.
        
        Args:
            encoding: Symbolic encoding to decode
            
        Returns:
            Decoded text (or description of symbols)
        """
        if not encoding:
            return ""
        
        # Simple decoding based on learned vocabulary
        decoded_parts = []
        for symbol in encoding:
            if symbol in self.symbolic_vocab:
                decoded_parts.append(self.symbolic_vocab[symbol])
            else:
                decoded_parts.append(f"SYM_{symbol}")
        
        return " ".join(decoded_parts) if decoded_parts else f"SYMBOLIC_MESSAGE_{len(encoding)}_TOKENS"