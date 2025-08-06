# 📚 API Reference

## Overview

This document provides comprehensive API reference for the Multilingual Multi-Agent Support System. The system provides multiple interfaces including Python APIs, REST APIs, and CLI interfaces.

## 🐍 Python API Reference

### Core Components

#### AgentCoordinator

Central coordinator for managing multi-agent interactions.

```python
class AgentCoordinator:
    def __init__(self)
    
    def register_agent(self, agent: BaseAgent) -> None
        """Register an agent with the coordinator."""
    
    def unregister_agent(self, agent_id: str) -> None
        """Unregister an agent."""
    
    async def run_cycle(self) -> None
        """Run one coordination cycle."""
    
    def start_all_agents(self) -> None
        """Start all registered agents."""
    
    def stop_all_agents(self) -> None
        """Stop all registered agents."""
    
    def get_system_stats(self) -> Dict[str, Any]
        """Get system-wide statistics."""
```

#### BaseAgent

Abstract base class for all agents.

```python
class BaseAgent(ABC):
    def __init__(self, agent_id: str, config: Dict[str, Any] = None)
    
    @abstractmethod
    async def process_message(self, message: Message) -> Optional[Message]
        """Process an incoming message."""
    
    @abstractmethod
    def get_capabilities(self) -> List[str]
        """Get list of agent capabilities."""
    
    def send_message(self, recipient: str, content: str, 
                    message_type: MessageType = MessageType.RESPONSE,
                    symbolic_encoding: Optional[List[int]] = None,
                    metadata: Dict[str, Any] = None) -> Message
        """Send a message to another agent."""
    
    def receive_message(self, message: Message) -> None
        """Receive a message from another agent."""
    
    async def run_cycle(self) -> List[Message]
        """Run one processing cycle."""
    
    def start(self) -> None
        """Start the agent."""
    
    def stop(self) -> None
        """Stop the agent."""
    
    def get_stats(self) -> Dict[str, Any]
        """Get agent statistics."""
    
    def reset(self) -> None
        """Reset agent state."""
```

### Agent Implementations

#### CommunicationAgent

Handles symbolic encoding and reinforcement learning.

```python
class CommunicationAgent(BaseAgent):
    def __init__(self, agent_id: str = "communication_agent", config: Dict[str, Any] = None)
    
    def get_capabilities(self) -> List[str]
        """Returns: ['symbolic_encoding', 'multilingual_processing', 'message_translation', 
                    'emergent_communication', 'reinforcement_learning']"""
    
    async def process_message(self, message: Message) -> Optional[Message]
        """Process message and generate symbolic encoding."""
    
    def update_from_reward(self, reward: float, message_index: int = -1) -> None
        """Update model based on reward feedback."""
    
    def start_episode(self) -> None
        """Start a new training episode."""
    
    def end_episode(self) -> Dict[str, Any]
        """End current episode and return statistics."""
    
    def set_training_mode(self, training: bool) -> None
        """Set training mode."""
    
    def save_model(self, path: str) -> None
        """Save the encoder model."""
    
    def load_model(self, path: str) -> None
        """Load the encoder model."""
    
    def get_symbolic_vocab(self) -> Dict[int, str]
        """Get current symbolic vocabulary."""
    
    def decode_symbolic_message(self, encoding: List[int]) -> str
        """Decode symbolic message to text."""
```

#### RetrievalAgent

Handles knowledge base search and information retrieval.

```python
class RetrievalAgent(BaseAgent):
    def __init__(self, agent_id: str = "retrieval_agent", config: Dict[str, Any] = None)
    
    def get_capabilities(self) -> List[str]
        """Returns: ['knowledge_retrieval', 'semantic_search', 'multilingual_query_processing',
                    'symbolic_message_interpretation', 'result_ranking', 'context_synthesis']"""
    
    async def process_message(self, message: Message) -> Optional[Message]
        """Process retrieval request and return relevant information."""
    
    def get_retrieval_stats(self) -> Dict[str, Any]
        """Get retrieval statistics."""
    
    def clear_cache(self) -> None
        """Clear query caches."""
    
    def warm_up_knowledge_base(self) -> None
        """Warm up knowledge base with common queries."""
    
    async def process_batch_queries(self, queries: List[str]) -> List[List[SearchResult]]
        """Process multiple queries in batch."""
```

#### CriticAgent

Evaluates responses and provides feedback for learning.

```python
class CriticAgent(BaseAgent):
    def __init__(self, agent_id: str = "critic_agent", config: Dict[str, Any] = None)
    
    def get_capabilities(self) -> List[str]
        """Returns: ['response_evaluation', 'quality_assessment', 'reward_generation',
                    'feedback_provision', 'learning_optimization', 'performance_tracking']"""
    
    async def process_message(self, message: Message) -> Optional[Message]
        """Evaluate response and provide feedback."""
    
    def get_evaluation_stats(self) -> Dict[str, Any]
        """Get evaluation statistics."""
    
    def reset_evaluation_history(self) -> None
        """Reset evaluation history."""
```

#### EscalationAgent

Handles severity detection and automated escalation.

```python
class EscalationAgent(BaseAgent):
    def __init__(self, agent_id: str = "escalation_agent", config: Dict[str, Any] = None)
    
    def get_capabilities(self) -> List[str]
        """Returns: ['severity_assessment', 'escalation_detection', 'email_notification',
                    'urgency_analysis', 'multilingual_severity_detection', 'automated_escalation']"""
    
    async def process_message(self, message: Message) -> Optional[Message]
        """Process message for severity assessment and escalation."""
    
    def get_escalation_stats(self) -> Dict[str, Any]
        """Get escalation statistics."""
    
    def get_escalation_history(self, limit: int = 50) -> List[Dict[str, Any]]
        """Get escalation history."""
    
    def mark_escalation_resolved(self, escalation_id: str) -> bool
        """Mark escalation as resolved."""
    
    def test_email_configuration(self) -> Dict[str, Any]
        """Test email configuration."""
```

### Knowledge Base API

#### UnifiedKnowledgeBase

Central knowledge base for document storage and retrieval.

```python
class UnifiedKnowledgeBase:
    def __init__(self, config_path: str = None, index_path: str = None, metadata_path: str = None)
    
    def add_document(self, file_path: str, force_reprocess: bool = False) -> bool
        """Add a document to the knowledge base."""
    
    def add_documents_from_directory(self, directory: str, recursive: bool = True,
                                   file_patterns: List[str] = None) -> Tuple[int, int]
        """Add all supported documents from a directory."""
    
    def search(self, query: str, max_results: int = None, language: str = None,
              file_types: List[str] = None, min_score: float = None) -> List[SearchResult]
        """Search the knowledge base."""
    
    def get_similar_chunks(self, chunk_id: str, max_results: int = 5) -> List[SearchResult]
        """Get chunks similar to a given chunk."""
    
    def get_stats(self) -> KnowledgeBaseStats
        """Get knowledge base statistics."""
    
    def save_index(self) -> bool
        """Save vector index and metadata to disk."""
    
    def load_index(self) -> bool
        """Load vector index and metadata from disk."""
    
    def clear(self) -> None
        """Clear all data from knowledge base."""
```

#### SearchResult

Result from knowledge base search.

```python
@dataclass
class SearchResult:
    chunk: DocumentChunk
    score: float
    rank: int
    
    def to_dict(self) -> Dict[str, Any]
        """Convert to dictionary representation."""
```

#### DocumentChunk

Represents a chunk of processed document content.

```python
@dataclass
class DocumentChunk:
    id: str
    content: str
    metadata: Dict[str, Any]
    source_file: str
    chunk_index: int
    language: str
    embedding: Optional[List[float]]
    
    def to_dict(self) -> Dict[str, Any]
        """Convert chunk to dictionary."""
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'DocumentChunk'
        """Create chunk from dictionary."""
```

### Document Processors

#### BaseDocumentProcessor

Abstract base for document processors.

```python
class BaseDocumentProcessor(ABC):
    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 50,
                 min_chunk_size: int = 50, max_chunk_size: int = 1000)
    
    @abstractmethod
    def can_process(self, file_path: str) -> bool
        """Check if processor can handle the file."""
    
    @abstractmethod
    def extract_text(self, file_path: str) -> str
        """Extract raw text from document."""
    
    @abstractmethod
    def extract_metadata(self, file_path: str) -> Dict[str, Any]
        """Extract metadata from document."""
    
    def process_document(self, file_path: str) -> ProcessingResult
        """Process document into chunks."""
```

#### Specific Processors

```python
class CSVProcessor(BaseDocumentProcessor):
    """Process CSV files with intelligent column detection."""

class XLSXProcessor(BaseDocumentProcessor):
    """Process Excel files with multi-sheet support."""

class DOCXProcessor(BaseDocumentProcessor):
    """Process Word documents with table extraction."""

class PDFProcessor(BaseDocumentProcessor):
    """Process PDF files with text and table extraction."""

class TXTProcessor(BaseDocumentProcessor):
    """Process plain text files with encoding detection."""
```

### Reinforcement Learning API

#### REINFORCETrainer

Trainer for REINFORCE algorithm.

```python
class REINFORCETrainer:
    def __init__(self, policy_network: nn.Module, config: Dict[str, Any] = None)
    
    def start_episode(self) -> None
        """Start a new episode."""
    
    def add_step(self, state: torch.Tensor, action: List[int], reward: float,
                log_prob: torch.Tensor) -> None
        """Add a step to current episode."""
    
    def end_episode(self) -> TrainingStats
        """End episode and perform REINFORCE update."""
    
    def get_training_stats(self) -> Dict[str, Any]
        """Get comprehensive training statistics."""
    
    def save_checkpoint(self, filepath: str) -> None
        """Save training checkpoint."""
    
    def load_checkpoint(self, filepath: str) -> None
        """Load training checkpoint."""
    
    def evaluate_policy(self, num_episodes: int = 10) -> Dict[str, Any]
        """Evaluate current policy."""
    
    def reset_training(self) -> None
        """Reset training state."""
```

#### SupportEnvironment

RL environment for training.

```python
class SupportEnvironment:
    def __init__(self, config: Dict[str, Any] = None)
    
    def reset(self, task: Optional[SupportTask] = None) -> Dict[str, Any]
        """Reset environment for new episode."""
    
    async def step(self, action: Optional[Dict[str, Any]] = None) -> Tuple[Dict, float, bool, Dict]
        """Execute one environment step."""
    
    def get_episode_summary(self) -> Dict[str, Any]
        """Get summary of completed episode."""
    
    def close(self) -> None
        """Clean up environment resources."""
```

### Language Processing API

#### LanguageDetector

Detect text language with confidence scoring.

```python
class LanguageDetector:
    def __init__(self)
    
    def detect_language(self, text: str, min_length: int = 10) -> LanguageDetectionResult
        """Detect language of input text."""
```

#### MultilingualTranslator

Translate text between languages.

```python
class MultilingualTranslator:
    def __init__(self)
    
    def translate_text(self, text: str, target_language: str = None,
                      source_language: str = None) -> TranslationResult
        """Translate text to target language."""
```

#### TextPreprocessor

Preprocess text for multilingual content.

```python
class TextPreprocessor:
    def __init__(self)
    
    def preprocess_text(self, text: str, language: str = 'en',
                       remove_stopwords: bool = True, stem_words: bool = False,
                       normalize_case: bool = True) -> str
        """Preprocess text for better analysis."""
```

### Configuration API

#### ConfigLoader

Load and manage system configuration.

```python
class ConfigLoader:
    def __init__(self, config_path: Optional[str] = None)
    
    def load_config(self, override_path: Optional[str] = None) -> SystemConfig
        """Load configuration from YAML file."""
    
    def save_config(self, config: SystemConfig, output_path: Optional[str] = None) -> None
        """Save configuration to YAML file."""
    
    def get_config(self) -> SystemConfig
        """Get loaded configuration."""

# Global functions
def get_config() -> SystemConfig
    """Get global configuration instance."""

def load_config(config_path: Optional[str] = None) -> SystemConfig
    """Load configuration from file."""
```

## 🌐 REST API Reference

### Authentication

All API endpoints require authentication in production mode.

```http
Authorization: Bearer <api_key>
Content-Type: application/json
```

### Base URL

```
http://localhost:8000/api/v1
```

### Endpoints

#### System Status

```http
GET /system/status
```

Response:
```json
{
  "status": "healthy",
  "agents": {
    "communication_agent": {"status": "active", "uptime": 3600},
    "retrieval_agent": {"status": "active", "uptime": 3600},
    "critic_agent": {"status": "active", "uptime": 3600},
    "escalation_agent": {"status": "active", "uptime": 3600}
  },
  "knowledge_base": {
    "total_documents": 1250,
    "total_chunks": 15678,
    "languages": ["en", "es", "de", "fr"]
  }
}
```

#### Process Query

```http
POST /query/process
```

Request:
```json
{
  "query": "I cannot access my email account",
  "language": "en",
  "metadata": {
    "user_id": "user123",
    "session_id": "session456"
  }
}
```

Response:
```json
{
  "query_id": "query_789",
  "response": "To resolve email access issues, please follow these steps...",
  "language": "en",
  "evaluation": {
    "overall_score": 0.85,
    "relevance_score": 0.90,
    "accuracy_score": 0.82,
    "completeness_score": 0.88,
    "language_quality_score": 0.80
  },
  "escalation": {
    "triggered": false,
    "severity_level": "medium",
    "severity_score": 0.45
  },
  "retrieval_context": {
    "sources_found": 3,
    "max_similarity": 0.89,
    "search_time_ms": 245
  }
}
```

#### Knowledge Base Search

```http
POST /knowledge/search
```

Request:
```json
{
  "query": "password reset procedure",
  "max_results": 10,
  "min_score": 0.7,
  "language": "en"
}
```

Response:
```json
{
  "results": [
    {
      "chunk_id": "chunk_123",
      "content": "To reset your password, navigate to...",
      "source_file": "IT_Support_Manual.docx",
      "score": 0.92,
      "rank": 1,
      "metadata": {
        "chunk_index": 15,
        "language": "en",
        "content_length": 342
      }
    }
  ],
  "total_results": 7,
  "search_time_ms": 123
}
```

#### Training Control

```http
POST /training/start
```

Request:
```json
{
  "episodes": 100,
  "learning_rate": 0.001,
  "save_checkpoints": true
}
```

Response:
```json
{
  "training_id": "training_456",
  "status": "started",
  "estimated_duration_minutes": 45
}
```

```http
GET /training/{training_id}/status
```

Response:
```json
{
  "training_id": "training_456",
  "status": "running",
  "progress": {
    "current_episode": 45,
    "total_episodes": 100,
    "completion_percentage": 45.0
  },
  "metrics": {
    "average_reward": 0.678,
    "best_reward": 0.892,
    "episodes_per_hour": 120
  }
}
```

#### Escalation Management

```http
GET /escalations
```

Response:
```json
{
  "escalations": [
    {
      "escalation_id": "ESC_20250107_143052_abc123",
      "severity_level": "high",
      "timestamp": "2025-01-07T14:30:52Z",
      "original_query": "URGENT: Cannot access critical systems",
      "email_sent": true,
      "status": "pending"
    }
  ],
  "total": 15,
  "pending": 3
}
```

```http
POST /escalations/{escalation_id}/resolve
```

Request:
```json
{
  "resolution_notes": "Issue resolved by IT team",
  "resolved_by": "admin_user"
}
```

#### Document Management

```http
POST /documents/upload
```

Request: Multipart form data with file upload

Response:
```json
{
  "document_id": "doc_789",
  "filename": "new_policy.pdf",
  "status": "processing",
  "chunks_created": 23,
  "processing_time_seconds": 3.2
}
```

## 🖥️ CLI Reference

### Installation

The CLI is included with the main package installation.

### Basic Usage

```bash
# Get system status
python -m support_system status

# Process a query
python -m support_system query "How do I reset my password?"

# Train the system
python -m support_system train --episodes 1000

# Build knowledge base
python -m support_system kb build --directory dataset/

# Search knowledge base
python -m support_system kb search "email configuration"
```

### Command Reference

#### System Commands

```bash
# Check system status
support-system status [--detailed] [--json]

# Start all agents
support-system start [--config CONFIG_PATH]

# Stop all agents
support-system stop

# Reset system state
support-system reset [--confirm]
```

#### Training Commands

```bash
# Start training
support-system train [OPTIONS]

Options:
  --episodes INTEGER     Number of training episodes [default: 1000]
  --config PATH         Configuration file path
  --learning-rate FLOAT Learning rate [default: 0.001]
  --save-interval INT   Save interval in episodes [default: 100]
  --eval-interval INT   Evaluation interval [default: 50]
  --output-dir PATH     Output directory for models [default: models/]

# Evaluate model
support-system evaluate MODEL_PATH [--episodes INTEGER]

# Resume training
support-system train --resume CHECKPOINT_PATH
```

#### Knowledge Base Commands

```bash
# Build knowledge base
support-system kb build [OPTIONS]

Options:
  --directory PATH      Directory containing documents
  --recursive          Search directories recursively
  --patterns TEXT      File patterns to include (e.g., "*.pdf,*.docx")
  --force              Force reprocessing of existing documents

# Search knowledge base
support-system kb search QUERY [OPTIONS]

Options:
  --max-results INTEGER Maximum results to return [default: 10]
  --min-score FLOAT    Minimum similarity score [default: 0.7]
  --language TEXT      Search language [default: en]
  --format TEXT        Output format (json|table|text) [default: table]

# Knowledge base statistics
support-system kb stats [--detailed]

# Clear knowledge base
support-system kb clear [--confirm]
```

#### Query Commands

```bash
# Process single query
support-system query TEXT [OPTIONS]

Options:
  --language TEXT      Query language [default: auto-detect]
  --format TEXT        Output format (json|text) [default: text]
  --save-conversation  Save conversation to file
  --interactive        Enter interactive mode

# Interactive mode
support-system interactive [--language TEXT]
```

#### Configuration Commands

```bash
# Show configuration
support-system config show [--section SECTION]

# Validate configuration
support-system config validate [CONFIG_PATH]

# Generate default configuration
support-system config init [--output CONFIG_PATH]

# Update configuration value
support-system config set KEY VALUE
```

#### Monitoring Commands

```bash
# Show system metrics
support-system metrics [--interval SECONDS] [--format FORMAT]

# Export logs
support-system logs export [--start DATE] [--end DATE] [--format FORMAT]

# Show agent statistics
support-system agents stats [--agent AGENT_ID]
```

### Environment Variables

```bash
# Configuration file path
export NEXACORP_CONFIG_PATH=/path/to/config.yaml

# Debug mode
export NEXACORP_DEBUG=true

# API settings
export NEXACORP_API_PORT=8000
export NEXACORP_API_HOST=0.0.0.0

# Email settings
export NEXACORP_SMTP_SERVER=smtp.gmail.com
export NEXACORP_SENDER_EMAIL=support@company.com
export NEXACORP_SENDER_PASSWORD=app_password

# Ollama settings
export NEXACORP_OLLAMA_URL=http://localhost:11434
```

### Exit Codes

- `0`: Success
- `1`: General error
- `2`: Configuration error
- `3`: Agent initialization error
- `4`: Knowledge base error
- `5`: Training error
- `130`: Interrupted by user (Ctrl+C)

## 📝 Examples

### Python API Examples

#### Basic Agent Setup

```python
import asyncio
from agents.base_agent import AgentCoordinator
from agents.communication.communication_agent import CommunicationAgent
from agents.retrieval.retrieval_agent import RetrievalAgent
from agents.critic.critic_agent import CriticAgent

async def setup_basic_system():
    # Initialize coordinator
    coordinator = AgentCoordinator()
    
    # Create agents
    comm_agent = CommunicationAgent()
    retrieval_agent = RetrievalAgent()
    critic_agent = CriticAgent()
    
    # Register agents
    coordinator.register_agent(comm_agent)
    coordinator.register_agent(retrieval_agent)
    coordinator.register_agent(critic_agent)
    
    # Start system
    coordinator.start_all_agents()
    
    return coordinator

# Run setup
coordinator = asyncio.run(setup_basic_system())
```

#### Processing Queries

```python
from agents.base_agent import Message, MessageType

async def process_user_query(coordinator, query_text, language="en"):
    # Create user message
    user_message = Message(
        type=MessageType.QUERY,
        content=query_text,
        sender="user",
        recipient="communication_agent",
        language=language
    )
    
    # Send to communication agent
    comm_agent = coordinator.agents["communication_agent"]
    comm_agent.receive_message(user_message)
    
    # Run coordination cycles until response
    for _ in range(10):  # Max 10 cycles
        await coordinator.run_cycle()
    
    # Get system stats
    return coordinator.get_system_stats()

# Process query
result = asyncio.run(process_user_query(
    coordinator, 
    "I need help resetting my password", 
    "en"
))
```

#### Knowledge Base Operations

```python
from kb.unified_knowledge_base import get_knowledge_base

# Get knowledge base instance
kb = get_knowledge_base()

# Add documents
success = kb.add_document("path/to/document.pdf")
successful, total = kb.add_documents_from_directory("dataset/")

# Search knowledge base
results = kb.search(
    query="password reset procedure",
    max_results=10,
    min_score=0.75
)

# Display results
for result in results:
    print(f"Score: {result.score:.3f}")
    print(f"Source: {result.chunk.source_file}")
    print(f"Content: {result.chunk.content[:200]}...")
    print("-" * 50)

# Get statistics
stats = kb.get_stats()
print(f"Documents: {stats.total_documents}")
print(f"Chunks: {stats.total_chunks}")
print(f"Languages: {stats.languages}")
```

#### Training Example

```python
from rl.algorithms.reinforce import REINFORCEAgent
from rl.environments.support_environment import SupportEnvironment

async def train_communication_agent():
    # Setup components
    env = SupportEnvironment()
    comm_agent = CommunicationAgent()
    rl_agent = REINFORCEAgent(comm_agent)
    
    # Training loop
    for episode in range(100):
        # Reset environment and start episode
        env.reset()
        rl_agent.start_training_episode()
        
        done = False
        total_reward = 0
        
        # Episode loop
        while not done:
            obs, reward, done, info = await env.step()
            if reward != 0:
                rl_agent.receive_reward(reward)
            total_reward += reward
        
        # End episode
        stats = rl_agent.end_training_episode()
        print(f"Episode {episode}: Reward = {total_reward:.3f}")
    
    # Save trained model
    rl_agent.save_model("models/trained_communication_model.pt")

# Run training
asyncio.run(train_communication_agent())
```

### REST API Examples

#### Python Requests

```python
import requests

# Base URL
BASE_URL = "http://localhost:8000/api/v1"

# Process query
response = requests.post(f"{BASE_URL}/query/process", json={
    "query": "How do I configure my email client?",
    "language": "en"
})

result = response.json()
print(f"Response: {result['response']}")
print(f"Score: {result['evaluation']['overall_score']}")

# Search knowledge base
response = requests.post(f"{BASE_URL}/knowledge/search", json={
    "query": "email configuration",
    "max_results": 5,
    "min_score": 0.7
})

results = response.json()
for result in results['results']:
    print(f"{result['score']:.3f}: {result['content'][:100]}...")
```

#### cURL Examples

```bash
# Process query
curl -X POST "http://localhost:8000/api/v1/query/process" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "I cannot access my account",
    "language": "en"
  }'

# Search knowledge base
curl -X POST "http://localhost:8000/api/v1/knowledge/search" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "password reset",
    "max_results": 10
  }'

# Get system status
curl -X GET "http://localhost:8000/api/v1/system/status"
```

### CLI Examples

```bash
# Interactive training session
support-system train \
  --episodes 500 \
  --learning-rate 0.001 \
  --eval-interval 25 \
  --save-interval 50

# Build knowledge base from multiple sources
support-system kb build \
  --directory "documents/" \
  --recursive \
  --patterns "*.pdf,*.docx,*.txt"

# Interactive query session
support-system interactive --language en

# Monitor system in real-time
support-system metrics --interval 5 --format json

# Search and export results
support-system kb search "troubleshooting guide" \
  --max-results 20 \
  --format json > search_results.json
```

This API reference provides comprehensive documentation for all interfaces and components of the multilingual multi-agent support system.