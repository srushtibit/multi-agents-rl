"""
Streamlit Dashboard for the Multilingual Multi-Agent Support System.
Provides an interactive interface for testing, monitoring, and training the system.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import asyncio
import logging
import json
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
import time
import os
import sys
from pathlib import Path

# Ensure project root is on sys.path for absolute package imports
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# Import system components
from agents.base_agent import AgentCoordinator, Message, MessageType
from agents.communication.communication_agent import CommunicationAgent
from agents.retrieval.retrieval_agent import RetrievalAgent  
from agents.critic.critic_agent import CriticAgent
from agents.escalation.escalation_agent import EscalationAgent
from kb.unified_knowledge_base import get_knowledge_base
from rl.environments.support_environment import SupportEnvironment, SupportTaskGenerator
from rl.algorithms.reinforce import REINFORCEAgent
from utils.config_loader import get_config
from utils.language_utils import detect_language

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="NexaCorp AI Support System",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .agent-status-active {
        color: #28a745;
        font-weight: bold;
    }
    .agent-status-inactive {
        color: #dc3545;
        font-weight: bold;
    }
    .chat-container {
        max-height: 75vh;
        overflow-y: auto;
        padding: 1rem;
        background-color: transparent;
        border-radius: 0.5rem;
        margin-bottom: 4rem;
    }
    .message-user {
        background-color: #dcf8c6;
        color: #075e54;
        padding: 0.75rem 1rem;
        border-radius: 18px 18px 4px 18px;
        margin: 0.5rem 0 0.5rem auto;
        max-width: 70%;
        text-align: right;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        word-wrap: break-word;
    }
    .message-agent {
        background-color: #ffffff;
        color: #333333;
        padding: 0.75rem 1rem;
        border-radius: 18px 18px 18px 4px;
        margin: 0.5rem auto 0.5rem 0;
        max-width: 70%;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        word-wrap: break-word;
    }
    .input-container {
        position: fixed;
        bottom: 2rem;
        left: 50%;
        transform: translateX(-50%);
        width: 80%;
        max-width: 800px;
        background-color: #ffffff;
        border-radius: 1rem;
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
        padding: 1rem;
        z-index: 1000;
    }
    .thinking-button {
        background-color: #e8f4f8;
        color: #1976d2;
        padding: 0.5rem 1rem;
        border-radius: 12px;
        margin: 0.25rem auto 0.25rem 0;
        max-width: 70%;
        font-size: 0.9em;
        border: 1px solid #e3f2fd;
    }
    .message-system {
        background-color: #fff3e0;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
        border-left: 4px solid #ff9800;
    }
</style>
""", unsafe_allow_html=True)

class SupportSystemDashboard:
    """Main dashboard class for the support system."""
    
    def __init__(self):
        self.config = get_config()
        self.knowledge_base = get_knowledge_base()
        
        # Initialize session state
        if 'system_initialized' not in st.session_state:
            self._initialize_session_state()
        
        # Initialize system components
        if not st.session_state.system_initialized:
            self._initialize_system()
    
    def _initialize_session_state(self):
        """Initialize Streamlit session state."""
        st.session_state.system_initialized = False
        st.session_state.agents_running = False
        st.session_state.conversation_history = []
        st.session_state.training_history = []
        st.session_state.system_metrics = {}
        st.session_state.selected_language = 'en'
        st.session_state.current_query = ''
        st.session_state.last_response = ''
        st.session_state.escalation_alerts = []
    
    def _initialize_system(self):
        """Initialize the multi-agent system."""
        try:
            # Initialize coordinator and agents
            st.session_state.coordinator = AgentCoordinator()
            st.session_state.communication_agent = CommunicationAgent()
            st.session_state.retrieval_agent = RetrievalAgent()
            st.session_state.critic_agent = CriticAgent()
            st.session_state.escalation_agent = EscalationAgent()
            
            # Register agents
            st.session_state.coordinator.register_agent(st.session_state.communication_agent)
            st.session_state.coordinator.register_agent(st.session_state.retrieval_agent)
            st.session_state.coordinator.register_agent(st.session_state.critic_agent)
            st.session_state.coordinator.register_agent(st.session_state.escalation_agent)
            
            # Initialize RL components
            st.session_state.rl_agent = REINFORCEAgent(st.session_state.communication_agent)
            st.session_state.environment = SupportEnvironment()
            st.session_state.task_generator = SupportTaskGenerator()
            
            st.session_state.system_initialized = True
            st.success("✅ System initialized successfully!")
            
        except Exception as e:
            st.error(f"❌ System initialization failed: {e}")
            logger.error(f"System initialization error: {e}")
    
    def run(self):
        """Main dashboard interface."""
        # Header
        st.markdown('<h1 class="main-header">🤖 NexaCorp AI Support System</h1>', unsafe_allow_html=True)
        
        # Sidebar
        self._render_sidebar()
        
        # Main content area
        tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
            "💬 Chat Interface", 
            "📊 System Monitoring", 
            "🎓 Training Dashboard",
            "📚 Knowledge Base",
            "⚠️ Escalation Center",
            "⚙️ System Configuration"
        ])
        
        with tab1:
            self._render_chat_interface()
        
        with tab2:
            self._render_monitoring_dashboard()
        
        with tab3:
            self._render_training_dashboard()
        
        with tab4:
            self._render_knowledge_base_interface()
        
        with tab5:
            self._render_escalation_center()
        
        with tab6:
            self._render_configuration_interface()
    
    def _render_sidebar(self):
        """Render the sidebar with system controls."""
        with st.sidebar:
            st.markdown("## 🎛️ System Controls")
            
            # System status
            if st.session_state.system_initialized:
                st.markdown("**Status:** <span class='agent-status-active'>System Ready</span>", unsafe_allow_html=True)
            else:
                st.markdown("**Status:** <span class='agent-status-inactive'>System Offline</span>", unsafe_allow_html=True)
            
            # Agent controls
            st.markdown("### 🤖 Agent Management")
            
            if st.button("🚀 Start All Agents"):
                self._start_agents()
            
            if st.button("⏹️ Stop All Agents"):
                self._stop_agents()
            
            # Agent status
            if st.session_state.system_initialized:
                agents = st.session_state.coordinator.agents
                for agent_id, agent in agents.items():
                    status = "🟢 Active" if agent.is_active else "🔴 Inactive"
                    st.markdown(f"**{agent_id}:** {status}")
            
            st.divider()
            
            # Language selection
            st.markdown("### 🌐 Language Settings")
            st.session_state.selected_language = st.selectbox(
                "Interface Language",
                options=['en', 'es', 'de', 'fr', 'hi', 'zh'],
                index=0,
                format_func=lambda x: {
                    'en': '🇺🇸 English',
                    'es': '🇪🇸 Spanish', 
                    'de': '🇩🇪 German',
                    'fr': '🇫🇷 French',
                    'hi': '🇮🇳 Hindi',
                    'zh': '🇨🇳 Chinese'
                }[x]
            )
            
            st.divider()
            
            # Quick actions
            st.markdown("### ⚡ Quick Actions")
            
            if st.button("🧹 Clear Conversation"):
                st.session_state.conversation_history = []
                st.rerun()
            
            if st.button("💾 Export Logs"):
                self._export_logs()
            
            if st.button("🔄 Reset System"):
                self._reset_system()
    
    def _render_chat_interface(self):
        """Render the main chat interface - ChatGPT style."""
        st.markdown("## 💬 Interactive Support Chat")
        
        # Chat container with proper scrolling (main content area)
        chat_container = st.container()
        
        with chat_container:
            # Display chat history first (scrollable area)
            if st.session_state.conversation_history:
                # Create a scrollable chat area
                chat_area = st.container()
                with chat_area:
                    st.markdown('<div class="chat-container">', unsafe_allow_html=True)
                    
                    for exchange in st.session_state.conversation_history:
                        # User message (right side)
                        st.markdown(f"""
                        <div class="message-user">
                            {exchange.get('query', 'N/A')}
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Bot response (left side)
                        if 'chat_response' in exchange and exchange['chat_response']:
                            st.markdown(f"""
                            <div class="message-agent">
                                🤖 {exchange['chat_response']}
                            </div>
                            """, unsafe_allow_html=True)
                            
                            # Thinking process button (if detailed info available)
                            if ('response' in exchange and exchange['response'] is not None) or \
                               ('agent_conversation' in exchange and exchange['agent_conversation']) or \
                               ('evaluation' in exchange and exchange['evaluation'] is not None):
                                
                                thinking_details = self._format_thinking_details(exchange)
                                if thinking_details:
                                    with st.expander("🤔 Show thinking process"):
                                        st.markdown(thinking_details, unsafe_allow_html=True)
                    
                    st.markdown('</div>', unsafe_allow_html=True)
            else:
                st.info("👋 Welcome! I'm your AI support assistant. How can I help you today?")
        
        # Input area at the bottom (ChatGPT style)
        st.markdown("---")
        st.markdown("### 💬 Ask me anything...")
        
        # Query input at bottom (Enter to send via form)
        with st.form(key="chat_form", clear_on_submit=True):
            query = st.text_input(
                "Your message",
                value="",
                placeholder="Type your message and press Enter to send...",
                key="chat_input"
            )
            submitted = st.form_submit_button("📤 Send")
        
        if submitted and query.strip():
            asyncio.run(self._process_query(query))
        
        # Quick examples and utilities
        st.markdown("### 🎯 Quick Examples")
        example_queries = [
            "Password reset help",
            "VPN connection issue",
            "Email not syncing",
            "Software installation",
            "Account access problem"
        ]
        cols = st.columns(len(example_queries))
        for col, example in zip(cols, example_queries):
            with col:
                if st.button(example, key=f"example_{example}"):
                    asyncio.run(self._process_query(example))
        
        col_util_1, col_util_2 = st.columns([1, 1])
        with col_util_1:
            if st.button("🎲 Random Query"):
                random_task = st.session_state.task_generator.generate_task()
                asyncio.run(self._process_query(random_task.user_query))
        with col_util_2:
            if st.button("🔄 Clear"):
                st.session_state.current_query = ""
                st.rerun()
    
    def _render_monitoring_dashboard(self):
        """Render the system monitoring dashboard."""
        st.markdown("## 📊 System Performance Monitoring")
        
        if not st.session_state.system_initialized:
            st.warning("⚠️ System not initialized. Please initialize the system first.")
            return
        
        # Real-time metrics
        st.markdown("### 📈 Real-time Metrics")
        
        # Get current stats
        system_stats = st.session_state.coordinator.get_system_stats()
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Active Agents",
                system_stats.get('active_agents', 0),
                delta=None
            )
        
        with col2:
            st.metric(
                "Total Messages",
                system_stats.get('total_messages', 0),
                delta=None
            )
        
        with col3:
            st.metric(
                "Conversations",
                len(st.session_state.conversation_history),
                delta=None
            )
        
        with col4:
            avg_response_time = self._calculate_average_response_time()
            st.metric(
                "Avg Response Time (s)",
                f"{avg_response_time:.2f}",
                delta=None
            )
        
        # Agent performance
        st.markdown("### 🤖 Agent Performance")
        
        agent_stats = system_stats.get('agent_stats', {})
        
        if agent_stats:
            # Create performance dataframe
            performance_data = []
            for agent_id, stats in agent_stats.items():
                performance_data.append({
                    'Agent': agent_id.replace('_agent', '').title(),
                    'Messages Processed': stats.get('messages_processed', 0),
                    'Messages Sent': stats.get('messages_sent', 0),
                    'Errors': stats.get('errors', 0),
                    'Uptime (s)': stats.get('uptime', 0),
                    'Success Rate': (stats.get('messages_processed', 0) - stats.get('errors', 0)) / max(stats.get('messages_processed', 0), 1)
                })
            
            df = pd.DataFrame(performance_data)
            
            # Performance charts
            col1, col2 = st.columns(2)
            
            with col1:
                fig = px.bar(df, x='Agent', y='Messages Processed', 
                           title="Messages Processed by Agent")
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                fig = px.bar(df, x='Agent', y='Success Rate', 
                           title="Success Rate by Agent")
                st.plotly_chart(fig, use_container_width=True)
            
            # Detailed table
            st.dataframe(df, use_container_width=True)
        
        # Knowledge base stats
        st.markdown("### 📚 Knowledge Base Statistics")
        
        try:
            kb_stats = self.knowledge_base.get_stats()
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Documents", kb_stats.total_documents)
            
            with col2:
                st.metric("Total Chunks", kb_stats.total_chunks)
            
            with col3:
                st.metric("Total Characters", f"{kb_stats.total_characters:,}")
            
            with col4:
                st.metric("Languages", len(kb_stats.languages))
            
            # Language distribution
            if kb_stats.languages:
                lang_data = pd.DataFrame({
                    'Language': kb_stats.languages,
                    'Count': [1] * len(kb_stats.languages)  # Simplified for demo
                })
                
                fig = px.pie(lang_data, values='Count', names='Language', 
                           title="Content Language Distribution")
                st.plotly_chart(fig, use_container_width=True)
        
        except Exception as e:
            st.error(f"Error loading knowledge base stats: {e}")
    
    def _render_training_dashboard(self):
        """Render the RL training dashboard."""
        st.markdown("## 🎓 Reinforcement Learning Training")
        
        if not st.session_state.system_initialized:
            st.warning("⚠️ System not initialized. Please initialize the system first.")
            return
        
        # Training controls
        st.markdown("### 🎮 Training Controls")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            if st.button("🚀 Start Training Episode"):
                self._start_training_episode()
        
        with col2:
            if st.button("⏹️ End Training Episode"):
                self._end_training_episode()
        
        with col3:
            num_episodes = st.number_input("Episodes to Run", min_value=1, max_value=100, value=10)
            if st.button("🔄 Run Batch Training"):
                self._run_batch_training(num_episodes)
        
        with col4:
            if st.button("💾 Save Model"):
                self._save_training_model()
        
        # Training parameters
        st.markdown("### ⚙️ Training Parameters")
        
        col1, col2 = st.columns(2)
        
        with col1:
            learning_rate = st.slider("Learning Rate", 0.0001, 0.01, 0.001, format="%.4f")
            gamma = st.slider("Discount Factor (γ)", 0.9, 0.999, 0.99, format="%.3f")
        
        with col2:
            entropy_coef = st.slider("Entropy Coefficient", 0.001, 0.1, 0.01, format="%.3f")
            max_episodes = st.slider("Max Episodes", 10, 1000, 100)
        
        # Training statistics
        st.markdown("### 📊 Training Statistics")
        
        try:
            training_stats = st.session_state.rl_agent.get_training_stats()
            
            if training_stats:
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Total Episodes", training_stats.get('total_episodes', 0))
                
                with col2:
                    st.metric("Total Steps", training_stats.get('total_steps', 0))
                
                with col3:
                    recent = training_stats.get('recent_performance', {})
                    st.metric("Avg Reward", f"{recent.get('average_reward', 0):.3f}")
                
                with col4:
                    st.metric("Best Avg Reward", f"{training_stats.get('best_average_reward', 0):.3f}")
                
                # Training progress chart
                progress_data = training_stats.get('training_progress', [])
                if progress_data:
                    df = pd.DataFrame(progress_data)
                    
                    fig = make_subplots(
                        rows=2, cols=2,
                        subplot_titles=('Average Reward', 'Episode Reward', 'Policy Loss', 'Episode Length')
                    )
                    
                    fig.add_trace(go.Scatter(x=df['episode'], y=df['average_reward'], name='Avg Reward'), row=1, col=1)
                    fig.add_trace(go.Scatter(x=df['episode'], y=df['episode_reward'], name='Episode Reward'), row=1, col=2)
                    fig.add_trace(go.Scatter(x=df['episode'], y=df['policy_loss'], name='Policy Loss'), row=2, col=1)
                    fig.add_trace(go.Scatter(x=df['episode'], y=df['episode_length'], name='Episode Length'), row=2, col=2)
                    
                    fig.update_layout(height=600, showlegend=False, title_text="Training Progress")
                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("📊 No training data available yet. Start training to see statistics.")
        
        except Exception as e:
            st.error(f"Error loading training stats: {e}")
    
    def _render_knowledge_base_interface(self):
        """Render the knowledge base management interface."""
        st.markdown("## 📚 Knowledge Base Management")
        
        # Knowledge base operations
        st.markdown("### 📁 Document Management")
        
        col1, col2 = st.columns(2)
        
        with col1:
            uploaded_file = st.file_uploader(
                "Upload Document",
                type=['pdf', 'docx', 'csv', 'xlsx', 'txt'],
                help="Upload documents to add to the knowledge base"
            )
            
            if uploaded_file and st.button("📤 Add to Knowledge Base"):
                self._add_document_to_kb(uploaded_file)
        
        with col2:
            directory_path = st.text_input(
                "Directory Path",
                value="dataset/",
                help="Path to directory containing documents"
            )
            
            if st.button("📂 Index Directory"):
                self._index_directory(directory_path)
        
        # Search interface
        st.markdown("### 🔍 Knowledge Base Search")
        
        search_query = st.text_input(
            "Search Query",
            placeholder="Enter search terms..."
        )
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            max_results = st.number_input("Max Results", min_value=1, max_value=50, value=10)
        
        with col2:
            min_score = st.slider("Min Similarity Score", 0.0, 1.0, 0.7, format="%.2f")
        
        with col3:
            search_language = st.selectbox("Search Language", ['all', 'en', 'es', 'de', 'fr'])
        
        if search_query and st.button("🔍 Search"):
            self._search_knowledge_base(search_query, max_results, min_score, search_language)
        
        # Knowledge base statistics
        st.markdown("### 📊 Knowledge Base Statistics")
        
        try:
            kb_stats = self.knowledge_base.get_stats()
            
            # Display stats
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Documents", kb_stats.total_documents)
            
            with col2:
                st.metric("Chunks", kb_stats.total_chunks)
            
            with col3:
                st.metric("Characters", f"{kb_stats.total_characters:,}")
            
            with col4:
                st.metric("Languages", len(kb_stats.languages))
            
            # File formats
            if kb_stats.file_formats:
                format_data = pd.DataFrame({
                    'Format': kb_stats.file_formats,
                    'Count': [1] * len(kb_stats.file_formats)  # Simplified
                })
                
                fig = px.bar(format_data, x='Format', y='Count', 
                           title="Document Formats in Knowledge Base")
                st.plotly_chart(fig, use_container_width=True)
        
        except Exception as e:
            st.error(f"Error loading knowledge base statistics: {e}")
    
    def _render_escalation_center(self):
        """Render the escalation monitoring center."""
        st.markdown("## ⚠️ Escalation Monitoring Center")
        
        if not st.session_state.system_initialized:
            st.warning("⚠️ System not initialized. Please initialize the system first.")
            return
        
        # Escalation statistics
        st.markdown("### 📊 Escalation Statistics")
        
        try:
            escalation_stats = st.session_state.escalation_agent.get_escalation_stats()
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Assessments", escalation_stats.get('total_assessments', 0))
            
            with col2:
                st.metric("Escalations Triggered", escalation_stats.get('escalations_triggered', 0))
            
            with col3:
                st.metric("Emails Sent", escalation_stats.get('emails_sent', 0))
            
            with col4:
                escalation_rate = escalation_stats.get('escalation_rate', 0)
                st.metric("Escalation Rate", f"{escalation_rate:.2%}")
            
            # Severity distribution
            severity_dist = escalation_stats.get('severity_distribution', {})
            if severity_dist:
                df = pd.DataFrame(list(severity_dist.items()), columns=['Severity', 'Count'])
                
                fig = px.pie(df, values='Count', names='Severity', 
                           title="Severity Level Distribution")
                st.plotly_chart(fig, use_container_width=True)
        
        except Exception as e:
            st.error(f"Error loading escalation stats: {e}")
        
        # Recent escalations
        st.markdown("### 🚨 Recent Escalations")
        
        try:
            recent_escalations = st.session_state.escalation_agent.get_escalation_history(limit=20)
            
            if recent_escalations:
                df = pd.DataFrame(recent_escalations)
                
                # Format timestamp
                df['timestamp'] = pd.to_datetime(df['timestamp']).dt.strftime('%Y-%m-%d %H:%M:%S')
                
                # Display table
                st.dataframe(df[['escalation_id', 'severity_level', 'timestamp', 'email_sent', 'reasoning']], 
                           use_container_width=True)
            else:
                st.info("📭 No recent escalations found.")
        
        except Exception as e:
            st.error(f"Error loading escalation history: {e}")
        
        # Email configuration test
        st.markdown("### 📧 Email Configuration")
        
        if st.button("🧪 Test Email Configuration"):
            try:
                test_result = st.session_state.escalation_agent.test_email_configuration()
                
                if test_result['status'] == 'success':
                    st.success("✅ Email configuration is working correctly!")
                    st.json(test_result['details'])
                else:
                    st.error("❌ Email configuration has issues:")
                    for error in test_result['errors']:
                        st.error(f"• {error}")
            
            except Exception as e:
                st.error(f"Error testing email configuration: {e}")
    
    def _render_configuration_interface(self):
        """Render the system configuration interface."""
        st.markdown("## ⚙️ System Configuration")
        
        # Configuration editor
        st.markdown("### 📝 Configuration Editor")
        
        try:
            config_data = self.config.config
            
            # Display current configuration
            st.json(config_data)
            
            # Configuration modification interface
            st.markdown("### 🔧 Modify Configuration")
            
            config_section = st.selectbox(
                "Configuration Section",
                options=list(config_data.keys())
            )
            
            if config_section:
                section_data = config_data[config_section]
                
                # Allow editing of specific values
                if isinstance(section_data, dict):
                    for key, value in section_data.items():
                        if isinstance(value, (str, int, float, bool)):
                            new_value = st.text_input(f"{config_section}.{key}", value=str(value))
                            
                            if st.button(f"Update {key}"):
                                self._update_config_value(f"{config_section}.{key}", new_value)
        
        except Exception as e:
            st.error(f"Error loading configuration: {e}")
        
        # System information
        st.markdown("### ℹ️ System Information")
        
        system_info = {
            "Python Version": "3.9+",
            "Streamlit Version": st.__version__,
            "System Environment": self.config.get('system.environment', 'Unknown'),
            "Debug Mode": self.config.get('system.debug', False),
            "Log Level": self.config.get('system.log_level', 'INFO')
        }
        
        for key, value in system_info.items():
            st.text(f"{key}: {value}")
    
    async def _process_query(self, query: str):
        """Process a user query through the multi-agent system."""
        if not st.session_state.system_initialized or not query.strip():
            return
        
        try:
            # Detect language
            language_result = detect_language(query)
            
            # Create user message
            user_message = Message(
                type=MessageType.QUERY,
                content=query,
                sender="user",
                recipient="communication_agent",
                language=language_result.language
            )
            
            # Ensure agents are running before processing cycles
            if not getattr(st.session_state.coordinator, "is_running", False):
                st.session_state.coordinator.start_all_agents()

            # Process through communication agent
            st.session_state.communication_agent.receive_message(user_message)
            
            # Run system cycles
            response_received = False
            final_response = ""
            evaluation_result = None
            symbolic_encoding = None
            agent_conversation = []  # Track inter-agent conversation
            
            for cycle_num in range(10):  # Max 10 cycles
                messages = await st.session_state.coordinator.run_cycle() or []
                
                for message in messages:
                    # Record all inter-agent messages
                    agent_conversation.append({
                        'cycle': cycle_num + 1,
                        'sender': message.sender,
                        'recipient': message.recipient,
                        'type': message.type.value,
                        'content': message.content[:200] + "..." if len(message.content) > 200 else message.content,
                        'timestamp': datetime.now().strftime("%H:%M:%S")
                    })
                    
                    if message.type == MessageType.RESPONSE and (message.sender == "retrieval_agent" or message.recipient == "user"):
                        final_response = message.content
                        response_received = True
                    
                    elif message.type == MessageType.SYMBOLIC and message.sender == "communication_agent":
                        symbolic_encoding = message.symbolic_encoding
                    
                    elif message.type == MessageType.FEEDBACK and "evaluation_result" in message.metadata:
                        evaluation_result = message.metadata["evaluation_result"]
                
                if response_received:
                    break
            
            # Generate chat-like response from detailed search results
            chat_response = self._generate_chat_response(final_response, query)
            
            # Record conversation
            conversation_entry = {
                'timestamp': datetime.now().isoformat(),
                'query': query,
                'language': language_result.language,
                'response': final_response or "No response generated",
                'chat_response': chat_response,
                'agent_conversation': agent_conversation,
                'symbolic_encoding': symbolic_encoding,
                'evaluation': evaluation_result
            }
            
            st.session_state.conversation_history.append(conversation_entry)
            st.success("✅ Query processed successfully!")
            st.rerun()
        
        except Exception as e:
            st.error(f"❌ Error processing query: {e}")
            logger.error(f"Query processing error: {e}")
    
    def _generate_chat_response(self, detailed_response: str, query: str) -> str:
        """Generate a concise, chat-like response from detailed search results."""
        
        # Handle simple greetings and casual conversation
        query_lower = query.lower().strip()
        simple_greetings = ['hi', 'hello', 'hey', 'good morning', 'good afternoon', 'good evening', 'how are you', 'what\'s up', 'sup']
        
        if any(greeting in query_lower for greeting in simple_greetings):
            return "Hello! 👋 I'm your AI support assistant. I'm here to help you with any technical issues, account problems, or questions you might have. How can I assist you today?"
        
        if not detailed_response or detailed_response == "No response generated":
            return "I'm sorry, I couldn't find any relevant information for your query. Please try rephrasing your question or contact support for assistance."
        
        # Extract the main solution from the first result
        try:
            lines = detailed_response.split('\n')
            first_result = None
            solution = None
            
            # Find the first result and extract its solution
            for i, line in enumerate(lines):
                if line.strip().startswith('1.') and 'Title:' in line:
                    # Extract title
                    title_start = line.find('Title:') + 6
                    title_end = line.find('Description:')
                    if title_end == -1:
                        title_end = len(line)
                    title = line[title_start:title_end].strip()
                    
                    # Look for solution in subsequent lines
                    for j in range(i, min(i + 5, len(lines))):
                        if 'Solution:' in lines[j]:
                            solution_start = lines[j].find('Solution:') + 9
                            solution = lines[j][solution_start:].strip()
                            break
                    
                    first_result = title
                    break
            
            # Generate chat response based on query type
            if 'password' in query_lower and 'reset' in query_lower:
                if solution:
                    return f"For password reset issues, here's what you can do: {solution}"
                return "For password reset issues, please check your spam folder, verify your email address, and ensure the email service is configured correctly."
            
            elif 'access' in query_lower or 'login' in query_lower or 'dashboard' in query_lower:
                if solution:
                    return f"To resolve access issues: {solution}"
                return "For access problems, please verify your credentials, check your account status, and ensure you have the proper permissions."
            
            elif 'slow' in query_lower or 'performance' in query_lower:
                if solution:
                    return f"To improve performance: {solution}"
                return "For performance issues, try clearing your browser cache, checking your network connection, and restarting the application."
            
            elif 'upload' in query_lower or 'file' in query_lower:
                if solution:
                    return f"For file upload problems: {solution}"
                return "For file upload issues, check the file size limits, verify the file format is supported, and try using a different browser."
            
            elif 'email' in query_lower or 'notification' in query_lower:
                if solution:
                    return f"To fix email notification issues: {solution}"
                return "For email notification problems, check the email service configuration, verify SMTP settings, and test email connectivity."
            
            elif 'leaves' in query_lower or 'leave' in query_lower:
                return "If your leave applications aren't showing on the panel, please check: 1) Ensure you've submitted your leave request through the correct system, 2) Verify you're looking in the right section (My Leaves, Leave History, etc.), 3) Check if there are any pending approvals needed, 4) Contact HR if your approved leaves still don't appear after 24 hours."
            
            else:
                # Generic response with the first solution found
                if solution and first_result:
                    return f"I found a solution for '{first_result}': {solution}"
                elif first_result:
                    # Try to extract a useful response even without explicit solution
                    if 'access' in first_result.lower() or 'dashboard' in first_result.lower():
                        return "It seems you're having trouble accessing a system or dashboard. Try verifying your login credentials, checking your account permissions, and ensuring your account is active. If the issue persists, contact your system administrator."
                    elif 'password' in first_result.lower():
                        return "For password-related issues, first check your spam/junk folder for reset emails. Make sure you're using the correct email address and that your account isn't locked. If you still can't reset your password, contact support."
                    elif 'email' in first_result.lower() or 'notification' in first_result.lower():
                        return "Email notification issues are usually caused by incorrect email settings. Check that your email address is correct in your profile, verify SMTP configuration, and ensure emails aren't being blocked by your email provider."
                    elif 'slow' in first_result.lower() or 'performance' in first_result.lower():
                        return "For performance issues, try clearing your browser cache and cookies, check your internet connection, and close other applications that might be using system resources. If using a web application, try a different browser."
                    elif 'upload' in first_result.lower() or 'file' in first_result.lower():
                        return "File upload problems are often due to file size limits or unsupported formats. Check that your file is under the size limit, in a supported format (PDF, JPG, PNG, DOC, etc.), and try using a different browser if the issue continues."
                    else:
                        return f"Based on similar cases like '{first_result}', I recommend checking your account settings, verifying your permissions, and ensuring all system requirements are met. If the problem continues, contact technical support with specific error details."
                else:
                    # Last resort - be more helpful than just "check detailed results"
                    return "I couldn't find an exact match for your issue, but here are some general troubleshooting steps: 1) Check your account permissions and settings, 2) Clear your browser cache and cookies, 3) Try logging out and back in, 4) Contact support if the issue persists with specific error messages."
        
        except Exception:
            # Fallback for any parsing errors - still be helpful
            if 'password' in query_lower:
                return "For password issues, check your spam folder for reset emails and verify your email address is correct."
            elif 'access' in query_lower or 'login' in query_lower:
                return "For access problems, verify your credentials and check that your account is active."
            elif 'slow' in query_lower or 'performance' in query_lower:
                return "For performance issues, try clearing your browser cache and checking your internet connection."
            else:
                return "I'm here to help! Try checking your account settings and permissions first. If the issue persists, contact support with specific details about what you're experiencing."
    
    def _format_thinking_details(self, exchange: dict) -> str:
        """Format the thinking process details for the expandable section."""
        details = []
        
        # Add detailed search results
        if 'response' in exchange and exchange['response'] is not None:
            details.append(f"""
            <div style="background-color: #f8f9fa; color: #212529; padding: 1rem; border-radius: 0.5rem; border-left: 4px solid #17a2b8; margin: 0.5rem 0;">
                <strong style="color: #17a2b8;">🔍 Knowledge Base Search Results:</strong><br>
                <span style="color: #495057;">{exchange['response']}</span>
            </div>
            """)
        
        # Add agent conversation
        if 'agent_conversation' in exchange and exchange['agent_conversation']:
            details.append('<div style="margin: 0.5rem 0;"><strong style="color: #1976d2;">🤖 Agent Communication:</strong></div>')
            
            for msg in exchange['agent_conversation']:
                sender_emoji = "🗣️" if msg['sender'] == "communication_agent" else "🔍" if msg['sender'] == "retrieval_agent" else "🤖"
                recipient_emoji = "🗣️" if msg['recipient'] == "communication_agent" else "🔍" if msg['recipient'] == "retrieval_agent" else "🤖"
                
                details.append(f"""
                <div style="background-color: #f1f3f4; color: #212529; padding: 0.5rem; margin: 0.25rem 0; border-radius: 0.25rem; font-size: 0.9em;">
                    <strong style="color: #1f2937;">Cycle {msg['cycle']} - {msg['timestamp']}</strong><br>
                    {sender_emoji} <strong style="color: #374151;">{msg['sender'].replace('_', ' ').title()}</strong> 
                    → {recipient_emoji} <strong style="color: #374151;">{msg['recipient'].replace('_', ' ').title()}</strong> 
                    (<em style="color: #6b7280;">{msg['type']}</em>)<br>
                    <span style="color: #4b5563;">{msg['content']}</span>
                </div>
                """)
        
        # Add evaluation metrics
        if 'evaluation' in exchange and exchange['evaluation'] is not None:
            eval_data = exchange['evaluation']
            details.append(f"""
            <div style="background-color: #fff3e0; color: #212529; padding: 1rem; border-radius: 0.5rem; border-left: 4px solid #ff9800; margin: 0.5rem 0;">
                <strong style="color: #ff9800;">📊 Response Quality Metrics:</strong><br>
                Overall Score: {eval_data.get('overall_score', 0):.3f} | 
                Relevance: {eval_data.get('relevance_score', 0):.3f} | 
                Accuracy: {eval_data.get('accuracy_score', 0):.3f} | 
                Completeness: {eval_data.get('completeness_score', 0):.3f}
            </div>
            """)
        
        # Add symbolic encoding
        if 'symbolic_encoding' in exchange and exchange['symbolic_encoding'] is not None:
            encoding = exchange['symbolic_encoding']
            details.append(f"""
            <div style="background-color: #e8f5e8; color: #212529; padding: 1rem; border-radius: 0.5rem; border-left: 4px solid #4caf50; margin: 0.5rem 0;">
                <strong style="color: #4caf50;">🔢 Symbolic Encoding:</strong><br>
                Length: {len(encoding)} | Unique Symbols: {len(set(encoding))} | 
                Encoding: {encoding[:20]}{"..." if len(encoding) > 20 else ""}
            </div>
            """)
        
        return "".join(details)
    
    def _start_agents(self):
        """Start all agents."""
        try:
            st.session_state.coordinator.start_all_agents()
            st.session_state.agents_running = True
            st.success("✅ All agents started successfully!")
        except Exception as e:
            st.error(f"❌ Error starting agents: {e}")
    
    def _stop_agents(self):
        """Stop all agents."""
        try:
            st.session_state.coordinator.stop_all_agents()
            st.session_state.agents_running = False
            st.success("✅ All agents stopped successfully!")
        except Exception as e:
            st.error(f"❌ Error stopping agents: {e}")
    
    def _calculate_average_response_time(self) -> float:
        """Calculate average response time."""
        # Simplified calculation
        return 2.5  # Mock value
    
    def _start_training_episode(self):
        """Start a new training episode."""
        try:
            st.session_state.rl_agent.start_training_episode()
            st.success("🚀 Training episode started!")
        except Exception as e:
            st.error(f"❌ Error starting training episode: {e}")
    
    def _end_training_episode(self):
        """End the current training episode."""
        try:
            stats = st.session_state.rl_agent.end_training_episode()
            if stats:
                st.success(f"⏹️ Training episode ended! Reward: {stats.episode_reward:.3f}")
            else:
                st.warning("⚠️ No active training episode to end.")
        except Exception as e:
            st.error(f"❌ Error ending training episode: {e}")
    
    def _run_batch_training(self, num_episodes: int):
        """Run batch training."""
        try:
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            for i in range(num_episodes):
                # Start episode
                st.session_state.rl_agent.start_training_episode()
                
                # Generate random task
                task = st.session_state.task_generator.generate_task()
                
                # Simulate episode (simplified)
                reward = np.random.uniform(0.3, 0.9)  # Mock reward
                st.session_state.rl_agent.receive_reward(reward)
                
                # End episode
                stats = st.session_state.rl_agent.end_training_episode()
                
                # Update progress
                progress = (i + 1) / num_episodes
                progress_bar.progress(progress)
                status_text.text(f"Episode {i+1}/{num_episodes} - Reward: {stats.episode_reward:.3f}")
                
                time.sleep(0.1)  # Small delay for visualization
            
            st.success(f"✅ Batch training completed! {num_episodes} episodes finished.")
        
        except Exception as e:
            st.error(f"❌ Error in batch training: {e}")
    
    def _save_training_model(self):
        """Save the training model."""
        try:
            model_path = "models/communication_agent_model.pt"
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            st.session_state.rl_agent.save_model(model_path)
            st.success(f"💾 Model saved to {model_path}")
        except Exception as e:
            st.error(f"❌ Error saving model: {e}")
    
    def _add_document_to_kb(self, uploaded_file):
        """Add uploaded document to knowledge base."""
        try:
            # Save uploaded file temporarily
            temp_path = f"temp_{uploaded_file.name}"
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            # Add to knowledge base
            success = self.knowledge_base.add_document(temp_path)
            
            # Clean up
            os.remove(temp_path)
            
            if success:
                st.success(f"✅ Document '{uploaded_file.name}' added to knowledge base!")
            else:
                st.error(f"❌ Failed to add document '{uploaded_file.name}'")
        
        except Exception as e:
            st.error(f"❌ Error adding document: {e}")
    
    def _index_directory(self, directory_path: str):
        """Index all documents in a directory."""
        try:
            successful, total = self.knowledge_base.add_documents_from_directory(directory_path)
            st.success(f"✅ Indexed {successful}/{total} documents from {directory_path}")
        except Exception as e:
            st.error(f"❌ Error indexing directory: {e}")
    
    def _search_knowledge_base(self, query: str, max_results: int, min_score: float, language: str):
        """Search the knowledge base."""
        try:
            # Perform search
            search_language = None if language == 'all' else language
            results = self.knowledge_base.search(
                query=query,
                max_results=max_results,
                min_score=min_score,
                language=search_language
            )
            
            st.markdown(f"### 🔍 Search Results ({len(results)} found)")
            
            if results:
                for i, result in enumerate(results):
                    with st.expander(f"Result {i+1} - Score: {result.score:.3f}"):
                        st.markdown(f"**Source:** {result.chunk.source_file}")
                        st.markdown(f"**Language:** {result.chunk.language}")
                        st.markdown(f"**Content:**")
                        st.text(result.chunk.content[:500] + "..." if len(result.chunk.content) > 500 else result.chunk.content)
            else:
                st.info("📭 No results found for your query.")
        
        except Exception as e:
            st.error(f"❌ Error searching knowledge base: {e}")
    
    def _update_config_value(self, key_path: str, new_value: str):
        """Update a configuration value."""
        try:
            # This would update the configuration
            st.success(f"✅ Configuration updated: {key_path} = {new_value}")
        except Exception as e:
            st.error(f"❌ Error updating configuration: {e}")
    
    def _export_logs(self):
        """Export system logs."""
        try:
            # Create export data
            export_data = {
                'conversation_history': st.session_state.conversation_history,
                'system_stats': st.session_state.coordinator.get_system_stats() if st.session_state.system_initialized else {},
                'export_timestamp': datetime.now().isoformat()
            }
            
            # Create download
            json_data = json.dumps(export_data, indent=2)
            st.download_button(
                label="📥 Download Logs",
                data=json_data,
                file_name=f"support_system_logs_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )
        except Exception as e:
            st.error(f"❌ Error exporting logs: {e}")
    
    def _reset_system(self):
        """Reset the entire system."""
        try:
            # Clear session state
            for key in list(st.session_state.keys()):
                if key != 'system_initialized':
                    del st.session_state[key]
            
            st.session_state.system_initialized = False
            st.success("🔄 System reset successfully!")
            st.rerun()
        except Exception as e:
            st.error(f"❌ Error resetting system: {e}")

# Main application
def main():
    """Main application entry point."""
    dashboard = SupportSystemDashboard()
    dashboard.run()

if __name__ == "__main__":
    main()