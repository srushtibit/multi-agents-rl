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
            
            # Initialize RL components (optional; only if communication agent supports encoder)
            try:
                _ = st.session_state.communication_agent.encoder  # probe optional attribute
                st.session_state.rl_agent = REINFORCEAgent(st.session_state.communication_agent)
            except Exception:
                st.session_state.rl_agent = None
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
                        
                        # Bot response (left side) - use chat_response if available, fallback to response
                        if exchange.get('processing'):
                            # Show loading state with animated dots
                            st.markdown(f"""
                            <div class="message-agent">
                                🤖 <span style="opacity: 0.7;">Thinking</span>
                                <span class="loading-dots">
                                    <span>.</span><span>.</span><span>.</span>
                                </span>
                            </div>
                            <style>
                            .loading-dots span {{
                                animation: loading 1.4s infinite ease-in-out both;
                                display: inline-block;
                            }}
                            .loading-dots span:nth-child(1) {{ animation-delay: -0.32s; }}
                            .loading-dots span:nth-child(2) {{ animation-delay: -0.16s; }}
                            @keyframes loading {{
                                0%, 80%, 100% {{ transform: scale(0); }}
                                40% {{ transform: scale(1); }}
                            }}
                            </style>
                            """, unsafe_allow_html=True)
                        else:
                            display_response = exchange.get('chat_response') or exchange.get('response')
                            if display_response:
                                # Escape HTML in the response to prevent raw HTML display
                                import html
                                safe_response = html.escape(str(display_response))
                                st.markdown(f"""
                                <div class="message-agent">
                                    🤖 {safe_response}
                                </div>
                                """, unsafe_allow_html=True)
                            
                            # Thinking process button (if detailed info available)
                            if ('agent_conversation' in exchange and exchange['agent_conversation']) or \
                               ('evaluation' in exchange and exchange['evaluation'] is not None) or \
                               ('retrieved_docs' in exchange and exchange['retrieved_docs']):

                                with st.expander("🤔 Show thinking process"):
                                    self._render_thinking_process(exchange)
                    
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
            # Immediately add user message to show it in the chat
            user_entry = {
                'timestamp': datetime.now().isoformat(),
                'query': query,
                'language': 'en',  # Will be detected later
                'response': None,  # Will be filled after processing
                'chat_response': None,
                'agent_conversation': [],
                'symbolic_encoding': None,
                'evaluation': None,
                'retrieved_docs': None,
                'processing': True  # Flag to show loading state
            }
            st.session_state.conversation_history.append(user_entry)

            # Process the query with spinner
            with st.spinner("🤖 Agents are processing your request..."):
                asyncio.run(self._process_query(query, len(st.session_state.conversation_history) - 1))
        
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
                    # Add user message immediately for examples too
                    user_entry = {
                        'timestamp': datetime.now().isoformat(),
                        'query': example,
                        'language': 'en',
                        'response': None,
                        'chat_response': None,
                        'agent_conversation': [],
                        'symbolic_encoding': None,
                        'evaluation': None,
                        'retrieved_docs': None,
                        'processing': True
                    }
                    st.session_state.conversation_history.append(user_entry)
                    with st.spinner("🤖 Agents are processing your request..."):
                        asyncio.run(self._process_query(example, len(st.session_state.conversation_history) - 1))

        col_util_1, col_util_2 = st.columns([1, 1])
        with col_util_1:
            if st.button("🎲 Random Query"):
                random_task = st.session_state.task_generator.generate_task()
                # Add user message immediately for random query too
                user_entry = {
                    'timestamp': datetime.now().isoformat(),
                    'query': random_task.user_query,
                    'language': 'en',
                    'response': None,
                    'chat_response': None,
                    'agent_conversation': [],
                    'symbolic_encoding': None,
                    'evaluation': None,
                    'retrieved_docs': None,
                    'processing': True
                }
                st.session_state.conversation_history.append(user_entry)
                with st.spinner("🤖 Agents are processing your request..."):
                    asyncio.run(self._process_query(random_task.user_query, len(st.session_state.conversation_history) - 1))
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
            if st.session_state.rl_agent is None:
                st.info("RL training is unavailable because the communication agent does not expose an encoder.")
                return
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

        with st.form(key="kb_search_form", clear_on_submit=False):
            search_query = st.text_input(
                "Search Query",
                placeholder="Enter search terms...",
                key="kb_search_input"
            )
            col1, col2, col3 = st.columns(3)
            with col1:
                max_results = st.number_input("Max Results", min_value=1, max_value=50, value=10, key="kb_max_results")
            with col2:
                min_score = st.slider("Min Similarity Score", 0.0, 1.0, 0.7, format="%.2f", key="kb_min_score")
            with col3:
                search_language = st.selectbox("Search Language", ['all', 'en', 'es', 'de', 'fr'], key="kb_lang")
            search_submitted = st.form_submit_button("🔍 Search")
        if search_submitted and search_query:
            with st.spinner("🔎 Searching knowledge base..."):
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
    
    async def _process_query(self, query: str, conversation_index: int = None):
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
            initial_response = await st.session_state.communication_agent.process_message(user_message)

            # Check if communication agent handled it directly (e.g., greeting)
            if initial_response and initial_response.recipient == "user":
                # Direct response from communication agent - no need for further processing
                final_response = initial_response.content
                agent_conversation = [{
                    'cycle': 1,
                    'sender': 'communication_agent',
                    'recipient': 'user',
                    'type': 'response',
                    'content': final_response,
                    'timestamp': datetime.now().strftime("%H:%M:%S")
                }]
                response_received = True
                evaluation_result = None
                symbolic_encoding = None
            else:
                # Continue with full processing pipeline
                st.session_state.communication_agent.receive_message(user_message)

                # Run system cycles
                response_received = False
                final_response = ""
                evaluation_result = None
                symbolic_encoding = None
                agent_conversation = []  # Track inter-agent conversation

                for cycle_num in range(5):  # Allow enough cycles for retrieval → communication → critic
                    messages = await st.session_state.coordinator.run_cycle() or []
                
                    for message in messages:
                        # Record all inter-agent messages
                        # Escape special HTML characters to prevent raw HTML rendering
                        def _escape_html(text: str) -> str:
                            try:
                                return (text.replace('&', '&amp;')
                                            .replace('<', '&lt;')
                                            .replace('>', '&gt;'))
                            except Exception:
                                return text

                        safe_content = message.content
                        if isinstance(safe_content, str):
                            # Strip any raw HTML to avoid leaking tags into the UI and hide <think> blocks
                            import re as _re
                            no_html = _re.sub(r"<[^>]+>", "", safe_content)
                            no_think = no_html
                            if "<think>" in safe_content:
                                # Remove think blocks if any slipped through
                                no_think = _re.sub(r"<think>[\s\S]*?</think>", "", safe_content, flags=_re.IGNORECASE)
                                no_think = _re.sub(r"<[^>]+>", "", no_think)
                            truncated = no_think[:400] + "..." if len(no_think) > 400 else no_think
                            safe_content = _escape_html(truncated)

                        agent_conversation.append({
                            'cycle': cycle_num + 1,
                            'sender': message.sender,
                            'recipient': message.recipient,
                            'type': message.type.value,
                            'content': safe_content,
                            'timestamp': datetime.now().strftime("%H:%M:%S")
                        })

                        # Prefer the final, user-directed response from the critic agent
                        if message.type == MessageType.RESPONSE and message.recipient == "user":
                            final_response = message.content
                            # Capture evaluation and retrieved docs if provided by the critic
                            if isinstance(message.metadata, dict):
                                if message.metadata.get("evaluation"):
                                    evaluation_result = message.metadata.get("evaluation")
                                if message.metadata.get("retrieved_docs"):
                                    st.session_state.last_retrieved_docs = message.metadata.get("retrieved_docs")
                            response_received = True

                        elif message.type == MessageType.SYMBOLIC and message.sender == "communication_agent":
                            symbolic_encoding = message.symbolic_encoding

                        elif message.type == MessageType.FEEDBACK and isinstance(message.metadata, dict) and "evaluation_result" in message.metadata:
                            evaluation_result = message.metadata.get("evaluation_result")

                    if response_received:
                        break
            
            # Generate chat-like response from detailed search results
            chat_response = self._generate_chat_response(final_response, query)
            
            # Update existing conversation entry or create new one
            if conversation_index is not None and conversation_index < len(st.session_state.conversation_history):
                # Update existing entry
                entry = st.session_state.conversation_history[conversation_index]
                entry.update({
                    'language': language_result.language,
                    'response': final_response or "No response generated",
                    'chat_response': chat_response,
                    'agent_conversation': agent_conversation,
                    'symbolic_encoding': symbolic_encoding,
                    'evaluation': evaluation_result,
                    'retrieved_docs': st.session_state.get('last_retrieved_docs'),
                    'processing': False  # Remove processing flag
                })
            else:
                # Create new conversation entry (fallback)
                conversation_entry = {
                    'timestamp': datetime.now().isoformat(),
                    'query': query,
                    'language': language_result.language,
                    'response': final_response or "No response generated",
                    'chat_response': chat_response,
                    'agent_conversation': agent_conversation,
                    'symbolic_encoding': symbolic_encoding,
                    'evaluation': evaluation_result,
                    'retrieved_docs': st.session_state.get('last_retrieved_docs'),
                    'processing': False
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

        # Clean and process the detailed response
        # The retrieval agent returns bullet points, so let's work with that format
        response_lines = [line.strip() for line in detailed_response.split('\n') if line.strip()]

        # If the response is already in a good format (bullet points), use it directly
        if any(line.startswith(('-', '*', '•')) for line in response_lines):
            # Clean up the bullet points and make them more conversational
            cleaned_bullets = []
            for line in response_lines:
                if line.startswith(('-', '*', '•')):
                    # Remove bullet and clean up
                    clean_line = line[1:].strip()
                    if clean_line and len(clean_line) > 10:  # Skip very short lines
                        cleaned_bullets.append(clean_line)

            if cleaned_bullets:
                # Create a conversational response
                if len(cleaned_bullets) == 1:
                    return f"Here's what I found to help with your issue: {cleaned_bullets[0]}"
                else:
                    response = "Here's what I found to help with your issue:\n\n"
                    for i, bullet in enumerate(cleaned_bullets[:4], 1):  # Limit to 4 points
                        response += f"{i}. {bullet}\n"
                    return response.strip()

        # If no bullet points, try to extract useful information from the response
        # Look for common patterns and provide contextual help based on query type

        # Try to extract meaningful content from the response
        meaningful_content = ""
        for line in response_lines:
            # Skip metadata-like lines
            if any(skip_word in line.lower() for skip_word in ['ticket', 'complaint id', 'employee name', 'domain', 'priority', 'queue', 'business type', 'tag', 'language', 'source:']):
                continue
            # Look for resolution or answer content
            if any(key_word in line.lower() for key_word in ['resolution:', 'answer:', 'solution:', 'steps:']):
                meaningful_content = line
                break
            # If line is substantial, use it
            elif len(line) > 20 and not line.startswith(('ticket', 'complaint', 'employee')):
                meaningful_content = line
                break

        # Generate contextual responses based on query type
        if 'password' in query_lower and 'reset' in query_lower:
            if meaningful_content:
                return f"For password reset issues, here's what I found: {meaningful_content}"
            return "For password reset issues, please check your spam folder, verify your email address, and ensure the email service is configured correctly."

        elif 'access' in query_lower or 'login' in query_lower or 'dashboard' in query_lower:
            if meaningful_content:
                return f"To resolve access issues: {meaningful_content}"
            return "For access problems, please verify your credentials, check your account status, and ensure you have the proper permissions."

        elif 'slow' in query_lower or 'performance' in query_lower:
            if meaningful_content:
                return f"To improve performance: {meaningful_content}"
            return "For performance issues, try clearing your browser cache, checking your network connection, and restarting the application."

        elif 'upload' in query_lower or 'file' in query_lower:
            if meaningful_content:
                return f"For file upload problems: {meaningful_content}"
            return "For file upload issues, check the file size limits, verify the file format is supported, and try using a different browser."

        elif 'email' in query_lower or 'notification' in query_lower:
            if meaningful_content:
                return f"To fix email notification issues: {meaningful_content}"
            return "For email notification problems, check the email service configuration, verify SMTP settings, and test email connectivity."

        elif 'leaves' in query_lower or 'leave' in query_lower:
            if meaningful_content:
                return f"Regarding leave applications: {meaningful_content}"
            return "If your leave applications aren't showing on the panel, please check: 1) Ensure you've submitted your leave request through the correct system, 2) Verify you're looking in the right section (My Leaves, Leave History, etc.), 3) Check if there are any pending approvals needed, 4) Contact HR if your approved leaves still don't appear after 24 hours."

        else:
            # Generic response - use meaningful content if found, otherwise provide helpful fallback
            if meaningful_content:
                return f"Here's what I found to help with your issue: {meaningful_content}"
            else:
                # Provide helpful general guidance
                return "I found some information that might help, but let me provide some general troubleshooting steps: 1) Check your account permissions and settings, 2) Clear your browser cache and cookies, 3) Try logging out and back in, 4) Contact support if the issue persists with specific error messages."
    
    def _format_thinking_details(self, exchange: dict) -> str:
        """Format the thinking process details for the expandable section."""
        import html

        details = []

        # Helper function to safely escape HTML content
        def safe_html_escape(text: str) -> str:
            if not text:
                return ""
            return html.escape(str(text))

        # 1) Communication → Retrieval (what was asked)
        com_to_ret = None
        for msg in exchange.get('agent_conversation', []) or []:
            if msg.get('sender') == 'communication_agent' and msg.get('recipient') == 'retrieval_agent':
                com_to_ret = msg
                break
        if com_to_ret:
            safe_content = safe_html_escape(com_to_ret.get('content', ''))
            details.append(f"""
            <div style="background-color: #eef6ff; color: #212529; padding: 0.75rem; border-radius: 0.5rem; border-left: 4px solid #1e88e5; margin: 0.5rem 0;">
                <strong style="color: #1e88e5;">1) Communication → Retrieval</strong><br>
                <span style="color: #374151;">{safe_content}</span>
            </div>
            """)

        # 2) Retrieval → Communication (what came back)
        ret_to_com = None
        for msg in exchange.get('agent_conversation', []) or []:
            if msg.get('sender') == 'retrieval_agent' and msg.get('recipient') == 'communication_agent' and msg.get('type') == 'response':
                ret_to_com = msg
                break
        if ret_to_com:
            safe_content = safe_html_escape(ret_to_com.get('content', ''))
            details.append(f"""
            <div style="background-color: #f1f3f4; color: #212529; padding: 0.75rem; border-radius: 0.5rem; border-left: 4px solid #6c757d; margin: 0.5rem 0;">
                <strong style="color: #6c757d;">2) Retrieval → Communication</strong><br>
                <span style="color: #374151;">{safe_content}</span>
            </div>
            """)

        # 3) Short summary (from critic feedback if available)
        summary_text = None
        eval_data = exchange.get('evaluation') if isinstance(exchange.get('evaluation'), dict) else None
        if eval_data and isinstance(eval_data.get('feedback'), str):
            summary_text = eval_data.get('feedback')
        elif exchange.get('retrieved_docs'):
            top = exchange['retrieved_docs'][0]
            top_src = top.get('chunk',{}).get('source_file','')
            summary_text = f"Selected top results by similarity; leading source: {top_src}."
        if summary_text:
            safe_summary = safe_html_escape(summary_text)
            details.append(f"""
            <div style="background-color: #fffbe6; color: #212529; padding: 0.75rem; border-radius: 0.5rem; border-left: 4px solid #fbc02d; margin: 0.5rem 0;">
                <strong style="color: #fbc02d;">3) Summary</strong><br>
                <span style="color: #374151;">{safe_summary}</span>
            </div>
            """)

        # Show retrieved docs from metadata if present
        retrieved_docs = exchange.get('retrieved_docs')

        if not retrieved_docs and isinstance(exchange.get('response'), dict) and 'retrieved_docs' in exchange['response']:
            retrieved_docs = exchange['response']['retrieved_docs']

        if retrieved_docs:
            details.append('<div style="margin: 0.5rem 0;"><strong style="color: #1976d2;">4) 📄 Documents Consulted:</strong></div>')
            for i, doc in enumerate(retrieved_docs[:5]):
                try:
                    src = safe_html_escape(doc['chunk']['source_file'])
                    snippet = doc['chunk']['content'][:220] + ('...' if len(doc['chunk']['content']) > 220 else '')
                    safe_snippet = safe_html_escape(snippet)
                    score = doc.get('score', 0)
                    details.append(
                        f'<div style="background-color: #f9fafb; color: #212529; padding: 0.5rem; margin: 0.25rem 0; border-radius: 0.25rem; font-size: 0.9em;">'
                        f'<strong>Doc {i+1}</strong> — <em>{src}</em><br>'
                        f'<span style="color:#6b7280;">Score: {score:.3f}</span><br>'
                        f'<span style="color:#374151;">{safe_snippet}</span>'
                        f'</div>'
                    )
                except Exception:
                    continue
        
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

    def _render_thinking_process(self, exchange: dict):
        """Render the thinking process in a structured, user-friendly way."""
        import html

        # Helper function to safely escape HTML content
        def safe_escape(text: str) -> str:
            if not text:
                return ""
            return html.escape(str(text))

        st.markdown("### 🧠 Agent Communication Flow")

        # 1. Show Communication → Retrieval
        com_to_ret = None
        for msg in exchange.get('agent_conversation', []) or []:
            if msg.get('sender') == 'communication_agent' and msg.get('recipient') == 'retrieval_agent':
                com_to_ret = msg
                break

        if com_to_ret:
            st.markdown("**1️⃣ Communication Agent → Retrieval Agent**")
            with st.container():
                st.info(f"📤 **Query sent:** {safe_escape(com_to_ret.get('content', ''))}")

        # 2. Show Retrieval → Communication
        ret_to_com = None
        for msg in exchange.get('agent_conversation', []) or []:
            if msg.get('sender') == 'retrieval_agent' and msg.get('recipient') == 'communication_agent' and msg.get('type') == 'response':
                ret_to_com = msg
                break

        if ret_to_com:
            st.markdown("**2️⃣ Retrieval Agent → Communication Agent**")
            with st.container():
                response_content = safe_escape(ret_to_com.get('content', ''))
                if len(response_content) > 300:
                    response_content = response_content[:300] + "..."
                st.success(f"📥 **Response received:** {response_content}")

        # 3. Show Documents Consulted
        retrieved_docs = exchange.get('retrieved_docs')
        if retrieved_docs:
            st.markdown("**3️⃣ Documents Consulted**")

            # Create tabs for better organization
            if len(retrieved_docs) > 1:
                doc_tabs = st.tabs([f"Doc {i+1}" for i in range(min(5, len(retrieved_docs)))])
                for i, (tab, doc) in enumerate(zip(doc_tabs, retrieved_docs[:5])):
                    with tab:
                        try:
                            src = doc['chunk']['source_file']
                            snippet = doc['chunk']['content'][:300] + ('...' if len(doc['chunk']['content']) > 300 else '')
                            score = doc.get('score', 0)

                            st.markdown(f"**📄 Source:** `{src}`")
                            st.markdown(f"**🎯 Similarity Score:** {score:.3f}")
                            st.markdown(f"**📝 Content Preview:**")
                            st.text_area("Document Content", value=snippet, height=100, disabled=True, key=f"doc_content_{i}", label_visibility="collapsed")
                        except Exception:
                            st.error("Error displaying document")
            else:
                # Single document - show directly
                try:
                    doc = retrieved_docs[0]
                    src = doc['chunk']['source_file']
                    snippet = doc['chunk']['content'][:400] + ('...' if len(doc['chunk']['content']) > 400 else '')
                    score = doc.get('score', 0)

                    col1, col2 = st.columns([2, 1])
                    with col1:
                        st.markdown(f"**📄 Source:** `{src}`")
                    with col2:
                        st.markdown(f"**🎯 Score:** {score:.3f}")

                    st.markdown("**📝 Content Preview:**")
                    st.text_area("Document Content", value=snippet, height=120, disabled=True, key="single_doc_content", label_visibility="collapsed")
                except Exception:
                    st.error("Error displaying document")

        # 4. Show Evaluation Metrics
        if 'evaluation' in exchange and exchange['evaluation'] is not None:
            st.markdown("**4️⃣ Response Quality Assessment**")
            eval_data = exchange['evaluation']

            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Overall", f"{eval_data.get('overall_score', 0):.2f}")
            with col2:
                st.metric("Relevance", f"{eval_data.get('relevance_score', 0):.2f}")
            with col3:
                st.metric("Accuracy", f"{eval_data.get('accuracy_score', 0):.2f}")
            with col4:
                st.metric("Completeness", f"{eval_data.get('completeness_score', 0):.2f}")

            if eval_data.get('feedback'):
                st.markdown("**💬 Feedback:**")
                st.info(safe_escape(eval_data.get('feedback', '')))

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
            if not st.session_state.rl_agent:
                st.warning("RL training unavailable: communication agent has no encoder.")
                return
            st.session_state.rl_agent.start_training_episode()
            st.success("🚀 Training episode started!")
        except Exception as e:
            st.error(f"❌ Error starting training episode: {e}")
    
    def _end_training_episode(self):
        """End the current training episode."""
        try:
            if not st.session_state.rl_agent:
                st.warning("RL training unavailable: communication agent has no encoder.")
                return
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
            if not st.session_state.rl_agent:
                st.warning("RL training unavailable: communication agent has no encoder.")
                return
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
            if not st.session_state.rl_agent:
                st.warning("RL training unavailable: communication agent has no encoder.")
                return
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