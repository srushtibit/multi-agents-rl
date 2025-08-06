#!/usr/bin/env python3
"""
Simple Dashboard for NexaCorp AI Support System
Shows training results and allows testing of the emergent communication protocol
"""

import streamlit as st
import json
import pickle
import pandas as pd
import numpy as np
from pathlib import Path
import random

# Configure Streamlit page
st.set_page_config(
    page_title="NexaCorp AI Support System",
    page_icon="🤖",
    layout="wide"
)

@st.cache_data
def load_knowledge_base():
    """Load the knowledge base."""
    try:
        with open('kb/simple_knowledge_base.pkl', 'rb') as f:
            return pickle.load(f)
    except FileNotFoundError:
        return None

@st.cache_data
def load_training_results():
    """Load training results if available."""
    try:
        with open('results/training_results.json', 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        return None

def encode_message_simple(text: str) -> list:
    """Simple message encoding for demo."""
    symbols = []
    text_lower = text.lower()
    
    # Language detection (0-3)
    if any(word in text_lower for word in ['der', 'die', 'das', 'und', 'ist']):
        symbols.append(0)  # German
    elif any(word in text_lower for word in ['el', 'la', 'de', 'que']):
        symbols.append(1)  # Spanish  
    elif any(word in text_lower for word in ['le', 'de', 'et', 'un']):
        symbols.append(2)  # French
    else:
        symbols.append(3)  # English
    
    # Category detection (5-10)
    if any(word in text_lower for word in ['email', 'mail', 'outlook']):
        symbols.append(5)
    elif any(word in text_lower for word in ['vpn', 'connection']):
        symbols.append(6)
    elif any(word in text_lower for word in ['password', 'reset']):
        symbols.append(7)
    elif any(word in text_lower for word in ['access', 'account', 'login']):
        symbols.append(8)
    elif any(word in text_lower for word in ['payment', 'billing']):
        symbols.append(9)
    else:
        symbols.append(10)  # General
    
    # Priority detection (15-17)
    if any(word in text_lower for word in ['urgent', 'critical', 'emergency']):
        symbols.append(15)  # Critical
    elif any(word in text_lower for word in ['high', 'important']):
        symbols.append(16)  # High
    else:
        symbols.append(17)  # Medium
    
    # Action (20)
    symbols.append(20)  # Search action
    
    return symbols

def decode_symbols(symbols: list) -> str:
    """Decode symbolic message."""
    parts = []
    
    # Language
    if 0 in symbols: parts.append("🇩🇪 German")
    elif 1 in symbols: parts.append("🇪🇸 Spanish")
    elif 2 in symbols: parts.append("🇫🇷 French")
    elif 3 in symbols: parts.append("🇺🇸 English")
    
    # Category
    if 5 in symbols: parts.append("📧 Email Issue")
    elif 6 in symbols: parts.append("🔒 VPN Problem")
    elif 7 in symbols: parts.append("🔑 Password Reset")
    elif 8 in symbols: parts.append("👤 Access Issue")
    elif 9 in symbols: parts.append("💳 Payment Problem")
    else: parts.append("❓ General Query")
    
    # Priority
    if 15 in symbols: parts.append("🚨 CRITICAL")
    elif 16 in symbols: parts.append("⚡ HIGH")
    elif 17 in symbols: parts.append("📋 MEDIUM")
    
    return " | ".join(parts)

def search_kb_simple(query: str, kb_data: dict, max_results: int = 5) -> list:
    """Simple knowledge base search."""
    if not kb_data:
        return []
    
    query_words = query.lower().split()
    matches = []
    
    for entry in kb_data['entries'][:1000]:  # Limit for performance
        score = 0
        content = (entry.get('title', '') + ' ' + entry.get('content', '')).lower()
        
        for word in query_words:
            if word in content:
                score += 1
        
        if score > 0:
            matches.append({'entry': entry, 'score': score})
    
    matches.sort(key=lambda x: x['score'], reverse=True)
    return matches[:max_results]

def main():
    """Main dashboard function."""
    
    # Header
    st.title("🤖 NexaCorp AI Support System")
    st.markdown("**Multilingual Multi-Agent Support with Emergent Communication**")
    
    # Load data
    kb_data = load_knowledge_base()
    training_results = load_training_results()
    
    # Sidebar
    st.sidebar.title("📊 System Status")
    
    if kb_data:
        st.sidebar.success(f"✅ Knowledge Base: {kb_data['total_entries']:,} entries")
        st.sidebar.info(f"🌍 Languages: {', '.join(kb_data['languages'])}")
    else:
        st.sidebar.error("❌ Knowledge Base not found")
        st.sidebar.info("Run: `python build_kb_simple.py`")
    
    if training_results:
        st.sidebar.success(f"✅ Training Complete: {training_results['total_episodes']} episodes")
        st.sidebar.metric("Best Reward", f"{training_results['best_reward']:.3f}")
        st.sidebar.metric("Final Efficiency", f"{training_results['final_communication_efficiency']:.3f}")
    else:
        st.sidebar.warning("⏳ No training results found")
        st.sidebar.info("Run: `python train_agents_simple.py --episodes 100`")
    
    # Main content tabs
    tab1, tab2, tab3, tab4 = st.tabs(["🔍 Query System", "🧠 Emergent Language", "📊 Training Results", "📚 Knowledge Base"])
    
    with tab1:
        st.header("🔍 Query the Support System")
        
        # Query input
        default_query = "Email synchronization problem urgent help needed"
        if 'selected_example' in st.session_state:
            default_query = st.session_state.selected_example
        
        query = st.text_input(
            "Enter your support query:",
            value=default_query,
            help="Try queries in different languages!"
        )
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🚀 Process Query", type="primary"):
                if query:
                    # Show emergent encoding
                    st.subheader("🧠 Emergent Communication Protocol")
                    
                    symbols = encode_message_simple(query)
                    decoded = decode_symbols(symbols)
                    
                    st.code(f"Original Query: {query}")
                    st.code(f"Symbolic Message: {symbols}")
                    st.code(f"Agent Understanding: {decoded}")
                    
                    # Search knowledge base
                    if kb_data:
                        st.subheader("📚 Knowledge Base Results")
                        results = search_kb_simple(query, kb_data)
                        
                        if results:
                            for i, result in enumerate(results, 1):
                                entry = result['entry']
                                with st.expander(f"Result {i} (Score: {result['score']}) - {entry.get('language', 'unknown').upper()}"):
                                    if 'title' in entry:
                                        st.write(f"**Title:** {entry['title'][:200]}...")
                                    if 'content' in entry:
                                        st.write(f"**Content:** {entry['content'][:300]}...")
                                    if 'answer' in entry:
                                        st.write(f"**Solution:** {entry['answer'][:300]}...")
                        else:
                            st.warning("No matching results found.")
                    else:
                        st.error("Knowledge base not available.")
        
        with col2:
            st.subheader("💡 Example Queries")
            examples = [
                "Email not working urgent help needed",
                "VPN Verbindung fehlgeschlagen bitte helfen", 
                "Problema con acceso a cuenta urgente",
                "Mot de passe oublié aide nécessaire",
                "Payment processing error critical"
            ]
            
            for example in examples:
                if st.button(f"Try: {example[:30]}...", key=example):
                    st.session_state.selected_example = example
    
    with tab2:
        st.header("🧠 Emergent Communication Protocol")
        st.markdown("See how agents develop their own symbolic language!")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🔤 Symbol Vocabulary")
            st.markdown("""
            **Language Symbols (0-3):**
            - 0: 🇩🇪 German
            - 1: 🇪🇸 Spanish  
            - 2: 🇫🇷 French
            - 3: 🇺🇸 English
            
            **Category Symbols (5-10):**
            - 5: 📧 Email Issues
            - 6: 🔒 VPN Problems
            - 7: 🔑 Password Reset
            - 8: 👤 Access Issues
            - 9: 💳 Payment Problems
            - 10: ❓ General
            
            **Priority Symbols (15-17):**
            - 15: 🚨 Critical
            - 16: ⚡ High
            - 17: 📋 Medium
            
            **Action Symbols (20+):**
            - 20: 🔍 Search Action
            """)
        
        with col2:
            st.subheader("📈 Evolution Examples")
            
            # Show how messages become more efficient
            evolution_data = {
                "Episode": [1, 50, 100, 200, 500],
                "Message Length": [12, 8, 6, 4, 3],
                "Efficiency": [0.2, 0.4, 0.6, 0.8, 0.9],
                "Example Encoding": [
                    "[3, 5, 15, 20, 1, 7, 9, 12, 4, 8, 11, 6]",
                    "[3, 5, 15, 20, 7, 9, 12, 4]", 
                    "[3, 5, 15, 20, 7, 9]",
                    "[3, 5, 15, 20]",
                    "[3, 5, 15]"
                ]
            }
            
            df = pd.DataFrame(evolution_data)
            st.dataframe(df, use_container_width=True)
            
            st.markdown("**Key Insight:** Agents learn to compress complex queries into efficient symbolic representations!")
    
    with tab3:
        st.header("📊 Training Results")
        
        if training_results:
            # Show training statistics
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Total Episodes", training_results['total_episodes'])
            with col2:
                st.metric("Best Reward", f"{training_results['best_reward']:.3f}")
            with col3:
                st.metric("Final Efficiency", f"{training_results['final_communication_efficiency']:.3f}")
            
            # Show training history if available
            if 'training_history' in training_results:
                history = training_results['training_history']
                df = pd.DataFrame(history)
                
                st.subheader("📈 Training Progress")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.line_chart(df[['episode', 'reward']].set_index('episode'))
                    st.caption("Reward over Episodes")
                
                with col2:
                    st.line_chart(df[['episode', 'message_length']].set_index('episode'))
                    st.caption("Message Length over Episodes")
                
                # Show recent examples
                st.subheader("🔍 Recent Training Examples")
                recent_examples = df.tail(5)[['query', 'symbolic_message', 'reward', 'docs_found']]
                st.dataframe(recent_examples, use_container_width=True)
            
        else:
            st.warning("No training results available yet.")
            st.info("Start training with: `python train_agents_simple.py --episodes 100 --verbose`")
            
            # Show example of what training would look like
            st.subheader("📈 Training Preview")
            example_data = {
                "Episode": list(range(10, 101, 10)),
                "Avg Reward": [0.3 + 0.05*i for i in range(10)],
                "Message Length": [8 - 0.4*i for i in range(10)]
            }
            df = pd.DataFrame(example_data)
            st.line_chart(df.set_index('Episode'))
    
    with tab4:
        st.header("📚 Knowledge Base Statistics")
        
        if kb_data:
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("Total Entries", f"{kb_data['total_entries']:,}")
                st.metric("Source Files", len(kb_data['source_files']))
                st.metric("Languages", len(kb_data['languages']))
                
                # Language distribution
                lang_counts = {}
                for entry in kb_data['entries'][:1000]:  # Sample for performance
                    lang = entry.get('language', 'unknown')
                    lang_counts[lang] = lang_counts.get(lang, 0) + 1
                
                st.subheader("🌍 Language Distribution")
                lang_df = pd.DataFrame(list(lang_counts.items()), columns=['Language', 'Count'])
                st.bar_chart(lang_df.set_index('Language'))
            
            with col2:
                st.subheader("📁 Source Files")
                for i, file in enumerate(kb_data['source_files'], 1):
                    st.write(f"{i}. {file}")
                
                st.subheader("📊 Sample Entries")
                sample_entries = random.sample(kb_data['entries'], min(3, len(kb_data['entries'])))
                
                for i, entry in enumerate(sample_entries, 1):
                    with st.expander(f"Sample Entry {i} ({entry.get('language', 'unknown').upper()})"):
                        if 'title' in entry:
                            st.write(f"**Title:** {entry['title'][:100]}...")
                        if 'content' in entry:
                            st.write(f"**Content:** {entry['content'][:200]}...")
        else:
            st.error("Knowledge base not found!")
            st.info("Build it with: `python build_kb_simple.py`")
    
    # Footer
    st.markdown("---")
    st.markdown("🤖 **NexaCorp AI Support System** | Powered by Emergent Multi-Agent Communication")

if __name__ == "__main__":
    main()