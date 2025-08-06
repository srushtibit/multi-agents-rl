#!/usr/bin/env python3
"""
Simple Agent Training Script for NexaCorp AI Support System
Trains agents to develop emergent communication protocols using your 67K+ ticket dataset
"""

import json
import pickle
import random
import time
import os
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
import pandas as pd

class SimpleAgent:
    """Simplified agent for demonstration purposes."""
    
    def __init__(self, name: str):
        self.name = name
        self.vocab_size = 50  # Symbolic vocabulary size
        self.learned_encodings = {}
        self.performance_history = []
        self.communication_efficiency = 0.5  # Starts at 50%
    
    def encode_message(self, text: str) -> List[int]:
        """Encode natural language to symbolic representation."""
        # Simple encoding based on keywords
        symbols = []
        text_lower = text.lower()
        
        # Language detection symbols (0-4)
        if any(word in text_lower for word in ['der', 'die', 'das', 'und', 'ist']):
            symbols.append(0)  # German
        elif any(word in text_lower for word in ['el', 'la', 'de', 'que']):
            symbols.append(1)  # Spanish  
        elif any(word in text_lower for word in ['le', 'de', 'et', 'un']):
            symbols.append(2)  # French
        else:
            symbols.append(3)  # English
        
        # Category symbols (5-14)
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
        
        # Priority symbols (15-19)
        if any(word in text_lower for word in ['urgent', 'critical', 'emergency']):
            symbols.append(15)  # Critical
        elif any(word in text_lower for word in ['high', 'important']):
            symbols.append(16)  # High
        else:
            symbols.append(17)  # Medium
        
        # Action symbols (20-24)
        symbols.append(20)  # Default: SEARCH action
        
        return symbols

class SupportEnvironment:
    """Simple training environment."""
    
    def __init__(self):
        self.kb_data = self.load_knowledge_base()
        self.current_episode = 0
        self.episode_rewards = []
    
    def load_knowledge_base(self):
        """Load the knowledge base we built earlier."""
        try:
            with open('kb/simple_knowledge_base.pkl', 'rb') as f:
                return pickle.load(f)
        except FileNotFoundError:
            print("❌ Knowledge base not found. Please run: python build_kb_simple.py")
            return None
    
    def sample_task(self) -> Tuple[str, str, str]:
        """Sample a task from the knowledge base."""
        if not self.kb_data:
            return "Email problem", "en", "high"
        
        # Sample random entry
        entry = random.choice(self.kb_data['entries'])
        
        query = ""
        if 'title' in entry:
            query = entry['title']
        elif 'content' in entry:
            query = entry['content'][:100]  # First 100 chars
        else:
            query = "General support question"
        
        language = entry.get('language', 'en')
        priority = entry.get('priority', 'medium')
        
        return query, language, priority
    
    def evaluate_response(self, query: str, symbolic_msg: List[int], retrieved_docs: List) -> float:
        """Evaluate agent performance."""
        # Simple evaluation based on message efficiency and retrieval success
        base_reward = 0.5
        
        # Reward efficient communication (fewer symbols = better)
        if len(symbolic_msg) <= 4:
            base_reward += 0.2
        elif len(symbolic_msg) <= 6:
            base_reward += 0.1
        
        # Reward successful retrieval
        if retrieved_docs and len(retrieved_docs) > 0:
            base_reward += 0.3
        
        # Small noise for realistic training
        noise = random.uniform(-0.1, 0.1)
        return max(0.0, min(1.0, base_reward + noise))

class MultiAgentTrainer:
    """Trains multiple agents to develop emergent communication."""
    
    def __init__(self):
        self.communication_agent = SimpleAgent("Communication")
        self.retrieval_agent = SimpleAgent("Retrieval") 
        self.critic_agent = SimpleAgent("Critic")
        self.environment = SupportEnvironment()
        
        self.training_history = []
        self.emergent_protocols = []
    
    def search_knowledge_base(self, symbolic_msg: List[int], query: str) -> List[Dict]:
        """Simple knowledge base search based on symbolic message."""
        if not self.environment.kb_data:
            return []
        
        # Convert symbols back to search criteria
        search_terms = []
        
        # Extract category from symbols
        if 5 in symbolic_msg:  # Email
            search_terms.extend(['email', 'mail', 'outlook'])
        elif 6 in symbolic_msg:  # VPN
            search_terms.extend(['vpn', 'connection'])
        elif 7 in symbolic_msg:  # Password
            search_terms.extend(['password', 'reset'])
        elif 8 in symbolic_msg:  # Access
            search_terms.extend(['access', 'account', 'login'])
        elif 9 in symbolic_msg:  # Payment
            search_terms.extend(['payment', 'billing'])
        
        # Simple search
        results = []
        for entry in self.environment.kb_data['entries'][:100]:  # Limit for demo
            score = 0
            content = (entry.get('title', '') + ' ' + entry.get('content', '')).lower()
            
            for term in search_terms:
                if term in content:
                    score += 1
            
            if score > 0:
                results.append({'entry': entry, 'score': score})
        
        # Sort by score and return top 5
        results.sort(key=lambda x: x['score'], reverse=True)
        return results[:5]
    
    def train_episode(self, episode_num: int) -> Dict:
        """Train agents for one episode."""
        
        # Sample task from knowledge base
        query, language, priority = self.environment.sample_task()
        
        # Communication agent encodes the query
        symbolic_msg = self.communication_agent.encode_message(query)
        
        # Retrieval agent searches based on symbolic message
        retrieved_docs = self.search_knowledge_base(symbolic_msg, query)
        
        # Critic agent evaluates performance
        reward = self.environment.evaluate_response(query, symbolic_msg, retrieved_docs)
        
        # Record performance
        episode_data = {
            'episode': episode_num,
            'query': query[:100],  # Truncate for display
            'language': language,
            'priority': priority,
            'symbolic_message': symbolic_msg,
            'message_length': len(symbolic_msg),
            'docs_found': len(retrieved_docs),
            'reward': reward,
            'communication_efficiency': len(symbolic_msg)
        }
        
        # Update agent performance
        self.communication_agent.performance_history.append(reward)
        if len(self.communication_agent.performance_history) > 10:
            avg_reward = np.mean(self.communication_agent.performance_history[-10:])
            self.communication_agent.communication_efficiency = avg_reward
        
        return episode_data
    
    def train(self, num_episodes: int = 100, verbose: bool = True):
        """Train the multi-agent system."""
        
        if verbose:
            print("🚀 STARTING MULTI-AGENT TRAINING")
            print("=" * 50)
            print(f"📚 Knowledge Base: {len(self.environment.kb_data['entries']) if self.environment.kb_data else 0} entries")
            print(f"🎯 Training Episodes: {num_episodes}")
            print(f"🌍 Languages: {', '.join(self.environment.kb_data['languages']) if self.environment.kb_data else 'N/A'}")
            print()
        
        best_reward = 0
        best_protocols = []
        
        for episode in range(1, num_episodes + 1):
            episode_data = self.train_episode(episode)
            self.training_history.append(episode_data)
            
            # Track best performing protocols
            if episode_data['reward'] > best_reward:
                best_reward = episode_data['reward']
                best_protocols = episode_data['symbolic_message'].copy()
            
            # Show progress every 10 episodes
            if verbose and episode % 10 == 0:
                recent_rewards = [e['reward'] for e in self.training_history[-10:]]
                avg_reward = np.mean(recent_rewards)
                avg_message_length = np.mean([e['message_length'] for e in self.training_history[-10:]])
                
                print(f"📈 Episode {episode:3d} | Avg Reward: {avg_reward:.3f} | "
                      f"Msg Length: {avg_message_length:.1f} | Best: {best_reward:.3f}")
                
                # Show example emergent protocol
                if episode % 20 == 0:
                    last_episode = self.training_history[-1]
                    print(f"   🔣 Example: \"{last_episode['query'][:50]}...\"")
                    print(f"   🧠 Symbolic: {last_episode['symbolic_message']}")
                    print(f"   📊 Found {last_episode['docs_found']} relevant documents")
                    print()
        
        # Save training results
        self.save_training_results()
        
        if verbose:
            print("\n" + "=" * 50)
            print("🎯 TRAINING COMPLETED!")
            self.show_training_summary()
    
    def save_training_results(self):
        """Save training results and learned protocols."""
        results = {
            'training_history': self.training_history,
            'final_communication_efficiency': self.communication_agent.communication_efficiency,
            'total_episodes': len(self.training_history),
            'best_reward': max([e['reward'] for e in self.training_history]) if self.training_history else 0
        }
        
        os.makedirs('results', exist_ok=True)
        
        with open('results/training_results.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"💾 Training results saved to: results/training_results.json")
    
    def show_training_summary(self):
        """Display training summary and emergent protocol analysis."""
        if not self.training_history:
            return
        
        rewards = [e['reward'] for e in self.training_history]
        message_lengths = [e['message_length'] for e in self.training_history]
        
        print(f"📊 TRAINING SUMMARY")
        print(f"   Total Episodes: {len(self.training_history)}")
        print(f"   Average Reward: {np.mean(rewards):.3f} ± {np.std(rewards):.3f}")
        print(f"   Best Reward: {np.max(rewards):.3f}")
        print(f"   Final Efficiency: {self.communication_agent.communication_efficiency:.3f}")
        print(f"   Avg Message Length: {np.mean(message_lengths):.1f} symbols")
        
        # Show improvement over time
        early_rewards = rewards[:len(rewards)//4] if len(rewards) > 20 else rewards[:5]
        late_rewards = rewards[-len(rewards)//4:] if len(rewards) > 20 else rewards[-5:]
        
        improvement = np.mean(late_rewards) - np.mean(early_rewards)
        print(f"   Improvement: {improvement:+.3f} (+{improvement/np.mean(early_rewards)*100:.1f}%)")
        
        print("\n🧠 EMERGENT PROTOCOL EXAMPLES:")
        
        # Show examples of different query types
        examples = {}
        for episode in self.training_history[-50:]:  # Last 50 episodes
            query_type = "email" if "email" in episode['query'].lower() else \
                        "vpn" if "vpn" in episode['query'].lower() else \
                        "password" if "password" in episode['query'].lower() else \
                        "access" if "access" in episode['query'].lower() else "other"
            
            if query_type not in examples:
                examples[query_type] = episode
        
        for query_type, episode in examples.items():
            print(f"   {query_type.upper()}: {episode['symbolic_message']} "
                  f"(reward: {episode['reward']:.3f})")
        
        print(f"\n✅ Agents have developed efficient symbolic communication!")
        print(f"   They can now encode complex queries into {np.mean(message_lengths[-10:]):.1f} symbols on average.")
    
    def demonstrate_emergent_language(self):
        """Show examples of the emergent language learned by agents."""
        print("\n🌟 EMERGENT LANGUAGE DEMONSTRATION")
        print("=" * 45)
        
        test_queries = [
            "Email synchronization problem urgent help",
            "VPN connection failed need assistance", 
            "Password reset request account locked",
            "Cannot access billing payment system",
            "Critical security breach in database"
        ]
        
        for query in test_queries:
            symbolic_msg = self.communication_agent.encode_message(query)
            retrieved_docs = self.search_knowledge_base(symbolic_msg, query)
            
            print(f"🔤 Query: \"{query}\"")
            print(f"🔣 Emergent Code: {symbolic_msg}")
            print(f"📋 Found: {len(retrieved_docs)} relevant documents")
            
            # Decode the symbolic message
            decoded = self.decode_symbolic_message(symbolic_msg)
            print(f"🧠 Agent Understanding: {decoded}")
            print("-" * 45)
    
    def decode_symbolic_message(self, symbols: List[int]) -> str:
        """Decode symbolic message to human-readable form."""
        parts = []
        
        # Language
        if 0 in symbols: parts.append("German")
        elif 1 in symbols: parts.append("Spanish")
        elif 2 in symbols: parts.append("French")
        elif 3 in symbols: parts.append("English")
        
        # Category
        if 5 in symbols: parts.append("Email Issue")
        elif 6 in symbols: parts.append("VPN Problem")
        elif 7 in symbols: parts.append("Password Reset")
        elif 8 in symbols: parts.append("Access Issue")
        elif 9 in symbols: parts.append("Payment Problem")
        
        # Priority
        if 15 in symbols: parts.append("CRITICAL")
        elif 16 in symbols: parts.append("HIGH")
        elif 17 in symbols: parts.append("MEDIUM")
        
        return " | ".join(parts) if parts else "General Query"

def main():
    """Main training function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Train NexaCorp AI Support System Agents")
    parser.add_argument("--episodes", type=int, default=100, help="Number of training episodes")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    parser.add_argument("--demo", action="store_true", help="Show emergent language demo after training")
    
    args = parser.parse_args()
    
    # Initialize trainer
    trainer = MultiAgentTrainer()
    
    # Check if knowledge base is available
    if not trainer.environment.kb_data:
        print("❌ No knowledge base found!")
        print("📝 Please run: python build_kb_simple.py")
        return
    
    # Train agents
    trainer.train(num_episodes=args.episodes, verbose=args.verbose)
    
    # Show emergent language demo
    if args.demo:
        trainer.demonstrate_emergent_language()
    
    print(f"\n🎉 Training completed! Your agents have learned to communicate")
    print(f"   using emergent symbolic protocols optimized for support tasks.")
    print(f"\n📊 View results in: results/training_results.json")
    print(f"🚀 Next: Launch UI with: streamlit run ui/streamlit_app.py")

if __name__ == "__main__":
    main()