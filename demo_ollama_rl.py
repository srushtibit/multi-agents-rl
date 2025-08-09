"""
Demo script for Ollama Reinforcement Learning integration.

This script demonstrates how to use the new Ollama RL system
both programmatically and through the Streamlit interface.
"""

import asyncio
import logging
from datetime import datetime

from agents.communication.communication_agent import CommunicationAgent
from rl.algorithms.ollama_reinforce import OllamaREINFORCEAgent
from rl.environments.ollama_support_environment import OllamaSupportEnvironment

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def demo_ollama_rl():
    """Demonstrate Ollama RL training."""
    
    print("🚀 Ollama Reinforcement Learning Demo")
    print("=" * 50)
    
    # Initialize components
    print("📦 Initializing components...")
    communication_agent = CommunicationAgent()
    rl_agent = OllamaREINFORCEAgent(communication_agent)
    environment = OllamaSupportEnvironment()
    
    print("✅ Components initialized successfully!")
    
    # Demo 1: Single training episode
    print("\n🎯 Demo 1: Single Training Episode")
    print("-" * 30)
    
    # Start training episode
    rl_agent.start_training_episode()
    
    # Reset environment
    state = environment.reset()
    print(f"📝 Query: {state.current_query}")
    print(f"📊 Category: {state.query_category}")
    print(f"🔢 Complexity: {state.query_complexity:.2f}")
    
    total_reward = 0.0
    steps = 0
    
    while steps < 5:  # Limit steps for demo
        # Process query with RL strategy selection
        processed_query, strategy_used = rl_agent.process_query_with_rl(state.current_query)
        
        print(f"\n🔄 Step {steps + 1}:")
        print(f"   Strategy: {strategy_used}")
        print(f"   Processed: {processed_query}")
        
        # Simulate response quality (in real scenario, this comes from actual processing)
        import random
        response_quality = random.uniform(0.5, 0.9)
        
        # Create action for environment
        action = {
            'strategy': strategy_used,
            'processed_query': processed_query,
            'response_quality': response_quality
        }
        
        # Step environment
        next_state, reward, done, info = environment.step(action)
        
        print(f"   Response Quality: {response_quality:.3f}")
        print(f"   Reward: {reward:.3f}")
        print(f"   User Satisfaction: {next_state.user_satisfaction:.3f}")
        
        # Provide reward to RL agent
        rl_agent.receive_reward(reward, environment.get_state_representation())
        
        total_reward += reward
        steps += 1
        state = next_state
        
        if done:
            print(f"   ✅ Episode completed! Resolved: {state.resolved}")
            break
    
    # End episode
    stats = rl_agent.end_training_episode()
    
    print(f"\n📊 Episode Results:")
    print(f"   Total Reward: {total_reward:.3f}")
    print(f"   Steps: {steps}")
    print(f"   Average Reward: {stats.average_reward:.3f}")
    print(f"   Exploration Rate: {stats.exploration_rate:.3f}")
    
    # Demo 2: Multiple episodes for learning
    print("\n🎯 Demo 2: Multiple Training Episodes")
    print("-" * 40)
    
    num_episodes = 10
    episode_rewards = []
    
    for episode in range(num_episodes):
        # Start episode
        rl_agent.start_training_episode()
        state = environment.reset()
        
        episode_reward = 0.0
        episode_steps = 0
        
        while episode_steps < 3:  # Short episodes for demo
            processed_query, strategy_used = rl_agent.process_query_with_rl(state.current_query)
            
            response_quality = random.uniform(0.4, 0.9)
            action = {
                'strategy': strategy_used,
                'processed_query': processed_query,
                'response_quality': response_quality
            }
            
            next_state, reward, done, _ = environment.step(action)
            rl_agent.receive_reward(reward, environment.get_state_representation())
            
            episode_reward += reward
            episode_steps += 1
            state = next_state
            
            if done:
                break
        
        # End episode
        stats = rl_agent.end_training_episode()
        episode_rewards.append(episode_reward)
        
        print(f"Episode {episode + 1:2d}: Reward={episode_reward:.3f}, Avg={stats.average_reward:.3f}, ε={stats.exploration_rate:.3f}")
    
    # Show learning progress
    print(f"\n📈 Learning Progress:")
    print(f"   First 3 episodes avg: {sum(episode_rewards[:3])/3:.3f}")
    print(f"   Last 3 episodes avg: {sum(episode_rewards[-3:])/3:.3f}")
    print(f"   Improvement: {(sum(episode_rewards[-3:])/3 - sum(episode_rewards[:3])/3):.3f}")
    
    # Demo 3: Strategy performance analysis
    print("\n🎯 Demo 3: Strategy Performance Analysis")
    print("-" * 42)
    
    training_stats = rl_agent.get_training_stats()
    strategy_performance = training_stats.get('strategy_performance', {})
    
    print("Strategy Performance:")
    for strategy, perf in strategy_performance.items():
        print(f"   {strategy.replace('_', ' ').title():15}: "
              f"Uses={perf['total_uses']:2d}, "
              f"Avg Reward={perf['average_reward']:.3f}, "
              f"Success Rate={perf['success_rate']*100:.1f}%")
    
    # Demo 4: Save and load model
    print("\n🎯 Demo 4: Model Persistence")
    print("-" * 25)

    # Save model
    import os
    os.makedirs("demo_models", exist_ok=True)
    model_path = f"demo_models/demo_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    rl_agent.save_model(model_path)
    print(f"💾 Model saved to: {model_path}")
    
    # Show final statistics
    final_stats = rl_agent.get_training_stats()
    print(f"\n📊 Final Training Statistics:")
    print(f"   Total Episodes: {final_stats['total_episodes']}")
    print(f"   Total Steps: {final_stats['total_steps']}")
    print(f"   Best Average Reward: {final_stats['best_average_reward']:.3f}")
    print(f"   Current Exploration Rate: {final_stats['current_exploration_rate']:.3f}")
    
    print("\n🎉 Demo completed successfully!")
    print("\n💡 Next Steps:")
    print("   1. Run the Streamlit app: streamlit run ui/streamlit_app.py")
    print("   2. Go to the 'Training Dashboard' tab")
    print("   3. Use the training controls to run more episodes")
    print("   4. Monitor performance in the 'System Monitoring' tab")
    print("   5. Try different queries to see strategy adaptation")

def demo_streamlit_integration():
    """Show how to use the RL system through Streamlit."""
    
    print("\n🖥️  Streamlit Integration Guide")
    print("=" * 40)
    
    print("""
The Ollama RL system is now fully integrated with your Streamlit app!

🎓 Training Dashboard Features:
   • Real-time training status and metrics
   • Strategy performance comparison
   • Interactive training controls
   • Progress visualization
   • Model saving and loading

📊 System Monitoring Features:
   • RL performance metrics
   • Strategy usage statistics
   • Current exploration rate
   • Best performance tracking

🎮 How to Use:
   1. Start the Streamlit app
   2. Initialize the system (if not already done)
   3. Go to 'Training Dashboard' tab
   4. Click 'Start Training Episode' to begin
   5. Use the chat interface to interact with queries
   6. Click 'End Training Episode' to complete
   7. Run 'Batch Training' for multiple episodes
   8. Monitor progress in real-time
   9. Save models when satisfied with performance

🔄 Automatic Learning:
   • The system learns from every user interaction
   • Strategy selection improves over time
   • Performance metrics are tracked continuously
   • Models are automatically checkpointed

📈 Performance Tracking:
   • View strategy effectiveness in real-time
   • Monitor learning curves and progress
   • Compare different training runs
   • Export results for analysis
    """)

if __name__ == "__main__":
    print("🤖 Ollama Reinforcement Learning Demo")
    print("Choose demo mode:")
    print("1. Programmatic RL Demo")
    print("2. Streamlit Integration Guide")
    
    choice = input("\nEnter choice (1 or 2): ").strip()
    
    if choice == "1":
        asyncio.run(demo_ollama_rl())
    elif choice == "2":
        demo_streamlit_integration()
    else:
        print("Invalid choice. Running programmatic demo...")
        asyncio.run(demo_ollama_rl())
