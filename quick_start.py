#!/usr/bin/env python3
"""
Quick start script for the Multilingual Multi-Agent Support System.
Sets up a minimal working system for immediate testing.
"""

import os
import sys
import asyncio
from pathlib import Path

def setup_basic_system():
    """Setup basic system without complex dependencies."""
    print("🚀 Setting up basic NexaCorp AI Support System...")
    
    # Create basic directories
    directories = ["logs", "models", "cache", "results"]
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        print(f"✅ Created directory: {directory}")
    
    # Create a basic configuration
    basic_config = """
# Basic system configuration for quick start
system:
  name: "NexaCorp AI Support System - Quick Start"
  environment: "local"
  debug: true

languages:
  supported: ["en", "es", "de"]
  primary: "en"
  auto_detect: true

agents:
  communication:
    symbolic_vocab_size: 100
    learning_rate: 0.001
  
  retrieval:
    max_documents: 10
    
  escalation:
    severity_threshold: 0.8

knowledge_base:
  similarity_threshold: 0.7
  max_results: 5
"""
    
    # Write basic config
    with open("config/quick_start_config.yaml", "w") as f:
        f.write(basic_config)
    
    print("✅ Created basic configuration")

async def test_basic_functionality():
    """Test basic functionality without complex dependencies."""
    print("\n🧪 Testing basic functionality...")
    
    try:
        # Test basic imports
        import pandas as pd
        import numpy as np
        print("✅ Basic data processing imports working")
        
        # Test YAML config loading
        import yaml
        with open("config/quick_start_config.yaml", "r") as f:
            config = yaml.safe_load(f)
        print("✅ Configuration loading working")
        
        # Test basic language detection
        try:
            from langdetect import detect
            lang = detect("Hello, how are you?")
            print(f"✅ Language detection working: '{lang}'")
        except ImportError:
            print("⚠️ Language detection not available (install langdetect)")
        
        # Test document processing capabilities
        if os.path.exists("dataset"):
            files = list(Path("dataset").glob("*.csv"))[:2]  # Check first 2 CSV files
            if files:
                for file in files:
                    try:
                        df = pd.read_csv(file, nrows=5)  # Read just 5 rows
                        print(f"✅ Can read {file.name}: {len(df)} rows, {len(df.columns)} columns")
                    except Exception as e:
                        print(f"⚠️ Issue with {file.name}: {e}")
            else:
                print("ℹ️ No CSV files found in dataset directory")
        else:
            print("ℹ️ Dataset directory not found")
        
        return True
        
    except Exception as e:
        print(f"❌ Basic functionality test failed: {e}")
        return False

def create_simple_demo():
    """Create a simple demo script."""
    demo_script = '''#!/usr/bin/env python3
"""
Simple demo of the NexaCorp AI Support System.
Run this to test basic functionality.
"""

import pandas as pd
import yaml
from pathlib import Path

def demo_knowledge_base():
    """Demo knowledge base functionality."""
    print("📚 Demo: Knowledge Base Processing")
    
    # Create sample data
    sample_data = {
        "issue": [
            "Email not syncing",
            "Password reset required", 
            "VPN connection failed",
            "Software installation help",
            "Account access problem"
        ],
        "solution": [
            "Check network connection and reconfigure email client settings",
            "Visit the password reset page and follow the instructions",
            "Restart VPN client and verify credentials",
            "Download installer from company portal and run as administrator",
            "Contact IT support for account unlock procedures"
        ],
        "category": [
            "IT Support",
            "Account Management",
            "Network",
            "Software",
            "Security"
        ],
        "priority": ["Medium", "High", "Medium", "Low", "High"]
    }
    
    df = pd.DataFrame(sample_data)
    print(f"✅ Sample knowledge base: {len(df)} entries")
    print(df.head())
    
    return df

def demo_query_processing(kb_df):
    """Demo query processing."""
    print("\\n🔍 Demo: Query Processing")
    
    queries = [
        "I can't access my email",
        "How do I reset my password?",
        "VPN is not working",
        "Need help installing software"
    ]
    
    for query in queries:
        print(f"\\nQuery: '{query}'")
        
        # Simple keyword matching (basic version)
        matches = []
        query_words = query.lower().split()
        
        for idx, row in kb_df.iterrows():
            issue_words = row['issue'].lower().split()
            solution_words = row['solution'].lower().split()
            
            # Count word matches
            matches_count = 0
            for word in query_words:
                if any(word in issue_word for issue_word in issue_words):
                    matches_count += 2  # Issue matches are more important
                if any(word in solution_word for solution_word in solution_words):
                    matches_count += 1
            
            if matches_count > 0:
                matches.append((matches_count, row))
        
        # Sort by match score
        matches.sort(key=lambda x: x[0], reverse=True)
        
        if matches:
            best_match = matches[0][1]
            print(f"  → Best match: {best_match['issue']}")
            print(f"  → Solution: {best_match['solution']}")
            print(f"  → Category: {best_match['category']} | Priority: {best_match['priority']}")
        else:
            print("  → No matches found")

def demo_multilingual():
    """Demo multilingual capabilities."""
    print("\\n🌐 Demo: Multilingual Support")
    
    sample_queries = {
        "en": "I cannot access my account",
        "es": "No puedo acceder a mi cuenta", 
        "de": "Ich kann nicht auf mein Konto zugreifen",
        "fr": "Je ne peux pas accéder à mon compte"
    }
    
    try:
        from langdetect import detect
        
        for lang_code, query in sample_queries.items():
            detected = detect(query)
            status = "✅" if detected == lang_code else "⚠️"
            print(f"  {status} '{query}' → Detected: {detected} (Expected: {lang_code})")
            
    except ImportError:
        print("  ⚠️ Language detection not available (install langdetect)")
        for lang_code, query in sample_queries.items():
            print(f"  📝 {lang_code.upper()}: '{query}'")

def main():
    """Run the demo."""
    print("🤖 NexaCorp AI Support System - Simple Demo")
    print("=" * 50)
    
    # Demo knowledge base
    kb_df = demo_knowledge_base()
    
    # Demo query processing
    demo_query_processing(kb_df)
    
    # Demo multilingual
    demo_multilingual()
    
    print("\\n🎉 Demo completed!")
    print("\\nNext steps:")
    print("1. Install full dependencies: python install.py")
    print("2. Launch UI: streamlit run ui/streamlit_app.py")
    print("3. Build full knowledge base: python train_system.py --build-kb")

if __name__ == "__main__":
    main()
'''
    
    with open("simple_demo.py", "w") as f:
        f.write(demo_script)
    
    print("✅ Created simple demo script: simple_demo.py")

def main():
    """Main quick start process."""
    print("🚀 NexaCorp AI Support System - Quick Start Setup")
    print("=" * 60)
    
    # Setup basic system
    setup_basic_system()
    
    # Test functionality
    test_result = asyncio.run(test_basic_functionality())
    
    # Create demo
    create_simple_demo()
    
    print("\n" + "=" * 60)
    if test_result:
        print("🎉 QUICK START SETUP COMPLETED!")
        print("\n🚀 What you can do now:")
        print("1. Run simple demo: python simple_demo.py")
        print("2. Install full system: python install.py")
        print("3. Launch basic UI: streamlit run ui/streamlit_app.py")
        print("4. Process your dataset: python train_system.py --build-kb")
    else:
        print("⚠️ SETUP COMPLETED WITH ISSUES")
        print("Try installing minimal dependencies first:")
        print("pip install pandas numpy pyyaml langdetect")
    
    print("=" * 60)

if __name__ == "__main__":
    main()