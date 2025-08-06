#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simple demo of the NexaCorp AI Support System.
Run this to test basic functionality.
"""

import pandas as pd
import yaml
from pathlib import Path
import os

def demo_knowledge_base():
    """Demo knowledge base functionality."""
    print("Demo: Knowledge Base Processing")
    
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
    print(f"Sample knowledge base: {len(df)} entries")
    print(df.head())
    
    return df

def demo_query_processing(kb_df):
    """Demo query processing."""
    print("\nDemo: Query Processing")
    
    queries = [
        "I can't access my email",
        "How do I reset my password?",
        "VPN is not working",
        "Need help installing software"
    ]
    
    for query in queries:
        print(f"\nQuery: '{query}'")
        
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
            print(f"  -> Best match: {best_match['issue']}")
            print(f"  -> Solution: {best_match['solution']}")
            print(f"  -> Category: {best_match['category']} | Priority: {best_match['priority']}")
        else:
            print("  -> No matches found")

def demo_multilingual():
    """Demo multilingual capabilities."""
    print("\nDemo: Multilingual Support")
    
    sample_queries = {
        "en": "I cannot access my account",
        "es": "No puedo acceder a mi cuenta", 
        "de": "Ich kann nicht auf mein Konto zugreifen",
        "fr": "Je ne peux pas acceder a mon compte"
    }
    
    try:
        from langdetect import detect
        
        for lang_code, query in sample_queries.items():
            detected = detect(query)
            status = "OK" if detected == lang_code else "DIFF"
            print(f"  {status} '{query}' -> Detected: {detected} (Expected: {lang_code})")
            
    except ImportError:
        print("  Language detection not available (install langdetect)")
        for lang_code, query in sample_queries.items():
            print(f"  {lang_code.upper()}: '{query}'")

def demo_dataset_analysis():
    """Demo analysis of your existing dataset."""
    print("\nDemo: Your Dataset Analysis")
    
    dataset_dir = Path("dataset")
    if dataset_dir.exists():
        csv_files = list(dataset_dir.glob("*.csv"))
        
        for csv_file in csv_files[:3]:  # Analyze first 3 CSV files
            try:
                print(f"\nAnalyzing: {csv_file.name}")
                df = pd.read_csv(csv_file, nrows=10)  # Read first 10 rows
                
                print(f"  Shape: {df.shape[0]} rows x {df.shape[1]} columns")
                print(f"  Columns: {list(df.columns)}")
                
                # Check for multilingual content
                if 'language' in df.columns:
                    languages = df['language'].value_counts()
                    print(f"  Languages found: {dict(languages)}")
                
                # Show sample content
                if 'subject' in df.columns:
                    print(f"  Sample subject: {df['subject'].iloc[0][:100]}...")
                elif 'body' in df.columns:
                    print(f"  Sample content: {df['body'].iloc[0][:100]}...")
                
            except Exception as e:
                print(f"  Error reading {csv_file.name}: {e}")
    else:
        print("  Dataset directory not found")

def main():
    """Run the demo."""
    print("NexaCorp AI Support System - Simple Demo")
    print("=" * 50)
    
    # Demo knowledge base
    kb_df = demo_knowledge_base()
    
    # Demo query processing
    demo_query_processing(kb_df)
    
    # Demo multilingual
    demo_multilingual()
    
    # Demo your dataset
    demo_dataset_analysis()
    
    print("\nDemo completed!")
    print("\nNext steps:")
    print("1. Launch UI: streamlit run ui/streamlit_app.py")
    print("2. Build full knowledge base: python train_system.py --build-kb")
    print("3. Start training: python train_system.py --episodes 100")

if __name__ == "__main__":
    main()
