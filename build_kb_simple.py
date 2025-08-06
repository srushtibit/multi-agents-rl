#!/usr/bin/env python3
"""
Simple knowledge base builder for your existing dataset.
This will process your CSV files and create a searchable knowledge base.
"""

import pandas as pd
import os
import json
from pathlib import Path
import pickle
from datetime import datetime

def create_simple_knowledge_base():
    """Create a simple knowledge base from your CSV files."""
    print("Building Knowledge Base from Your Dataset...")
    print("=" * 50)
    
    knowledge_entries = []
    
    # Process each CSV file in dataset directory
    dataset_dir = Path("dataset")
    csv_files = list(dataset_dir.glob("*.csv"))
    
    print(f"Found {len(csv_files)} CSV files to process")
    
    for csv_file in csv_files:
        print(f"\nProcessing: {csv_file.name}")
        
        try:
            # Try different encodings
            df = None
            for encoding in ['utf-8', 'latin-1', 'cp1252']:
                try:
                    df = pd.read_csv(csv_file, encoding=encoding)
                    print(f"  Successfully read with {encoding} encoding")
                    break
                except UnicodeDecodeError:
                    continue
            
            if df is None:
                print(f"  Error: Could not read {csv_file.name}")
                continue
            
            print(f"  Shape: {df.shape[0]} rows x {df.shape[1]} columns")
            
            # Extract knowledge entries
            for idx, row in df.iterrows():
                entry = {
                    'id': f"{csv_file.stem}_{idx}",
                    'source_file': csv_file.name,
                    'row_index': idx
                }
                
                # Extract text content based on available columns
                if 'subject' in df.columns and pd.notna(row.get('subject')):
                    entry['title'] = str(row['subject'])
                
                if 'body' in df.columns and pd.notna(row.get('body')):
                    entry['content'] = str(row['body'])
                
                if 'answer' in df.columns and pd.notna(row.get('answer')):
                    entry['answer'] = str(row['answer'])
                
                if 'language' in df.columns and pd.notna(row.get('language')):
                    entry['language'] = str(row['language'])
                else:
                    entry['language'] = 'unknown'
                
                if 'priority' in df.columns and pd.notna(row.get('priority')):
                    entry['priority'] = str(row['priority'])
                
                if 'queue' in df.columns and pd.notna(row.get('queue')):
                    entry['category'] = str(row['queue'])
                
                # Add all other columns as metadata
                entry['metadata'] = {}
                for col in df.columns:
                    if col not in ['subject', 'body', 'answer', 'language', 'priority', 'queue']:
                        if pd.notna(row.get(col)):
                            entry['metadata'][col] = str(row[col])
                
                # Only add entries with meaningful content
                if 'title' in entry or 'content' in entry:
                    knowledge_entries.append(entry)
            
            print(f"  Extracted {len([e for e in knowledge_entries if e['source_file'] == csv_file.name])} entries")
            
        except Exception as e:
            print(f"  Error processing {csv_file.name}: {e}")
    
    print(f"\nTotal knowledge entries: {len(knowledge_entries)}")
    
    # Save knowledge base
    kb_data = {
        'entries': knowledge_entries,
        'created_at': datetime.now().isoformat(),
        'total_entries': len(knowledge_entries),
        'source_files': [f.name for f in csv_files],
        'languages': list(set(entry['language'] for entry in knowledge_entries))
    }
    
    # Create knowledge base directory
    os.makedirs('kb', exist_ok=True)
    
    # Save as JSON
    with open('kb/simple_knowledge_base.json', 'w', encoding='utf-8') as f:
        json.dump(kb_data, f, indent=2, ensure_ascii=False)
    
    # Save as pickle for Python
    with open('kb/simple_knowledge_base.pkl', 'wb') as f:
        pickle.dump(kb_data, f)
    
    print(f"\nKnowledge base saved!")
    print(f"  - JSON format: kb/simple_knowledge_base.json")
    print(f"  - Python format: kb/simple_knowledge_base.pkl")
    
    # Display statistics
    print(f"\nKnowledge Base Statistics:")
    print(f"  Total entries: {len(knowledge_entries)}")
    print(f"  Languages: {kb_data['languages']}")
    print(f"  Source files: {len(kb_data['source_files'])}")
    
    # Language distribution
    lang_counts = {}
    for entry in knowledge_entries:
        lang = entry['language']
        lang_counts[lang] = lang_counts.get(lang, 0) + 1
    
    print(f"  Language distribution:")
    for lang, count in sorted(lang_counts.items()):
        print(f"    {lang}: {count} entries")
    
    return kb_data

def search_knowledge_base(query, kb_data, max_results=5):
    """Simple search function for the knowledge base."""
    print(f"\nSearching for: '{query}'")
    print("-" * 30)
    
    query_words = query.lower().split()
    matches = []
    
    for entry in kb_data['entries']:
        score = 0
        
        # Search in title
        if 'title' in entry:
            title_words = entry['title'].lower().split()
            for word in query_words:
                if any(word in title_word for title_word in title_words):
                    score += 3
        
        # Search in content
        if 'content' in entry:
            content_words = entry['content'].lower().split()
            for word in query_words:
                if any(word in content_word for content_word in content_words):
                    score += 2
        
        # Search in answer
        if 'answer' in entry:
            answer_words = entry['answer'].lower().split()
            for word in query_words:
                if any(word in answer_word for answer_word in answer_words):
                    score += 4  # Answers are very relevant
        
        if score > 0:
            matches.append((score, entry))
    
    # Sort by score
    matches.sort(key=lambda x: x[0], reverse=True)
    
    # Display results
    for i, (score, entry) in enumerate(matches[:max_results]):
        print(f"Result {i+1} (Score: {score}):")
        print(f"  Source: {entry['source_file']}")
        print(f"  Language: {entry['language']}")
        
        if 'title' in entry:
            print(f"  Title: {entry['title'][:100]}...")
        
        if 'content' in entry:
            print(f"  Content: {entry['content'][:150]}...")
        
        if 'answer' in entry:
            print(f"  Answer: {entry['answer'][:150]}...")
        
        print()
    
    if not matches:
        print("No matches found.")
    
    return matches[:max_results]

def main():
    """Main function."""
    print("NexaCorp AI Support System - Knowledge Base Builder")
    print("=" * 60)
    
    # Build knowledge base
    kb_data = create_simple_knowledge_base()
    
    # Test search functionality
    print("\n" + "=" * 60)
    print("Testing Search Functionality")
    print("=" * 60)
    
    test_queries = [
        "email problem",
        "password reset",
        "VPN connection",
        "Shopify payment",
        "account access"
    ]
    
    for query in test_queries:
        search_knowledge_base(query, kb_data, max_results=3)
    
    print("\n" + "=" * 60)
    print("Knowledge Base Ready!")
    print("You can now:")
    print("1. Use the search function above")
    print("2. Launch the full system: streamlit run ui/streamlit_app.py")
    print("3. Train the AI agents: python train_system.py --build-kb")
    print("=" * 60)

if __name__ == "__main__":
    main()