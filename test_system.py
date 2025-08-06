#!/usr/bin/env python3
"""
Test the NexaCorp AI Support System with your multilingual dataset.
This version works with current dependencies.
"""

import json
import pickle
from pathlib import Path

def load_knowledge_base():
    """Load the knowledge base we built."""
    try:
        with open('kb/simple_knowledge_base.pkl', 'rb') as f:
            return pickle.load(f)
    except FileNotFoundError:
        print("Knowledge base not found. Please run: python build_kb_simple.py")
        return None

def detect_language_simple(text):
    """Simple language detection based on common words."""
    text_lower = text.lower()
    
    # German indicators
    german_words = ['der', 'die', 'das', 'und', 'ist', 'ich', 'ein', 'eine', 'mit', 'nicht', 'von', 'zu', 'sie', 'er', 'auf', 'für']
    german_score = sum(1 for word in german_words if word in text_lower)
    
    # English indicators
    english_words = ['the', 'and', 'is', 'to', 'of', 'a', 'in', 'that', 'have', 'it', 'for', 'not', 'on', 'with', 'he', 'as']
    english_score = sum(1 for word in english_words if word in text_lower)
    
    # Spanish indicators
    spanish_words = ['el', 'la', 'de', 'que', 'y', 'en', 'un', 'es', 'se', 'no', 'te', 'lo', 'le', 'da', 'su', 'por']
    spanish_score = sum(1 for word in spanish_words if word in text_lower)
    
    # French indicators
    french_words = ['le', 'de', 'et', 'à', 'un', 'il', 'être', 'et', 'en', 'avoir', 'que', 'pour', 'dans', 'ce', 'son', 'une']
    french_score = sum(1 for word in french_words if word in text_lower)
    
    scores = {
        'de': german_score,
        'en': english_score,
        'es': spanish_score,
        'fr': french_score
    }
    
    return max(scores, key=scores.get)

def smart_search(query, kb_data, max_results=5):
    """Enhanced search with language detection and better scoring."""
    print(f"\n🔍 Searching for: '{query}'")
    
    # Detect query language
    query_lang = detect_language_simple(query)
    print(f"🌍 Detected language: {query_lang}")
    
    query_words = query.lower().split()
    matches = []
    
    for entry in kb_data['entries']:
        score = 0
        
        # Language boost - prefer same language as query
        if entry.get('language', '').lower() == query_lang.lower():
            score += 5
        
        # Search in title (highest weight)
        if 'title' in entry:
            title_words = entry['title'].lower().split()
            for word in query_words:
                if any(word in title_word for title_word in title_words):
                    score += 4
                if word in entry['title'].lower():
                    score += 2
        
        # Search in content (medium weight)
        if 'content' in entry:
            content_words = entry['content'].lower().split()
            for word in query_words:
                if any(word in content_word for content_word in content_words):
                    score += 3
                if word in entry['content'].lower():
                    score += 1
        
        # Search in answer (highest weight - this is the solution)
        if 'answer' in entry:
            answer_words = entry['answer'].lower().split()
            for word in query_words:
                if any(word in answer_word for answer_word in answer_words):
                    score += 5
                if word in entry['answer'].lower():
                    score += 2
        
        # Priority boost
        if entry.get('priority', '').lower() in ['high', 'urgent', 'critical']:
            score += 1
        
        if score > 0:
            matches.append((score, entry))
    
    # Sort by score
    matches.sort(key=lambda x: x[0], reverse=True)
    
    print(f"📊 Found {len(matches)} matches")
    print("-" * 60)
    
    # Display results
    for i, (score, entry) in enumerate(matches[:max_results]):
        print(f"🎯 Result {i+1} (Score: {score} | Language: {entry.get('language', 'unknown')})")
        print(f"📁 Source: {entry['source_file']}")
        
        if 'title' in entry:
            title = entry['title'][:100] + "..." if len(entry['title']) > 100 else entry['title']
            print(f"📝 Title: {title}")
        
        if 'content' in entry:
            content = entry['content'][:200] + "..." if len(entry['content']) > 200 else entry['content']
            print(f"📄 Issue: {content}")
        
        if 'answer' in entry:
            answer = entry['answer'][:200] + "..." if len(entry['answer']) > 200 else entry['answer']
            print(f"✅ Solution: {answer}")
        
        if entry.get('priority'):
            print(f"⚡ Priority: {entry['priority']}")
        
        print("-" * 60)
    
    if not matches:
        print("❌ No matches found.")
        print("💡 Try different keywords or check spelling.")
    
    return matches[:max_results]

def interactive_demo():
    """Interactive demo of the system."""
    print("🤖 NexaCorp AI Support System - Interactive Demo")
    print("=" * 60)
    
    # Load knowledge base
    kb_data = load_knowledge_base()
    if not kb_data:
        return
    
    print(f"📚 Knowledge Base loaded: {kb_data['total_entries']} entries")
    print(f"🌍 Languages: {', '.join(kb_data['languages'])}")
    print(f"📁 Source files: {len(kb_data['source_files'])}")
    print()
    
    # Show language distribution
    lang_counts = {}
    for entry in kb_data['entries']:
        lang = entry['language']
        lang_counts[lang] = lang_counts.get(lang, 0) + 1
    
    print("📊 Language Distribution:")
    for lang, count in sorted(lang_counts.items()):
        percentage = (count / kb_data['total_entries']) * 100
        print(f"   {lang.upper()}: {count:,} entries ({percentage:.1f}%)")
    print()
    
    # Predefined demo queries
    demo_queries = [
        "Email synchronization problem",
        "VPN Verbindungsproblem",  # German
        "problema de acceso a cuenta",  # Spanish
        "problème de mot de passe",  # French
        "Shopify payment integration issue",
        "urgent security breach",
        "database connection error",
        "user authentication failure"
    ]
    
    print("🎯 Demo Queries (press Enter to run each, or type 'custom' for your own):")
    for i, query in enumerate(demo_queries, 1):
        print(f"   {i}. {query}")
    print()
    
    while True:
        choice = input("Enter query number (1-8), 'custom', or 'quit': ").strip()
        
        if choice.lower() in ['quit', 'exit', 'q']:
            break
        elif choice.lower() in ['custom', 'c']:
            query = input("Enter your query: ").strip()
            if query:
                smart_search(query, kb_data)
        elif choice.isdigit() and 1 <= int(choice) <= len(demo_queries):
            query = demo_queries[int(choice) - 1]
            smart_search(query, kb_data)
        else:
            print("Invalid choice. Try again.")
        
        print("\n" + "="*60 + "\n")

def benchmark_search():
    """Benchmark search performance."""
    print("⚡ Search Performance Benchmark")
    print("=" * 40)
    
    kb_data = load_knowledge_base()
    if not kb_data:
        return
    
    import time
    
    test_queries = [
        "email not working",
        "password reset",
        "VPN connection failed",
        "account locked",
        "payment processing error"
    ]
    
    total_time = 0
    total_results = 0
    
    for query in test_queries:
        start_time = time.time()
        results = smart_search(query, kb_data, max_results=10)
        end_time = time.time()
        
        search_time = end_time - start_time
        total_time += search_time
        total_results += len(results)
        
        print(f"Query: '{query}' - {len(results)} results in {search_time:.3f}s")
    
    avg_time = total_time / len(test_queries)
    avg_results = total_results / len(test_queries)
    
    print(f"\n📈 Performance Summary:")
    print(f"   Average search time: {avg_time:.3f} seconds")
    print(f"   Average results per query: {avg_results:.1f}")
    print(f"   Total knowledge base size: {kb_data['total_entries']:,} entries")
    print(f"   Search throughput: {kb_data['total_entries']/avg_time:.0f} entries/second")

def main():
    """Main function."""
    print("🚀 Welcome to NexaCorp AI Support System Test")
    print("=" * 50)
    
    while True:
        print("\nChoose an option:")
        print("1. Interactive Search Demo")
        print("2. Performance Benchmark")
        print("3. Knowledge Base Statistics")
        print("4. Exit")
        
        choice = input("\nEnter your choice (1-4): ").strip()
        
        if choice == '1':
            interactive_demo()
        elif choice == '2':
            benchmark_search()
        elif choice == '3':
            kb_data = load_knowledge_base()
            if kb_data:
                print(f"\n📊 Knowledge Base Statistics:")
                print(f"   Total entries: {kb_data['total_entries']:,}")
                print(f"   Languages: {', '.join(kb_data['languages'])}")
                print(f"   Created: {kb_data['created_at']}")
        elif choice == '4':
            print("👋 Goodbye!")
            break
        else:
            print("❌ Invalid choice. Please try again.")

if __name__ == "__main__":
    main()