#!/usr/bin/env python3
"""
Chatbot Runner - Simple Command Line Interface
Provides one-command execution for chatbot operations using chatbot_pipeline.py

Usage:
    python chatbot_runner.py run_indexing_pipeline
    python chatbot_runner.py query "What do visitors from Australia say?"
    python chatbot_runner.py launch_ui
"""

import sys
import os
import subprocess
import argparse
from typing import Optional

# Add current directory and parent directories to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'insights_extractor'))

from chatbot_config import ChatbotConfig
from chatbot_pipeline import ChatbotPipeline

def run_indexing_pipeline(batch_size: Optional[int] = None):
    """Execute the complete indexing pipeline from CSV to database"""
    if batch_size:
        print(f"🚀 Running Complete Indexing Pipeline with Batch Processing")
        print(f"📦 Batch size: {batch_size} records per batch")
    else:
        print("🚀 Running Complete Indexing Pipeline")
    print("=" * 50)
    
    try:
        # Initialize pipeline
        chatbot = ChatbotPipeline(
            ChatbotConfig.OPENAI_API_KEY,
            ChatbotConfig.DATABASE_URL,
            ChatbotConfig.INPUT_CSV_PATH
        )
        
        # Run indexing pipeline with optional batch size
        results = chatbot.run_indexing_pipeline(batch_size=batch_size)
        
        print("\n🎉 Indexing Pipeline Completed Successfully!")
        print(f"📊 Records processed: {results['inserted_count']}")
        if batch_size and 'total_batches' in results:
            print(f"📦 Batches processed: {results['total_batches']}")
            print(f"📊 Batch size: {results['batch_size']}")
        print(f"💰 Total cost: ${results['total_cost']:.4f}")
        print("✅ Ready for queries!")
        
    except Exception as e:
        print(f"❌ Indexing pipeline failed: {e}")
        sys.exit(1)

def run_indexing_pipeline_batch(batch_size: int = 500):
    """Execute the complete indexing pipeline with batch processing (default 500 records per batch)"""
    print(f"🚀 Running Complete Indexing Pipeline with Batch Processing")
    print(f"📦 Batch size: {batch_size} records per batch")
    print("=" * 50)
    
    try:
        # Initialize pipeline
        chatbot = ChatbotPipeline(
            ChatbotConfig.OPENAI_API_KEY,
            ChatbotConfig.DATABASE_URL,
            ChatbotConfig.INPUT_CSV_PATH
        )
        
        # Run indexing pipeline with batch processing
        results = chatbot.run_indexing_pipeline(batch_size=batch_size)
        
        print("\n🎉 Batch Indexing Pipeline Completed Successfully!")
        print(f"📊 Total records processed: {results['inserted_count']}")
        print(f"📦 Total batches processed: {results['total_batches']}")
        print(f"📊 Records per batch: {results['batch_size']}")
        print(f"💰 Total cost: ${results['total_cost']:.4f}")
        print("✅ Ready for queries!")
        
    except Exception as e:
        print(f"❌ Batch indexing pipeline failed: {e}")
        sys.exit(1)

def query_reviews(question: str):
    """Query reviews with a natural language question"""
    print(f"🔍 Querying: '{question}'")
    print("=" * 50)
    
    try:
        # Initialize pipeline
        chatbot = ChatbotPipeline(
            ChatbotConfig.OPENAI_API_KEY,
            ChatbotConfig.DATABASE_URL,
            ChatbotConfig.INPUT_CSV_PATH
        )
        
        # Query reviews
        result = chatbot.query_reviews(question)
        
        print("\n📊 Query Results:")
        print(f"   Filters applied: {result.get('filters', {})}")
        print(f"   Reviews found: {result.get('total_filtered', 0)}")
        print(f"   Similar reviews: {result.get('top_similar', 0)}")
        print(f"   Query cost: ${result.get('query_cost', 0):.4f}")
        
        print(f"\n💬 Summary:")
        print(result['summary'])
        
        if result.get('similar_reviews'):
            print(f"\n📄 Top Similar Reviews:")
            for i, review in enumerate(result['similar_reviews'][:3], 1):
                print(f"   {i}. Rating: {review['rating']}/5, {review['season']} {review['year']}, {review['reviewer_location']}")
                print(f"      Similarity: {review.get('similarity_score', 0):.3f}")
                print(f"      Text: {review['review_text'][:100]}...")
                print()
        
    except Exception as e:
        print(f"❌ Query failed: {e}")
        sys.exit(1)

def launch_ui():
    """Launch the Streamlit chatbot UI"""
    print("🌐 Launching Chatbot UI")
    print("=" * 30)
    
    try:
        # Check if streamlit is available
        try:
            import streamlit
        except ImportError:
            print("📦 Installing Streamlit...")
            subprocess.run([sys.executable, "-m", "pip", "install", "streamlit"], check=True)
        
        # Launch Streamlit
        print("🚀 Starting Streamlit server...")
        print("📱 Opening browser at http://localhost:8501")
        print("⌨️ Press Ctrl+C to stop the server")
        
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", 
            "chatbot_streamlit.py", 
            "--server.headless", "false"
        ])
        
    except FileNotFoundError:
        print("❌ chatbot_streamlit.py not found")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Failed to launch UI: {e}")
        sys.exit(1)

def main():
    """Main command line interface"""
    parser = argparse.ArgumentParser(
        description="Chatbot Runner - Simple CLI for Disney Review Chatbot",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python chatbot_runner.py run_indexing_pipeline
    python chatbot_runner.py run_indexing_pipeline --batch-size 500
    python chatbot_runner.py run_indexing_pipeline_batch
    python chatbot_runner.py run_indexing_pipeline_batch --batch-size 1000
    python chatbot_runner.py query "What do visitors from Australia say?"
    python chatbot_runner.py launch_ui
        """
    )
    
    parser.add_argument(
        'command',
        choices=['run_indexing_pipeline', 'run_indexing_pipeline_batch', 'query', 'launch_ui'],
        help='Command to execute'
    )
    
    parser.add_argument(
        'question',
        nargs='?',
        help='Question to ask (required for query command)'
    )
    
    parser.add_argument(
        '--batch-size',
        type=int,
        default=None,
        help='Batch size for processing (e.g., 500). If not specified, processes all data at once.'
    )
    
    # Show help if no arguments
    if len(sys.argv) == 1:
        parser.print_help()
        print("\n🎯 Quick Start:")
        print("1. python chatbot_runner.py run_indexing_pipeline")
        print("2. python chatbot_runner.py run_indexing_pipeline_batch --batch-size 500")
        print("3. python chatbot_runner.py query \"What do visitors think about rides?\"")
        print("4. python chatbot_runner.py launch_ui")
        print("\n📦 Batch Processing:")
        print("• Use run_indexing_pipeline_batch for automatic batch processing (default 500 records)")
        print("• Use --batch-size to customize batch size (e.g., --batch-size 1000)")
        print("• Batch processing is recommended for large datasets (>1000 records)")
        sys.exit(0)
    
    args = parser.parse_args()
    
    # Validate environment
    if not ChatbotConfig.OPENAI_API_KEY:
        print("❌ OPENAI_API_KEY not found!")
        print("💡 Set it with: export OPENAI_API_KEY='your-key-here'")
        sys.exit(1)
    
    if not os.path.exists(ChatbotConfig.INPUT_CSV_PATH):
        print(f"❌ Input CSV not found: {ChatbotConfig.INPUT_CSV_PATH}")
        sys.exit(1)
    
    # Execute command
    if args.command == 'run_indexing_pipeline':
        batch_size = args.batch_size
        if batch_size:
            print(f"🔧 Using custom batch size: {batch_size}")
        run_indexing_pipeline(batch_size=batch_size)
    
    elif args.command == 'run_indexing_pipeline_batch':
        batch_size = args.batch_size if args.batch_size else 500
        print(f"🔧 Using batch size: {batch_size}")
        run_indexing_pipeline_batch(batch_size=batch_size)
    
    elif args.command == 'query':
        if not args.question:
            print("❌ Question required for query command")
            print("💡 Usage: python chatbot_runner.py query \"Your question here\"")
            sys.exit(1)
        query_reviews(args.question)
    
    elif args.command == 'launch_ui':
        launch_ui()

if __name__ == "__main__":
    main() 