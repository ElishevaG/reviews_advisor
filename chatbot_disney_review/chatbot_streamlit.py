#!/usr/bin/env python3
"""
Streamlit Chatbot Interface for Disney Reviews
Provides natural language query interface for the indexed review data

This interface allows users to:
- Ask questions about Disney reviews in natural language
- Get AI-powered summaries of relevant reviews  
- See filter extraction and similarity search results
- Track query costs and performance
"""

import streamlit as st
import pandas as pd
import json
import os
import sys
from datetime import datetime
from typing import Dict, Any
from sqlalchemy import text

# Add paths for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'insights_extractor'))

# Import chatbot modules
from chatbot_config import ChatbotConfig
from chatbot_pipeline import ChatbotPipeline

# Page configuration
st.set_page_config(
    page_title="Disney Review Chatbot",
    page_icon="🏰",
    layout="wide"
)

# Initialize session state
if 'pipeline' not in st.session_state:
    st.session_state.pipeline = None
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'total_cost' not in st.session_state:
    st.session_state.total_cost = 0.0

def initialize_pipeline():
    """Initialize the chatbot pipeline"""
    try:
        pipeline = ChatbotPipeline(
            ChatbotConfig.OPENAI_API_KEY,
            ChatbotConfig.DATABASE_URL,
            ChatbotConfig.INPUT_CSV_PATH
        )
        return pipeline, None
    except Exception as e:
        return None, str(e)

def display_query_result(result: Dict[str, Any]):
    """Display query result in a formatted way"""
    
    # Main summary
    st.markdown("### 💬 Answer")
    st.markdown(result['summary'])
    
    # Query details in expander
    with st.expander("🔍 Query Details"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**📊 Extracted Filters:**")
            if result.get('filters'):
                for key, value in result['filters'].items():
                    if value is not None:
                        st.write(f"- {key}: {value}")
            else:
                st.write("No specific filters detected")
        
        with col2:
            st.markdown("**📈 Search Results:**")
            st.write(f"- Filtered reviews: {result.get('total_filtered', 0)}")
            st.write(f"- Similar reviews found: {result.get('top_similar', 0)}")
            st.write(f"- Query cost: ${result.get('query_cost', 0):.4f}")
    
    # Similar reviews in expander
    if result.get('similar_reviews'):
        with st.expander(f"📄 Top {len(result['similar_reviews'])} Similar Reviews"):
            for i, review in enumerate(result['similar_reviews'][:5], 1):
                st.markdown(f"**Review {i}** (Rating: {review['rating']}/5, {review['season']} {review['year']}, {review['reviewer_location']})")
                st.write(f"*Similarity: {review.get('similarity_score', 0):.3f}*")
                st.write(review['review_text'][:300] + "..." if len(review['review_text']) > 300 else review['review_text'])
                st.divider()

def main():
    """Main Streamlit application"""
    
    # Header
    st.title("🏰 Disney Hong Kong Review Chatbot")
    st.markdown("Ask questions about visitor experiences in natural language!")
    
    # Sidebar with information
    with st.sidebar:
        st.markdown("### ℹ️ About")
        st.markdown("This chatbot analyzes Disney visitor reviews using AI to provide insights about customer experiences.")
        
        st.markdown("### 📊 Dataset")
        st.write(f"- Source: First 50 reviews")
        st.write(f"- Database: PostgreSQL with vector search")
        st.write(f"- AI Models: GPT-4o-mini + text-embedding-3-small")
        
        st.markdown("### 💰 Usage")
        st.write(f"Total cost this session: ${st.session_state.total_cost:.4f}")
        
        if st.button("🔄 Reset Chat"):
            st.session_state.chat_history = []
            st.session_state.total_cost = 0.0
            st.rerun()
    
    # Initialize pipeline if not already done
    if st.session_state.pipeline is None:
        with st.spinner("🚀 Initializing chatbot pipeline..."):
            pipeline, error = initialize_pipeline()
            if pipeline:
                st.session_state.pipeline = pipeline
                st.success("✅ Chatbot pipeline initialized successfully!")
            else:
                st.error(f"❌ Failed to initialize pipeline: {error}")
                st.stop()
    
    # Check if data is indexed
    try:
        # Quick check if data exists in database
        with st.session_state.pipeline.engine.connect() as conn:
            result = conn.execute(text("SELECT COUNT(*) FROM disney_review"))
            row_count = result.scalar()
        
        if row_count == 0:
            st.warning("⚠️ No review data found in database. Please run the indexing pipeline first using `chatbot_runner.py`")
            st.markdown("### 🔧 Setup Instructions:")
            st.code("""
# Run the step-by-step pipeline to index reviews:
python -i chatbot_runner.py run_indexing_pipeline
            """)
            st.stop()
        else:
            st.info(f"📊 Database ready with {row_count} indexed reviews")
            
    except Exception as e:
        st.error(f"❌ Database connection error: {e}")
        st.stop()
    
    # Main chat interface
    st.markdown("### 💬 Ask about Disney experiences:")
    
    # Sample questions
    st.markdown("**💡 Try these example questions:**")
    example_questions = [
        "What do visitors from Australia say about the park?",
        "How were the experiences in 2019?",
        "What are the main complaints about wait times?",
        "What do people say about the food at the park?",
        "How do visitors rate their Spring visits?"
    ]
    
    cols = st.columns(3)
    for i, question in enumerate(example_questions):
        with cols[i % 3]:
            if st.button(question, key=f"example_{i}"):
                st.session_state.current_question = question
    
    # Question input
    question = st.text_input(
        "Your question:",
        value=getattr(st.session_state, 'current_question', ''),
        placeholder="e.g., What do visitors say about the rides?",
        key="question_input"
    )
    
    # Process question
    if st.button("🔍 Ask", type="primary") and question:
        with st.spinner("🤖 Analyzing reviews and generating response..."):
            try:
                # Query the chatbot
                result = st.session_state.pipeline.query_reviews(question)
                st.session_state.total_cost += st.session_state.pipeline.total_cost
                
                # Add to chat history
                st.session_state.chat_history.append({
                    'question': question,
                    'result': result,
                    'timestamp': datetime.now().strftime("%H:%M:%S")
                })
                
                # Display result
                display_query_result(result)
                
            except Exception as e:
                st.error(f"❌ Error processing question: {e}")
    
    # Chat history
    if st.session_state.chat_history:
        st.markdown("### 📜 Chat History")
        
        for i, chat in enumerate(reversed(st.session_state.chat_history)):
            with st.expander(f"Q{len(st.session_state.chat_history)-i}: {chat['question'][:50]}... ({chat['timestamp']})"):
                st.markdown(f"**Question:** {chat['question']}")
                st.markdown(f"**Answer:** {chat['result']['summary']}")
                
                if chat['result'].get('filters'):
                    st.markdown(f"**Filters:** {chat['result']['filters']}")
                st.markdown(f"**Results:** {chat['result'].get('total_filtered', 0)} filtered, {chat['result'].get('top_similar', 0)} similar")

if __name__ == "__main__":
    main() 