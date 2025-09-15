"""
Chatbot Configuration - Reuses existing pipeline components for shared functionality
Based on technical_design_document_chatboat.md specifications with PostgreSQL integration

This configuration file sets up the chatbot project while reusing:
- Database initialization and schema creation from pipeline.py
- Date feature engineering from pipeline.py  
- Vectorization and normalization from pipeline.py
"""

import os
from dotenv import load_dotenv
from typing import Dict, List, Any
from dataclasses import dataclass

# Load environment variables
load_dotenv(override=True)

@dataclass 
class ChatbotPipelineNode:
    """Single definition of a chatbot pipeline node"""
    id: int
    name: str
    function_name: str
    technology: str
    purpose: str
    description: str
    reuses_pipeline: bool = False  # Flag to indicate if it reuses existing pipeline function

class ChatbotConfig:
    """Configuration for the Disney Review Chatbot supporting multiple Disney parks: 
    Disneyland_Paris, Disneyland_HongKong, and Disneyland_California"""
    
    # ================================
    # 🗄️ DATABASE CONFIGURATION
    # ================================
    DATABASE_NAME = 'disneyland_reviews'
    DATABASE_USER = 'eghezail'
    DATABASE_PASSWORD = os.getenv('DATABASE_PASSWORD', '')
    DATABASE_HOST = os.getenv('DATABASE_HOST', 'localhost')
    DATABASE_PORT = os.getenv('DATABASE_PORT', '5432')
    DATABASE_URL = f"postgresql://{DATABASE_USER}:{DATABASE_PASSWORD}@{DATABASE_HOST}:{DATABASE_PORT}/{DATABASE_NAME}"
    DISNEY_REVIEW_TABLE = "disney_review"
    
    # ================================
    # 🤖 MODEL CONFIGURATION  
    # ================================
    OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
    LLM_MODEL = "gpt-4o-mini"
    EMBEDDING_MODEL = "text-embedding-3-small"
    EMBEDDING_DIMENSIONS = 512
    LLM_TEMPERATURE = 0.0
    
    # Validate API key on load
    if not OPENAI_API_KEY:
        print("⚠️ OPENAI_API_KEY not found in environment. Make sure to set it before running.")
    else:
        print(f"✅ OPENAI_API_KEY loaded: {OPENAI_API_KEY[:10]}...")
    
    # ================================
    # 📁 FILE PATHS
    # ================================
    INPUT_CSV_PATH = "/Users/eghezail/Desktop/LLM_Model/data/DisneylandReviews.csv"  # Test with 50 reviews
    OUTPUT_DIR = "chatbot_outputs"
    METADATA_FILE = "chatbot_metadata.json"
    
    # ================================
    # 🏗️ CHATBOT PIPELINE DEFINITION
    # ================================
    CHATBOT_PIPELINE_NODES = [
        ChatbotPipelineNode(
            id=0,
            name="Database Initialization",
            function_name="init_database",
            technology="PostgreSQL + pgvector",
            purpose="Initialize database connection and create schema for disney_review table",
            description="Reuse pipeline database initialization but create new disney_review schema",
            reuses_pipeline=True
        ),
        ChatbotPipelineNode(
            id=1,
            name="Date Feature Engineering", 
            function_name="extract_temporal_features",
            technology="pandas",
            purpose="Extract year, month, and season from Year_Month column",
            description="Reuse existing date feature engineering from pipeline.py",
            reuses_pipeline=True
        ),
        ChatbotPipelineNode(
            id=2,
            name="Metadata Enumeration",
            function_name="extract_metadata_values",
            technology="pandas",
            purpose="Extract unique values for filtering: year, month, season, reviewer_location",
            description="Create metadata file with all unique filter values for LLM parsing"
        ),
        ChatbotPipelineNode(
            id=3,
            name="Text Vectorization",
            function_name="vectorize_reviews", 
            technology="OpenAI text-embedding-3-small",
            purpose="Convert full review text to 512-dimensional embeddings",
            description="Reuse vectorization function from pipeline.py for full reviews",
            reuses_pipeline=True
        ),

        ChatbotPipelineNode(
            id=4,
            name="Database Write",
            function_name="store_disney_reviews",
            technology="psycopg2/SQLAlchemy",
            purpose="Store disney_review records with embeddings",
            description="Store complete reviews with metadata and vectors for chatbot search"
        ),
        ChatbotPipelineNode(
            id=5,
            name="Filter Parsing",
            function_name="parse_user_filters",
            technology="GPT-4o-mini via LangChain",
            purpose="Extract metadata filters from user questions",
            description="LLM-based parsing to extract year, location, season, etc. from natural language"
        ),
        ChatbotPipelineNode(
            id=6,
            name="Review Retrieval", 
            function_name="retrieve_filtered_reviews",
            technology="psycopg2/SQLAlchemy",
            purpose="Query reviews matching user filters",
            description="SQL-based filtering on metadata before similarity search"
        ),
        ChatbotPipelineNode(
            id=7,
            name="Similarity Search",
            function_name="find_similar_reviews",
            technology="numpy dot product",
            purpose="Top-10 cosine similarity matching",
            description="Compute cosine similarity using OpenAI embeddings (already normalized)"
        ),
        ChatbotPipelineNode(
            id=8,
            name="Summarization",
            function_name="summarize_reviews",
            technology="GPT-4o-mini via LangChain", 
            purpose="Summarize most relevant reviews for user",
            description="Generate natural language summary of top matching reviews"
        ),
        ChatbotPipelineNode(
            id=10,
            name="Chat Interface",
            function_name="streamlit_interface",
            technology="Streamlit",
            purpose="Provide natural language query interface",
            description="Interactive UI for chatbot queries and responses"
        )
    ]
    
    # ================================
    # 🗄️ DATABASE SCHEMA FOR CHATBOT
    # ================================
    CHATBOT_DATABASE_SCHEMA = {
        'table_creation_sql': """
        CREATE EXTENSION IF NOT EXISTS vector;
        CREATE TABLE IF NOT EXISTS disney_review (
            review_id VARCHAR(255) PRIMARY KEY,
            review_text TEXT NOT NULL,
            rating INTEGER NOT NULL,
            year INTEGER NOT NULL,
            month INTEGER NOT NULL,
            season VARCHAR(20),
            branch VARCHAR(255),
            reviewer_location VARCHAR(255),
            embedding_vector VECTOR(512),
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        """,
        'indexes_sql': [
            "CREATE INDEX IF NOT EXISTS idx_disney_review_year ON disney_review(year);",
            "CREATE INDEX IF NOT EXISTS idx_disney_review_month ON disney_review(month);", 
            "CREATE INDEX IF NOT EXISTS idx_disney_review_season ON disney_review(season);",
            "CREATE INDEX IF NOT EXISTS idx_disney_review_location ON disney_review(reviewer_location);",
            "CREATE INDEX IF NOT EXISTS idx_disney_review_rating ON disney_review(rating);",
            "CREATE INDEX IF NOT EXISTS idx_disney_review_branch ON disney_review(branch);",
            "CREATE INDEX IF NOT EXISTS idx_disney_review_embedding_hnsw ON disney_review USING hnsw (embedding_vector vector_cosine_ops);"
        ]
    }
    
    # ================================
    # 📋 PIPELINE INFO METHODS
    # ================================
    @classmethod
    def get_pipeline_info(cls) -> Dict[str, Any]:
        """Get chatbot pipeline information"""
        return {
            'name': 'Disneyland Review Chatbot Pipeline',
            'description': 'Natural language chatbot for analyzing customer reviews with vector similarity search',
            'nodes': [node.name for node in cls.CHATBOT_PIPELINE_NODES],
            'technologies': {node.name: node.technology for node in cls.CHATBOT_PIPELINE_NODES},
            'reused_functions': [node.name for node in cls.CHATBOT_PIPELINE_NODES if node.reuses_pipeline],
            'total_nodes': len(cls.CHATBOT_PIPELINE_NODES),
            'database_config': {
                'name': cls.DATABASE_NAME,
                'user': cls.DATABASE_USER,
                'table': 'disney_review'
            },
            'model_config': {
                'llm': cls.LLM_MODEL,
                'embedding': cls.EMBEDDING_MODEL, 
                'dimensions': cls.EMBEDDING_DIMENSIONS
            }
        }
    
    @classmethod
    def get_reused_nodes(cls) -> List[ChatbotPipelineNode]:
        """Get nodes that reuse existing pipeline functions"""
        return [node for node in cls.CHATBOT_PIPELINE_NODES if node.reuses_pipeline]
    
    @classmethod
    def get_new_nodes(cls) -> List[ChatbotPipelineNode]:
        """Get nodes that are new for the chatbot"""
        return [node for node in cls.CHATBOT_PIPELINE_NODES if not node.reuses_pipeline] 