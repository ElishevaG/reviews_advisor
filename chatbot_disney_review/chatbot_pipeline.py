#!/usr/bin/env python3
"""
Disneyland Review Chatbot Pipeline
Implementation based on technical_design_document_chatboat.md with PostgreSQL integration

This pipeline reuses components from the existing pipeline.py for:
- Database initialization and schema creation
- Date feature engineering  
- Text vectorization and normalization

New components specific to chatbot:
- Metadata enumeration for filters
- Filter parsing from natural language
- Similarity search and review retrieval
- Summarization for chat responses
"""

import pandas as pd
import numpy as np
import os
import json
import logging
from typing import List, Dict, Tuple, Any, Optional
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Core libraries

import matplotlib.pyplot as plt
import seaborn as sns

# Database libraries
import psycopg2
from psycopg2.extras import RealDictCursor
import sqlalchemy
from sqlalchemy import create_engine, text, MetaData
from sqlalchemy.pool import QueuePool
from sqlalchemy.sql import text as sql_text

# LangChain imports
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.schema import HumanMessage, SystemMessage
from langchain.callbacks import get_openai_callback
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

# Import existing pipeline components for reuse
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'insights_extractor'))
from pipeline import ReviewAnalysisPipeline
from chatbot_config import ChatbotConfig

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ChatbotPipeline:
    """
    Chatbot pipeline that reuses existing components from ReviewAnalysisPipeline
    and adds new chatbot-specific functionality
    """
    
    def __init__(self, openai_api_key: str, database_url: str, input_csv_path: str):
        """
        Initialize the chatbot pipeline
        
        Args:
            openai_api_key: OpenAI API key for LangChain
            database_url: PostgreSQL connection URL  
            input_csv_path: Path to input CSV file
        """
        self.openai_api_key = openai_api_key
        self.database_url = database_url
        self.input_csv_path = input_csv_path
        
        # Initialize database connection (reused from pipeline)
        self._init_database()
        
        # Initialize LangChain components
        os.environ["OPENAI_API_KEY"] = openai_api_key
        self.llm = ChatOpenAI(model_name=ChatbotConfig.LLM_MODEL, temperature=ChatbotConfig.LLM_TEMPERATURE)
        self.embeddings = OpenAIEmbeddings(model=ChatbotConfig.EMBEDDING_MODEL, dimensions=ChatbotConfig.EMBEDDING_DIMENSIONS)
        
        # Data storage
        self.df = None
        self.metadata = {}
        self.total_cost = 0.0
        
        # Load existing metadata if available
        self._load_metadata()
        
        logger.info("✅ Chatbot Pipeline initialized successfully!")
    
    def _load_metadata(self):
        """Load metadata from saved file if it exists"""
        try:
            metadata_path = os.path.join(ChatbotConfig.OUTPUT_DIR, ChatbotConfig.METADATA_FILE)
            if os.path.exists(metadata_path):
                with open(metadata_path, 'r') as f:
                    self.metadata = json.load(f)
                logger.info(f"✅ Loaded metadata with {len(self.metadata)} filter types")
            else:
                logger.warning("⚠️ No metadata file found. Run indexing pipeline first.")
        except Exception as e:
            logger.warning(f"⚠️ Could not load metadata: {e}")
            self.metadata = {}
    
    def _init_database(self):
        """Initialize PostgreSQL database connection (REUSED from pipeline.py)"""
        try:
            # Create SQLAlchemy engine with connection pooling
            self.engine = create_engine(
                self.database_url,
                poolclass=QueuePool,
                pool_size=5,
                max_overflow=10,
                pool_pre_ping=True
            )
            
            # Test connection
            with self.engine.connect() as conn:
                result = conn.execute(text("SELECT 1"))
                conn.commit()
            
            # Create chatbot schema
            self._create_chatbot_database_schema()
            
            logger.info("✅ Database connection established and chatbot schema ready")
            
        except Exception as e:
            logger.error(f"❌ Database initialization failed: {e}")
            raise
    
    def _create_chatbot_database_schema(self):
        """Create database schema for disney_review table"""
        try:
            # Check if vector extension is available
            vector_available = False
            try:
                with self.engine.connect() as conn:
                    conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
                    conn.commit()
                vector_available = True
                logger.info("✅ Vector extension enabled")
            except Exception as e:
                logger.warning(f"⚠️ Could not create vector extension: {e}")
                logger.info("📝 Proceeding with TEXT-based embedding storage")
            
            # Create table and indexes in separate transaction
            with self.engine.connect() as conn:
                # Create table with appropriate column types
                if vector_available:
                    table_sql = ChatbotConfig.CHATBOT_DATABASE_SCHEMA['table_creation_sql']
                else:
                    # Replace VECTOR(512) with TEXT for compatibility
                    table_sql = ChatbotConfig.CHATBOT_DATABASE_SCHEMA['table_creation_sql'].replace('VECTOR(512)', 'TEXT')
                    # Remove vector extension creation
                    table_sql = table_sql.replace('CREATE EXTENSION IF NOT EXISTS vector;', '')
                
                conn.execute(text(table_sql))
                
                # Create indexes (skip vector indexes if not available)
                for index_sql in ChatbotConfig.CHATBOT_DATABASE_SCHEMA['indexes_sql']:
                    if 'hnsw' in index_sql and not vector_available:
                        logger.info(f"⏭️ Skipping vector index (extension not available)")
                        continue
                    conn.execute(text(index_sql))
                
                conn.commit()
                
            logger.info("✅ Chatbot database schema created successfully")
            
        except Exception as e:
            logger.error(f"❌ Error creating chatbot database schema: {e}")
            raise
    
    def load_data(self) -> pd.DataFrame:
        """
        Load and validate input CSV data (REUSED from pipeline.py with minor modifications)
        
        Returns:
            pd.DataFrame: Loaded and validated dataframe
        """
        logger.info("📊 Loading data for chatbot...")
        
        try:
            # Try UTF-8 first, then fall back to latin-1 for encoding issues
            try:
                self.df = pd.read_csv(self.input_csv_path)
            except UnicodeDecodeError:
                logger.warning("⚠️ UTF-8 decode failed, trying latin-1 encoding...")
                self.df = pd.read_csv(self.input_csv_path, encoding='latin-1')
            
            logger.info(f"✅ Loaded {len(self.df)} reviews for chatbot")
            
            # Validate required columns for chatbot
            required_cols = ['Review_ID', 'Rating', 'Year_Month', 'Reviewer_Location', 'Review_Text', 'Branch']
            missing_cols = [col for col in required_cols if col not in self.df.columns]
            
            if missing_cols:
                raise ValueError(f"Missing required columns: {missing_cols}")
            
            # Basic data cleaning
            self.df = self.df.dropna(subset=['Review_Text'])
            self.df['Review_Text'] = self.df['Review_Text'].astype(str)
            
            logger.info(f"✅ Data validation complete for chatbot. Final dataset: {len(self.df)} reviews")
            return self.df
            
        except Exception as e:
            logger.error(f"❌ Error loading data for chatbot: {e}")
            raise
    
    def extract_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract temporal features (REUSED from pipeline.py node_1_date_feature_engineering)
        
        Args:
            df: Input dataframe
            
        Returns:
            pd.DataFrame: DataFrame with temporal features
        """
        logger.info("🗓️ Step 1: Date Feature Engineering (REUSED from pipeline.py)")
        
        # Parse Year_Month with missing value handling
        df['Year_Month'] = df['Year_Month'].fillna('2020-01')  # Default to 2020-01 for missing
        df['Year_Month'] = df['Year_Month'].replace('missing', '2020-01')
        
        df['Year'] = df['Year_Month'].str.split('-').str[0].astype(int)
        df['Month'] = df['Year_Month'].str.split('-').str[1].astype(int)
        
        # Create Season column as specified in technicall_design_document.md
        def get_season(month):
            if month in [3, 4, 5]:
                return 'Spring'
            elif month in [6, 7, 8]:
                return 'Summer'
            elif month in [9, 10, 11]:
                return 'Autumn'
            else:  # 12, 1, 2
                return 'Winter'
        
        df['Season'] = df['Month'].apply(get_season)
        
        logger.info(f"✅ Added temporal features: Year, Month, Season")
        return df
    
    def extract_metadata_values(self, df: pd.DataFrame) -> Dict[str, List]:
        """
        Extract unique values for metadata filtering (NEW for chatbot)
        
        Args:
            df: DataFrame with temporal features
            
        Returns:
            Dict[str, List]: Dictionary of unique values for each metadata field
        """
        logger.info("📋 Step 2: Metadata Enumeration (NEW for chatbot)")
        
        metadata = {
            'years': sorted(df['Year'].unique().tolist()),
            'months': sorted(df['Month'].unique().tolist()),
            'seasons': sorted(df['Season'].unique().tolist()),
            'reviewer_locations': sorted(df['Reviewer_Location'].unique().tolist()),
            'ratings': sorted(df['Rating'].unique().tolist()),
            'branch': sorted(df['Branch'].unique().tolist())
        }
        
        self.metadata = metadata
        
        # Save metadata to file
        os.makedirs(ChatbotConfig.OUTPUT_DIR, exist_ok=True)
        metadata_path = os.path.join(ChatbotConfig.OUTPUT_DIR, ChatbotConfig.METADATA_FILE)
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"✅ Extracted metadata values:")
        logger.info(f"   Years: {metadata['years']}")
        logger.info(f"   Months: {metadata['months']}")
        logger.info(f"   Seasons: {metadata['seasons']}")
        logger.info(f"   Locations: {len(metadata['reviewer_locations'])} unique locations")
        logger.info(f"   Ratings: {metadata['ratings']}")
        logger.info(f"   Branches: {metadata['branch']}")
        logger.info(f"📁 Metadata saved to: {metadata_path}")
        
        return metadata
    
    def vectorize_reviews(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Convert full review text to vectors (REUSED from pipeline.py node_3_text_vectorization)
        
        Args:
            df: DataFrame with reviews
            
        Returns:
            pd.DataFrame: DataFrame with embedding columns
        """
        logger.info("🔢 Step 3: Text Vectorization (REUSED from pipeline.py)")
        
        # Get embeddings for all review texts
        review_texts = df['Review_Text'].tolist()
        
        logger.info(f"   Getting embeddings for {len(review_texts)} reviews...")
        
        try:
            with get_openai_callback() as cb:
                embeddings = self.embeddings.embed_documents(review_texts)
                self.total_cost += cb.total_cost
                
            logger.info(f"   Embedding cost: ${cb.total_cost:.4f}")
            
            # Convert to numpy array and add to dataframe
            embeddings_array = np.array(embeddings)
            
            # Add embedding columns
            for i in range(embeddings_array.shape[1]):
                df[f'embedding_{i}'] = embeddings_array[:, i]
            
            logger.info(f"✅ Added {embeddings_array.shape[1]} embedding dimensions")
            return df
            
        except Exception as e:
            logger.error(f"❌ Error in vectorization: {e}")
            raise
    

    
    def store_disney_reviews(self, df: pd.DataFrame) -> int:
        """
        Store disney_review records with embeddings (NEW for chatbot)
        
        Args:
            df: DataFrame with embedding columns
            
        Returns:
            int: Number of rows inserted
        """
        logger.info("💾 Step 4: Database Write - Storing disney_review records...")
        
        inserted_count = 0
        
        try:
            with self.engine.connect() as conn:
                for _, row in df.iterrows():
                    # Prepare embedding vector as array string for PostgreSQL
                    embedding_cols = [col for col in df.columns if col.startswith('embedding_')]
                    embedding_values = [row[col] for col in embedding_cols]
                    embedding_vector_str = '[' + ','.join(map(str, embedding_values)) + ']'
                    
                    # Insert individual row
                    insert_sql = """
                    INSERT INTO disney_review (
                        review_id, review_text, rating, year, month, season,
                        branch, reviewer_location, embedding_vector
                    ) VALUES (
                        :review_id, :review_text, :rating, :year, :month, :season,
                        :branch, :reviewer_location, :embedding_vector
                    ) ON CONFLICT (review_id) DO UPDATE SET
                        review_text = EXCLUDED.review_text,
                        rating = EXCLUDED.rating,
                        year = EXCLUDED.year,
                        month = EXCLUDED.month,
                        season = EXCLUDED.season,
                        branch = EXCLUDED.branch,
                        reviewer_location = EXCLUDED.reviewer_location,
                        embedding_vector = EXCLUDED.embedding_vector,
                        updated_at = CURRENT_TIMESTAMP
                    """
                    
                    conn.execute(text(insert_sql), {
                        'review_id': str(row['Review_ID']),
                        'review_text': str(row['Review_Text']),
                        'rating': int(row['Rating']),
                        'year': int(row['Year']),
                        'month': int(row['Month']),
                        'season': str(row['Season']),
                        'branch': str(row.get('Branch')),
                        'reviewer_location': str(row['Reviewer_Location']),
                        'embedding_vector': embedding_vector_str
                    })
                    
                    inserted_count += 1
                
                conn.commit()
                
            logger.info(f"✅ Database write completed: {inserted_count} disney_review records stored")
            return inserted_count
            
        except Exception as e:
            logger.error(f"❌ Database write failed: {e}")
            raise
    
    def run_indexing_pipeline(self, batch_size: Optional[int] = None) -> Dict[str, Any]:
        """
        Execute the complete indexing pipeline for chatbot (one-time setup)
        
        Args:
            batch_size: Optional batch size for processing. If None, processes all data at once.
                       If provided, processes data in batches of this size.
        
        Returns:
            Dict[str, Any]: Pipeline results
        """
        logger.info("🚀 Starting Chatbot Indexing Pipeline")
        if batch_size:
            logger.info(f"📦 Batch processing enabled: {batch_size} records per batch")
        logger.info("=" * 60)
        
        try:
            # Step 0: Load data
            df = self.load_data()
            
            # Step 1: Date Feature Engineering (REUSED)
            df = self.extract_temporal_features(df)
            
            # Step 2: Metadata Enumeration (NEW) 
            metadata = self.extract_metadata_values(df)
            
            if batch_size and len(df) >= batch_size:
                # Process in batches
                total_inserted = self.run_batch_processing(df, batch_size)
                
                logger.info("=" * 60)
                logger.info("🎉 Chatbot Batch Indexing Pipeline completed successfully!")
                logger.info(f"💰 Total OpenAI API cost: ${self.total_cost:.4f}")
                logger.info(f"💾 Data stored in PostgreSQL database: disney_review table")
                logger.info(f"📊 Total records processed: {len(df)} in {(len(df) + batch_size - 1) // batch_size} batches")
                logger.info(f"📁 Metadata saved for filter parsing")
                
                return {
                    'processed_data': df,
                    'metadata': metadata,
                    'inserted_count': total_inserted,
                    'total_cost': self.total_cost,
                    'batch_size': batch_size,
                    'total_batches': (len(df) + batch_size - 1) // batch_size
                }
            else:
                # Process all at once (original behavior)
                # Step 3: Text Vectorization (REUSED)
                df = self.vectorize_reviews(df)
                
                # Step 4: Database Write (NEW)
                inserted_count = self.store_disney_reviews(df)
                
                logger.info("=" * 60)
                logger.info("🎉 Chatbot Indexing Pipeline completed successfully!")
                logger.info(f"💰 Total OpenAI API cost: ${self.total_cost:.4f}")
                logger.info(f"💾 Data stored in PostgreSQL database: disney_review table")
                logger.info(f"📊 Records processed: {len(df)}")
                logger.info(f"📁 Metadata saved for filter parsing")
                
                return {
                    'processed_data': df,
                    'metadata': metadata,
                    'inserted_count': inserted_count,
                    'total_cost': self.total_cost
                }
            
        except Exception as e:
            logger.error(f"❌ Chatbot Indexing Pipeline failed: {e}")
            raise

    def run_batch_processing(self, df: pd.DataFrame, batch_size: int) -> int:
        """
        Process data in batches to handle large datasets efficiently
        
        Args:
            df: Complete DataFrame with temporal features
            batch_size: Number of records to process per batch
            
        Returns:
            int: Total number of records inserted
        """
        total_batches = (len(df) + batch_size - 1) // batch_size
        total_inserted = 0
        
        logger.info(f"🔄 Processing {len(df)} records in {total_batches} batches of {batch_size}")
        
        for batch_num in range(total_batches):
            start_idx = batch_num * batch_size
            end_idx = min(start_idx + batch_size, len(df))
            batch_df = df.iloc[start_idx:end_idx].copy()
            
            logger.info(f"📦 Processing batch {batch_num + 1}/{total_batches}: records {start_idx + 1}-{end_idx}")
            
            try:
                # Step 3: Text Vectorization for batch
                batch_df = self.vectorize_reviews_batch(batch_df, batch_num + 1, total_batches)
                
                # Step 4: Database Write for batch
                batch_inserted = self.store_disney_reviews_batch(batch_df, batch_num + 1, total_batches)
                
                total_inserted += batch_inserted
                
                logger.info(f"✅ Batch {batch_num + 1} completed: {batch_inserted} records inserted")
                logger.info(f"💰 Cumulative cost so far: ${self.total_cost:.4f}")
                
                # Clear batch data from memory
                del batch_df
                
            except Exception as e:
                logger.error(f"❌ Batch {batch_num + 1} failed: {e}")
                raise
        
        logger.info(f"🎉 All {total_batches} batches completed successfully!")
        return total_inserted

    def vectorize_reviews_batch(self, df: pd.DataFrame, batch_num: int, total_batches: int) -> pd.DataFrame:
        """
        Convert review text to vectors for a batch (optimized for memory and API calls)
        
        Args:
            df: DataFrame batch with reviews
            batch_num: Current batch number (for logging)
            total_batches: Total number of batches (for logging)
            
        Returns:
            pd.DataFrame: DataFrame with embedding columns
        """
        logger.info(f"🔢 Batch {batch_num}/{total_batches}: Text Vectorization")
        
        # Get embeddings for batch review texts
        review_texts = df['Review_Text'].tolist()
        
        logger.info(f"   Getting embeddings for {len(review_texts)} reviews in batch {batch_num}...")
        
        try:
            with get_openai_callback() as cb:
                embeddings = self.embeddings.embed_documents(review_texts)
                self.total_cost += cb.total_cost
                
            logger.info(f"   Batch {batch_num} embedding cost: ${cb.total_cost:.4f}")
            
            # Convert to numpy array and add to dataframe
            embeddings_array = np.array(embeddings)
            
            # Add embedding columns
            for i in range(embeddings_array.shape[1]):
                df[f'embedding_{i}'] = embeddings_array[:, i]
            
            logger.info(f"✅ Batch {batch_num}: Added {embeddings_array.shape[1]} embedding dimensions")
            return df
            
        except Exception as e:
            logger.error(f"❌ Error in batch {batch_num} vectorization: {e}")
            raise

    def store_disney_reviews_batch(self, df: pd.DataFrame, batch_num: int, total_batches: int) -> int:
        """
        Store disney_review records for a batch (optimized for database performance)
        
        Args:
            df: DataFrame batch with embeddings
            batch_num: Current batch number (for logging)
            total_batches: Total number of batches (for logging)
            
        Returns:
            int: Number of rows inserted in this batch
        """
        logger.info(f"💾 Batch {batch_num}/{total_batches}: Database Write")
        
        inserted_count = 0
        
        try:
            with self.engine.connect() as conn:
                # Use batch insert for better performance
                batch_data = []
                
                for _, row in df.iterrows():
                    # Prepare embedding vector as array string for PostgreSQL
                    embedding_cols = [col for col in df.columns if col.startswith('embedding_')]
                    embedding_values = [row[col] for col in embedding_cols]
                    embedding_vector_str = '[' + ','.join(map(str, embedding_values)) + ']'
                    
                    batch_data.append({
                        'review_id': str(row['Review_ID']),
                        'review_text': str(row['Review_Text']),
                        'rating': int(row['Rating']),
                        'year': int(row['Year']),
                        'month': int(row['Month']),
                        'season': str(row['Season']),
                        'branch': str(row.get('Branch')),
                        'reviewer_location': str(row['Reviewer_Location']),
                        'embedding_vector': embedding_vector_str
                    })
                
                # Batch insert all records for this batch
                insert_sql = """
                INSERT INTO disney_review (
                    review_id, review_text, rating, year, month, season,
                    branch, reviewer_location, embedding_vector
                ) VALUES (
                    :review_id, :review_text, :rating, :year, :month, :season,
                    :branch, :reviewer_location, :embedding_vector
                ) ON CONFLICT (review_id) DO UPDATE SET
                    review_text = EXCLUDED.review_text,
                    rating = EXCLUDED.rating,
                    year = EXCLUDED.year,
                    month = EXCLUDED.month,
                    season = EXCLUDED.season,
                    branch = EXCLUDED.branch,
                    reviewer_location = EXCLUDED.reviewer_location,
                    embedding_vector = EXCLUDED.embedding_vector,
                    updated_at = CURRENT_TIMESTAMP
                """
                
                # Execute batch insert
                conn.execute(text(insert_sql), batch_data)
                conn.commit()
                inserted_count = len(batch_data)
                
            logger.info(f"✅ Batch {batch_num} database write completed: {inserted_count} records stored")
            return inserted_count
            
        except Exception as e:
            logger.error(f"❌ Batch {batch_num} database write failed: {e}")
            raise
    
    # ================================
    # CHATBOT QUERY FUNCTIONS (NEW)
    # ================================
    
    def parse_user_filters(self, user_question: str) -> Dict[str, Any]:
        """
        Extract metadata filters from user question using LLM (NEW for chatbot)
        
        Args:
            user_question: Natural language question from user
            
        Returns:
            Dict[str, Any]: Extracted filters
        """
        logger.info(f"🔍 Parsing filters from question: '{user_question[:50]}...'")
        
        # Create filter parsing prompt
        filter_parsing_prompt = PromptTemplate(
            input_variables=["question", "years", "months", "seasons", "reviewer_locations", "ratings", "branches"],
            template="""You are a filter extraction system for a Disney review chatbot that covers multiple Disney parks.

Extract metadata filters from the user's question. Available filter values:

Years: {years}
Months: {months} (1=Jan, 2=Feb, etc.)
Seasons: {seasons}
Reviewer Locations: {reviewer_locations}
Ratings: {ratings}
Disney Branches: {branches}

Return ONLY a JSON object with extracted filters. Use null for unspecified filters.

Example outputs:
{{"year": 2019, "season": null, "rating": null, "reviewer_location": "Australia", "branch": "Disneyland_HongKong"}}
{{"year": null, "season": "Summer", "rating": [1,2,3], "reviewer_location": null, "branch": "Disneyland_Paris"}}
{{"year": 2018, "season": null, "rating": 5, "reviewer_location": "United States", "branch": "Disneyland_California"}}

User question: {question}"""
        )
        
        # Create LLM chain
        filter_parsing_chain = LLMChain(
            llm=self.llm,
            prompt=filter_parsing_prompt
        )
        
        try:
            with get_openai_callback() as cb:
                response = filter_parsing_chain.run({
                    "question": user_question,
                    "years": self.metadata.get('years', []),
                    "months": self.metadata.get('months', []),
                    "seasons": self.metadata.get('seasons', []),
                    "reviewer_locations": self.metadata.get('reviewer_locations', []),
                    "ratings": self.metadata.get('ratings', []),
                    "branches": self.metadata.get('branches', [])
                })
                self.total_cost += cb.total_cost
                
            # Parse JSON response
            try:
                filters = json.loads(response.strip())
                logger.info(f"✅ Extracted filters: {filters}")
                return filters
            except json.JSONDecodeError:
                logger.warning(f"⚠️ Could not parse filter JSON: {response}")
                return {}
                
        except Exception as e:
            logger.error(f"❌ Error parsing filters: {e}")
            return {}
    
    def retrieve_filtered_reviews(self, filters: Dict[str, Any]) -> pd.DataFrame:
        """
        Query reviews matching user filters (NEW for chatbot)
        
        Args:
            filters: Extracted metadata filters
            
        Returns:
            pd.DataFrame: Filtered reviews from database
        """
        logger.info("📊 Retrieving filtered reviews from database...")
        
        # Build SQL query with filters
        where_conditions = []
        params = {}
        
        if filters.get('year'):
            where_conditions.append("year = %(year)s")
            params['year'] = filters['year']
            
        if filters.get('season'):
            where_conditions.append("season = %(season)s")
            params['season'] = filters['season']
            
        if filters.get('reviewer_location'):
            where_conditions.append("reviewer_location = %(reviewer_location)s")
            params['reviewer_location'] = filters['reviewer_location']
            
        if filters.get('rating'):
            if isinstance(filters['rating'], list):
                where_conditions.append("rating = ANY(%(rating)s)")
                params['rating'] = filters['rating']
            else:
                where_conditions.append("rating = %(rating)s")
                params['rating'] = filters['rating']
        
        if filters.get('branch'):
            where_conditions.append("branch = %(branch)s")
            params['branch'] = filters['branch']
        
        # Base query
        query = """
        SELECT review_id, review_text, rating, year, month, season, 
               reviewer_location, branch, embedding_vector
        FROM disney_review
        """
        
        if where_conditions:
            query += " WHERE " + " AND ".join(where_conditions)
        
        query += " ORDER BY review_id"
        
        try:
            df = pd.read_sql(query, self.engine, params=params)
            logger.info(f"✅ Retrieved {len(df)} reviews matching filters")
            return df
            
        except Exception as e:
            logger.error(f"❌ Error retrieving filtered reviews: {e}")
            return pd.DataFrame()
    
    def find_similar_reviews(self, question: str, filtered_reviews: pd.DataFrame, top_k: int = 10) -> pd.DataFrame:
        """
        Find top-K most similar reviews using cosine similarity (NEW for chatbot)
        
        Args:
            question: User question text
            filtered_reviews: Pre-filtered reviews dataframe
            top_k: Number of top reviews to return
            
        Returns:
            pd.DataFrame: Top similar reviews with similarity scores
        """
        logger.info(f"🎯 Finding top-{top_k} similar reviews...")
        
        if len(filtered_reviews) == 0:
            logger.warning("⚠️ No filtered reviews to search")
            return pd.DataFrame()
        
        try:
            # Get question embedding
            with get_openai_callback() as cb:
                question_embedding = self.embeddings.embed_query(question)
                self.total_cost += cb.total_cost
            
            # Question embedding is already normalized from OpenAI
            question_embedding = np.array(question_embedding)
            
            # Extract review embeddings from database result (stored as strings)
            review_embeddings = []
            for _, row in filtered_reviews.iterrows():
                # Parse the embedding vector string from database
                embedding_str = row['embedding_vector']
                if embedding_str.startswith('[') and embedding_str.endswith(']'):
                    embedding_values = [float(x) for x in embedding_str[1:-1].split(',')]
                    review_embeddings.append(embedding_values)
                else:
                    logger.warning(f"⚠️ Could not parse embedding for review {row['review_id']}")
                    continue
            
            if not review_embeddings:
                logger.warning("⚠️ No valid embeddings found")
                return pd.DataFrame()
            
            review_embeddings = np.array(review_embeddings)
            
            # Compute cosine similarities
            similarities = np.dot(review_embeddings, question_embedding)
            
            # Get top-k indices
            top_indices = np.argsort(similarities)[-top_k:][::-1]
            
            # Create result dataframe
            result_reviews = filtered_reviews.iloc[top_indices].copy()
            result_reviews['similarity_score'] = similarities[top_indices]
            
            logger.info(f"✅ Found {len(result_reviews)} similar reviews")
            logger.info(f"   Top similarity score: {similarities[top_indices[0]]:.3f}")
            
            return result_reviews
            
        except Exception as e:
            logger.error(f"❌ Error finding similar reviews: {e}")
            return pd.DataFrame()
    
    def summarize_reviews(self, question: str, similar_reviews: pd.DataFrame) -> str:
        """
        Summarize most relevant reviews for user (NEW for chatbot)
        
        Args:
            question: Original user question
            similar_reviews: Top similar reviews dataframe
            
        Returns:
            str: Natural language summary
        """
        logger.info("📝 Generating summary of relevant reviews...")
        
        if len(similar_reviews) == 0:
            return "I couldn't find any reviews matching your criteria. Please try a different question or filters."
        
        # Prepare review texts for summarization
        review_texts = []
        for idx, row in similar_reviews.head(5).iterrows():  # Use top 5 for summarization
            review_texts.append(f"Review {idx+1} (Rating: {row['rating']}/5, {row['season']} {row['year']}, {row['reviewer_location']}): {row['review_text'][:200]}...")
        
        reviews_text = "\n\n".join(review_texts)
        
        # Create summarization prompt
        summarization_prompt = PromptTemplate(
            input_variables=["question", "reviews"],
            template="""You are a helpful Disney review analyst. Summarize the most relevant reviews to answer the user's question.

User Question: {question}

Relevant Reviews:
{reviews}

Provide a helpful summary that:
1. Directly addresses the user's question
2. Highlights key insights from the reviews
3. Mentions specific examples when relevant
4. Is conversational and helpful
5. Acknowledges limitations if the data is limited

Keep the response focused and under 200 words."""
        )
        
        # Create LLM chain
        summarization_chain = LLMChain(
            llm=self.llm,
            prompt=summarization_prompt
        )
        
        try:
            with get_openai_callback() as cb:
                summary = summarization_chain.run({
                    "question": question,
                    "reviews": reviews_text
                })
                self.total_cost += cb.total_cost
                
            logger.info("✅ Generated summary successfully")
            return summary.strip()
            
        except Exception as e:
            logger.error(f"❌ Error generating summary: {e}")
            return "Sorry, I encountered an error while summarizing the reviews. Please try again."
    
    def query_reviews(self, user_question: str) -> Dict[str, Any]:
        """
        Complete query pipeline: parse filters, retrieve, search, and summarize (NEW for chatbot)
        
        Args:
            user_question: Natural language question from user
            
        Returns:
            Dict[str, Any]: Query results with summary and metadata
        """
        logger.info(f"🤖 Processing query: '{user_question}'")
        
        try:
            # Step 1: Parse filters from question
            filters = self.parse_user_filters(user_question)
            
            # Step 2: Retrieve filtered reviews
            filtered_reviews = self.retrieve_filtered_reviews(filters)
            
            # Step 3: Find similar reviews
            similar_reviews = self.find_similar_reviews(user_question, filtered_reviews)
            
            # Step 4: Generate summary
            summary = self.summarize_reviews(user_question, similar_reviews)
            
            result = {
                'question': user_question,
                'filters': filters,
                'total_filtered': len(filtered_reviews),
                'top_similar': len(similar_reviews),
                'summary': summary,
                'similar_reviews': similar_reviews.to_dict('records') if len(similar_reviews) > 0 else [],
                'query_cost': self.total_cost
            }
            
            logger.info(f"✅ Query completed successfully")
            return result
            
        except Exception as e:
            logger.error(f"❌ Query failed: {e}")
            return {
                'question': user_question,
                'error': str(e),
                'summary': "Sorry, I encountered an error processing your question. Please try again."
            }

def main():
    """
    Main execution function for testing the chatbot pipeline
    """
    # Test the indexing pipeline with first 50 reviews
    
    # Configuration
    OPENAI_API_KEY = ChatbotConfig.OPENAI_API_KEY
    DATABASE_URL = ChatbotConfig.DATABASE_URL
    INPUT_CSV_PATH = ChatbotConfig.INPUT_CSV_PATH
    
    if not OPENAI_API_KEY:
        logger.error("❌ Please set OPENAI_API_KEY environment variable")
        return
    
    logger.info("🎯 Testing Chatbot Pipeline with First 50 Reviews")
    logger.info("=" * 60)
    
    # Initialize pipeline
    chatbot = ChatbotPipeline(OPENAI_API_KEY, DATABASE_URL, INPUT_CSV_PATH)
    
    # Run indexing pipeline
    indexing_results = chatbot.run_indexing_pipeline()
    
    # Test query functionality
    logger.info("\n🔍 Testing Query Functionality")
    logger.info("=" * 40)
    
    test_questions = [
        "What do visitors from Australia say about the park?",
        "How were the experiences during 2019?",
        "What are the main complaints about wait times?"
    ]
    
    for question in test_questions:
        logger.info(f"\n📝 Testing: {question}")
        result = chatbot.query_reviews(question)
        logger.info(f"💬 Summary: {result['summary'][:100]}...")
        
    logger.info(f"\n💰 Total cost: ${chatbot.total_cost:.4f}")

if __name__ == "__main__":
    main() 