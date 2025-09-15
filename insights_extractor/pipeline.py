#!/usr/bin/env python3
"""
Customer Review Analysis Pipeline
Implementation based on technicall_design_document.md specification with PostgreSQL integration

This pipeline processes Disney Hong Kong reviews to extract actionable insights
from customer pain points using the enhanced architecture defined in technicall_design_document.md.
"""

import pandas as pd
import numpy as np
import os
import json
import logging
import hashlib
from typing import List, Dict, Tuple, Any, Optional
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Core libraries
import hdbscan
import matplotlib.pyplot as plt
import seaborn as sns

# Database libraries
import psycopg2
from psycopg2.extras import RealDictCursor
import sqlalchemy
from sqlalchemy import create_engine, text
from sqlalchemy.pool import QueuePool

# NLP libraries
import nltk
from nltk.tokenize import sent_tokenize
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

# LangChain imports
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.schema import HumanMessage, SystemMessage
from langchain.callbacks import get_openai_callback
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ReviewAnalysisPipeline:
    """
    Main pipeline class implementing the enhanced review analysis system
    as specified in technicall_design_document.md with PostgreSQL integration
    """
    
    def __init__(self, openai_api_key: str, database_url: str, input_csv_path: str = "DisneylandReviews.csv", cache_dir: str = "llm_cache"):
        """
        Initialize the pipeline with OpenAI API key, database connection, and input data path
        
        Args:
            openai_api_key: OpenAI API key for LangChain
            database_url: PostgreSQL connection URL
            input_csv_path: Path to input CSV file
            cache_dir: Directory to store LLM cache files
        """
        self.openai_api_key = openai_api_key
        self.database_url = database_url
        self.input_csv_path = input_csv_path
        self.cache_dir = cache_dir
        
        # Initialize database connection
        self._init_database()
        
        # Initialize LangChain components
        from config import Config
        os.environ["OPENAI_API_KEY"] = openai_api_key
        self.llm = ChatOpenAI(model_name=Config.LLM_MODEL, temperature=Config.LLM_TEMPERATURE)
        self.embeddings = OpenAIEmbeddings(model=Config.EMBEDDING_MODEL, dimensions=Config.EMBEDDING_DIMENSIONS)
        
        # Data storage
        self.df = None
        self.processed_df = None
        self.clusters = None
        self.themes = {}
        self.priority_results = None
        self.total_cost = 0.0
        
        # Initialize cache
        self._init_cache()
        
        logger.info("✅ Pipeline initialized successfully with PostgreSQL integration!")
    
    def _init_database(self):
        """Initialize PostgreSQL database connection and schema"""
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
                conn.execute(text("SELECT 1"))
            
            # Create table and indexes
            self._create_database_schema()
            
            logger.info("✅ Database connection established and schema ready")
            
        except Exception as e:
            logger.error(f"❌ Database initialization failed: {e}")
            raise
    
    def _create_database_schema(self):
        """Create database schema from centralized Config.DATABASE_SCHEMA"""
        from config import Config
        
        try:
            # Try to create extension in separate transaction
            try:
                with self.engine.connect() as conn:
                    conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
                    conn.commit()
                logger.info("✅ Vector extension enabled")
            except Exception as e:
                logger.warning(f"⚠️ Could not create vector extension (may need admin privileges): {e}")
                logger.info("📝 Proceeding with TEXT-based embedding storage")
            
            # Check if table exists and has TEXT columns for vectors
            with self.engine.connect() as conn:
                # Check if table exists and has TEXT columns for vectors
                check_sql = """
                SELECT column_name, data_type 
                FROM information_schema.columns 
                WHERE table_name = 'pain_point' 
                AND column_name IN ('embedding_vector')
                AND data_type = 'text'
                """
                result = conn.execute(text(check_sql))
                old_text_columns = result.fetchall()
                
                if old_text_columns:
                    logger.info("🔄 Found existing table with TEXT vector columns, dropping to recreate with VECTOR type...")
                    conn.execute(text("DROP TABLE IF EXISTS pain_point CASCADE"))
                    conn.commit()
                    logger.info("✅ Old table dropped")
            
            # Create table and indexes in separate transaction
            with self.engine.connect() as conn:
                # Create table using centralized schema
                conn.execute(text(Config.DATABASE_SCHEMA['table_creation_sql']))
                
                # Create indexes using centralized schema
                for index_sql in Config.DATABASE_SCHEMA['indexes_sql']:
                    try:
                        conn.execute(text(index_sql))
                    except Exception as e:
                        # HNSW indexes might fail if pgvector not fully installed
                        if "hnsw" in index_sql.lower():
                            logger.warning(f"⚠️ Could not create HNSW index (may need pgvector extension): {e}")
                        else:
                            raise e
                
                # Reset sequence to start from 0 if database is empty
                count_result = conn.execute(text("SELECT COUNT(*) FROM pain_point"))
                row_count = count_result.scalar()
                
                if row_count == 0:
                    # Modify sequence to allow starting from 0, then reset it
                    conn.execute(text("ALTER SEQUENCE pain_point_id_seq MINVALUE 0"))
                    conn.execute(text("ALTER SEQUENCE pain_point_id_seq RESTART WITH 0"))
                    logger.info("✅ Database is empty - sequence reset to start from ID 0")
                else:
                    logger.info(f"📊 Database contains {row_count} existing records - keeping current sequence")
                
                conn.commit()
                
            logger.info("✅ Database schema created successfully")
            
        except Exception as e:
            logger.error(f"❌ Error creating database schema: {e}")
            raise
    
    def _db_write_1_after_normalization(self, df: pd.DataFrame) -> int:
        """
        DB Write 1: Store pain points with embeddings (simplified)
        Individual row writes after vectorization (Node 3)
        
        Args:
            df: DataFrame with embeddings
            
        Returns:
            int: Number of rows inserted
        """
        logger.info("💾 DB Write 1: Storing pain points with embeddings...")
        
        inserted_count = 0
        
        try:
            with self.engine.connect() as conn:
                for _, row in df.iterrows():
                    # Prepare embedding vectors for PostgreSQL VECTOR type
                    embedding_cols = [col for col in df.columns if col.startswith('embedding_')]
                    embedding_values = [float(row[col]) for col in embedding_cols]
                    
                    # Insert individual row
                    insert_sql = """
                    INSERT INTO pain_point (
                        review_id, sentence_index, review_text, rating, year, month, season,
                        branch, reviewer_location, pain_point_text, embedding_vector
                    ) VALUES (
                        :review_id, :sentence_index, :review_text, :rating, :year, :month, :season,
                        :branch, :reviewer_location, :pain_point_text, :embedding_vector
                    ) ON CONFLICT (review_id, sentence_index) DO NOTHING
                    """
                    
                    # Convert vectors to proper string format for PostgreSQL VECTOR type
                    embedding_vector_str = '[' + ','.join(map(str, embedding_values)) + ']'
                    
                    conn.execute(text(insert_sql), {
                        'review_id': str(row['Review_ID']),
                        'sentence_index': int(row.get('Sentence_Index', 0)),
                        'review_text': str(row.get('Original_Review_Text', row['Review_Text'])),
                        'rating': int(row['Rating']),
                        'year': int(row['Year']),
                        'month': int(row['Month']),
                        'season': str(row['Season']),
                        'branch': str(row.get('Branch', 'Hong Kong')),
                        'reviewer_location': str(row.get('Reviewer_Location', '')),
                        'pain_point_text': str(row['Review_Text']),
                        'embedding_vector': embedding_vector_str
                    })
                    
                    inserted_count += 1
                
                conn.commit()
                
            logger.info(f"✅ DB Write 1 completed: {inserted_count} pain points stored")
            return inserted_count
            
        except Exception as e:
            logger.error(f"❌ DB Write 1 failed: {e}")
            raise
    
    def _db_write_2_after_theme_generation(self, df: pd.DataFrame) -> int:
        """
        DB Write 2: Update records with cluster IDs and theme information
        Individual row updates after theme generation (Node 5)
        
        Args:
            df: DataFrame with themes
            
        Returns:
            int: Number of rows updated
        """
        logger.info("💾 DB Write 2: Updating with cluster IDs and themes...")
        
        updated_count = 0
        
        try:
            with self.engine.connect() as conn:
                for _, row in df.iterrows():
                    # Update individual row
                    update_sql = """
                    UPDATE pain_point 
                    SET cluster_id = :cluster_id, 
                        theme_label = :theme_label,
                        updated_at = CURRENT_TIMESTAMP
                    WHERE review_id = :review_id AND sentence_index = :sentence_index
                    """
                    
                    result = conn.execute(text(update_sql), {
                        'cluster_id': int(row['Cluster_ID']) if row['Cluster_ID'] != -1 else None,
                        'theme_label': str(row.get('Theme', '')) if pd.notna(row.get('Theme')) else None,
                        'review_id': str(row['Review_ID']),
                        'sentence_index': int(row.get('Sentence_Index', 0))
                    })
                    
                    updated_count += result.rowcount
                
                conn.commit()
                
            logger.info(f"✅ DB Write 2 completed: {updated_count} records updated")
            return updated_count
            
        except Exception as e:
            logger.error(f"❌ DB Write 2 failed: {e}")
            raise
    
    def _load_data_from_database(self) -> pd.DataFrame:
        """
        Load processed data from database for analysis
        
        Returns:
            pd.DataFrame: Data loaded from pain_point table
        """
        try:
            query = """
            SELECT id, review_id, sentence_index, review_text, rating, year, month, season,
                   branch, reviewer_location, pain_point_text, cluster_id, theme_label,
                   created_at, updated_at
            FROM pain_point
            ORDER BY review_id, sentence_index
            """
            
            df = pd.read_sql(query, self.engine)
            logger.info(f"📊 Loaded {len(df)} records from database")
            return df
            
        except Exception as e:
            logger.error(f"❌ Error loading data from database: {e}")
            raise

    def _init_cache(self):
        """Initialize the LLM cache system"""
        # Create cache directory if it doesn't exist
        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir)
            logger.info(f"📁 Created cache directory: {self.cache_dir}")
        
        self.cache_file = os.path.join(self.cache_dir, "pain_point_cache.json")
        
        # Load existing cache
        if os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, 'r', encoding='utf-8') as f:
                    self.cache = json.load(f)
                logger.info(f"💾 Loaded cache with {len(self.cache)} entries")
            except Exception as e:
                logger.warning(f"⚠️  Error loading cache: {e}")
                self.cache = {}
        else:
            self.cache = {}
            logger.info("💾 Initialized empty cache")
    
    def _get_cache_key(self, review_text: str) -> str:
        """Generate a cache key for the review text"""
        # Create a hash of the review text for consistent caching
        return hashlib.md5(review_text.strip().lower().encode('utf-8')).hexdigest()
    
    def _save_cache(self):
        """Save the cache to disk"""
        try:
            with open(self.cache_file, 'w', encoding='utf-8') as f:
                json.dump(self.cache, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.warning(f"⚠️  Error saving cache: {e}")
    
    def _get_from_cache(self, review_text: str) -> Optional[List[str]]:
        """Get pain points from cache if available"""
        cache_key = self._get_cache_key(review_text)
        return self.cache.get(cache_key)
    
    def _save_to_cache(self, review_text: str, pain_points: List[str]):
        """Save pain points to cache"""
        cache_key = self._get_cache_key(review_text)
        self.cache[cache_key] = pain_points
        # Save cache every 10 entries to avoid losing data
        if len(self.cache) % 10 == 0:
            self._save_cache()
    
    def load_data(self) -> pd.DataFrame:
        """
        Load and validate input CSV data according to technicall_design_document.md specification
        
        Returns:
            pd.DataFrame: Loaded and validated dataframe
        """
        logger.info("📊 Loading data...")
        
        try:
            # Try UTF-8 first, then fall back to latin-1 for encoding issues
            try:
                self.df = pd.read_csv(self.input_csv_path)
            except UnicodeDecodeError:
                logger.warning("⚠️ UTF-8 decode failed, trying latin-1 encoding...")
                self.df = pd.read_csv(self.input_csv_path, encoding='latin-1')
            
            logger.info(f"✅ Loaded {len(self.df)} reviews")
            
            # Validate required columns from technicall_design_document.md
            required_cols = ['Review_ID', 'Rating', 'Year_Month', 'Reviewer_Location', 'Review_Text', 'Branch']
            missing_cols = [col for col in required_cols if col not in self.df.columns]
            
            if missing_cols:
                raise ValueError(f"Missing required columns: {missing_cols}")
            
            # Basic data cleaning
            self.df = self.df.dropna(subset=['Review_Text'])
            self.df['Review_Text'] = self.df['Review_Text'].astype(str)
            
            logger.info(f"✅ Data validation complete. Final dataset: {len(self.df)} reviews")
            return self.df
            
        except Exception as e:
            logger.error(f"❌ Error loading data: {e}")
            raise
    
    def node_1_date_feature_engineering(self) -> pd.DataFrame:
        """
        Node 1: Date Feature Engineering
        Extract temporal features for analysis as per technicall_design_document.md
        
        Returns:
            pd.DataFrame: DataFrame with additional date features
        """
        logger.info("🗓️  Node 1: Date Feature Engineering")
        
        df = self.df.copy()
        
        # Parse Year_Month with missing value handling
        # Handle missing or invalid Year_Month values
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
        
        logger.info(f"✅ Added date features: Year, Month, Season")
        return df
    
    def node_2_sentence_splitting_pain_extraction(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Node 2: Sentence Splitting & Pain Point Extraction
        Split reviews into sentences and extract pain points in one LLM call using GPT-4o-mini
        
        Args:
            df: Input dataframe with date features
            
        Returns:
            pd.DataFrame: Expanded dataframe with pain point sentences
        """
        logger.info("✂️  Node 2: Sentence Splitting & Pain Point Extraction")
        
        pain_sentences = []
        total_processed = 0
        
        for idx, row in df.iterrows():
            review_text = row['Review_Text']
            
            # Extract pain points in one LLM call (splitting + filtering)
            pain_points = self._extract_pain_points_single_call(review_text)
            
            # Create new rows for each pain point
            for sentence_idx, sentence in enumerate(pain_points):
                new_row = row.copy()
                new_row['Review_Text'] = sentence
                new_row['Original_Review_Text'] = row['Review_Text']  # Keep original
                new_row['Review_ID'] = row['Review_ID']
                new_row['Sentence_Index'] = sentence_idx
                pain_sentences.append(new_row)
            
            total_processed += 1
            if total_processed % 10 == 0:
                logger.info(f"   Processed {total_processed}/{len(df)} reviews...")
        
        result_df = pd.DataFrame(pain_sentences)
        
        # Save cache after processing all reviews
        self._save_cache()
        logger.info(f"💾 Cache saved with {len(self.cache)} entries")
        
        logger.info(f"✅ Extracted {len(result_df)} pain point sentences from {len(df)} reviews")
        return result_df
    
    def _extract_pain_points_single_call(self, review_text: str) -> List[str]:
        """
        Use GPT-4o-mini to split review into sentences and extract pain points in one call using LangChain LLMChain
        Cache results to avoid duplicate API calls.
        
        Args:
            review_text: Full review text to analyze
            
        Returns:
            List[str]: Pain point sentences extracted from the review
        """
        if not review_text or not review_text.strip():
            return []
        
        # Check cache first
        cached_result = self._get_from_cache(review_text)
        if cached_result is not None:
            logger.debug(f"💾 Cache hit for review (length: {len(review_text)})")
            return cached_result
        
        
        # Create pain point extraction prompt template  
        pain_point_extraction_prompt = PromptTemplate(
            input_variables=["review_text"],
            template="""You are an analyst working for Disneyland Park in Hong Kong, tasked with analyzing online reviews submitted by visitors.

Instructions:
- Your goal is to extract **only pain points** from each review.
- A "pain point" is a **concrete source of dissatisfaction**, such as: service issues, unmet expectations, park limitations, or emotional frustration.
- Do **not include** positive or neutral statements.
- If there are no pain points, return an empty list.

Extraction Guidelines:
- Return a **JSON list** of the pain points, translated to English if needed.
- Use short phrases, like: "long queue to aquarium", "overcrowded restaurant", "limited ride options", "rainy weather affecting experience"
- Avoid generic expressions like "bad experience"
- If the review mentions **weather**, **crowds**, **wait times**, or **lack of rides**, treat them as valid pain points only if they clearly impacted the experience.
- Do not invent or infer — extract only what is clearly stated or implied.

Format:
```json
["<pain point 1>", "<pain point 2>", ...]
```

For example:
Review: "I came with my 2 kids. I didn't like the park, as it was rainy and the queue to the aquarium were awfully long and the restaurant was overcrowded"
Output: ["long queue to aquarium", "overcrowded restaurant"]

Review to analyze:
{review_text}"""
        )
        
        # Create LLM chain
        pain_point_chain = LLMChain(
            llm=self.llm,
            prompt=pain_point_extraction_prompt
        )
        
        try:
            with get_openai_callback() as cb:
                response = pain_point_chain.run({"review_text": review_text})
                self.total_cost += cb.total_cost
                
            result = response.strip()
            
            if result.lower() == "none":
                return []
            
            # Try to parse JSON first (improved for LangChain)
            try:
                # Look for JSON array in the response
                start_idx = result.find('[')
                end_idx = result.rfind(']') + 1
                if start_idx != -1 and end_idx != 0:
                    json_str = result[start_idx:end_idx]
                    pain_points = json.loads(json_str)
                    if isinstance(pain_points, list):
                        # Save to cache and return
                        self._save_to_cache(review_text, pain_points)
                        return pain_points
                    else:
                        return []
            except (json.JSONDecodeError, ValueError):
                pass
            
            # Fallback: Parse the numbered/bulleted response
            pain_points = []
            for line in result.split('\n'):
                line = line.strip()
                if line and (line[0].isdigit() or line.startswith('•') or line.startswith('-')):
                    # Remove numbering and clean up
                    sentence = line.split('.', 1)[-1].strip() if '.' in line else line
                    sentence = sentence.lstrip('•-').strip()
                    if sentence:
                        pain_points.append(sentence)
            
            # Save to cache and return
            self._save_to_cache(review_text, pain_points)
            return pain_points
                
        except Exception as e:
            logger.warning(f"⚠️  LLM error in pain point extraction: {e}")
            # Fallback to keyword-based filtering with sentence splitting
            sentences = sent_tokenize(review_text)
            negative_keywords = ['bad', 'terrible', 'awful', 'disappointing', 'problem', 'issue', 'complain', 'worst', 'horrible', 'annoying', 'frustrating']
            fallback_result = [sent for sent in sentences if any(keyword in sent.lower() for keyword in negative_keywords)]
            # Save fallback result to cache too
            self._save_to_cache(review_text, fallback_result)
            return fallback_result
    
    def node_3_text_vectorization(self, df: pd.DataFrame) -> pd.DataFrame:
        
        """
        Node 3: Text Vectorization
        Convert sentences to numerical vectors using OpenAI text-embedding-3-small
        
        Args:
            df: DataFrame with pain point sentences
            
        Returns:
            pd.DataFrame: DataFrame with embedding columns
        """
        logger.info("🔢 Node 3: Text Vectorization")
        
        # Get embeddings for all sentences
        sentences = df['Review_Text'].tolist()
        
        logger.info(f"   Getting embeddings for {len(sentences)} sentences...")
        
        try:
            with get_openai_callback() as cb:
                embeddings = self.embeddings.embed_documents(sentences)
                self.total_cost += cb.total_cost
                
            logger.info(f"   Embedding cost: ${cb.total_cost:.4f}")
            
            # Convert to numpy array and add to dataframe
            embeddings_array = np.array(embeddings)
            
            # Add embedding columns
            for i in range(embeddings_array.shape[1]):
                df[f'embedding_{i}'] = embeddings_array[:, i]
            
            logger.info(f"✅ Added {embeddings_array.shape[1]} embedding dimensions")
            
            # DB Write 1: Store pain points with embeddings (OpenAI embeddings are already normalized)
            self._db_write_1_after_normalization(df)
            
            return df
            
        except Exception as e:
            logger.error(f"❌ Error in vectorization: {e}")
            raise
    
    def run_pipeline_part_1(self, batch_size: int = None) -> Dict[str, Any]:
        """
        Execute pipeline Part 1: Nodes 1-3 (Date Features → Pain Extraction → Vectorization → DB Write 1)
        
        Args:
            batch_size: If provided, processes data in batches
        
        Returns:
            Dict[str, Any]: Part 1 results with cost and record count
        """
        logger.info("🚀 Starting Pipeline Part 1: Nodes 1-3 (Data Processing & Vectorization)")
        logger.info("=" * 60)
        
        try:
            # Load data
            self.load_data()
            
            # Node 1: Date Feature Engineering
            df = self.node_1_date_feature_engineering()
            
            if batch_size:
                # Batch processing for expensive LLM/API calls
                logger.info(f"📦 Using batch processing with batch size: {batch_size}")
                total_inserted = self.run_batch_processing(df, batch_size)
                logger.info(f"✅ Batch processing completed: {total_inserted} records processed")
            else:
                # Original sequential processing
                # Node 2: Sentence Splitting & Pain Point Extraction
                df = self.node_2_sentence_splitting_pain_extraction(df)
                
                # Node 3: Text Vectorization + DB Write 1
                df = self.node_3_text_vectorization(df)
                total_inserted = len(df)
            
            logger.info("=" * 60)
            logger.info("🎉 Pipeline Part 1 completed successfully!")
            logger.info(f"💰 Part 1 Cost: ${self.total_cost:.4f}")
            logger.info(f"💾 Records processed and stored: {total_inserted}")
            
            return {
                'total_inserted': total_inserted,
                'total_cost': self.total_cost,
                'part': 1,
                'status': 'completed'
            }
            
        except Exception as e:
            logger.error(f"❌ Pipeline Part 1 failed: {e}")
            raise

    def run_pipeline_part_2(self) -> Dict[str, Any]:
        """
        Execute pipeline Part 2: Nodes 4-5 (Clustering → Theme Generation → DB Write 2)
        
        Returns:
            Dict[str, Any]: Part 2 results with cluster and theme information
        """
        logger.info("🚀 Starting Pipeline Part 2: Nodes 4-5 (Clustering & Theme Generation)")
        logger.info("=" * 60)
        
        try:
            # Node 4: Clustering (reads from database)
            df = self.node_4_clustering()
            
            # Node 5: Theme Generation + DB Write 2
            df = self.node_5_theme_generation(df)
            
            logger.info("=" * 60)
            logger.info("🎉 Pipeline Part 2 completed successfully!")
            logger.info(f"💰 Part 2 Cost: ${self.total_cost:.4f}")
            logger.info(f"🎯 Clusters found: {self.clusters['n_clusters']}")
            logger.info(f"🏷️  Themes generated: {len(self.themes)}")
            
            return {
                'cluster_info': self.clusters,
                'themes': self.themes,
                'total_cost': self.total_cost,
                'part': 2,
                'status': 'completed'
            }
            
        except Exception as e:
            logger.error(f"❌ Pipeline Part 2 failed: {e}")
            raise

    def run_pipeline_part_3(self) -> Dict[str, Any]:
        """
        Execute pipeline Part 3: Nodes 6-8 (Grouping → Prioritization → Solutions)
        
        Returns:
            Dict[str, Any]: Part 3 results with analysis and solutions
        """
        logger.info("🚀 Starting Pipeline Part 3: Nodes 6-8 (Analysis & Solutions)")
        logger.info("=" * 60)
        
        try:
            # Load processed data from database
            df = self._load_data_from_database()
            
            # Node 6: Theme Grouping (using database)
            theme_grouping = self.node_6_theme_grouping(df)
            
            # Node 7: Impact Prioritization
            priority_themes = self.node_7_impact_prioritization(df, theme_grouping)
            
            # Node 8: Solution Generation & Visualization
            solutions_data = self.node_8_solution_generation_visualization(df, priority_themes)
            
            # Save processed data to CSV for reports
            self.processed_df = df
            
            # Ensure output directory exists
            from config import Config
            os.makedirs(Config.OUTPUT_DIR, exist_ok=True)
            priority_themes.to_csv(Config.PRIORITY_REPORT_PATH, index=False)
            
            logger.info("=" * 60)
            logger.info("🎉 Pipeline Part 3 completed successfully!")
            logger.info(f"💰 Part 3 Cost: ${self.total_cost:.4f}")
            logger.info(f"📈 Top Priority Theme: {priority_themes.iloc[0]['Theme']}")
            logger.info(f"📁 Priority analysis saved to: {Config.PRIORITY_REPORT_PATH}")
            
            return {
                'processed_data': df,
                'priority_themes': priority_themes,
                'theme_grouping': theme_grouping,
                'solutions': solutions_data,
                'total_cost': self.total_cost,
                'part': 3,
                'status': 'completed'
            }
            
        except Exception as e:
            logger.error(f"❌ Pipeline Part 3 failed: {e}")
            raise

    def run_pipeline_part_4(self, results_part_3: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute pipeline Part 4: Generate additional outputs and final report
        
        Args:
            results_part_3: Results from part 3 containing all pipeline data
        
        Returns:
            Dict[str, Any]: Complete pipeline results
        """
        logger.info("🚀 Starting Pipeline Part 4: Additional Outputs & Final Report")
        logger.info("=" * 60)
        
        try:
            # Load cluster and theme information from database if not available
            if self.clusters is None or self.themes is None:
                logger.info("   Loading cluster and theme information from database...")
                self._load_cluster_and_theme_info_from_database()
            
            # Combine all results for complete output
            complete_results = {
                'processed_data': results_part_3['processed_data'],
                'priority_themes': results_part_3['priority_themes'],
                'theme_grouping': results_part_3['theme_grouping'],
                'solutions': results_part_3['solutions'],
                'cluster_info': self.clusters,
                'themes': self.themes,
                'total_cost': self.total_cost
            }
            
            logger.info("=" * 60)
            logger.info("🎉 Pipeline Part 4 completed successfully!")
            logger.info(f"💰 Total Pipeline Cost: ${self.total_cost:.4f}")
            logger.info(f"💾 Database Records: {len(results_part_3['processed_data'])}")
            logger.info(f"📁 All deliverables ready for additional outputs generation")
            
            return complete_results
            
        except Exception as e:
            logger.error(f"❌ Pipeline Part 4 failed: {e}")
            raise

    def _load_cluster_and_theme_info_from_database(self):
        """Load cluster and theme information from database for part 4"""
        try:
            # Get cluster statistics
            cluster_stats_query = """
            SELECT 
                COUNT(DISTINCT cluster_id) FILTER (WHERE cluster_id IS NOT NULL AND cluster_id != -1) as n_clusters,
                COUNT(*) FILTER (WHERE cluster_id = -1 OR cluster_id IS NULL) as n_outliers,
                COUNT(*) as total_records
            FROM pain_point
            """
            
            with self.engine.connect() as conn:
                result = conn.execute(text(cluster_stats_query))
                stats = result.fetchone()
                
                if stats:
                    n_clusters = int(stats[0]) if stats[0] else 0
                    n_outliers = int(stats[1]) if stats[1] else 0
                    total_records = int(stats[2]) if stats[2] else 0
                    
                    self.clusters = {
                        'n_clusters': n_clusters,
                        'n_outliers': n_outliers,
                        'cluster_labels': [],  # Not needed for part 4
                        'outlier_ratio': n_outliers / total_records if total_records > 0 else 0
                    }
                else:
                    self.clusters = {'n_clusters': 0, 'n_outliers': 0, 'cluster_labels': [], 'outlier_ratio': 0}
                
                # Get unique themes
                themes_query = """
                SELECT DISTINCT cluster_id, theme_label 
                FROM pain_point 
                WHERE theme_label IS NOT NULL AND cluster_id IS NOT NULL
                ORDER BY cluster_id
                """
                
                result = conn.execute(text(themes_query))
                theme_rows = result.fetchall()
                
                self.themes = {}
                for row in theme_rows:
                    cluster_id = int(row[0]) if row[0] != -1 else -1
                    theme_label = row[1]
                    self.themes[cluster_id] = theme_label
                
                # Add miscellaneous for outliers if not present
                if -1 not in self.themes:
                    self.themes[-1] = "Miscellaneous Issues"
                
                logger.info(f"   Loaded cluster info: {n_clusters} clusters, {n_outliers} outliers")
                logger.info(f"   Loaded {len(self.themes)} themes from database")
                
        except Exception as e:
            logger.error(f"❌ Error loading cluster/theme info from database: {e}")
            # Set default values to prevent further errors
            self.clusters = {'n_clusters': 0, 'n_outliers': 0, 'cluster_labels': [], 'outlier_ratio': 0}
            self.themes = {-1: "Miscellaneous Issues"}

    def node_4_clustering(self, df: pd.DataFrame = None) -> pd.DataFrame:
        """
        Node 4: Clustering
        Group similar pain points automatically using HDBSCAN with optimized parameters
        
        Args:
            df: DataFrame with embeddings (optional, will load from database if not provided)
            
        Returns:
            pd.DataFrame: DataFrame with cluster assignments
        """
        logger.info("🎯 Node 4: Clustering")
        
        # Load data from database if not provided
        if df is None:
            logger.info("   Loading embedding data from database...")
            # Load embeddings from database
            embedding_query = """
            SELECT id, review_id, sentence_index, pain_point_text, embedding_vector
            FROM pain_point 
            WHERE embedding_vector IS NOT NULL
            ORDER BY review_id, sentence_index
            """
            
            try:
                with self.engine.connect() as conn:
                    result = conn.execute(text(embedding_query))
                    rows = result.fetchall()
                
                # Convert to DataFrame and parse embeddings
                data_rows = []
                embeddings_list = []
                
                for row in rows:
                    data_rows.append({
                        'id': row[0],
                        'Review_ID': row[1],
                        'Sentence_Index': row[2],
                        'Review_Text': row[3],
                    })
                    
                    # Parse embedding vector from string format [1.1,2.2,3.3,...]
                    embedding_str = row[4]
                    if embedding_str.startswith('[') and embedding_str.endswith(']'):
                        embedding_values = [float(x.strip()) for x in embedding_str[1:-1].split(',')]
                        embeddings_list.append(embedding_values)
                    else:
                        logger.error(f"Invalid embedding format for record {row[0]}")
                        raise ValueError(f"Invalid embedding format")
                
                df = pd.DataFrame(data_rows)
                embeddings_matrix = np.array(embeddings_list)
                
                logger.info(f"   Loaded {len(df)} records with embeddings from database")
                
            except Exception as e:
                logger.error(f"❌ Error loading embeddings from database: {e}")
                raise
                
        else:
            # Extract embedding columns from provided DataFrame
            embedding_cols = [col for col in df.columns if col.startswith('embedding_')]
            embeddings_matrix = df[embedding_cols].values
        
        # Apply HDBSCAN clustering with parameters from config
        from config import Config
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=Config.HDBSCAN_MIN_CLUSTER_SIZE,
            min_samples=Config.HDBSCAN_MIN_SAMPLES,
            metric=Config.HDBSCAN_METRIC,
            cluster_selection_method=Config.HDBSCAN_CLUSTER_SELECTION,
            cluster_selection_epsilon=0.1  # Added epsilon for more granular cluster selection
        )
        
        cluster_labels = clusterer.fit_predict(embeddings_matrix)
        
        # If we get too few clusters, try even more granular parameters
        n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
        
        if n_clusters < 5:  # If still too few clusters, try more aggressive parameters
            logger.info(f"   Initial clustering found {n_clusters} clusters, trying more granular parameters...")
            clusterer = hdbscan.HDBSCAN(
                min_cluster_size=2,  # More aggressive
                min_samples=1,
                metric=Config.HDBSCAN_METRIC,
                cluster_selection_method=Config.HDBSCAN_CLUSTER_SELECTION,
                cluster_selection_epsilon=0.05
            )
            cluster_labels = clusterer.fit_predict(embeddings_matrix)
            n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
        
        df['Cluster_ID'] = cluster_labels
        
        # Calculate cluster info
        unique_clusters = set(cluster_labels)
        n_outliers = sum(1 for label in cluster_labels if label == -1)
        
        self.clusters = {
            'n_clusters': n_clusters,
            'n_outliers': n_outliers,
            'cluster_labels': cluster_labels.tolist(),
            'outlier_ratio': n_outliers / len(cluster_labels)
        }
        
        logger.info(f"✅ Clustering completed: {n_clusters} clusters, {n_outliers} outliers")
        logger.info(f"   Outlier ratio: {self.clusters['outlier_ratio']:.2%}")
        
        return df
    
    def node_5_theme_generation(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Node 5: Theme Generation
        Create human-readable theme labels using GPT-4o-mini
        
        Args:
            df: DataFrame with cluster assignments
            
        Returns:
            pd.DataFrame: DataFrame with theme labels
        """
        logger.info("🏷️  Node 5: Theme Generation")
        
        # Get unique clusters (excluding outliers)
        unique_clusters = [int(c) for c in df['Cluster_ID'].unique() if c != -1]  # Convert to Python int
        
        themes = {}
        
        for cluster_id in unique_clusters:
            cluster_sentences = df[df['Cluster_ID'] == cluster_id]['Review_Text'].tolist()
            
            # Sample up to 10 representative sentences
            sample_sentences = cluster_sentences[:10] if len(cluster_sentences) > 10 else cluster_sentences
            
            theme = self._generate_theme_label(sample_sentences, cluster_id)
            themes[int(cluster_id)] = theme  # Ensure key is Python int
            
            logger.info(f"   Cluster {cluster_id}: {theme}")
        
        # Handle outliers (ensure key is Python int)
        themes[-1] = "Miscellaneous Issues"
        
        # Add theme column
        df['Theme'] = df['Cluster_ID'].map(themes)
        
        self.themes = themes
        logger.info(f"✅ Generated themes for {len(unique_clusters)} clusters")
        
        # DB Write 2: Update records with cluster IDs and theme information
        self._db_write_2_after_theme_generation(df)
        
        return df
    
    def _generate_theme_label(self, sentences: List[str], cluster_id: int) -> str:
        """
        Generate a theme label for a cluster using GPT-4o-mini with LangChain
        
        Args:
            sentences: Sample sentences from the cluster
            cluster_id: Cluster identifier
            
        Returns:
            str: Generated theme label
        """
        sentences_text = "\n".join([f"- {sent}" for sent in sentences])
        
        # Create theme generation prompt template
        theme_generation_prompt = PromptTemplate(
            input_variables=["cluster_id", "sentences_text"],
            template="""You are an expert at analyzing customer feedback themes. 
Analyze the following customer complaint sentences and provide a short, 
descriptive label (2-4 words) that captures the main shared problem theme.

Focus on the core issue being expressed. Be specific and actionable.
Examples: "Long Wait Times", "Poor Food Quality", "Staff Attitude", "Ride Maintenance"

Return only the theme label, nothing else.

Cluster {cluster_id} sentences:
{sentences_text}"""
        )
        
        # Create LLM chain
        theme_generation_chain = LLMChain(
            llm=self.llm,
            prompt=theme_generation_prompt
        )
        
        try:
            with get_openai_callback() as cb:
                response = theme_generation_chain.run({
                    "cluster_id": cluster_id,
                    "sentences_text": sentences_text
                })
                self.total_cost += cb.total_cost
                
            theme = response.strip()
            return theme if theme else f"Theme {cluster_id}"
            
        except Exception as e:
            logger.warning(f"⚠️  Error generating theme for cluster {cluster_id}: {e}")
            return f"Theme {cluster_id}"
    
    def node_6_theme_grouping(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Node 6: Theme Grouping
        Group data by theme only using PostgreSQL, excluding outliers (cluster_id = -1)
        
        Args:
            df: DataFrame with themes (can be None, will load from database)
            
        Returns:
            pd.DataFrame: Theme grouping dataframe
        """
        logger.info("📊 Node 6: Theme Grouping")
        
        # Load data from database for grouping analysis
        db_df = self._load_data_from_database()
        
        # Filter out outliers (cluster_id = -1) for theme grouping
        df_filtered = db_df[db_df['cluster_id'].notna() & (db_df['cluster_id'] != -1)].copy()
        outliers_count = len(db_df) - len(df_filtered)
        
        if outliers_count > 0:
            logger.info(f"   Excluding {outliers_count} outliers (cluster_id = -1) from theme grouping")
        
        # Group by Theme only
        theme_grouping = df_filtered.groupby('theme_label').agg({
            'id': 'count',
            'rating': 'mean',
            'cluster_id': 'first'
        }).reset_index().rename(columns={'id': 'Volume', 'theme_label': 'Theme'})
        
        logger.info(f"   Theme grouping: {len(theme_grouping)} themes")
        logger.info("✅ Theme grouping created (outliers excluded)")
        return theme_grouping
    
    def node_7_impact_prioritization(self, df: pd.DataFrame, theme_grouping: pd.DataFrame) -> pd.DataFrame:
        """
        Node 7: Impact Prioritization
        Rank themes by volume and rating impact
        
        Args:
            df: DataFrame with themes (can be None, will use database data)
            theme_grouping: Theme grouping dataframe
            
        Returns:
            pd.DataFrame: Prioritized themes sorted by volume and rating
        """
        logger.info("🎯 Node 7: Impact Prioritization")
        
        # Use theme grouping as base for analysis
        theme_analysis = theme_grouping.copy()
        
        # Prioritization: 
        # 1. Primary sort by theme volume (descending)
        # 2. Secondary sort by rating (ascending - lower is worse)
        theme_analysis = theme_analysis.sort_values(
            ['Volume', 'rating'], 
            ascending=[False, True]
        )
        
        # Add priority rank
        theme_analysis['Priority_Rank'] = range(1, len(theme_analysis) + 1)
        
        self.priority_results = theme_analysis
        
        logger.info("✅ Themes prioritized by volume and rating impact")
        logger.info("\n📈 Top 5 Priority Themes:")
        for idx, row in theme_analysis.head().iterrows():
            logger.info(f"   #{row['Priority_Rank']} {row['Theme']}: {row['Volume']} complaints, Avg Rating={row['rating']:.2f}")
        
        return theme_analysis
    
    def node_8_solution_generation_visualization(self, df: pd.DataFrame, priority_themes: pd.DataFrame) -> Dict[str, Any]:
        """
        Node 8: Solution Generation & Visualization
        Generate recommendations and visualizations using GPT-4o-mini + matplotlib
        
        Args:
            df: DataFrame with themes (can be None, will use database data)
            priority_themes: Prioritized themes dataframe
            
        Returns:
            Dict[str, Any]: Solutions and visualization data
        """
        logger.info("💡 Node 8: Solution Generation & Visualization")
        
        # Load data from database if needed
        if df is None or len(df) == 0:
            df = self._load_data_from_database()
        
        solutions = {}
        
        # Generate solutions for top themes
        top_themes = priority_themes.head(5)
        
        for idx, row in top_themes.iterrows():
            theme = row['Theme']
            volume = int(row['Volume'])  # Convert to Python int
            avg_rating = float(row['rating'])  # Convert to Python float
            
            # Get sample sentences for context from database
            theme_sentences_query = """
            SELECT pain_point_text 
            FROM pain_point 
            WHERE theme_label = :theme_name 
            LIMIT 3
            """
            
            try:
                with self.engine.connect() as conn:
                    result = conn.execute(text(theme_sentences_query), {"theme_name": theme})
                    theme_sentences = [row[0] for row in result.fetchall()]
            except Exception as e:
                logger.warning(f"⚠️ Could not fetch theme sentences: {e}")
                theme_sentences = []
            
            solution = self._generate_solution(theme, theme_sentences, volume, avg_rating)
            solutions[theme] = solution
            
            logger.info(f"   Generated solution for: {theme}")
        
        # Create visualizations
        visualizations = self._create_visualizations(df, priority_themes)
        
        results = {
            'solutions': solutions,
            'visualizations': visualizations,
            'summary_stats': {
                'total_pain_points': int(len(df)),
                'unique_themes': int(len(priority_themes)),
                'top_theme': top_themes.iloc[0]['Theme'] if len(top_themes) > 0 else 'N/A',
                'avg_rating_impact': float(top_themes['rating'].mean()) if len(top_themes) > 0 else 0.0
            }
        }
        
        logger.info("✅ Solutions generated and visualizations created")
        return results
    
    def _generate_solution(self, theme: str, sample_sentences: List[str], volume: int, avg_rating: float) -> str:
        """
        Generate actionable solution for a theme using GPT-4o-mini with LangChain
        
        Args:
            theme: Theme name
            sample_sentences: Sample sentences from the theme
            volume: Number of complaints
            avg_rating: Average rating for this theme
            
        Returns:
            str: Generated solution recommendations
        """
        sample_complaints_text = "\n".join([f"- {sent}" for sent in sample_sentences])
        
        # Create solution generation prompt template
        solution_generation_prompt = PromptTemplate(
            input_variables=["theme", "volume", "avg_rating", "sample_complaints"],
            template="""You are a customer experience consultant for Disney Hong Kong. 
Analyze the customer pain point theme and provide specific, actionable 
improvement recommendations that Disney can implement.

Structure your response as:
1. Root Cause Analysis (2-3 sentences)
2. Immediate Actions (3-4 bullet points)
3. Long-term Improvements (2-3 bullet points)
4. Success Metrics (2-3 measurable outcomes)

Be specific, practical, and focused on guest experience improvement.

Theme: {theme}
Volume: {volume} complaints
Average Rating: {avg_rating}/5.0

Sample complaints:
{sample_complaints}"""
        )
        
        # Create LLM chain
        solution_generation_chain = LLMChain(
            llm=self.llm,
            prompt=solution_generation_prompt
        )
        
        try:
            with get_openai_callback() as cb:
                response = solution_generation_chain.run({
                    "theme": theme,
                    "volume": volume,
                    "avg_rating": f"{avg_rating:.2f}",
                    "sample_complaints": sample_complaints_text
                })
                self.total_cost += cb.total_cost
                
            return response.strip()
            
        except Exception as e:
            logger.warning(f"⚠️  Error generating solution for {theme}: {e}")
            return f"Solution analysis needed for {theme} (Volume: {volume}, Rating: {avg_rating:.2f})"
    
    def _create_visualizations(self, df: pd.DataFrame, priority_themes: pd.DataFrame) -> Dict[str, str]:
        """
        Create and save visualizations
        
        Args:
            df: DataFrame with themes
            priority_themes: Prioritized themes dataframe
            
        Returns:
            Dict[str, str]: Dictionary of visualization file paths
        """
        from config import Config
        plt.style.use('seaborn-v0_8')
        visualization_files = {}
        
        # Ensure output directory exists
        os.makedirs(Config.OUTPUT_DIR, exist_ok=True)
        
        # 1. Theme Volume Distribution
        plt.figure(figsize=(12, 6))
        top_themes = priority_themes.head(10)
        plt.barh(top_themes['Theme'], top_themes['Volume'])
        plt.title('Top 10 Pain Point Themes by Volume')
        plt.xlabel('Number of Complaints')
        plt.tight_layout()
        plt.savefig(Config.VISUALIZATION_FILES['theme_volume'], dpi=300, bbox_inches='tight')
        plt.close()
        visualization_files['theme_volume'] = Config.VISUALIZATION_FILES['theme_volume']
        
        # 2. Priority Ranking Chart
        plt.figure(figsize=(10, 6))
        plt.scatter(priority_themes['Volume'], priority_themes['rating'], 
                   s=100, alpha=0.7, c=priority_themes['Priority_Rank'], cmap='RdYlBu_r')
        plt.colorbar(label='Priority Rank')
        plt.xlabel('Volume (Number of Complaints)')
        plt.ylabel('Average Rating')
        plt.title('Theme Priority Analysis: Volume vs Rating')
        
        # Annotate top themes
        for idx, row in priority_themes.head(5).iterrows():
            plt.annotate(row['Theme'], (row['Volume'], row['rating']), 
                        xytext=(5, 5), textcoords='offset points', fontsize=8)
        
        plt.tight_layout()
        plt.savefig(Config.VISUALIZATION_FILES['priority_analysis'], dpi=300, bbox_inches='tight')
        plt.close()
        visualization_files['priority_analysis'] = Config.VISUALIZATION_FILES['priority_analysis']
        
        # 3. Seasonal Distribution (from database)
        try:
            seasonal_query = """
            SELECT season, theme_label, COUNT(*) as count
            FROM pain_point 
            WHERE theme_label IS NOT NULL AND season IS NOT NULL
            GROUP BY season, theme_label
            """
            
            with self.engine.connect() as conn:
                seasonal_df = pd.read_sql(seasonal_query, conn)
            
            seasonal_pivot = seasonal_df.pivot(index='season', columns='theme_label', values='count').fillna(0)
            
            plt.figure(figsize=(12, 8))
            seasonal_pivot.plot(kind='bar', stacked=True)
            plt.title('Pain Point Themes by Season')
            plt.xlabel('Season')
            plt.ylabel('Number of Complaints')
            plt.legend(title='Theme', bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.tight_layout()
            plt.savefig(Config.VISUALIZATION_FILES['seasonal'], dpi=300, bbox_inches='tight')
            plt.close()
            visualization_files['seasonal'] = Config.VISUALIZATION_FILES['seasonal']
            
        except Exception as e:
            logger.warning(f"⚠️ Could not create seasonal visualization: {e}")
        
        return visualization_files
    
    def run_pipeline(self, batch_size: int = None) -> Dict[str, Any]:
        """
        Execute the complete enhanced pipeline as specified in technicall_design_document.md with PostgreSQL integration
        This method now calls the 4 separate pipeline parts for modularity
        
        Args:
            batch_size: If provided, processes data in batches up to clustering step
        
        Returns:
            Dict[str, Any]: Complete pipeline results
        """
        logger.info("🚀 Starting Complete Review Analysis Pipeline (technicall_design_document.md Implementation with PostgreSQL)")
        logger.info("=" * 60)
        
        try:
            # Part 1: Data Processing & Vectorization (Nodes 1-3)
            part1_results = self.run_pipeline_part_1(batch_size)
            
            # Part 2: Clustering & Theme Generation (Nodes 4-5)
            part2_results = self.run_pipeline_part_2()
            
            # Part 3: Analysis & Solutions (Nodes 6-8)
            part3_results = self.run_pipeline_part_3()
            
            # Part 4: Additional Outputs (ready for external generation)
            complete_results = self.run_pipeline_part_4(part3_results)
            
            logger.info("=" * 60)
            logger.info("🎉 Complete Pipeline executed successfully!")
            logger.info(f"💰 Total OpenAI API cost: ${self.total_cost:.4f}")
            logger.info(f"💾 Data stored in PostgreSQL database: {self.database_url.split('@')[1]}")
            logger.info(f"📁 All pipeline parts completed successfully")
            
            return complete_results
            
        except Exception as e:
            logger.error(f"❌ Complete Pipeline failed: {e}")
            raise

    def run_batch_processing(self, df: pd.DataFrame, batch_size: int) -> int:
        """
        Run batch processing for pain point extraction, vectorization, and DB writes
        
        Args:
            df: DataFrame with date features
            batch_size: Number of reviews to process per batch
            
        Returns:
            int: Total number of records inserted
        """
        logger.info(f"🚀 Starting batch processing with batch size: {batch_size}")
        
        total_batches = len(df) // batch_size + (1 if len(df) % batch_size > 0 else 0)
        total_inserted = 0
        
        logger.info(f"📊 Processing {len(df)} reviews in {total_batches} batches")
        
        for batch_num in range(total_batches):
            start_idx = batch_num * batch_size
            end_idx = min(start_idx + batch_size, len(df))
            batch_df = df.iloc[start_idx:end_idx].copy()
            
            logger.info(f"📦 Processing batch {batch_num + 1}/{total_batches}: records {start_idx + 1}-{end_idx}")
            
            try:
                # Node 2: Sentence Splitting & Pain Point Extraction (LLM calls)
                logger.info(f"   ✂️ Extracting pain points from {len(batch_df)} reviews...")
                batch_df = self.node_2_sentence_splitting_pain_extraction(batch_df)
                
                # Node 3: Text Vectorization (OpenAI embedding calls) + DB Write 1
                logger.info(f"   🔢 Vectorizing {len(batch_df)} pain point sentences...")
                batch_df = self.node_3_text_vectorization(batch_df)
                
                batch_inserted = len(batch_df)
                total_inserted += batch_inserted
                
                logger.info(f"✅ Batch {batch_num + 1} completed: {batch_inserted} records processed and stored")
                logger.info(f"💰 Current total cost: ${self.total_cost:.4f}")
                
            except Exception as e:
                logger.error(f"❌ Batch {batch_num + 1} failed: {e}")
                raise
        
        logger.info(f"🎉 All batches completed! Total records processed: {total_inserted}")
        return total_inserted
    
    def generate_executive_summary(self, results: Dict[str, Any]) -> str:
        """
        Generate executive summary of findings
        
        Args:
            results: Pipeline results
            
        Returns:
            str: Executive summary text
        """
        summary_data = results['solutions']['summary_stats']
        top_themes = results['priority_themes'].head(3)
        
        summary = f"""
# Executive Summary: Disney Hong Kong Review Analysis

## Key Findings

**Total Analysis Scope:**
- {summary_data['total_pain_points']} customer pain points analyzed
- {summary_data['unique_themes']} distinct problem themes identified
- Average rating impact: {summary_data['avg_rating_impact']:.2f}/5.0
- Total processing cost: ${results['total_cost']:.4f}
- Data stored in PostgreSQL database for future analysis

## Top Priority Issues

"""
        
        for idx, (_, row) in enumerate(top_themes.iterrows(), 1):
            summary += f"""
### {idx}. {row['Theme']}
- **Volume:** {row['Volume']} complaints
- **Average Rating:** {row['rating']:.2f}/5.0
- **Priority Rank:** #{row['Priority_Rank']}
"""
        
        summary += """

## Recommended Actions

Based on the analysis, immediate attention should be focused on the top 3 themes above, 
as they represent the highest impact on customer satisfaction.

## Data Storage

All processed data has been stored in the PostgreSQL database for:
- Real-time querying and analysis
- Historical trend analysis
- Integration with other business systems
- Scalable processing of future reviews

## Next Steps

1. Review detailed solutions for each priority theme
2. Implement quick wins from immediate actions
3. Develop long-term improvement roadmap
4. Monitor success metrics post-implementation
5. Set up automated pipeline runs for continuous insights

*Detailed solutions and visualizations are available in the generated reports.*
"""
        
        return summary

def main():
    """
    Main execution function
    """
    from config import Config
    
    # Configuration
    OPENAI_API_KEY = Config.OPENAI_API_KEY
    DATABASE_URL = Config.DATABASE_URL
    
    if not OPENAI_API_KEY:
        logger.error("❌ Please set OPENAI_API_KEY environment variable")
        return
    
    # Initialize and run pipeline with explicit 1000 reviews file
    pipeline = ReviewAnalysisPipeline(
        OPENAI_API_KEY, 
        DATABASE_URL, 
        input_csv_path=Config.INPUT_CSV_PATH  # Use config path
    )
    results = pipeline.run_pipeline()
    
    # Generate executive summary
    summary = pipeline.generate_executive_summary(results)
    
    # Ensure output directory exists
    os.makedirs(Config.OUTPUT_DIR, exist_ok=True)
    with open(Config.EXECUTIVE_SUMMARY_PATH, 'w') as f:
        f.write(summary)
    
    logger.info(f"📄 Executive summary saved to: {Config.EXECUTIVE_SUMMARY_PATH}")

if __name__ == "__main__":
    main() 