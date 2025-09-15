"""
Configuration file for the Review Analysis Pipeline
Based on technicall_design_document.md specifications with PostgreSQL integration

🎯 SINGLE SOURCE OF TRUTH
This file is the ONLY place where pipeline configuration should be modified.
All other files reference this configuration to avoid duplication.

📝 HOW TO MAKE CHANGES:
1. Want to change models, clustering params, or database settings? → Edit this file only
2. Want to add/remove pipeline nodes? → Edit PIPELINE_NODES and PIPELINE_TECHNOLOGIES only
3. Want to change database schema? → Edit DATABASE_SCHEMA only
4. All other files will automatically use these changes

🚫 DO NOT EDIT:
- Technology definitions in other files (they import from here)
- Node descriptions in runner.py or step_by_step_runner.py
"""

import os
from dotenv import load_dotenv
from typing import Dict, List, Any
from dataclasses import dataclass

# Load environment variables
load_dotenv(override=True)

@dataclass
class PipelineNode:
    """Single definition of a pipeline node"""
    id: int
    name: str
    function_name: str
    technology: str
    purpose: str
    description: str
    database_operation: str = None

class Config:
    """
    🎯 MASTER CONFIGURATION CLASS
    Single source of truth for all pipeline settings
    """
    
    # ================================
    # 🔑 CORE API CONFIGURATION
    # ================================
    OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
    
    # ================================
    # 🗄️ DATABASE CONFIGURATION
    # ================================
    DATABASE_HOST = os.getenv('DATABASE_HOST', 'localhost')
    DATABASE_PORT = os.getenv('DATABASE_PORT', '5432')
    DATABASE_NAME = 'disneyland_reviews'
    DATABASE_USER = 'eghezail'
    DATABASE_PASSWORD = os.getenv('DATABASE_PASSWORD', '')
    DATABASE_URL = f"postgresql://{DATABASE_USER}:{DATABASE_PASSWORD}@{DATABASE_HOST}:{DATABASE_PORT}/{DATABASE_NAME}"
    PAIN_POINT_TABLE = "pain_point"
    
    # ================================
    # 🤖 MODEL CONFIGURATION
    # ================================
    LLM_MODEL = "gpt-4o-mini"
    EMBEDDING_MODEL = "text-embedding-3-small"
    EMBEDDING_DIMENSIONS = 512
    LLM_TEMPERATURE = 0.1
    
    # ================================
    # 📊 CLUSTERING CONFIGURATION
    # ================================
    HDBSCAN_MIN_CLUSTER_SIZE = 20
    HDBSCAN_MIN_SAMPLES = 2
    HDBSCAN_METRIC = 'euclidean'
    HDBSCAN_CLUSTER_SELECTION = 'eom'
    
    # ================================
    # 📁 DATA CONFIGURATION
    # ================================
    INPUT_CSV_PATH = "/Users/eghezail/Desktop/LLM_Model/data/DisneylandReviews.csv"  # Test with 50 reviews
    REQUIRED_COLUMNS = ['Review_ID', 'Rating', 'Year_Month', 'Reviewer_Location', 'Review_Text', 'Branch']
    
    # ================================
    # 📤 OUTPUT CONFIGURATION
    # ================================
    OUTPUT_DIR = "/Users/eghezail/Desktop/LLM_Model/insights_extractor/reports/"
    PRIORITY_REPORT_PATH = OUTPUT_DIR + "priority_analysis_report.csv"
    EXECUTIVE_SUMMARY_PATH = OUTPUT_DIR + "executive_summary.md"
    SOLUTIONS_PATH = OUTPUT_DIR + "solution_recommendations.json"
    
    # ================================
    # ⚙️ PROCESSING CONFIGURATION
    # ================================
    MAX_SENTENCES_PER_THEME = 10
    TOP_THEMES_FOR_SOLUTIONS = 5
    BATCH_SIZE_PROCESSING = 10
    MAX_PROCESSING_COST = 10.0  # Maximum allowed cost in USD for processing
    
    # ================================
    # 📊 VISUALIZATION CONFIGURATION
    # ================================
    FIGURE_SIZE = (12, 6)
    DPI = 300
    PLOT_STYLE = 'seaborn-v0_8'
    VISUALIZATION_FILES = {
        'theme_volume': OUTPUT_DIR + 'theme_volume_distribution.png',
        'priority_analysis': OUTPUT_DIR + 'priority_analysis.png',
        'seasonal': OUTPUT_DIR + 'seasonal_distribution.png'
    }
    
    # ================================
    # 🎯 SUCCESS METRICS
    # ================================
    MIN_COVERAGE_THRESHOLD = 0.8  # 80% of negative sentences clustered
    MIN_THEME_QUALITY_SCORE = 0.7  # Manual validation threshold
    MIN_ACTIONABILITY_SCORE = 0.8  # Feasibility assessment threshold
    
    # ================================
    # 🗓️ SEASON MAPPING
    # ================================
    SEASON_MAPPING = {
        'Spring': [3, 4, 5],
        'Summer': [6, 7, 8],
        'Autumn': [9, 10, 11],
        'Winter': [12, 1, 2]
    }
    
    # ================================
    # 🚀 PIPELINE DEFINITION (SINGLE SOURCE OF TRUTH)
    # ================================
    PIPELINE_NODES = [
        PipelineNode(
            id=0,
            name="Date Feature Engineering",
            function_name="node_0_date_feature_engineering",
            technology="pandas",
            purpose="Extract temporal features for analysis",
            description="Parse Year_Month to extract year, month, and season components"
        ),
        PipelineNode(
            id=1,
            name="Sentence Splitting & Pain Point Extraction",
            function_name="node_1_sentence_splitting_pain_extraction",
            technology="GPT-4o-mini",
            purpose="Split reviews into sentences and extract only pain points in one LLM call",
            description="Use GPT-4o-mini to identify and extract negative feedback sentences"
        ),
        PipelineNode(
            id=2,
            name="Text Vectorization",
            function_name="node_2_text_vectorization",
            technology="OpenAI text-embedding-3-small size 512",
            purpose="Convert sentences to numerical vectors",
            description="Generate 512-dimensional embeddings for semantic similarity analysis"
        ),
        PipelineNode(
            id=3,
            name="Database Write",
            function_name="node_3_database_write",
            technology="psycopg2/SQLAlchemy",
            purpose="Store pain points with embeddings",
            description="Store pain points with OpenAI embeddings (already normalized)",
            database_operation="DB Write 1: Store pain points with embeddings"
        ),
        PipelineNode(
            id=4,
            name="Clustering",
            function_name="node_4_clustering",
            technology="HDBSCAN",
            purpose="Group similar pain points automatically",
            description="Use HDBSCAN to cluster similar pain point sentences"
        ),
        PipelineNode(
            id=5,
            name="Theme Generation",
            function_name="node_5_theme_generation",
            technology="GPT-4o-mini",
            purpose="Create human-readable theme labels",
            description="Generate descriptive theme names for each cluster",
            database_operation="DB Write 2: Update with cluster IDs and theme labels"
        ),
        PipelineNode(
            id=6,
            name="Dimensional Grouping",
            function_name="node_6_dimensional_grouping",
            technology="pandas + PostgreSQL",
            purpose="Group data across multiple dimensions",
            description="Aggregate pain points by theme, rating, season using PostgreSQL queries"
        ),
        PipelineNode(
            id=7,
            name="Impact Prioritization",
            function_name="node_7_impact_prioritization",
            technology="pandas + PostgreSQL",
            purpose="Rank themes by business impact",
            description="Analyze volume and rating correlations to prioritize improvement areas"
        ),
        PipelineNode(
            id=8,
            name="Solution Generation & Visualization",
            function_name="node_8_solution_generation_visualization",
            technology="GPT-4o-mini + matplotlib",
            purpose="Generate actionable recommendations",
            description="Create improvement suggestions and visualizations for top priority themes"
        )
    ]
    
    # ================================
    # 💾 DATABASE SCHEMA (SINGLE SOURCE OF TRUTH)
    # ================================
    DATABASE_SCHEMA = {
        'table_creation_sql': """
        CREATE EXTENSION IF NOT EXISTS vector;
        CREATE TABLE IF NOT EXISTS pain_point (
            id SERIAL PRIMARY KEY,
            review_id VARCHAR(255) NOT NULL,
            sentence_index INTEGER NOT NULL,
            review_text TEXT NOT NULL,
            rating INTEGER NOT NULL,
            year INTEGER NOT NULL,
            month INTEGER NOT NULL,
            season VARCHAR(20),
            branch VARCHAR(255) DEFAULT 'Hong Kong',
            reviewer_location VARCHAR(255),
            pain_point_text TEXT NOT NULL,
            embedding_vector VECTOR(512),
            cluster_id INTEGER,
            theme_label VARCHAR(255),
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(review_id, sentence_index)
        );
        """,
        'indexes_sql': [
            "CREATE INDEX IF NOT EXISTS idx_pain_point_cluster ON pain_point(cluster_id);",
            "CREATE INDEX IF NOT EXISTS idx_pain_point_theme ON pain_point(theme_label);",
            "CREATE INDEX IF NOT EXISTS idx_pain_point_rating ON pain_point(rating);",
            "CREATE INDEX IF NOT EXISTS idx_pain_point_year ON pain_point(year);",
            "CREATE INDEX IF NOT EXISTS idx_pain_point_month ON pain_point(month);",
            "CREATE INDEX IF NOT EXISTS idx_pain_point_season ON pain_point(season);",
            "CREATE INDEX IF NOT EXISTS idx_pain_point_embedding_hnsw ON pain_point USING hnsw (embedding_vector vector_cosine_ops);"
        ]
    }
    
    # ================================
    # 📋 DERIVED PROPERTIES (AUTO-GENERATED)
    # ================================
    @classmethod
    def get_pipeline_info(cls) -> Dict[str, Any]:
        """
        🎯 AUTO-GENERATED pipeline information from SINGLE SOURCE
        This method generates all pipeline metadata from the centralized definitions above
        """
        return {
            'name': 'Customer Review Analysis Pipeline',
            'description': 'Technical implementation based on technicall_design_document.md specification with PostgreSQL storage',
            'nodes': [node.name for node in cls.PIPELINE_NODES],
            'technologies': {node.name: node.technology for node in cls.PIPELINE_NODES},
            'database_operations': {
                node.name: node.database_operation 
                for node in cls.PIPELINE_NODES 
                if node.database_operation
            },
            'output_deliverables': [
                'PostgreSQL Database',
                'Priority Analysis Report',
                'Solution Recommendations',
                'Visualizations',
                'Executive Summary'
            ],
            'total_nodes': len(cls.PIPELINE_NODES),
            'database_config': {
                'name': cls.DATABASE_NAME,
                'user': cls.DATABASE_USER,
                'table': cls.PAIN_POINT_TABLE
            },
            'model_config': {
                'llm': cls.LLM_MODEL,
                'embedding': cls.EMBEDDING_MODEL,
                'dimensions': cls.EMBEDDING_DIMENSIONS
            }
        }
    
    @classmethod
    def get_node_by_id(cls, node_id: int) -> PipelineNode:
        """Get specific node configuration by ID"""
        return next((node for node in cls.PIPELINE_NODES if node.id == node_id), None)
    
    @classmethod
    def get_node_by_name(cls, name: str) -> PipelineNode:
        """Get specific node configuration by name"""
        return next((node for node in cls.PIPELINE_NODES if node.name == name), None)
    
    @classmethod
    def get_database_nodes(cls) -> List[PipelineNode]:
        """Get all nodes that perform database operations"""
        return [node for node in cls.PIPELINE_NODES if node.database_operation]
    
    # ================================
    # ✅ VALIDATION METHODS
    # ================================
    @classmethod
    def validate_config(cls):
        """Validate all configuration settings"""
        errors = []
        
        # API Configuration
        if not cls.OPENAI_API_KEY:
            errors.append("OPENAI_API_KEY not found in environment variables")
        
        # File Configuration
        if not os.path.exists(cls.INPUT_CSV_PATH):
            errors.append(f"Input CSV file not found: {cls.INPUT_CSV_PATH}")
        
        # Model Configuration
        if cls.EMBEDDING_DIMENSIONS != 512:
            errors.append("Embedding dimensions must be 512 as per technicall_design_document.md")
        
        if cls.LLM_MODEL != "gpt-4o-mini":
            errors.append("LLM model must be gpt-4o-mini as per technicall_design_document.md")
        
        if cls.EMBEDDING_MODEL != "text-embedding-3-small":
            errors.append("Embedding model must be text-embedding-3-small as per technicall_design_document.md")
        
        # Database Configuration
        if not cls.DATABASE_USER:
            errors.append("Database user must be specified as per technicall_design_document.md")
        
        # Pipeline Configuration
        if len(cls.PIPELINE_NODES) == 0:
            errors.append("No pipeline nodes defined")
        
        if errors:
            raise ValueError(f"Configuration validation failed:\n" + "\n".join(f"- {error}" for error in errors))
        
        return True
    
    # ================================
    # 📋 USAGE GUIDANCE
    # ================================
    @classmethod
    def print_usage_guide(cls):
        """Print guidance on how to modify the pipeline"""
        guide = """
🎯 PIPELINE CONFIGURATION GUIDE

📝 TO MAKE CHANGES, EDIT ONLY config.py:

1. 🤖 Change Models:
   - LLM_MODEL = "gpt-4o-mini"
   - EMBEDDING_MODEL = "text-embedding-3-small"
   
2. 🎯 Adjust Clustering:
   - HDBSCAN_MIN_CLUSTER_SIZE = 3
   - HDBSCAN_MIN_SAMPLES = 2
   
3. 🗄️ Database Settings:
   - DATABASE_NAME = "disneyland_reviews"
   - DATABASE_USER = "eghezail"
   
4. 🚀 Add/Remove Pipeline Nodes:
   - Edit PIPELINE_NODES list
   - All files will automatically update
   
5. 💾 Modify Database Schema:
   - Edit DATABASE_SCHEMA dictionary
   - Schema will be auto-applied

🚫 DO NOT EDIT THESE FILES:
- Technology stack in README.md
- Node definitions in other files

✅ THESE FILES AUTO-UPDATE:
- runner.py (uses Config.get_pipeline_info())
- step_by_step_runner.py (uses Config.PIPELINE_NODES)
- setup.py (uses Config for validation)
- All generated metadata files
"""
        print(guide)

# ================================
# 🔧 LEGACY SUPPORT (TEMPORARY)
# ================================
# Keep these for backward compatibility during transition
PAIN_POINT_TABLE_SQL = Config.DATABASE_SCHEMA['table_creation_sql']
PAIN_POINT_INDEXES_SQL = Config.DATABASE_SCHEMA['indexes_sql'] 

CACHE_DIR = "/Users/eghezail/Desktop/LLM_Model/llm_cache"  # Absolute path 