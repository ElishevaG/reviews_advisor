# Technical Design Document: Customer Review Analysis Pipeline

## Overview
This document outlines the technical implementation of a customer review analysis pipeline that processes Disney Hong Kong reviews to extract actionable insights from customer pain points, with all data stored in PostgreSQL for unified persistence and analysis.

## Pipeline Architecture

| Component | Technology | Purpose |
|-----------|------------|---------|
| **Database** | **PostgreSQL 16+** with **pgvector** | Unified storage for structured data and vector embeddings |
| **Date Feature Engineering** | pandas | Extract temporal features for analysis |
| **Sentence Splitting & Pain Point Extraction** | GPT-4o-mini | Split reviews into sentences and extract only pain points |
| **Text Vectorization** | OpenAI text-embedding-3-small size 512 | Convert sentences to numerical vectors (already normalized) |
| **Database Write 1** | psycopg2/SQLAlchemy | Store pain points with embeddings |
| **Clustering** | HDBSCAN | Group similar pain points automatically |
| **Theme Generation** | GPT-4o-mini | Create human-readable theme labels |
| **Database Write 2** | psycopg2/SQLAlchemy | Update records with cluster IDs and theme information |
| **Dimensional Grouping** | pandas + PostgreSQL | Group data across multiple dimensions |
| **Impact Prioritization** | pandas + PostgreSQL | Rank themes by volume and rating impact |
| **Solution Generation & Visualization** | GPT-4o-mini + matplotlib | Generate recommendations and visualizations |

## Data Flow

```
Raw CSV → [Node 1] → Date Features → [Node 2] → Pain Point Sentences → 
[Node 3] → Embeddings → [DB Write 1: Individual Inserts] →
[Node 4] → Clusters → [Node 5] → Themes → [DB Write 2: Individual Updates] → [Node 6] → Theme Groups → 
[Node 7] → Priorities → [Node 8] → Solutions & Visualizations
```

## Input Data Structure
- **File Format**: CSV (input only)
- **Columns**:
  - `Review_ID`: Unique identifier for each review
  - `Rating`: Customer rating (1-5 stars)
  - `Year_Month`: Review date in YYYY-MM format
  - `Reviewer_Location`: Geographic location of reviewer
  - `Review_Text`: Full review content
  - `Branch`: Constant value "Hong Kong"

## Database Configuration
- **Database**: PostgreSQL 16+ with pgvector extension
- **Database Name**: `disneyland_reviews`
- **User**: `eghezail`
- **Required Extensions**: `pgvector` for vector storage and similarity operations

## Database Schema

```sql
CREATE TABLE pain_point (
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
    -- Metadata
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    -- Unique constraint
    UNIQUE(review_id, sentence_index)
);

-- Indexes for performance
CREATE INDEX idx_pain_point_cluster ON pain_point(cluster_id);
CREATE INDEX idx_pain_point_theme ON pain_point(theme_label);
CREATE INDEX idx_pain_point_rating ON pain_point(rating);
CREATE INDEX idx_pain_point_year ON pain_point(year);
CREATE INDEX idx_pain_point_month ON pain_point(month);
CREATE INDEX idx_pain_point_season ON pain_point(season);
```



### Database Write Operations

**DB Write 1 (After Node 3 - Text Vectorization)**:
- Insert individual rows into `pain_point` table
- Include: review metadata, pain point text, embeddings (already normalized)
- Leave clustering fields as NULL initially

**DB Write 2 (After Node 5 - Theme Generation)**:
- Update existing rows with clustering results
- Set: cluster_id, theme_label
- Update: updated_at timestamp

## Output Deliverables

1. **PostgreSQL Database**: Complete dataset stored in `pain_point` table with all computed features
2. **Priority Analysis Report**: SQL-generated ranked list of themes by business impact
3. **Solution Recommendations**: Actionable improvements for top priority themes
4. **Visualizations**: Charts showing volume distribution, impact analysis, and seasonal patterns
5. **Executive Summary**: Business-friendly report with key findings and recommendations

## Success Metrics

- **Coverage**: % of negative sentences successfully clustered and stored
- **Theme Quality**: Manual validation of generated themes in database
- **Actionability**: Feasibility assessment of generated solutions
- **Business Impact**: Correlation between identified issues and rating patterns
- **Data Integrity**: Successful storage and retrieval of all pipeline outputs

## Implementation Notes

- All external API calls routed through LangChain for consistency
- PostgreSQL connection pooling for efficient database operations
- Individual row writes for real-time processing capability
- Configurable parameters for clustering and theme generation
- Database transactions for data consistency during updates
- Scalable architecture supporting concurrent processing
- Vector similarity search capabilities using pgvector for future analysis

## Pipeline Execution Options

### Standard Execution
```bash
python runner.py
```
Processes all reviews sequentially in a single batch.

### Batch Processing (Recommended for Large Datasets)
```bash
python runner.py --batch-size 10    # Process 10 reviews per batch
python runner.py --batch-size 25    # Process 25 reviews per batch
python runner.py --batch-size 50    # Process 50 reviews per batch
```

**Batch Processing Benefits:**
- **API Rate Limit Management**: Avoids OpenAI rate limits with large datasets
- **Cost Control**: Incremental processing with real-time cost monitoring
- **Memory Efficiency**: Processes data in manageable chunks
- **Fault Tolerance**: Resume processing from failed batches
- **Progress Tracking**: Real-time batch progress and cost updates

**Batch Processing Flow:**
1. **Batch 1-N**: Pain Point Extraction (LLM) + Text Vectorization (OpenAI) + DB Write 1
2. **Load Complete Dataset**: Retrieve all processed data from database
3. **Continue Pipeline**: Clustering → Theme Generation → Analysis (on complete dataset)
