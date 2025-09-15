# Technical Design Document: Disney Review Chatbot

## Overview

This document outlines the technical implementation of a natural language chatbot for analyzing customer reviews of multiple Disney parks: Disneyland_Paris, Disneyland_HongKong, and Disneyland_California. The chatbot processes queries using vector similarity and LLM summarization, enabling internal stakeholders to explore visitor feedback in a conversational interface. All data is persisted in PostgreSQL using pgvector for similarity search.

## Input Data Structure

* **Source**: PostgreSQL table `disney_review`
* **Columns**:

  * `review_id`: Unique review identifier
  * `review_text`: Full customer review in English
  * `rating`: Integer rating (1–5)
  * `year`: Year of visit
  * `month`: Month of visit (integer)
  * `season`: Visit season (e.g., "Spring")
  * `reviewer_location`: Country of reviewer
  * `branch`: Disney park branch ("Disneyland_HongKong", "Disneyland_Paris", "Disneyland_California")
  * `embedding_vector`: 512-dimensional vector (OpenAI, already normalized)

## Database Configuration

* **Database**: PostgreSQL 16+ with pgvector extension
* **Database Name**: `disneyland_reviews`
* **User**: `eghezail`
* **Required Extensions**: `pgvector`

## Database Schema

```sql
CREATE TABLE disney_review (
    review_id VARCHAR(255) NOT NULL,
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

-- Indexes for metadata filters
CREATE INDEX idx_year ON disney_review(year);
CREATE INDEX idx_month ON disney_review(month);
CREATE INDEX idx_season ON disney_review(season);
CREATE INDEX idx_location ON disney_review(reviewer_location);
CREATE INDEX idx_branch ON disney_review(branch);
```

## Pipeline Architecture

### 1. Indexing & Preprocessing

| Component                    | Technology                      | Purpose                                                                                                            |
| ---------------------------- | ------------------------------- | ------------------------------------------------------------------------------------------------------------------ |
| **Date Feature Engineering** | pandas                          | Extract `year`, `month`, and `season` from the CSV `Year_Month` column                                             |
| **Metadata Enumeration**     | pandas                          | Extract unique values of metadata fields (e.g., year, month, season, reviewer\_location) and save to metadata file |                                                                           |
| **Vectorization**            | OpenAI `text-embedding-3-small` | Embed review text and questions (already normalized)                                                                                    |
| **Database Write** | psycopg2/SQLAlchemy | Store disney_review with embeddings |

### 2. Search & Interaction (runtime)

| Component                    | Technology                      | Purpose                                                                                                            |
| ---------------------------- | ------------------------------- | ------------------------------------------------------------------------------------------------------------------ |
| **Filter Parsing**           | GPT-4o-mini via LangChain       | Generate query metadata filters                                                                             |
| **Top Similar Reviews**         | psycopg2            | Query reviews matching filters, calculate cosine similarity,rank                                                                                     |                                                                                     |
| **Summarization**            | GPT-4o-mini via LangChain       | Summarize most relevant reviews                                                                                    |
| **Chat Interface**           | Streamlit                       | UI for natural language queries                                                                                    |


## Data Flow

```
User Question → [1] LangChain Filter Parser → [2] SQL Filter Query →
[3] Retrieve Metadata + Embeddings → [4] Get Question Embedding (normalized) →
[5] Cosine Similarity Computation → [6] Select Top 10 Reviews →
[7] Summarization (GPT-4o-mini) → [8] Display Summary in Streamlit
```

### Database Write Operations

* All reviews are preloaded with:

  * OpenAI embeddings (already normalized)
  * Indexed metadata for fast filtering

## Output Deliverables

1. **Streamlit Chatbot**: Internal UI for natural language review analysis
2. **Database Table**: Populated `disney_review` table with normalized vectors
3. **Summarization Logic**: GPT-4o-mini LangChain prompt pipeline
4. **Cosine Similarity Engine**: In-memory ranking function
5. **Filter Extractor**: LangChain-based parser for country/year/month/season

## Success Metrics

* **Query Precision**: Top-10 review relevance (manual check)
* **Summarization Quality**: Human evaluation of coherence & insight
* **Latency**: Sub-5s response time per query
* **Cache Effectiveness**: Reduction in LLM calls for repeated queries
* **System Integrity**: No missing embeddings or normalization errors

## Implementation Notes

* Embedding stored in two separate fields: raw and normalized
* Query vector normalized in-app before similarity search
* Cosine similarity = dot product of normalized vectors
* All LangChain calls modularized for testability
* Metadata filters automatically extracted from input CSV: all unique values of `year`, `month`, `season`, and `reviewer_location` will be stored as a metadata file during preprocessing
* Streamlit supports dev/debug toggle to inspect filters and matched reviews
* Summary is the only output shown in production mode (no raw reviews)
