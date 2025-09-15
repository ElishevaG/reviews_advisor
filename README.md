# Reviews Advisor - Project Overview

This project contains two main modules for analyzing Disney visitor reviews:

## 📁 Project Structure

```
LLM_Model/
├── 🤖 chatbot_disney_review/        # Natural Language Chatbot Module
├── 📊 insights_extractor/  # Analytics Pipeline Module
├── 📂 data/                         # Shared datasets
├── 📂 reports/                      # Generated reports
└── 📂 other-analyses/               # Additional analysis modules
```

## 🤖 Chatbot Disney Review Module

**Location:** `chatbot_disney_review/`

**Purpose:** Natural language chatbot for querying Disney review insights using vector similarity search and AI summarization.

**Quick Start:**
```bash
cd chatbot_disney_review
python chatbot_runner.py run_indexing_pipeline
python chatbot_runner.py query "What do visitors from Australia say?"
python chatbot_runner.py launch_ui
```

**Documentation:** See `chatbot_disney_review/CHATBOT_README.md`

## 📊 Insights Extractor

**Location:** `insights_extractor/`

**Purpose:** Comprehensive analytics pipeline for extracting visitor pain points and actionable business insights from review data.

**Quick Start:**
```bash
cd insights_extractor
python runner.py run_full_pipeline
```

## 🚀 Getting Started

1. **Choose your module:**
   - For **natural language queries**: Use `chatbot_disney_review/`
   - For **analytics and insights**: Use `insights_extractor/`

2. **Setup environment:**
   ```bash
   # From project root
   source venv/bin/activate
   export OPENAI_API_KEY="your-api-key-here"
   ```

## 🔗 Module Integration

Both modules share:
- **Database:** PostgreSQL `disneyland_reviews` 
- **Data:** `/data/` directory with Disney review datasets
- **Dependencies:** Python virtual environment in `venv`

The chatbot module **reuses components** from the analytics pipeline for efficiency.

**Choose your path and start exploring Disney visitor insights! 🏰** 