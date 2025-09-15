#!/usr/bin/env python3
"""
Setup script for Disney Review Analysis Pipeline
Based on technicall_design_document.md specifications with PostgreSQL integration
"""

import os
import sys
import subprocess
from pathlib import Path

def create_env_file():
    """Create .env file template with PostgreSQL configuration"""
    env_template = """# Disney Review Analysis Pipeline Environment Variables

OPENAI_API_KEY=your-openai-api-key-here
DATABASE_HOST=localhost
DATABASE_PORT=5432
DATABASE_PASSWORD=your-database-password-here

# INPUT_CSV_PATH=DisneylandReviews.csv
# PRIORITY_REPORT_PATH=priority_analysis_report.csv
# EXECUTIVE_SUMMARY_PATH=executive_summary.md
# SOLUTIONS_PATH=solution_recommendations.json

# LLM_MODEL=gpt-4o-mini
# EMBEDDING_MODEL=text-embedding-3-small
# EMBEDDING_DIMENSIONS=512
"""
    
    if not os.path.exists('.env'):
        with open('.env', 'w') as f:
            f.write(env_template)
        print("✅ Created .env file template")
        print("⚠️  Please edit .env file and add your OpenAI API key and database password")
    else:
        print("ℹ️  .env file already exists")

def create_requirements_file():
    """Create requirements.txt with PostgreSQL dependencies"""
    requirements = """# Core dependencies for Disney Hong Kong Review Analysis Pipeline
# Based on technicall_design_document.md specifications with PostgreSQL integration

# Data processing
pandas>=1.5.0
numpy>=1.21.0

# Machine learning
scikit-learn>=1.1.0
hdbscan>=0.8.28

# NLP and text processing
nltk>=3.7

# LangChain for LLM integration
langchain>=0.0.200
openai>=0.27.0

# Database dependencies (PostgreSQL)
psycopg2-binary>=2.9.5
SQLAlchemy>=2.0.0
pgvector>=0.2.0

# Visualization
matplotlib>=3.5.0
seaborn>=0.11.0

# Environment management
python-dotenv>=0.19.0

# Jupyter notebook support (optional)
jupyter>=1.0.0
ipykernel>=6.0.0

# Development dependencies (optional)
pytest>=7.0.0
black>=22.0.0
"""
    
    if not os.path.exists('requirements.txt'):
        with open('requirements.txt', 'w') as f:
            f.write(requirements)
        print("✅ Created requirements.txt with PostgreSQL dependencies")
    else:
        print("ℹ️  requirements.txt already exists")

def setup_virtual_environment():
    """Setup venv virtual environment"""
    print("🐍 Setting up venv virtual environment...")
    
    # Check if virtual environment exists
    venv_path = Path("venv")
    if venv_path.exists():
        print("ℹ️  Virtual environment 'venv' already exists")
    else:
        try:
            subprocess.check_call([sys.executable, "-m", "venv", "venv"])
            print("✅ Created virtual environment 'venv'")
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to create virtual environment: {e}")
            return False
    
    return True

def install_requirements():
    """Install required packages in venv environment"""
    print("📦 Installing required packages in venv environment...")
    
    # Determine the correct pip path based on OS
    if sys.platform == "win32":
        pip_path = Path("venv/Scripts/pip")
    else:
        pip_path = Path("venv/bin/pip")
    
    try:
        subprocess.check_call([str(pip_path), "install", "-r", "requirements.txt"])
        print("✅ Requirements installed successfully in venv environment")
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install requirements: {e}")
        return False
    return True

def create_directories():
    """Create necessary directories"""
    directories = [
        "outputs",
        "logs",
        "visualizations",
        "reports"
    ]
    
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        print(f"✅ Created directory: {directory}")

def download_nltk_data():
    """Download required NLTK data"""
    print("📚 Downloading NLTK data...")
    try:
        import nltk
        nltk.download('punkt', quiet=True)
        print("✅ NLTK data downloaded successfully")
    except Exception as e:
        print(f"⚠️  Warning: Could not download NLTK data: {e}")

def validate_setup():
    """Validate setup"""
    print("🔍 Validating setup...")
    
    # Test imports first
    try:
        from config import Config
        print("✅ Configuration module imported successfully")
    except ImportError as e:
        print(f"❌ Error importing configuration: {e}")
        return False
    
    # Check if Disney reviews file exists using config
    if not os.path.exists(Config.INPUT_CSV_PATH):
        print(f"⚠️  Warning: {Config.INPUT_CSV_PATH} not found")
        print("   Please ensure the Disney reviews dataset is in the current directory")
    else:
        print(f"✅ Disney reviews dataset found: {Config.INPUT_CSV_PATH}")
    
    # Check if .env file has API key
    if os.path.exists('.env'):
        with open('.env', 'r') as f:
            env_content = f.read()
            if 'your-openai-api-key-here' in env_content:
                print("⚠️  Warning: Please update your OpenAI API key in .env file")
            if 'your-database-password-here' in env_content:
                print("⚠️  Warning: Please update your database password in .env file")
            if 'your-openai-api-key-here' not in env_content and 'your-database-password-here' not in env_content:
                print("✅ Environment configuration appears complete")
    
    return True

def check_postgresql():
    """Check PostgreSQL installation and provide setup guidance"""
    print("🐘 Checking PostgreSQL setup...")
    
    try:
        import psycopg2
        print("✅ psycopg2 (PostgreSQL adapter) available")
    except ImportError:
        print("⚠️  psycopg2 not available - will be installed with requirements")
    
    print("\n📋 PostgreSQL Setup Checklist:")
    print("   1. Ensure PostgreSQL 16+ is installed and running")
    print("   2. Create database 'disneyland_reviews':")
    print("      CREATE DATABASE disneyland_reviews;")
    print("   3. Install pgvector extension (if available):")
    print("      CREATE EXTENSION vector;")
    print("   4. Ensure user 'eghezail' has access to the database")
    print("   5. Update DATABASE_PASSWORD in .env file")

def print_usage_instructions():
    """Print usage instructions with PostgreSQL information"""
    instructions = """
╔══════════════════════════════════════════════════════════════════════════════╗
║                              SETUP COMPLETE                                  ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  To run the pipeline:                                                        ║
║                                                                              ║
║  1. Ensure PostgreSQL is running with database 'disneyland_reviews'         ║
║  2. Ensure DisneylandReviews.csv is in the current directory                ║
║  3. Update your OpenAI API key and database password in .env file           ║
║  4. Activate the virtual environment and run the pipeline:                  ║
║                                                                              ║
║     # On macOS/Linux:                                                        ║
║     source venv/bin/activate                                          ║
║     # On Windows:                                                            ║
║     # venv\Scripts\activate                                           ║
║                                                                              ║
║     python runner.py                                                         ║
║                                                                              ║
║  The pipeline will generate all deliverables specified in technicall_design_document.md:           ║
║  - PostgreSQL Database with complete dataset                                 ║
║  - Priority Analysis Report                                                  ║
║  - Solution Recommendations                                                  ║
║  - Visualizations                                                            ║
║  - Executive Summary                                                         ║
║                                                                              ║
║  Database Schema:                                                            ║
║  - Table: pain_point                                                         ║
║  - Individual row writes after normalization and theme generation           ║
║  - Vector embeddings stored for similarity search                           ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""
    print(instructions)

def main():
    """Main setup function"""
    print("🚀 Setting up Disney Hong Kong Review Analysis Pipeline")
    print("   Implementation based on technicall_design_document.md specifications with PostgreSQL")
    print("="*60)
    
    # Create requirements file first
    create_requirements_file()
    
    # Setup virtual environment
    if not setup_virtual_environment():
        print("❌ Setup failed at virtual environment creation")
        return
    
    # Create environment file
    create_env_file()
    
    # Install requirements
    if not install_requirements():
        print("❌ Setup failed during requirements installation")
        return 1
    
    # Create directories
    create_directories()
    
    # Download NLTK data
    download_nltk_data()
    
    # Check PostgreSQL
    check_postgresql()
    
    # Validate setup
    if not validate_setup():
        print("❌ Setup validation failed")
        return 1
    
    # Print usage instructions
    print_usage_instructions()
    
    print("✅ Setup completed successfully!")
    return 0

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code) 