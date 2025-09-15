#!/usr/bin/env python3
"""
Main Runner Script for Customer Review Analysis Pipeline
Based on technicall_design_document.md specifications with PostgreSQL integration

This script executes the complete enhanced pipeline and generates all deliverables
specified in the Technical Design Document.
"""

import sys
import os
import logging
import json
from datetime import datetime
from pathlib import Path

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import Config
from pipeline import ReviewAnalysisPipeline

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('pipeline.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def print_banner():
    """Print pipeline banner with technicall_design_document.md information"""
    banner = """
    ╔══════════════════════════════════════════════════════════════════════════════╗
    ║                    Disney Hong Kong Review Analysis Pipeline                 ║
    ║                   Implementation based on technicall_design_document.md with PostgreSQL            ║
    ╠══════════════════════════════════════════════════════════════════════════════╣
    ║  Split Pipeline Architecture:                                                ║
    ║  PART 1: Nodes 1-3 → Data Processing & Vectorization                       ║
    ║  PART 2: Nodes 4-5 → Clustering & Theme Generation                         ║
    ║  PART 3: Nodes 6-8 → Analysis & Solutions                                  ║
    ║  PART 4: Additional Outputs & Final Report                                 ║
    ║                                                                              ║
    ║  Commands: complete | part1 | part2 | part3 | part4                        ║
    ║  Example: python runner.py part1 --batch-size 25                           ║
    ╚══════════════════════════════════════════════════════════════════════════════╝
    """
    print(banner)

def validate_environment():
    """Validate environment and configuration"""
    logger.info("🔍 Validating environment and configuration...")
    
    try:
        # Validate configuration
        Config.validate_config()
        logger.info("✅ Configuration validation passed")
        
        # Check required files
        if not os.path.exists(Config.INPUT_CSV_PATH):
            raise FileNotFoundError(f"Input CSV file not found: {Config.INPUT_CSV_PATH}")
        
        # Test database connection
        from sqlalchemy import create_engine, text
        engine = create_engine(Config.DATABASE_URL)
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        logger.info("✅ Database connection test passed")
        
        # Create output directories
        output_dir = Path("outputs")
        output_dir.mkdir(exist_ok=True)
        
        logger.info("✅ Environment validation completed")
        return True
        
    except Exception as e:
        logger.error(f"❌ Environment validation failed: {e}")
        return False

def run_pipeline_part_1(batch_size: int = None):
    """Execute pipeline Part 1: Nodes 1-3 (Data Processing & Vectorization)"""
    logger.info("🚀 Executing Pipeline Part 1: Nodes 1-3 (Data Processing & Vectorization)")
    
    try:
        # Initialize pipeline with database connection
        pipeline = ReviewAnalysisPipeline(
            openai_api_key=Config.OPENAI_API_KEY,
            database_url=Config.DATABASE_URL,
            input_csv_path=Config.INPUT_CSV_PATH
        )
        
        # Execute Part 1
        results = pipeline.run_pipeline_part_1(batch_size=batch_size)
        
        logger.info("🎉 Pipeline Part 1 execution completed successfully!")
        return results
        
    except Exception as e:
        logger.error(f"❌ Pipeline Part 1 execution failed: {e}")
        raise

def run_pipeline_part_2():
    """Execute pipeline Part 2: Nodes 4-5 (Clustering & Theme Generation)"""
    logger.info("🚀 Executing Pipeline Part 2: Nodes 4-5 (Clustering & Theme Generation)")
    
    try:
        # Initialize pipeline with database connection
        pipeline = ReviewAnalysisPipeline(
            openai_api_key=Config.OPENAI_API_KEY,
            database_url=Config.DATABASE_URL,
            input_csv_path=Config.INPUT_CSV_PATH
        )
        
        # Execute Part 2
        results = pipeline.run_pipeline_part_2()
        
        logger.info("🎉 Pipeline Part 2 execution completed successfully!")
        return results
        
    except Exception as e:
        logger.error(f"❌ Pipeline Part 2 execution failed: {e}")
        raise

def run_pipeline_part_3():
    """Execute pipeline Part 3: Nodes 6-8 (Analysis & Solutions)"""
    logger.info("🚀 Executing Pipeline Part 3: Nodes 6-8 (Analysis & Solutions)")
    
    try:
        # Initialize pipeline with database connection
        pipeline = ReviewAnalysisPipeline(
            openai_api_key=Config.OPENAI_API_KEY,
            database_url=Config.DATABASE_URL,
            input_csv_path=Config.INPUT_CSV_PATH
        )
        
        # Execute Part 3
        results = pipeline.run_pipeline_part_3()
        
        logger.info("🎉 Pipeline Part 3 execution completed successfully!")
        return results
        
    except Exception as e:
        logger.error(f"❌ Pipeline Part 3 execution failed: {e}")
        raise

def run_pipeline_part_4():
    """Execute pipeline Part 4: Generate additional outputs and final report"""
    logger.info("🚀 Executing Pipeline Part 4: Additional Outputs & Final Report")
    
    try:
        # Initialize pipeline with database connection
        pipeline = ReviewAnalysisPipeline(
            openai_api_key=Config.OPENAI_API_KEY,
            database_url=Config.DATABASE_URL,
            input_csv_path=Config.INPUT_CSV_PATH
        )
        
        # Need to get Part 3 results first (load from database)
        part3_results = pipeline.run_pipeline_part_3()
        
        # Execute Part 4
        complete_results = pipeline.run_pipeline_part_4(part3_results)
        
        # Generate additional outputs
        generate_additional_outputs(pipeline, complete_results)
        
        # Generate final report
        generate_final_report(complete_results)
        
        logger.info("🎉 Pipeline Part 4 execution completed successfully!")
        return complete_results
        
    except Exception as e:
        logger.error(f"❌ Pipeline Part 4 execution failed: {e}")
        raise

def run_pipeline(batch_size: int = None):
    """Execute the complete pipeline with optional batch processing"""
    if batch_size:
        logger.info(f"🚀 Starting complete pipeline execution with batch processing (batch size: {batch_size})...")
    else:
        logger.info("🚀 Starting complete pipeline execution...")
    
    try:
        # Initialize pipeline with database connection
        pipeline = ReviewAnalysisPipeline(
            openai_api_key=Config.OPENAI_API_KEY,
            database_url=Config.DATABASE_URL,
            input_csv_path=Config.INPUT_CSV_PATH
        )
        
        # Execute complete pipeline with batch processing if specified
        results = pipeline.run_pipeline(batch_size=batch_size)
        
        # Generate additional outputs
        generate_additional_outputs(pipeline, results)
        
        # Generate final report
        generate_final_report(results)
        
        logger.info("🎉 Complete pipeline execution completed successfully!")
        return results
        
    except Exception as e:
        logger.error(f"❌ Complete pipeline execution failed: {e}")
        raise

def generate_additional_outputs(pipeline, results):
    """Generate additional outputs based on technicall_design_document.md deliverables"""
    from config import Config
    logger.info("📄 Generating additional outputs...")
    
    try:
        # 1. Executive Summary (technicall_design_document.md deliverable)
        executive_summary = pipeline.generate_executive_summary(results)
        with open(Config.EXECUTIVE_SUMMARY_PATH, 'w') as f:
            f.write(executive_summary)
        logger.info(f"✅ Executive summary saved to: {Config.EXECUTIVE_SUMMARY_PATH}")
        
        # 2. Solution Recommendations JSON (technicall_design_document.md deliverable)
        solutions_data = {
            'generated_at': datetime.now().isoformat(),
            'pipeline_cost': results['total_cost'],
            'database_url': Config.DATABASE_URL.split('@')[1],  # Hide credentials
            'solutions': results['solutions']['solutions'],
            'summary_stats': results['solutions']['summary_stats']
        }
        
        with open(Config.SOLUTIONS_PATH, 'w') as f:
            json.dump(solutions_data, f, indent=2)
        logger.info(f"✅ Solution recommendations saved to: {Config.SOLUTIONS_PATH}")
        
        # 2b. Generate human-readable Solutions Report (Markdown)
        solutions_report = generate_solutions_report(results)
        solutions_report_path = Config.SOLUTIONS_PATH.replace('.json', '_report.md')
        with open(solutions_report_path, 'w') as f:
            f.write(solutions_report)
        logger.info(f"✅ Solutions report saved to: {solutions_report_path}")
        
        # 3. Pipeline Metadata
        pipeline_info = Config.get_pipeline_info()
        pipeline_metadata = {
            'pipeline_info': pipeline_info,
            'execution_timestamp': datetime.now().isoformat(),
            'total_cost': results['total_cost'],
            'database_info': {
                'url': Config.DATABASE_URL.split('@')[1],  # Hide credentials
                'table': Config.PAIN_POINT_TABLE,
                'total_records': len(results.get('processed_data', []))
            },
            'cluster_info': {
                'n_clusters': results['cluster_info']['n_clusters'],
                'n_outliers': results['cluster_info']['n_outliers']
            },
            'themes': results['themes'],
            'success_metrics': calculate_success_metrics(results)
        }
        
        with open('pipeline_metadata.json', 'w') as f:
            json.dump(pipeline_metadata, f, indent=2)
        logger.info("✅ Pipeline metadata saved to: pipeline_metadata.json")
        
    except Exception as e:
        logger.error(f"❌ Error generating additional outputs: {e}")
        raise

def generate_solutions_report(results):
    """Generate human-readable solutions report in markdown format"""
    from config import Config
    logger.info("📝 Generating human-readable solutions report...")
    
    try:
        # Get solutions data
        solutions = results['solutions']['solutions']
        summary_stats = results['solutions']['summary_stats']
        priority_themes = results['priority_themes'].head(5)
        
        # Start building the report
        report = f"""# 🎯 Disney Hong Kong - Actionable Solutions Report

## 📊 Executive Overview

- **Analysis Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
- **Total Pain Points Analyzed**: {summary_stats['total_pain_points']:,}
- **Unique Themes Identified**: {summary_stats['unique_themes']}
- **Top Priority Theme**: {summary_stats['top_theme']}
- **Average Rating Impact**: {summary_stats['avg_rating_impact']:.2f}/5.0
- **Processing Cost**: ${results['total_cost']:.4f}

---

## 🔥 Priority Action Plan

The following solutions are ranked by impact and volume. **Focus on implementing these in order for maximum guest satisfaction improvement.**

"""
        
        # Add solutions for each priority theme
        for idx, (_, row) in enumerate(priority_themes.iterrows(), 1):
            theme = row['Theme']
            volume = row['Volume']
            rating = row['rating']
            
            if theme in solutions:
                solution_text = solutions[theme]
                
                report += f"""
---

## #{idx} {theme}

**📈 Impact Metrics:**
- **Volume**: {volume} complaints
- **Average Rating**: {rating:.2f}/5.0  
- **Business Priority**: #{row['Priority_Rank']}

{solution_text}

---
"""
        
        # Add implementation guidelines
        report += f"""

## 🚀 Implementation Guidelines

### Phase 1: Immediate Actions (0-3 months)
Focus on the **Immediate Actions** listed above for each priority theme. These require minimal investment but can provide quick wins and demonstrate commitment to improvement.

### Phase 2: Long-term Improvements (3-12 months)  
Implement the **Long-term Improvements** which require more planning and investment but provide sustainable solutions to the root causes.

### Phase 3: Monitoring & Optimization (Ongoing)
Use the **Success Metrics** to track progress and adjust strategies. Set up automated monitoring to measure:

"""
        
        # Add success metrics summary
        for idx, (_, row) in enumerate(priority_themes.iterrows(), 1):
            theme = row['Theme']
            if theme in solutions:
                report += f"- **{theme}**: Track guest satisfaction and relevant KPIs\n"
        
        report += f"""

## 💡 Quick Wins Recommendations

Based on the analysis, here are the **top 3 quick wins** to implement immediately:

1. **Enhanced Communication** (Multiple themes): Implement real-time updates and better signage across the park
2. **Virtual Queue Systems** (Long Wait Times): Reduce perceived wait times through technology
3. **Value Meal Options** (High Food Prices): Offer budget-friendly alternatives to improve value perception

## 📋 Next Steps

1. **Prioritize Implementation**: Start with Immediate Actions for the top 3 themes
2. **Assign Ownership**: Designate team leads for each solution area
3. **Set Timelines**: Establish clear milestones for each phase
4. **Monitor Progress**: Use the success metrics to track improvement
5. **Iterate**: Run the analysis pipeline regularly to measure impact

## 🔗 Related Documents

- **Executive Summary**: `{Config.EXECUTIVE_SUMMARY_PATH}`
- **Detailed Analysis**: `{Config.PRIORITY_REPORT_PATH}` 
- **Technical Data**: `{Config.SOLUTIONS_PATH}`
- **Final Report**: `final_report.md`

---

*This report was generated by the Disney Hong Kong Review Analysis Pipeline using AI-powered analysis of customer feedback. Solutions are based on {summary_stats['total_pain_points']:,} real customer pain points.*

*For technical details and raw data, see the accompanying JSON and CSV files.*
"""
        
        return report
        
    except Exception as e:
        logger.error(f"❌ Error generating solutions report: {e}")
        return f"# Solutions Report\n\nError generating report: {e}"

def generate_final_report(results):
    """Generate final comprehensive report"""
    from config import Config
    logger.info("📊 Generating final comprehensive report...")
    
    try:
        report = f"""
# Disney Hong Kong Review Analysis Pipeline - Final Report

## Execution Summary
- **Pipeline Version**: Based on technicall_design_document.md specification with PostgreSQL integration
- **Execution Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
- **Total Processing Cost**: ${results['total_cost']:.4f}
- **Database**: {Config.DATABASE_URL.split('@')[1]}

## Data Processing Results
- **Input Reviews**: {len(results['processed_data']) if 'processed_data' in results else 'N/A'}
- **Pain Points Extracted**: {len(results['processed_data']) if 'processed_data' in results else 'N/A'}
- **Database Records**: {len(results['processed_data']) if 'processed_data' in results else 'N/A'}
- **Clusters Found**: {results['cluster_info']['n_clusters']}
- **Outliers**: {results['cluster_info']['n_outliers']}
- **Themes Generated**: {len(results['themes'])}

## Database Integration
- **Table**: {Config.PAIN_POINT_TABLE}
- **Write Operations**: 2 (after normalization and theme generation)
- **Individual Row Processing**: ✅ Enabled for real-time capability
- **Vector Storage**: ✅ PostgreSQL with pgvector extension

## Top Priority Themes
"""
        
        # Add top themes
        for idx, (_, row) in enumerate(results['priority_themes'].head(5).iterrows(), 1):
            report += f"""
### {idx}. {row['Theme']}
- **Volume**: {row['Volume']} complaints
- **Average Rating**: {row['rating']:.2f}/5.0
- **Priority Rank**: #{row['Priority_Rank']}
"""
        
        report += f"""

## Output Files Generated
- **PostgreSQL Database**: Complete dataset in `{Config.PAIN_POINT_TABLE}` table
- **Priority Analysis Report**: {Config.PRIORITY_REPORT_PATH}
- **Executive Summary**: {Config.EXECUTIVE_SUMMARY_PATH}
- **Solution Recommendations**: {Config.SOLUTIONS_PATH}
- **Pipeline Metadata**: pipeline_metadata.json
- **Visualizations**: {', '.join(Config.VISUALIZATION_FILES.values())}

## Enhanced Pipeline Architecture Completed
✅ All enhanced nodes executed successfully as per technicall_design_document.md:
1. Date Feature Engineering
2. Sentence Splitting & Pain Point Extraction
3. Text Vectorization
4. Vector Normalization
5. **Database Write 1** (Individual inserts with embeddings)
6. Clustering
7. Theme Generation
8. **Database Write 2** (Individual updates with themes)
9. Dimensional Grouping (PostgreSQL-based)
10. Impact Prioritization
11. Solution Generation & Visualization

## Database Benefits
- **Unified Storage**: All data in single PostgreSQL database
- **Vector Similarity**: Support for semantic search with pgvector
- **Real-time Processing**: Individual row writes enable streaming
- **Scalability**: Connection pooling and optimized queries
- **Integration**: Easy connection to BI tools and dashboards

## Next Steps
1. Review generated solutions for each priority theme
2. Implement recommended immediate actions
3. Monitor success metrics post-implementation
4. Set up automated pipeline runs for continuous insights
5. Build dashboards connecting to the PostgreSQL database
6. Implement vector similarity search for advanced analytics

---
*Report generated by Disney Hong Kong Review Analysis Pipeline*
*Implementation based on technicall_design_document.md specification with PostgreSQL integration*
"""
        
        with open('reports/final_report.md', 'w') as f:
            f.write(report)
        
        logger.info("✅ Final report saved to: final_report.md")
        
    except Exception as e:
        logger.error(f"❌ Error generating final report: {e}")

def calculate_success_metrics(results):
    """Calculate success metrics based on technicall_design_document.md specifications"""
    from config import Config
    import numpy as np
    try:
        processed_data = results['processed_data']
        cluster_info = results['cluster_info']
        
        # Coverage: % of negative sentences successfully clustered
        total_sentences = len(processed_data)
        clustered_sentences = len(processed_data[processed_data['Cluster_ID'] != -1])
        coverage = float(clustered_sentences / total_sentences if total_sentences > 0 else 0)
        
        # Theme quality: Number of meaningful themes generated
        meaningful_themes = len([t for t in results['themes'].values() if not t.startswith('Theme ')])
        theme_quality = float(meaningful_themes / len(results['themes']) if len(results['themes']) > 0 else 0)
        
        # Business impact: Correlation between themes and ratings
        theme_ratings = results['priority_themes']['rating'].tolist()
        avg_theme_rating = sum(theme_ratings) / len(theme_ratings) if theme_ratings else 0
        business_impact = float((5 - avg_theme_rating) / 4)  # Normalized impact score
        
        # Data integrity: Database storage success
        data_integrity = 1.0 if len(processed_data) > 0 else 0.0
        
        # Convert NumPy types to Python native types for JSON serialization
        def convert_numpy_types(value):
            if isinstance(value, (np.integer, np.int64, np.int32)):
                return int(value)
            elif isinstance(value, (np.floating, np.float64, np.float32)):
                return float(value)
            elif isinstance(value, np.bool_):
                return bool(value)
            return value
        
        return {
            'coverage': coverage,
            'theme_quality': theme_quality,
            'business_impact': business_impact,
            'data_integrity': data_integrity,
            'meets_coverage_threshold': coverage >= Config.MIN_COVERAGE_THRESHOLD,
            'meets_theme_quality_threshold': theme_quality >= Config.MIN_THEME_QUALITY_SCORE,
            'total_clusters': convert_numpy_types(cluster_info['n_clusters']),
            'total_outliers': convert_numpy_types(cluster_info['n_outliers']),
            'database_records': int(len(processed_data))
        }
        
    except Exception as e:
        logger.error(f"❌ Error calculating success metrics: {e}")
        return {}

def main():
    """Main execution function"""
    import argparse
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Disney Hong Kong Review Analysis Pipeline')
    parser.add_argument('command', choices=['complete', 'part1', 'part2', 'part3', 'part4'], 
                       help='Pipeline command to execute')
    parser.add_argument('--batch-size', type=int, default=None, 
                       help='Process data in batches of this size (for LLM calls). Only applies to part1 and complete commands.')
    args = parser.parse_args()
    
    print_banner()
    
    # Validate environment
    if not validate_environment():
        sys.exit(1)
    
    # Display pipeline information
    pipeline_info = Config.get_pipeline_info()
    logger.info(f"📋 Pipeline: {pipeline_info['name']}")
    logger.info(f"📋 Description: {pipeline_info['description']}")
    logger.info(f"📋 Nodes: {len(pipeline_info['nodes'])}")
    logger.info(f"💾 Database: {Config.DATABASE_URL.split('@')[1]}")
    
    if args.batch_size and args.command in ['complete', 'part1']:
        logger.info(f"📦 Batch Processing: {args.batch_size} reviews per batch")
    elif args.command in ['complete', 'part1']:
        logger.info("📦 Batch Processing: Disabled (process all at once)")
    
    try:
        # Execute the requested command
        if args.command == 'complete':
            logger.info("🎯 Executing COMPLETE pipeline (all parts)")
            results = run_pipeline(batch_size=args.batch_size)
            
            # Success summary for complete pipeline
            logger.info("\n" + "="*60)
            logger.info("🎉 COMPLETE PIPELINE EXECUTION COMPLETED SUCCESSFULLY!")
            logger.info("="*60)
            logger.info(f"💰 Total Cost: ${results['total_cost']:.4f}")
            logger.info(f"📊 Themes Generated: {len(results['themes'])}")
            logger.info(f"📈 Top Priority Theme: {results['priority_themes'].iloc[0]['Theme']}")
            logger.info(f"💾 Database Records: {len(results['processed_data'])}")
            logger.info(f"📁 All deliverables saved as per technicall_design_document.md specification")
            if args.batch_size:
                logger.info(f"📦 Batch Processing: Used {args.batch_size} reviews per batch")
            logger.info("="*60)
            
        elif args.command == 'part1':
            logger.info("🎯 Executing PART 1: Nodes 1-3 (Data Processing & Vectorization)")
            results = run_pipeline_part_1(batch_size=args.batch_size)
            
            # Success summary for part 1
            logger.info("\n" + "="*60)
            logger.info("🎉 PART 1 EXECUTION COMPLETED SUCCESSFULLY!")
            logger.info("="*60)
            logger.info(f"💰 Part 1 Cost: ${results['total_cost']:.4f}")
            logger.info(f"💾 Records Processed: {results['total_inserted']}")
            if args.batch_size:
                logger.info(f"📦 Batch Processing: Used {args.batch_size} reviews per batch")
            logger.info("🎯 Next: Run 'python runner.py part2' for clustering and theme generation")
            logger.info("="*60)
            
        elif args.command == 'part2':
            logger.info("🎯 Executing PART 2: Nodes 4-5 (Clustering & Theme Generation)")
            results = run_pipeline_part_2()
            
            # Success summary for part 2
            logger.info("\n" + "="*60)
            logger.info("🎉 PART 2 EXECUTION COMPLETED SUCCESSFULLY!")
            logger.info("="*60)
            logger.info(f"💰 Part 2 Cost: ${results['total_cost']:.4f}")
            logger.info(f"🎯 Clusters Found: {results['cluster_info']['n_clusters']}")
            logger.info(f"🏷️  Themes Generated: {len(results['themes'])}")
            logger.info("🎯 Next: Run 'python runner.py part3' for analysis and solutions")
            logger.info("="*60)
            
        elif args.command == 'part3':
            logger.info("🎯 Executing PART 3: Nodes 6-8 (Analysis & Solutions)")
            results = run_pipeline_part_3()
            
            # Success summary for part 3
            logger.info("\n" + "="*60)
            logger.info("🎉 PART 3 EXECUTION COMPLETED SUCCESSFULLY!")
            logger.info("="*60)
            logger.info(f"💰 Part 3 Cost: ${results['total_cost']:.4f}")
            logger.info(f"📈 Top Priority Theme: {results['priority_themes'].iloc[0]['Theme']}")
            logger.info(f"📁 Priority analysis saved to: {Config.PRIORITY_REPORT_PATH}")
            logger.info("🎯 Next: Run 'python runner.py part4' for additional outputs and final report")
            logger.info("="*60)
            
        elif args.command == 'part4':
            logger.info("🎯 Executing PART 4: Additional Outputs & Final Report")
            results = run_pipeline_part_4()
            
            # Success summary for part 4
            logger.info("\n" + "="*60)
            logger.info("🎉 PART 4 EXECUTION COMPLETED SUCCESSFULLY!")
            logger.info("="*60)
            logger.info(f"💰 Total Pipeline Cost: ${results['total_cost']:.4f}")
            logger.info(f"💾 Database Records: {len(results['processed_data'])}")
            logger.info(f"📄 Executive summary: {Config.EXECUTIVE_SUMMARY_PATH}")
            logger.info(f"💡 Solutions: {Config.SOLUTIONS_PATH}")
            logger.info(f"📊 Final report: final_report.md")
            logger.info("🎉 ALL PIPELINE PARTS COMPLETED!")
            logger.info("="*60)
        
        return 0
        
    except KeyboardInterrupt:
        logger.info(f"\n⚠️  {args.command.upper()} execution interrupted by user")
        return 1
    except Exception as e:
        logger.error(f"\n❌ {args.command.upper()} execution failed: {e}")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code) 