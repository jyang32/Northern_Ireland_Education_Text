# main.py
import sys
from pathlib import Path
import logging
import pandas as pd

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from scripts.config import *
from scripts.file_reader import FileReader
from scripts.utils import save_processed_data, load_metadata, match_metadata_to_data

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("outputs/pipeline.log", mode='w', encoding='utf-8'), # overwrite the file, if append, choose 'a'
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def main():
    """Main data processing pipeline."""
    logger.info("Starting Catholic/Protestant text processing pipeline")
    
    # Create output directory
    OUTPUTS_DIR.mkdir(exist_ok=True)
    
    # Initialize file reader
    config = {
        'FILE_TYPES': FILE_TYPES,
        'RELIGIOUS_GROUPS': RELIGIOUS_GROUPS,
        'CHUNK_SIZE': CHUNK_SIZE,
        'MIN_CHUNK_LENGTH': MIN_CHUNK_LENGTH,
        'CATHOLIC_DIR': CATHOLIC_DIR,
        'PROTESTANT_DIR': PROTESTANT_DIR,
        'BOTH_DIR': BOTH_DIR,
        'FETCH_URLS': True,
        'MAX_URL_CHARS': 8000,
        'USE_OPENAI_FALLBACK': True,
        'OPENAI_MODEL': 'gpt-4o-mini',
        'MAX_AI_SUMMARY_CHARS': 2000,
        'USE_AI_AGENT': True,
        'AI_AGENT_MODEL': 'gpt-4o-mini'
    }
    
    reader = FileReader(config)
    
    # Process all data
    logger.info("Processing all data files...")
    df = reader.process_all_data(chunk_large_files=True)
    
    if df.empty:
        logger.error("No data was processed. Check your file paths and file types.")
        return
    
    # Load and match metadata
    logger.info("Loading metadata and matching to processed data...")
    metadata_path = DATA_DIR / "metadata.csv"
    metadata_df = load_metadata(metadata_path)
    
    if not metadata_df.empty:
        df = match_metadata_to_data(df, metadata_df)
        logger.info("Metadata matching completed")
    else:
        logger.warning("No metadata found, proceeding without metadata matching")
    
    # Save processed data
    logger.info("Saving processed data...")
    processed_data_file = OUTPUTS_DIR / OUTPUT_FILES['processed_data']
    df.to_csv(processed_data_file, index=False, encoding='utf-8')
    logger.info(f"Processed data saved to {processed_data_file}")
    
    # Print summary
    logger.info("\n=== DATA PROCESSING SUMMARY ===")
    logger.info(f"Total records: {len(df)}")
    logger.info(f"Unique files: {df['filename'].nunique()}")
    logger.info(f"File types: {df['file_type'].value_counts().to_dict()}")
    logger.info(f"Religious groups: {df['religious_group'].value_counts().to_dict()}")
    
    # Print metadata matching summary
    if 'publication_policy_year' in df.columns:
        matched_count = df['publication_policy_year'].notna().sum()
        logger.info(f"Records with publication-policy-year: {matched_count} out of {len(df)}")
        if matched_count > 0:
            years = df['publication_policy_year'].dropna().unique()
            logger.info(f"Publication years found: {sorted(years)}")
    
    # Print AI summary information
    if 'has_ai_summary' in df.columns:
        ai_summary_count = df['has_ai_summary'].sum()
        logger.info(f"Records with AI-generated summaries: {ai_summary_count} out of {len(df)}")
    
    # Print detailed breakdown
    logger.info("\n" + "="*50)
    logger.info("DATA PROCESSING COMPLETE")
    logger.info("="*50)
    
    # Print breakdown by religious group
    for group in ['catholic', 'protestant', 'both']:
        group_df = df[df['religious_group'] == group]
        if not group_df.empty:
            logger.info(f"\n{group.upper()}:")
            logger.info(f"  Files: {group_df['filename'].nunique()}")
            logger.info(f"  Chunks: {len(group_df)}")
            logger.info(f"  File types: {group_df['file_type'].value_counts().to_dict()}")
            logger.info(f"  Avg content length: {group_df['content_length'].mean():.0f}")
    
    # Print breakdown by file type
    logger.info(f"\nFILE TYPE BREAKDOWN:")
    for file_type in df['file_type'].unique():
        type_df = df[df['file_type'] == file_type]
        logger.info(f"  {file_type}: {len(type_df)} chunks from {type_df['filename'].nunique()} files")
    
    logger.info(f"\n=== PROCESSING COMPLETE ===")
    logger.info(f"Processed data saved to: {processed_data_file}")
    logger.info("Ready for analysis in the next stage!")

if __name__ == "__main__":
    main() 