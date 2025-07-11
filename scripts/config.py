# config.py
import os
from pathlib import Path

# Base paths
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
OUTPUTS_DIR = BASE_DIR / "outputs"

# Data structure paths
STRAND1_DIR = DATA_DIR / "strand1"
CATHOLIC_DIR = STRAND1_DIR / "catholic"
PROTESTANT_DIR = STRAND1_DIR / "protestant"
BOTH_DIR = STRAND1_DIR / "both"

# File type mappings
FILE_TYPES = {
    'textbook': ['Madden', 'Doherty', 'Johnston'],
    'policy': ['Planning Framework', 'specification'],
    'combined': ['combined all'],
    'interview': ['Teacher']
}

# Content categories
CONTENT_CATEGORIES = {
    'textbook': 'Educational textbook content',
    'policy': 'Policy and planning documents', 
    'combined': 'Combined resource materials',
    'interview': 'Teacher interview transcripts'
}

# Religious group mappings
RELIGIOUS_GROUPS = {
    'catholic': 'Catholic perspective',
    'protestant': 'Protestant perspective', 
    'both': 'Shared/neutral perspective'
}

# Analysis parameters
CHUNK_SIZE = 1000  # For text chunking
MIN_CHUNK_LENGTH = 100  # Minimum chunk length
MAX_CHUNK_LENGTH = 2000  # Maximum chunk length

# URL processing parameters
FETCH_URLS = True  # Whether to fetch content from URLs in combined documents
MAX_URL_CHARS = 8000  # Maximum characters to extract from each URL
URL_TIMEOUT = 15  # Timeout for URL requests in seconds

# OpenAI fallback parameters
USE_OPENAI_FALLBACK = True  # Whether to use OpenAI when URL fetching fails
OPENAI_MODEL = "gpt-4o-mini"  # OpenAI model to use for summarization
# OpenAI API key will be loaded from .env file or environment variable
MAX_AI_SUMMARY_CHARS = 2000  # Maximum characters for AI-generated summaries

# AI Agent parameters removed - using simpler OpenAI approach

# Output file names
OUTPUT_FILES = {
    'processed_data': 'processed_text_data.csv',
    'analysis_results': 'analysis_results.csv',
    'comparison_results': 'group_comparison_results.csv'
} 