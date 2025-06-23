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

# Output file names
OUTPUT_FILES = {
    'processed_data': 'processed_text_data.csv',
    'analysis_results': 'analysis_results.csv',
    'comparison_results': 'group_comparison_results.csv'
} 