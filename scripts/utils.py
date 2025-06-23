import re
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any
import logging

logger = logging.getLogger(__name__)

def extract_clean_text(text: str, remove_headers: bool = False, remove_bullet_points: bool = False) -> str:
    """
    Clean and preprocess text content.
    
    Args:
        text: Raw text content
        remove_headers: Whether to remove header patterns
        remove_bullet_points: Whether to remove bullet points
    
    Returns:
        Cleaned text
    """
    # Basic cleanup
    text = re.sub(r'\s+', ' ', text)
    
    if remove_headers:
        text = re.sub(r'(?i)header:.*?\n', '', text)

    if remove_bullet_points:
        text = re.sub(r'^\s*[-*]\s+', '', text, flags=re.MULTILINE)
    
    return text.strip()

def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 100) -> List[str]:
    """
    Split text into overlapping chunks for analysis.
    
    Args:
        text: Text to chunk
        chunk_size: Size of each chunk
        overlap: Overlap between chunks
    
    Returns:
        List of text chunks
    """
    if len(text) <= chunk_size:
        return [text]
    
    chunks = []
    start = 0
    
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        
        # Try to break at sentence boundary
        if end < len(text):
            last_period = chunk.rfind('.')
            if last_period > chunk_size * 0.7:  # If period is in last 30% of chunk
                chunk = chunk[:last_period + 1]
                end = start + last_period + 1
        
        chunks.append(chunk.strip())
        start = end - overlap
    
    return chunks

def classify_file_type(filename: str, file_types: Dict[str, List[str]]) -> str:
    """
    Classify file based on filename patterns.
    
    Args:
        filename: Name of the file
        file_types: Dictionary mapping file types to keywords
    
    Returns:
        File type classification
    """
    filename_lower = filename.lower()
    
    for file_type, keywords in file_types.items():
        for keyword in keywords:
            if keyword.lower() in filename_lower:
                return file_type
    
    return 'unknown'

def determine_religious_group(filepath: Path, religious_groups: Dict[str, str]) -> str:
    """
    Determine religious group based on file location.
    
    Args:
        filepath: Path to the file
        religious_groups: Dictionary mapping group names to descriptions
    
    Returns:
        Religious group classification
    """
    path_str = str(filepath).lower()
    
    if 'catholic' in path_str:
        return 'catholic'
    elif 'protestant' in path_str:
        return 'protestant'
    elif 'reconciled' in path_str or 'interview' in path_str:
        return 'both'
    else:
        return 'unknown'

def create_metadata_row(filepath: Path, content: str, file_type: str, 
                       religious_group: str, chunk_id: int = None) -> Dict[str, Any]:
    """
    Create a metadata row for the DataFrame.
    
    Args:
        filepath: Path to the file
        content: Text content
        file_type: Type of file
        religious_group: Religious group classification
        chunk_id: Chunk identifier (if text is chunked)
    
    Returns:
        Dictionary with metadata
    """
    return {
        'filename': filepath.name,
        'filepath': str(filepath),
        'file_type': file_type,
        'religious_group': religious_group,
        'content': content,
        'content_length': len(content),
        'word_count': len(content.split()),
        'chunk_id': chunk_id,
        'chunk_count': 1 if chunk_id is None else None
    }

def save_processed_data(data: List[Dict[str, Any]], output_path: Path, filename: str):
    """
    Save processed data to CSV.
    
    Args:
        data: List of data dictionaries
        output_path: Output directory path
        filename: Output filename
    """
    df = pd.DataFrame(data)
    output_file = output_path / filename
    df.to_csv(output_file, index=False, encoding='utf-8')
    logger.info(f"Saved {len(df)} records to {output_file}")
    return df

# --- Textbook pre-processing ---
def preprocess_textbook(text):
    # Remove Table of Contents (from 'CONTENTS' to first 'Part' or 'Unit')
    text = re.sub(r'CONTENTS.*?(?=Part|Unit|Chapter)', '', text, flags=re.DOTALL | re.IGNORECASE)
    # Remove lines that are just numbers (page numbers)
    text = re.sub(r'^\s*\d+\s*$', '', text, flags=re.MULTILINE)
    # Remove lines that look like '1.1 1 Page 4' (unit/page listings)
    text = re.sub(r'^\s*\d+\.\d+\s+\d+\s+Page\s+\d+.*$', '', text, flags=re.MULTILINE)
    # Standardize section headings (add newlines before/after)
    text = re.sub(r'(Part \d+|Unit \d+|Chapter \d+)', r'\n\1\n', text)
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

# --- Policy document pre-processing ---
def preprocess_policy(text):
    # Standardize section headings (add newlines before/after)
    text = re.sub(r'(Unit Overview|Assessment Overview)', r'\n\1\n', text, flags=re.IGNORECASE)
    # Standardize bullet points (replace various bullets with '- ')
    text = re.sub(r'[\u2022\u2023\u25E6\u2043\u2219•]', '-', text)  # common bullet unicode chars
    text = re.sub(r'^\s*-\s*', '- ', text, flags=re.MULTILINE)
    # Remove extra whitespace and blank lines
    text = re.sub(r'\n\s*\n', '\n', text)
    text = re.sub(r'[ \t]+', ' ', text)
    text = text.strip()
    return text

# --- Teacher interview pre-processing ---
def extract_teacher_answers(text, teacher_label="Teacher"):
    # Remove timestamps
    text = re.sub(r'\d+:\d+:\d+\.\d+\s*--?>?\s*\d+:\d+:\d+\.\d+', '', text)
    # Split by speaker
    utterances = re.split(r'(Teacher|Interviewer|I:)', text)
    teacher_text = []
    for i in range(1, len(utterances), 2):
        if utterances[i].strip().lower().startswith(teacher_label.lower()):
            teacher_text.append(utterances[i+1].strip())
    return '\n'.join(teacher_text)

# --- Combined type pre-processing ---
def preprocess_combined(text):
    text = re.sub(r'^\s*(Q|A|Section|Part|Chapter|Unit)[\s\d\.:-]*\n', '', text, flags=re.MULTILINE | re.IGNORECASE)
    text = re.sub(r'\[IMAGE.*?\]', '', text, flags=re.IGNORECASE)
    text = re.sub(r'https?://\S+', '', text) # removing the urls for now, can read in urls as a next step
    text = re.sub(r'\n\s*\n', '\n', text)
    text = re.sub(r'[ \t]+', ' ', text)
    text = text.strip()
    return text