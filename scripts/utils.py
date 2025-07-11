import re
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any
import logging
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
import time
import os

# Optional imports for environment variables
try:
    from dotenv import load_dotenv
    load_dotenv()  # Load environment variables from .env file
    DOTENV_AVAILABLE = True
except ImportError:
    DOTENV_AVAILABLE = False
    logging.warning("python-dotenv not available. Install with: pip install python-dotenv")

# Optional OpenAI import
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    logging.warning("OpenAI library not available. Install with: pip install openai")

# LangChain imports removed - using simpler OpenAI approach
LANGCHAIN_AVAILABLE = False

logger = logging.getLogger(__name__)

# Headers for web scraping to avoid being blocked
HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; Northern_Ireland_Education_Analysis/1.0)"}

def fetch_url_content_with_ai_fallback(url: str, max_chars: int = 8000, timeout: int = 15, 
                                      use_ai_fallback: bool = True, openai_model: str = "gpt-4o-mini",
                                      max_ai_chars: int = 2000) -> tuple[str, str]:
    """
    Fetch content from a URL with OpenAI fallback for failed requests.
    
    Args:
        url: URL to fetch
        max_chars: Maximum characters to extract from raw content
        timeout: Request timeout in seconds
        use_ai_fallback: Whether to use OpenAI when raw fetching fails
        openai_model: OpenAI model to use for summarization
        max_ai_chars: Maximum characters for AI-generated summary
    
    Returns:
        Tuple of (content, content_type) where content_type is 'raw' or 'ai_summary'
    """
    # First try to fetch raw content
    try:
        logger.info(f"Fetching raw content from: {url}")
        resp = requests.get(url, headers=HEADERS, timeout=timeout)
        resp.raise_for_status()
        
        soup = BeautifulSoup(resp.text, "html.parser")
        
        # Remove script and style elements
        for script in soup(["script", "style"]):
            script.decompose()
        
        # Get text content
        text = soup.get_text(" ", strip=True)
        
        # Clean up the text
        text = re.sub(r'\s+', ' ', text)
        text = text[:max_chars]
        
        logger.info(f"Successfully extracted {len(text)} characters from {url}")
        return text, 'raw'
        
    except Exception as exc:
        error_msg = f"[Error fetching {url}: {exc}]"
        logger.warning(error_msg)
        
        # Try OpenAI fallback if enabled and available
        if use_ai_fallback and OPENAI_AVAILABLE:
            try:
                logger.info(f"Attempting OpenAI fallback for: {url}")
                ai_summary = fetch_with_ai_agent(url)
                if ai_summary and not ai_summary.startswith('[AI Error'):
                    logger.info(f"Successfully generated AI summary for {url}")
                    return ai_summary, 'ai_summary'
            except Exception as ai_exc:
                logger.error(f"OpenAI fallback also failed for {url}: {ai_exc}")
        
        # Return error message if both methods fail
        return error_msg, 'error'





def fetch_with_ai_agent(url: str) -> str:
    """
    Use AI to generate a summary based on knowledge about the URL/domain.
    This is used when raw URL fetching fails.
    
    Args:
        url: URL to generate knowledge-based summary for
    
    Returns:
        AI-generated summary based on training data
    """
    try:
        logger.info(f"Using AI to generate knowledge-based summary for: {url}")
        
        # Get API key
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            return f"[AI Error: No API key available for {url}]"
        
        # Use OpenAI directly for knowledge-based summary
        client = OpenAI(api_key=api_key)
        
        prompt = f"""
        Based on your training data, provide information about this URL: {url}
        
        Please provide a summary that includes:
        - What this website/domain typically contains
        - General information about the topic
        - Relevance to Northern Ireland education or history (if applicable)
        - Educational value and potential content
        
        Keep the summary under 2000 characters and focus on educational content.
        Note: This is based on general knowledge, not live website content.
        """
        
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that provides information based on your training data."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=1000,
            temperature=0.3
        )
        
        summary = response.choices[0].message.content.strip()
        return f"[AI-GENERATED SUMMARY FROM KNOWLEDGE BASE] {summary}"
        
    except Exception as exc:
        logger.error(f"Knowledge-based summarization failed for {url}: {exc}")
        return f"[AI Error: {exc} for {url}]"



def fetch_url_content(url: str, max_chars: int = 8000, timeout: int = 15) -> str:
    """
    Fetch content from a URL and return cleaned text (legacy function for backward compatibility).
    
    Args:
        url: URL to fetch
        max_chars: Maximum characters to extract
        timeout: Request timeout in seconds
    
    Returns:
        Extracted text content or error message
    """
    content, content_type = fetch_url_content_with_ai_fallback(url, max_chars, timeout, use_ai_fallback=False)
    return content

def extract_urls_from_text(text: str) -> List[str]:
    """
    Extract URLs from text using regex.
    
    Args:
        text: Text to search for URLs
    
    Returns:
        List of found URLs
    """
    url_pattern = r'https?://[^\s<>"{}|\\^`\[\]]+'
    urls = re.findall(url_pattern, text)
    return urls

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
    elif 'both' in path_str or 'reconciled' in path_str or 'interview' in path_str:
        return 'both'
    else:
        return 'unknown'

def create_metadata_row(filepath: Path, content: str, file_type: str, 
                       religious_group: str, chunk_id: int = None, 
                       has_url_content: bool = False, has_ai_summary: bool = False) -> Dict[str, Any]:
    """
    Create a metadata row for the DataFrame.
    
    Args:
        filepath: Path to the file
        content: Text content
        file_type: Type of file
        religious_group: Religious group classification
        chunk_id: Chunk identifier (if text is chunked)
        has_url_content: Whether the content includes fetched URL content
        has_ai_summary: Whether the content includes AI-generated summaries
    
    Returns:
        Dictionary with metadata
    """
    return {
        'filename': filepath.name,
        # 'filepath': str(filepath),
        'file_type': file_type,
        'religious_group': religious_group,
        'content': content,
        'content_length': len(content),
        'word_count': len(content.split()),
        'chunk_id': chunk_id,
        'chunk_count': 1 if chunk_id is None else None,
        'has_url_content': has_url_content,
        'has_ai_summary': has_ai_summary
    }

def load_metadata(metadata_path: Path) -> pd.DataFrame:
    """
    Load metadata from CSV file.
    
    Args:
        metadata_path: Path to metadata.csv file
    
    Returns:
        DataFrame with metadata
    """
    try:
        metadata_df = pd.read_csv(metadata_path)
        logger.info(f"Loaded metadata with {len(metadata_df)} records")
        return metadata_df
    except Exception as e:
        logger.error(f"Error loading metadata from {metadata_path}: {e}")
        return pd.DataFrame()

def match_metadata_to_data(processed_df: pd.DataFrame, metadata_df: pd.DataFrame) -> pd.DataFrame:
    """
    Match metadata information to processed data based on filename.
    
    Args:
        processed_df: DataFrame with processed text data
        metadata_df: DataFrame with metadata information
    
    Returns:
        DataFrame with publication-policy-year column added
    """
    if metadata_df.empty:
        logger.warning("No metadata available for matching")
        return processed_df
    
    # Create a copy to avoid modifying the original
    result_df = processed_df.copy()
    
    # Add publication-policy-year column with default None
    result_df['publication_policy_year'] = None
    
    # Match based on filename
    matched_count = 0
    for idx, row in result_df.iterrows():
        filename = row['filename']
        
        # Find matching row in metadata
        metadata_match = metadata_df[metadata_df['strand-one-filename'] == filename]
        
        if not metadata_match.empty:
            # Get the publication-policy-year value
            pub_year = metadata_match.iloc[0]['publication-policy-year']
            if pd.notna(pub_year) and pub_year != '':
                result_df.at[idx, 'publication_policy_year'] = pub_year
                matched_count += 1
    
    logger.info(f"Matched publication-policy-year for {matched_count} out of {len(result_df)} records")
    
    return result_df

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
def preprocess_combined(text, fetch_urls: bool = True, max_url_chars: int = 8000, 
                       use_ai_fallback: bool = True, openai_model: str = "gpt-4o-mini",
                       max_ai_chars: int = 2000):
    """
    Preprocess combined documents with optional URL content fetching and AI fallback.
    
    Args:
        text: Text to preprocess
        fetch_urls: Whether to fetch content from URLs found in text
        max_url_chars: Maximum characters to extract from each URL
        use_ai_fallback: Whether to use OpenAI when URL fetching fails
        openai_model: OpenAI model to use for summarization
        max_ai_chars: Maximum characters for AI-generated summaries
    
    Returns:
        Tuple of (preprocessed_text, url_contents_dict, has_ai_summary)
    """
    # Extract URLs before removing them
    all_urls = extract_urls_from_text(text) if fetch_urls else []
    
    # Get unique URLs to avoid repeated fetching
    unique_urls = list(set(all_urls))
    
    # Basic preprocessing
    text = re.sub(r'^\s*(Q|A|Section|Part|Chapter|Unit)[\s\d\.:-]*\n', '', text, flags=re.MULTILINE | re.IGNORECASE)
    text = re.sub(r'\[IMAGE.*?\]', '', text, flags=re.IGNORECASE)
    text = re.sub(r'https?://\S+', '', text)  # Remove URLs from original text
    text = re.sub(r'\n\s*\n', '\n', text)
    text = re.sub(r'[ \t]+', ' ', text)
    text = text.strip()
    
    # Store URL contents separately for per-chunk analysis
    url_contents_dict = {}
    has_ai_summary = False
    
    # Fetch URL content if requested
    if fetch_urls and unique_urls:
        logger.info(f"Found {len(all_urls)} total URLs, {len(unique_urls)} unique URLs in combined document")
        
        for i, url in enumerate(unique_urls):
            logger.info(f"Processing unique URL {i+1}/{len(unique_urls)}: {url}")
            content, content_type = fetch_url_content_with_ai_fallback(
                url, max_chars=max_url_chars, use_ai_fallback=use_ai_fallback,
                openai_model=openai_model, max_ai_chars=max_ai_chars
            )
            
            url_contents_dict[url] = content
            
            # Track if any AI summaries were generated
            if content_type == 'ai_summary':
                has_ai_summary = True
            
            # Add a small delay to be respectful to servers
            time.sleep(1)
        
        # Append URL contents to the text with markers
        if url_contents_dict:
            url_sections = []
            for i, (url, content) in enumerate(url_contents_dict.items(), 1):
                if content.startswith('[AI-GENERATED SUMMARY]'):
                    url_sections.append(f"\n\n--- AI SUMMARY {i}: {url} ---\n{content}")
                elif not content.startswith('[Error'):
                    url_sections.append(f"\n\n--- URL Content {i}: {url} ---\n{content}")
                else:
                    url_sections.append(f"\n\n--- URL Error {i}: {url} ---\n{content}")
            
            text += "\n\n" + "\n".join(url_sections)
    
    return text, url_contents_dict, has_ai_summary

def check_chunk_has_url_content(chunk_text: str, url_contents_dict: dict) -> bool:
    """
    Check if a chunk contains URL content by looking for URL markers.
    
    Args:
        chunk_text: The text chunk to check
        url_contents_dict: Dictionary of URL contents
    
    Returns:
        True if chunk contains URL content, False otherwise
    """
    # Look for URL content markers in the chunk (both raw content and AI summaries)
    url_markers = [
        "--- URL Content",
        "--- AI SUMMARY", 
        "[AI-GENERATED SUMMARY FROM KNOWLEDGE BASE]",
        "[AI-GENERATED SUMMARY FROM LIVE CONTENT]"
    ]
    
    for url in url_contents_dict.keys():
        for marker in url_markers:
            if marker in chunk_text and url in chunk_text:
                return True
    return False

def check_chunk_has_ai_summary(chunk_text: str) -> bool:
    """
    Check if a chunk contains AI summary content by looking for AI summary markers.
    
    Args:
        chunk_text: The text chunk to check
    
    Returns:
        True if chunk contains AI summary content, False otherwise
    """
    # Look for AI summary markers in the chunk
    ai_markers = [
        "--- AI SUMMARY",
        "[AI-GENERATED SUMMARY FROM KNOWLEDGE BASE]",
        "[AI-GENERATED SUMMARY FROM LIVE CONTENT]"
    ]
    
    for marker in ai_markers:
        if marker in chunk_text:
            return True
    return False