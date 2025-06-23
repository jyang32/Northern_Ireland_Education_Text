# file_reader.py
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any
import logging
logger = logging.getLogger(__name__)
from docx import Document
import PyPDF2
import io
import re

from .config import FILE_TYPES, RELIGIOUS_GROUPS, CHUNK_SIZE, MIN_CHUNK_LENGTH
from .utils import (
    extract_clean_text, chunk_text, classify_file_type, 
    determine_religious_group, create_metadata_row,
    preprocess_textbook, preprocess_policy, extract_teacher_answers, preprocess_combined
)

class FileReader:
    """Class to handle reading and processing different file types."""
    
    def __init__(self, config):
        self.config = config
        self.processed_data = []
    
    def read_docx_file(self, filepath: Path) -> str:
        """Read and extract text from a DOCX file."""
        try:
            doc = Document(filepath)
            text = ""
            for paragraph in doc.paragraphs:
                text += paragraph.text + "\n"
            return text
        except Exception as e:
            logger.error(f"Error reading DOCX file {filepath}: {e}")
            return ""
    
    # a single interface for all file types
    ## expandable to other files types pdf, text, etc.
    ### this is reading a single file
    def read_file(self, filepath: Path) -> str:
        """Read and extract text from a DOCX file (only)."""
        return self.read_docx_file(filepath)
    
    # process a single file
    ## this is the main function that will be used to process a single file
    ## it will return a list of metadata dictionaries
    ## the metadata dictionaries will contain the file path, the file type, the religious group, and the content
    ## the content will be the text of the file
    ## the file type will be the type of the file
    ## the religious group will be the religious group of the file
    
    def process_file(self, filepath: Path, chunk_large_files: bool = True) -> List[Dict[str, Any]]:
        """
        Process a single file and return metadata rows.
        
        Args:
            filepath: Path to the file
            chunk_large_files: Whether to chunk large files
        
        Returns:
            List of metadata dictionaries
        """
        logger.info(f"Processing file: {filepath.name}")
        raw_text = self.read_file(filepath)
        if not raw_text.strip():
            logger.warning(f"No content extracted from {filepath.name}")
            return []
        file_type = classify_file_type(filepath.name, self.config['FILE_TYPES'])
        religious_group = determine_religious_group(filepath, self.config['RELIGIOUS_GROUPS'])
        # Use custom pre-processing for each file type
        if file_type == 'textbook':
            clean_text = preprocess_textbook(raw_text)
        elif file_type == 'policy':
            clean_text = preprocess_policy(raw_text)
        elif file_type == 'interview':
            clean_text = extract_teacher_answers(raw_text)
        elif file_type == 'combined':
            clean_text = preprocess_combined(raw_text)
        else:
            clean_text = extract_clean_text(raw_text)
        # Handle large files by chunking
        if chunk_large_files and len(clean_text) > self.config['CHUNK_SIZE']:
            chunks = chunk_text(clean_text, self.config['CHUNK_SIZE'])
            rows = []
            for i, chunk in enumerate(chunks):
                if len(chunk) >= self.config['MIN_CHUNK_LENGTH']:
                    row = create_metadata_row(
                        filepath=filepath,
                        content=chunk,
                        file_type=file_type,
                        religious_group=religious_group,
                        chunk_id=i
                    )
                    row['chunk_count'] = len(chunks)
                    rows.append(row)
            logger.info(f"Chunked {filepath.name} into {len(rows)} chunks")
            return rows
        else:
            row = create_metadata_row(
                filepath=filepath,
                content=clean_text,
                file_type=file_type,
                religious_group=religious_group
            )
            return [row]
    
    # process a directory
    ## this is the main function that will be used to process all files in a directory
 
    def process_directory(self, directory: Path, chunk_large_files: bool = True) -> List[Dict[str, Any]]:
        """
        Process all files in a directory.
        
        Args:
            directory: Directory to process
            chunk_large_files: Whether to chunk large files
        
        Returns:
            List of all metadata dictionaries
        """
        all_data = []
        supported_extensions = {'.docx', '.pdf', '.txt'}
        
        for filepath in directory.rglob('*'):
            if filepath.is_file() and filepath.suffix.lower() in supported_extensions:
                file_data = self.process_file(filepath, chunk_large_files)
                all_data.extend(file_data)
        
        logger.info(f"Processed {len(all_data)} records from {directory}")
        return all_data
    
    # process all data
    ## this is the main function that will be used to process all data
  
    def process_all_data(self, chunk_large_files: bool = True) -> pd.DataFrame:
        """
        Process all data directories and return a DataFrame.
        
        Args:
            chunk_large_files: Whether to chunk large files
        
        Returns:
            DataFrame with all processed data
        """
        all_data = []
        
        # Process Catholic files
        if self.config['CATHOLIC_DIR'].exists():
            catholic_data = self.process_directory(self.config['CATHOLIC_DIR'], chunk_large_files)
            all_data.extend(catholic_data)
        
        # Process Protestant files
        if self.config['PROTESTANT_DIR'].exists():
            protestant_data = self.process_directory(self.config['PROTESTANT_DIR'], chunk_large_files)
            all_data.extend(protestant_data)
        
        # Process Both files
        if self.config['BOTH_DIR'].exists():
            both_data = self.process_directory(self.config['BOTH_DIR'], chunk_large_files)
            all_data.extend(both_data)
        
        # Create DataFrame
        df = pd.DataFrame(all_data)
        
        if not df.empty:
            logger.info(f"Created DataFrame with {len(df)} records")
            logger.info(f"File types: {df['file_type'].value_counts().to_dict()}")
            logger.info(f"Religious groups: {df['religious_group'].value_counts().to_dict()}")
        else:
            logger.warning("No data was processed")
        
        return df 