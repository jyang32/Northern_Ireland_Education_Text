import os
from docx import Document
import re

def read_docx(path):
    doc = Document(path)
    return "\n".join([p.text for p in doc.paragraphs])

def extract_clean_text(filepath, remove_headers=False, remove_bullet_points=False):
    ext = os.path.splitext(filepath)[1].lower()
    if ext == '.docx':
        text = read_docx(filepath)
    elif ext == '.txt':
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            text = f.read()
    else:
        return None  # Skip unsupported types for now

    text = re.sub(r'\s+', ' ', text)
    if remove_headers:
        text = re.sub(r'(?i)header:.*?\n', '', text)
    if remove_bullet_points:
        text = re.sub(r'^\s*[-*]\s+', '', text, flags=re.MULTILINE)
    return text.strip()

def read_and_process_file(row):
    path = os.path.join('data/documents', row['strand-one-filename'])
    doc_type = row['strand-one-type']
    
    if doc_type == 'txtbk':
        return extract_clean_text(path)
    elif doc_type == 'teacherinterview':
        return extract_clean_text(path, remove_bullet_points=True)
    elif doc_type == 'policy':
        return extract_clean_text(path, remove_headers=True)
    elif doc_type == 'teacher-online-gpt':
        return extract_clean_text(path)
    else:
        return None