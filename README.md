# Northern Ireland Education Text Analysis

This project analyzes educational texts from Catholic and Protestant perspectives in Northern Ireland, comparing content across different document types including textbooks, policy documents, and teacher interviews.

## Project Structure

```
Northern_Ireland_Education_Text/
├── README.md
├── requirements.txt
├── scripts/
│   ├── config.py
│   ├── utils.py
│   ├── file_reader.py
│   └── main.py
├── data/
│   └── strand1/
│       ├── catholic/
│       │   ├── Madden (2011) CCEA revision guide Chp 3. Changing Relationships.docx
│       │   ├── Doherty (2001) Northern Ireland since c.1960.docx
│       │   └── ... (more textbooks)
│       ├── protestant/
│       │   ├── Madden (2007) History for CCEA GCSE Revision Guide - Chapter 3.docx
│       │   └── ... (more textbooks)
│       └── both/
│           ├── Reconciled_interviews/
│           │   ├── TeacherA_reconciled.docx
│           │   ├── TeacherB_reconciled.docx
│           │   └── ... (more interviews)
│           └── GCSE History (2017)-specification-Standard.docx
├── outputs/
│   └── processed_text_data.csv
└── CS_student_work/                   # Additional student work
```
- `catholic/`: All Catholic perspective documents (textbooks, etc.)
- `protestant/`: All Protestant perspective documents (textbooks, etc.)
- `both/`: All shared/interview/policy documents (e.g., teacher interviews, policy docs)

## Document Types

- **Textbooks**: Educational materials by Madden, Doherty, Johnston
- **Policy Documents**: GCSE Planning Frameworks and specifications
- **Combined Resources**: Comprehensive resource collections
- **Teacher Interviews**: Reconciled interview transcripts (under `both/`)

## Usage

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run the processing pipeline:
```bash
python -m scripts.main
```

3. Check outputs in the `outputs/` directory for processed data.

## Data Processing Features

- Automatic file type classification
- Text cleaning and preprocessing
- Large file chunking for analysis
- Religious group categorization
- Comprehensive metadata tracking
- URL content extraction: For combined documents, automatically fetches and includes content from referenced URLs
- Content source tracking: The`has_url_content` column indicates whether content includes fetched web resources

## URL Processing

The pipeline now includes advanced URL processing capabilities for combined documents:

- Automatic URL detection: Extracts URLs from text using regex patterns
- Content fetching: Downloads and processes web content using requests and BeautifulSoup
- Configurable limits: Control character limits and timeouts via `config.py`
- Error handling: Handles failed requests and network issues
- Rate limiting: Includes delays between requests to be respectful to servers

### Configuration

URL processing can be configured in `scripts/config.py`:

```python
# URL processing parameters
FETCH_URLS = True  # Whether to fetch content from URLs in combined documents
MAX_URL_CHARS = 8000  # Maximum characters to extract from each URL
URL_TIMEOUT = 15  # Timeout for URL requests in seconds
```

### Testing (test file currently ignored)

Run the URL processing test:

```bash
python test_url_processing.py
```

### Output Format

The processed data CSV now includes a `has_url_content` column:

- `True`: Content includes fetched web resources from URLs found in the document
- `False`: Content is from the original document only

This helps distinguish between:
- Original content: Text directly from the source document
- Enhanced content: Original text + fetched web resources (for combined documents with URLs)
