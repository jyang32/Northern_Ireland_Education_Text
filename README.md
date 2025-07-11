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
- AI fallback summarization: When URL fetching fails, uses OpenAI to generate summaries of the content
- Content source tracking: The `has_url_content` and `has_ai_summary` columns indicate the source of content

## URL Processing with AI Fallback

The pipeline includes URL processing capabilities for combined documents:

- Raw Content Fetching: Uses enhanced web scraping to fetch live content from URLs
- AI Knowledge-Based Fallback: When raw fetching fails, uses OpenAI to generate summaries based on training data
- Educational Focus: AI summaries focus on Northern Ireland education and history relevance
- Automatic URL detection: Extracts URLs from text using regex patterns
- Configurable limits: Control character limits and timeouts via `config.py`
- Error handling: Handles failed requests and network issues
- Rate limiting: Includes delays between requests to be respectful to servers

### Configuration

URL processing can be configured in `scripts/config.py`:

```python
# URL processing parameters
FETCH_URLS = False  # Set this to False to skip all URL processing
MAX_URL_CHARS = 8000  # Maximum characters to extract from each URL
URL_TIMEOUT = 15  # Timeout for URL requests in seconds

# OpenAI fallback parameters
USE_OPENAI_FALLBACK = False  # Set this to False to disable AI completely
OPENAI_MODEL = "gpt-4o-mini"  # OpenAI model to use for summarization
# OpenAI API key will be loaded from .env file or environment variable
MAX_AI_SUMMARY_CHARS = 2000  # Maximum characters for AI-generated summaries
```

### AI Fallback Setup

To use the AI fallback functionality:

1. Install the required libraries:
```bash
pip install openai python-dotenv
```

2. Set your OpenAI API key in the `.env` file:
```bash
# In your .env file
OPENAI_API_KEY=your-api-key-here
```

3. The system will automatically:
   - Try to fetch raw content from URLs using enhanced web scraping
   - If raw fetching fails, use OpenAI to generate knowledge-based summaries
   - Focus on Northern Ireland education and history relevance
   - Provide summaries based on AI's training data about the domain

### Testing (test file currently ignored)

Run the URL processing test:

```bash
python test_url_processing.py
```

### Output Format

The processed data CSV now includes two content tracking columns:

- `has_url_content`: Indicates whether content includes fetched web resources
  - `True`: Content includes fetched web resources from URLs found in the document
  - `False`: Content is from the original document only

- `has_ai_summary`: Indicates whether content includes AI-generated summaries
  - `True`: Content includes AI-generated summaries (live content or knowledge-based)
  - `False`: Content is from raw URL fetching or original document only

This helps distinguish between:
- Original content: Text directly from the source document
- Live web content: Actually fetched from live websites using web scraping
- Knowledge-based summaries: AI summaries based on training data (when live access fails)

### Content Labels

The system uses clear labels to identify content sources:
- `[AI-GENERATED SUMMARY FROM KNOWLEDGE BASE]`: AI summary based on training data
- `--- URL Content {i}: {url} ---`: Raw live web content
- `--- AI SUMMARY {i}: {url} ---`: AI-generated summary

### Content Flag Combinations

The output CSV includes two content tracking columns that work together:

| `has_url_content` | `has_ai_summary` | Content Type | Description |
|-------------------|------------------|--------------|-------------|
| `False` | `False` | Original document content only | Pure text from the source document, no URL content |
| `True` | `False` | Raw URL content | Successfully fetched live web content from URLs |
| `True` | `True` | AI-generated URL content | AI summaries generated when raw URL fetching failed |

Note: When `has_ai_summary=True`, `has_url_content` will always be `True` because AI summaries are URL-derived content.
