# Northern Ireland Education Text Analysis

This project performs comprehensive analysis of educational texts from Option2 and Option1 perspectives in Northern Ireland, including content analysis, topic modeling with BERTopic, and sentiment analysis using OpenAI. The analysis compares content across different document types including textbooks, policy documents, and teacher interviews.

## Project Structure

```
Northern_Ireland_Education_Text/
├── README.md
├── environment.yml
├── environment_setup.md
├── scripts/
│   ├── config.py
│   ├── utils.py
│   ├── file_reader.py
│   ├── main.py
│   ├── preprocess.py
│   ├── stopwords.txt
│   └── phrases.txt
├── analysis/
│   ├── BERTopic.ipynb
│   ├── descriptives.ipynb
│   ├── descriptives_no_interview.ipynb
│   └── sentiment.ipynb
├── data/
│   ├── metadata.csv
│   └── strand1/
│       ├── both/
│       │   ├── GCSE History (2017)-specification-Standard.docx
│       │   └── Reconciled_interviews/
│       │       ├── TeacherF_reconciled.docx
│       │       └── TeacherI_reconciled.docx
│       ├── option1/
│       │   ├── GCSE Planning Framework History Unit 1 Section B Option 1.docx
│       │   ├── Madden (2007) History for CCEA GCSE Revision Guide - Chapter 3 (Peace, War and Neutrality).docx
│       │   ├── Madden (2009) History for CCEA GCSE Second Edition - Chapter 2 (Peace, War and Neutrality) (1).docx
│       │   ├── Madden (2011) ccea revision guide chp 2 peace war neutrality.docx
│       │   ├── Madden and McBride History for CCEA GCSE Chapter 2.docx
│       │   ├── option1_combined all.docx
│       │   ├── TeacherC_reconciled.docx
│       │   ├── TeacherD_reconciled.docx
│       │   ├── TeacherE_reconciled.docx
│       │   ├── TeacherH_reconciled.docx
│       │   ├── TeacherL_reconciled.docx
│       │   ├── TeacherN_reconciled.docx
│       │   ├── TeacherO_reconciled.docx
│       │   ├── TeacherP_reconciled.docx
│       │   ├── TeacherQ_reconciled.docx
│       │   └── Updated Johnston&Johnston no textboxes.docx
│       └── option2/
│           ├── Doherty (2001) Northern Ireland since c.1960 .docx
│           ├── GCSE Planning Framework History Unit 1 Section B Option 2.docx
│           ├── Madden (2007) History for CCEA GCSE Revision Guide - Chapter 4 (Changing Relationships) (2).docx
│           ├── Madden (2009) History for CCEA GCSE Second Edition - Chapter 3 (Changing Relationships).docx
│           ├── Madden (2011) CCEA revision guide Chp 3. Changing Relationships.docx
│           ├── Madden and McBride History for CCEA GCSE Chapter 3.docx
│           ├── option2_combined all.docx
│           ├── TeacherA_reconciled.docx
│           ├── TeacherB_reconciled.docx
│           ├── TeacherG_reconciled.docx
│           ├── TeacherJ_reconciled.docx
│           ├── TeacherK_reconciled.docx
│           ├── TeacherM_reconciled.docx
│           └── Updated Doherty (2001) no textboxes.docx
├── outputs/
│   ├── processed_text_data.csv
│   ├── cleaned_text_data.csv
│   └── analysis_results/
│       ├── models/
│       ├── raw/
│       ├── no_interview/
│       ├── reduce_outlier/
│       ├── all_visuals/
│       ├── option1_sentiment_analysis_fixed/
│       └── option2_sentiment_analysis_fixed/
```
- `option2/`: All Option2 perspective documents (textbooks, teacher interviews, etc.)
- `option1/`: All Option1 perspective documents (textbooks, teacher interviews, etc.)
- `both/`: All shared/interview/policy documents (e.g., reconciled teacher interviews, policy docs)

## Document Types

- **Textbooks**: Educational materials by Madden, Doherty, Johnston
- **Policy Documents**: GCSE Planning Frameworks and specifications
- **Combined Resources**: Comprehensive resource collections
- **Teacher Interviews**: Teacher interview transcripts (can be under `option2/`, `option1/`, or `both/`)

## Quick Start for New Users

### 1. Environment Setup

This project uses conda for dependency management. Follow these steps to set up your environment:

```bash
# Clone or download the project
cd Northern_Ireland_Education_Text

# Create the conda environment (recommended)
conda env create -f environment.yml

# Activate the environment
conda activate bertopic_env

```

### 2. OpenAI API Setup (for Sentiment Analysis)

To use sentiment analysis features, you'll need an OpenAI API key:

1. Get an API key from [OpenAI](https://platform.openai.com/api-keys)
2. Create a `.env` file in the project root:
```bash
# In your .env file
OPENAI_API_KEY=your_actual_api_key_here
```

### 3. Running the Analysis Pipeline

#### Step 1: Process Raw Data
```bash
# Read and process raw documents
python -m scripts.main

# Clean and preprocess text
python scripts/preprocess.py
```
#### Step 2: Descriptive Analysis
```bash
# Generate descriptive statistics
jupyter notebook analysis/descriptives.ipynb
```
#### Step 3: Topic Modeling
```bash
# Run Jupyter notebook for topic analysis
jupyter notebook analysis/BERTopic.ipynb
```

#### Step 4: Sentiment Analysis
```bash
# Run sentiment analysis on key terms
jupyter notebook analysis/sentiment.ipynb
```

### 4. Output Files

Your analysis will generate:
- `outputs/processed_text_data.csv` - Processed document data
- `outputs/cleaned_text_data.csv` - Cleaned text for analysis
- `outputs/analysis_results/` - Topic modeling results and visualizations; Sentiment analysis results

## URL Processing with AI Fallback

The pipeline includes URL processing capabilities for combined documents:

- Raw Content Fetching: Uses enhanced web scraping to fetch live content from URLs
- AI Knowledge-Based Fallback: When raw fetching fails, uses OpenAI to generate summaries based on training data

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
