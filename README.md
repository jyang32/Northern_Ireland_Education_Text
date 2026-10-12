# Northern Ireland Education Text Analysis

This project analyzes educational texts from the Option 1 and Option 2 perspectives in Northern Ireland GCSE history. It covers content analysis, topic modeling with BERTopic, sentiment analysis, and knowledge graph (KG) analysis of narrative structure. The analysis compares content across document types including textbooks, policy documents, and teacher interviews.

## Project Structure

```
Northern_Ireland_Education_Text/
├── README.md
├── environment.yml
├── environment_setup.md
├── sentiment predicate lists.txt
├── scripts/
│   ├── config.py
│   ├── utils.py
│   ├── file_reader.py
│   ├── main.py
│   ├── preprocess.py
│   ├── stopwords.txt
│   ├── phrases.txt
│   ├── enhanced_knowledge_extractor.py
│   ├── refined_ontology.py
│   ├── kg_utilities.py
│   ├── sentiment_lexicon.py
│   └── neo4j_loader_v2.py
├── analysis/
│   ├── BERTopic.ipynb
│   ├── descriptives.ipynb
│   ├── descriptives_no_interview.ipynb
│   ├── sentiment.ipynb
│   ├── knowledge_graph.ipynb
│   ├── preselected_kg_extraction.ipynb
│   ├── preselected_kg_extraction_updated.ipynb
│   ├── narrative_network_analysis.ipynb
│   ├── narrative_network_analysis_new.ipynb
│   ├── narrative_network_analysis_updated.ipynb
│   └── narrative_network_analysis_v3.ipynb
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
- `option2/`: All Option 2 perspective documents (textbooks, teacher interviews, etc.)
- `option1/`: All Option 1 perspective documents (textbooks, teacher interviews, etc.)
- `both/`: All shared/interview/policy documents (e.g., reconciled teacher interviews, policy docs)

Raw data and knowledge graph outputs (`graph_data/`, `knowledge_graphs/`) are not tracked in the repository.

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

### 2. API and Database Credentials

Sentiment analysis and knowledge graph extraction use the OpenAI API, and the knowledge graph is stored in a Neo4j Aura database. Create a `.env` file in the project root with:

```bash
OPENAI_API_KEY=your_openai_api_key
NEO4J_URI=neo4j+s://your-instance.databases.neo4j.io
NEO4J_USER=neo4j
NEO4J_PASSWORD=your_neo4j_password
```

The `.env` file and Neo4j credential files are excluded by `.gitignore`. Never commit them.

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

#### Step 5: Knowledge Graph Extraction
```bash
# Extract Subject-Predicate-Object (SPO) triples from the preselected corpus
jupyter notebook analysis/preselected_kg_extraction_updated.ipynb
```
Triples are extracted with GPT-4o-mini and can be loaded into Neo4j with `scripts/neo4j_loader_v2.py` for cross-corpus queries.

#### Step 6: Narrative Network Analysis
```bash
# Build Option 1 and Option 2 networks and run the network analysis
jupyter notebook analysis/narrative_network_analysis_v3.ipynb
```
`narrative_network_analysis_v3.ipynb` is the source of record for all knowledge graph figures and tables in the manuscript. Earlier versions (`narrative_network_analysis.ipynb`, `_new`, `_updated`) are kept for reference only.

### 4. Output Files

Your analysis will generate:
- `outputs/processed_text_data.csv` - Processed document data
- `outputs/cleaned_text_data.csv` - Cleaned text for analysis
- `outputs/analysis_results/` - Topic modeling results and visualizations; sentiment analysis results
- `graph_data/pi_report/figures/` and `graph_data/pi_report/tables/` - Knowledge graph figures and tables

## Knowledge Graph Analysis

The knowledge graph component compares how the Option 1 and Option 2 corpora structure their historical narratives.

**Scripts**
- `enhanced_knowledge_extractor.py` - LLM-based SPO triple extraction
- `refined_ontology.py` - Entity and relation type definitions
- `kg_utilities.py` - Entity normalization (including case-insensitive deduplication) and graph helpers
- `sentiment_lexicon.py` - Predicate sentiment lexicon
- `neo4j_loader_v2.py` - Loads extracted triples into Neo4j Aura

**Methods**
- Triples are converted into a directed network per corpus, with entities as nodes and predicates as edges.
- Communities are detected with a two-tier Leiden approach on the largest connected component; peripheral nodes are labeled separately.
- Node importance is measured with betweenness centrality; PageRank is used as a robustness check.
- Edge sentiment combines an LLM-coded predicate lexicon (`sentiment predicate lists.txt`) with the AFINN lexicon (Nielsen 2011). Where the AFINN score is non-zero, its sign determines the sentiment category; otherwise the lexicon label is used.

## URL Processing

The pipeline can fetch live content from URLs referenced in the combined documents. This is configured in `scripts/config.py`:

```python
# URL processing parameters
FETCH_URLS = False  # Set this to False to skip all URL processing
MAX_URL_CHARS = 8000  # Maximum characters to extract from each URL
URL_TIMEOUT = 15  # Timeout for URL requests in seconds
```
