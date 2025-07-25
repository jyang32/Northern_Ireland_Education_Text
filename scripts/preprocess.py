# this script is use to clean the text data for more traditional/non LLM text analysis

import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import string
import os

# Download NLTK resources if not already present
#nltk.download('punkt')
#nltk.download('stopwords')
#nltk.download('wordnet')

# File paths
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_CSV = os.path.join(SCRIPT_DIR, '..', 'outputs', 'processed_text_data.csv')
OUTPUT_CSV = os.path.join(SCRIPT_DIR, '..', 'outputs', 'cleaned_text_data.csv')

# Load data
print('Reading processed data...')
df = pd.read_csv(INPUT_CSV)

# Initialize tools
stop_words = set(stopwords.words('english'))
# Read custom stop words from file
stopwords_path = os.path.join(SCRIPT_DIR, 'stopwords.txt')
custom_stop_words = set()
if os.path.exists(stopwords_path):
    with open(stopwords_path, 'r', encoding='utf-8') as f:
        for line in f:
            word = line.strip()
            if word:
                custom_stop_words.add(word)
stop_words.update(custom_stop_words)
lemmatizer = WordNetLemmatizer()

# Preprocessing function
def preprocess_text(text):
    if pd.isna(text):
        return ''
    # Lowercase
    text = text.lower()
    # Tokenize
    tokens = word_tokenize(text)
    # Remove punctuation and stopwords, lemmatize
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word.isalpha() and word not in stop_words]
    return ' '.join(tokens)

print('Preprocessing text...')
df['cleaned_content'] = df['content'].apply(preprocess_text)

# create a new directory for the cleaned data
os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)

# Save cleaned data
print(f'Saving cleaned data to {OUTPUT_CSV}...')
df.to_csv(OUTPUT_CSV, index=False)
print('Done.')

