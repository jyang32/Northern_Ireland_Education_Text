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

# Load phrases to preserve
phrases_path = os.path.join(SCRIPT_DIR, 'phrases.txt')
preserve_phrases = []
if os.path.exists(phrases_path):
    with open(phrases_path, 'r', encoding='utf-8') as f:
        for line in f:
            phrase = line.strip()
            if phrase and not phrase.startswith('#'):  # Skip comments and empty lines
                preserve_phrases.append(phrase.lower())
    print(f"Loaded {len(preserve_phrases)} phrases to preserve")
else:
    print("No phrases.txt found, proceeding without phrase preservation")

lemmatizer = WordNetLemmatizer()

# Preprocessing function with phrase preservation
def preprocess_text(text):
    if pd.isna(text):
        return ''
    
    # Text normalization
    text = text.lower()
    
    # Replace "world war ii" with "second world war"
    text = text.replace('world war ii', 'second world war')
    text = text.replace('world war 2', 'second world war')
    text = text.replace('wwii', 'second world war')
    text = text.replace('ww2', 'second world war')
    
    # Replace "sinn fein" with "sinn féin" (proper Irish spelling)
    text = text.replace('sinn fein', 'sinn féin')
    
    # Step 1: Preserve phrases by replacing them with placeholders
    phrase_placeholders = {}
    protected_text = text
    
    for i, phrase in enumerate(preserve_phrases):
        if phrase in protected_text:
            placeholder = f"__PHRASE_{i}__"
            phrase_placeholders[placeholder] = phrase.replace(' ', '_')  # Replace spaces with underscores
            protected_text = protected_text.replace(phrase, placeholder)
    
    # Step 2: Tokenize the protected text
    tokens = word_tokenize(protected_text)
    
    # Step 3: Process tokens (lemmatize, remove punctuation, remove stopwords)
    processed_tokens = []
    for token in tokens:
        if token.startswith('__PHRASE_') and token.endswith('__'):
            # This is a preserved phrase - restore it
            if token in phrase_placeholders:
                processed_tokens.append(phrase_placeholders[token])
        elif token.isalpha():
            # Regular word processing
            lemmatized = lemmatizer.lemmatize(token)
            if lemmatized not in stop_words:
                processed_tokens.append(lemmatized)
    
    return ' '.join(processed_tokens)

print('Preprocessing text...')
df['cleaned_content'] = df['content'].apply(preprocess_text)

# create a new directory for the cleaned data
os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)

# Save cleaned data
print(f'Saving cleaned data to {OUTPUT_CSV}...')
df.to_csv(OUTPUT_CSV, index=False)
print('Done.')

