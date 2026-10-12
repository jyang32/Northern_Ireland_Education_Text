"""
Utility functions for Knowledge Graph Extraction Notebook
Focused on 4 pre-selected texts
"""

import re
import time
from typing import List, Dict
from datetime import datetime
from refined_ontology import RefinedOntology
from dataclasses import dataclass

@dataclass
class KnowledgeTriple:
    """Structure for storing extracted knowledge triples"""
    subject: str
    predicate: str
    object: str
    subject_type: str
    object_type: str
    confidence: float
    source_corpus: str
    chunk_id: str


class SimpleRateLimiter:
    """Simple rate limiter with cost tracking for API calls"""
    
    def __init__(self, delay_seconds: float = 1.0):
        self.delay_seconds = delay_seconds
        self.request_count = 0
        self.start_time = datetime.now()
        self.total_cost = 0.0
    
    def safe_wait(self):
        """Wait between requests"""
        time.sleep(self.delay_seconds)
        self.request_count += 1
        
        if self.request_count % 5 == 0:
            elapsed = (datetime.now() - self.start_time).seconds
            rate = self.request_count / max(elapsed, 1)
            print(f"  ðŸ“Š {self.request_count} requests | {elapsed}s elapsed | {rate:.1f} req/s")
    
    def estimate_cost(self, input_tokens: int, output_tokens: int, model: str = "gpt-4o-mini"):
        """
        Estimate cost for API call
        
        GPT-4o-mini pricing (as of 2024):
        - Input: $0.15 per 1M tokens
        - Output: $0.60 per 1M tokens
        """
        input_cost = (input_tokens * 0.15) / 1_000_000
        output_cost = (output_tokens * 0.60) / 1_000_000
        total_cost = input_cost + output_cost
        
        self.total_cost += total_cost
        return total_cost
    
    def get_summary(self) -> Dict:
        """Get rate limiter statistics"""
        elapsed = (datetime.now() - self.start_time).seconds
        return {
            'total_requests': self.request_count,
            'elapsed_seconds': elapsed,
            'avg_rate': self.request_count / max(elapsed, 1),
            'total_cost': self.total_cost
        }


class TextChunker:
    """Chunk text for knowledge extraction"""
    
    def __init__(self, chunk_size: int = 800, chunk_overlap: int = 150):
        """
        Initialize text chunker
        
        Args:
            chunk_size: Target size for each chunk in characters
            chunk_overlap: Overlap between chunks to maintain context
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
    
    def clean_text(self, text: str) -> str:
        """Clean text for processing"""
        if not text or text.strip() == '':
            return ""
        
        text = str(text).strip()
        
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'\n+', '\n', text)
        
        return text.strip()
    
    def chunk_by_sentences(self, text: str) -> List[Dict]:
        """
        Chunk text by sentences with overlap
        
        Returns list of dicts with 'text', 'start', 'end', 'length'
        """
        text = self.clean_text(text)
        
        if not text or len(text) < 100:
            return [{'text': text, 'start': 0, 'end': len(text), 'length': len(text)}] if text else []
        
        # Split by sentences
        sentences = re.split(r'[.!?]+\s+', text)
        sentences = [s.strip() + '.' for s in sentences if s.strip()]
        
        chunks = []
        current_chunk = ""
        start_pos = 0
        
        for sentence in sentences:
            # If adding this sentence exceeds chunk_size, save current chunk
            if len(current_chunk + sentence) > self.chunk_size and current_chunk:
                end_pos = start_pos + len(current_chunk)
                chunks.append({
                    'text': current_chunk.strip(),
                    'start': start_pos,
                    'end': end_pos,
                    'length': len(current_chunk)
                })
                
                # Create overlap by keeping last few sentences
                sentences_in_chunk = current_chunk.split('.')
                if len(sentences_in_chunk) >= 2:
                    overlap_text = '.'.join(sentences_in_chunk[-2:]) + '.'
                    current_chunk = overlap_text + ' ' + sentence
                    start_pos = end_pos - len(overlap_text)
                else:
                    current_chunk = sentence
                    start_pos = end_pos
            else:
                current_chunk += ' ' + sentence if current_chunk else sentence
        
        # Add the last chunk
        if current_chunk.strip():
            chunks.append({
                'text': current_chunk.strip(),
                'start': start_pos,
                'end': start_pos + len(current_chunk),
                'length': len(current_chunk)
            })
        
        return chunks


def format_extraction_summary(extractor_stats: Dict, elapsed_time: float = None):
    """Format extraction statistics for display"""
    print("\n" + "="*70)
    print("ðŸ“Š EXTRACTION SUMMARY")
    print("="*70)
    
    stats = extractor_stats
    total_chunks = stats.get('total_chunks', 0)
    successful = stats.get('successful_extractions', 0)
    failed = stats.get('failed_extractions', 0)
    total_triples = stats.get('total_triples', 0)
    
    print(f"Chunks processed:     {total_chunks:,}")
    print(f"Successful:           {successful:,}")
    print(f"Failed:               {failed:,}")
    
    if total_chunks > 0:
        success_rate = (successful / total_chunks) * 100
        print(f"Success rate:         {success_rate:.1f}%")
    
    print(f"\nRelations extracted:  {total_triples:,}")
    
    if successful > 0:
        avg_per_chunk = total_triples / successful
        print(f"Avg per chunk:        {avg_per_chunk:.1f}")
    
    print(f"\nAPI calls:            {stats.get('api_calls', 0):,}")
    print(f"Total cost:           ${stats.get('total_cost', 0):.3f}")
    
    if elapsed_time:
        print(f"Total time:           {elapsed_time:.1f}s")
    
    print("="*70)


def load_target_documents(df, target_filenames: List[str]):
    """
    Load ALL chunks for the target documents from dataframe
    
    Args:
        df: DataFrame with cleaned text data (pre-chunked)
        target_filenames: List of filenames to extract
    
    Returns:
        Dict with filename as key, document data as value
        Each document contains all its chunks
    """
    documents = {}
    
    for filename in target_filenames:
        # Get ALL rows matching this filename (all chunks)
        matches = df[df['filename'] == filename]
        
        if len(matches) > 0:
            documents[filename] = {
                'filename': filename,
                'religious_group': matches.iloc[0]['religious_group'],
                'chunks': matches['cleaned_content'].tolist(),  # All chunk texts
                'chunk_ids': matches['chunk_id'].tolist(),      # Chunk indices
                'total_chunks': len(matches),
                'total_words': matches['word_count'].sum(),
                'total_chars': matches['content_length'].sum(),
                'original_indices': matches.index.tolist()
            }
            
            print(f"âœ… Loaded: {filename}")
            print(f"   Group: {matches.iloc[0]['religious_group']} | {len(matches)} chunks | {matches['word_count'].sum():,} total words")
        else:
            print(f"âŒ Not found: {filename}")
    
    return documents


def prepare_corpus_data(documents: Dict, chunker: TextChunker = None):
    """
    Prepare documents for extraction - handles pre-chunked data
    
    Args:
        documents: Dict of document data (with 'chunks' already present)
        chunker: TextChunker instance (not used if data is pre-chunked)
    
    Returns:
        Dict with 'option1' and 'option2' keys containing chunks and metadata
    """
    corpus_data = {
        'option1': {'chunks': [], 'metadata': []},
        'option2': {'chunks': [], 'metadata': []}
    }
    
    for filename, doc_data in documents.items():
        group = doc_data['religious_group']
        
        print(f"\nðŸ“„ {filename[:60]}...")
        print(f"   Group: {group}")
        print(f"   Chunks: {doc_data['total_chunks']}")
        
        # Data is already chunked - just organize by group
        for i, (chunk_text, chunk_id, orig_idx) in enumerate(zip(
            doc_data['chunks'], 
            doc_data['chunk_ids'],
            doc_data['original_indices']
        )):
            corpus_data[group]['chunks'].append(chunk_text)
            corpus_data[group]['metadata'].append({
                'filename': filename,
                'chunk_index': chunk_id,
                'original_index': orig_idx,
                'doc_chunk_number': i
            })
    
    return corpus_data


def normalize_entity_name(entity_name: str, ontology: RefinedOntology) -> str:
    """
    Normalize a single entity name to its canonical form.
    
    Pipeline:
      1. Unicode NFC normalization + strip whitespace
      2. Ontology alias lookup (exact match)
      3. Force-uppercase known abbreviations
      4. Smart title-casing for known person names (preserving particles)
    
    Args:
        entity_name: Raw entity name from extraction
        ontology: Ontology with canonical mappings
        
    Returns:
        Canonical entity name
    """
    import unicodedata
    
    # Step 1: Unicode NFC + strip
    name = unicodedata.normalize('NFC', entity_name.strip())
    if not name:
        return name
    
    # Step 2: Ontology alias lookup
    canonical = ontology.get_canonical_name(name)
    if canonical != name:
        return canonical
    
    # Step 3: Force-uppercase known abbreviations
    ABBREVIATIONS = {
        'ira', 'uvf', 'uda', 'dup', 'sdlp', 'gfa', 'nicra',
        'ruc', 'udr', 'pira', 'inla', 'oira', 'uup', 'sf',
    }
    if name.lower() in ABBREVIATIONS:
        return name.upper()
    
    # Step 4: Smart title-casing for known person names
    # Preserves lowercase particles: de, van, von, of, the
    KNOWN_PERSONS = [
        'paisley', 'valera', 'thatcher', 'adams', 'hume', 'churchill',
        'faulkner', 'craig', 'sands', 'chichester', 'brooke', 'lemass',
    ]
    LOWERCASE_PARTICLES = {'de', 'van', 'von', 'of', 'the', 'le', 'la', 'di'}
    
    if any(word in name.lower() for word in KNOWN_PERSONS):
        words = name.split()
        result = []
        for i, word in enumerate(words):
            if word.lower() in LOWERCASE_PARTICLES and i > 0:
                result.append(word.lower())
            else:
                result.append(word.capitalize())
        return ' '.join(result)
    
    return name


def _build_case_dedup_map(triples):
    """
    Build a mapping from variant entity names to their canonical form
    by grouping case-insensitive duplicates.
    
    Selection rules for canonical form:
      - Known abbreviations (IRA, etc.) -> UPPERCASE
      - Otherwise -> most frequent form wins
      - Tie-break: prefer the form with more uppercase letters (proper noun)
    
    Returns:
        Tuple of (merge_map dict, merged_report list)
    """
    import unicodedata
    from collections import Counter
    
    # Count every entity occurrence across all subject/object positions
    name_freq = Counter()
    for t in triples:
        name_freq[t.subject] += 1
        name_freq[t.object] += 1
    
    # Group by NFC-lowercased key
    groups = {}
    for name in name_freq:
        key = unicodedata.normalize('NFC', name.strip()).lower()
        if key not in groups:
            groups[key] = []
        groups[key].append(name)
    
    FORCE_UPPER = {
        'ira', 'uvf', 'uda', 'dup', 'sdlp', 'gfa', 'nicra',
        'ruc', 'udr', 'pira', 'inla', 'oira', 'uup', 'sf',
    }
    
    merge_map = {}
    merged_report = []
    
    for key, variants in groups.items():
        unique_variants = list(set(variants))
        if len(unique_variants) <= 1:
            continue  # no duplicates for this key
        
        # Pick canonical form
        if key in FORCE_UPPER:
            canonical = key.upper()
        else:
            # Most frequent form
            by_freq = sorted(unique_variants, key=lambda v: -name_freq[v])
            top_freq = name_freq[by_freq[0]]
            tied = [v for v in by_freq if name_freq[v] == top_freq]
            # Tie-break: more uppercase letters = proper noun form
            canonical = max(tied, key=lambda v: sum(1 for c in v if c.isupper()))
        
        for v in unique_variants:
            if v != canonical:
                merge_map[v] = canonical
        merged_report.append((canonical, unique_variants))
    
    return merge_map, merged_report


def normalize_triples(triples, ontology):
    """
    Normalize entity names in all triples (two-pass approach).
    
    Pass 1: Per-entity ontology alias resolution + abbreviation handling
    Pass 2: Case-insensitive deduplication across all entities in the batch
    
    Args:
        triples: List of extracted KnowledgeTriple objects
        ontology: RefinedOntology with canonical mappings
        
    Returns:
        List of triples with normalized entity names
    """
    # Pass 1: ontology-based normalization
    pass1 = []
    for triple in triples:
        pass1.append(KnowledgeTriple(
            subject=normalize_entity_name(triple.subject, ontology),
            predicate=triple.predicate,
            object=normalize_entity_name(triple.object, ontology),
            subject_type=triple.subject_type,
            object_type=triple.object_type,
            confidence=triple.confidence,
            source_corpus=triple.source_corpus,
            chunk_id=triple.chunk_id,
        ))
    
    # Pass 2: case-insensitive batch deduplication
    merge_map, merged_report = _build_case_dedup_map(pass1)
    
    if merged_report:
        print(f"  Entity dedup: merged {len(merge_map)} case-variant names")
        for canonical, variants in sorted(merged_report, key=lambda x: -len(x[1])):
            print(f"    {variants} -> \'{canonical}\'")
    
    if not merge_map:
        return pass1
    
    pass2 = []
    for triple in pass1:
        pass2.append(KnowledgeTriple(
            subject=merge_map.get(triple.subject, triple.subject),
            predicate=triple.predicate,
            object=merge_map.get(triple.object, triple.object),
            subject_type=triple.subject_type,
            object_type=triple.object_type,
            confidence=triple.confidence,
            source_corpus=triple.source_corpus,
            chunk_id=triple.chunk_id,
        ))
    
    return pass2


if __name__ == "__main__":
    print("Knowledge Graph Extraction Utilities (v5)")
    print("="*50)
    print("\nKey features:")
    print("  - Two-pass entity normalization:")
    print("    Pass 1: ontology alias resolution")
    print("    Pass 2: case-insensitive batch dedup")
    print("  - NFC unicode normalization")
    print("  - Smart title-casing (preserves \'de\', \'von\', etc.)")
    print("  - Automatic abbreviation uppercasing")
    print("\nAvailable classes and functions:")
    print("  - SimpleRateLimiter: API rate limiting")
    print("  - TextChunker: Text chunking with overlap")
    print("  - format_extraction_summary: Display statistics")
    print("  - load_target_documents: Load specific files")
    print("  - prepare_corpus_data: Organize by perspective")
    print("  - normalize_triples: Two-pass entity normalization")