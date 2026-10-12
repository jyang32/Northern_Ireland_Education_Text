"""
Updated NIKnowledgeExtractor with refined prompts and ontology usage
Improvements:
1. System/user message separation
2. Full ontology (entities + relationships) in prompt
3. Corpus option detection
4. Stronger constraints
5. Better formatting
"""

import json
import re
from typing import List, Dict
from dataclasses import dataclass
from openai import OpenAI
from refined_ontology import RefinedOntology, CorpusOption, get_ontology


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


def extract_json_from_response(content: str) -> str:
    """Extract JSON content from markdown-wrapped responses"""
    json_pattern = r'```json\s*(.*?)\s*```'
    match = re.search(json_pattern, content, re.DOTALL)
    if match:
        return match.group(1).strip()
    
    json_array_pattern = r'(\[.*\])'
    match = re.search(json_array_pattern, content, re.DOTALL)
    if match:
        return match.group(1).strip()
    
    return content.strip()


class NIKnowledgeExtractor:
    """
    Enhanced Knowledge extractor for Northern Ireland education texts
    Uses refined ontology with separate Option 1 and Option 2 definitions
    """
    
    def __init__(self, client: OpenAI, model: str = "gpt-4o-mini"):
        self.client = client
        self.model = model
        
        # Ontology will be set per corpus
        self.ontology = None
        self.corpus_option = None
        self.entity_types = []
        self.relationship_types = []
        
        self.extraction_stats = {
            'total_chunks': 0,
            'successful_extractions': 0,
            'failed_extractions': 0,
            'total_triples': 0,
            'api_calls': 0,
            'total_cost': 0.0
        }
        
        self.invalid_predicates = {}
        self.invalid_predicate_examples = {}

        print(f"Knowledge extractor initialized with model: {model}")
    
    def set_corpus_ontology(self, corpus_name: str):
        """
        Set the appropriate ontology based on corpus name
        Automatically detects Option 1 or Option 2
        """
        self.ontology, self.corpus_option = get_ontology(corpus_name)
        self.entity_types = self.ontology.get_all_labels(self.corpus_option)
        self.relationship_types = self.ontology.get_relationships(self.corpus_option)
        
        print(f"\nðŸ“š Ontology set for: {self.corpus_option.value.upper()}")
        print(f"   Entity types: {len(self.entity_types)}")
        print(f"   Relationship types: {len(self.relationship_types)}")
    
    def system_message(self) -> str:
        """
        System message containing role, ontology, and extraction instructions
        This follows best practices by separating instructions from data
        """
        if not self.ontology or not self.corpus_option:
            raise ValueError("Ontology not set. Call set_corpus_ontology() first.")
        
        # Format ontology for display
        ontology_formatted = self.ontology.to_formatted_string(self.corpus_option)
        
        return f"""You are an expert at creating Knowledge Graphs from Northern Ireland educational texts.

Your task is to extract structured knowledge relationships from educational texts that represent different perspectives on Northern Ireland's history.

=== ONTOLOGY ===
{ontology_formatted}

=== CRITICAL EXTRACTION RULES ===

1. **STRICT ONTOLOGY COMPLIANCE**:
   - Use ONLY the entity types listed above
   - Use ONLY the relationship types listed above
   - Do not create new entity types or relationships

2. **ENTITY NAMING RULES**:
   - Use FULL, COMPLETE names for people: "Ian Paisley" not "Ian" or "Paisley"
   - Use FULL, COMPLETE names for organizations: "IRA" or "Irish Republican Army" consistently
   - Use full event names: "Good Friday Agreement" not "GFA" or "agreement"
   - When a person is mentioned by first name only, infer the full name from context
   - Be consistent: if you use "Ian Paisley" once, use it throughout

   **CAPITALIZATION IS CRITICAL - follow these rules strictly**:
   - Person names MUST use Title Case: "Hitler" not "hitler", "Churchill" not "churchill"
   - Place names MUST use Title Case: "Belfast" not "belfast", "Germany" not "germany"
   - Organisation names MUST use their standard form: "IRA" not "ira", "UVF" not "uvf"
   - Event names MUST use Title Case: "Easter Rising" not "easter rising"
   - NEVER output an entity name in all-lowercase unless it is a common noun (e.g., "weapons", "economy")
   - If in doubt, use Title Case for proper nouns

   Examples:
   CORRECT: "Ian Paisley", "Hitler", "Belfast", "IRA", "Good Friday Agreement"
   WRONG: "Ian", "hitler", "belfast", "ira", "agreement"

3. **ENTITY EXTRACTION**:
   - Extract entities exactly as specified in the ontology
   - Preserve multi-word phrases (e.g., "Good Friday Agreement", "Ian Paisley")
   - Match entities to the correct type from the ontology
   - Remove articles ("the", "a", "an") from entity names
   - Use consistent capitalization

4. **RELATIONSHIP EXTRACTION**:
   - Find both explicit and implied relationships
   - Multiple relationships can exist between the same pair of entities
   - Relationships must be from the approved list only
   - Use the most specific relationship type that fits

5. **DISCOURSE ANALYSIS FOCUS**:
   - How events are portrayed and interpreted
   - Roles assigned to figures and organizations
   - Emotional and evaluative language about entities
   - Connections between historical events and contemporary issues
   - Community perspectives and narratives

6. **OUTPUT FORMAT**:
   Return a JSON array where each relationship is an object with:
   - "subject": entity name (string)
   - "subject_type": entity type from ontology (string)
   - "predicate": relationship type from ontology (string)
   - "object": entity name (string)
   - "object_type": entity type from ontology (string)
   - "confidence": how confident you are (0.0-1.0, float)

7. **OUTPUT REQUIREMENTS**:
   - Respond ONLY with a valid JSON array
   - No markdown formatting, no code blocks
   - No comments before or after the JSON
   - Ensure the JSON can be directly parsed by json.loads()

Example output format:
[
  {{
    "subject": "Ian Paisley",
    "subject_type": "Figure",
    "predicate": "Oppose",
    "object": "Good Friday Agreement",
    "object_type": "Event",
    "confidence": 0.95
  }},
  {{
    "subject": "IRA",
    "subject_type": "Entity",
    "predicate": "Decommission",
    "object": "weapons",
    "object_type": "Entity",
    "confidence": 0.88
  }}
]

Remember: Be consistent with the ontology. Every entity type and relationship type must exist in the lists provided above."""

    def user_message(self, text_chunk: str, corpus_name: str) -> str:
        """
        User message containing only the input text
        This follows best practices by separating data from instructions
        """
        return f"""Text Source: {corpus_name}

Input text to analyze:
```
{text_chunk}
```"""
    
    def extract_knowledge_from_chunk(self, text_chunk: str, corpus_name: str, 
                                   chunk_id: str, rate_limiter=None) -> List[KnowledgeTriple]:
        """
        Extract knowledge triples from a single text chunk
        
        Args:
            text_chunk: Text to extract knowledge from
            corpus_name: Name of the corpus (must contain 'option1' or 'option2')
            chunk_id: Unique identifier for this chunk
            rate_limiter: Optional rate limiter object
        
        Returns:
            List of KnowledgeTriple objects
        """
        # Set ontology if not already set or if corpus changed
        if not self.ontology or corpus_name.lower() not in ['option1', 'option2', self.corpus_option.value]:
            self.set_corpus_ontology(corpus_name)
        
        try:
            # Rate limiting
            if rate_limiter:
                rate_limiter.safe_wait()
            
            # Make API call with separated system and user messages
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.system_message()},
                    {"role": "user", "content": self.user_message(text_chunk, corpus_name)}
                ],
                max_tokens=2000,  # Increased for larger ontology
                temperature=0.1   # Low temperature for consistency
            )
            
            self.extraction_stats['api_calls'] += 1
            
            # Extract and parse response
            content = response.choices[0].message.content
            json_content = extract_json_from_response(content)
            
            # Track costs
            if hasattr(response, 'usage') and rate_limiter:
                cost = rate_limiter.estimate_cost(
                    response.usage.prompt_tokens,
                    response.usage.completion_tokens
                )
                self.extraction_stats['total_cost'] += cost
            
            # Parse JSON response
            try:
                relations_data = json.loads(json_content)
                if isinstance(relations_data, dict) and 'relations' in relations_data:
                    relations_data = relations_data['relations']
                elif not isinstance(relations_data, list):
                    relations_data = []
            except json.JSONDecodeError as e:
                print(f"  ⚠️  JSON parsing error for {chunk_id}: {e}")
                self.extraction_stats['failed_extractions'] += 1
                return []
            
            # Convert to KnowledgeTriple objects with validation
            triples = []
            VALID_CATEGORIES = {'event', 'figure', 'entity'}

            for relation in relations_data:
                try:
                    # Validate entity types
                    subject_type = str(relation['subject_type']).strip().lower()
                    if subject_type not in VALID_CATEGORIES:
                        print(f"  ⚠️  Invalid subject_type: {relation['subject_type']}")
                        continue

                    object_type = str(relation['object_type']).strip().lower()
                    if object_type not in VALID_CATEGORIES:
                        print(f"  ⚠️  Invalid object_type: {relation['object_type']}")
                        continue
                    
                    # Validate relationship type
                    predicate = str(relation['predicate']).strip()
                    predicate_lower = predicate.lower()
                    relationship_types_lower = [r.lower() for r in self.relationship_types]

                    if predicate_lower not in relationship_types_lower:
                        # track invalid predicates for analysis
                        self.invalid_predicates[predicate] = self.invalid_predicates.get(predicate, 0) + 1
                        # store examples
                        if predicate not in self.invalid_predicate_examples:
                            self.invalid_predicate_examples[predicate] = []
                        
                        example = (relation.get('subject', ''), relation.get('object', ''))
                        if len(self.invalid_predicate_examples[predicate]) < 3:
                            self.invalid_predicate_examples[predicate].append(example)
                        
                        print(f"  ⚠️  Invalid predicate: {relation['predicate']}")
                        continue

                    triple = KnowledgeTriple(
                        subject=str(relation['subject']).strip(),
                        predicate=str(relation['predicate']).strip(),
                        object=str(relation['object']).strip(),
                        subject_type=str(relation['subject_type']).strip(),
                        object_type=str(relation['object_type']).strip(),
                        confidence=float(relation.get('confidence', 0.8)),
                        source_corpus=corpus_name,
                        chunk_id=chunk_id
                    )
                    triples.append(triple)
                except (KeyError, ValueError, TypeError) as e:
                    print(f"  ⚠️  Error creating triple: {e}")
                    continue
            
            self.extraction_stats['successful_extractions'] += 1
            self.extraction_stats['total_triples'] += len(triples)
            
            return triples
            
        except Exception as e:
            print(f"  ⚠️ Extraction failed for {chunk_id}: {e}")
            self.extraction_stats['failed_extractions'] += 1
            return []
    
    def extract_from_corpus(self, chunks: List[str], metadata: List[Dict], 
                          corpus_name: str, max_chunks: int = None,
                          save_progress: bool = True, rate_limiter=None) -> List[KnowledgeTriple]:
        """Extract knowledge from an entire corpus"""
        from tqdm import tqdm
        from datetime import datetime
        import pickle
        from kg_utilities import normalize_triples  # Import normalizer


        print(f"\n{'='*70}")
        print(f"Starting extraction from {corpus_name}...")
        print(f"{'='*70}")
        print(f"Total chunks: {len(chunks)}")
        
        # Set ontology for this corpus
        self.set_corpus_ontology(corpus_name)
        
        if max_chunks:
            chunks = chunks[:max_chunks]
            metadata = metadata[:max_chunks]
            print(f"Limited to: {max_chunks} chunks")
        
        all_triples = []
        self.extraction_stats['total_chunks'] += len(chunks)
        
        for i, (chunk, meta) in enumerate(tqdm(zip(chunks, metadata), 
                                              total=len(chunks), 
                                              desc=f"Extracting {corpus_name}")):
            
            chunk_id = f"{corpus_name}_{i}_{meta.get('original_index', i)}"
            triples = self.extract_knowledge_from_chunk(chunk, corpus_name, chunk_id, rate_limiter)
            all_triples.extend(triples)
            
            # Progress updates
            if (i + 1) % 25 == 0:
                print(f"\n  💼 Progress: {i+1}/{len(chunks)} chunks")
                print(f"     Extracted: {len(all_triples)} relations")
                print(f"     Cost so far: ${self.extraction_stats['total_cost']:.3f}")
                
                if save_progress:
                    self.save_progress(all_triples, f"{corpus_name}_progress_{i+1}")
        
        print(f"\n{'='*70}")
        print(f"✅ Extraction complete for {corpus_name}!")
        print(f"{'='*70}")
        print(f"  Extracted: {len(all_triples)} knowledge relations")
        print(f"  Success rate: {self.extraction_stats['successful_extractions']/(self.extraction_stats['total_chunks'])*100:.1f}%")
        print(f"  Total cost: ${self.extraction_stats['total_cost']:.3f}")
        print(f"{'='*70}")

        # After extraction, normalize entity names
        print(f"\n🛠️ Normalizing entity names...")
        all_triples = normalize_triples(all_triples, self.ontology)
        print(f"✅ Entity names normalized")


        if save_progress:
            self.save_progress(all_triples, f"{corpus_name}_final")
        # Save invalid predicates report
        if self.invalid_predicates:
            self.save_invalid_predicates_report(corpus_name)

        return all_triples
    
    def save_invalid_predicates_report(self, corpus_name: str):
        """Save report of invalid predicates for ontology refinement"""
        from datetime import datetime
        import json
        import os
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = f"../graph_data/invalid_predicates_{corpus_name}_{timestamp}.json"
        
        # Sort by frequency
        sorted_predicates = sorted(
            self.invalid_predicates.items(), 
            key=lambda x: x[1], 
            reverse=True
        )
        
        report = {
            'corpus': corpus_name,
            'timestamp': timestamp,
            'total_invalid_count': sum(self.invalid_predicates.values()),
            'unique_invalid_predicates': len(self.invalid_predicates),
            'predicates': [
                {
                    'predicate': pred,
                    'count': count,
                    'examples': self.invalid_predicate_examples.get(pred, [])
                }
                for pred, count in sorted_predicates
            ]
        }
        
        os.makedirs('../graph_data', exist_ok=True)
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        print(f"\n📋 Invalid predicates report saved: {report_file}")
        print(f"   Total invalid: {report['total_invalid_count']}")
        print(f"   Unique predicates: {report['unique_invalid_predicates']}")
        print(f"\n   Top 10 invalid predicates:")
        for item in report['predicates'][:10]:
            print(f"      {item['predicate']:20s} : {item['count']:3d} times")


    def save_progress(self, triples: List[KnowledgeTriple], filename_suffix: str):
        """Save extraction progress"""
        from datetime import datetime
        import pickle
        import os
        
        # Create directory if needed
        os.makedirs('../graph_data', exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        pickle_path = f"../graph_data/kg_triples_{filename_suffix}_{timestamp}.pkl"
        
        with open(pickle_path, 'wb') as f:
            pickle.dump(triples, f)
        print(f"  💼 Progress saved: {pickle_path}")
    
    def print_final_summary(self):
        """Print final extraction summary"""
        stats = self.extraction_stats
        print("\n" + "="*70)
        print("⚠️ FINAL KNOWLEDGE GRAPH EXTRACTION SUMMARY")
        print("="*70)
        print(f"Total chunks processed:  {stats['total_chunks']:,}")
        print(f"Successful extractions:  {stats['successful_extractions']:,}")
        print(f"Failed extractions:      {stats['failed_extractions']:,}")
        print(f"Success rate:            {stats['successful_extractions']/max(stats['total_chunks'], 1)*100:.1f}%")
        print(f"Total relations:         {stats['total_triples']:,}")
        print(f"Avg relations/chunk:     {stats['total_triples']/max(stats['successful_extractions'], 1):.1f}")
        print(f"API calls made:          {stats['api_calls']:,}")
        print(f"Total cost:              ${stats['total_cost']:.3f}")

        if self.invalid_predicates:
            print(f"\n⚠️  VALIDATION STATISTICS")
            print(f"Invalid predicates found: {sum(self.invalid_predicates.values()):,}")
            print(f"Unique invalid predicates: {len(self.invalid_predicates)}")
            print(f"\nTop 5 most common:")
            for pred, count in sorted(self.invalid_predicates.items(), key=lambda x: x[1], reverse=True)[:5]:
                print(f"  {pred:20s} : {count:3d} times")
        print("="*70)


# Usage example for notebook
if __name__ == "__main__":
    print("NIKnowledgeExtractor - Enhanced Version")
    print("="*70)
    print("\nKey Features:")
    print("âœ“ System/user message separation")
    print("âœ“ Full ontology (entities + relationships)")
    print("âœ“ Automatic corpus option detection")
    print("âœ“ Strict ontology validation")
    print("âœ“ Enhanced error handling")
    print("\nTo use in notebook:")
    print("  extractor = NIKnowledgeExtractor(client=llm_clients['openai'])")
    print("  triples = extractor.extract_from_corpus(chunks, metadata, 'option1', rate_limiter=rate_limiter)")