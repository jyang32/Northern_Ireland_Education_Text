"""
Neo4j Knowledge Graph Loader (v2)
- Merges duplicate edges: same (subject, predicate, object) pairs get a single edge with weight
- Edge thickness in Neo4j Browser can represent the weight

CHANGELOG (v2 - 2026-02-10):
  - load_relationships now aggregates duplicate (subject, predicate, object, corpus) edges
  - Each merged edge has: weight (count), avg_confidence, chunk_ids list
  - Added merge_weighted_relationships() helper
  - Updated Cypher queries for weighted graph exploration

Usage in notebook (Cell 11):
    from neo4j_loader_v2 import (
        Neo4jConfig, CORPUS_COLORS, CORPUS_LABELS,
        connect_to_neo4j, clear_database, create_indexes,
        load_entities, load_relationships_weighted, verify_data,
        print_sample_queries
    )
"""

from collections import defaultdict
from typing import List, Dict


# Color scheme for visualization
CORPUS_COLORS = {
    'option1': '#E74C3C',   # Red (Protestant/Unionist)
    'option2': '#3498DB',   # Blue (Catholic/Nationalist)
    'both': '#2ECC71'       # Green (Shared)
}

CORPUS_LABELS = {
    'option1': 'Protestant/Unionist',
    'option2': 'Catholic/Nationalist',
    'both': 'Shared'
}


class Neo4jConfig:
    """Neo4j connection configuration - reads from environment variables"""
    def __init__(self):
        import os
        self.URI = os.getenv('NEO4J_URI', 'bolt://localhost:7687')
        self.USER = os.getenv('NEO4J_USER', 'neo4j')
        self.PASSWORD = os.getenv('NEO4J_PASSWORD', 'password')
        self.DATABASE = os.getenv('NEO4J_DATABASE', 'neo4j')


def connect_to_neo4j(config):
    """Connect to Neo4j database"""
    from neo4j import GraphDatabase
    try:
        driver = GraphDatabase.driver(
            config.URI,
            auth=(config.USER, config.PASSWORD)
        )
        driver.verify_connectivity()
        print(f"✅ Connected to Neo4j at {config.URI}")
        return driver
    except Exception as e:
        print(f"❌ Connection failed: {e}")
        print("\n💡 Make sure Neo4j is running and credentials are correct")
        print("   Update .env file with:")
        print("   NEO4J_URI=bolt://localhost:7687")
        print("   NEO4J_USER=neo4j")
        print("   NEO4J_PASSWORD=your_password")
        return None


def clear_database(driver, database):
    """Clear existing data (optional - use with caution!)"""
    print("\n⚠️  Clearing existing data...")
    with driver.session(database=database) as session:
        session.run("MATCH (n) DETACH DELETE n")
        print("✅ Database cleared")


def create_indexes(driver, database):
    """Create indexes for better performance"""
    print("\n📊 Creating indexes...")
    with driver.session(database=database) as session:
        try:
            session.run("CREATE INDEX entity_name IF NOT EXISTS FOR (n:Entity) ON (n.name)")
            print("✅ Index on entity names created")
        except Exception as e:
            print(f"⚠️  Index creation: {e}")


def load_entities(driver, database, option1_triples, option2_triples):
    """Load all entities as nodes"""
    print("\n📦 Loading entities...")
    
    # Collect all unique entities with their types and corpus
    entities = {}
    
    for triple in option1_triples:
        if triple.subject not in entities:
            entities[triple.subject] = {'type': triple.subject_type, 'corpora': set()}
        entities[triple.subject]['corpora'].add('option1')
        
        if triple.object not in entities:
            entities[triple.object] = {'type': triple.object_type, 'corpora': set()}
        entities[triple.object]['corpora'].add('option1')
    
    for triple in option2_triples:
        if triple.subject not in entities:
            entities[triple.subject] = {'type': triple.subject_type, 'corpora': set()}
        entities[triple.subject]['corpora'].add('option2')
        
        if triple.object not in entities:
            entities[triple.object] = {'type': triple.object_type, 'corpora': set()}
        entities[triple.object]['corpora'].add('option2')
    
    print(f"   Found {len(entities)} unique entities")
    
    with driver.session(database=database) as session:
        created = 0
        for entity_name, entity_data in entities.items():
            corpora_list = list(entity_data['corpora'])
            if len(corpora_list) == 2:
                corpus = 'both'
            else:
                corpus = corpora_list[0]
            
            color = CORPUS_COLORS[corpus]
            corpus_label = CORPUS_LABELS[corpus]
            
            session.run("""
                MERGE (e:Entity {name: $name})
                SET e.type = $type,
                    e.corpus = $corpus,
                    e.corpus_label = $corpus_label,
                    e.color = $color,
                    e.occurrences = $occurrences
            """,
            name=entity_name,
            type=entity_data['type'],
            corpus=corpus,
            corpus_label=corpus_label,
            color=color,
            occurrences=len(corpora_list)
            )
            created += 1
        
        print(f"✅ Created {created} entity nodes")
    
    return entities


def _aggregate_relationships(all_triples):
    """
    Aggregate duplicate (subject, predicate, object, corpus) edges.
    
    Returns a list of dicts, each representing a merged edge:
      {
        'subject': str,
        'predicate': str,
        'object': str,
        'corpus': str,
        'weight': int,
        'avg_confidence': float,
        'chunk_ids': list[str],
        'corpus_label': str,
        'color': str,
      }
    """
    # Group by (subject, predicate, object, corpus)
    edge_groups = defaultdict(lambda: {
        'confidences': [],
        'chunk_ids': [],
    })
    
    for triple in all_triples:
        key = (triple.subject, triple.predicate, triple.object, triple.source_corpus)
        edge_groups[key]['confidences'].append(triple.confidence)
        edge_groups[key]['chunk_ids'].append(triple.chunk_id)
    
    merged_edges = []
    for (subject, predicate, obj, corpus), data in edge_groups.items():
        merged_edges.append({
            'subject': subject,
            'predicate': predicate,
            'object': obj,
            'corpus': corpus,
            'weight': len(data['confidences']),
            'avg_confidence': sum(data['confidences']) / len(data['confidences']),
            'chunk_ids': data['chunk_ids'],
            'corpus_label': CORPUS_LABELS[corpus],
            'color': CORPUS_COLORS[corpus],
        })
    
    return merged_edges


def load_relationships_weighted(driver, database, option1_triples, option2_triples):
    """
    Load relationships with edge merging.
    
    Duplicate (subject, predicate, object, corpus) triples are merged into
    a single RELATED edge with:
      - weight: number of occurrences (use for edge thickness)
      - avg_confidence: mean confidence across occurrences
      - chunk_ids: list of source chunk IDs
    """
    print("\n🔗 Loading relationships (with edge merging)...")
    
    all_triples = option1_triples + option2_triples
    merged_edges = _aggregate_relationships(all_triples)
    
    # Stats
    total_raw = len(all_triples)
    total_merged = len(merged_edges)
    duplicates_removed = total_raw - total_merged
    multi_weight = [e for e in merged_edges if e['weight'] > 1]
    
    print(f"   Raw edges: {total_raw}")
    print(f"   After merging: {total_merged} (removed {duplicates_removed} duplicates)")
    print(f"   Edges with weight > 1: {len(multi_weight)}")
    if multi_weight:
        max_weight = max(e['weight'] for e in multi_weight)
        print(f"   Max edge weight: {max_weight}")
    
    with driver.session(database=database) as session:
        created = 0
        skipped = 0
        
        for edge in merged_edges:
            try:
                session.run("""
                    MATCH (a:Entity {name: $subject})
                    MATCH (b:Entity {name: $object})
                    CREATE (a)-[r:RELATED {
                        type: $predicate,
                        weight: $weight,
                        avg_confidence: $avg_confidence,
                        corpus: $corpus,
                        corpus_label: $corpus_label,
                        color: $color,
                        chunk_ids: $chunk_ids
                    }]->(b)
                """,
                subject=edge['subject'],
                object=edge['object'],
                predicate=edge['predicate'],
                weight=edge['weight'],
                avg_confidence=round(edge['avg_confidence'], 3),
                corpus=edge['corpus'],
                corpus_label=edge['corpus_label'],
                color=edge['color'],
                chunk_ids=edge['chunk_ids']
                )
                created += 1
                
            except Exception as e:
                skipped += 1
                if skipped <= 3:
                    print(f"  ⚠️  Skipped relationship: {e}")
        
        print(f"✅ Created {created} merged relationships")
        if skipped > 0:
            print(f"⚠️  Skipped {skipped} relationships")
    
    return merged_edges


def verify_data(driver, database):
    """Verify loaded data"""
    print("\n🔍 Verifying data...")
    
    with driver.session(database=database) as session:
        # Count nodes
        result = session.run("MATCH (n:Entity) RETURN count(n) as count")
        node_count = result.single()['count']
        
        # Count relationships
        result = session.run("MATCH ()-[r:RELATED]->() RETURN count(r) as count")
        rel_count = result.single()['count']
        
        # Total weight (original edge count)
        result = session.run("MATCH ()-[r:RELATED]->() RETURN sum(r.weight) as total_weight")
        total_weight = result.single()['total_weight']
        
        # Count by corpus
        result = session.run("""
            MATCH (n:Entity)
            RETURN n.corpus as corpus, count(n) as count
            ORDER BY count DESC
        """)
        corpus_counts = {record['corpus']: record['count'] for record in result}
        
        # Weighted edge stats
        result = session.run("""
            MATCH ()-[r:RELATED]->()
            WHERE r.weight > 1
            RETURN count(r) as multi_count, max(r.weight) as max_weight, avg(r.weight) as avg_weight
        """)
        weight_stats = result.single()
        
        print(f"✅ Nodes in Neo4j: {node_count:,}")
        print(f"✅ Merged relationships: {rel_count:,}")
        print(f"   (representing {total_weight:,} original edges)")
        print(f"\n   By corpus:")
        for corpus, count in corpus_counts.items():
            label = CORPUS_LABELS.get(corpus, corpus)
            print(f"      {label:25s} : {count:3d} entities")
        
        if weight_stats['multi_count'] and weight_stats['multi_count'] > 0:
            print(f"\n   Edge weight stats:")
            print(f"      Edges with weight > 1: {weight_stats['multi_count']}")
            print(f"      Max weight: {weight_stats['max_weight']}")
            print(f"      Avg weight (of multi): {weight_stats['avg_weight']:.1f}")


def print_sample_queries():
    """Print useful Cypher queries for weighted graph"""
    print("\n" + "="*70)
    print("🔍 USEFUL NEO4J QUERIES (Weighted Graph)")
    print("="*70)
    
    queries = {
        "View all nodes": """
MATCH (n:Entity)
RETURN n
LIMIT 25
        """,
        
        "View relationships with weight": """
MATCH (a:Entity)-[r:RELATED]->(b:Entity)
RETURN a.name, r.type, b.name, r.weight, r.avg_confidence, r.corpus_label
ORDER BY r.weight DESC
LIMIT 25
        """,
        
        "Heaviest edges (most repeated relationships)": """
MATCH (a:Entity)-[r:RELATED]->(b:Entity)
WHERE r.weight > 1
RETURN a.name AS from, r.type AS relationship, b.name AS to, 
       r.weight AS weight, r.corpus_label AS perspective
ORDER BY r.weight DESC
LIMIT 20
        """,
        
        "Option 1 only": """
MATCH (a:Entity)-[r:RELATED]->(b:Entity)
WHERE r.corpus = 'option1'
RETURN a, r, b
        """,
        
        "Option 2 only": """
MATCH (a:Entity)-[r:RELATED]->(b:Entity)
WHERE r.corpus = 'option2'
RETURN a, r, b
        """,
        
        "Shared entities": """
MATCH (n:Entity)
WHERE n.corpus = 'both'
RETURN n.name, n.type
ORDER BY n.occurrences DESC
        """,
        
        "Most connected entities": """
MATCH (n:Entity)
RETURN n.name, n.type, n.corpus_label, 
       size((n)-[:RELATED]->()) as out_degree,
       size((n)<-[:RELATED]-()) as in_degree,
       size((n)-[:RELATED]-()) as total_degree
ORDER BY total_degree DESC
LIMIT 10
        """,
        
        "Style edges by weight (run in Neo4j Browser)": """
// In Neo4j Browser Settings > Style, use this GRASS rule:
// relationship.RELATED { width: {weight}px; }
//
// Or use this query and check the visual:
MATCH p=(a:Entity)-[r:RELATED]->(b:Entity)
WHERE r.weight >= 2
RETURN p
        """,
    }
    
    for title, query in queries.items():
        print(f"\n{title}:")
        print(query)


if __name__ == "__main__":
    print("Neo4j Knowledge Graph Loader (v2 - Weighted Edges)")
    print("="*60)
    print("\nKey Features:")
    print("  ✔ Merges duplicate edges into weighted relationships")
    print("  ✔ Edge weight = occurrence count (for thickness visualization)")
    print("  ✔ Stores avg_confidence and source chunk_ids per edge")
    print("  ✔ Updated Cypher queries for weighted graph exploration")
