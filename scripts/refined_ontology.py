"""
Refined Ontology for Northern Ireland Education Knowledge Graph
Separate definitions for Option 1 and Option 2 with granular entity and relationship types

CHANGELOG (v2 - 2026-02-10):
  - Added 33 new relationship types to Option 1 from invalid predicates analysis (count > 2)
  - Added 50 new relationship types to Option 2 from invalid predicates analysis (count > 2)
  - Added entity aliases: Neutrality/neutrality, Eire/Éire, hunger strike (1981)/hunger strike,
    Eamon de Valera/Eamon De valera
"""

from dataclasses import dataclass, field
from typing import List, Set, Dict
from enum import Enum
import unicodedata


class CorpusOption(Enum):
    """Corpus options for different textbook perspectives"""
    OPTION_1 = "option_1"
    OPTION_2 = "option_2"


@dataclass
class EntityDefinition:
    """Definition of an entity type with specific instances"""
    category: str  # Events, Figures, Entities
    instances: List[str]
    description: str = ""


@dataclass
class RefinedOntology:
    """Refined ontology with separate definitions for each corpus option"""
    
    # Entity definitions by option
    events_option1: List[str] = field(default_factory=lambda: [
        "Easter Rising",
        "Anglo-Irish Treaty",
        "battle of the somme",
        "Belfast Blitz",
        "Internment",
        "Emergency",
        "Neutrality",
        "world war ii",
        "second world war",
        "world war one",
        "first world war",
        "Home rule"
    ])
    
    events_option2: List[str] = field(default_factory=lambda: [
        "bloody sunday",
        "good friday agreement",
        "anglo irish agreement",
        "sunningdale agreement",
        "hunger strike",
        "hunger strike (1981)",
        "civil rights movement",
        "Internment",
        "Direct rule",
        "Power sharing"
    ])
    
    figures_option1: List[str] = field(default_factory=lambda: [
        "éamon de valera",
        "de valera",
        "Winston churchill"
    ])
    
    figures_option2: List[str] = field(default_factory=lambda: [
        "ian paisley",
        "brian faulkner",
        "terence o'neill",
        "james craig",
        "gerry adams",
        "bobby sands",
        "margaret thatcher",
        "john hume"
    ])
    
    entities_shared: List[str] = field(default_factory=lambda: [
        "ulster",
        "sinn féin",
        "sinn fein",
        "IRA",
        "Irish Republican Army",
        "provisional ira",
        "official ira",
        "ulster volunteer force",
        "UVF",
        "ulster defence association",
        "UDA",
        "democratic unionist party",
        "DUP",
        "social democratic labour party",
        "SDLP",
        "Catholic",
        "Protestant",
        "Orange order",
        "Éire",
        "Irish Free State",
        "Republic of Ireland",
        "Unionist",
        "Loyalist",
        "Nationalist",
        "Republicanist",
        "Republican",
        "nicra",
        "Northern Ireland Civil Rights Association"
    ])
    
    # =========================================================================
    # Relationship types by option
    # =========================================================================
    # v2: Added predicates from invalid_predicates analysis (count > 2)
    #     New additions are marked with # NEW
    # =========================================================================
    
    relationships_option1: List[str] = field(default_factory=lambda: [
        # --- Original relationships ---
        "Introduce", "Establish", "Remain", "Fear", "React", "Demand", 
        "Propose", "Agree", "Reject", "Result", "Attack", "Abandon", 
        "Retain", "Control", "Argue", "Challenge", "Encourage", "Allow", 
        "Win", "Dominate", "Resent", "Kill", "Force", "Refuse", 
        "Recognise", "Believe", "Undermine", "Divide", "Create", "Join", 
        "Become", "Condemn", "Remove", "Damage", "Retaliate", "Benefit", 
        "Increase", "Avoid", "Lead", "Lose", "Claim", "Elect", 
        "Strengthen", "Reinforce", "Oppose", "Solve", "Impact", "Threaten", 
        "Protect", "Support", "Defend", "Weaken", "Dismantle", "Sign", 
        "End", "Declare", "Prepare", "Implement", "Order", "Build", 
        "Fight", "Cause", "Affect", "Contribute", "Evacuate", "Destroy", 
        "Object", "Intervene", "Smuggle", "Set up", "Realise", "Outrage", 
        "Express", "Intern", "Import", "Export", "Start", "Regard", 
        "Guarantee", "Buy", "Rule", "Stop", "Tax", "Sell", 
        "Invade", "Rearm", "Appease", "Change", "Bomb", "Surrender", "Help",
        # --- NEW: from invalid predicates analysis (count > 2) ---
        "Want", "Give", "Provide", "Carry", "Call", "Feel",
        "Decide", "Continue", "Respond", "Need", "Leave", "Include",
        "Take", "Use", "Send", "See", "Attempt", "Visit",
        "Consider", "Live", "Annoy", "Announce", "Demonstrate", "Appoint",
        "Suffer", "Require", "Maintain", "Pass", "Take control", "Enter",
        "Think", "Treat",
    ])
    
    relationships_option2: List[str] = field(default_factory=lambda: [
        # --- Original relationships ---
        "Win", "Establish", "Control", "Introduce", "Propose", "Open", 
        "Create", "Lose", "Change", "Increase", "Visit", "Offer", 
        "Object", "Fear", "Die", "Ban", "Weaken", "Divide", 
        "React", "End", "Organise", "Threaten", "Respond", "Reject", 
        "Condemn", "Resign", "Argue", "Result", "Riot", "Deploy", 
        "Emerge", "Allow", "Reassure", "Prevent", "Improve", "Defend", 
        "Declare", "Welcome", "Maintain", "Grow", "Demand", "Replace", 
        "Challenge", "Fail", "Pursue", "Arrest", "Intern", "Become", 
        "Remove", "Suspend", "Announce", "Believe", "Continue", "Vote", 
        "Refuse", "Solve", "Agree", "Begin", "Join", "Compromise", 
        "Grant", "Reduce", "Deteriorate", "Cooperate", "Form", "Accuse", 
        "Allocate", "Damage", "Call", "Act", "Protest", "Target", 
        "Commemorate", "Attempt", "Share", "Accept", "Decommission", 
        "Renounce", "Release", "March", "Publish",
        # --- NEW: from invalid predicates analysis (count > 2) ---
        "Want", "Feel", "Give", "Provide", "Need", "Decide",
        "Carry", "Support", "Achieve", "See", "Leave", "Include",
        "Take", "Oppose", "Demonstrate", "Use", "Pass", "Hold",
        "Undermine", "Aim", "Face", "Follow", "Mean", "Appoint",
        "View", "Send", "Consider", "Live", "Annoy", "Seem",
        "Lead", "Analyse", "Ensure", "Involve", "Suffer", "Show",
        "Gain", "Require", "Enjoy", "Put", "Ask", "Take control",
        "Enter", "Think", "Depend", "Treat", "Link", "Explain", "Impact",
    ])
    
    # Phrases to preserve during tokenization
    preserved_phrases: List[str] = field(default_factory=lambda: [
        # Political figures
        "éamon de valera",
        "de valera",
        "ian paisley",
        "brian faulkner",
        "terence o'neill",
        "james craig",
        "bobby sands",
        "gerry adams",
        "margaret thatcher",
        "john hume",
        "winston churchill",
        
        # Political organizations
        "sinn féin",
        "sinn fein",
        "ira",
        "irish republican army",
        "provisional ira",
        "official ira",
        "ulster volunteer force",
        "ulster defence association",
        "democratic unionist party",
        "social democratic labour party",
        "orange order",
        
        # Historical events
        "easter rising",
        "bloody sunday",
        "good friday agreement",
        "belfast agreement",
        "anglo irish agreement",
        "sunningdale agreement",
        "hunger strike",
        "hunger strikes",
        "civil rights movement",
        "battle of the somme",
        "belfast blitz",
        "anglo-irish treaty",
        
        # Places and political entities
        "northern ireland",
        "irish free state",
        "republic of ireland",
        "united kingdom",
        "stormont parliament",
        "westminster parliament",
        
        # Political/Educational terms
        "world war ii",
        "second world war",
        "world war one",
        "first world war",
        "home rule",
        "direct rule",
        "power sharing",
        
        # Religious/sectarian terms
        "roman catholic",
        "catholic church",
        "protestant unionist"
    ])
    
    # =========================================================================
    # Canonical entity mappings
    # =========================================================================
    # v2: Added mappings for Neutrality, Éire, hunger strike, Éamon de Valera
    # =========================================================================
    entity_aliases: Dict[str, List[str]] = field(default_factory=lambda: {
        # Canonical name -> list of aliases/variations
        "Ian Paisley": ["Ian", "Paisley", "ian paisley", "Rev Paisley", "Reverend Paisley"],
        "Éamon de Valera": [
            "de Valera", "de_valera", "éamon de valera", "eamon de valera", "Dev",
            "Eamon de Valera", "Eamon De valera", "Eamon De Valera",
            "eamon De valera", "De Valera",
            "Éamon De Valera", "Éamon De valera", "Éamon DE VALERA",       # accented variants
        ],
        "IRA": ["ira", "I.R.A.", "Irish Republican Army"],
        "Sinn Féin": ["sinn fein", "Sinn Fein", "SF"],
        "Good Friday Agreement": ["good friday agreement", "GFA", "Belfast Agreement"],
        "Bloody Sunday": ["bloody sunday", "Bogside Massacre"],
        "Democratic Unionist Party": ["DUP", "dup", "Democratic Unionists"],
        "Social Democratic Labour Party": ["SDLP", "sdlp", "Social Democratic Party"],
        "Ulster Volunteer Force": ["UVF", "uvf"],
        "Ulster Defence Association": ["UDA", "uda"],
        "Margaret Thatcher": ["Thatcher", "Mrs Thatcher"],
        "Gerry Adams": ["Adams", "gerry adams"],
        "Bobby Sands": ["Bobby", "bobby sands", "Robert Sands"],
        "John Hume": ["Hume", "john hume"],
        "Winston Churchill": ["Churchill", "winston churchill"],
        "Anglo-Irish Treaty": ["anglo irish treaty", "Anglo Irish Treaty", "Treaty"],
        "Easter Rising": ["easter rising", "1916 Rising", "Easter Rebellion"],
        # --- NEW entity merges (v2) ---
        "Neutrality": ["neutrality", "NEUTRALITY"],
        "Éire": ["Eire", "eire", "EIRE", "éire"],
        "hunger strike": ["hunger strike (1981)", "Hunger Strike (1981)", "Hunger Strike", "hunger strikes"],
        "World War II": ["world war ii", "second world war", "Second World War", "WWII", "WW2", "ww2"],
    })

    def get_canonical_name(self, entity_name: str) -> str:
        """
        Get canonical name for an entity.
        Uses Unicode NFC normalization so that é (single char) and e+combining accent
        are treated as identical — prevents 'Éamon de Valera' vs 'Éamon De Valera' splits.
        
        Args:
            entity_name: Entity name as extracted
            
        Returns:
            Canonical name if alias found, otherwise original name
        """
        name = unicodedata.normalize('NFC', entity_name.strip())
        name_lower = name.lower()
        
        # Check if it's already canonical
        for canonical in self.entity_aliases.keys():
            if name == unicodedata.normalize('NFC', canonical):
                return canonical
        
        # Check if it's an alias (case-insensitive, Unicode-normalized)
        for canonical, aliases in self.entity_aliases.items():
            for alias in aliases:
                if name_lower == unicodedata.normalize('NFC', alias).lower():
                    return canonical
        
        # Return original if no mapping found
        return name
    
    def get_all_labels(self, option: CorpusOption) -> List[str]:
        """Get all entity labels for a specific corpus option"""
        if option == CorpusOption.OPTION_1:
            return (self.events_option1 + 
                   self.figures_option1 + 
                   self.entities_shared)
        elif option == CorpusOption.OPTION_2:
            return (self.events_option2 + 
                   self.figures_option2 + 
                   self.entities_shared)
        else:
            raise ValueError(f"Unknown option: {option}")
    
    def get_relationships(self, option: CorpusOption) -> List[str]:
        """Get relationship types for a specific corpus option"""
        if option == CorpusOption.OPTION_1:
            return self.relationships_option1
        elif option == CorpusOption.OPTION_2:
            return self.relationships_option2
        else:
            raise ValueError(f"Unknown option: {option}")
    
    def get_entity_categories(self, option: CorpusOption) -> Dict[str, List[str]]:
        """Get entities organized by category"""
        if option == CorpusOption.OPTION_1:
            return {
                "Event": self.events_option1,
                "Figure": self.figures_option1,
                "Entity": self.entities_shared
            }
        elif option == CorpusOption.OPTION_2:
            return {
                "Event": self.events_option2,
                "Figure": self.figures_option2,
                "Entity": self.entities_shared
            }
        else:
            raise ValueError(f"Unknown option: {option}")
    
    def to_formatted_string(self, option: CorpusOption) -> str:
        """Format ontology as a readable string for LLM prompts"""
        categories = self.get_entity_categories(option)
        relationships = self.get_relationships(option)
        
        output = f"=== Ontology for {option.value.upper()} ===\n\n"
        
        output += "ENTITY TYPES:\n"
        for category, entities in categories.items():
            output += f"\n{category}s:\n"
            for entity in entities:
                output += f"  - {entity}\n"
        
        output += f"\nRELATIONSHIP TYPES:\n"
        for rel in relationships:
            output += f"  - {rel}\n"
        
        return output
    
    def get_preserved_phrases_set(self) -> Set[str]:
        """Get preserved phrases as a set for efficient lookup"""
        return set(phrase.lower() for phrase in self.preserved_phrases)


# Create singleton instances
ONTOLOGY_OPTION_1 = RefinedOntology()
ONTOLOGY_OPTION_2 = RefinedOntology()


def get_ontology(corpus_name: str) -> tuple[RefinedOntology, CorpusOption]:
    """
    Get the appropriate ontology based on corpus name
    
    Args:
        corpus_name: Name of the corpus (should contain 'option1' or 'option2')
    
    Returns:
        Tuple of (ontology, option)
    """
    corpus_lower = corpus_name.lower()
    
    if 'option1' in corpus_lower or 'option_1' in corpus_lower:
        return ONTOLOGY_OPTION_1, CorpusOption.OPTION_1
    elif 'option2' in corpus_lower or 'option_2' in corpus_lower:
        return ONTOLOGY_OPTION_2, CorpusOption.OPTION_2
    else:
        raise ValueError(
            f"Cannot determine corpus option from name: {corpus_name}. "
            f"Name should contain 'option1' or 'option2'"
        )


# Example usage
if __name__ == "__main__":
    ontology = RefinedOntology()
    
    # Test Option 1
    print("=" * 60)
    print(ontology.to_formatted_string(CorpusOption.OPTION_1))
    print("\n" + "=" * 60)
    
    # Test Option 2
    print(ontology.to_formatted_string(CorpusOption.OPTION_2))
    print("\n" + "=" * 60)
    
    # Show relationship counts
    opt1_rels = ontology.get_relationships(CorpusOption.OPTION_1)
    opt2_rels = ontology.get_relationships(CorpusOption.OPTION_2)
    print(f"\nOption 1 relationship types: {len(opt1_rels)}")
    print(f"Option 2 relationship types: {len(opt2_rels)}")
    
    # Test new entity aliases
    print("\n" + "=" * 60)
    print("Testing new entity aliases:")
    test_cases = [
        ("neutrality", "Neutrality"),
        ("Eire", "Éire"),
        ("hunger strike (1981)", "hunger strike"),
        ("Eamon De valera", "Éamon de Valera"),
        ("Eamon de Valera", "Éamon de Valera"),
    ]
    for input_name, expected in test_cases:
        result = ontology.get_canonical_name(input_name)
        status = "✅" if result == expected else "❌"
        print(f"  {status} '{input_name}' -> '{result}' (expected: '{expected}')")