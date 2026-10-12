"""
Sentiment Lexicon for Northern Ireland Education Knowledge Graph
================================================================
Classifies ALL predicates found in the extracted data into:
  - POSITIVE: Constructive, cooperative, beneficial actions
  - NEGATIVE: Adversarial, destructive, harmful, coercive actions
  - NEUTRAL:  Descriptive, procedural, or genuinely ambiguous actions

Sources:
  - 182 predicates from refined_ontology.py (validated by pipeline)
  - 198 additional predicates from invalid_predicates reports
    (these appear in preselected_kg_20260210_044054.csv after
     ontology expansion / relaxed validation)

Coverage: 380 unique predicates — every predicate that could appear
in Option 1 or Option 2 extraction data.

Classification principles (for NI conflict discourse analysis):
  1. Does this predicate frame the SUBJECT as constructive or destructive?
  2. In the NI conflict context, does this action imply
     cooperation/resolution or conflict/antagonism?
  3. Ambiguous predicates (context-dependent) default to NEUTRAL.

These classifications are research judgments, not objective truths.
Review and adjust for your specific analytical framework.
"""

# ── POSITIVE predicates (48) ───────────────────────────────────────
# Constructive, cooperative, beneficial, resolution-oriented actions
SENTIMENT_POSITIVE = {
    # --- Core positive (from ontology) ---
    'Accept',          # accepting terms, agreements
    'Achieve',         # accomplishment
    'Agree',           # consensus, compromise
    'Allow',           # permitting, enabling
    'Allowed',         # variant of Allow
    'Appreciate',      # valuing
    'Assist',          # helping
    'Be Friendly Towards',  # positive social relation
    'Be Happy',        # positive emotional state
    'Benefit',         # gaining advantage
    'Build',           # constructive
    'Commemorate',     # honouring, remembrance
    'Compromise',      # mutual concession
    'Cooperate',       # working together
    'Decommission',    # disarmament, peace process
    'Defend',          # protection
    'Encourage',       # positive influence
    'Enjoy',           # positive experience
    'Ensure',          # guaranteeing safety/rights
    'Favour',          # supporting, preferring
    'Favourable',      # positive stance
    'Favoured',        # preferred, supported
    'Glad',            # positive emotional state
    'Grant',           # giving rights/concessions
    'Guarantee',       # securing promises
    'Help',            # assistance
    'Hope',            # positive outlook
    'Improve',         # making better
    'Inspire',         # motivating positively
    'Join',            # uniting, participating
    'Keen',            # enthusiastic
    'Modernise',       # progressive improvement
    'Offer',           # extending proposals
    'Open',            # creating access/dialogue
    'Praise',          # commendation
    'Protect',         # safeguarding
    'Publish',         # transparency, information sharing
    'Reassure',        # calming, building confidence
    'Reassured',       # variant of Reassure
    'Recognise',       # acknowledging legitimacy
    'Regain',          # recovering positively
    'Release',         # freeing, letting go
    'Renounce',        # giving up violence/claims
    'Resolve',         # settling disputes
    'Save',            # rescuing, preserving
    'Share',           # distributing, power-sharing
    'Solve',           # resolving problems
    'Strengthen',      # reinforcing positively
    'Succeed',         # achieving success
    'Support',         # backing, endorsement
    'Sympathetic',     # positive disposition
    'Welcome',         # positive reception
    'Win',             # achieving victory/success
}

# ── NEGATIVE predicates (82) ──────────────────────────────────────
# Adversarial, destructive, coercive, conflict-oriented actions
SENTIMENT_NEGATIVE = {
    # --- Core negative (from ontology) ---
    'Abandon',         # deserting, leaving behind
    'Abolish',         # eliminating, dismantling
    'Accuse',          # blaming
    'Alarm',           # causing fear
    'Annoy',           # irritation, provocation
    'Appease',         # capitulation framing (negative in NI discourse)
    'Argue',           # conflict, disagreement
    'Arrest',          # coercive state action
    'Attack',          # violence, aggression
    'Ban',             # prohibition, suppression
    'Bomb',            # violence
    'Break',           # breaking agreements/structures
    'Campaign',        # sustained opposition/pressure
    'Challenge',       # confrontation
    'Condemn',         # moral denunciation
    'Criticise',       # negative evaluation
    'Damage',          # causing harm
    'Demand',          # coercive insistence
    'Denounce',        # public condemnation
    'Destroy',         # annihilation
    'Deteriorate',     # worsening
    'Detonate',        # explosive violence
    'Die',             # death
    'Disagree',        # opposition
    'Dislike',         # negative disposition
    'Dismantle',       # taking apart (typically negative framing)
    'Dissatisfied',    # negative evaluation
    'Divide',          # creating division
    'Dominate',        # power imbalance
    'Emigrate',        # forced/pressured departure (NI context)
    'Enforce',         # coercive implementation
    'Evacuate',        # forced displacement
    'Explode',         # explosive violence
    'Fail',            # failure
    'Fear',            # anxiety, threat perception
    'Fight',           # armed conflict
    'Fire',            # shooting, violence
    'Flee',            # escaping danger
    'Force',           # coercion
    'Furious',         # intense anger
    'Hit back',        # retaliation
    'Horrify',         # causing horror
    'Hunger strike',   # protest through self-harm
    'Impose',          # forcing onto others
    'Injure',          # causing physical harm
    'Insist',          # forceful demanding
    'Intern',          # detention without trial
    'Invade',          # military aggression
    'Isolate',         # excluding, cutting off
    'Justify',         # defending controversial actions
    'Kill',            # lethal violence
    'Lay claim',       # asserting contested ownership
    'Limit',           # restricting
    'Lose',            # defeat
    'Object',          # opposition
    'Oppose',          # resistance, antagonism
    'Outrage',         # anger, moral shock
    'Perish',          # death
    'Persuade',        # influencing (often coercive in NI context)
    'Protest',         # active opposition
    'Raid',            # aggressive incursion
    'Rampage',         # violent disorder
    'Refuse',          # blocking, denial
    'Regret',          # negative emotional response
    'Reject',          # dismissal
    'Remove',          # taking away, stripping
    'Resent',          # bitterness
    'Resign',          # forced/pressured departure
    'Resist',          # opposition, defiance
    'Retaliate',       # revenge, escalation
    'Riot',            # violent disorder
    'Search',          # security forces searching (NI context)
    'Sell',            # in NI context: "selling out" framing
    'Smeared',         # defamation
    'Smuggle',         # illegal activity
    'Split',           # fracturing, division
    'Steal',           # theft, illegal taking
    'Stop',            # blocking, halting
    'Suffer',          # enduring hardship
    'Surrender',       # capitulation
    'Suspect',         # suspicion, distrust
    'Suspend',         # halting governance/rights
    'Target',          # singling out for attack
    'Tax',             # economic burden framing
    'Threaten',        # intimidation
    'Undermine',       # subversion
    'Weaken',          # diminishing
    'Worsen',          # deteriorating
}

# ── NEUTRAL predicates (250) ──────────────────────────────────────
# Descriptive, procedural, cognitive, or genuinely ambiguous actions
SENTIMENT_NEUTRAL = {
    # --- Core neutral (from ontology) ---
    'Access',          # accessing
    'Act',             # generic action
    'Adopt',           # adopting policy/position
    'Affect',          # influence (direction ambiguous)
    'Aim',             # intention
    'Allocate',        # distribution (administrative)
    'Analyse',         # cognitive
    'Announce',        # communication
    'Appear',          # becoming visible
    'Apply',           # implementing
    'Appoint',         # administrative
    'Ask',             # requesting
    'Assume',          # taking on role
    'Attempt',         # trying
    'Attend',          # being present
    'Avoid',           # evasion (ambiguous)
    'Bail',            # legal process
    'Be Seen',         # perception
    'Become',          # transformation
    'Become known as', # naming
    'Begin',           # starting
    'Believe',         # cognitive
    'Born',            # birth
    'Borrow',          # temporary acquisition
    'Bring',           # transporting
    'Broadcast',       # media communication
    'Buy',             # economic transaction
    'Call',            # naming, summoning
    'Carry',           # transporting, carrying out
    'Cause',           # causation (direction ambiguous)
    'Cease',           # stopping (ambiguous: could stop good or bad)
    'Change',          # transformation (ambiguous)
    'Check',           # verifying
    'Claim',           # asserting
    'Clear',           # clarifying
    'Come to power',   # political transition
    'Command',         # directing
    'Commit',          # committing to action
    'Communicate',     # information exchange
    'Compare',         # analytical
    'Consider',        # cognitive
    'Contact',         # reaching out
    'Contain',         # holding within
    'Continue',        # persistence
    'Contribute',      # providing (context-dependent)
    'Control',         # governance (ambiguous)
    'Cope',            # managing difficulty
    'Cost',            # expense
    'Count',           # enumerating
    'Cover',           # spanning, reporting
    'Create',          # making (ambiguous)
    'Deal',            # handling
    'Decide',          # decision-making
    'Declare',         # announcing formally
    'Demonstrate',     # showing/protesting (ambiguous)
    'Depend',          # reliance
    'Deploy',          # military/strategic (ambiguous)
    'Desire',          # wanting
    'Develop',         # growth
    'Discuss',         # deliberation
    'Drive',           # propelling
    'Drop',            # releasing, abandoning
    'Effort',          # exertion
    'Elect',           # democratic process
    'Elected',         # democratic outcome
    'Emerge',          # appearing
    'Employ',          # hiring, using
    'End',             # concluding (could end good or bad things)
    'Engage',          # participating
    'Enter',           # joining, starting
    'Establish',       # creating institutions
    'Examine',         # analysing
    'Exercise',        # using power/rights
    'Expended',        # spent, used
    'Expense',         # cost
    'Experience',      # undergoing
    'Explain',         # cognitive
    'Export',          # economic
    'Express',         # communication
    'Extend',          # stretching, offering
    'Face',            # confronting
    'Fall',            # declining
    'Feel',            # emotional state
    'Find',            # discovering
    'Finish',          # completing
    'Fly',             # aviation
    'Follow',          # succeeding
    'Form',            # creating
    'Formed',          # variant of Form
    'Gain',            # acquiring
    'Get',             # obtaining
    'Give',            # transferring
    'Given',           # variant of Give
    'Go',              # moving
    'Go Ahead',        # proceeding
    'Grow',            # increasing
    'Happen',          # occurring
    'Have',            # possessing
    'Head',            # leading
    'Hear',            # perceiving
    'Higher',          # comparative
    'Highlight',       # emphasising
    'Hold',            # maintaining, possessing
    'Housed',          # located
    'Identify',        # recognising
    'Illustrate',      # demonstrating
    'Impact',          # affecting (direction ambiguous)
    'Implement',       # executing policy
    'Import',          # economic
    'Include',         # incorporating
    'Increase',        # growing (direction ambiguous)
    'Indicate',        # signalling
    'Inform',          # communicating
    'Intervene',       # intervention (ambiguous)
    'Introduce',       # bringing in
    'Investigate',     # examining
    'Involve',         # participating
    'Issue',           # releasing, publishing
    'Keep',            # maintaining
    'Kiss',            # (appears in data)
    'Know',            # cognitive
    'Land',            # arriving
    'Last',            # enduring
    'Lead',            # directing
    'Leave',           # departing
    'Link',            # connecting
    'Live',            # residing
    'Maintain',        # sustaining
    'Mainly',          # (modifier, not true predicate)
    'Make',            # creating, causing
    'March',           # organised movement (ambiguous)
    'Mean',            # signifying
    'Meet',            # assembling
    'Member',          # membership (not true predicate)
    'Might Give',      # conditional giving
    'Mount',           # organising
    'Move',            # relocating
    'Need',            # requiring
    'Notice',          # observing
    'Order',           # commanding (ambiguous authority)
    'Organise',        # arranging (ambiguous)
    'Organize',        # variant of Organise
    'Outline',         # describing
    'Part',            # separating
    'Participate',     # taking part
    'Partition',       # dividing (ambiguous: historical/political act)
    'Partly seen',     # perception
    'Pass',            # enacting legislation
    'Patrol',          # security activity
    'Pay',             # financial
    'Phone',           # communicating
    'Place',           # positioning
    'Plan',            # preparing
    'Plant',           # placing (ambiguous)
    'Play',            # performing role
    'Position',        # stance
    'Preferred name',  # naming (not true predicate)
    'Prepare',         # readying
    'Present',         # showing
    'Prevent',         # stopping (ambiguous)
    'Propose',         # suggesting
    'Provide',         # supplying
    'Pursue',          # following, seeking
    'Push',            # urging
    'Put',             # placing
    'Rather',          # (modifier, not true predicate)
    'Reach',           # attaining
    'React',           # responding
    'Read',            # consuming text
    'Realise',         # understanding
    'Rearm',           # military (descriptive)
    'Receive',         # getting
    'Redefine',        # reinterpreting
    'Reduce',          # decreasing (ambiguous)
    'Refer',           # referencing
    'Refuel',          # logistical
    'Regard',          # viewing, considering
    'Reinforce',       # strengthening (ambiguous)
    'Rely',            # depending
    'Remain',          # staying
    'Replace',         # substituting
    'Reply',           # responding
    'Report',          # communicating findings
    'Represent',       # standing for
    'Require',         # needing
    'Respond',         # answering
    'Result',          # causing outcome
    'Retain',          # keeping
    'Return',          # coming back
    'Reveal',          # disclosing
    'Rule',            # governing
    'Run',             # operating
    'Say',             # stating
    'See',             # perceiving
    'Seek',            # searching for
    'Seem',            # appearing
    'Send',            # dispatching
    'Serve',           # serving
    'Set',             # placing
    'Set up',          # establishing
    'Show',            # displaying
    'Sign',            # formalising agreement
    'Source',          # (not true predicate)
    'Spend',           # expending
    'Spot',            # noticing
    'Stand',           # positioning
    'Start',           # beginning
    'Start making',    # beginning production
    'State',           # declaring
    'Stay',            # remaining
    'Stay part',       # remaining within
    'Success',         # (noun, not true predicate)
    'Summarise',       # condensing
    'Sunk',            # sinking (naval)
    'Supply',          # providing
    'Swear',           # pledging
    'Swear loyalty to',# pledging allegiance
    'Switch',          # changing
    'Symbolise',       # representing
    'Take',            # acquiring
    'Take control',    # seizing power (ambiguous)
    'Take place',      # occurring
    'Tell',            # communicating
    'Term',            # naming
    'Think',           # cognitive
    'Thought',         # variant of Think
    'Trade',           # economic exchange
    'Transfer',        # moving
    'Treat',           # handling (ambiguous)
    'Try',             # attempting
    'Turn',            # changing direction
    'Understand',      # cognitive
    'Use',             # utilising
    'View',            # perceiving
    'Visit',           # travelling to
    'Voice',           # expressing
    'Vote',            # democratic participation
    'Want',            # desiring
    'Work',            # labouring
}


def get_sentiment(predicate: str) -> str:
    """Classify a predicate as positive, negative, or neutral.
    Unknown predicates default to neutral."""
    if predicate in SENTIMENT_POSITIVE:
        return 'positive'
    elif predicate in SENTIMENT_NEGATIVE:
        return 'negative'
    return 'neutral'


# ── Validation ─────────────────────────────────────────────────────
def validate_lexicon():
    """Verify no predicate is classified in multiple categories"""
    all_classified = SENTIMENT_POSITIVE | SENTIMENT_NEGATIVE | SENTIMENT_NEUTRAL

    pos_neg = SENTIMENT_POSITIVE & SENTIMENT_NEGATIVE
    pos_neu = SENTIMENT_POSITIVE & SENTIMENT_NEUTRAL
    neg_neu = SENTIMENT_NEGATIVE & SENTIMENT_NEUTRAL

    errors = []
    if pos_neg: errors.append(f"In both POSITIVE and NEGATIVE: {pos_neg}")
    if pos_neu: errors.append(f"In both POSITIVE and NEUTRAL: {pos_neu}")
    if neg_neu: errors.append(f"In both NEGATIVE and NEUTRAL: {neg_neu}")

    print(f"Sentiment Lexicon Validation")
    print(f"  Positive: {len(SENTIMENT_POSITIVE)} predicates")
    print(f"  Negative: {len(SENTIMENT_NEGATIVE)} predicates")
    print(f"  Neutral:  {len(SENTIMENT_NEUTRAL)} predicates")
    print(f"  Total:    {len(all_classified)} predicates")

    if errors:
        for e in errors:
            print(f"  ERROR: {e}")
    else:
        print(f"  No overlaps detected")

    return len(errors) == 0


def validate_against_data(csv_path: str):
    """Check coverage against actual extracted data CSV"""
    import pandas as pd
    from collections import Counter

    df = pd.read_csv(csv_path)
    data_preds = set(df['predicate'].unique())
    all_classified = SENTIMENT_POSITIVE | SENTIMENT_NEGATIVE | SENTIMENT_NEUTRAL

    missing = data_preds - all_classified
    unused = all_classified - data_preds

    print(f"\nCoverage against {csv_path}:")
    print(f"  Predicates in data:    {len(data_preds)}")
    print(f"  Predicates in lexicon: {len(all_classified)}")
    print(f"  Missing from lexicon:  {len(missing)}")
    if missing:
        for p in sorted(missing):
            freq = df[df['predicate'] == p].shape[0]
            print(f"    {p:<25} (freq: {freq}) → defaults to NEUTRAL")
    print(f"  Unused in lexicon:     {len(unused)}")

    # Distribution
    df['sentiment'] = df['predicate'].apply(get_sentiment)
    dist = df['sentiment'].value_counts()
    print(f"\n  Sentiment distribution in data:")
    for sent, count in dist.items():
        print(f"    {sent:<10} {count:>5} ({100*count/len(df):.1f}%)")

    # By corpus
    for corpus in sorted(df['corpus'].unique()):
        sub = df[df['corpus'] == corpus]
        sub_dist = sub['predicate'].apply(get_sentiment).value_counts()
        print(f"\n  {corpus}:")
        for sent in ['positive', 'negative', 'neutral']:
            count = sub_dist.get(sent, 0)
            print(f"    {sent:<10} {count:>5} ({100*count/len(sub):.1f}%)")


if __name__ == "__main__":
    validate_lexicon()
