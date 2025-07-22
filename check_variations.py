import pandas as pd
from shared_deduplication_utils import SharedDeduplicationUtils
from fuzzywuzzy import fuzz
import jellyfish
from doublemetaphone import doublemetaphone

# Copy the same functions from code-material_list.py
def clean_group_name(name):
    utils = SharedDeduplicationUtils()
    return utils.clean_company_name(name)

def get_keywords(name):
    utils = SharedDeduplicationUtils()
    return utils.get_company_keywords(name)

def fuzzy_jaccard_similarity(name1, name2):
    keywords1 = get_keywords(name1)
    keywords2 = get_keywords(name2)
    if not keywords1 or not keywords2:
        return 0
    intersection = len(keywords1.intersection(keywords2))
    union = len(keywords1.union(keywords2))
    jaccard_sim = intersection / union if union > 0 else 0
    token_sort_ratio = fuzz.token_sort_ratio(name1, name2) / 100
    token_set_ratio = fuzz.token_set_ratio(name1, name2) / 100
    similarity = (jaccard_sim * 0.4 + token_sort_ratio * 0.3 + token_set_ratio * 0.3)
    return similarity

def jaro_winkler_similarity(name1, name2):
    if not name1 or not name2:
        return 0
    return jellyfish.jaro_winkler_similarity(name1, name2)

def metaphone_similarity(name1, name2):
    if not name1 or not name2:
        return 0
    mp1 = doublemetaphone(name1)
    mp2 = doublemetaphone(name2)
    similarities = []
    for code1 in mp1:
        for code2 in mp2:
            if code1 and code2:
                sim = jellyfish.jaro_winkler_similarity(code1, code2)
                similarities.append(sim)
    return max(similarities) if similarities else 0

def calculate_ensemble_confidence(name, group):
    """
    Calculate confidence score for assignment using ensemble methods
    """
    name_clean = clean_group_name(name)
    group_clean = clean_group_name(group)
    
    # Calculate individual similarity scores
    fuzzy_score = fuzzy_jaccard_similarity(name_clean, group_clean)
    jaro_score = jaro_winkler_similarity(name_clean, group_clean)
    metaphone_score = metaphone_similarity(name_clean, group_clean)
    
    # Calculate agreement (how many methods agree above their thresholds)
    thresholds = {'fuzzy_jaccard': 0.85, 'jaro_winkler': 0.89, 'metaphone': 0.91}
    agreements = 0
    if fuzzy_score >= thresholds['fuzzy_jaccard']:
        agreements += 1
    if jaro_score >= thresholds['jaro_winkler']:
        agreements += 1
    if metaphone_score >= thresholds['metaphone']:
        agreements += 1
    
    # Calculate weighted average score
    avg_score = (fuzzy_score * 0.4 + jaro_score * 0.3 + metaphone_score * 0.3)
    
    # Calculate confidence based on agreement and average score
    if agreements == 3:  # All methods agree
        confidence = avg_score * 1.1  # Boost confidence
    elif agreements == 2:  # Two methods agree
        confidence = avg_score * 0.9
    elif agreements == 1:  # One method agrees
        confidence = avg_score * 0.7
    else:  # No methods agree
        confidence = avg_score * 0.5
    
    return min(confidence, 1.0)  # Cap at 1.0

# Load actual data from database
utils = SharedDeduplicationUtils()
engine = utils.create_database_connection()
query = """
    SELECT materialGroupId, materialGroupName
    FROM rematter_default.material_group
"""
df = pd.read_sql(query, engine)

print("Looking for material groups that DON'T exactly match target names:")
print("=" * 70)

TARGET_GROUPS = ['Aluminum', 'Brass', 'Lead', 'Electronics', 'Copper', 'ICW', 'Steel', 'Autos', 'Stainless', 'Alloys', 'Other']

# Find material groups that don't exactly match any target group
non_exact_matches = []
for _, row in df.iterrows():
    name = row['materialGroupName']
    if name not in TARGET_GROUPS:  # Only check non-exact matches
        non_exact_matches.append(name)

print(f"Found {len(non_exact_matches)} material groups that don't exactly match target names")
print()

if non_exact_matches:
    print("Testing these variations:")
    print("-" * 50)
    
    for name in non_exact_matches[:10]:  # Test first 10
        # Find best match
        best_group = None
        best_confidence = 0
        
        for group in TARGET_GROUPS:
            confidence = calculate_ensemble_confidence(name, group)
            if confidence > best_confidence:
                best_confidence = confidence
                best_group = group
        
        if best_group and best_confidence > 0.3:  # Show all matches above 0.3
            print(f"{name:30} -> {best_group:12} | Confidence: {best_confidence:.3f}")
            
            if best_confidence >= 0.9:
                print(f"  ✓ Auto-approved (confidence >= 0.9)")
            elif best_confidence > 0.5:
                print(f"  ? Manual approval needed (0.5 < confidence < 0.9)")
            else:
                print(f"  ✗ Auto-rejected (confidence <= 0.5)")
            print()
else:
    print("All material groups exactly match target group names!")
    print("This explains why no manual approval was needed.") 