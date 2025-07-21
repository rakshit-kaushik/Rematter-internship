import pandas as pd
from shared_deduplication_utils import SharedDeduplicationUtils
from fuzzywuzzy import fuzz
import jellyfish
from doublemetaphone import doublemetaphone
import os
import sys

# 1. Define the 11 target groups
TARGET_GROUPS = [
    'Aluminum', 'Brass', 'Lead', 'Electronics', 'Copper', 'ICW',
    'Steel', 'Autos', 'Stainless', 'Alloys', 'Other'
]

# 2. Clean and normalize group names (reuse company cleaning logic)
def clean_group_name(name):
    utils = SharedDeduplicationUtils()
    return utils.clean_company_name(name)

def get_keywords(name):
    utils = SharedDeduplicationUtils()
    return utils.get_company_keywords(name)

# 3. Similarity functions (adapted from your code)
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

# 4. Ensemble assignment logic
def assign_group(name):
    """
    Assign material group name to one of the target groups using ensemble similarity
    """
    if pd.isna(name):
        return None
    
    name_clean = clean_group_name(name)
    best_group = None
    best_score = 0
    
    for group in TARGET_GROUPS:
        group_clean = clean_group_name(group)
        
        # Calculate ensemble similarity
        fuzzy_score = fuzzy_jaccard_similarity(name_clean, group_clean)
        jaro_score = jaro_winkler_similarity(name_clean, group_clean)
        metaphone_score = metaphone_similarity(name_clean, group_clean)
        
        # Weighted ensemble score
        ensemble_score = (fuzzy_score * 0.4 + jaro_score * 0.3 + metaphone_score * 0.3)
        
        if ensemble_score > best_score:
            best_score = ensemble_score
            best_group = group
    
    # Only assign if score is above threshold
    if best_score >= 0.6:  # Increased threshold to avoid false matches
        return best_group
    return None

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

def main():
    try:
        # 5. Load material groups from DB
        utils = SharedDeduplicationUtils()
        engine = utils.create_database_connection()
        query = """
            SELECT materialGroupId, materialGroupName
            FROM rematter_default.material_group
        """
        df = pd.read_sql(query, engine)
        print(f"Loaded {len(df)} material groups from database.")

        # 6. Pre-process all assignments for better performance
        print("\nPre-processing material group assignments...")
        assignments = []
        for _, row in df.iterrows():
            name = row['materialGroupName']
            mgid = row['materialGroupId']
            assigned_group = assign_group(name)
            if assigned_group:
                confidence = calculate_ensemble_confidence(name, assigned_group)
                assignments.append({
                    'name': name,
                    'mgid': mgid,
                    'group': assigned_group,
                    'confidence': confidence
                })

        # Group assignments by target group
        assignments_by_group = {}
        for assignment in assignments:
            group = assignment['group']
            if group not in assignments_by_group:
                assignments_by_group[group] = []
            assignments_by_group[group].append(assignment)

        # 7. Cluster/group assignment with manual approval
        clusters_by_id = {group: [] for group in TARGET_GROUPS}
        clusters_by_name = {group: [] for group in TARGET_GROUPS}
        print("\nProcessing material group assignments:")
        print("- Auto-approving: confidence >= 0.9")
        print("- Manual approval: 0.5 < confidence < 0.9")
        print("- Auto-rejecting: confidence <= 0.5")
        
        for group in TARGET_GROUPS:
            print(f"\n=== {group} ===")
            group_assignments = assignments_by_group.get(group, [])
            
            for assignment in group_assignments:
                name = assignment['name']
                mgid = assignment['mgid']
                confidence = assignment['confidence']
                
                if confidence >= 0.9:
                    # High confidence - auto approve
                    print(f"  ✓ Auto-approved '{name}' (ID: {mgid}) for group '{group}' (confidence: {confidence:.3f})")
                    clusters_by_id[group].append(mgid)
                    clusters_by_name[group].append(name)
                elif confidence <= 0.5:
                    # Low confidence - auto reject
                    print(f"  ✗ Auto-rejected '{name}' (ID: {mgid}) for group '{group}' (confidence: {confidence:.3f})")
                else:
                    # Medium confidence - manual approval needed
                    while True:
                        response = input(f"Approve '{name}' (ID: {mgid}) for group '{group}'? (confidence: {confidence:.3f}) (y/n): ").lower().strip()
                        if response in ['y', 'yes']:
                            print(f"  ✓ Approved")
                            clusters_by_id[group].append(mgid)
                            clusters_by_name[group].append(name)
                            break
                        elif response in ['n', 'no']:
                            print(f"  ✗ Rejected")
                            break
                        else:
                            print("Please enter 'y' or 'n'")

        # Remove empty groups
        clusters_by_id = {k: v for k, v in clusters_by_id.items() if v}
        clusters_by_name = {k: v for k, v in clusters_by_name.items() if v}

        # 8. Output CSVs with error handling
        outdir = os.path.dirname(__file__)
        id_csv = os.path.join(outdir, 'material_group_clusters_by_id.csv')
        name_csv = os.path.join(outdir, 'material_group_clusters_by_name.csv')

        # Prepare DataFrames
        df_id = pd.DataFrame({
            'group': list(clusters_by_id.keys()),
            'materialGroupIds': [v for v in clusters_by_id.values()]
        })
        df_name = pd.DataFrame({
            'group': list(clusters_by_name.keys()),
            'materialGroupNames': [v for v in clusters_by_name.values()]
        })

        try:
            df_id.to_csv(id_csv, index=False)
            df_name.to_csv(name_csv, index=False)
            print(f"\nSaved clusters by ID to: {id_csv}")
            print(f"Saved clusters by Name to: {name_csv}")
        except Exception as e:
            print(f"Error saving CSV files: {e}")
            sys.exit(1)

        # Only print clusters by name to stdout
        print("\nMaterial Group Clusters (by Name):")
        for group, names in clusters_by_name.items():
            print(f"\n=== {group} ===")
            for n in sorted(set(names)):
                print(f"  - {n}")

    except Exception as e:
        print(f"Error in main execution: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 