import pandas as pd
from shared_deduplication_utils import SharedDeduplicationUtils
from fuzzywuzzy import fuzz
import jellyfish
from doublemetaphone import doublemetaphone
import os
import sys
import json

# 1. Define the 12 target groups (same as material_list.py)
TARGET_GROUPS = [
    'Aluminum', 'Brass', 'Lead', 'Electronics', 'Copper', 'ICW',
    'Steel', 'Autos', 'Stainless', 'Alloys', 'Electric Motors', 'Other'
]

# 2. Similarity functions (same as material_list.py)
def clean_material_name(name):
    utils = SharedDeduplicationUtils()
    return utils.clean_company_name(name)  # Reuse company cleaning logic

def get_material_keywords(name):
    utils = SharedDeduplicationUtils()
    return utils.get_company_keywords(name)  # Reuse keyword extraction

def fuzzy_jaccard_similarity(name1, name2):
    keywords1 = get_material_keywords(name1)
    keywords2 = get_material_keywords(name2)
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

def calculate_ensemble_similarity(name1, name2):
    """
    Calculate ensemble similarity between two material names
    """
    name1_clean = clean_material_name(name1)
    name2_clean = clean_material_name(name2)
    
    fuzzy_score = fuzzy_jaccard_similarity(name1_clean, name2_clean)
    jaro_score = jaro_winkler_similarity(name1_clean, name2_clean)
    metaphone_score = metaphone_similarity(name1_clean, name2_clean)
    
    # Weighted ensemble score
    ensemble_score = (fuzzy_score * 0.4 + jaro_score * 0.3 + metaphone_score * 0.3)
    return ensemble_score

def calculate_ensemble_confidence(name1, name2):
    """
    Calculate confidence score for material similarity using ensemble methods
    """
    name1_clean = clean_material_name(name1)
    name2_clean = clean_material_name(name2)
    
    # Calculate individual similarity scores
    fuzzy_score = fuzzy_jaccard_similarity(name1_clean, name2_clean)
    jaro_score = jaro_winkler_similarity(name1_clean, name2_clean)
    metaphone_score = metaphone_similarity(name1_clean, name2_clean)
    
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

def cluster_materials_by_similarity(materials_df, threshold=0.7):
    """
    Cluster materials within a group based on name similarity
    Returns: list of clusters, where each cluster is a list of material IDs
    """
    if len(materials_df) <= 1:
        return [[m['materialId'] for m in materials_df.to_dict('records')]]
    
    # Convert to list of dictionaries for easier processing
    materials = materials_df.to_dict('records')
    clusters = []
    processed = set()
    
    for i, material1 in enumerate(materials):
        if material1['materialId'] in processed:
            continue
            
        # Start a new cluster with this material
        current_cluster = [material1['materialId']]
        processed.add(material1['materialId'])
        
        # Find similar materials
        for j, material2 in enumerate(materials):
            if i == j or material2['materialId'] in processed:
                continue
                
            similarity = calculate_ensemble_similarity(
                material1['materialName'], 
                material2['materialName']
            )
            
            if similarity >= threshold:
                current_cluster.append(material2['materialId'])
                processed.add(material2['materialId'])
        
        clusters.append(current_cluster)
    
    return clusters

def select_main_material_id(cluster_materials, materials_df):
    """
    Select the main material ID for a cluster
    Strategy: Choose the material with the shortest name (most generic)
    """
    if not cluster_materials:
        return None
    
    # Get the materials in this cluster
    cluster_df = materials_df[materials_df['materialId'].isin(cluster_materials)]
    
    # Sort by name length (shortest first) and return the first one
    cluster_df = cluster_df.sort_values('materialName', key=lambda x: x.str.len())
    return cluster_df.iloc[0]['materialId']

def load_material_group_clusters():
    """
    Load the material group clusters from the previous step
    """
    try:
        # Try to load from the CSV file created by material_list.py
        cluster_file = os.path.join(os.path.dirname(__file__), 'material_group_clusters_by_id.csv')
        if os.path.exists(cluster_file):
            df = pd.read_csv(cluster_file)
            clusters = {}
            for _, row in df.iterrows():
                group = row['group']
                # Parse the material group IDs from string representation
                try:
                    if isinstance(row['materialGroupIds'], str):
                        # Handle string representation like "['uuid1', 'uuid2', 'uuid3']"
                        ids_str = row['materialGroupIds'].strip('[]')
                        # Split by comma and strip quotes and whitespace
                        ids = [id.strip().strip("'\"") for id in ids_str.split(',') if id.strip()]
                    else:
                        ids = row['materialGroupIds']
                    clusters[group] = ids
                except Exception as parse_error:
                    print(f"Warning: Could not parse material group IDs for {group}: {parse_error}")
                    clusters[group] = []
            return clusters
        else:
            print(f"Warning: {cluster_file} not found. Using empty clusters.")
            return {group: [] for group in TARGET_GROUPS}
    except Exception as e:
        print(f"Error loading material group clusters: {e}")
        return {group: [] for group in TARGET_GROUPS}

def main():
    try:
        # 1. Load material group clusters from previous step
        print("Loading material group clusters...")
        material_group_clusters = load_material_group_clusters()
        
        # 2. Connect to database and load materials
        print("Connecting to database and loading materials...")
        utils = SharedDeduplicationUtils()
        engine = utils.create_database_connection()
        
        # Load all materials with their group IDs
        query = """
            SELECT materialId, materialName, materialGroupId
            FROM rematter_default.material
            WHERE materialGroupId IS NOT NULL
        """
        all_materials_df = pd.read_sql(query, engine)
        print(f"Loaded {len(all_materials_df)} materials from database.")
        
        # 3. Process each target group
        all_clusters = []
        
        for target_group in TARGET_GROUPS:
            print(f"\n=== Processing {target_group} ===")
            
            # Get material group IDs for this target group
            group_ids = material_group_clusters.get(target_group, [])
            if not group_ids:
                print(f"  No material groups found for {target_group}")
                continue
            
            # Get all materials that belong to these material groups
            group_materials = all_materials_df[
                all_materials_df['materialGroupId'].isin(group_ids)
            ].copy()
            
            if len(group_materials) == 0:
                print(f"  No materials found for {target_group}")
                continue
            
            print(f"  Found {len(group_materials)} materials in {len(group_ids)} material groups")
            
            # 4. Cluster materials within this group
            print(f"  Clustering materials by name similarity...")
            material_clusters = cluster_materials_by_similarity(group_materials, threshold=0.7)
            
            print(f"  Created {len(material_clusters)} clusters")
            
            # 5. Process each cluster
            for cluster in material_clusters:
                if len(cluster) == 1:
                    # Single material cluster
                    main_material_id = cluster[0]
                    all_clusters.append({
                        'target_group': target_group,
                        'main_material_id': main_material_id,
                        'clustered_material_ids': cluster,
                        'cluster_size': 1
                    })
                else:
                    # Multi-material cluster
                    main_material_id = select_main_material_id(cluster, group_materials)
                    all_clusters.append({
                        'target_group': target_group,
                        'main_material_id': main_material_id,
                        'clustered_material_ids': cluster,
                        'cluster_size': len(cluster)
                    })
                    
                    # Show cluster details
                    cluster_materials = group_materials[group_materials['materialId'].isin(cluster)]
                    print(f"    Cluster {len(cluster)} materials:")
                    for _, material in cluster_materials.iterrows():
                        print(f"      - {material['materialName']} (ID: {material['materialId']})")
        
        # 6. Create output DataFrame
        print(f"\nCreating output...")
        output_data = []
        for cluster_info in all_clusters:
            output_data.append({
                'target_group': cluster_info['target_group'],
                'main_material_id': cluster_info['main_material_id'],
                'clustered_material_ids': json.dumps(cluster_info['clustered_material_ids']),
                'cluster_size': cluster_info['cluster_size']
            })
        
        output_df = pd.DataFrame(output_data)
        
        # 7. Save to CSV
        output_file = os.path.join(os.path.dirname(__file__), 'material_clusters.csv')
        output_df.to_csv(output_file, index=False)
        print(f"Saved material clusters to: {output_file}")
        
        # 8. Create human-readable text file with material names
        print(f"\nCreating human-readable output...")
        text_output_file = os.path.join(os.path.dirname(__file__), 'material_clusters_readable.txt')
        
        with open(text_output_file, 'w') as f:
            f.write("MATERIAL CLUSTERS - HUMAN READABLE FORMAT\n")
            f.write("=" * 50 + "\n\n")
            
            # Group clusters by target group
            for target_group in TARGET_GROUPS:
                group_clusters = [c for c in all_clusters if c['target_group'] == target_group]
                if not group_clusters:
                    continue
                
                f.write(f"=== {target_group.upper()} ===\n")
                f.write(f"Total clusters: {len(group_clusters)}\n")
                total_materials = sum(c['cluster_size'] for c in group_clusters)
                f.write(f"Total materials: {total_materials}\n\n")
                
                for i, cluster_info in enumerate(group_clusters, 1):
                    cluster_ids = cluster_info['clustered_material_ids']
                    main_id = cluster_info['main_material_id']
                    
                    # Get material names for this cluster
                    cluster_materials = all_materials_df[
                        all_materials_df['materialId'].isin(cluster_ids)
                    ]
                    
                    # Get main material name
                    main_material = cluster_materials[
                        cluster_materials['materialId'] == main_id
                    ].iloc[0]['materialName']
                    
                    f.write(f"Cluster {i} (Size: {len(cluster_ids)})\n")
                    f.write(f"Main Material: {main_material} (ID: {main_id})\n")
                    f.write("All Materials in Cluster:\n")
                    
                    for _, material in cluster_materials.iterrows():
                        material_name = material['materialName']
                        material_id = material['materialId']
                        if material_id == main_id:
                            f.write(f"  → {material_name} (ID: {material_id}) [MAIN]\n")
                        else:
                            f.write(f"  - {material_name} (ID: {material_id})\n")
                    
                    f.write("\n")
                
                f.write("\n" + "-" * 50 + "\n\n")
        
        print(f"Saved human-readable clusters to: {text_output_file}")
        
        # 8.5. Create human-readable CSV file with material names
        print(f"\nCreating human-readable CSV...")
        csv_readable_output_file = os.path.join(os.path.dirname(__file__), 'material_clusters_readable.csv')
        
        # Prepare data for readable CSV
        readable_csv_data = []
        for target_group in TARGET_GROUPS:
            group_clusters = [c for c in all_clusters if c['target_group'] == target_group]
            if not group_clusters:
                continue
            
            for i, cluster_info in enumerate(group_clusters, 1):
                cluster_ids = cluster_info['clustered_material_ids']
                main_id = cluster_info['main_material_id']
                
                # Get material names for this cluster
                cluster_materials = all_materials_df[
                    all_materials_df['materialId'].isin(cluster_ids)
                ]
                
                # Get main material name
                main_material = cluster_materials[
                    cluster_materials['materialId'] == main_id
                ].iloc[0]['materialName']
                
                # Get all material names in cluster
                all_material_names = []
                for _, material in cluster_materials.iterrows():
                    material_name = material['materialName']
                    material_id = material['materialId']
                    if material_id == main_id:
                        all_material_names.append(f"{material_name} [MAIN]")
                    else:
                        all_material_names.append(material_name)
                
                readable_csv_data.append({
                    'target_group': target_group,
                    'cluster_number': i,
                    'cluster_size': len(cluster_ids),
                    'main_material_name': main_material,
                    'main_material_id': main_id,
                    'all_material_names': ' | '.join(all_material_names),
                    'all_material_ids': json.dumps(cluster_ids)
                })
        
        # Create and save readable CSV
        readable_csv_df = pd.DataFrame(readable_csv_data)
        readable_csv_df.to_csv(csv_readable_output_file, index=False)
        print(f"Saved human-readable CSV to: {csv_readable_output_file}")
        
        # 9. Print summary
        print(f"\n=== Summary ===")
        print(f"Total clusters created: {len(all_clusters)}")
        for group in TARGET_GROUPS:
            group_clusters = [c for c in all_clusters if c['target_group'] == group]
            if group_clusters:
                total_materials = sum(c['cluster_size'] for c in group_clusters)
                print(f"{group}: {len(group_clusters)} clusters, {total_materials} materials")
        
        # 10. Show some example clusters
        print(f"\n=== Example Clusters ===")
        for cluster_info in all_clusters[:5]:  # Show first 5 clusters
            if cluster_info['cluster_size'] > 1:
                group = cluster_info['target_group']
                main_id = cluster_info['main_material_id']
                cluster_ids = cluster_info['clustered_material_ids']
                
                # Get material names for display
                cluster_materials = all_materials_df[
                    all_materials_df['materialId'].isin(cluster_ids)
                ]
                main_material = cluster_materials[
                    cluster_materials['materialId'] == main_id
                ].iloc[0]['materialName']
                
                print(f"{group} - Main: '{main_material}' (ID: {main_id})")
                print(f"  Cluster: {cluster_ids}")
                print()
        
    except Exception as e:
        print(f"Error in main execution: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
