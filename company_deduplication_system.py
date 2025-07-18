# ensemble location-first deduplication with interactive approval
# combines location-based clustering with ensemble name similarity methods
# includes interactive approval for low-confidence clusters

import pandas as pd
import numpy as np
import networkx as nx
from tqdm import tqdm
import math
import os
from fuzzywuzzy import fuzz
from doublemetaphone import doublemetaphone
import jellyfish
from datetime import datetime
from shared_deduplication_utils import SharedDeduplicationUtils

# ------------------------------
# STEP 1: Location-based clustering (first pass)
# ------------------------------

def haversine_distance(lat1, lon1, lat2, lon2):
    """calculate great circle distance between two points"""
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
    c = 2 * math.asin(math.sqrt(a))
    return c * 6371  # radius of earth in km

def get_geographic_blocking_key(lat, lon, precision=2):
    """create geographic blocking key"""
    lat_rounded = round(lat, precision)
    lon_rounded = round(lon, precision)
    return f"{lat_rounded},{lon_rounded}"

def location_similarity_function(row1, row2, distance_threshold):
    """check if companies are within distance threshold"""
    lat1, lon1 = row1['latitude'], row1['longitude']
    lat2, lon2 = row2['latitude'], row2['longitude']
    
    if pd.isna(lat1) or pd.isna(lon1) or pd.isna(lat2) or pd.isna(lon2):
        return False
    
    distance = haversine_distance(lat1, lon1, lat2, lon2)
    return distance <= distance_threshold

def build_location_clusters(df, distance_threshold=0.01):  # increased to 10 meters
    """build initial location-based clusters"""
    G = nx.Graph()
    G.add_nodes_from(df.index)
    
    # create geographic blocking keys with higher precision for better accuracy
    df['blocking_key'] = df.apply(
        lambda row: get_geographic_blocking_key(row['latitude'], row['longitude'], precision=4), 
        axis=1
    )
    
    print("building location-based clusters...")
    for key, group in tqdm(df.groupby('blocking_key'), desc="processing geographic blocks"):
        if len(group) <= 1:
            continue
            
        block_indices = group.index.tolist()
        for i in range(len(block_indices)):
            for j in range(i + 1, len(block_indices)):
                idx1, idx2 = block_indices[i], block_indices[j]
                
                if location_similarity_function(df.loc[idx1], df.loc[idx2], distance_threshold):
                    G.add_edge(idx1, idx2)
    
    return G

# ------------------------------
# STEP 2: Ensemble name similarity within locations
# ------------------------------

def calculate_fuzzy_jaccard_similarity(name1, name2):
    """calculate fuzzy jaccard similarity with improved thresholds"""
    utils = SharedDeduplicationUtils()
    keywords1 = utils.get_company_keywords(name1)
    keywords2 = utils.get_company_keywords(name2)
    
    if not keywords1 or not keywords2:
        return 0
    
    intersection = len(keywords1.intersection(keywords2))
    union = len(keywords1.union(keywords2))
    jaccard_sim = intersection / union if union > 0 else 0
    
    # improved fuzzy matching with better thresholds
    token_sort_ratio = fuzz.token_sort_ratio(name1, name2) / 100
    token_set_ratio = fuzz.token_set_ratio(name1, name2) / 100
    partial_ratio = fuzz.partial_ratio(name1, name2) / 100
    
    # weighted combination with more emphasis on token-based methods
    similarity = (jaccard_sim * 0.3 + token_sort_ratio * 0.25 + token_set_ratio * 0.25 + partial_ratio * 0.2)
    
    return similarity

def calculate_metaphone_similarity(name1, name2):
    """calculate metaphone similarity with improved handling"""
    if pd.isna(name1) or pd.isna(name2) or not name1 or not name2:
        return 0
    
    utils = SharedDeduplicationUtils()
    keywords1 = utils.get_company_keywords(name1)
    keywords2 = utils.get_company_keywords(name2)
    
    filtered_name1 = ' '.join(keywords1)
    filtered_name2 = ' '.join(keywords2)
    
    if not filtered_name1 or not filtered_name2:
        return 0
    
    primary1, secondary1 = doublemetaphone(filtered_name1)
    primary2, secondary2 = doublemetaphone(filtered_name2)
    
    similarities = []
    # primary to primary (highest weight)
    if primary1 and primary2:
        sim = jellyfish.jaro_winkler_similarity(primary1, primary2)
        similarities.append(sim * 1.0)
    
    # primary to secondary
    if primary1 and secondary2:
        sim = jellyfish.jaro_winkler_similarity(primary1, secondary2)
        similarities.append(sim * 0.8)
    
    # secondary to primary
    if secondary1 and primary2:
        sim = jellyfish.jaro_winkler_similarity(secondary1, primary2)
        similarities.append(sim * 0.8)
    
    # secondary to secondary (lowest weight)
    if secondary1 and secondary2:
        sim = jellyfish.jaro_winkler_similarity(secondary1, secondary2)
        similarities.append(sim * 0.6)
    
    return max(similarities) if similarities else 0

def calculate_jaro_winkler_similarity(name1, name2):
    """calculate jaro-winkler similarity with improved preprocessing"""
    if not name1 or not name2:
        return 0
    
    utils = SharedDeduplicationUtils()
    keywords1 = utils.get_company_keywords(name1)
    keywords2 = utils.get_company_keywords(name2)
    
    keywords1_str = ' '.join(keywords1)
    keywords2_str = ' '.join(keywords2)
    
    if not keywords1_str or not keywords2_str:
        return 0
    
    # use jaro_winkler_similarity for better accuracy
    return jellyfish.jaro_winkler_similarity(keywords1_str, keywords2_str)

def calculate_name_ensemble_confidence(pair_methods, total_methods):
    """calculate ensemble confidence based on method agreement"""
    return len(pair_methods) / total_methods

def are_companies_similar_ensemble(row1, row2):
    """ensemble similarity function combining multiple methods with individual thresholds"""
    name1 = row1['companyName']
    name2 = row2['companyName']
    
    # calculate similarities using different methods
    fuzzy_jaccard = calculate_fuzzy_jaccard_similarity(name1, name2)
    metaphone = calculate_metaphone_similarity(name1, name2)
    jaro_winkler = calculate_jaro_winkler_similarity(name1, name2)
    
    # individual thresholds for each method
    thresholds = {
        'fuzzy_jaccard': 0.85,
        'metaphone': 0.91,
        'jaro_winkler': 0.89
    }
    
    # determine which methods agree (with individual thresholds)
    methods = []
    method_scores = {}
    
    if fuzzy_jaccard >= thresholds['fuzzy_jaccard']:
        methods.append('fuzzy_jaccard')
        method_scores['fuzzy_jaccard'] = fuzzy_jaccard
    
    if metaphone >= thresholds['metaphone']:
        methods.append('metaphone')
        method_scores['metaphone'] = metaphone
    
    if jaro_winkler >= thresholds['jaro_winkler']:
        methods.append('jaro_winkler')
        method_scores['jaro_winkler'] = jaro_winkler
    
    # calculate ensemble confidence
    confidence = calculate_name_ensemble_confidence(methods, 3)
    
    # companies are similar if ANY method agrees (very permissive)
    is_similar = len(methods) > 0
    
    # calculate weighted average score for confidence
    if method_scores:
        weighted_score = sum(method_scores.values()) / len(method_scores)
        final_confidence = (confidence + weighted_score) / 2
    else:
        final_confidence = confidence
    
    return is_similar, final_confidence

# ------------------------------
# STEP 3: Interactive approval system
# ------------------------------

def get_user_approval(cluster_companies, cluster_id):
    """get user approval for low-confidence clusters with improved interface"""
    print(f"\n{'='*60}")
    print(f"CLUSTER {cluster_id} - MANUAL REVIEW REQUIRED")
    print(f"{'='*60}")
    
    # Calculate confidence statistics for this cluster
    avg_confidence = cluster_companies['ensemble_confidence'].mean()
    min_confidence = cluster_companies['ensemble_confidence'].min()
    max_confidence = cluster_companies['ensemble_confidence'].max()
    
    print(f"Cluster size: {len(cluster_companies)}")
    print(f"Confidence scores:")
    print(f"  Average: {avg_confidence:.3f}")
    print(f"  Range: {min_confidence:.3f} - {max_confidence:.3f}")
    print(f"  Threshold: 0.600 (clusters below this need manual review)")
    print(f"  Note: Confidence combines Fuzzy Jaccard, Metaphone, and Jaro-Winkler similarities")
    print()
    
    print("Companies in this cluster:")
    
    for i, (_, company) in enumerate(cluster_companies.iterrows(), 1):
        confidence = company['ensemble_confidence']
        print(f"{i}. {company['companyName']}")
        print(f"   Confidence: {confidence:.3f}")
        print(f"   Location: {company.get('city', 'N/A')}, {company.get('state', 'N/A')}")
        print(f"   Coordinates: ({company.get('latitude', 'N/A')}, {company.get('longitude', 'N/A')})")
        print(f"   ID: {company['centralizedCompanyId']}")
        print()
    
    while True:
        response = input("Are these companies duplicates? (y/n/q to quit and save progress): ").lower().strip()
        if response in ['y', 'yes']:
            return True
        elif response in ['n', 'no']:
            return False
        elif response in ['q', 'quit']:
            print("Saving partial progress and exiting...")
            raise KeyboardInterrupt("User requested quit during manual review")
        else:
            print("Please enter 'y', 'n', or 'q'")

def process_low_confidence_clusters(df, confidence_threshold=0.60):  # lowered threshold
    """process low-confidence clusters with interactive approval"""
    print(f"\n{'='*60}")
    print("PROCESSING LOW-CONFIDENCE CLUSTERS")
    print(f"{'='*60}")
    
    # analyze all clusters first
    total_clusters = 0
    singleton_clusters = 0
    high_confidence_clusters = 0
    low_confidence_clusters = []
    
    for cluster_id in df['cluster_id'].unique():
        if cluster_id == -1:
            continue
        cluster = df[df['cluster_id'] == cluster_id]
        total_clusters += 1
        
        if len(cluster) == 1:
            singleton_clusters += 1
        else:
            avg_confidence = cluster['ensemble_confidence'].mean()
            if avg_confidence >= confidence_threshold:
                high_confidence_clusters += 1
            else:
                low_confidence_clusters.append(cluster_id)
    
    low_confidence_clusters = sorted(low_confidence_clusters)
    
    # print detailed statistics
    print(f"📊 CLUSTER ANALYSIS SUMMARY:")
    print(f"   Total clusters: {total_clusters}")
    print(f"   Singleton clusters (1 company): {singleton_clusters} - NO REVIEW NEEDED")
    print(f"   High-confidence clusters: {high_confidence_clusters} - NO REVIEW NEEDED")
    print(f"   Low-confidence clusters: {len(low_confidence_clusters)} - MANUAL REVIEW REQUIRED")
    print(f"   Confidence threshold: {confidence_threshold}")
    print()
    
    if len(low_confidence_clusters) == 0:
        print("✅ No low-confidence clusters found! All clusters are either singletons or high-confidence.")
        return df
    
    print(f"🔍 Found {len(low_confidence_clusters)} low-confidence clusters for manual review")
    
    # Show confidence ranges for low-confidence clusters
    if len(low_confidence_clusters) > 0:
        print(f"   Confidence ranges for low-confidence clusters:")
        for cluster_id in low_confidence_clusters[:5]:  # Show first 5
            cluster = df[df['cluster_id'] == cluster_id]
            avg_conf = cluster['ensemble_confidence'].mean()
            min_conf = cluster['ensemble_confidence'].min()
            max_conf = cluster['ensemble_confidence'].max()
            print(f"     Cluster {cluster_id}: {min_conf:.3f} - {max_conf:.3f} (avg: {avg_conf:.3f})")
        
        if len(low_confidence_clusters) > 5:
            print(f"     ... and {len(low_confidence_clusters) - 5} more clusters")
    
    print()
    
    approved_clusters = set()
    rejected_clusters = set()
    
    # Training data for ML model (same format as code-ensemble_loc.py)
    manual_decisions = []
    training_data = []
    
    for cluster_id in low_confidence_clusters:
        cluster_companies = df[df['cluster_id'] == cluster_id]
        
        # Calculate cluster metrics (same format as code-ensemble_loc.py)
        avg_confidence = cluster_companies['ensemble_confidence'].mean()
        max_confidence = cluster_companies['ensemble_confidence'].max()
        min_confidence = cluster_companies['ensemble_confidence'].min()
        
        # Determine confidence level
        if avg_confidence > 0.8:
            confidence_level = 'HIGH'
        elif avg_confidence >= 0.7:
            confidence_level = 'MEDIUM'
        else:
            confidence_level = 'LOW'
        
        # Create location key
        location_lat = cluster_companies['latitude'].iloc[0]
        location_lon = cluster_companies['longitude'].iloc[0]
        location_key = f"{location_lat}_{location_lon}"
        
        # Prepare training data features (same format as code-ensemble_loc.py)
        cluster_features = {
            'cluster_id': f"ENSEMBLE_{cluster_id}",
            'location_key': location_key,
            'location_lat': location_lat,
            'location_lon': location_lon,
            'cluster_size': len(cluster_companies),
            'avg_confidence': avg_confidence,
            'max_confidence': max_confidence,
            'min_confidence': min_confidence,
            'confidence_level': confidence_level,
            'num_edges': len(cluster_companies) * (len(cluster_companies) - 1) // 2,  # approximate
            'company_names': cluster_companies['companyName'].tolist(),
            'company_ids': cluster_companies['centralizedCompanyId'].tolist(),
            'company_domains': [''] * len(cluster_companies),  # placeholder
            'timezones': [''] * len(cluster_companies),  # placeholder
            'methods_agreeing': ['ensemble'],  # placeholder
            'timestamp': datetime.now().isoformat()
        }
        
        # Get user decision
        user_decision = get_user_approval(cluster_companies, cluster_id)
        
        if user_decision:
            approved_clusters.add(cluster_id)
            print(f"✓ Cluster {cluster_id} APPROVED")
            cluster_features['manual_decision'] = 'APPROVED'
            cluster_features['decision_reason'] = 'manual_approval'
            manual_decisions.append(cluster_features)
        else:
            rejected_clusters.add(cluster_id)
            print(f"✗ Cluster {cluster_id} REJECTED")
            cluster_features['manual_decision'] = 'REJECTED'
            cluster_features['decision_reason'] = 'manual_rejection'
            manual_decisions.append(cluster_features)
    
    # collect rejected companies for separate tracking
    rejected_companies = []
    for cluster_id in rejected_clusters:
        cluster = df[df['cluster_id'] == cluster_id]
        for _, company in cluster.iterrows():
            rejected_companies.append({
                'original_cluster_id': cluster_id,
                'centralizedCompanyId': company['centralizedCompanyId'],
                'companyName': company['companyName'],
                'city': company.get('city', 'N/A'),
                'state': company.get('state', 'N/A'),
                'latitude': company.get('latitude', 'N/A'),
                'longitude': company.get('longitude', 'N/A'),
                'ensemble_confidence': company['ensemble_confidence'],
                'rejection_reason': 'Manual rejection - low confidence cluster'
            })
    
    # update cluster assignments based on approval
    for cluster_id in rejected_clusters:
        df.loc[df['cluster_id'] == cluster_id, 'cluster_id'] = -1  # unclustered
    
    print(f"\n✅ APPROVAL SUMMARY:")
    print(f"   Approved clusters: {len(approved_clusters)}")
    print(f"   Rejected clusters: {len(rejected_clusters)}")
    print(f"   Rejected companies: {len(rejected_companies)}")
    print(f"   Total reviewed: {len(approved_clusters) + len(rejected_clusters)}")
    print(f"   Remaining clusters: {total_clusters - len(approved_clusters) - len(rejected_clusters)} (singletons + high-confidence)")
    
    # save rejected companies to separate file
    if rejected_companies:
        rejected_df = pd.DataFrame(rejected_companies)
        rejected_file = 'individual_output/new/csv_files/rejected/ensemble_location_rejected.csv'
        os.makedirs(os.path.dirname(rejected_file), exist_ok=True)
        rejected_df.to_csv(rejected_file, index=False)
        print(f"   📁 Rejected companies saved to: {rejected_file}")
    
    # save training data for ML model
    if manual_decisions:
        save_training_data(manual_decisions, training_data, 'ensemble_location_system')
        print(f"   🤖 Training data saved for ML model")
    
    return df, rejected_companies, manual_decisions

def save_training_data(manual_decisions, training_data, method_name):
    """Save training data for ML model in the same format as code-ensemble_loc.py"""
    from datetime import datetime
    import json
    
    # Create training data directory
    training_dir = 'individual_output/new/training_data'
    os.makedirs(training_dir, exist_ok=True)
    
    # Combine all training data
    all_training_data = manual_decisions + training_data
    
    if all_training_data:
        # Save as JSON for detailed analysis (same format as code-ensemble_loc.py)
        json_file = os.path.join(training_dir, f"{method_name}_training_data.json")
        with open(json_file, 'w') as f:
            json.dump(all_training_data, f, indent=2, default=str)
        
        # Save as CSV for ML training (same format as code-ensemble_loc.py)
        csv_file = os.path.join(training_dir, f"{method_name}_training_data.csv")
        
        # Flatten the data for CSV (same format as code-ensemble_loc.py)
        flattened_data = []
        for item in all_training_data:
            flat_item = {
                'cluster_id': item['cluster_id'],
                'location_key': item['location_key'],
                'location_lat': item['location_lat'],
                'location_lon': item['location_lon'],
                'cluster_size': item['cluster_size'],
                'avg_confidence': item['avg_confidence'],
                'max_confidence': item['max_confidence'],
                'min_confidence': item['min_confidence'],
                'confidence_level': item['confidence_level'],
                'num_edges': item['num_edges'],
                'company_names_count': len(item['company_names']),
                'company_domains_count': len([d for d in item['company_domains'] if d]),
                'timezones_count': len([t for t in item['timezones'] if t]),
                'methods_agreeing_count': len(item['methods_agreeing']),
                'methods_agreeing': ','.join(item['methods_agreeing']),
                'manual_decision': item['manual_decision'],
                'decision_reason': item['decision_reason'],
                'timestamp': item['timestamp']
            }
            flattened_data.append(flat_item)
        
        df_training = pd.DataFrame(flattened_data)
        df_training.to_csv(csv_file, index=False)
        
        print(f"\nTraining data saved:")
        print(f"  JSON: {json_file}")
        print(f"  CSV: {csv_file}")
        print(f"  Total samples: {len(all_training_data)}")
        print(f"  Manual decisions: {len(manual_decisions)}")
        print(f"  Auto-approved: {len(training_data)}")
    else:
        print("\nNo training data to save")

# ------------------------------
# STEP 4: Main ensemble pipeline
# ------------------------------

def main():
    """main ensemble location-first deduplication pipeline"""
    utils = SharedDeduplicationUtils()
    
    print("=" * 60)
    print("ENSEMBLE LOCATION-FIRST DEDUPLICATION PIPELINE")
    print("=" * 60)
    
    # create database connection
    engine = utils.create_database_connection()
    
    # load companies with location data
    print("loading company and location data...")
    query = """
    SELECT DISTINCT
      centralized_company.centralizedCompanyId,
      centralized_company.companyName,
      centralized_company.createdAt,
      centralized_company.updatedAt,
      MIN(location.streetAddress) AS streetAddress,
      MIN(location.city) AS city,
      MIN(location.state) AS state,
      MIN(location.zipcode) AS zipcode,
      MIN(location.country) AS country,
      MIN(ST_X(location.coordinate)) AS latitude,
      MIN(ST_Y(location.coordinate)) AS longitude
    FROM centralized_company
    JOIN company
      ON company.credentialledCentralizedCompanyId = centralized_company.centralizedCompanyId
    JOIN location
      ON location.locatableId = company.companyId
    WHERE location.coordinate IS NOT NULL
    GROUP BY 
      centralized_company.centralizedCompanyId,
      centralized_company.companyName,
      centralized_company.createdAt,
      centralized_company.updatedAt
    """
    
    companies_df = pd.read_sql(query, engine)
    print(f"loaded {len(companies_df)} companies with location data")
    
    # clean company names
    companies_df['clean_name'] = companies_df['companyName'].apply(utils.clean_company_name)
    
    # STEP 1: Build location-based clusters
    print("\nSTEP 1: Building location-based clusters...")
    location_G = build_location_clusters(companies_df, distance_threshold=0.01)  # 10 meters
    companies_df = utils.assign_clusters(companies_df, location_G)
    print(f"created {companies_df['cluster_id'].nunique() - 1} location-based clusters")
    
    # STEP 2: Apply ensemble name similarity within location clusters
    print("\nSTEP 2: Applying ensemble name similarity within location clusters...")
    
    # create new graph for ensemble clustering
    ensemble_G = nx.Graph()
    ensemble_G.add_nodes_from(companies_df.index)
    
    # initialize confidence column
    companies_df['ensemble_confidence'] = 0.0
    
    # process each location cluster
    for cluster_id in tqdm(companies_df['cluster_id'].unique(), desc="processing location clusters"):
        if cluster_id == -1:
            continue
            
        location_cluster = companies_df[companies_df['cluster_id'] == cluster_id]
        
        if len(location_cluster) <= 1:
            continue
        
        # apply ensemble name similarity within this location cluster
        indices = location_cluster.index.tolist()
        for i in range(len(indices)):
            for j in range(i + 1, len(indices)):
                idx1, idx2 = indices[i], indices[j]
                
                is_similar, confidence = are_companies_similar_ensemble(
                    companies_df.loc[idx1], 
                    companies_df.loc[idx2]
                )
                
                if is_similar:
                    ensemble_G.add_edge(idx1, idx2)
                    # store confidence for later use
                    companies_df.loc[idx1, 'ensemble_confidence'] = max(companies_df.loc[idx1, 'ensemble_confidence'], confidence)
                    companies_df.loc[idx2, 'ensemble_confidence'] = max(companies_df.loc[idx2, 'ensemble_confidence'], confidence)
    
    # STEP 3: Assign final ensemble clusters
    print("\nSTEP 3: Assigning final ensemble clusters...")
    companies_df = utils.assign_clusters(companies_df, ensemble_G)
    print(f"created {companies_df['cluster_id'].nunique() - 1} final ensemble clusters")
    
    # STEP 4: Interactive approval for low-confidence clusters with training data collection
    print("\nSTEP 4: Interactive approval for low-confidence clusters...")
    
    # Initialize training data collection (same format as code-ensemble_loc.py)
    manual_decisions = []
    training_data = []
    
    try:
        companies_df, rejected_companies, training_data = process_low_confidence_clusters(companies_df, confidence_threshold=0.60)
    except KeyboardInterrupt as e:
        if "User requested quit" in str(e):
            print(f"\n{'='*60}")
            print("🛑 MANUAL REVIEW INTERRUPTED BY USER")
            print(f"{'='*60}")
            print("Saving partial progress and training data...")
            
            # Save current state before quitting
            partial_file = 'individual_output/new/csv_files/partial/ensemble_location_partial.csv'
            os.makedirs(os.path.dirname(partial_file), exist_ok=True)
            companies_df.to_csv(partial_file, index=False)
            print(f"📁 Partial progress saved to: {partial_file}")
            
            # Save any training data collected so far
            if 'training_data' in locals() and training_data:
                save_training_data(manual_decisions, training_data, 'ensemble_location_system_partial')
                print(f"🤖 Partial training data saved for ML model")
            
            print("You can resume later by loading the partial file.")
            print("Exiting gracefully...")
            return
        else:
            raise e
    
    # STEP 5: Generate canonical companies
    print("\nSTEP 5: Generating canonical companies...")
    canonical_companies = []
    
    for cluster_id in companies_df['cluster_id'].unique():
        if cluster_id == -1:
            continue
            
        cluster = companies_df[companies_df['cluster_id'] == cluster_id]
        canonical = cluster.sort_values('updatedAt', ascending=False).iloc[0]
        
        canonical_companies.append({
            'canonical_id': f'ENSEMBLE_{cluster_id}',
            'original_id': canonical['centralizedCompanyId'],
            'company_name': canonical['companyName'],
            'cluster_size': len(cluster),
            'latitude': canonical['latitude'],
            'longitude': canonical['longitude'],
            'address': canonical['streetAddress'],
            'city': canonical['city'],
            'state': canonical['state'],
            'country': canonical['country'],
            'created_at': canonical['createdAt'],
            'updated_at': canonical['updatedAt']
        })
    
    canonical_companies = pd.DataFrame(canonical_companies)
    print(f"generated {len(canonical_companies)} canonical companies")
    
    # STEP 6: Save results
    print("\nSTEP 6: Saving results...")
    utils.save_results(companies_df, canonical_companies, 'ensemble_location')
    
    # STEP 7: Generate analysis report
    print("\nSTEP 7: Generating analysis report...")
    analysis_file = 'individual_output/new/analysis_files/ensemble_location_analysis.txt'
    
    with open(analysis_file, 'w') as f:
        f.write("ENSEMBLE LOCATION-FIRST CLUSTER ANALYSIS\n")
        f.write("=" * 60 + "\n\n")
        
        # overall statistics
        total_clusters = companies_df['cluster_id'].nunique() - 1
        total_clustered = len(companies_df[companies_df['cluster_id'] != -1])
        total_unclustered = len(companies_df[companies_df['cluster_id'] == -1])
        
        f.write(f"Overall Statistics:\n")
        f.write(f"-----------------\n")
        f.write(f"Total clusters: {total_clusters}\n")
        f.write(f"Total clustered companies: {total_clustered}\n")
        f.write(f"Total unclustered companies: {total_unclustered}\n")
        f.write(f"Average cluster size: {total_clustered/total_clusters:.2f}\n\n")
        
        # confidence distribution
        if 'ensemble_confidence' in companies_df.columns:
            confidence_stats = companies_df[companies_df['cluster_id'] != -1]['ensemble_confidence'].describe()
            f.write(f"Ensemble Confidence Statistics:\n")
            f.write(f"-----------------------------\n")
            f.write(f"Mean confidence: {confidence_stats['mean']:.3f}\n")
            f.write(f"Std confidence: {confidence_stats['std']:.3f}\n")
            f.write(f"Min confidence: {confidence_stats['min']:.3f}\n")
            f.write(f"Max confidence: {confidence_stats['max']:.3f}\n\n")
        
        # detailed cluster information (only clusters with >1 company)
        f.write("Detailed Cluster Information:\n")
        f.write("---------------------------\n")
        for cluster_id in sorted(companies_df['cluster_id'].unique()):
            if cluster_id == -1:
                continue
                
            cluster = companies_df[companies_df['cluster_id'] == cluster_id]
            
            # Only show clusters with more than one company
            if len(cluster) <= 1:
                continue
                
            avg_confidence = cluster['ensemble_confidence'].mean() if 'ensemble_confidence' in cluster.columns else 0.0
            
            f.write(f"\nCluster ENSEMBLE_{cluster_id} (Size: {len(cluster)}, Confidence: {avg_confidence:.3f}):\n")
            f.write("-" * 60 + "\n")
            
            for _, company in cluster.sort_values('updatedAt', ascending=False).iterrows():
                f.write(f"Company: {company['companyName']}\n")
                f.write(f"  Location: {company.get('city', 'N/A')}, {company.get('state', 'N/A')}\n")
                f.write(f"  Coordinates: ({company.get('latitude', 'N/A')}, {company.get('longitude', 'N/A')})\n")
                f.write(f"  Last updated: {company['updatedAt']}\n")
                f.write(f"  ID: {company['centralizedCompanyId']}\n")
                f.write("-" * 40 + "\n")
    
    print(f"analysis report saved to {analysis_file}")
    
    # Final summary
    print(f"\n{'='*60}")
    print("🎉 FINAL SUMMARY")
    print(f"{'='*60}")
    
    final_clusters = companies_df['cluster_id'].nunique() - 1
    final_clustered = len(companies_df[companies_df['cluster_id'] != -1])
    final_unclustered = len(companies_df[companies_df['cluster_id'] == -1])
    
    # Count clusters by size
    cluster_sizes = {}
    for cluster_id in companies_df['cluster_id'].unique():
        if cluster_id == -1:
            continue
        cluster = companies_df[companies_df['cluster_id'] == cluster_id]
        size = len(cluster)
        cluster_sizes[size] = cluster_sizes.get(size, 0) + 1
    
    print(f"📈 FINAL RESULTS:")
    print(f"   Total companies processed: {len(companies_df)}")
    print(f"   Final clusters created: {final_clusters}")
    print(f"   Companies clustered: {final_clustered}")
    print(f"   Companies unclustered: {final_unclustered}")
    print(f"   Clustering rate: {final_clustered/len(companies_df)*100:.1f}%")
    print()
    
    print(f"📊 CLUSTER SIZE DISTRIBUTION:")
    for size in sorted(cluster_sizes.keys()):
        count = cluster_sizes[size]
        print(f"   {size} company clusters: {count}")
    
    # Save training data (same format as code-ensemble_loc.py)
    save_training_data(manual_decisions, training_data, 'ensemble_location_system')
    
    print(f"\n✅ Ensemble location-first deduplication completed successfully!")
    print(f"   ✅ Metaphone similarity: IMPLEMENTED")
    print(f"   ✅ Multiple companies per cluster: SUPPORTED")
    print(f"   ✅ Interactive approval: COMPLETED")
    print(f"   ✅ Training data collection: COMPLETED")

if __name__ == "__main__":
    main() 