#!/usr/bin/env python3
"""
ensemble location-based company deduplication
combines multiple similarity methods with location data for robust deduplication
"""

import pandas as pd
from sqlalchemy import create_engine
from getpass import getpass
import re
from fuzzywuzzy import fuzz
import networkx as nx
from tqdm import tqdm
import numpy as np
from collections import Counter
import os
from collections import defaultdict
import json
import math
import jellyfish
from doublemetaphone import doublemetaphone
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from datetime import datetime
from shared_deduplication_utils import *

# Add missing functions that are called in main()

def load_and_prepare_data(input_file):
    """Load and prepare data from CSV file"""
    try:
        df = pd.read_csv(input_file)
        print(f"Loaded {len(df)} companies from {input_file}")
        
        # Ensure required columns exist
        required_columns = ['centralizedCompanyId', 'centralized_company_name', 'lat', 'lon']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            print(f"Warning: Missing columns: {missing_columns}")
            # Create dummy columns if missing
            for col in missing_columns:
                if col == 'centralizedCompanyId':
                    df[col] = range(len(df))
                elif col == 'centralized_company_name':
                    df[col] = f"Company_{range(len(df))}"
                elif col in ['lat', 'lon']:
                    df[col] = 0.0
        
        # Clean company names
        df['centralized_company_name'] = df['centralized_company_name'].fillna('').astype(str).apply(clean_company_name)
        
        return df
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

def perform_clustering(similarity_matrix, threshold=0.7):
    """Perform clustering using the similarity matrix"""
    # Use hierarchical clustering
    from scipy.cluster.hierarchy import linkage, fcluster
    from scipy.spatial.distance import squareform
    
    # Convert similarity to distance (1 - similarity)
    distance_matrix = 1 - similarity_matrix
    
    # Convert to condensed distance matrix
    condensed_distances = squareform(distance_matrix)
    
    # Perform hierarchical clustering
    linkage_matrix = linkage(condensed_distances, method='ward')
    
    # Get cluster labels
    cluster_labels = fcluster(linkage_matrix, t=1-threshold, criterion='distance')
    
    return cluster_labels

def create_cluster_dataframe(df, cluster_labels):
    """Create a dataframe with cluster information"""
    df_clustered = df.copy()
    df_clustered['cluster_id'] = cluster_labels
    
    # Calculate cluster statistics
    cluster_stats = []
    for cluster_id in df_clustered['cluster_id'].unique():
        cluster_data = df_clustered[df_clustered['cluster_id'] == cluster_id]
        cluster_stats.append({
            'cluster_id': cluster_id,
            'size': len(cluster_data),
            'companies': cluster_data['centralizedCompanyId'].tolist(),
            'names': cluster_data['centralized_company_name'].tolist()
        })
    
    return df_clustered, cluster_stats

def save_results(df_clustered, canonical_df, cluster_stats, output_dir, prefix):
    """Save results to files"""
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Save clustered data
    clustered_file = os.path.join(output_dir, f"{prefix}_clustered.csv")
    df_clustered.to_csv(clustered_file, index=False)
    
    # Save canonical companies
    canonical_file = os.path.join(output_dir, f"{prefix}_canonical.csv")
    canonical_df.to_csv(canonical_file, index=False)
    
    # Save analysis
    analysis_file = os.path.join(output_dir, f"{prefix}_analysis.txt")
    with open(analysis_file, 'w') as f:
        f.write(f"Ensemble Location Deduplication Analysis\n")
        f.write(f"Generated on: {datetime.now()}\n\n")
        f.write(f"Total companies processed: {len(df_clustered)}\n")
        f.write(f"Total clusters found: {len(cluster_stats)}\n")
        f.write(f"Average cluster size: {np.mean([c['size'] for c in cluster_stats]):.2f}\n")
        f.write(f"Largest cluster size: {max([c['size'] for c in cluster_stats])}\n")
        f.write(f"Smallest cluster size: {min([c['size'] for c in cluster_stats])}\n\n")
        
        f.write("Cluster Details:\n")
        for cluster in cluster_stats:
            f.write(f"Cluster {cluster['cluster_id']}: {cluster['size']} companies\n")
            for name in cluster['names']:
                f.write(f"  - {name}\n")
            f.write("\n")
    
    return clustered_file, canonical_file, analysis_file

def create_canonical_companies(df_clustered, cluster_stats):
    """Create canonical companies from clustered data"""
    canonical_list = []
    
    for cluster in cluster_stats:
        cluster_data = df_clustered[df_clustered['cluster_id'] == cluster['cluster_id']]
        
        # Select canonical record (most recent or first)
        try:
            canonical = cluster_data.sort_values('updatedAt', ascending=False).iloc[0]
        except:
            canonical = cluster_data.iloc[0]
        
        # Calculate average location
        avg_lat = cluster_data['lat'].mean()
        avg_lon = cluster_data['lon'].mean()
        
        canonical_list.append({
            'canonical_id': f"CLUSTER_{cluster['cluster_id']}",
            'original_id': canonical['centralizedCompanyId'],
            'company_name': canonical['centralized_company_name'],
            'cluster_size': len(cluster_data),
            'location_lat': avg_lat,
            'location_lon': avg_lon,
            'company_ids': cluster_data['centralizedCompanyId'].tolist(),
            'company_names': cluster_data['centralized_company_name'].tolist()
        })
    
    return pd.DataFrame(canonical_list)

# building a few helper functions

def clean_company_name(name):
    """
    Normalize company name: lowercase, strip punctuation, unify suffixes.
    """
    if pd.isna(name):
        return ""
    name = str(name).lower()
    name = re.sub(r'[^\w\s]', ' ', name)
    name = re.sub(r'\s+', ' ', name).strip()
    suffixes = {'inc': 'inc', 'incorporated': 'inc',
                'llc': 'llc', 'ltd': 'ltd', 'limited': 'ltd',
                'corp': 'corp', 'corporation': 'corp',
                'co': 'co', 'company': 'co'}
    words = name.split()
    if words and words[-1] in suffixes:
        words[-1] = suffixes[words[-1]]
    return ' '.join(words)


def get_company_keywords(name):
    """
    Extract key identifying words from company name.
    """
    if pd.isna(name):
        return set()
    
    # added stop words like city, town and all too because of how common they were. 
    stop_words = {'the', 'and', 'of', 'in', 'to', 'for', 'a', 'an', 'on', 'at', 'with', 'by', 'city', 'town', 'county', 'of', 'in'}
    
    # split the name into words
    name = clean_company_name(name)
    words = set(name.split())
    
    # removing stop words
    words = {w for w in words if w not in stop_words and len(w) > 2}
    
    return words

def calculate_name_similarity(name1, name2):
    """
    Calculate similarity between two company names using multiple metrics.
    """
    keywords1 = get_company_keywords(name1)
    keywords2 = get_company_keywords(name2)
    
    if not keywords1 or not keywords2:
        return 0
    
    # getting jaccard similarity of keywords
    intersection = len(keywords1.intersection(keywords2))
    union = len(keywords1.union(keywords2))
    jaccard_sim = intersection / union if union > 0 else 0
    
    # Calculate token sort ratio (handles word order differences)
    token_sort_ratio = fuzz.token_sort_ratio(name1, name2) / 100
    
    # Calculate token set ratio (handles extra/missing words)
    token_set_ratio = fuzz.token_set_ratio(name1, name2) / 100
    
    # Weighted combination of metrics
    similarity = (jaccard_sim * 0.4 + token_sort_ratio * 0.3 + token_set_ratio * 0.3)
    
    return similarity

def calculate_jaro_winkler_score(name1, name2):
    if not name1 or not name2:
        return 0
    return jellyfish.jaro_winkler_similarity(name1, name2)

def calculate_metaphone_similarity(name1, name2):
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

def are_companies_similar(row1, row2, name_thresh=0.85):
    """
    Determine if two companies are similar using multiple criteria.
    """
    # Calculate name similarity
    name_sim = calculate_name_similarity(row1['clean_name'], row2['clean_name'])
    
    # Check timezone match
    timezone_match = (row1['timezone'] == row2['timezone']) and row1['timezone']
    
    # Companies are similar if:
    # 1. Names are very similar (high threshold)
    # 2. AND they're in the same timezone (if timezone exists)
    
    
    if timezone_match and name_sim >= name_thresh:
        return True
    return False


def get_blocking_key(name):
    """
    Create a blocking key for efficient comparison.
    Returns first 3 characters of first word and first 2 of second word if exists.
    """
    if pd.isna(name):
        return ""
    words = str(name).lower().split()
    if not words:
        return ""
    key = words[0][:3]
    if len(words) > 1:
        key += words[1][:2]
    return key

def build_similarity_graph(df):
    """
    Build a graph where edges connect similar companies (no blocking, all pairs compared).
    """
    G = nx.Graph()
    G.add_nodes_from(df.index)

    # Compare all pairs within the dataframe
    indices = df.index.tolist()
    for i in range(len(indices)):
        for j in range(i + 1, len(indices)):
            idx1, idx2 = indices[i], indices[j]
            if are_companies_similar(df.loc[idx1], df.loc[idx2]):
                G.add_edge(idx1, idx2)
    return G

def assign_clusters(df, G):
    """
    Assign connected component ID as cluster ID.
    """
    cluster_id_map = {}
    for cluster_id, component in enumerate(nx.connected_components(G)):
        for idx in component:
            cluster_id_map[idx] = cluster_id
    df['cluster_id'] = df.index.map(cluster_id_map).fillna(-1).astype(int)
    return df

def generate_canonical_companies(df):
    """
    For each cluster, select canonical row by latest updatedAt.
    """
    canonical_list = []
    for cluster_id in df['cluster_id'].unique():
        if cluster_id == -1:
            continue
        cluster = df[df['cluster_id'] == cluster_id]
        canonical = cluster.sort_values('updatedAt', ascending=False).iloc[0]
        canonical_list.append({
            'canonical_id': f'CANON_{cluster_id}',
            'original_id': canonical['centralizedCompanyId'],
            'company_name': canonical['centralized_company_name'],
            'timezone': canonical['timezone'],
            'cluster_size': len(cluster),
            'created_at': canonical['createdAt'],
            'updated_at': canonical['updatedAt']
        })
    return pd.DataFrame(canonical_list)

def calculate_name_ensemble_confidence(pair_methods, total_methods):
    """
    Calculate ensemble confidence for a pair based on how many methods agree.
    pair_methods: list of methods that agree for this pair
    total_methods: list of all possible methods
    """
    agreement_count = len(pair_methods)
    agreement_ratio = agreement_count / len(total_methods)
    
    # Calculate weighted average of similarity scores (like company_deduplication_system.py)
    method_scores = {}
    if 'jaccard_fuzzy' in pair_methods:
        method_scores['jaccard_fuzzy'] = 0.85  # threshold value
    if 'jaro_winkler' in pair_methods:
        method_scores['jaro_winkler'] = 0.89  # threshold value
    if 'metaphone' in pair_methods:
        method_scores['metaphone'] = 0.91  # threshold value
    
    # Calculate weighted average score for confidence (like company_deduplication_system.py)
    if method_scores:
        weighted_score = sum(method_scores.values()) / len(method_scores)
        final_confidence = (agreement_ratio + weighted_score) / 2
    else:
        final_confidence = agreement_ratio
    
    return {
        'agreement_count': agreement_count,
        'total_methods': len(total_methods),
        'final_confidence': final_confidence
    }

# STEP 3: Load location data from database (like code-location-2.py)

def load_location_data_from_db():
    """Load location data directly from database like code-location-2.py"""
    print("Loading company and location data from database...")
    
    db_password = getpass("Enter your database password: ")
    username = 'rematter_api_service'
    host = 'mysql.alpha.rematter.com'
    port = 3306
    database = 'rematter_default'
    
    connection_string = f"mysql+pymysql://{username}:{db_password}@{host}:{port}/{database}"
    engine = create_engine(connection_string)
    
    query = """
    SELECT DISTINCT
      centralized_company.centralizedCompanyId,
      centralized_company.companyName AS centralized_company_name,
      centralized_company.companyDomain,
      centralized_company.createdAt,
      centralized_company.updatedAt,
      centralized_company.timezone,
      -- Take the first location for each centralized company
      MIN(location.streetAddress) AS streetAddress,
      MIN(location.city) AS city,
      MIN(location.state) AS state,
      MIN(location.zipcode) AS zipcode,
      MIN(location.country) AS country,
      -- Use the first non-null coordinate
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
      centralized_company.companyDomain,
      centralized_company.createdAt,
      centralized_company.updatedAt,
      centralized_company.timezone
    """
    
    companies_df = pd.read_sql(query, engine)
    print(f"Loaded {len(companies_df)} companies with location data from database.")
    
    # Clean company names
    print("Cleaning company names...")
    companies_df['clean_name'] = companies_df['centralized_company_name'].apply(clean_company_name)
    
    return companies_df

def create_location_clusters(location_df, decimal_places=4):
    """Create location clusters by rounding coordinates to 4 decimal places"""
    print(f"Creating location clusters (rounded to {decimal_places} decimal places)...")
    
    # Ensure latitude and longitude are numeric
    location_df['latitude'] = pd.to_numeric(location_df['latitude'], errors='coerce')
    location_df['longitude'] = pd.to_numeric(location_df['longitude'], errors='coerce')
    # Add rounded coordinates
    location_df['rounded_lat'] = location_df['latitude'].apply(lambda x: round(x, decimal_places) if pd.notna(x) else None)
    location_df['rounded_lon'] = location_df['longitude'].apply(lambda x: round(x, decimal_places) if pd.notna(x) else None)
    
    # Create location key
    location_df['location_key'] = location_df.apply(
        lambda row: f"{row['rounded_lat']}_{row['rounded_lon']}" 
        if pd.notna(row['rounded_lat']) and pd.notna(row['rounded_lon']) 
        else 'NO_LOCATION', axis=1
    )
    
    # Group by location key
    location_clusters = {}
    cluster_id = 0
    
    for location_key, group in location_df.groupby('location_key'):
        if location_key != 'NO_LOCATION' and len(group) > 1:
            location_clusters[location_key] = {
                'cluster_id': cluster_id,
                'companies': group,
                'lat': group['rounded_lat'].iloc[0],
                'lon': group['rounded_lon'].iloc[0],
                'size': len(group)
            }
            cluster_id += 1
        else:
            # Single companies or no location - assign individual clusters
            for _, company in group.iterrows():
                location_clusters[f"INDIVIDUAL_{company['centralizedCompanyId']}"] = {
                    'cluster_id': cluster_id,
                    'companies': pd.DataFrame([company]),
                    'lat': company.get('rounded_lat'),
                    'lon': company.get('rounded_lon'),
                    'size': 1
                }
                cluster_id += 1
    
    print(f"Created {len(location_clusters)} location clusters")
    
    # Show some statistics
    cluster_sizes = [cluster['size'] for cluster in location_clusters.values()]
    print(f"Location cluster statistics:")
    print(f"  Total clusters: {len(location_clusters)}")
    print(f"  Average size: {np.mean(cluster_sizes):.2f}")
    print(f"  Max size: {max(cluster_sizes)}")
    print(f"  Clusters with >1 company: {len([s for s in cluster_sizes if s > 1])}")
    
    return location_clusters

def cluster_within_location(location_cluster, total_name_methods):
    companies = location_cluster['companies']
    company_ids = companies['centralizedCompanyId'].tolist()
    n = len(company_ids)
    pair_methods = {}

    # For every pair, calculate all three similarities
    for i in range(n):
        for j in range(i + 1, n):
            idx1 = company_ids[i]
            idx2 = company_ids[j]
            row1 = companies.iloc[i]
            row2 = companies.iloc[j]
            methods_agree = []
            # Jaccard+Fuzzy
            sim_jf = calculate_name_similarity(row1['clean_name'], row2['clean_name'])
            if sim_jf > 0.85:
                methods_agree.append('jaccard_fuzzy')
            # Jaro-Winkler
            sim_jw = calculate_jaro_winkler_score(row1['clean_name'], row2['clean_name'])
            if sim_jw > 0.89:
                methods_agree.append('jaro_winkler')
            # Metaphone (relaxed: Jaro-Winkler similarity of metaphone codes > 0.91)
            sim_meta = calculate_metaphone_similarity(row1['clean_name'], row2['clean_name'])
            if sim_meta > 0.91:
                methods_agree.append('metaphone')
            if methods_agree:
                pair_methods[(idx1, idx2)] = methods_agree

    # Build ensemble graph: connect pairs if ANY method matches
    G = nx.Graph()
    G.add_nodes_from(company_ids)
    for (idx1, idx2), methods in pair_methods.items():
        if len(methods) >= 1:
            G.add_edge(idx1, idx2, methods=methods)

    # Assign cluster IDs based on connected components
    cluster_id_map = {}
    for cluster_id, component in enumerate(nx.connected_components(G)):
        for company_id in component:
            cluster_id_map[company_id] = cluster_id
    companies_with_clusters = companies.copy()
    companies_with_clusters['name_ensemble_cluster_id'] = companies_with_clusters['centralizedCompanyId'].map(cluster_id_map).fillna(-1).astype(int)
    # Return the DataFrame and the pair_methods dictionary
    return companies_with_clusters, pair_methods

def generate_canonical_companies_ensemble(location_clusters, total_name_methods):
    """Generate canonical companies with location-first approach"""
    canonical_list = []
    rejected_clusters = []
    
    # ML Training Data Collection
    training_data = []
    manual_decisions = []
    
    print(f"Processing {len(location_clusters)} location clusters...")
    
    processed_clusters = 0
    confidence_stats = {'HIGH': 0, 'MEDIUM': 0, 'LOW': 0, 'REJECTED': 0}
    
    # First pass: collect all clusters and identify low-confidence ones
    low_confidence_clusters = []
    high_confidence_clusters = []
    
    print("Analyzing all location clusters...")
    for location_key, location_cluster in tqdm(location_clusters.items(), desc="Analyzing clusters"):
        if location_cluster['size'] == 1:
            # Single company in location - high confidence
            high_confidence_clusters.append((location_key, location_cluster, None, None))
            continue
        
        # Multiple companies in location - need name ensemble clustering
        companies_with_clusters, pair_methods = cluster_within_location(
            location_cluster, total_name_methods
        )

        # Process each name ensemble cluster within this location
        for cluster_id in companies_with_clusters['name_ensemble_cluster_id'].unique():
            cluster = companies_with_clusters[companies_with_clusters['name_ensemble_cluster_id'] == cluster_id]
            if len(cluster) == 1:
                # Singleton cluster: high confidence
                high_confidence_clusters.append((location_key, location_cluster, cluster_id, cluster))
                continue
            
            # Clustered companies - calculate confidence
            cluster_indices = set(cluster['centralizedCompanyId'])
            cluster_confidence_scores = []
            methods_agreeing = set()
            cluster_id_list = list(cluster_indices)
            for i in range(len(cluster_id_list)):
                for j in range(i + 1, len(cluster_id_list)):
                    pair = (cluster_id_list[i], cluster_id_list[j])
                    if pair in pair_methods:
                        confidence = calculate_name_ensemble_confidence(pair_methods[pair], total_name_methods)['final_confidence']
                        cluster_confidence_scores.append(confidence)
                        methods_agreeing.update(pair_methods[pair])
                    elif (cluster_id_list[j], cluster_id_list[i]) in pair_methods:
                        confidence = calculate_name_ensemble_confidence(pair_methods[(cluster_id_list[j], cluster_id_list[i])], total_name_methods)['final_confidence']
                        cluster_confidence_scores.append(confidence)
                        methods_agreeing.update(pair_methods[(cluster_id_list[j], cluster_id_list[i])])
            
            # Calculate cluster metrics
            if cluster_confidence_scores:
                avg_cluster_confidence = np.mean(cluster_confidence_scores)
                max_cluster_confidence = max(cluster_confidence_scores)
                min_cluster_confidence = min(cluster_confidence_scores)
            else:
                avg_cluster_confidence = 0
                max_cluster_confidence = 0
                min_cluster_confidence = 0
            
            # Classify confidence level
            if avg_cluster_confidence > 0.8:
                confidence_level = 'HIGH'
            elif avg_cluster_confidence >= 0.7:
                confidence_level = 'MEDIUM'
            else:
                confidence_level = 'LOW'
            
            # Store cluster info
            cluster_info = {
                'location_key': location_key,
                'location_cluster': location_cluster,
                'cluster_id': cluster_id,
                'cluster': cluster,
                'avg_confidence': avg_cluster_confidence,
                'max_confidence': max_cluster_confidence,
                'min_confidence': min_cluster_confidence,
                'confidence_level': confidence_level,
                'methods_agreeing': methods_agreeing,
                'pair_methods': pair_methods
            }
            
            if avg_cluster_confidence < 0.60:
                low_confidence_clusters.append(cluster_info)
            else:
                high_confidence_clusters.append((location_key, location_cluster, cluster_id, cluster))
    
    # Show statistics
    print(f"\n{'='*60}")
    print("CLUSTER ANALYSIS SUMMARY")
    print(f"{'='*60}")
    print(f"Total location clusters: {len(location_clusters)}")
    print(f"High-confidence clusters: {len(high_confidence_clusters)} - NO REVIEW NEEDED")
    print(f"Low-confidence clusters: {len(low_confidence_clusters)} - MANUAL REVIEW REQUIRED")
    print(f"Confidence threshold: 0.60")
    print()
    
    if len(low_confidence_clusters) == 0:
        print("✅ No low-confidence clusters found! All clusters are high-confidence.")
        # Process all high-confidence clusters
        print("Processing all high-confidence clusters...")
        for location_key, location_cluster, cluster_id, cluster in tqdm(high_confidence_clusters, desc="Processing high-confidence clusters"):
            # Process high-confidence cluster (same logic as before)
            if location_cluster['size'] == 1:
                # Single company in location - add directly
                company = location_cluster['companies'].iloc[0]
                canonical_list.append({
                    'canonical_id': f"LOC_{location_cluster['cluster_id']}_SINGLE",
                    'original_id': company['centralizedCompanyId'],
                    'company_name': company['centralized_company_name'],
                    'company_domain': company.get('companyDomain', ''),
                    'timezone': company.get('timezone', ''),
                    'cluster_size': 1,
                    'avg_cluster_confidence': 1.0,
                    'confidence_level': 'HIGH',
                    'location_lat': location_cluster['lat'],
                    'location_lon': location_cluster['lon'],
                    'location_key': location_key,
                    'company_ids': [company['centralizedCompanyId']],
                    'company_names': [company['centralized_company_name']],
                    'created_at': company.get('createdAt', ''),
                    'updated_at': company.get('updatedAt', '')
                })
                confidence_stats['HIGH'] += 1
                processed_clusters += 1
            elif cluster_id is None:
                # Singleton name cluster
                company = cluster.iloc[0]
                canonical_list.append({
                    'canonical_id': f"LOC_{location_cluster['cluster_id']}_SINGLE_{company['centralizedCompanyId']}",
                    'original_id': company['centralizedCompanyId'],
                    'company_name': company['centralized_company_name'],
                    'company_domain': company.get('companyDomain', ''),
                    'timezone': company.get('timezone', ''),
                    'cluster_size': 1,
                    'avg_cluster_confidence': 1.0,
                    'confidence_level': 'HIGH',
                    'location_lat': location_cluster['lat'],
                    'location_lon': location_cluster['lon'],
                    'location_key': location_key,
                    'company_ids': [company['centralizedCompanyId']],
                    'company_names': [company['centralized_company_name']],
                    'created_at': company.get('createdAt', ''),
                    'updated_at': company.get('updatedAt', '')
                })
                confidence_stats['HIGH'] += 1
                processed_clusters += 1
            else:
                # High-confidence multi-company cluster
                try:
                    canonical = cluster.sort_values('updatedAt', ascending=False).iloc[0]
                except:
                    canonical = cluster.iloc[0]
                
                canonical_list.append({
                    'canonical_id': f"LOC_{location_cluster['cluster_id']}_NAME_{cluster_id}",
                    'original_id': canonical['centralizedCompanyId'],
                    'company_name': canonical['centralized_company_name'],
                    'company_domain': canonical.get('companyDomain', ''),
                    'timezone': canonical.get('timezone', ''),
                    'cluster_size': len(cluster),
                    'avg_cluster_confidence': 1.0,  # High confidence
                    'confidence_level': 'HIGH',
                    'location_lat': location_cluster['lat'],
                    'location_lon': location_cluster['lon'],
                    'location_key': location_key,
                    'company_ids': cluster['centralizedCompanyId'].tolist(),
                    'company_names': cluster['centralized_company_name'].tolist(),
                    'created_at': canonical.get('createdAt', ''),
                    'updated_at': canonical.get('updatedAt', '')
                })
                confidence_stats['HIGH'] += 1
                processed_clusters += 1
        
        # Save training data
        save_training_data(manual_decisions, training_data, 'ensemble_location_loc')
        return pd.DataFrame(canonical_list), rejected_clusters
    
    print(f"🔍 Found {len(low_confidence_clusters)} low-confidence clusters for manual review")
    print()
    
    # Process all high-confidence clusters first
    print("Processing all high-confidence clusters...")
    for location_key, location_cluster, cluster_id, cluster in tqdm(high_confidence_clusters, desc="Processing high-confidence clusters"):
        # Process high-confidence cluster (same logic as before)
        if location_cluster['size'] == 1:
            # Single company in location - add directly
            company = location_cluster['companies'].iloc[0]
            canonical_list.append({
                'canonical_id': f"LOC_{location_cluster['cluster_id']}_SINGLE",
                'original_id': company['centralizedCompanyId'],
                'company_name': company['centralized_company_name'],
                'company_domain': company.get('companyDomain', ''),
                'timezone': company.get('timezone', ''),
                'cluster_size': 1,
                'avg_cluster_confidence': 1.0,
                'confidence_level': 'HIGH',
                'location_lat': location_cluster['lat'],
                'location_lon': location_cluster['lon'],
                'location_key': location_key,
                'company_ids': [company['centralizedCompanyId']],
                'company_names': [company['centralized_company_name']],
                'created_at': company.get('createdAt', ''),
                'updated_at': company.get('updatedAt', '')
            })
            confidence_stats['HIGH'] += 1
            processed_clusters += 1
        elif cluster_id is None:
            # Singleton name cluster
            company = cluster.iloc[0]
            canonical_list.append({
                'canonical_id': f"LOC_{location_cluster['cluster_id']}_SINGLE_{company['centralizedCompanyId']}",
                'original_id': company['centralizedCompanyId'],
                'company_name': company['centralized_company_name'],
                'company_domain': company.get('companyDomain', ''),
                'timezone': company.get('timezone', ''),
                'cluster_size': 1,
                'avg_cluster_confidence': 1.0,
                'confidence_level': 'HIGH',
                'location_lat': location_cluster['lat'],
                'location_lon': location_cluster['lon'],
                'location_key': location_key,
                'company_ids': [company['centralizedCompanyId']],
                'company_names': [company['centralized_company_name']],
                'created_at': company.get('createdAt', ''),
                'updated_at': company.get('updatedAt', '')
            })
            confidence_stats['HIGH'] += 1
            processed_clusters += 1
        else:
            # High-confidence multi-company cluster
            try:
                canonical = cluster.sort_values('updatedAt', ascending=False).iloc[0]
            except:
                canonical = cluster.iloc[0]
            
            canonical_list.append({
                'canonical_id': f"LOC_{location_cluster['cluster_id']}_NAME_{cluster_id}",
                'original_id': canonical['centralizedCompanyId'],
                'company_name': canonical['centralized_company_name'],
                'company_domain': canonical.get('companyDomain', ''),
                'timezone': canonical.get('timezone', ''),
                'cluster_size': len(cluster),
                'avg_cluster_confidence': 1.0,  # High confidence
                'confidence_level': 'HIGH',
                'location_lat': location_cluster['lat'],
                'location_lon': location_cluster['lon'],
                'location_key': location_key,
                'company_ids': cluster['centralizedCompanyId'].tolist(),
                'company_names': cluster['centralized_company_name'].tolist(),
                'created_at': canonical.get('createdAt', ''),
                'updated_at': canonical.get('updatedAt', '')
            })
            confidence_stats['HIGH'] += 1
            processed_clusters += 1
    
    print(f"✅ Processed {len(high_confidence_clusters)} high-confidence clusters")
    print()
    
    # Now process low-confidence clusters with interactive prompts
    print("Processing low-confidence clusters with manual review...")
    for i, cluster_info in enumerate(low_confidence_clusters, 1):
        location_key = cluster_info['location_key']
        location_cluster = cluster_info['location_cluster']
        cluster_id = cluster_info['cluster_id']
        cluster = cluster_info['cluster']
        avg_cluster_confidence = cluster_info['avg_confidence']
        max_cluster_confidence = cluster_info['max_confidence']
        min_cluster_confidence = cluster_info['min_confidence']
        confidence_level = cluster_info['confidence_level']
        methods_agreeing = cluster_info['methods_agreeing']
        pair_methods = cluster_info['pair_methods']
        
        print(f"\nLow confidence cluster {i}/{len(low_confidence_clusters)}:")
        print(f"  Location: ({location_cluster['lat']:.4f}, {location_cluster['lon']:.4f})")
        print(f"  Cluster size: {len(cluster)}")
        print(f"  Average confidence: {avg_cluster_confidence:.3f}")
        print(f"  Companies:")
        for _, company in cluster.iterrows():
            print(f"    - {company['centralized_company_name']} (ID: {company['centralizedCompanyId']})")
        
        # Prepare training data features
        cluster_features = {
            'cluster_id': f"LOC_{location_cluster['cluster_id']}_NAME_{cluster_id}",
            'location_key': location_key,
            'location_lat': location_cluster['lat'],
            'location_lon': location_cluster['lon'],
            'cluster_size': len(cluster),
            'avg_confidence': avg_cluster_confidence,
            'max_confidence': max_cluster_confidence,
            'min_confidence': min_cluster_confidence,
            'confidence_level': confidence_level,
            'num_edges': len(cluster_info.get('cluster_confidence_scores', [])) // 2,
            'company_names': cluster['centralized_company_name'].tolist(),
            'company_ids': cluster['centralizedCompanyId'].tolist(),
            'company_domains': cluster.get('companyDomain', '').tolist(),
            'timezones': cluster.get('timezone', '').tolist(),
            'methods_agreeing': list(methods_agreeing),
            'timestamp': datetime.now().isoformat()
        }
        
        while True:
            response = input(f"\nDo you want to cluster these companies? (y/n/q to quit): ").lower().strip()
            if response in ['y', 'yes']:
                print("  ✓ Clustering accepted")
                cluster_features['manual_decision'] = 'APPROVED'
                cluster_features['decision_reason'] = 'manual_approval'
                manual_decisions.append(cluster_features)
                
                # Add to canonical list
                try:
                    canonical = cluster.sort_values('updatedAt', ascending=False).iloc[0]
                except:
                    canonical = cluster.iloc[0]
                
                canonical_list.append({
                    'canonical_id': f"LOC_{location_cluster['cluster_id']}_NAME_{cluster_id}",
                    'original_id': canonical['centralizedCompanyId'],
                    'company_name': canonical['centralized_company_name'],
                    'company_domain': canonical.get('companyDomain', ''),
                    'timezone': canonical.get('timezone', ''),
                    'cluster_size': len(cluster),
                    'avg_cluster_confidence': avg_cluster_confidence,
                    'confidence_level': confidence_level,
                    'location_lat': location_cluster['lat'],
                    'location_lon': location_cluster['lon'],
                    'location_key': location_key,
                    'company_ids': cluster['centralizedCompanyId'].tolist(),
                    'company_names': cluster['centralized_company_name'].tolist(),
                    'created_at': canonical.get('createdAt', ''),
                    'updated_at': canonical.get('updatedAt', '')
                })
                confidence_stats[confidence_level] += 1
                processed_clusters += 1
                break
            elif response in ['n', 'no']:
                print("  ✗ Clustering rejected - adding companies individually")
                cluster_features['manual_decision'] = 'REJECTED'
                cluster_features['decision_reason'] = 'manual_rejection'
                manual_decisions.append(cluster_features)
                
                # Add companies individually instead of as a cluster
                for _, company in cluster.iterrows():
                    canonical_list.append({
                        'canonical_id': f"LOC_{location_cluster['cluster_id']}_REJECTED_{company['centralizedCompanyId']}",
                        'original_id': company['centralizedCompanyId'],
                        'company_name': company['centralized_company_name'],
                        'company_domain': company.get('companyDomain', ''),
                        'timezone': company.get('timezone', ''),
                        'cluster_size': 1,
                        'avg_cluster_confidence': 0.0,
                        'confidence_level': 'REJECTED',
                        'location_lat': location_cluster['lat'],
                        'location_lon': location_cluster['lon'],
                        'location_key': location_key,
                        'company_ids': [company['centralizedCompanyId']],
                        'company_names': [company['centralized_company_name']],
                        'created_at': company.get('createdAt', ''),
                        'updated_at': company.get('updatedAt', '')
                    })
                    confidence_stats['REJECTED'] += 1
                    processed_clusters += 1
                break
            elif response in ['q', 'quit']:
                print("  🚪 Quitting - saving partial progress...")
                # Save training data collected so far
                save_training_data(manual_decisions, training_data)
                return pd.DataFrame(canonical_list), rejected_clusters
            else:
                print("Please enter 'y', 'n', or 'q'")
    
    print(f"Completed - {processed_clusters} canonical companies generated")
    print(f"Rejected clusters: {len(rejected_clusters)}")
    print(f"Confidence Level Distribution:")
    print(f"  HIGH confidence: {confidence_stats['HIGH']} clusters")
    print(f"  MEDIUM confidence: {confidence_stats['MEDIUM']} clusters") 
    print(f"  LOW confidence: {confidence_stats['LOW']} clusters")
    print(f"  REJECTED clusters: {confidence_stats['REJECTED']} clusters")
    
    # Save training data
    save_training_data(manual_decisions, training_data)
    
    return pd.DataFrame(canonical_list), rejected_clusters

def save_training_data(manual_decisions, training_data):
    """Save training data for ML model development"""
    output_dir = "individual_output/new/training_data"
    os.makedirs(output_dir, exist_ok=True)
    
    # Combine all training data
    all_training_data = manual_decisions + training_data
    
    if all_training_data:
        # Save as JSON for detailed analysis
        json_file = os.path.join(output_dir, "ensemble_location_training_data.json")
        with open(json_file, 'w') as f:
            json.dump(all_training_data, f, indent=2, default=str)
        
        # Save as CSV for ML training
        csv_file = os.path.join(output_dir, "ensemble_location_training_data.csv")
        
        # Flatten the data for CSV
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

def generate_cleaned_cluster_list(canonical_companies):
    """Generate a cleaned list with company IDs and top company name"""
    cleaned_list = []
    
    for _, cluster in canonical_companies.iterrows():
        cleaned_list.append({
            'cluster_id': cluster['canonical_id'],
            'top_company_name': cluster['company_name'],
            'top_company_id': cluster['original_id'],
            'cluster_size': cluster['cluster_size'],
            'confidence_score': cluster['avg_cluster_confidence'],
            'confidence_level': cluster['confidence_level'],
            'location_lat': cluster['location_lat'],
            'location_lon': cluster['location_lon'],
            'company_ids': cluster['company_ids']
        })
    
    return pd.DataFrame(cleaned_list)

def ensemble_similarity(row1, row2, weights={'name': 0.4, 'location': 0.6}):
    """calculate ensemble similarity combining name and location"""
    name_sim = fuzz.ratio(str(row1['company_name']), str(row2['company_name'])) / 100.0
    loc_sim = location_similarity(row1, row2)
    
    # weighted combination
    ensemble_sim = weights['name'] * name_sim + weights['location'] * loc_sim
    return ensemble_sim

def main():
    print("=" * 60)
    print("ENSEMBLE LOCATION-BASED DEDUPLICATION (Location-first, CSV import)")
    print("=" * 60)

    # Step 1: Load data from the database
    df = load_location_data_from_db()
    if df is None or len(df) == 0:
        print("No data loaded from database.")
        return

    # Clean name for ensemble methods
    if 'clean_name' not in df.columns:
        df['clean_name'] = df['centralized_company_name'].apply(clean_company_name)

    # Rename lat/lon columns if needed
    if 'latitude' not in df.columns and 'lat' in df.columns:
        df['latitude'] = df['lat']
    if 'longitude' not in df.columns and 'lon' in df.columns:
        df['longitude'] = df['lon']

    # Step 2: Create location clusters by rounding coordinates
    location_clusters = create_location_clusters(df, decimal_places=4)

    # Step 3: Use existing CSV data (already loaded)
    print(f"Using existing data with {len(df)} companies")

    # Step 4: For each location cluster, run ensemble name clustering
    total_name_methods = ['jaccard_fuzzy', 'jaro_winkler', 'metaphone']
    canonical_df, _ = generate_canonical_companies_ensemble(location_clusters, total_name_methods)

    # Step 5: Save results
    output_dir = "individual_output/new"
    os.makedirs(output_dir, exist_ok=True)
    canonical_file = os.path.join(output_dir, "ensemble_location_canonical.csv")
    canonical_df.to_csv(canonical_file, index=False)
    print(f"\nEnsemble location deduplication complete!")
    print(f"Canonical companies saved to {canonical_file}")

if __name__ == "__main__":
    main()