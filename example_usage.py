#!/usr/bin/env python3
"""
Example Usage of Company Deduplication System
============================================

This script demonstrates how to use the company deduplication system
with sample data for testing and development purposes.
"""

import pandas as pd
import numpy as np
from company_deduplication_system import (
    build_location_clusters,
    are_companies_similar_ensemble,
    process_low_confidence_clusters
)
from shared_deduplication_utils import SharedDeduplicationUtils

def create_sample_data():
    """Create sample company data for testing"""
    
    sample_companies = [
        # Similar companies in same location
        {
            'centralizedCompanyId': 'COMP001',
            'companyName': 'Acme Corporation Inc',
            'latitude': 40.7128,
            'longitude': -74.0060,
            'city': 'New York',
            'state': 'NY',
            'streetAddress': '123 Main St',
            'timezone': 'America/New_York',
            'createdAt': '2023-01-01',
            'updatedAt': '2023-12-01'
        },
        {
            'centralizedCompanyId': 'COMP002',
            'companyName': 'Acme Corp',
            'latitude': 40.7128,
            'longitude': -74.0060,
            'city': 'New York',
            'state': 'NY',
            'streetAddress': '123 Main St',
            'timezone': 'America/New_York',
            'createdAt': '2023-01-02',
            'updatedAt': '2023-12-02'
        },
        {
            'centralizedCompanyId': 'COMP003',
            'companyName': 'ACME Corporation',
            'latitude': 40.7128,
            'longitude': -74.0060,
            'city': 'New York',
            'state': 'NY',
            'streetAddress': '123 Main St',
            'timezone': 'America/New_York',
            'createdAt': '2023-01-03',
            'updatedAt': '2023-12-03'
        },
        
        # Different companies in same location
        {
            'centralizedCompanyId': 'COMP004',
            'companyName': 'Tech Solutions LLC',
            'latitude': 40.7128,
            'longitude': -74.0060,
            'city': 'New York',
            'state': 'NY',
            'streetAddress': '123 Main St',
            'timezone': 'America/New_York',
            'createdAt': '2023-01-04',
            'updatedAt': '2023-12-04'
        },
        
        # Similar companies in different location
        {
            'centralizedCompanyId': 'COMP005',
            'companyName': 'Acme Corporation',
            'latitude': 34.0522,
            'longitude': -118.2437,
            'city': 'Los Angeles',
            'state': 'CA',
            'streetAddress': '456 Oak Ave',
            'timezone': 'America/Los_Angeles',
            'createdAt': '2023-01-05',
            'updatedAt': '2023-12-05'
        },
        
        # Similar names, different companies
        {
            'centralizedCompanyId': 'COMP006',
            'companyName': 'Acme Technologies',
            'latitude': 40.7128,
            'longitude': -74.0060,
            'city': 'New York',
            'state': 'NY',
            'streetAddress': '123 Main St',
            'timezone': 'America/New_York',
            'createdAt': '2023-01-06',
            'updatedAt': '2023-12-06'
        },
        
        # Standalone company
        {
            'centralizedCompanyId': 'COMP007',
            'companyName': 'Global Industries Ltd',
            'latitude': 41.8781,
            'longitude': -87.6298,
            'city': 'Chicago',
            'state': 'IL',
            'streetAddress': '789 Pine St',
            'timezone': 'America/Chicago',
            'createdAt': '2023-01-07',
            'updatedAt': '2023-12-07'
        }
    ]
    
    return pd.DataFrame(sample_companies)

def run_sample_deduplication():
    """Run the deduplication system on sample data"""
    
    print("=" * 60)
    print("SAMPLE COMPANY DEDUPLICATION DEMO")
    print("=" * 60)
    
    # Create sample data
    print("Creating sample company data...")
    companies_df = create_sample_data()
    print(f"Created {len(companies_df)} sample companies")
    
    # Initialize components
    utils = SharedDeduplicationUtils()
    
    # Clean company names
    print("\nCleaning company names...")
    companies_df['clean_name'] = companies_df['companyName'].apply(utils.clean_company_name)
    
    # Step 1: Build location-based clusters
    print("\nSTEP 1: Building location-based clusters...")
    location_G = build_location_clusters(companies_df, distance_threshold=0.01)  # 10 meters
    companies_df = utils.assign_clusters(companies_df, location_G)
    print(f"Created {companies_df['cluster_id'].nunique() - 1} location-based clusters")
    
    # Display location clusters
    print("\nLocation-based clusters:")
    for cluster_id in sorted(companies_df['cluster_id'].unique()):
        if cluster_id == -1:
            continue
        cluster = companies_df[companies_df['cluster_id'] == cluster_id]
        print(f"  Location Cluster {cluster_id}: {len(cluster)} companies")
        for _, company in cluster.iterrows():
            print(f"    - {company['companyName']} ({company['city']}, {company['state']})")
    
    # Step 2: Apply ensemble name similarity within location clusters
    print("\nSTEP 2: Applying ensemble name similarity within location clusters...")
    
    # Create new graph for ensemble clustering
    import networkx as nx
    ensemble_G = nx.Graph()
    ensemble_G.add_nodes_from(companies_df.index)
    
    # Initialize confidence column
    companies_df['ensemble_confidence'] = 0.0
    
    # Process each location cluster
    for cluster_id in companies_df['cluster_id'].unique():
        if cluster_id == -1:
            continue
            
        location_cluster = companies_df[companies_df['cluster_id'] == cluster_id]
        
        if len(location_cluster) <= 1:
            continue
        
        print(f"\nProcessing location cluster {cluster_id}...")
        
        # Apply ensemble name similarity within this location cluster
        indices = location_cluster.index.tolist()
        for i in range(len(indices)):
            for j in range(i + 1, len(indices)):
                idx1, idx2 = indices[i], indices[j]
                
                is_similar, confidence = are_companies_similar_ensemble(
                    companies_df.loc[idx1], 
                    companies_df.loc[idx2]
                )
                
                print(f"  Comparing: '{companies_df.loc[idx1, 'companyName']}' vs '{companies_df.loc[idx2, 'companyName']}'")
                print(f"    Similar: {is_similar}, Confidence: {confidence:.3f}")
                
                if is_similar:
                    ensemble_G.add_edge(idx1, idx2)
                    # Store confidence for later use
                    companies_df.loc[idx1, 'ensemble_confidence'] = max(
                        companies_df.loc[idx1, 'ensemble_confidence'], confidence
                    )
                    companies_df.loc[idx2, 'ensemble_confidence'] = max(
                        companies_df.loc[idx2, 'ensemble_confidence'], confidence
                    )
    
    # Step 3: Assign final ensemble clusters
    print("\nSTEP 3: Assigning final ensemble clusters...")
    companies_df = utils.assign_clusters(companies_df, ensemble_G)
    print(f"Created {companies_df['cluster_id'].nunique() - 1} final ensemble clusters")
    
    # Display final clusters
    print("\nFinal ensemble clusters:")
    for cluster_id in sorted(companies_df['cluster_id'].unique()):
        if cluster_id == -1:
            continue
        cluster = companies_df[companies_df['cluster_id'] == cluster_id]
        avg_confidence = cluster['ensemble_confidence'].mean()
        print(f"  Ensemble Cluster {cluster_id}: {len(cluster)} companies (avg confidence: {avg_confidence:.3f})")
        for _, company in cluster.iterrows():
            print(f"    - {company['companyName']} (confidence: {company['ensemble_confidence']:.3f})")
    
    # Step 4: Generate canonical companies
    print("\nSTEP 4: Generating canonical companies...")
    canonical_companies = utils.generate_canonical_companies(companies_df)
    print(f"Generated {len(canonical_companies)} canonical companies")
    
    # Display canonical companies
    print("\nCanonical companies:")
    for _, canonical in canonical_companies.iterrows():
        print(f"  {canonical['canonical_id']}: {canonical['company_name']} (cluster size: {canonical['cluster_size']})")
    
    # Step 5: Save results
    print("\nSTEP 5: Saving results...")
    utils.save_results(companies_df, canonical_companies, 'sample_ensemble')
    
    # Step 6: Generate analysis report
    print("\nSTEP 6: Generating analysis report...")
    analysis_file = 'individual_output/new/analysis_files/sample_ensemble_analysis.txt'
    utils.generate_analysis_report(companies_df, analysis_file, 'sample_ensemble')
    
    print("\nSample deduplication completed successfully!")
    print(f"Check the output files in individual_output/new/ for detailed results.")

def test_similarity_methods():
    """Test individual similarity methods"""
    
    print("\n" + "=" * 60)
    print("TESTING SIMILARITY METHODS")
    print("=" * 60)
    
    from company_deduplication_system import (
        calculate_fuzzy_jaccard_similarity,
        calculate_metaphone_similarity,
        calculate_jaro_winkler_similarity
    )
    
    # Test cases
    test_cases = [
        ("Acme Corporation Inc", "Acme Corp"),
        ("Acme Corporation Inc", "ACME Corporation"),
        ("Acme Corporation Inc", "Acme Technologies"),
        ("Tech Solutions LLC", "Tech Solutions Limited"),
        ("Global Industries Ltd", "Global Industries Limited")
    ]
    
    for name1, name2 in test_cases:
        print(f"\nComparing: '{name1}' vs '{name2}'")
        
        fuzzy_jaccard = calculate_fuzzy_jaccard_similarity(name1, name2)
        metaphone = calculate_metaphone_similarity(name1, name2)
        jaro_winkler = calculate_jaro_winkler_similarity(name1, name2)
        
        print(f"  Fuzzy Jaccard: {fuzzy_jaccard:.3f}")
        print(f"  Metaphone: {metaphone:.3f}")
        print(f"  Jaro-Winkler: {jaro_winkler:.3f}")
        
        # Test ensemble logic
        is_similar, confidence = are_companies_similar_ensemble(
            pd.Series({'companyName': name1}),
            pd.Series({'companyName': name2})
        )
        print(f"  Ensemble Similar: {is_similar}, Confidence: {confidence:.3f}")

if __name__ == "__main__":
    # Test similarity methods
    test_similarity_methods()
    
    # Run sample deduplication
    run_sample_deduplication() 