# jaro-winkler deduplication using shared utilities
# only contains the unique jaro-winkler logic

import pandas as pd
import jellyfish
import os
from shared_deduplication_utils import SharedDeduplicationUtils

def jaro_winkler_similarity(name1, name2):
    """
    calculate jaro-winkler similarity between two company names
    this is the unique logic for jaro-winkler method
    """
    if not name1 or not name2:
        return 0
    
    # get filtered keywords for both names
    utils = SharedDeduplicationUtils()
    keywords1 = utils.get_company_keywords(name1)
    keywords2 = utils.get_company_keywords(name2)
    
    # convert sets to strings for jaro-winkler processing
    keywords1_str = ' '.join(keywords1)
    keywords2_str = ' '.join(keywords2)
    
    # if either name has no meaningful keywords after filtering, return 0
    if not keywords1_str or not keywords2_str:
        return 0
    
    return jellyfish.jaro_winkler_similarity(keywords1_str, keywords2_str)

def are_companies_similar(row1, row2, jw_thresh):
    """
    determine if two companies are similar using jaro-winkler criteria
    this is the unique logic for jaro-winkler method
    """
    name_sim = jaro_winkler_similarity(row1['clean_name'], row2['clean_name'])
    timezone_match = (row1['timezone'] == row2['timezone']) and row1['timezone']
    return timezone_match and name_sim >= jw_thresh

def process_threshold(utils, jw_thresh, output_dir):
    """
    process data for a specific jaro-winkler threshold
    this is the unique logic for jaro-winkler method (multiple thresholds)
    """
    print(f"\n{'='*60}")
    print(f"processing threshold: {jw_thresh}")
    print(f"{'='*60}")
    
    # create database connection
    engine = utils.create_database_connection()
    
    # load companies
    companies_df = utils.load_companies_from_database(engine)
    
    # build similarity graph for this threshold
    G = utils.build_similarity_graph(companies_df, are_companies_similar, jw_thresh)
    
    # assign clusters
    df_clustered = utils.assign_clusters(companies_df.copy(), G)
    num_clusters = df_clustered['cluster_id'].nunique() - 1
    print(f"found {num_clusters} clusters for threshold {jw_thresh}")
    
    # generate canonical companies
    canonical_companies = utils.generate_canonical_companies(df_clustered)
    print(f"generated {len(canonical_companies)} canonical companies")
    
    # save results
    threshold_suffix = f"jw_{jw_thresh}".replace('.', '_')
    
    # save clustered data to organized folder structure
    clustered_file = f'individual_output/new/csv_files/clustered/jaro_{threshold_suffix}_clustered.csv'
    df_clustered.to_csv(clustered_file, index=False)
    print(f"saved clustered companies to {clustered_file}")
    
    # save canonical companies to organized folder structure
    canonical_file = f'individual_output/new/csv_files/canonical/jaro_{threshold_suffix}_canonical.csv'
    canonical_companies.to_csv(canonical_file, index=False)
    print(f"saved canonical companies to {canonical_file}")
    
    # generate detailed analysis
    analysis_file = f'individual_output/new/analysis_files/jaro_{threshold_suffix}_analysis.txt'
    utils.generate_analysis_report(df_clustered, analysis_file, f"jaro-winkler (threshold: {jw_thresh})", jw_thresh)
    print(f"saved analysis report to {analysis_file}")
    
    return {
        'threshold': jw_thresh,
        'num_clusters': num_clusters,
        'num_canonical': len(canonical_companies),
        'total_clustered': len(df_clustered[df_clustered['cluster_id'] != -1]),
        'total_unclustered': len(df_clustered[df_clustered['cluster_id'] == -1])
    }

def main():
    """main function to run jaro-winkler deduplication with multiple thresholds"""
    
    # create shared utilities instance
    utils = SharedDeduplicationUtils()
    
    # define thresholds to test
    thresholds = [0.85, 0.87, 0.89, 0.91, 0.93]
    
    # create output directory
    output_dir = 'jaro_threshold_results'
    os.makedirs(output_dir, exist_ok=True)
    
    # process each threshold
    results = []
    for threshold in thresholds:
        result = process_threshold(utils, threshold, output_dir)
        results.append(result)
    
    # generate summary report
    summary_file = 'individual_output/new/analysis_files/jaro_threshold_comparison_summary.txt'
    with open(summary_file, 'w') as f:
        f.write("JARO-WINKLER THRESHOLD COMPARISON SUMMARY\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"{'Threshold':<12} {'Clusters':<10} {'Canonical':<10} {'Clustered':<10} {'Unclustered':<12} {'Avg Size':<10}\n")
        f.write("-" * 70 + "\n")
        
        for result in results:
            avg_size = result['total_clustered'] / result['num_clusters'] if result['num_clusters'] > 0 else 0
            f.write(f"{result['threshold']:<12} {result['num_clusters']:<10} {result['num_canonical']:<10} {result['total_clustered']:<10} {result['total_unclustered']:<12} {avg_size:<10.2f}\n")
        
        f.write("\n" + "=" * 50 + "\n")
        f.write("RECOMMENDATIONS:\n")
        f.write("- Lower thresholds (0.85-0.87): More aggressive clustering, catches more duplicates\n")
        f.write("- Higher thresholds (0.91-0.93): Conservative clustering, fewer false positives\n")
        f.write("- Medium thresholds (0.89): Balanced approach\n")
    
    print(f"\nall thresholds processed. summary saved to {summary_file}")
    print("jaro-winkler deduplication completed successfully!")

if __name__ == "__main__":
    main() 