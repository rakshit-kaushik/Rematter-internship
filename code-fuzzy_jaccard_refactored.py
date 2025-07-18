# fuzzy jaccard deduplication using shared utilities
# only contains the unique fuzzy jaccard logic

import pandas as pd
from fuzzywuzzy import fuzz
from shared_deduplication_utils import SharedDeduplicationUtils

def calculate_name_similarity(name1, name2):
    """
    calculate similarity between two company names using multiple metrics
    this is the unique logic for fuzzy jaccard method
    """
    # get keyword sets
    utils = SharedDeduplicationUtils()
    keywords1 = utils.get_company_keywords(name1)
    keywords2 = utils.get_company_keywords(name2)
    
    if not keywords1 or not keywords2:
        return 0
    
    # calculate jaccard similarity of keywords
    intersection = len(keywords1.intersection(keywords2))
    union = len(keywords1.union(keywords2))
    jaccard_sim = intersection / union if union > 0 else 0
    
    # calculate token sort ratio (handles word order differences)
    token_sort_ratio = fuzz.token_sort_ratio(name1, name2) / 100
    
    # calculate token set ratio (handles extra/missing words)
    token_set_ratio = fuzz.token_set_ratio(name1, name2) / 100
    
    # weighted combination of metrics
    similarity = (jaccard_sim * 0.4 + token_sort_ratio * 0.3 + token_set_ratio * 0.3)
    
    return similarity

def are_companies_similar(row1, row2, name_thresh=0.85):
    """
    determine if two companies are similar using fuzzy jaccard criteria
    this is the unique logic for fuzzy jaccard method
    """
    # calculate name similarity
    name_sim = calculate_name_similarity(row1['clean_name'], row2['clean_name'])
    
    # check timezone match
    timezone_match = (row1['timezone'] == row2['timezone']) and row1['timezone']
    
    # companies are similar if:
    # 1. names are very similar (high threshold)
    # 2. and they're in the same timezone (if timezone exists)
    
    if timezone_match and name_sim >= name_thresh:
        return True
    return False

def main():
    """main function to run fuzzy jaccard deduplication"""
    
    # create shared utilities instance
    utils = SharedDeduplicationUtils()
    
    # run the pipeline with fuzzy jaccard method
    results = utils.run_deduplication_pipeline(
        similarity_function=are_companies_similar,
        threshold=0.85,
        method_name="fuzzy jaccard",
        output_prefix="fuzzy_jaccard"
    )
    
    print("fuzzy jaccard deduplication completed successfully!")

if __name__ == "__main__":
    main() 