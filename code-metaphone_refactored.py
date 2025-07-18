# metaphone deduplication using shared utilities
# only contains the unique metaphone logic

import pandas as pd
from doublemetaphone import doublemetaphone
import jellyfish
from shared_deduplication_utils import SharedDeduplicationUtils

def get_metaphone_codes(name):
    """
    get metaphone codes for a company name after removing stop words
    returns both primary and secondary metaphone codes
    this is the unique logic for metaphone method
    """
    if pd.isna(name) or not name:
        return "", ""
    
    # get filtered keywords first
    utils = SharedDeduplicationUtils()
    keywords = utils.get_company_keywords(name)
    
    # convert set to string for metaphone processing
    filtered_name = ' '.join(keywords)
    
    # if no meaningful keywords after filtering, return empty codes
    if not filtered_name:
        return "", ""
    
    # get metaphone codes from filtered name
    primary, secondary = doublemetaphone(filtered_name)
    
    return primary, secondary

def calculate_metaphone_similarity(name1, name2):
    """
    calculate similarity between two company names using metaphone codes
    this is the unique logic for metaphone method
    """
    if pd.isna(name1) or pd.isna(name2) or not name1 or not name2:
        return 0
    
    # get metaphone codes for both names
    primary1, secondary1 = get_metaphone_codes(name1)
    primary2, secondary2 = get_metaphone_codes(name2)
    
    # calculate jaro-winkler similarity between metaphone codes
    similarities = []
    
    # primary to primary
    if primary1 and primary2:
        sim = jellyfish.jaro_winkler_similarity(primary1, primary2)
        similarities.append(sim)
    
    # primary to secondary
    if primary1 and secondary2:
        sim = jellyfish.jaro_winkler_similarity(primary1, secondary2)
        similarities.append(sim)
    
    # secondary to primary
    if secondary1 and primary2:
        sim = jellyfish.jaro_winkler_similarity(secondary1, primary2)
        similarities.append(sim)
    
    # secondary to secondary
    if secondary1 and secondary2:
        sim = jellyfish.jaro_winkler_similarity(secondary1, secondary2)
        similarities.append(sim)
    
    # return the highest similarity found
    return max(similarities) if similarities else 0

def are_companies_similar(row1, row2, metaphone_thresh=0.85):
    """
    determine if two companies are similar using metaphone matching
    this is the unique logic for metaphone method
    """
    # calculate metaphone similarity
    metaphone_sim = calculate_metaphone_similarity(row1['companyName'], row2['companyName'])
    
    # check timezone match (optional - can be disabled)
    timezone_match = (row1['timezone'] == row2['timezone']) and row1['timezone']
    
    # companies are similar if:
    # 1. metaphone similarity is high enough
    # 2. and they're in the same timezone (if timezone exists)
    
    if timezone_match and metaphone_sim >= metaphone_thresh:
        return True
    
    # alternative: only use metaphone similarity without timezone requirement
    # uncomment the line below and comment out the above if you want to ignore timezone
    # return metaphone_sim >= metaphone_thresh
    
    return False

def main():
    """main function to run metaphone deduplication"""
    
    # create shared utilities instance
    utils = SharedDeduplicationUtils()
    
    # run the pipeline with metaphone method
    results = utils.run_deduplication_pipeline(
        similarity_function=are_companies_similar,
        threshold=0.85,
        method_name="metaphone",
        output_prefix="metaphone"
    )
    
    print("metaphone deduplication completed successfully!")

if __name__ == "__main__":
    main() 