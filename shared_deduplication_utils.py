# shared deduplication utilities
# contains all common functionality used across different deduplication methods

import pandas as pd
from sqlalchemy import create_engine
from getpass import getpass
import re
import networkx as nx
from tqdm import tqdm
import numpy as np
import os

class SharedDeduplicationUtils:
    """shared utilities for all deduplication methods"""
    
    def __init__(self):
        pass
    
    def clean_company_name(self, name):
        """
        normalize company name: lowercase, strip punctuation, unify suffixes
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

    def get_company_keywords(self, name):
        """
        extract key identifying words from company name
        """
        if pd.isna(name):
            return set()
        
        # common words to exclude
        stop_words = {'the', 'and', 'of', 'in', 'to', 'for', 'a', 'an', 'on', 'at', 'with', 'by', 'city', 'town', 'county', 'of', 'in'}
        
        # clean and split
        name = self.clean_company_name(name)
        words = set(name.split())
        
        # remove stop words and short words
        words = {w for w in words if w not in stop_words and len(w) > 2}
        
        return words

    def get_blocking_key(self, name):
        """
        create a blocking key for efficient comparison
        returns first 3 characters of first word and first 2 of second word if exists
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

    def build_similarity_graph(self, df, similarity_function, threshold):
        """
        build a graph where edges connect similar companies using blocking
        """
        G = nx.Graph()
        G.add_nodes_from(df.index)
        
        # create blocking keys
        df['blocking_key'] = df['clean_name'].apply(self.get_blocking_key)
        
        # group by blocking key
        for key, group in tqdm(df.groupby('blocking_key')):
            if len(group) <= 1:
                continue
                
            # only compare within the same block
            indices = group.index.tolist()
            for i in range(len(indices)):
                for j in range(i + 1, len(indices)):
                    idx1, idx2 = indices[i], indices[j]
                    
                    if similarity_function(df.loc[idx1], df.loc[idx2], threshold):
                        G.add_edge(idx1, idx2)
        
        return G

    def assign_clusters(self, df, G):
        """
        assign connected component id as cluster id
        """
        cluster_id_map = {}
        for cluster_id, component in enumerate(nx.connected_components(G)):
            for idx in component:
                cluster_id_map[idx] = cluster_id
        df['cluster_id'] = df.index.map(cluster_id_map).fillna(-1).astype(int)
        return df

    def generate_canonical_companies(self, df):
        """
        for each cluster, select canonical row by latest updatedat
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
                'company_name': canonical['companyName'],
                'timezone': canonical['timezone'],
                'cluster_size': len(cluster),
                'created_at': canonical['createdAt'],
                'updated_at': canonical['updatedAt']
            })
        return pd.DataFrame(canonical_list)

    def create_database_connection(self):
        """
        create database connection
        """
        db_password = getpass("enter your database password: ")
        username = 'rematter_api_service'
        host = 'mysql.alpha.rematter.com'
        port = 3306
        database = 'rematter_default'

        connection_string = f"mysql+pymysql://{username}:{db_password}@{host}:{port}/{database}"
        engine = create_engine(connection_string)
        return engine

    def load_companies_from_database(self, engine):
        """
        load all companies from database
        """
        # first, get total count
        count_query = "SELECT COUNT(*) as count FROM rematter_default.centralized_company;"
        total_count = pd.read_sql(count_query, engine).iloc[0]['count']
        print(f"total companies in database: {total_count}")

        # load all companies
        print("loading all companies...")
        query = """
        SELECT centralizedCompanyId, companyName, timezone, createdAt, updatedAt
        FROM rematter_default.centralized_company;
        """
        companies_df = pd.read_sql(query, engine)
        print(f"loaded {len(companies_df)} rows.")

        # clean company names
        print("cleaning company names...")
        companies_df['clean_name'] = companies_df['companyName'].apply(self.clean_company_name)
        
        return companies_df

    def save_results(self, companies_df, canonical_companies, output_prefix):
        """
        save results to csv files in organized folder structure
        """
        print("saving results to csv files...")
        
        # save clustered companies to clustered folder
        clustered_file = f'individual_output/new/csv_files/clustered/{output_prefix}_clustered.csv'
        companies_df.to_csv(clustered_file, index=False)
        print(f"saved clustered companies to {clustered_file}")
        
        # save canonical companies to canonical folder
        canonical_file = f'individual_output/new/csv_files/canonical/{output_prefix}_canonical.csv'
        canonical_companies.to_csv(canonical_file, index=False)
        print(f"saved canonical companies to {canonical_file}")

    def generate_analysis_report(self, companies_df, output_file, method_name, threshold=None):
        """
        generate detailed cluster analysis report
        """
        print("generating detailed cluster analysis...")
        with open(output_file, 'w') as f:
            f.write(f"DETAILED CLUSTER ANALYSIS ({method_name.upper()})\n")
            if threshold:
                f.write(f"Threshold: {threshold}\n")
            f.write("=" * 60 + "\n\n")
            
            # overall statistics
            total_clusters = companies_df['cluster_id'].nunique() - 1  # -1 for unclustered
            total_clustered = len(companies_df[companies_df['cluster_id'] != -1])
            total_unclustered = len(companies_df[companies_df['cluster_id'] == -1])
            
            f.write(f"overall statistics:\n")
            f.write(f"-----------------\n")
            f.write(f"total number of clusters: {total_clusters}\n")
            f.write(f"total companies in clusters: {total_clustered}\n")
            f.write(f"total unclustered companies: {total_unclustered}\n")
            f.write(f"average cluster size: {total_clustered/total_clusters:.2f}\n\n")
            
            # cluster size distribution
            cluster_sizes = companies_df[companies_df['cluster_id'] != -1]['cluster_id'].value_counts()
            f.write("cluster size distribution:\n")
            f.write("------------------------\n")
            for size, count in cluster_sizes.value_counts().sort_index().items():
                f.write(f"clusters with {size} companies: {count}\n")
            f.write("\n")
            
            # detailed cluster information
            f.write("detailed cluster information:\n")
            f.write("---------------------------\n")
            for cluster_id in sorted(companies_df['cluster_id'].unique()):
                if cluster_id == -1:
                    continue
                    
                cluster = companies_df[companies_df['cluster_id'] == cluster_id]
                f.write(f"\ncluster {cluster_id} (size: {len(cluster)}):\n")
                f.write("-" * 50 + "\n")
                
                # sort by updatedat to show most recent first
                for _, company in cluster.sort_values('updatedAt', ascending=False).iterrows():
                    f.write(f"company: {company['companyName']}\n")
                    f.write(f"  timezone: {company['timezone']}\n")
                    f.write(f"  last updated: {company['updatedAt']}\n")
                    f.write(f"  id: {company['centralizedCompanyId']}\n")
                    f.write("-" * 30 + "\n")

    def run_deduplication_pipeline(self, similarity_function, threshold, method_name, output_prefix):
        """
        run complete deduplication pipeline
        """
        print("=" * 60)
        print(f"STARTING {method_name.upper()} DEDUPLICATION PIPELINE")
        print("=" * 60)
        
        # create database connection
        engine = self.create_database_connection()
        
        # load companies
        companies_df = self.load_companies_from_database(engine)
        
        # build similarity graph
        print(f"building similarity graph with {method_name} matching...")
        G = self.build_similarity_graph(companies_df, similarity_function, threshold)
        
        # assign clusters
        print("assigning cluster ids...")
        companies_df = self.assign_clusters(companies_df, G)
        print(f"found {companies_df['cluster_id'].nunique() - 1} clusters.")
        
        # generate canonical companies
        print("generating canonical companies...")
        canonical_companies = self.generate_canonical_companies(companies_df)
        print(f"generated {len(canonical_companies)} canonical companies.")
        
        # save results
        self.save_results(companies_df, canonical_companies, output_prefix)
        
        # generate analysis
        analysis_file = f'individual_output/new/analysis_files/{output_prefix}_analysis.txt'
        self.generate_analysis_report(companies_df, analysis_file, method_name, threshold)
        
        print("done — results saved to csv and analysis file")
        return canonical_companies 