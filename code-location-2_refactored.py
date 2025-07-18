# ------------------------------
# STEP 1: Import libraries
# ------------------------------

import pandas as pd
import math
import networkx as nx
from tqdm import tqdm
from shared_deduplication_utils import SharedDeduplicationUtils

# ------------------------------
# STEP 2: Location-specific helper functions
# ------------------------------

def haversine_distance(lat1, lon1, lat2, lon2):
    """
    calculate the great circle distance between two points 
    on the earth (specified in decimal degrees)
    """
    # convert decimal degrees to radians
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    
    # haversine formula
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
    c = 2 * math.asin(math.sqrt(a))
    
    # radius of earth in kilometers
    r = 6371
    
    return c * r

def get_geographic_blocking_key(lat, lon, precision=2):
    """
    create a geographic blocking key by rounding coordinates to reduce comparisons
    precision=2 means round to 2 decimal places (roughly 1km blocks)
    precision=3 means round to 3 decimal places (roughly 100m blocks)
    """
    lat_rounded = round(lat, precision)
    lon_rounded = round(lon, precision)
    return f"{lat_rounded},{lon_rounded}"

def location_similarity_function(row1, row2, distance_threshold):
    """
    similarity function for location-based deduplication
    returns true if companies are within distance_threshold km of each other
    """
    lat1, lon1 = row1['latitude'], row1['longitude']
    lat2, lon2 = row2['latitude'], row2['longitude']
    
    # check for null coordinates
    if pd.isna(lat1) or pd.isna(lon1) or pd.isna(lat2) or pd.isna(lon2):
        return False
    
    distance = haversine_distance(lat1, lon1, lat2, lon2)
    return distance <= distance_threshold

def build_location_similarity_graph(df, distance_threshold=0.005):
    """
    build similarity graph for location-based deduplication using geographic blocking
    """
    G = nx.Graph()
    G.add_nodes_from(df.index)
    
    # create geographic blocking keys (round to 3 decimal places for ~100m blocks)
    df['blocking_key'] = df.apply(
        lambda row: get_geographic_blocking_key(row['latitude'], row['longitude'], precision=3), 
        axis=1
    )
    
    # track statistics
    total_blocks = 0
    total_edges_created = 0
    
    print("processing companies with geographic blocking strategy...")
    
    # process each geographic block
    for key, group in tqdm(df.groupby('blocking_key'), desc="processing geographic blocks"):
        if len(group) <= 1:
            continue
            
        total_blocks += 1
        # get indices for this block
        block_indices = group.index.tolist()
        
        # add edges between companies that are close enough
        block_edges = 0
        for i in range(len(block_indices)):
            for j in range(i + 1, len(block_indices)):
                idx1, idx2 = block_indices[i], block_indices[j]
                
                if location_similarity_function(df.loc[idx1], df.loc[idx2], distance_threshold):
                    G.add_edge(idx1, idx2)
                    block_edges += 1
        
        total_edges_created += block_edges
    
    print(f"\nclustering statistics:")
    print(f"  total geographic blocks processed: {total_blocks}")
    print(f"  total edges created: {total_edges_created}")
    print(f"  distance threshold: {distance_threshold} km")
    
    return G

# ------------------------------
# STEP 3: Main execution
# ------------------------------

def main():
    # initialize shared utilities
    utils = SharedDeduplicationUtils()
    
    print("=" * 60)
    print("STARTING LOCATION-BASED DEDUPLICATION PIPELINE")
    print("=" * 60)
    
    # create database connection
    engine = utils.create_database_connection()
    
    # load companies with location data
    print("loading company and location data from database...")
    query = """
    SELECT DISTINCT
      centralized_company.centralizedCompanyId,
      centralized_company.companyName AS centralized_company_name,
      centralized_company.companyDomain,
      centralized_company.createdAt,
      centralized_company.updatedAt,
      -- take the first location for each centralized company
      MIN(location.streetAddress) AS streetAddress,
      MIN(location.city) AS city,
      MIN(location.state) AS state,
      MIN(location.zipcode) AS zipcode,
      MIN(location.country) AS country,
      -- use the first non-null coordinate
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
      centralized_company.updatedAt
    """
    
    companies_df = pd.read_sql(query, engine)
    print(f"loaded {len(companies_df)} companies with location data from database.")
    
    # build location similarity graph
    print("finding companies by geographic proximity with blocking strategy...")
    G = build_location_similarity_graph(companies_df, distance_threshold=0.005)  # 5 meter threshold
    
    # assign clusters
    print("assigning cluster ids...")
    companies_df = utils.assign_clusters(companies_df, G)
    print(f"found {companies_df['cluster_id'].nunique() - 1} clusters.")
    
    # generate canonical companies (using location-specific canonical generation)
    print("generating canonical company records...")
    canonical_companies = []
    
    for cluster_id in companies_df['cluster_id'].unique():
        if cluster_id == -1:
            continue
            
        cluster = companies_df[companies_df['cluster_id'] == cluster_id]
        
        # select the most recent record as canonical
        canonical = cluster.sort_values('updatedAt', ascending=False).iloc[0]
        
        canonical_companies.append({
            'canonical_id': f'CANON_{cluster_id}',
            'original_id': canonical['centralizedCompanyId'],
            'company_name': canonical['centralized_company_name'],
            'company_domain': canonical.get('companyDomain', ''),
            'cluster_size': len(cluster),
            'latitude': canonical['latitude'],
            'longitude': canonical['longitude'],
            'address': canonical['streetAddress'],
            'city': canonical['city'],
            'state': canonical['state'],
            'country': canonical['country'],
            'created_at': canonical.get('createdAt', ''),
            'updated_at': canonical.get('updatedAt', '')
        })
    
    canonical_companies = pd.DataFrame(canonical_companies)
    print(f"generated {len(canonical_companies)} canonical companies.")
    
    # save results using shared utilities
    utils.save_results(companies_df, canonical_companies, 'location')
    
    # generate location-specific analysis report
    print("generating detailed cluster analysis...")
    analysis_file = 'individual_output/new/analysis_files/location_analysis.txt'
    
    with open(analysis_file, 'w') as f:
        f.write("DETAILED CLUSTER ANALYSIS (GEOGRAPHIC PROXIMITY - COMPANIES WITH LOCATIONS)\n")
        f.write("======================================================================\n\n")
        
        # overall statistics
        total_clusters = companies_df['cluster_id'].nunique() - 1  # -1 for unclustered
        total_clustered = len(companies_df[companies_df['cluster_id'] != -1])
        total_unclustered = len(companies_df[companies_df['cluster_id'] == -1])
        
        f.write(f"overall statistics:\n")
        f.write(f"-----------------\n")
        f.write(f"total number of clusters: {total_clusters}\n")
        f.write(f"total companies in clusters: {total_clustered}\n")
        f.write(f"total unclustered companies: {total_unclustered}\n")
        if total_clusters > 0:
            f.write(f"average cluster size: {total_clustered/total_clusters:.2f}\n")
        f.write(f"distance threshold: 5 meters (0.005 km)\n\n")
        
        # cluster size distribution
        if total_clusters > 0:
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
            
            # calculate cluster center
            cluster_lat = cluster['latitude'].mean()
            cluster_lon = cluster['longitude'].mean()
            f.write(f"cluster center: ({cluster_lat:.6f}, {cluster_lon:.6f})\n")
            f.write(f"location: {cluster.iloc[0]['streetAddress']}, {cluster.iloc[0]['city']}, {cluster.iloc[0]['state']} {cluster.iloc[0]['zipcode']}, {cluster.iloc[0]['country']}\n\n")
            
            # sort by updatedat to show most recent first
            for _, company in cluster.sort_values('updatedAt', ascending=False).iterrows():
                f.write(f"company: {company['centralized_company_name']}\n")
                f.write(f"  domain: {company.get('companyDomain', 'N/A')}\n")
                f.write(f"  coordinates: ({company['latitude']:.6f}, {company['longitude']:.6f})\n")
                f.write(f"  address: {company['streetAddress']}\n")
                f.write(f"  last updated: {company.get('updatedAt', 'N/A')}\n")
                f.write(f"  id: {company['centralizedCompanyId']}\n")
                f.write("-" * 30 + "\n")
    
    print("done — results saved to csv and analysis file")
    return canonical_companies

if __name__ == "__main__":
    main() 