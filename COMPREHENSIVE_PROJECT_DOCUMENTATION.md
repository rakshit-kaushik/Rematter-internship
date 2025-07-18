# Company Deduplication System - Comprehensive Documentation

## Project Overview

This project implements a sophisticated company deduplication system that identifies and groups similar companies in a database using multiple similarity algorithms and ensemble methods. The system has evolved from individual method implementations to a modular, shared-utilities approach, culminating in a location-first ensemble clustering system with interactive approval capabilities.

**Last Updated:** July 3, 2025  
**Data Snapshot:** July 2, 2025

---

## Evolution of the System

### Phase 1: Individual Method Implementations (OLD)
Initially, each deduplication method was implemented as a separate, standalone script with significant code duplication:

- **`code-fuzzy_jaccard_no_domain.py`** (280 lines)
- **`code-metaphone.py`** (361 lines)  
- **`code-jaro_only.py`** (285 lines)
- **`code-location-2.py`** (323 lines)

**Problems with this approach:**
- ~800 lines of duplicate code across files
- Inconsistent database connections and output formats
- Difficult to maintain and extend
- No standardized analysis reporting

### Phase 2: Refactored Shared Utilities System (NEW)
Recognizing the code duplication problem, we refactored the system to use a shared utilities module:

#### **`shared_deduplication_utils.py`** (269 lines)
Contains all common functionality:
- Database connection and data loading
- Company name cleaning and keyword extraction
- Blocking key generation for efficiency
- Similarity graph building and clustering
- Canonical company generation
- Result saving and analysis report generation
- Complete pipeline orchestration

#### **Refactored Method Files** (Only Unique Logic)
- **`code-fuzzy_jaccard_refactored.py`** (73 lines) - Only fuzzy jaccard similarity logic
- **`code-metaphone_refactored.py`** (113 lines) - Only metaphone similarity logic
- **`code-jaro_refactored.py`** (135 lines) - Only jaro-winkler similarity logic
- **`code-location-2_refactored.py`** (262 lines) - Only location-based clustering logic

**Benefits of refactoring:**
- Eliminated ~400 lines of duplicate code
- Consistent behavior across all methods
- Easy to add new similarity methods
- Standardized output format and analysis

### Phase 3: Ensemble Clustering with Interactive Approval (FINAL)
The final approach combines multiple methods using ensemble clustering with location-first strategy and manual approval for low-confidence clusters.

---

## File Organization

### Current Working Directory
```
rematter-internship-2025/
├── shared_deduplication_utils.py          # Shared utilities (NEW)
├── code-fuzzy_jaccard_refactored.py       # Fuzzy jaccard method (NEW)
├── code-metaphone_refactored.py           # Metaphone method (NEW)
├── code-jaro_refactored.py                # Jaro-winkler method (NEW)
├── code-location-2_refactored.py          # Location method (NEW)
├── code-ensemble_loc.py                   # Final ensemble approach (FINAL)
├── code-ensemble-combine.py               # Alternative ensemble method
├── requirements.txt                       # Python dependencies
└── extras/                                # Additional utilities
    ├── view_clusters_by_confidence.py     # Analysis tool
    └── test_scripts/                      # Testing utilities
```

### Output Organization

#### **NEW Outputs** (Current System - July 2025)
```
individual_output/new/
├── csv_files/
│   ├── canonical/                         # Canonical company records
│   │   ├── fuzzy_jaccard_canonical.csv
│   │   ├── metaphone_canonical.csv
│   │   ├── jaro_jw_0_85_canonical.csv
│   │   ├── jaro_jw_0_87_canonical.csv
│   │   ├── jaro_jw_0_89_canonical.csv
│   │   ├── jaro_jw_0_91_canonical.csv
│   │   ├── jaro_jw_0_93_canonical.csv
│   │   └── location_canonical.csv
│   └── clustered/                         # All companies with cluster assignments
│       ├── fuzzy_jaccard_clustered.csv
│       ├── metaphone_clustered.csv
│       ├── jaro_jw_0_85_clustered.csv
│       ├── jaro_jw_0_87_clustered.csv
│       ├── jaro_jw_0_89_clustered.csv
│       ├── jaro_jw_0_91_clustered.csv
│       ├── jaro_jw_0_93_clustered.csv
│       └── location_clustered.csv
└── analysis_files/                        # Detailed analysis reports
    ├── fuzzy_jaccard_analysis.txt
    ├── metaphone_analysis.txt
    ├── jaro_threshold_comparison_summary.txt
    ├── jaro_jw_0_85_analysis.txt
    ├── jaro_jw_0_87_analysis.txt
    ├── jaro_jw_0_89_analysis.txt
    ├── jaro_jw_0_91_analysis.txt
    ├── jaro_jw_0_93_analysis.txt
    └── location_analysis.txt
```

#### **OLD Outputs** (Previous System - Deprecated)
```
individual_output/old/
├── csv_files/
│   ├── canonical/                         # Old canonical files
│   │   ├── canonical_companies_ensemble_combined.csv
│   │   ├── canonical_companies_ensemble_location.csv
│   │   ├── canonical_companies_fuzzy_jaccard_no_domain.csv
│   │   ├── canonical_companies_location.csv
│   │   └── canonical_companies_metaphone.csv
│   └── clustered/                         # Old clustered files
│       ├── clustered_companies_ensemble_combined.csv
│       ├── clustered_companies_fuzzy_jaccard_no_domain.csv
│       ├── clustered_companies_location.csv
│       └── clustered_companies_metaphone.csv
└── analysis_files/                        # Old analysis files
    ├── cluster_analysis_ensemble_combined.txt
    ├── cluster_analysis_fuzzy_jaccard_no_domain.txt
    ├── cluster_analysis_location.txt
    └── cluster_analysis_metaphone.txt
```

---

## How the Shared Deduplication System Works

### Core Architecture

The refactored system uses a shared utilities class that provides common functionality while individual method files contain only their unique similarity logic.

#### **SharedDeduplicationUtils Class**
```python
class SharedDeduplicationUtils:
    def clean_company_name(self, name)           # Normalize company names
    def get_company_keywords(self, name)         # Extract key words
    def get_blocking_key(self, name)             # Create blocking keys
    def build_similarity_graph(self, df, similarity_function, threshold)
    def assign_clusters(self, df, G)             # Assign cluster IDs
    def generate_canonical_companies(self, df)   # Create canonical records
    def create_database_connection(self)         # Database connection
    def load_companies_from_database(self, engine) # Load data
    def save_results(self, companies_df, canonical_companies, output_prefix)
    def generate_analysis_report(self, companies_df, output_file, method_name, threshold)
    def run_deduplication_pipeline(self, similarity_function, threshold, method_name, output_prefix)
```

#### **Method-Specific Files**
Each method file only contains:
1. **Similarity calculation functions** (unique to each method)
2. **Similarity criteria function** (how to determine if companies match)
3. **Main execution function** that uses shared utilities

### Example: Fuzzy Jaccard Method
```python
# code-fuzzy_jaccard_refactored.py
from shared_deduplication_utils import SharedDeduplicationUtils

def calculate_fuzzy_jaccard_similarity(name1, name2):
    # Unique fuzzy jaccard logic
    pass

def are_companies_similar(row1, row2, threshold):
    # Unique similarity criteria
    pass

def main():
    utils = SharedDeduplicationUtils()
    results = utils.run_deduplication_pipeline(
        similarity_function=are_companies_similar,
        threshold=0.85,
        method_name="fuzzy_jaccard",
        output_prefix="fuzzy_jaccard"
    )
```

---

## The Final Approach: Ensemble Location-First Clustering

### **`code-ensemble_loc.py`** - The Final System

This is the most sophisticated approach that combines location-based clustering with ensemble name similarity methods and interactive approval for quality control.

### How It Works

#### **Step 1: Location-First Clustering**
```python
def create_location_clusters(location_df, decimal_places=4):
    # Groups companies by rounded coordinates (4 decimal places ≈ 11 meters)
    # Creates geographic blocks for efficient processing
```

#### **Step 2: Ensemble Name Similarity Within Locations**
For each location cluster, applies three name similarity methods:
1. **Fuzzy Jaccard** (weight: 0.85) - Jaccard similarity + token ratios
2. **Jaro-Winkler** (weight: 0.9) - String similarity algorithm
3. **Metaphone** (weight: 0.9) - Phonetic similarity

#### **Step 3: Confidence Calculation**
```python
def calculate_name_ensemble_confidence(pair_methods, total_methods):
    # Calculates confidence based on:
    # - Number of methods that agree
    # - Weighted scores for each method
    # - Agreement ratio
```

#### **Step 4: Interactive Approval System**
For clusters with confidence < 0.75, the system prompts for manual approval:

```
================================================================================
CLUSTER 123 - Confidence: 0.720
================================================================================
Top Company: Example Company Inc
Cluster Size: 3
Confidence Level: MEDIUM
Methods Agreeing: jaro_winkler, metaphone

Companies in this cluster:
--------------------------------------------------
 1. Example Company Inc (ID: comp_001)
     Domain: example.com
 2. Example Company (ID: comp_002)
     Domain: example.org
 3. ExampleCompany (ID: comp_003)

Approve this cluster? (y/n/s to skip all remaining): 
```

**User Options:**
- `y` or `yes`: Approve the cluster
- `n` or `no`: Reject the cluster (adds to manual review list)
- `s` or `skip`: Skip all remaining clusters below 0.75 confidence

#### **Step 5: Output Generation**
The system generates three main output files:

1. **`cleaned_cluster_list_ensemble_location.csv`** - Main output
2. **`canonical_companies_ensemble_location.csv`** - Detailed canonical information
3. **`cluster_analysis_ensemble_location.txt`** - Comprehensive analysis

---

## Output File Structure

### **Main Output: `cleaned_cluster_list_ensemble_location.csv`**
```csv
cluster_id,top_company_name,top_company_id,cluster_size,confidence_score,confidence_level,location_lat,location_lon,company_ids
LOC_0_SINGLE,Arringtong George,c8aae890-1138-11f0-8e97-598de960eaf5,1,1.0,HIGH,-1.3055,36.8358,['c8aae890-1138-11f0-8e97-598de960eaf5']
LOC_27_NAME_0,Amc Metals Recyclers Pty Ltd,a8643412-f616-497c-a0d6-0bf83a4576ce,2,0.953,HIGH,-20.7752,116.8692,"['595904af-10a2-40b5-a061-7be13c69e503', 'a8643412-f616-497c-a0d6-0bf83a4576ce']"
```

**Columns:**
- `cluster_id`: Unique identifier (LOC_X_SINGLE for single companies, LOC_X_NAME_Y for clusters)
- `top_company_name`: Representative company name
- `top_company_id`: Representative company ID
- `cluster_size`: Number of companies in cluster
- `confidence_score`: Average confidence (0.0-1.0)
- `confidence_level`: HIGH/MEDIUM/LOW classification
- `location_lat/lon`: Geographic coordinates
- `company_ids`: List of all company IDs in cluster

### **Detailed Output: `canonical_companies_ensemble_location.csv`**
Contains comprehensive information including:
- All basic cluster information
- Company domains and timezones
- Detailed confidence metrics
- All company names and IDs
- Creation and update timestamps

### **Analysis Output: `cluster_analysis_ensemble_location.txt`**
Provides detailed statistics and cluster information:
- Overall statistics and confidence distribution
- Multi-company clusters by confidence level
- Detailed information for each cluster
- Geographic and temporal analysis

---

## Manual Approval and Rejected Clusters

### How Rejected Clusters Are Handled

When a user rejects a cluster during the interactive approval process:

1. **Cluster is marked as REJECTED** in the confidence statistics
2. **Cluster is added to rejected_clusters list** for later processing
3. **Cluster is excluded** from the main output files
4. **User can continue** with the next cluster

### Rejected Clusters File Structure
The system maintains a list of rejected clusters that can be used for:
- Manual review and correction
- Training data for machine learning models
- Quality analysis and threshold tuning

### Future ML Training Pipeline

The rejected clusters provide the foundation for training a machine learning model to automate manual annotations:

#### **Phase 1: Manual Annotation Collection**
1. **Review rejected clusters** manually
2. **Annotate correct groupings** based on domain knowledge
3. **Create training dataset** with features and labels

#### **Training Features**
- Company name similarity scores (all methods)
- Geographic distance
- Domain similarity
- Timezone matching
- Industry classification
- Company size indicators

#### **Training Labels**
- Binary: Should companies be grouped (1) or not (0)
- Multi-class: Confidence level (HIGH/MEDIUM/LOW)

#### **Model Training**
```python
# Future implementation
def train_approval_model(rejected_clusters, manual_annotations):
    # Extract features from rejected clusters
    # Train classification model
    # Validate on held-out data
    # Deploy for automatic approval
```

#### **Automated Approval Integration**
```python
# Future enhancement
def auto_approve_clusters(clusters, trained_model):
    # Use trained model to predict approval
    # Apply confidence thresholds
    # Generate automatic approvals/rejections
```

---

## Running the System

### Individual Methods (NEW)
```bash
# Run individual similarity methods
python code-fuzzy_jaccard_refactored.py
python code-metaphone_refactored.py
python code-jaro_refactored.py
python code-location-2_refactored.py
```

### Ensemble Location-First (FINAL)
```bash
# Run the final ensemble approach
python code-ensemble_loc.py
```

### Analysis Tools
```bash
# View clusters by confidence level
python extras/view_clusters_by_confidence.py summary
python extras/view_clusters_by_confidence.py high 2 5
```

---

## Performance and Results

### Current System Performance (July 2025)
- **Total companies processed:** ~200,000
- **Location clusters:** ~150,000
- **Multi-company clusters:** ~10,000
- **Single companies:** ~140,000
- **Average confidence:** 0.756 (75.6%)

### Confidence Distribution
- **HIGH confidence:** ~46.8% of clusters
- **MEDIUM confidence:** ~53.2% of clusters
- **LOW confidence:** ~0% of clusters
- **Rejected clusters:** Varies based on manual review

### Geographic Coverage
- **Global coverage** with coordinates in 4 decimal precision
- **Location-based blocking** for efficient processing
- **Haversine distance** calculations for accuracy

---

## Future Enhancements

### 1. Machine Learning Integration
- Train models on manual annotations
- Automate approval decisions
- Improve confidence scoring

### 2. Advanced Similarity Methods
- Industry-specific similarity algorithms
- Domain name similarity
- Logo/image similarity (if available)

### 3. Real-time Processing
- Incremental clustering for new companies
- Streaming data processing
- Real-time confidence updates

### 4. Quality Metrics
- Precision/recall analysis
- Cross-validation with known duplicates
- A/B testing of different approaches

---

## Conclusion

This company deduplication system represents a comprehensive approach to identifying and grouping similar companies using multiple similarity algorithms, ensemble methods, and interactive quality control. The evolution from individual method implementations to a shared utilities system demonstrates the importance of code organization and maintainability in data science projects.

The final ensemble location-first approach provides the best balance of accuracy, efficiency, and quality control, while the interactive approval system ensures high-quality results for downstream applications.

**Key Takeaways:**
1. **Modular design** enables easy maintenance and extension
2. **Ensemble methods** improve accuracy over single algorithms
3. **Interactive approval** ensures quality control
4. **Geographic clustering** provides efficient processing
5. **Comprehensive documentation** enables future development

The system is ready for production use and provides a solid foundation for future enhancements including machine learning integration and automated approval systems. 