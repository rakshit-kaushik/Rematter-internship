# Company Deduplication System

A comprehensive ensemble-based company deduplication system that uses location-first clustering combined with multiple name similarity methods.

## Overview

This system implements two approaches to company deduplication, both using location-first clustering combined with ensemble name similarity methods:

### System 1: `company_deduplication_system.py` (Advanced)
- **Location Clustering**: Uses Haversine distance with 10-meter threshold for precise geographic grouping
- **Ensemble Methods**: Combines Fuzzy Jaccard, Jaro-Winkler, and Metaphone similarity
- **Confidence Calculation**: Sophisticated weighted average combining ensemble agreement and similarity scores
- **Interactive Approval**: Manual review for low-confidence clusters
- **Comprehensive Analysis**: Detailed reporting and ML training data generation

### System 2: `code-ensemble_loc.py` (Fast & Efficient) - **I recommend this fs**
- **Location Clustering**: Uses coordinate rounding (4 decimal places) for faster processing
- **Ensemble Methods**: Same three similarity methods as System 1
- **Confidence Calculation**: Identical to System 1 for consistency
- **Interactive Approval**: Manual review for low-confidence clusters
- **Speed Advantage**: Significantly faster due to simpler location clustering

## **Recommended Results**

**Use the results from `code-ensemble_loc.py`** located at:
```
individual_output/new/ensemble_location_canonical.csv
```

**Why choose `code-ensemble_loc.py`?**
- **Speed**: ~3x faster than the advanced system
- **Similar Accuracy**: Produces nearly identical results (47 vs 45 low-confidence clusters)
- **Same Confidence Logic**: Uses identical ensemble confidence calculation
- **Efficient Processing**: Coordinate rounding vs. pairwise distance calculations

## Features

- **Geographic Clustering**: Groups companies by location proximity
- **Ensemble Methods**: Combines multiple similarity algorithms for robust matching
- **Interactive Review**: Manual approval system for uncertain matches
- **Comprehensive Output**: CSV files and detailed analysis reports
- **Configurable Thresholds**: Adjustable similarity thresholds for each method
- **Database Integration**: Direct connection to MySQL database
- **Training Data Generation**: Collects manual decisions for ML model development

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd company-deduplication-system
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Ensure you have access to the MySQL database with the required credentials.

## Usage

### Recommended Usage (Fast System)

Run the efficient deduplication pipeline:

```bash
python code-ensemble_loc.py
```

### Alternative Usage (Advanced System)

Run the comprehensive deduplication pipeline:

```bash
python company_deduplication_system.py
```

Both systems will:
1. Prompt for database password
2. Load company data with location information
3. Perform location-based clustering
4. Apply ensemble name similarity within location clusters
5. Present low-confidence clusters for manual review
6. Generate canonical companies and save results
7. Create detailed analysis reports

### Configuration

You can modify the following parameters in the code:

- **Distance Threshold**: 
  - System 1: 0.01 km (10 meters) for location clustering
  - System 2: ~11 meters (coordinate rounding to 4 decimal places)
- **Similarity Thresholds**:
  - Fuzzy Jaccard: 0.85
  - Jaro-Winkler: 0.89
  - Metaphone: 0.91
- **Interactive Approval Threshold**: 0.60 (clusters below this confidence need manual review)

### Output Files

The system generates several output files:

1. **Canonical Companies**: `individual_output/new/ensemble_location_canonical.csv` - **MAIN OUTPUT**
   - Contains one representative company per cluster with all company IDs in the cluster
   - Includes `original_id` (canonical company ID) and `company_ids` (array of all company IDs)

2. **Training Data**: `individual_output/new/training_data/ensemble_location_training_data.csv`
   - Contains manual decisions and auto-approved clusters for ML model development

3. **Analysis Reports**: Various analysis files in `individual_output/new/analysis_files/`

## Algorithm Details

### Location Clustering

#### System 1 (Advanced)
1. **Geographic Blocking**: Companies are grouped by rounded coordinates (4 decimal places)
2. **Distance Calculation**: Haversine formula calculates great circle distances
3. **Threshold Filtering**: Companies within 10 meters are considered co-located

#### System 2 (Fast)
1. **Coordinate Rounding**: Companies are grouped by coordinates rounded to 4 decimal places
2. **Approximate Distance**: ~11 meters at the equator
3. **Faster Processing**: No pairwise distance calculations needed

### Name Similarity Methods

Both systems use identical similarity methods:

#### 1. Fuzzy Jaccard Similarity
- Extracts keywords from company names (removes stop words)
- Calculates Jaccard similarity on keyword sets
- Combines with fuzzy string matching (token sort, token set, partial ratios)
- Weighted combination: Jaccard (30%) + Token Sort (25%) + Token Set (25%) + Partial (20%)

#### 2. Jaro-Winkler Similarity
- Preprocesses names by extracting keywords
- Applies Jaro-Winkler algorithm for character-level similarity
- Accounts for transpositions and common prefixes

#### 3. Metaphone Similarity
- Generates phonetic codes using Double Metaphone
- Compares primary and secondary metaphone codes
- Uses Jaro-Winkler similarity on phonetic representations
- Weighted by code importance: Primary-Primary (100%), Primary-Secondary (80%), etc.

### Ensemble Logic

Both systems use identical ensemble logic:
- **Individual Thresholds**: Each method has its own threshold
- **Combination Rule**: Companies are clustered if ANY method exceeds its threshold
- **Confidence Calculation**: `(ensemble_ratio + weighted_score) / 2`
- **Interactive Review**: Clusters below 0.60 confidence threshold require manual approval

## Performance Comparison

| Metric | System 1 (Advanced) | System 2 (Fast) |
|--------|-------------------|-----------------|
| **Speed** | Baseline | ~3x faster |
| **Location Precision** | 10m Haversine | ~11m rounding |
| **Low-Confidence Clusters** | 45 | 47 |
| **Processing Time** | ~3-4 minutes | ~1-2 minutes |
| **Memory Usage** | Higher | Lower |

## Database Schema

The system expects the following database structure:

```sql
-- Centralized company information
centralized_company (
    centralizedCompanyId VARCHAR,
    companyName VARCHAR,
    createdAt TIMESTAMP,
    updatedAt TIMESTAMP
)

-- Company instances
company (
    companyId VARCHAR,
    credentialledCentralizedCompanyId VARCHAR
)

-- Location information
location (
    locatableId VARCHAR,
    streetAddress VARCHAR,
    city VARCHAR,
    state VARCHAR,
    zipcode VARCHAR,
    country VARCHAR,
    coordinate POINT
)
```

## Output Format

The main output CSV (`ensemble_location_canonical.csv`) contains:

- **`canonical_id`**: Generated cluster identifier
- **`original_id`**: Individual `centralizedCompanyId` of the canonical company
- **`company_ids`**: Array of all `centralizedCompanyId`s in the cluster (including canonical)
- **`cluster_size`**: Number of companies in the cluster
- **`confidence_level`**: HIGH, MEDIUM, LOW, or REJECTED
- **`location_lat`**, **`location_lon`**: Geographic coordinates
- **`company_name`**: Name of the canonical company
- **`company_domain`**, **`timezone`**: Additional company details

## Performance Considerations

- **Blocking**: Geographic blocking reduces comparison complexity
- **Efficient Algorithms**: Uses optimized string similarity libraries
- **Progress Tracking**: Progress bars show processing status
- **Memory Management**: Processes data in chunks where possible
- **Coordinate Rounding**: Faster than pairwise distance calculations

## Troubleshooting

### Common Issues

1. **Database Connection Error**:
   - Verify database credentials
   - Ensure network access to database server
   - Check MySQL user permissions

2. **Missing Dependencies**:
   - Run `pip install -r requirements.txt`
   - Install system-level dependencies if needed (e.g., `python-Levenshtein`)

3. **Memory Issues**:
   - Use System 2 (code-ensemble_loc.py) for better memory efficiency
   - Reduce geographic precision in blocking
   - Process smaller datasets

### Debugging

- Enable verbose logging by modifying print statements
- Check intermediate CSV files for data quality
- Review analysis reports for clustering patterns

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request


## Contact

me - rakshit@rematter.com
