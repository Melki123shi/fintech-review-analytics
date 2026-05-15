# Data Collection and Preprocessing Notebooks

## Overview

This directory contains Jupyter notebooks for collecting and preprocessing app reviews from the Google Play Store. The primary workflow focuses on the Commercial Bank of Ethiopia (CBE) mobile banking application.

---

## Notebooks

A comprehensive notebook that handles data collection, cleaning, and preprocessing for CBE app reviews.

**Key Sections:**
- Data collection from Google Play Store
- Data exploration and inspection
- Data preprocessing and cleaning
- Data validation and quality assurance
- Final dataset preparation

---

## Scraping Methodology

### Data Source
- **Platform**: Google Play Store
- **Application**: Commercial Bank of Ethiopia Mobile Banking (`com.combanketh.mobilebanking`)
- **Tool**: `google-play-scraper` Python library

### Collection Parameters
| Parameter | Value | Purpose |
|-----------|-------|---------|
| **Review Count** | 700 | Maximum number of reviews to collect per request |
| **Sort Order** | NEWEST | Retrieves most recent reviews first |
| **Language** | English | Filters reviews to English language only |
| **Country** | Ethiopia (et) | Localizes app metadata and region-specific content |
| **Rating Filter** | None | Collects reviews across all star ratings (1-5) |

### Review Data Collected

Each review includes the following fields:

| Field | Source | Description |
|-------|--------|-------------|
| `review_id` | reviewId | Unique identifier for each review |
| `review` | content | Full text content of the user's review |
| `rating` | score | Star rating (1-5) given by the user |
| `date` | at | Timestamp of when the review was posted |
| `bank` | N/A | Bank identifier (hardcoded as "CBE Bank") |
| `source` | N/A | Data source identifier (hardcoded as "Google Play") |

---

## Date Range

The date range of collected reviews varies based on available Google Play Store data at the time of collection. The preprocessing notebook automatically:

1. **Normalizes** all dates to ISO 8601 format (YYYY-MM-DD)
2. **Reports** the minimum and maximum dates in the final dataset
3. **Validates** date consistency across all records

The actual date range is displayed in the preprocessing report generated at the end of `cbe_preprocessing.ipynb`.

---

## Preprocessing & Data Cleaning

### Processing Steps (in order)

1. **Remove Duplicates**
   - Duplicates identified by `review_id`
   - Ensures each review is included only once

2. **Handle Missing Data**
   - Critical columns: `review` and `rating`
   - Rows with missing values in these columns are removed
   - Maintains data integrity

3. **Normalize Dates**
   - Converts all date formats to YYYY-MM-DD
   - Handles timestamp conversion
   - Ensures consistency across the dataset

4. **Validate Ratings**
   - Verifies all ratings are integers between 1-5
   - Removes reviews with invalid ratings
   - Ensures data quality

5. **Clean Review Text**
   - Converts text to lowercase
   - Removes URLs and web references
   - Removes extra whitespace
   - Removes emojis and non-ASCII characters
   - Retains alphanumeric text and basic punctuation

6. **Final Dataset Preparation**
   - Selects only essential columns: `review`, `rating`, `date`, `bank`, `source`
   - Sorts by date (newest first)
   - Resets index for clean output

### Data Quality Metrics

The preprocessing report provides:
- Raw reviews collected
- Final reviews after cleaning
- Total reviews removed
- Data retention rate (percentage)
- Data quality assessment (EXCELLENT/GOOD/NEEDS ATTENTION)
- Date range of the dataset
- Rating distribution breakdown

---

## Known Limitations

### API & Collection Limitations

1. **Review Count Cap**
   - Google Play Store API limits requests to 700 reviews per call
   - Actual collected count may be lower if fewer reviews are available
   - Use continuation tokens for pagination if needed

2. **Language Filtering**
   - Collection limited to English-language reviews only
   - Non-English reviews are excluded from the dataset
   - May miss relevant feedback in other languages

3. **Geographic Filtering**
   - Data limited to reviews from Ethiopia (et) region
   - May not capture international user feedback
   - Region-specific biases may exist in review patterns

4. **Sort Order**
   - Default sort is NEWEST (most recent first)
   - Affects which reviews are captured when hitting the 700 review limit
   - Older reviews may be underrepresented

### Data Quality Limitations

1. **Text Cleaning Trade-offs**
   - Emojis and non-ASCII characters are removed
   - May lose sentiment indicators (emoji reactions)
   - Special characters that carry meaning may be stripped

2. **Missing Metadata**
   - App version at time of review not consistently captured
   - Reviewer profile information not collected
   - Review helpfulness scores not included

3. **Duplicates**
   - Some duplicate reviews may still exist after deduplication
   - Duplicates identified solely by `review_id`
   - Content-based duplicates are not detected

4. **Invalid Data**
   - Invalid ratings (outside 1-5 range) are removed
   - May indicate API inconsistencies or data corruption
   - Affected reviews are permanently excluded

### Time-Based Limitations

1. **Snapshot Data**
   - Each collection represents a point-in-time snapshot
   - Historical review changes and edits are not captured
   - Review deletion from Google Play is not tracked

2. **Rate Limiting**
   - Google Play Scraper may face rate limits
   - Large-scale recurring collections may fail
   - Unexpected delays can occur

---

## Output

### Final Dataset Location
```
data/processed/cbe_reviews_cleaned.csv
```

### Dataset Schema
```
review         : string (cleaned review text)
rating         : integer (1-5 star rating)
date           : string (YYYY-MM-DD format)
bank           : string ("CBE Bank")
source         : string ("Google Play")
```

---

## Running the Notebook

### Prerequisites
- Python 3.7+
- Required packages (see `../requirements.txt`):
  - pandas
  - numpy
  - google-play-scraper

### Execution Steps

1. Navigate to the notebooks directory:
   ```bash
   cd notebooks/
   ```

2. Start Jupyter:
   ```bash
   jupyter notebook cbe_preprocessing.ipynb
   ```

3. Run all cells in order (Cell → Run All)

4. Review the preprocessing report output

5. Check the final dataset in `data/processed/cbe_reviews_cleaned.csv`

---

## Notes & Recommendations

- **Reproducibility**: Data collection from Google Play Store may yield different results over time as new reviews are added and removed
- **Scaling**: For analyzing multiple apps, consider creating separate notebooks for each app to maintain organization
- **Updates**: Periodically re-run the notebook to capture new reviews and maintain dataset freshness
- **Error Handling**: Monitor console output for warnings about invalid data or API issues

---
