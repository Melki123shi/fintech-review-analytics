import warnings

import pandas as pd
import logging
from pathlib import Path

# Suppress warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_cleaned_data(data_dir='../data/processed'):
    """
    Load cleaned CSV files for all available banks.
    
    Parameters:
    -----------
    data_dir : str
        Directory containing cleaned CSV files
        
    Returns:
    --------
    dict : Dictionary with bank names as keys and DataFrames as values
    list : List of available banks
    """
    data_path = Path(data_dir)
    banks_data = {}
    available_banks = []
  
    
    # Map common file patterns to bank names
    bank_patterns = {
        'boa': 'BOA Bank',
        'cbe': 'CBE Bank',
        'dashen': 'Dashen Bank'
    }
    
    for pattern, bank_name in bank_patterns.items():
        csv_file = data_path / f'{pattern}_reviews_cleaned.csv'
        if csv_file.exists():
            try:
                df = pd.read_csv(csv_file)
                banks_data[bank_name] = df.reset_index(drop=True).rename_axis('review_id').reset_index()
                available_banks.append(bank_name)
                logger.info(f"✓ Loaded {bank_name}: {len(df)} reviews")
            except Exception as e:
                logger.error(f"✗ Failed to load {csv_file}: {e}")
        else:
            logger.warning(f"! File not found: {csv_file}")
    
    if not banks_data:
        logger.error("No cleaned data files found!")
        return {}, []
    
    return banks_data, available_banks
