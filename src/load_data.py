import pandas as pd

def load_data(file_path):
    """
    Load the cleaned review data from a CSV file.

    Parameters:
    file_path (str): The path to the cleaned CSV file.

    Returns:
    pd.DataFrame: A DataFrame containing the cleaned review data.
    """
    try:
        data = pd.read_csv(file_path)
        df = pd.DataFrame(data)
        print(f"Data loaded successfully from {file_path}")
        return df
    except Exception as e:
        print(f"Error loading data: {e}")
        return None