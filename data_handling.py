import pandas as pd
import logging

logger = logging.getLogger(__name__)

def extract_data_from_csv(file_path: str) -> pd.DataFrame:
    """
    Extract relevant data from the CSV file.

    Args:
        file_path (str): Path to the CSV file.

    Returns:
        pd.DataFrame: DataFrame containing PDB IDs and kcat_mut values.
    """
    logger.info(f"Extracting data from {file_path}")
    df = pd.read_csv(file_path)
    logger.info(f"Extracted data for {len(df)} structures")
    return df