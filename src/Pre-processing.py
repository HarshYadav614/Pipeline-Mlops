import pandas as pd
import os
import logging
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# 1. LOGGING CONFIGURATION

log_dir = 'logs'
os.makedirs(log_dir, exist_ok=True)

logger = logging.getLogger('car_data_pipeline')
logger.setLevel('DEBUG')

# Handlers (Console and File)
console_handler = logging.StreamHandler()
file_handler = logging.FileHandler(os.path.join(log_dir, 'pipeline.log'))

# Formatter
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

# 2. FUNCTION DEFINITIONS

def load_data(data_url: str) -> pd.DataFrame:
    """Load data from a CSV file or URL."""
    try:
        df = pd.read_csv(data_url)
        logger.debug('Data loaded successfully from %s', data_url)
        return df
    except Exception as e:
        logger.error('Unexpected error occurred while loading the data: %s', e)
        raise

def label_encode_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Apply Label Encoding to categorical features (from your Jupyter Notebook)."""
    try:
        df_encoded = df.copy()
        categorical_cols = ['Car_Name', 'Fuel_Type', 'Seller_Type', 'Transmission']
        
        le = LabelEncoder()
        for col in categorical_cols:
            if col in df_encoded.columns:
                df_encoded[col] = le.fit_transform(df_encoded[col])
                logger.debug('Encoded column: %s', col)
        
        return df_encoded
    except Exception as e:
        logger.error('Error during Label Encoding: %s', e)
        raise

def save_data(train_data: pd.DataFrame, test_data: pd.DataFrame, data_path: str, folder_name: str) -> None:
    """Generic function to save data into specific folders (raw or interim)."""
    try:
        target_path = os.path.join(data_path, folder_name)
        os.makedirs(target_path, exist_ok=True)
        
        train_data.to_csv(os.path.join(target_path, f"train_{folder_name}.csv"), index=False)
        test_data.to_csv(os.path.join(target_path, f"test_{folder_name}.csv"), index=False)
        
        logger.debug(f'Data saved successfully to {target_path}')
    except Exception as e:
        logger.error(f'Error saving data to {folder_name}: {e}')
        raise

# 3. MAIN PIPELINE (INGESTION + INTERIM PREPROCESSING)

def main():
    try:
        # Configuration - (Ensure this is the RAW github link)
        data_url = 'https://raw.githubusercontent.com/HarshYadav614/Datasets/refs/heads/main/car%20data.csv' 
        base_data_path = './data'

        # --- STAGE 1: DATA INGESTION ---
        logger.info("Starting Data Ingestion stage...")
        df = load_data(data_url)
        
        # Initial Split (Raw)
        train_raw, test_raw = train_test_split(df, test_size=0.2, random_state=42)
        
        # Save to 'raw' folder
        save_data(train_raw, test_raw, base_data_path, folder_name='raw')
        
        # --- STAGE 2: PREPROCESSING (INTERIM) ---
        logger.info("Starting Preprocessing (Interim) stage...")
        
        # Apply the Encoding logic from your Jupyter Notebook
        train_interim = label_encode_columns(train_raw)
        test_interim = label_encode_columns(test_raw)
        
        # Save to 'interim' folder (The intermediate step from the video)
        save_data(train_interim, test_interim, base_data_path, folder_name='interim')
        
        logger.info("Pipeline Execution Successful: Raw and Interim data folders are ready.")

    except Exception as e:
        logger.error('Pipeline failed: %s', e)

if __name__ == '__main__':
    main()