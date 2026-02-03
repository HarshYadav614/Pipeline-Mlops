import pandas as pd
import os
import logging
import joblib
from sklearn.preprocessing import StandardScaler

# 1. LOGGING CONFIGURATION

log_dir = 'logs'
os.makedirs(log_dir, exist_ok=True)

logger = logging.getLogger('feature_engineering')
logger.setLevel('DEBUG')

# Console and File Handlers
console_handler = logging.StreamHandler()
file_handler = logging.FileHandler(os.path.join(log_dir, 'feature_engineering.log'))

# Formatter
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

# 2. FUNCTION DEFINITIONS

def load_data(file_path: str) -> pd.DataFrame:
    """Load data from the interim folder."""
    try:
        df = pd.read_csv(file_path)
        logger.debug('Data loaded for feature engineering from %s', file_path)
        return df
    except Exception as e:
        logger.error('Failed to load interim data: %s', e)
        raise

def scale_features(train_df: pd.DataFrame, test_df: pd.DataFrame, target_column: str):
    """Apply Standard Scaling to the features."""
    try:
        # Separate Features (X) and Target (y)
        X_train = train_df.drop(columns=[target_column])
        y_train = train_df[target_column]
        
        X_test = test_df.drop(columns=[target_column])
        y_test = test_df[target_column]
        
        # Initialize Scaler
        scaler = StandardScaler()
        
        # Fit on train, transform both
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Convert back to DataFrame to keep it clean
        X_train_final = pd.DataFrame(X_train_scaled, columns=X_train.columns)
        X_test_final = pd.DataFrame(X_test_scaled, columns=X_test.columns)
        
        # Add targets back
        X_train_final[target_column] = y_train.values
        X_test_final[target_column] = y_test.values
        
        logger.debug('Standard Scaling completed successfully.')
        return X_train_final, X_test_final, scaler

    except Exception as e:
        logger.error('Error during feature scaling: %s', e)
        raise

def save_processed_data(train_data: pd.DataFrame, test_data: pd.DataFrame, data_path: str) -> None:
    """Save the final scaled data into the 'processed' folder."""
    try:
        processed_path = os.path.join(data_path, 'processed')
        os.makedirs(processed_path, exist_ok=True)
        
        train_data.to_csv(os.path.join(processed_path, "train_processed.csv"), index=False)
        test_data.to_csv(os.path.join(processed_path, "test_processed.csv"), index=False)
        
        logger.debug('Final processed data saved to %s', processed_path)
    except Exception as e:
        logger.error('Unexpected error while saving processed data: %s', e)
        raise

# 3. MAIN FEATURE ENGINEERING PIPELINE

def main():
    try:
        base_data_path = './data'
        target_col = 'Selling_Price'
        
        # Step 1: Load Interim Data (Encoded but not scaled)
        interim_train_path = os.path.join(base_data_path, 'interim', 'train_interim.csv')
        interim_test_path = os.path.join(base_data_path, 'interim', 'test_interim.csv')
        
        train_df = load_data(interim_train_path)
        test_df = load_data(interim_test_path)
        
        # Step 2: Scale Features
        train_processed, test_processed, scaler = scale_features(train_df, test_df, target_col)
        
        # Step 3: Save to Processed folder
        save_processed_data(train_processed, test_processed, data_path=base_data_path)
        
        # Optional: Save the scaler object for later prediction
        models_dir = 'models'
        os.makedirs(models_dir, exist_ok=True)
        joblib.dump(scaler, os.path.join(models_dir, 'scaler.pkl'))
        
        logger.info("Feature Engineering stage completed successfully. Processed files created.")

    except Exception as e:
        logger.error('Feature Engineering process failed: %s', e)

if __name__ == '__main__':
    main()