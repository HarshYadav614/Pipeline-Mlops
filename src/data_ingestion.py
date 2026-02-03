import pandas as pd
import os
import numpy as np
import logging
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn import metrics
import yaml

# 1. LOGGING CONFIGURATION
log_dir = 'logs'
os.makedirs(log_dir, exist_ok=True)

logger = logging.getLogger('car_price_pipeline')
logger.setLevel('DEBUG')

# Handlers
console_handler = logging.StreamHandler()
file_handler = logging.FileHandler(os.path.join(log_dir, 'car_pipeline.log'))

# Formatter
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

def load_params(params_path: str) -> dict:
    """Load parameters from a YAML file."""
    try:
        with open(params_path, 'r') as file:
            params = yaml.safe_load(file)
        logger.debug('Parameters retrieved from %s', params_path)
        return params
    except FileNotFoundError:
        logger.error('File not found: %s', params_path)
        raise
    except yaml.YAMLError as e:
        logger.error('YAML error: %s', e)
        raise
    except Exception as e:
        logger.error('Unexpected error: %s', e)
        raise

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

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """Perform label encoding on categorical features."""
    try:
        df_encoded = df.copy()
        le = LabelEncoder()
        categorical_cols = ['Car_Name', 'Fuel_Type', 'Seller_Type', 'Transmission']
        
        for col in categorical_cols:
            df_encoded[col] = le.fit_transform(df_encoded[col])
            
        logger.debug('Data preprocessing (Label Encoding) completed.')
        return df_encoded
    except Exception as e:
        logger.error('Error during preprocessing: %s', e)
        raise

def save_data(train_data: pd.DataFrame, test_data: pd.DataFrame, data_path: str) -> None:
    """Save the split datasets into a 'raw' directory."""
    try:
        raw_data_path = os.path.join(data_path, 'raw')
        os.makedirs(raw_data_path, exist_ok=True)
        
        train_data.to_csv(os.path.join(raw_data_path, "train.csv"), index=False)
        test_data.to_csv(os.path.join(raw_data_path, "test.csv"), index=False)
        
        logger.debug('Train and test data saved to %s', raw_data_path)
    except Exception as e:
        logger.error('Error saving data: %s', e)
        raise

def train_and_evaluate(X_train, X_test, y_train, y_test):
    """Train multiple models and return their R2 scores."""
    try:
        models = {
            'LR': LinearRegression(),
            'Lasso': Lasso(),
            'Ridge': Ridge(),
            'ElasticNet': ElasticNet(),
            'DT': DecisionTreeRegressor(),
            'SVR': SVR(),
            'KNN': KNeighborsRegressor(),
            'RF': RandomForestRegressor(),
            'XGB': XGBRegressor()
        }
        
        scores = {}
        for name, model in models.items():
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            scores[name] = metrics.r2_score(y_test, y_pred)
            logger.debug(f"Model {name} trained. Score: {scores[name]:.4f}")
            
        return scores
    except Exception as e:
        logger.error('Error during model training: %s', e)
        raise

# 3. MAIN PIPELINE EXECUTION

def main():
    try:
        # Configuration - Replace with your GitHub Raw URL
        data_url = 'https://raw.githubusercontent.com/HarshYadav614/Datasets/refs/heads/main/car%20data.csv' 
        target_column = 'Selling_Price'
        
        # Step 1: Ingestion
        df = load_data(data_url)
        
        # Step 2: Preprocessing
        final_df = preprocess_data(df)
        
        # Step 3: Splitting
        params = load_params(params_path='params.yaml')
        test_size = params['data_ingestion']['test_size']
        train_df, test_df = train_test_split(df, test_size=test_size, random_state=42)
        
        # Step 4: Save Raw Splits (Your requested format)
        save_data(train_df, test_df, data_path='./data')
        
        # Step 5: Feature Scaling for Training
        X_train = train_df.drop(columns=[target_column])
        y_train = train_df[target_column]
        X_test = test_df.drop(columns=[target_column])
        y_test = test_df[target_column]
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Step 6: Evaluation
        performance = train_and_evaluate(X_train_scaled, X_test_scaled, y_train, y_test)
        
        # Display Final Summary
        summary_df = pd.DataFrame({
            'Model': list(performance.keys()),
            'R2_Score': list(performance.values())
        }).sort_values(by='R2_Score', ascending=False)
        
        print("\n--- Final Model Performance ---")
        print(summary_df)
        logger.info("Pipeline completed successfully.")

    except Exception as e:
        logger.error('Failed to complete the pipeline: %s', e)
        print(f"Error: {e}")

if __name__ == '__main__':
    main()