import pandas as pd
import os
import logging
import joblib
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn import metrics

# 1. LOGGING CONFIGURATION

log_dir = 'logs'
os.makedirs(log_dir, exist_ok=True)

logger = logging.getLogger('model_training')
logger.setLevel('DEBUG')

# Handlers
console_handler = logging.StreamHandler()
file_handler = logging.FileHandler(os.path.join(log_dir, 'model_training.log'))

# Formatter
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

# 2. FUNCTION DEFINITIONS

def load_processed_data(file_path: str) -> pd.DataFrame:
    """Load the final processed data from the processed folder."""
    try:
        df = pd.read_csv(file_path)
        logger.debug('Processed data loaded from %s', file_path)
        return df
    except Exception as e:
        logger.error('Failed to load processed data: %s', e)
        raise

def train_and_evaluate_models(train_df: pd.DataFrame, test_df: pd.DataFrame, target_column: str):
    """Train multiple models and return scores and the best model object."""
    try:
        # Split features and target
        X_train = train_df.drop(columns=[target_column])
        y_train = train_df[target_column]
        X_test = test_df.drop(columns=[target_column])
        y_test = test_df[target_column]

        # Model Dictionary based on your Jupyter Notebook
        models = {
            'LinearRegression': LinearRegression(),
            'Lasso': Lasso(),
            'Ridge': Ridge(),
            'ElasticNet': ElasticNet(),
            'DecisionTree': DecisionTreeRegressor(),
            'SVR': SVR(),
            'KNN': KNeighborsRegressor(),
            'RandomForest': RandomForestRegressor(),
            'XGBoost': XGBRegressor()
        }

        model_scores = {}
        best_score = -float('inf')
        best_model_name = ""
        best_model_obj = None

        logger.info("Starting model training loop...")

        for name, model in models.items():
            # Train
            model.fit(X_train, y_train)
            # Predict
            y_pred = model.predict(X_test)
            # Evaluate (R2 Score as per your notebook)
            score = metrics.r2_score(y_test, y_pred)
            model_scores[name] = score
            
            logger.debug(f"Model: {name} | R2 Score: {score:.4f}")

            # Track the best performer
            if score > best_score:
                best_score = score
                best_model_name = name
                best_model_obj = model

        return model_scores, best_model_name, best_model_obj

    except Exception as e:
        logger.error('Error during model training/evaluation: %s', e)
        raise

def save_best_model(model_obj, model_name: str, models_path: str) -> None:
    """Save the best performing model to a .pkl file."""
    try:
        os.makedirs(models_path, exist_ok=True)
        model_filename = os.path.join(models_path, 'best_model.pkl')
        joblib.dump(model_obj, model_filename)
        logger.info(f"Best model ({model_name}) saved successfully to {model_filename}")
    except Exception as e:
        logger.error('Failed to save the model: %s', e)
        raise

# 3. MAIN TRAINING PIPELINE

def main():
    try:
        base_data_path = './data'
        models_dir = './models'
        target_col = 'Selling_Price'
        
        # Step 1: Load Processed Data
        processed_train_path = os.path.join(base_data_path, 'processed', 'train_processed.csv')
        processed_test_path = os.path.join(base_data_path, 'processed', 'test_processed.csv')
        
        train_df = load_processed_data(processed_train_path)
        test_df = load_processed_data(processed_test_path)
        
        # Step 2: Train and Compare
        scores, best_name, best_model = train_and_evaluate_models(train_df, test_df, target_col)
        
        # Step 3: Print summary and save best model
        print("\n--- Model Training Results ---")
        for m_name, m_score in scores.items():
            print(f"{m_name}: {m_score:.4f}")
        
        print(f"\nWinning Model: {best_name} with Score: {scores[best_name]:.4f}")
        
        save_best_model(best_model, best_name, models_dir)
        
        logger.info("Model Training stage completed successfully.")

    except Exception as e:
        logger.error('Model Training process failed: %s', e)

if __name__ == '__main__':
    main()