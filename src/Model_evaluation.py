import pandas as pd
import os
import logging
import joblib
import matplotlib.pyplot as plt
import json  
import yaml
from sklearn import metrics
from dvclive import Live

# 1. LOGGING CONFIGURATION

log_dir = 'logs'
os.makedirs(log_dir, exist_ok=True)

logger = logging.getLogger('model_evaluation')
logger.setLevel('DEBUG')

# Handlers
console_handler = logging.StreamHandler()
file_handler = logging.FileHandler(os.path.join(log_dir, 'model_evaluation.log'))

# Formatter
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

# 2. FUNCTION DEFINITIONS

def load_params(params_path: str) -> dict:
    """Load parameters from a YAML file."""
    try:
        with open(params_path, 'r') as file:
            params = yaml.safe_load(file)
        logger.debug('Parameters retrieved from %s', params_path)
        return params
    except Exception as e:
        logger.error('Unexpected error loading params: %s', e)
        raise

def load_evaluation_artifacts(model_path: str, train_path: str, test_path: str):
    """Load the model and both processed datasets for evaluation."""
    try:
        model = joblib.load(model_path)
        train_df = pd.read_csv(train_path)
        test_df = pd.read_csv(test_path)
        logger.debug('Successfully loaded model and processed data.')
        return model, train_df, test_df
    except Exception as e:
        logger.error('Failed to load evaluation artifacts: %s', e)
        raise

def evaluate_performance(model, df: pd.DataFrame, target_column: str, stage_name: str):
    """Calculate R2 Score and MAE for a specific dataset stage."""
    try:
        X = df.drop(columns=[target_column])
        y = df[target_column]
        
        predictions = model.predict(X)
        r2 = metrics.r2_score(y, predictions)
        mae = metrics.mean_absolute_error(y, predictions)
        
        logger.info(f"{stage_name} Results -> R2: {r2:.4f}, MAE: {mae:.4f}")
        return y, predictions, r2, mae

    except Exception as e:
        logger.error(f'Error during {stage_name} evaluation: {e}')
        raise

def save_evaluation_plots(y_actual, y_pred, stage_name: str, reports_dir: str):
    """Generate and save scatter plots."""
    try:
        plt.figure(figsize=(10, 6))
        plt.scatter(y_actual, y_pred, color='blue', alpha=0.5)
        plt.xlabel('Actual Price')
        plt.ylabel('Predicted Price')
        plt.title(f'Actual vs Predicted Price - {stage_name}')
        
        plot_filename = os.path.join(reports_dir, f'{stage_name.lower()}_comparison.png')
        plt.savefig(plot_filename)
        plt.close()
        logger.debug(f'Saved comparison plot for {stage_name} at {plot_filename}')
    except Exception as e:
        logger.error(f'Failed to save plot for {stage_name}: {e}')
        raise

# 3. MAIN EVALUATION PIPELINE (WITH DVCLIVE)

def main():
    try:
        # Load params from YAML
        params = load_params(params_path='params.yaml')
        
        # Paths
        base_data_path = './data/processed'
        model_path = './models/best_model.pkl'
        reports_dir = './reports'
        os.makedirs(reports_dir, exist_ok=True)
        
        target_col = 'Selling_Price'

        # Step 1: Load everything
        best_model, train_processed, test_processed = load_evaluation_artifacts(
            model_path,
            os.path.join(base_data_path, 'train_processed.csv'),
            os.path.join(base_data_path, 'test_processed.csv')
        )

        # Step 2: Evaluate Performance
        y_train_act, y_train_pred, r2_train, mae_train = evaluate_performance(
            best_model, train_processed, target_col, "Training_Set"
        )
        y_test_act, y_test_pred, r2_test, mae_test = evaluate_performance(
            best_model, test_processed, target_col, "Testing_Set"
        )

        # Step 3: Experiment tracking using DVCLIVE (From Image Logic)
        with Live(save_dvc_exp=True) as live:
            # Log Metrics
            live.log_metric('train/r2', float(r2_train))
            live.log_metric('test/r2', float(r2_test))
            live.log_metric('test/mae', float(mae_test))
            
            # Log Parameters
            live.log_params(params)
            
            # Save visual plots
            save_evaluation_plots(y_train_act, y_train_pred, "Training_Set", reports_dir)
            save_evaluation_plots(y_test_act, y_test_pred, "Testing_Set", reports_dir)

        # Step 4: Legacy JSON save (to keep your dvc.yaml happy)
        metrics_data = {
            "train_r2": float(r2_train),
            "test_r2": float(r2_test),
            "mae": float(mae_test)
        }
        with open(os.path.join(reports_dir, 'metrics.json'), 'w') as f:
            json.dump(metrics_data, f, indent=4)
            
        print("\n--- Model Evaluation Complete ---")
        print(f"Test R2 Score:  {r2_test:.4f}")
        logger.info("Model Evaluation stage completed successfully.")

    except Exception as e:
        logger.error('Evaluation process failed: %s', e)

if __name__ == '__main__':
    main()