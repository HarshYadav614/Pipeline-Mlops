import pandas as pd
import os
import logging
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics

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

def load_evaluation_artifacts(model_path: str, train_path: str, test_path: str):
    """Load the model and both processed datasets for evaluation."""
    try:
        model = joblib.load(model_path)
        train_df = pd.read_csv(train_path)
        test_df = pd.read_csv(test_path)
        logger.debug('Successfully loaded model and processed data for evaluation.')
        return model, train_df, test_df
    except Exception as e:
        logger.error('Failed to load evaluation artifacts: %s', e)
        raise

def evaluate_performance(model, df: pd.DataFrame, target_column: str, stage_name: str):
    """Calculate R2 Score and Error for a specific dataset stage."""
    try:
        X = df.drop(columns=[target_column])
        y = df[target_column]
        
        predictions = model.predict(X)
        r2 = metrics.r2_score(y, predictions)
        mae = metrics.mean_absolute_error(y, predictions)
        
        logger.info(f"{stage_name} Results -> R2: {r2:.4f}, MAE: {mae:.4f}")
        return y, predictions, r2

    except Exception as e:
        logger.error(f'Error during {stage_name} evaluation: {e}')
        raise

def save_evaluation_plots(y_actual, y_pred, stage_name: str, reports_dir: str):
    """Generate and save scatter plots (Actual vs Predicted) as per Jupyter cells 57-58."""
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

# 3. MAIN EVALUATION PIPELINE

def main():
    try:
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

        # Step 2: Evaluate Training Set (Cell 57 logic)
        y_train_act, y_train_pred, r2_train = evaluate_performance(
            best_model, train_processed, target_col, "Training_Set"
        )
        save_evaluation_plots(y_train_act, y_train_pred, "Training_Set", reports_dir)

        # Step 3: Evaluate Testing Set (Cell 58 logic)
        y_test_act, y_test_pred, r2_test = evaluate_performance(
            best_model, test_processed, target_col, "Testing_Set"
        )
        save_evaluation_plots(y_test_act, y_test_pred, "Testing_Set", reports_dir)

        print("\n--- Model Evaluation Complete ---")
        print(f"Train R2 Score: {r2_train:.4f}")
        print(f"Test R2 Score:  {r2_test:.4f}")
        print(f"Reports saved in: {os.path.abspath(reports_dir)}")
        
        logger.info("Model Evaluation stage completed successfully.")

    except Exception as e:
        logger.error('Evaluation process failed: %s', e)

if __name__ == '__main__':
    main()