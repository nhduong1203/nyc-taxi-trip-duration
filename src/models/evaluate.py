import numpy as np
import pandas as pd


def evaluate_metric(y, y_pred):
    # Mean Absolute Error (MAE)
    mae = np.mean(np.abs(y - y_pred))

    # Mean Squared Error (MSE)
    mse = np.mean((y - y_pred) ** 2)

    # Root Mean Squared Error (RMSE)
    rmse = np.sqrt(np.mean((y - y_pred) ** 2))

    # Root Mean Squared Log Error (RMSLE)
    rmsle = np.sqrt(np.mean((np.log(y_pred + 1) - np.log(y + 1)) ** 2))

    # R-squared
    r2 = 1 - np.sum((y - y_pred) ** 2) / np.sum((y - np.mean(y)) ** 2)

    # Mean Absolute Percentage Error (MAPE)
    mape = np.mean(2 * np.abs(y - y_pred) / (np.abs(y) + np.abs(y_pred))) * 100

    # Symmetric Mean Absolute Percentage Error (SMAPE)
    smape = np.mean(2 * np.abs(y - y_pred) / (np.abs(y) + np.abs(y_pred))) * 100

    # Mean Directional Accuracy (MDA)
    mda = np.mean(np.sign(y[2:] - y[1:-1]) == np.sign(y_pred[2:] - y[1:-1]))

    # Median Absolute Error (MedAE)
    medAE = np.median(np.abs(y - y_pred))

    # Explained Variance Score (EVS)
    evs = 1 - np.var(y - y_pred) / np.var(y)

    metrics_df = pd.DataFrame({
        'Metric': ['MAE', 'MSE', 'RMSE', 'RMSLE', 'R2', 'MAPE', 'SMAPE', 'MDA', 'MedAE', 'EVS'],
        'Value': [mae, mse, rmse, rmsle, r2, mape, smape, mda, medAE, evs]
    })

    return metrics_df