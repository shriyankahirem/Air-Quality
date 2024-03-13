import numpy as np


def count_params(model):
    num_params = sum([p.numel() for p in model.parameters() if p.requires_grad])
    return num_params

def model_size(model):
    """Calculate the size of the model in mb, given the number of parameters"""
    num_params = count_params(model)
    return num_params * 4 / 1024 / 1024

def prediction_summary(y_true, y_pred, print_output=True):
    """Calculate various metrics for the model's predictions"""
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    cvrmse = rmse / y_true.mean()
    mae = np.mean(np.abs(y_true - y_pred))
    r2 = 1 - np.sum((y_true - y_pred) ** 2) / np.sum((y_true - y_true.mean()) ** 2)
    
    if print_output:
        print(f'RMSE: {rmse:.4f}, CVRMSE: {cvrmse:.4f}, MAE: {mae:.4f}, R2: {r2:.4f}')
    else:
        return rmse, cvrmse, mae, r2