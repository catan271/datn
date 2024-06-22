import pandas as pd
import datetime
import matplotlib.pyplot as plt
import os
import numpy as np
import json
from scipy import stats

def data_scale(array: np.ndarray, name: str):
    with open('../datasets/parameters.json', 'r') as file:
        parameters = json.load(file)
    # val_mean = parameters[f'{name}_mean']
    # val_std = parameters[f'{name}_std']
    val_min = parameters[f'{name}_min']
    val_max = parameters[f'{name}_max']
    return (array - val_min) / (val_max - val_min)

def data_descale(array: np.ndarray, name: str):
    with open('../datasets/parameters.json', 'r') as file:
        parameters = json.load(file)
    # val_mean = parameters[f'{name}_mean']
    # val_std = parameters[f'{name}_std']
    val_min = parameters[f'{name}_min']
    val_max = parameters[f'{name}_max']
    return array * (val_max - val_min) + val_min


def date_range(start: datetime.date, end: datetime.date, step: datetime.timedelta):
    while start < end:
        yield start
        start += step  

def df_to_X_y(df: pd.DataFrame):
    df_as_np = df.to_numpy()
    X = df_as_np[:, 4:]
    y = df_as_np[:, :4]
    return X, y

def preprocess_b(df: pd.DataFrame):
    prev_days = int(os.getenv('PREV_DAYS'))
    array = df.to_numpy()
    
    y = array[:, 0:3]
    y[:, 0] = data_scale(y[:, 0], 'view')
    y[:, 1] = data_scale(y[:, 1], 'cart')
    y[:, 2] = data_scale(y[:, 2], 'purchase')

    X = array[:, 3:prev_days * 3 + 6]
    X[:, 0] = (1 - data_scale(X[:, 0], 'rank_in_category'))
    X[:, 1] = (1 - data_scale(X[:, 1], 'days_on_shelf'))
    X[:, 2] = data_scale(X[:, 2], 'price')
    
    X[:, 3::3] = data_scale(X[:, 3::3], 'view')
    X[:, 4::3] = data_scale(X[:, 4::3], 'cart')    
    X[:, 5::3] = data_scale(X[:, 5::3], 'purchase')

    return X, y
    

def preprocess_c(df: pd.DataFrame):
    prev_days = int(os.getenv('PREV_DAYS'))
    array = df.to_numpy()
    y = array[:, 0:3]
    y[:, 0] = data_scale(y[:, 0], 'view')
    y[:, 1] = data_scale(y[:, 1], 'cart')
    y[:, 2] = data_scale(y[:, 2], 'purchase')

    X1 = array[:, 3:6]
    X1 = X1.reshape(X1.shape[0], 1, 3)
    X1[:, :, 0] = (1 - data_scale(X1[:, :, 0], 'rank_in_category'))
    X1[:, :, 1] = data_scale(X1[:, :, 1], 'days_on_shelf')
    X1[:, :, 2] = data_scale(X1[:, :, 2], 'price')
    

    X2 = array[:, 6:prev_days * 3 + 6]
    X2 = X2.reshape(X2.shape[0], prev_days, 3)
    X2[:, :, 0] = data_scale(X2[:, :, 0], 'view')
    X2[:, :, 1] = data_scale(X2[:, :, 1], 'cart')
    X2[:, :, 2] = data_scale(X2[:, :, 2], 'purchase')
    return X1, X2, y  

def plot_accuracy(predicted_values, actual_values, feature_name='Values', limit :tuple[int, int] = None):
    plt.figure(figsize=(10, 10))
    # Create scatter plot
    plt.scatter(x=actual_values, y=predicted_values, alpha=0.1, s=8, color="blue")
    # heatmap
    # plt.hexbin(actual_values, predicted_values, gridsize=100, cmap='viridis', mincnt=1, vmin=0, vmax=50)


    # Add line of equality for reference
    min_val = min(min(actual_values), min(predicted_values))
    max_val = max(max(actual_values), max(predicted_values))
    plt.plot([min_val, max_val], [min_val, max_val], color='red', linestyle='--', label='Line of Equality')

    slope, intercept, r_value, p_value, std_err = stats.linregress(actual_values, predicted_values)
    p_display = 'p < 22e-16' if p_value < 22e-16 else f'p = {p_value:.2e}'
    plt.text(min_val + (max_val - min_val) * 0.05, max_val - (max_val - min_val) * 0.1, f'R = {r_value:.2f}, {p_display}')

    # Add labels and title
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.title(f'Scatter Plot of Predicted vs Actual {feature_name}')
    plt.legend()

    if limit:
        plt.xlim(*limit)
        plt.ylim(*limit)

    # Show plot
    plt.show()