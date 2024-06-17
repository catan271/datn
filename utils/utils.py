import pandas as pd
import datetime
import matplotlib.pyplot as plt
import os
import numpy as np
import json

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
    
    y = array[:, 0:4]
    y[:, 0] = data_scale(y[:, 0], 'view')
    y[:, 1] = data_scale(y[:, 1], 'cart')
    y[:, 2] = data_scale(y[:, 2], 'remove_from_cart')
    y[:, 3] = data_scale(y[:, 3], 'purchase')

    X = array[:, 4:prev_days * 4 + 8]
    X[:, 0] = (1 - data_scale(X[:, 0], 'rank'))
    X[:, 1] = (1 - data_scale(X[:, 1], 'rank_in_category'))
    X[:, 2] = data_scale(X[:, 2], 'days_on_shelf')
    X[:, 3] = data_scale(X[:, 3], 'price')
    
    X[:, 4::4] = data_scale(X[:, 4::4], 'view')
    X[:, 5::4] = data_scale(X[:, 5::4], 'cart')
    X[:, 6::4] = data_scale(X[:, 6::4], 'remove_from_cart')
    X[:, 7::4] = data_scale(X[:, 7::4], 'purchase')

    return X, y
    

def preprocess_c(df: pd.DataFrame):
    prev_days = int(os.getenv('PREV_DAYS'))
    array = df.to_numpy()
    y = array[:, 0:4]
    y[:, 0] = data_scale(y[:, 0], 'view')
    y[:, 1] = data_scale(y[:, 1], 'cart')
    y[:, 2] = data_scale(y[:, 2], 'remove_from_cart')
    y[:, 3] = data_scale(y[:, 3], 'purchase')

    X1 = array[:, 4:8]
    X1 = X1.reshape(X1.shape[0], 1, 4)
    X1[:, :, 0] = (1 - data_scale(X1[:, :, 0], 'rank'))
    X1[:, :, 1] = (1 - data_scale(X1[:, :, 1], 'rank_in_category'))
    X1[:, :, 2] = data_scale(X1[:, :, 2], 'days_on_shelf')
    X1[:, :, 3] = data_scale(X1[:, :, 3], 'price')
    

    X2 = array[:, 8:prev_days * 4 + 8]
    X2 = X2.reshape(X2.shape[0], prev_days, 4)
    X2[:, :, 0] = data_scale(X2[:, :, 0], 'view')
    X2[:, :, 1] = data_scale(X2[:, :, 1], 'cart')
    X2[:, :, 2] = data_scale(X2[:, :, 2], 'remove_from_cart')
    X2[:, :, 3] = data_scale(X2[:, :, 3], 'purchase')
    return X1, X2, y  

def postprocess_c(y):
    y[:, 0] = data_descale(y[:, 0], 'view')
    y[:, 1] = data_descale(y[:, 1], 'cart')
    y[:, 2] = data_descale(y[:, 2], 'remove_from_cart')
    y[:, 3] = data_descale(y[:, 3], 'purchase')
    return y

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