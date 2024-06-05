import pandas as pd
import datetime
import matplotlib.pyplot as plt
import seaborn as sns


def df_to_X_y(df: pd.DataFrame):
    df_as_np = df.to_numpy()
    X = df_as_np[:, 4:]
    y = df_as_np[:, :4]
    return X, y

def date_range(start: datetime.date, end: datetime.date, step: datetime.timedelta):
    while start < end:
        yield start
        start += step        

def plot_accuracy(predicted_values, actual_values, feature_name='Values', limit :tuple[int, int] = None):
    plt.figure(figsize=(15, 14))
    # Create scatter plot
    plt.scatter(x=actual_values, y=predicted_values, alpha=0.5, s=1, color="blue")
    # heatmap
    # sns.kdeplot(x=actual_values, y=predicted_values, cmap="viridis", fill=False, thresh=0, levels=100)

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