import pandas as pd
import numpy as np


def average_hour(df, columns=["longitude", "latitude", "pm25"]):
    """
    Set negative values to zero and average readings for each hour
    Input:
        df: a dataframe with a column named "timestamp" and another column named "pm25
        columns: a list of columns to be averaged
    Ouput:
        df: a dataframe with averaged pm25 readings for each hour
    """

    # decompose timestamp
    df['timestamp'] = pd.to_datetime(df['timestamp'], format='mixed')
    df['year'] = df['timestamp'].dt.year
    df['month'] = df['timestamp'].dt.month
    df['day'] = df['timestamp'].dt.day
    df['hour'] = df['timestamp'].dt.hour
    df['weekday'] = df['timestamp'].dt.weekday

    # set negative pm25 to 0
    df.loc[df["pm25"] < 0, "pm25"] = 0

    # averaging
    df = df.loc[:, columns + ["year", "month", "day", "hour", "weekday"]]
    df = df.groupby(["year", "month", "day", "hour", "weekday"]).mean().reset_index(drop=False)
    
    return df


def mean_nan(x):
    """
    Compute the mean of a sequence with NaN values
    Input:
        x: a sequence of numbers
    Output:
        mean: the mean of x
    """
    mask = ~np.isnan(x)
    if mask.sum() == 0:
        return np.nan
    mean = np.mean(x[mask])
    return mean

def median_nan(x):
    """
    Compute the median of a sequence with NaN values
    Input:
        x: a sequence of numbers
    Output:
        median: the median of x
    """
    mask = ~np.isnan(x)
    if mask.sum() == 0:
        return np.nan
    median = np.median(x[mask])
    return median


def average_correlation_nan(corr_matrix, method='mean'):
    """
    Analyze correlation matrix with NaN values
    """
    # Set diagonal to Nan
    corr_matrix = corr_matrix.copy()
    np.fill_diagonal(corr_matrix, np.nan)
    # Compute the average correlation
    avg_corr = []
    for row in corr_matrix:
        if method == 'mean':
            avg_corr.append(mean_nan(row))
        elif method == 'median':
            avg_corr.append(median_nan(row))
    return np.array(avg_corr)


def corrcoef_nan(x, y):
    """
    Compute the correlation coefficient between two sequences with NaN values
    Input:
        x: a sequence of numbers
        y: a sequence of numbers
    Output:
        corr: the correlation coefficient between x and y
    """
    mask = ~np.isnan(x) & ~np.isnan(y)
    if mask.sum() < 2:
        return np.array([[np.nan, np.nan], [np.nan, np.nan]])
    corr = np.corrcoef(x[mask], y[mask])
    return corr
