import datetime
import pandas as pd


sensor_id_list = ['ucm']


def column_preprocessing(df, start_date, end_date):
    df = df[["Local Date/Time", "PM2.5 (μg/m³)", "Temperature (°C)", "Humidity (%)"]]
    df.columns = ["timestamp", "pm25", "temperature", "humidity"]
    df.dropna(inplace=True)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df.sort_values(by=["timestamp"], inplace=True)

    df = df[(df["timestamp"] >= start_date) & (df["timestamp"] < end_date)]
    return df


def clean_timestamp(df):
    """
    Convert timestamp to PDT and add year, month, day, hour, minute columns
    :param df: the dataframe with a timestamp column
    :return: cleand dataframe
    """
    ts = pd.to_datetime(df["time_stamp"], unit="s") - datetime.timedelta(hours=7)   # convert to PDT
    df["year"] = ts.dt.year
    df["month"] = ts.dt.month
    df["day"] = ts.dt.day
    df["hour"] = ts.dt.hour
    df["minute"] = ts.dt.minute

    df = df.drop(columns=["time_stamp"])

    return df


def average_30min(df):
    """
    Average the data in each 30 minutes
    :param df: dataframe with cleaned timestamp
    :return: averaged dataframe
    """
    df["minute"] = df["minute"].apply(lambda x: 0 if x < 30 else 30)
    df = df.groupby(["year", "month", "day", "hour", "minute"]).mean().reset_index()
    return df
