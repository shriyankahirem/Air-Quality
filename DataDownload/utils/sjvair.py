import pandas as pd
import requests
from io import StringIO


sensor_id_list = ["9tYuybX_ROqhTm5m6je79Q", "CCA4NsviS-K4reZPsMUTYQ"]


def get_one_sensor(sensor_id, start_date, end_date, save_path="./data/"):
    """
    Get data of one sensor from the SJV Air API.
    :param sensor_id: SJV Air sensor identifier
    :param start_date: datetime.datetime object
    :param end_date: datetime.datetime object
    :param save_path: folder path to save the file
    :return: a dataframe of the data in the time range
    """
    start_year = start_date.year
    start_month = start_date.month
    start_day = start_date.day
    start_hour = start_date.hour
    start_minute = start_date.minute
    start_second = start_date.second
    end_year = end_date.year
    end_month = end_date.month
    end_day = end_date.day
    end_hour = end_date.hour
    end_minute = end_date.minute
    end_second = end_date.second

    url_template = "https://www.sjvair.com/api/1.0/monitors/{}/entries/csv/?timestamp__gte={}-{}-{}+{}%3A{}%3A{}&timestamp__lte={}-{}-{}+{}%3A{}%3A{}"
    url = url_template.format(sensor_id,
                              start_year, start_month, start_day,
                              start_hour, start_minute, start_second,
                              end_year, end_month, end_day,
                              end_hour, end_minute, end_second)
    r = requests.get(url)
    df = pd.read_csv(StringIO(r.text), ",")
    df.to_csv(save_path + "sj_{}{}{}_{}{}{}_{}.csv".format(
        start_month, start_day, start_year,
        end_month, end_day, end_year,
        sensor_id
    ), index=False)

    return df


def clean_timestamp(df):
    """
    Convert timestamp to PDT and add year, month, day, hour, minute columns
    :param df: the dataframe with a timestamp column
    :return: cleand dataframe
    """
    df["timestamp"] = pd.to_datetime(df["timestamp"]).dt.tz_convert("US/Pacific")
    ts = pd.to_datetime(df["timestamp"])
    df["year"] = ts.dt.year
    df["month"] = ts.dt.month
    df["day"] = ts.dt.day
    df["hour"] = ts.dt.hour
    df["minute"] = ts.dt.minute

    df = df.drop(columns=["timestamp"])
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
