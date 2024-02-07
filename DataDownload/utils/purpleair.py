import datetime
import pandas as pd
import requests
import time
from io import StringIO


sensor_id_list = ["155565", "163173"]


def get_one_sensor_one_field(sensor_id, start_date, end_date, field="pm2.5_alt"):
    """
    Get one sensor's one field data from purpleair api
    :param sensor_id: sensor identifier
    :param start_date: a datetime object
    :param end_date: a datetime object
    :param field: field name
    :return: a dataframe of the data in the time range and given field
    """
    api_headers = {'X-API-Key': 'A9084A32-4059-11EE-A77F-42010A800009'}
    sensor_id_int = int(sensor_id)
    start_date = start_date.timestamp()
    end_date = end_date.timestamp()
    r = requests.get(f"https://api.purpleair.com/v1/sensors/{sensor_id_int}/history/csv", headers=api_headers, params={
        "start_timestamp": start_date,
        "average": 0,
        "end_timestamp": end_date,
        "fields": [field]
    })
    df = pd.read_csv(StringIO(r.text))
    # print(df)
    df = df.sort_values("time_stamp", ascending=True)
    return df


def get_one_sensor_multi_fields(sensor_id, start_date, end_date, fields):
    """
    Get one sensor's multiple fields data from purpleair api. The time range should be less than 2 days.
    :param sensor_id: sensor identifier
    :param start_date: a datetime object
    :param end_date: a datetime object
    :param fields: a list of field names
    :return: a dataframe of the data in the time range and given fields
    """
    for field in fields:
        time.sleep(0.5)  # frequently call purpleair api will cause error
        # print(field)
        df = get_one_sensor_one_field(sensor_id, start_date, end_date, field)
        if field == fields[0]:
            df_all = df
        else:
            df_all = df_all.merge(df, on=["time_stamp", "sensor_index"])
    return df_all


def get_sensor_data(sensor_id, start_date, end_date, fields):
    """
    Get one sensor's multiple fields data from purpleair api. No time range limit.
    :param sensor_id: sensor identifier
    :param start_date: a datetime object
    :param end_date: a datetime object
    :param fields: a list of field names
    :return: a dataframe of the data in the time range and given fields
    """
    # create a date sequence, since Purple Air API only allows 2 days interval for real-time data
    date_sequence = []
    while start_date < end_date:
        date_sequence.append(start_date)
        start_date += datetime.timedelta(days=1)
    date_sequence.append(end_date)

    for i in range(len(date_sequence) - 1):
        start_date = date_sequence[i]
        end_date = date_sequence[i + 1]
        df = get_one_sensor_multi_fields(sensor_id, start_date, end_date, fields)
        if i == 0:
            df_all = df
        else:
            df_all = pd.concat([df_all, df], axis=0)

    return df_all


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
