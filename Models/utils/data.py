import os
import torch
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
    # df['timestamp'] = pd.to_datetime(df['timestamp'])
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

def load_data(train_ratio=0.7, val_ratio=0.1, test_ratio=0.2, batch_size=32, seed=42):
    data_dir = '/Users/shangjiedu/Desktop/aJay/Merced/Research/Air Quality/InterpolationBaseline/data/Oct0123_Jan3024/'
    data_dir = '../InterpolationBaseline/data/Oct0123_Jan3024/'

    files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]
    files.sort()
    data = []
    for i, file in enumerate(files):
        df = pd.read_csv(data_dir + file)
        df = average_hour(df)

        # remove sensors with missing data
        if len(df) < 2928:
            print("File{}: {} contains missing hours".format(i, file))
            continue

        # remove sensors with outliers
        if df["pm25"].max() > 500:
            print("File{}: {} contains outliers".format(i, file))
            continue

        data.append(df.loc[:, ['pm25', 'longitude', 'latitude']])

    data = np.array(data).transpose(1, 0, 2)
    
    np.random.seed(seed)
    perm = np.random.permutation(data.shape[1])
    n_train = int(train_ratio * data.shape[1])
    n_val = int(val_ratio * data.shape[1])
    n_test = int(test_ratio * data.shape[1])
    train_idx = perm[:n_train]
    val_idx = perm[n_train: n_train + n_val]
    test_idx = perm[n_train + n_val:]
    idx_list = [train_idx, val_idx, test_idx]

    train_dataset = AirQualityDataset(data, idx_list=idx_list, dataset_type="train")
    val_dataset = AirQualityDataset(data, idx_list=idx_list, dataset_type="val")
    test_dataset = AirQualityDataset(data, idx_list=idx_list, dataset_type="test")

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, test_loader


def load_data2(train_ratio=0.7, test_ratio=0.3, batch_size=256, seed=42):
    data_dir = '/Users/shangjiedu/Desktop/aJay/Merced/Research/Air Quality/InterpolationBaseline/data/Oct0123_Jan3024/'
    data_dir = '../InterpolationBaseline/data/Oct0123_Jan3024/'

    files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]
    files.sort()
    data = []
    for i, file in enumerate(files):
        df = pd.read_csv(data_dir + file)
        df = average_hour(df)

        # remove sensors with missing data
        if len(df) < 2928:
            print("File{}: {} contains missing hours".format(i, file))
            continue

        # remove sensors with outliers
        if df["pm25"].max() > 500:
            print("File{}: {} contains outliers".format(i, file))
            continue

        data.append(df.loc[:, ['pm25', 'longitude', 'latitude']])

    data = np.array(data).transpose(1, 0, 2)
    
    np.random.seed(seed)
    perm = np.random.permutation(data.shape[1])
    n_train = int(train_ratio * data.shape[1])
    n_test = int(test_ratio * data.shape[1])
    train_idx = perm[:n_train]
    val_idx = None
    test_idx = perm[n_train:]
    idx_list = [train_idx, val_idx, test_idx]

    train_dataset = AirQualityDataset(data, idx_list=idx_list, dataset_type="train")
    test_dataset = AirQualityDataset(data, idx_list=idx_list, dataset_type="test")

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader

class AirQualityDataset(torch.utils.data.Dataset):
    def __init__(self, data, window=24, idx_list=None, dataset_type="train"):
        """
        Input:
            data: a 3D numpy array with shape (n_steps, n_sensors, 3), where the last dimension contains pm25 readings, longitude and latitude
            window: the number of lookback hours
        """
        self.readings = torch.from_numpy(data[:, :, 0]).float()   # (n_steps, n_sensors)

        self.locations = torch.from_numpy(data[0, :, 1:]).float()   # (n_sensors, 2)
        self.locations[:, 0] /= 180
        self.locations[:, 1] /= 90

        self.window = window
        self.train_idx = idx_list[0]
        self.val_idx = idx_list[1]
        self.test_idx = idx_list[2]
        self.type= dataset_type

    def __len__(self):
        if self.type == "train":
            return (self.readings.shape[0] - self.window + 1) * len(self.train_idx)
        elif self.type == "val":
            return (self.readings.shape[0] - self.window + 1) * len(self.val_idx)
        else:
            return (self.readings.shape[0] - self.window + 1) * len(self.test_idx)
        
    def __getitem__(self, idx):
        if self.type == "train":
            target_idx = self.train_idx[idx % len(self.train_idx)]
            monitored_idx = np.concatenate([self.train_idx[: idx % len(self.train_idx)], self.train_idx[idx % len(self.train_idx) + 1:]])
            start = idx // len(self.train_idx)
            end = start + self.window
        elif self.type == "val":
            target_idx = self.val_idx[idx % len(self.val_idx)]
            monitored_idx = self.train_idx
            start = idx // len(self.val_idx)
            end = start + self.window
        else:
            target_idx = self.test_idx[idx % len(self.test_idx)]
            monitored_idx = self.train_idx
            start = idx // len(self.test_idx)
            end = start + self.window

        target_location = self.locations[target_idx, :]
        target_reading = self.readings[start:end, target_idx]
        monitored_locations = self.locations[monitored_idx, :]
        monitored_readings = self.readings[start:end, monitored_idx]

        return monitored_locations, monitored_readings, target_location, target_reading
    