import os

import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd

class TestData:
    def __init__(self):
        self.data_list = [
            [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 1],
            [2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 0],
            [3.0, 6.0, 9.0, 12.0, 15.0, 18.0, 1],
            [4.0, 8.0, 12.0, 16.0, 20.0, 24.0, 0],
            [5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 1],
            [6.0, 12.0, 18.0, 24.0, 30.0, 36.0, 2],
            [7.0, 14.0, 21.0, 28.0, 35.0, 42.0, 1],
            [8.0, 16.0, 24.0, 32.0, 40.0, 48.0, 0],
            [9.0, 18.0, 27.0, 36.0, 45.0, 54.0, 2],
            [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 2],
            [11.0, 22.0, 33.0, 44.0, 55.0, 66.0, 1]
        ]

    def get_data(self):
        return self.data_list

class NumericalDataset(Dataset):
    def __init__(self, data, labels):
        if isinstance(data, pd.DataFrame):
            self.data=data.values

        if isinstance(labels, pd.Series):
            self.labels=labels.values

        self.data = torch.tensor(data, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


if __name__ == '__main__':
    data_file_path: str = 'data/data.csv'
    if not os.path.exists(data_file_path):
        raise FileNotFoundError(f'Data file not found: {data_file_path}')
    df = pd.read_csv('data/data.csv')
    # drop Address column
    df = df.drop('Address', axis=1)
    # split data into X and y
    X = df.drop('Price', axis=1)
    y = df['Price']





