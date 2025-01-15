import os

import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from sklearn.preprocessing import StandardScaler
from torch import nn
import torch.nn.functional as F

# We are not using this class in this file
# We are loading the dataset from the csv file
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
    def __init__(self, input_data, input_labels):
        if isinstance(input_data, pd.DataFrame):
            input_data = input_data.values

        if isinstance(input_labels, pd.Series):
            input_labels = input_labels.values

        self.data = torch.FloatTensor(input_data)
        self.labels = torch.FloatTensor(input_labels) # for loss calculation we need this to be float

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

    def get_data(self):
        return self.data

    def get_labels(self):
        return self.labels

def get_data_loader(data_file_path: str = '../data/data.csv'):
    if not os.path.exists(data_file_path):
        raise FileNotFoundError(f'Data file not found: {data_file_path}')
    df = pd.read_csv(data_file_path)
    # drop Address column
    df = df.drop('Address', axis=1)
    # split data into X and y
    X = df.drop('Price', axis=1)
    y = df['Price']

    # split the data into train, test and validation sets
    train_test_split_ratio = 0.2
    test_validation_split_ratio = 0.5
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=train_test_split_ratio, random_state=42, shuffle=True)
    X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, test_size=test_validation_split_ratio, random_state=42, shuffle=True)

    # standard scalar
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    X_val = scaler.transform(X_val)

    # create train, test and validation datasets
    train_dataset = NumericalDataset(X_train, y_train)
    test_dataset = NumericalDataset(X_test, y_test)
    val_dataset = NumericalDataset(X_val, y_val)

    # data loader
    train_loader = DataLoader(train_dataset, batch_size=10, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=2, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=2, shuffle=True)

    return train_loader, test_loader, val_loader

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.fc1 = nn.Linear(5, 10)
        self.fc2 = nn.Linear(10, 3)
        self.fc3 = nn.Linear(3, 1)
        self.dropout1 = nn.Dropout(0.2)
        self.dropout2 = nn.Dropout(0.4)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        x = self.softmax(x)
        return x



if __name__ == '__main__':
    # get the dataloader
    train, test, val = get_data_loader()
    # print the data
    for data, labels in train:
        print(f"Data shape: {data.shape}, Labels shape: {labels.shape}")

    # create the model
    model = Model()
    print(model)
    no_of_epochs = 200
    criterion = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    # train the model
    for epoch in range(no_of_epochs):
        for X_batch, y_batch in train:
            optimizer.zero_grad()
            y_pred = model(X_batch)
            loss = criterion(y_pred, y_batch)
            loss.backward()
            optimizer.step()
            print(f'Epoch: {epoch}, Loss: {loss.item()}')


    # test the model
    with torch.no_grad():
        for X_batch, y_batch in test:
            y_pred = model(X_batch)
            loss = criterion(y_pred, y_batch)
            print(f'Test Loss: {loss.item()}')
            print(f'Predicted: {y_pred}')
            print(f'Actual: {y_batch}')
            break









