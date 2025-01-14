import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

class IrisClassification(nn.Module):
    def __init__(self):
        super(IrisClassification, self).__init__()
        # here the iris dataset has 4 features as input, and we are predicting from three classes
        # input = 5 final output = 3
        self.input_size = 4
        self.output_size = 3
        self.fc1 = nn.Linear(self.input_size, 40)
        self.fc2 = nn.Linear(40, 80)
        self.fc3 = nn.Linear(80, 100)
        self.fc4 = nn.Linear(100, self.output_size)
        self.dropout1 = nn.Dropout(p=0.2)
        self.dropout2 = nn.Dropout(p=0.5)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        self.dropout1(x)
        x = F.relu(self.fc2(x))
        self.dropout2(x)
        x = F.relu(self.fc3(x))
        self.dropout1(x)
        x = self.fc4(x)
        # the dim parameter specifies the dimension along which the softmax operation should be applied
        # When dim=1, the softmax function is applied along the rows, so each row sums to 1.
        # When dim=0, the softmax function is applied along the columns, so each column sums to 1.
        # CrossEntropyLoss in PyTorch combines both softmax and the negative log-likelihood loss in one single function.
        # So, you don't need to explicitly apply softmax before using CrossEntropyLoss.
        # x = F.softmax(x , dim=1)
        return x

class DataManager:
    def __init__(self):
        # Load the iris dataset
        self.iris = load_iris()
        self.X = self.iris.data
        self.y = self.iris.target

        # Split the dataset into training and test set
        # Split the dataset into training and test set
        # after the train test split we need to convert the data to tensors
        # train test split will return the data as ndarray this needs to be translated to tensors
        # X should be float32 and y should be long
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=42)
        # Standardize the dataset
        self.scaler = StandardScaler()
        self.X_train = self.scaler.fit_transform(self.X_train)
        self.X_test = self.scaler.transform(self.X_test)

        # Convert the dataset to tensor
        self.X_train = torch.FloatTensor(self.X_train)
        self.X_test = torch.FloatTensor(self.X_test)
        self.y_train = torch.LongTensor(self.y_train)
        self.y_test = torch.LongTensor(self.y_test)

    def get_data(self):
        return self.X_train, self.X_test, self.y_train, self.y_test

    def get_scaler(self):
        return self.scaler

    def get_iris(self):
        return self.iris

    def get_X(self):
        return self.X

    def get_y(self):
        return self.y

    def get_X_train(self):
        return self.X_train

    def get_X_test(self):
        return self.X_test

    def get_y_train(self):
        return self.y_train

    def get_y_test(self):
        return self.y_test

    def train(self):
        model = IrisClassification()
        # Define the loss function and optimizer
        # CrossEntropyLoss in PyTorch combines both softmax and the negative log-likelihood loss in one single function.
        # So, you don't need to explicitly apply softmax before using CrossEntropyLoss.
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.01)
        for epoch in range(100):
            optimizer.zero_grad()
            predicted = model(self.X_train)
            loss = criterion(predicted, self.y_train)
            loss.backward()
            optimizer.step()
            if epoch % 10 == 0:
                print(f'Epoch [{epoch}/100], Loss: {loss.item()}')
        return model

    def test(self, input_model):
        with torch.no_grad():
            outputs = input_model(self.X_test)
            _, predicted = torch.max(outputs, 1)
            acc = accuracy_score(self.y_test, predicted)
            print(f'Accuracy: {acc}')


if __name__ == "__main__":
    data = DataManager()
    iris_model = data.train()
    data.test(iris_model)