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
        self.fc1 = nn.Linear(self.input_size, 100)
        self.fc2 = nn.Linear(100, 100)
        self.fc3 = nn.Linear(100, self.output_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        # the dim parameter specifies the dimension along which the softmax operation should be applied
        # When dim=1, the softmax function is applied along the rows, so each row sums to 1.
        # When dim=0, the softmax function is applied along the columns, so each column sums to 1.
        # CrossEntropyLoss in PyTorch combines both softmax and the negative log-likelihood loss in one single function.
        # So, you don't need to explicitly apply softmax before using CrossEntropyLoss.
        # x = F.softmax(x , dim=1)
        return x




if __name__ == "__main__":
    # Load the iris dataset
    iris = load_iris()
    X = iris.data
    y = iris.target

    # Split the dataset into training and test set
    # after the train test split we need to convert the data to tensors
    # train test split will return the data as ndarray this needs to be translated to tensors
    # X should be float32 and y should be long
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Standardize the dataset
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Convert the dataset to tensor
    X_train = torch.FloatTensor(X_train)
    X_test = torch.FloatTensor(X_test)
    y_train = torch.LongTensor(y_train)
    y_test = torch.LongTensor(y_test)

    # Create the model
    model = IrisClassification()

    # Define the loss function and optimizer
    # CrossEntropyLoss in PyTorch combines both softmax and the negative log-likelihood loss in one single function.
    # So, you don't need to explicitly apply softmax before using CrossEntropyLoss.
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    # Train the model
    for epoch in range(100):
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()

        if epoch % 10 == 0:
            print(f'Epoch [{epoch}/100], Loss: {loss.item()}')

    # Test the model
    with torch.no_grad():
        outputs = model(X_test)
        _, predicted = torch.max(outputs, 1)
        acc = accuracy_score(y_test, predicted)
        print(f'Accuracy: {acc}')