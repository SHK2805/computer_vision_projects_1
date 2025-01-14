import torch
import torch.nn as nn

class FunctionalModel(nn.Module):
    def __init__(self):
        # Call the parent class constructor
        super(FunctionalModel, self).__init__()
        # Define the input layer
        self.input = nn.Linear(20, 64)
        # Define the hidden layers
        self.hidden1 = nn.Linear(64, 32)
        self.hidden2 = nn.Linear(32, 16)
        # Define the output layer
        self.output = nn.Linear(16, 5)
        # relu activation function
        self.relu = nn.ReLU()

    # Define the model
    def forward(self, x):
        # Define a functional model
        x = self.relu(self.input(x))
        x = self.relu(self.hidden1(x))
        x = self.relu(self.hidden2(x))
        x = self.output(x)
        return x

if __name__ == "__main__":
    X = torch.randint(1, 10, (10, 20)).to(torch.float32)
    y = torch.randint(1, 10, (10, 5)).to(torch.float32)
    model = FunctionalModel()
    no_of_epochs = 250
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()
    for epoch in range(no_of_epochs):
        optimizer.zero_grad()
        y_pred = model(X)
        loss = criterion(y_pred, y)
        loss.backward()
        optimizer.step()
        print(f"Epoch: {epoch+1}, Loss: {loss.item() :.4f}")

    print(f"Weight: {model.output.weight.data}, Bias: {model.output.bias.data}")

    test = torch.randint(1, 10, (1, 20)).to(torch.float32)
    with torch.no_grad():
        print(f"Prediction for tensor {test} is : {model(test)}")
