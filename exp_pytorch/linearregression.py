import torch
import torch.nn as nn


class LinearRegression(nn.Module):
    def __init__(self):
        # Call the parent class constructor
        super(LinearRegression, self).__init__()
        self.linear = nn.Linear(in_features=1, out_features=1)
        self.relu = nn.ReLU()

    # Define the model
    def forward(self, x):
        # Define a linear model
        return self.relu(self.linear(x))

if __name__ == "__main__":
    model = LinearRegression()
    # Define the input tensor
    X = torch.tensor([[1.0], [2.0], [3.0], [4.0], [5.0], [6.0]])
    y = torch.tensor([[2.0], [4.0], [6.0], [8.0], [10.0], [12.0]])

    # train the model
    # Define the number of epochs
    # change the number of epochs to improve the model loss close to zero
    no_of_epochs: int = 250
    # Define the optimizer
    # change the learning rate to improve the model loss close to zero
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
    # Define the loss function
    criterion = nn.MSELoss()
    for epoch in range(no_of_epochs):
        # reset the gradients in the optimizer
        optimizer.zero_grad()
        # forward pass that uses random weights and biases to predict the output
        y_pred = model(X)
        # calculate the loss
        loss = criterion(y_pred, y)
        # backward pass optimizer that calculates the gradients of the loss w.r.t. the weights and biases
        loss.backward()
        # update the weights and biases in the optimizer
        optimizer.step()
        print(f"Epoch: {epoch+1}, Loss: {loss.item() :.4f}")
    # Print the model parameters
    print(f"Weight: {model.linear.weight.data}, Bias: {model.linear.bias.data}")

    # Test the model
    test = torch.tensor([7.0])
    # this will ask the model to not calculate the gradients or update the weights and biases
    with torch.no_grad():
        print(f"Prediction for tensor {test.item()} is : {model(test).item() :.4f}")





