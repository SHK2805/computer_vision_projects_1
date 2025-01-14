import torch
import torch.nn as nn

class SequentialModel(nn.Module):
    def __init__(self):
        # Call the parent class constructor
        super(SequentialModel, self).__init__()
        self.sequential = nn.Sequential(
            # input 20 features, output 64 features
            nn.Linear(20, 64),
            nn.ReLU(),
            # input 64 features, output 32 features
            nn.Linear(64, 32),
            nn.ReLU(),
            # input 32 features, output 16 feature
            nn.Linear(32, 16),
            nn.ReLU(),
            # input 16 features, output 5 features
            nn.Linear(16, 5)
        )
    # Define the model
    def forward(self, x):
        # Define a sequential model
        return self.sequential(x)

if __name__ == "__main__":
    model = SequentialModel()
    # Define the input tensor
    # create a random tensor of size 10x20
    # 10 is the number of samples and 20 is the number of features
    # 10 rows and 20 columns i.e. each column is a feature
    # get the X values between 1 and 10 integers
    # torch expects the input tensor to be of type float32
    X = torch.randint(1, 10, (10, 20))
    # convert the tensor to float32
    X = X.to(torch.float32)
    # print(f"X: {X}")
    # y is the output tensor of size 10x5
    # torch expects the output tensor to be of type float32
    y = torch.randint(1, 10, (10, 5))
    # convert the tensor to float32
    y = y.to(torch.float32)
    # print(f"y: {y}")

    # train the model
    # Define the number of epochs
    # change the number of epochs to improve the model loss close to zero
    no_of_epochs: int = 250
    # Define the optimizer
    # change the learning rate to improve the model loss close to zero
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
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
        print(f"Epoch: {epoch + 1}, Loss: {loss.item() :.4f}")
    # Print the model parameters
    print(f"Model: {model}")
    print(f"Model parameters: {model.parameters}")
    print(f"Model Weight: {model.sequential[0].weight.data}")
    print(f"Model Bias: {model.sequential[0].bias.data}")

    # Test the model
    test = torch.randn(1, 20)
    # this will ask the model to not calculate the gradients or update the weights and biases
    with torch.no_grad():
        print(f"Prediction for tensor {test} is : {model(test)}")


