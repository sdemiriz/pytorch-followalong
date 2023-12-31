import torch
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self):
        """Initialize network, inherit from nn.Module"""
        super(Net, self).__init__()

        # One in channel (image)
        # Six out channels (feature maps)
        # Convolution kernel of size 5x5
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5)

        self.fc1 = nn.Linear(in_features=16 * 5 * 5, out_features=120)
        self.fc2 = nn.Linear(in_features=120, out_features=84)
        self.fc3 = nn.Linear(in_features=84, out_features=10)

    def forward(self, x):
        """Initialize learning hyperparameters"""
        # Max pooling over 2x2 window
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)

        # Flatten all dimensions except batch dimension
        x = torch.flatten(x, 1)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x


net = Net()
print(net)

# Parameters of the network
params = list(net.parameters())

# List all parameters
for p in params:
    print(p.size())
print("\n")

# Trying a random 32x32 input
input = torch.randn(1, 1, 32, 32)
out = net(input)
print(input.size(), "\n", out, "\n")

# Clear gradients of all parameters and backpropagate with random gradients
net.zero_grad()
out.backward(torch.randn(1, 10))

# Compute loss
output = net(input)
print(f"output: {output}")
target = torch.randn(10)  # dummy target
target = target.view(1, -1)  # make target same shape as output
print(f"target: {target}")
criterion = nn.MSELoss()
print(f"criterion: {criterion}")

loss = criterion(output, target)
print(f"loss: {loss}")

import torch.optim as optim

# create optimizer
optimizer = optim.SGD(net.parameters(), lr=0.01)

optimizer.zero_grad()
output = net(input)
loss = criterion(output, target)
loss.backward()
optimizer.step()
