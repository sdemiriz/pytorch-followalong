import torch
from torchvision.models import resnet18, ResNet18_Weights

# Initialize model, data and labels
model = resnet18(weights=ResNet18_Weights.DEFAULT)
data = torch.rand(1, 3, 64, 64)
labels = torch.rand(1, 1000)

# Run forward propagation
prediction = model(data)

# Calculate loss
loss = (prediction - labels).sum()

# Backpropagate loss
loss.backward()

# Load an optimizer
optim = torch.optim.SGD(
    model.parameters(), lr=1e-2, momentum=0.9
)  # stochastic gradient descent

# Gradient descent
optim.step()
