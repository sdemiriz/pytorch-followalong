import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# ToTensor imports image into a torch Tensor
# Normalize will normalize each channel RGB to values between -1,1
transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    ]
)

# Work with 4 images at a time
batch_size = 4

# Download training images
trainset = torchvision.datasets.CIFAR10(
    root="./data", train=True, download=True, transform=transform
)

# Load training images
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=batch_size, shuffle=True, num_workers=2
)

# Download testing dataset
testset = torchvision.datasets.CIFAR10(
    root="./data", train=False, download=True, transform=transform
)

# Load testing dataset
testloader = torch.utils.data.DataLoader(
    testset, batch_size=batch_size, shuffle=True, num_workers=2
)

# Set up classifier classes
classes = (
    "plane",
    "car",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
)


# Define the CNN
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


net = Net()

# Define Loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

# Train network
for epoch in range(2):
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # Get inputs and labels from data
        inputs, labels = data

        # zero the parameter gradients
        optimizer.zero_grad()

        # SGD, backpropagate, optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # Print stats
        running_loss += loss.item()
        if i % 2000 == 1999:
            print(f"[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}")
            running_loss = 0.0

print("Finished Training")

PATH = "./cifar_net.pth"
torch.save(net.state_dict(), PATH)
net.load_state_dict(torch.load(PATH))

dataiter = iter(testloader)
images, labels = next(dataiter)

outputs = net(images)
_, predicted = torch.max(outputs, 1)
print("Predicted: ", " ".join(f"{classes[predicted[j]]:5s}" for j in range(4)))

correct = 0
total = 0

# No need to calculate gradients, this is not training
with torch.no_grad():
    for data in testloader:
        images, labels = data
        # Run images through network
        outputs = net(images)
        # Choose the highest energy result as prediction
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f"Accuracy of network on the 10000 test images: {100 * correct // total} %")

# Counts for predictions
correct_pred = {classname: 0 for classname in classes}
total_pred = {classname: 0 for classname in classes}

# Not training
with torch.no_grad():
    for data in testloader:
        images, labels = data
        output = net(images)
        _, predictions = torch.max(output, 1)

        # Get correct predictions for each class
        for label, prediction in zip(labels, predictions):
            if label == prediction:
                correct_pred[classes[label]] += 1
            total_pred[classes[label]] += 1

# Print accuracy per class
for classname, correct_count in correct_pred.items():
    accuracy = 100 * float(correct_count) / total_pred[classname]
    print(f"Accuracy for class: {classname:5s} is {accuracy:.1f} %")
