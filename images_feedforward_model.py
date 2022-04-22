import torch
import torchvision
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
import torchvision.transforms as transform
from torchvision.utils import make_grid
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import random_split

# Calculate accuracy
def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    return torch.tensor(torch.sum(preds==labels).item() / len(preds))

# Send tensor to selected device
def to_device(data, device):
    if isinstance(data, (list, tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)

# Feedforward network with 1 hiddel layer
class MnistModel(nn.Module):
    def __init__(self, in_size, hidden_size, out_size):
        super().__init__()
        # Hidden layer
        self.linear1 = nn.Linear(in_size, hidden_size)
        # Output layer
        self.linear2 = nn.Linear(hidden_size, out_size)
    
    def forward(self, xb):
        # Flatten image
        xb = xb.view(xb.size(0), -1)
        # Output hidden layer
        out = self.linear1(xb)
        # ReLU activation function
        out = F.relu(out)
        # Predictions using output layer
        out = self.linear2(out)
        return out

    def training_step(self, batch):
        images, labels = batch
        # Generate predictions
        out = self(images)
        # Calculate loss function
        loss = F.cross_entropy(out, labels)
        return loss

    def validation_step(self, batch):
        images, labels = batch
        # Generate predictions
        out = self(images)
        # Calculate loss function
        loss = F.cross_entropy(out, labels)
        # Calculate accuracy
        acc = accuracy(out, labels)
        return {'val_loss': loss, 'val_acc': acc}

    def validation_step_epoch_end(self, outputs):
        # Combine losses
        batch_loss = [x['val_loss'] for x in outputs]
        epoch_loss = torch.stack(batch_loss).mean()
        # Combine accuracy
        batch_acc = [x['val_acc'] for x in outputs]
        epoch_acc = torch.stack(batch_acc).mean()
        return {'val_loss': epoch_loss.item(), 'val_acc': epoch_acc.item()}

    def epoch_end(self, epoch, result):
        print("Epoch [{}], val_loss: {:.4f}, val_acc: {:.4f}".format(epoch, result['val_loss'], result['val_acc']))
        # Transforms weights to image
        tr = transform.ToPILImage()
        img1 = tr(self.linear1.weight.reshape(896, 28))
        img1.save("weight1.jpg")
        img2 = tr(self.linear2.weight.reshape(10, 32))
        img2.save("weight2.jpg")

# Moves data to device (CPU or GPU)
class DeviceDataLoader():
    def __init__(self, dl, device):
        self.dl = dl
        self.device = device
    
    def __iter__(self):
        # Yields data before moving to device
        for b in self.dl:
            yield to_device(b, self.device)
        
    # Number of batches
    def __len__(self):
        return len(self.dl)


dataset = MNIST(root='data/', download=True, transform=ToTensor())

val_size = 10000
train_size = len(dataset) - val_size

# Split training and validation sets
train_ds, val_ds = random_split(dataset, [train_size, val_size])
batch_size = 128

# Load training and validation sets
train_loader = DataLoader(train_ds, batch_size, shuffle=True)
val_loader = DataLoader(val_ds, batch_size*2)

input_size = 784
hidden_size = 32
num_classes = 10
device = torch.device('cpu')

model = MnistModel(input_size, hidden_size=32, out_size=num_classes)
to_device(model, device)

model.load_state_dict(torch.load('mnist-feedforward.pth'))

# Initial pass of first batch of 128 images
for images, labels in train_loader:
    outputs = model(images)
    loss = F.cross_entropy(outputs, labels)
    print("Initial Loss: ", loss.item())
    break

for images, labels in train_loader:
    images = to_device(images, device)
    break

train_loader = DeviceDataLoader(train_loader, device)
val_loader = DeviceDataLoader(val_loader, device)

def evaluate(model, val_loader):
    outputs = [model.validation_step(batch) for batch in val_loader]
    return model.validation_step_epoch_end(outputs)

def fit(epoch, lr, model, train_loader, val_loader, opt_func=torch.optim.SGD):
    history = []
    optimizer = opt_func(model.parameters(), lr)

    for epoch in range(epoch):
        for batch in train_loader:
            loss = model.training_step(batch)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        result = evaluate(model, val_loader)
        model.epoch_end(epoch, result)
        history.append(result)
    return history

history = [evaluate(model, val_loader)]

history += fit(5, 0.05, model, train_loader, val_loader)

torch.save(model.state_dict(), 'mnist-feedforward.pth')

# Define test dataset
test_dataset = MNIST(root='data/', 
                    train=False,
                    transform=ToTensor())

def predict_image(img, model):
    xb = to_device(img.unsqueeze(0), device)
    yb = model(xb)
    _, preds  = torch.max(yb, dim=1)
    return preds[0].item()

for i in range(100):
    img, label = test_dataset[i]
    predicted_image = predict_image(img, model)
    print('Label:', label, ', Predicted:', predicted_image, 'Correct:', label==predicted_image)