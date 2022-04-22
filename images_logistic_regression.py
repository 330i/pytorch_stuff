from ast import Import
import torch
import torchvision
from torchvision.datasets import MNIST
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from torch.utils.data import random_split
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

# Download training dataset
dataset = MNIST(root='data/', download=True)
print("Length of training dataset: ", len(dataset))

# Download test dataset
test_dataset = MNIST(root='data/', train=False)
print("Length of test dataset: ", len(test_dataset))

# Shows image within dataset
image, label = dataset[0]
plt.imshow(image, cmap='gray')
print("Label: ", label)

dataset = MNIST(root='data/', train=True, transform=transforms.ToTensor())

# Images transformed to tensor
img_tensor, label = dataset[0]
print("Image Shape: ", img_tensor.shape, "Label: ", label)

# Image data shown from index 10 to 15
print(img_tensor[0, 10:15, 10:15])
# Data ranges from 0 to 1
print(torch.min(img_tensor), torch.max(img_tensor))

# Splits 50000 images to training dataset and 10000 images to validation dataset
train_ds, val_ds = random_split(dataset, [50000, 10000])

# Divides images into 128 image batches
batch_size = 128
train_loader = DataLoader(train_ds, batch_size, shuffle=True)
val_loader = DataLoader(val_ds, batch_size)

input_size = 28*28
num_classes = 10

# Logistic regression
model = nn.Linear(input_size, num_classes)

# Model info
print("Model Weight Shape: ", model.weight.shape)
print("Model Weight: ", model.weight)
print("Model Bias Shape: ", model.bias.shape)
print("Model Bias: ", model.bias)

# Training and validation model
class MnistModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(input_size, num_classes)

    def forward(self, xb):
        # Flattening the image
        xb = xb.reshape(-1, 784)
        out = self.linear(xb)
        return out

    def training_step(self, batch):
        images, label = batch
        # Generate prediction
        out = self(images)
        # Calculate loss
        loss = F.cross_entropy(out, label)
        return loss
    
    def validation_step(self, batch):
        images, label = batch
        # Generate predictions
        out = self(images)
        # Calculate loss
        loss = F.cross_entropy(out, label)
        # Calculate accuracy
        acc = accuracy(out, label)
        return {'val_loss':loss, 'val_acc':acc}

    def validation_epoch_end(self, outputs):
        # Combine losses
        batch_losses = [x['val_loss'] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()
        # Combine accuracies
        batch_accs = [x['val_acc'] for x in outputs]
        epoch_acc = torch.stack(batch_accs).mean()
        return {'val_loss':epoch_loss, 'val_acc':epoch_acc}

    def epoch_end(self, epoch, result):
        print("Epoch [{}], val_loss: {:.4f}, val_acc: {:.4f}".format(epoch, result['val_loss'], result['val_acc']))
        # Transforms weights to image
        tr = transforms.ToPILImage()
        img = tr(self.linear.weight.reshape(280, 28))
        img.save("weight.jpg")

model = MnistModel()

model.load_state_dict(torch.load('mnist-logistic.pth'))

for images, label in train_loader:
    outputs = model(images)
    break

# Adds softmax function to model
probs = F.softmax(outputs, dim=1)

print("Probabilities: ", probs.data)

# This should add up to 1
print("Sum: ", torch.sum(probs[0]).item())

# Calculation of accuracy for loss function
def accuracy(outputs, label):
    _, preds = torch.max(outputs, dim=1)
    return torch.tensor(torch.sum(preds==label).item()/len(preds))

# Initial prediction of number with maximum probability
max_probs, preds = torch.max(probs, dim=1)
print("Predicted Numbers: ", preds)
print("Probability: ", max_probs)
print("Sample Size: ", preds.shape)
print("Batch Size: ", batch_size)
print("Labels: ", label)
print("Accuracy: ", accuracy(outputs, label))

# Cross entropy as loss function
loss_fn = F.cross_entropy
loss = loss_fn(outputs, label)
print("Loss: ", loss)

# Evaluation function in fit function
def evaluate(model, val_loader):
    outputs = [model.validation_step(batch) for batch in val_loader]
    return model.validation_epoch_end(outputs)

# Training and validation
def fit(epoch, lr, model, train_loader, val_loader, opt_func=torch.optim.SGD):
    optimizer = opt_func(model.parameters(), lr)
    # Records epoch history
    history = []

    for epoch in range(epoch):
        # Training phase
        for batch in train_loader:
            # Generate predictions and calculate loss
            loss = model.training_step(batch)
            # Compute gradients
            loss.backward()
            # Update weights
            optimizer.step()
            # Reset gradients
            optimizer.zero_grad()
        
        # Validation phase
        # Generate predictions and calculate loss and calculate metrics
        result = evaluate(model, val_loader)
        # Calculate average validation loss & metrics
        model.epoch_end(epoch, result)
        # Log epoch, loss & metrics for inspection
        history.append(result)
    
    return history

# CPU fans will start running in this step
history0 = evaluate(model, val_loader)
history1 = fit(5, 0.001, model, train_loader, val_loader)
history2 = fit(5, 0.001, model, train_loader, val_loader)
history3 = fit(5, 0.001, model, train_loader, val_loader)
history4 = fit(5, 0.001, model, train_loader, val_loader)

history = [history0] + history1 + history2 + history3 + history4

accuracies = [result['val_acc'] for result in history]

# Saves the model to mnist-logistic.pth
torch.save(model.state_dict(), 'mnist-logistic.pth')