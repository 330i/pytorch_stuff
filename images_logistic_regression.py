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

class MnistModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(input_size, num_classes)
    # Flattens dataset
    def forward(self, xb):
        xb = xb.reshape(-1, 28*28)
        out = self.linear(xb)
        return out

model = MnistModel()
print(model.linear)

for images, label in train_loader:
    print(images.shape)
    outputs = model(images)
    break

probs = F.softmax(outputs, dim=1)