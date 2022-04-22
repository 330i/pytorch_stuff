import torch
import torchvision
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
from torchvision.utils import make_grid
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import random_split

dataset = MNIST(root='data/', download=True, transform=ToTensor())

val_size = 10000
train_size = len(dataset) - val_size

# Split training and validation sets
train_ds, val_ds = random_split(dataset, [train_size, val_size])
batch_size = 128

# Load training and validation sets
train_loader = DataLoader(train_ds, batch_size, shuffle=True)
val_loader = DataLoader(val_ds, batch_size*2)

for images, labels in train_loader:
    # Flatten inputs
    inputs = images.reshape(-1, 784)
    print("Input shape: ", inputs.shape)
    break

# Set hidden layer as linear with 32 outputs
input_size = inputs.shape[-1]
hidden_size = 32
layer1 = nn.Linear(input_size, hidden_size)
layer1_outputs = layer1(inputs)

# Matrix multiplication with input and hidden layer
layer1_outputs_direct = inputs @ layer1.weight.t() + layer1.bias
print("Layer 1 Output Shape: ", layer1_outputs_direct.shape)

# Verify that layer1_outputs and layer1_outputs_direct are similar
print("layer1_outputs_direct is the composition of layer1_outputs verified by: ", torch.allclose(layer1_outputs, layer1_outputs_direct, 1e-3))

# Activation function: some values turned off while some values turned on
# Linear function remains for positive values but the overall function is not linear anymore
# ReLU zeros or turns off negative values
relu_outputs = F.relu(layer1_outputs)
print("layer1_outputs minimum: ", torch.min(layer1_outputs))
print("layer1_outputs after going through ReLU: ", torch.min(relu_outputs))

# Set output layer as linear with 10 outputs (since there are 10 digits to determine)
output_size = 10
layer2 = nn.Linear(hidden_size, output_size)
layer2_outputs = layer2(relu_outputs)

# Calculate loss
print("Loss: ", F.cross_entropy(layer2_outputs, labels))

# Expanded version of layer2(F.relu(layer1(inputs))) or what we did above
outputs = (F.relu(inputs @ layer1.weight.t() + layer1.bias)) @ layer2.weight.t() + layer2.bias

# Verify outputs and layer2_outputs are similar
print("outputs is the composition of layer2_outputs verified by: ", torch.allclose(outputs, layer2_outputs, 1e-3))

