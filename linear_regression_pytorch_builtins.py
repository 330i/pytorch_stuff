import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
import torch.nn.functional as F

# Input (temp, rainfall, humidity)
inputs = np.array([[73, 67, 43], 
                   [91, 88, 64], 
                   [87, 134, 58], 
                   [102, 43, 37], 
                   [69, 96, 70], 
                   [74, 66, 43], 
                   [91, 87, 65], 
                   [88, 134, 59], 
                   [101, 44, 37], 
                   [68, 96, 71], 
                   [73, 66, 44], 
                   [92, 87, 64], 
                   [87, 135, 57], 
                   [103, 43, 36], 
                   [68, 97, 70]], 
                  dtype='float32')

# Targets (apples, oranges)
targets = np.array([[56, 70], 
                    [81, 101], 
                    [119, 133], 
                    [22, 37], 
                    [103, 119],
                    [57, 69], 
                    [80, 102], 
                    [118, 132], 
                    [21, 38], 
                    [104, 118], 
                    [57, 69], 
                    [82, 100], 
                    [118, 134], 
                    [20, 38], 
                    [102, 120]], 
                   dtype='float32')

inputs = torch.from_numpy(inputs)
targets = torch.from_numpy(targets)

print("Initial Inputs: ", inputs)
print()

train_ds = TensorDataset(inputs, targets)
# Prints inputs and targets from index 0 to 3
print(train_ds[0:3])
print()

# Splitting the data into batches size of 5 in random order
batch_size = 5
train_dl = DataLoader(train_ds, batch_size, shuffle=True)

for xb, yb in train_dl:
    print("Input Batch: ", xb)
    print("Target Batch: ", yb)
    print()

# Define model
model = nn.Linear(3,2)
print("Initial Weight: ", model.weight)
print("Initial Bias: ", model.bias)
# model.parameters() does the same thing

# Generate predictions
preds = model(inputs)

# Define loss function
loss_fn = F.mse_loss

# Calculate loss
loss = loss_fn(model(inputs), targets)
print(loss)

# Define optimizer
# Stochastic gradient descent uses one random batch to update parameter
# SGD is fast but not accurate
opt = torch.optim.SGD(model.parameters(), lr=1e-5)

# Training function
def fit(num_epochs, model, loss_fn, opt, train_dl):
    for epoch in range(num_epochs):
        for xb, yb in train_dl:
            # Calculate prediction
            pred = model(xb)
            # Calculate loss
            loss = loss_fn(pred, yb)
            # Calculate gradient
            loss.backward()
            # Adjust weight and bias
            opt.step()
            # Reset gradient to zero
            opt.zero_grad()
        # Print progress
        if (epoch+1) % 10 == 0:
            print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, num_epochs, loss.item()))

fit(100, model, loss_fn, opt, train_dl)
print("Model Predictions: ", model(inputs))
print("Target: ", targets)

# Non-training value example
print("Non-Training Value Example: [75, 63, 44] ", model(torch.tensor([[75, 63, 44.]])))