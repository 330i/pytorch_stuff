from cgi import print_arguments
import torch
import numpy as np

def model(x):
    return x @ w.t() + b

def mse(t1,t2):
    diff = t1-t2
    return torch.sum(diff*diff)/diff.numel()

# Input (temp, rainfall, humidity)
inputs = np.array([[73, 67, 43], 
                    [91, 88, 64], 
                    [87, 134, 58], 
                    [102, 43, 37], 
                    [69, 96, 70]], dtype='float32')
# Targets (apples, oranges)
targets = np.array([[56, 70], 
                    [81, 101], 
                    [119, 133], 
                    [22, 37], 
                    [103, 119]], dtype='float32')
inputs = torch.from_numpy(inputs)
targets = torch.from_numpy(targets)

# Assigns weight and bias randomly
w = torch.randn(2,3, requires_grad=True)
b = torch.randn(2, requires_grad=True)
print("Weight: ",w)
print("Bias: ", b)
# ML model predicts
preds = model(inputs)
print("Predictions: ", preds)
print("Targets: ", targets)
# Calculates loss equation from prediction and target
loss = mse(preds, targets)
print("Loss: ", loss)
# Calculates gradient
loss.backward()
print("Weight: ", w)
print("Weight Autograd: ", w.grad)

# Assigns learning rate (1e-5) so that weight and bias don't change too much and modifies weight and bias
# torch.no_grad() indicates to not modify or track gradients
with torch.no_grad():
    w -= w.grad * 1e-5
    b -= b.grad * 1e-5

# Observe if the predictions are better
loss = mse(preds, targets)
print(loss)

# Resets grads to prevent grads stacking each other
w.grad.zero_()
b.grad.zero_()

# Training

# Generate predictions
preds = model(inputs)
print("Predictions: ", preds)

# Calculate loss
loss = mse(preds, targets)
print("Loss: ", loss)

# Calculate gradient
loss.backward()
print("Weight Gradient: ", w.grad)
print("Bias Gradient: ", b.grad)

# Adjust weight and reset gradients
with torch.no_grad():
    w -= w.grad * 1e-5
    b -= b.grad * 1e-5
    w.grad.zero_()
    b.grad.zero_()

# New weight and bias
print("New Weight: ", w)
print("New Bias: ", b)

# Calculate loss
preds = model(inputs)
loss = mse(preds, targets)
print("Loss: ", loss)

# Train for 100 epochs
# Repeats the train process 100 times
for i in range(100):
    preds = model(inputs)
    loss = mse(preds, targets)
    loss.backward()
    with torch.no_grad():
        w -= w.grad * 1e-5
        b -= b.grad * 1e-5
        w.grad.zero_()
        b.grad.zero_()

# Calculate Loss
preds = model(inputs)
loss = mse(preds, targets)
print(loss)

# Compare predictions and targets
print("Prediction: ", preds)
print("Target: ", targets)