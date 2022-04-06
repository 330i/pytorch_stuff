from tkinter import Y
import torch

x = torch.tensor(3.)
w = torch.tensor(4., requires_grad=True)
b = torch.tensor(5., requires_grad=True)
print(w,b)
y = w*x+b
y.backward()
print('dy/dx:', x.grad)
print('dy/dw:', w.grad)
print('dy/db:', b.grad)