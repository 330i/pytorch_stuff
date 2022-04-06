import torch
import numpy as np

x = np.array([[1.0,2],[3,4]])
print(x)
y = torch.from_numpy(x)
print(y)