import numpy as np
import torch


a = np.zeros([10, 20, 30])
a = torch.from_numpy(a)

print(a.shape)

a.unsqueeze_(3)

print(a.shape)