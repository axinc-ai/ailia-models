import numpy as np
import torch

# numpy
a = np.random.rand(10, 20)
tmp0 = np.split(a, indices_or_sections=5, axis=0) # split into 5 sections
for t in tmp0:
    print(t.shape)

np.split(a, indices_or_sections=7, axis=0) # error, since no equal division

tmp1 = np.split(a, [5, 7], 0) # use indices ([:5], [5:7], [7:])
for t in tmp1:
    print(t.shape)

# PyTorch
x = torch.randn(10, 20)

tmp2 = torch.split(x, split_size_or_sections=4, dim=0) # use size 4
for t in tmp2:
    print(t.shape) # last split might be smaller

tmp3 = torch.split(x, split_size_or_sections=[5, 2, 3], dim=0)
for t in tmp3:
    print(t.shape)
    
torch.split(x, split_size_or_sections=[5, 4], dim=0) # error, since 5+4 != dim(0)
# Should it return Tensors of size [5, 20], [4, 20] and [1, 20]?
