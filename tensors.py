import torch
import numpy as np

# Initialize tensor directly from def
data = [[1, 2], [3, 4]]
x_data = torch.tensor(data)

print(x_data)

# Initialize tensor via numpy array
np_array = np.array(data)
x_np = torch.from_numpy(np_array)

print(x_np)

# Initialize tensor from another tensor
x_ones = torch.ones_like(x_data)
print(f"Ones Tensor \n {x_ones} \n")

# Override datatype
x_rand = torch.rand_like(x_data, dtype=torch.float)
print(f"Random Tensor \n {x_rand} \n")

# Tensor with random or constant values
shape = (
    2,
    3,
)
rand_tensor = torch.rand(shape)
ones_tensor = torch.ones(shape)
zeros_tensor = torch.zeros(shape)

print(f"Random Tensor \n {rand_tensor} \n")
print(f"Ones Tensor \n {ones_tensor} \n")
print(f"Zeros Tensor \n {zeros_tensor} \n")

# Attributes of a tensor
tensor = torch.rand(3, 4)

print(f"Shape of tensor: {tensor.shape}")
print(f"Datatype of tensor: {tensor.dtype}")
print(f"Device tensor is stored on: {tensor.device}")  # cpu

print(torch.cuda.is_available())
