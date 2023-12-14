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

# A GPU is available
print(torch.cuda.is_available())

if torch.cuda.is_available():
    tensor = tensor.to("cuda")
    print(f"Device tensor is stored on: {tensor.device}")

# Indexing and slicing
tensor = torch.ones(4, 4)
tensor[:, 1] = 0
print(tensor)

# Joining tensors
t1 = torch.cat([tensor, tensor, tensor], dim=1)
print(t1)

# dim=1 is rowwise, dim=2 is columnwise concatenation
t1 = torch.cat([tensor, tensor, tensor], dim=0)
print(t1)

# Elementwise multiplication (Hadamard product)
print(f"tensor.mul(tensor) \n {tensor.mul(tensor)} \n")
print(f"tensor*tensor \n {tensor*tensor} \n")

# Matrix multiplication
print(f"tensor.matmul(tensor.T) \n {tensor.matmul(tensor.T)} \n")
print(f"tensor@tensor.T \n {tensor @ tensor.T} \n")

# In-place operations (save on memory but loses history)
print(tensor, "\n")
tensor.add_(5)
print(tensor, "\n")

# Numpy interactions: changing a tensor changes the array as well
t = torch.ones(5)
print(f"t: {t}")
n = t.numpy()
print(f"n: {n}")

t.add_(1)
print(f"t: {t}")
print(f"n: {n}")

# Numpy to Tensor
n = np.ones(5)
t = torch.from_numpy(n)

np.add(n, 1, out=n)
print(f"t: {t}")
print(f"n: {n}")
