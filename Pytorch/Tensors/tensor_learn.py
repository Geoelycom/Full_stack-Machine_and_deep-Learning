import torch
import numpy as np

    # Innitailise a tensor
    # tensors can be initialised directly from data:
    
data = [[1, 2], [3, 4]]
x_data =  torch.tensor(data)
    
    
    # Tensors can also be created from NumPy arrays
np_array = np.array(data)
x_np = torch.from_numpy(np_array)


    # Or from another tensor
    # The new tensor retains the properties (shape, datatype) of the argument tensor, unless explicitly overridden.

x_ones = torch.ones_like(x_data)  # retains the properties of x_data  
print(f"Ones Tensor: \n {x_ones} \n")

x_rand = torch.rand_like(x_data, dtype=torch.float)  # overrides the datatype of x_data

print(f"random Tensor: \n {x_rand} \n")


# with random or constant values
# shape is a tuple of tensor dimensions. in the below functions, it determines the dimensionality of the output tensor.

shape = (2,3,)
rand_tensor = torch.rand(shape)
ones_tensor = torch.ones(shape)
zeros_tensor = torch.zeros(shape)

print(f"Random Tensor: \n {rand_tensor} \n")
print(f"Ones Tensor: \n {ones_tensor} \n")
print(f"Zeros Tensor: \n {zeros_tensor}")


# Attributes of a Tensor

# Tensor attributes describe their shape, datatype, and the device on which they are stored.

tensor = torch.rand(3,4)

print(f"Shape of tensor: {tensor.shape}")
print(f"Datatype of tensor: {tensor.dtype}")
print(f"Device tensor is stored on: {tensor.device}")


# We move our tensor to the current accelerator if available
# if torch.accelerator.is_available():
#     tensor = tensor.to(torch.accelerator.current_accelerator())
    
    
# Operations on Tensors
t1 = torch.cat([tensor, tensor, tensor], dim=1)
print(t1)