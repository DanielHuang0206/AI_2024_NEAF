import numpy as np
import torch ## The function of torch is the same as numpy, but it can run on GPU

## first part : convert data type

"""
your code
"""
#tensor轉np
np_data = np.arange(6).reshape((2, 3)) 
# 使用np.arange(6)生成一個包含0到5的數組，然後使用reshape((2, 3))將其重新塑形為2行3列的矩陣。
print(np_data)
# 輸出np_data，這是一個2行3列的NumPy矩陣。

torch_data = torch.from_numpy(np_data)
# 使用torch.from_numpy(np_data)將NumPy數組np_data轉換為PyTorch的tensor張量。
print(torch_data)
# 輸出torch_data，這是一個PyTorch張量，包含與np_data相同的數據。

tensor2array = torch_data.numpy()
# 使用torch_data.numpy()將PyTorch張量轉換回NumPy數組。

data = [-1,-2,1,-2]
tensor = torch.FloatTensor(data)
print(
    '\nabs',
    '\nnumpy', np.sin(data),
    '\ntorch',torch.sin(tensor)

)
print(
    '\nmean',
    '\nnumpy', np.sin(data),
    '\ntorch',torch.sin(tensor)

)

# second part : some basic usage
data = [[1,2],[3,4]]
tensor = torch.FloatTensor(data)
print(
    '\nmatrix.multiplication(matmul)',
    '\nnumpy', np.matmul(data,data),
    '\ntorch',torch.mm(tensor,tensor)

)
data = np.array(data)

print(
    '\nmatrix.multiplication(dot)',
    '\nnumpy', np.dot(data),
    '\ntorch',torch.flatten().dot(tensor.flatten())

)
