import torch

## first part : Create tensor
x = torch.randn(3,3, requires_grad=True)
print("Initual Tensor:\n",x)
"""
your code
"""

## second part : calculate the gradient
"""
your code
"""
y = x + 2
z = y * y * 3

out = z.mean()
print("\nOutput:\n",out)

out.backward()
print("\nGradiants:\n",x.grad)

with torch.no_grad():
    y = x+2
    print("\nTensor with no grad:\n",y)
    #y.backward()