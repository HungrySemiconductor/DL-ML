import torch

a = torch.tensor([1])
print(id(a),a)

a = a + torch.tensor([1])
print(id(a),a)

a += torch.tensor([1])
print(id(a),a)