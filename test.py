import torch

print(torch.cuda.is_available())
print(torch.tensor(1).cuda() + torch.tensor(2).cuda())