import torch
print("CUDA available:", torch.cuda.is_available())
print("CUDA version:", torch.version.cuda)
print("CUDA device count:", torch.cuda.device_count())

print(torch.cuda.is_available())