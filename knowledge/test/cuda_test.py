import torch

print(f"{torch.cuda.is_available()}")
print(f"{torch.cuda.get_device_name()}")
print("PyTorch版本:", torch.__version__)
print("CUDA是否可用:", torch.cuda.is_available())
print("CUDA版本:", torch.version.cuda)
print("GPU名称:", torch.cuda.get_device_name(0))