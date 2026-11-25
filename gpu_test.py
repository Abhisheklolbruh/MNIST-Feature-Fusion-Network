import torch

print("PyTorch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("CUDA runtime:", torch.version.cuda)

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))

    x = torch.rand(5000, 5000, device="cuda")
    y = torch.rand(5000, 5000, device="cuda")

    z = x @ y

    print("Tensor device:", z.device)
    print("GPU test successful!")
else:
    print("CUDA NOT DETECTED")