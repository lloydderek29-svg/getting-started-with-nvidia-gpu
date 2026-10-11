import torch

if not torch.cuda.is_available():
    raise SystemExit("No CUDA device visible to PyTorch")

for i in range(torch.cuda.device_count()):
    p = torch.cuda.get_device_properties(i)
    major, minor = torch.cuda.get_device_capability(i)
    print(f"Device {i}: {p.name}")
    print(f"  Capability : {major}.{minor}")
    print(f"  SM count   : {p.multi_processor_count}")
    print(f"  Memory     : {p.total_memory / 2**30:.2f} GiB")
