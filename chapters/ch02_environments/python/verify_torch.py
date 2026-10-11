import time

import torch


def main() -> None:
    print("torch          :", torch.__version__)
    print("built for CUDA :", torch.version.cuda)
    print("cuDNN          :", torch.backends.cudnn.version())
    print("CUDA available :", torch.cuda.is_available())
    if not torch.cuda.is_available():
        raise SystemExit(1)

    dev = torch.device("cuda:0")
    name = torch.cuda.get_device_name(dev)
    cap = torch.cuda.get_device_capability(dev)
    print("device         :", name)
    print("capability     :", cap)
    print("arch list      :", torch.cuda.get_arch_list())
    print("bf16 supported :", torch.cuda.is_bf16_supported())

    a = torch.randn(4096, 4096, device=dev)
    b = torch.randn(4096, 4096, device=dev)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(10):
        c = a @ b
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    flops = 10 * 2 * 4096 ** 3
    print(f"matmul         : {flops / elapsed / 1e12:.1f} TFLOPS")
    print("result finite  :", bool(torch.isfinite(c).all()))


if __name__ == "__main__":
    main()
