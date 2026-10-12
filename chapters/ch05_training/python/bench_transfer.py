import torch

from gpubook.bench import bench_cuda


def main() -> None:
    dev = torch.device("cuda")
    shape = (256, 3, 224, 224)
    f32 = torch.rand(shape)
    u8 = (f32 * 255).to(torch.uint8)
    f32_pin = f32.pin_memory()
    u8_pin = u8.pin_memory()
    mean = torch.tensor(
        [0.485, 0.456, 0.406], device=dev).view(1, 3, 1, 1) * 255
    std = torch.tensor(
        [0.229, 0.224, 0.225], device=dev).view(1, 3, 1, 1) * 255

    cases = {
        "float32 pageable": lambda: f32.to(dev),
        "float32 pinned": lambda: f32_pin.to(
            dev, non_blocking=True),
        "uint8 pinned + normalize": lambda: (
            u8_pin.to(dev, non_blocking=True).float() - mean
        ) / std,
    }
    for name, fn in cases.items():
        s = bench_cuda(fn)
        print(f"{name:26s} {s.median_ms:8.2f} ms")


if __name__ == "__main__":
    main()
