import sys

import torch

from gpubook.bench import bench_cuda
from gpubook.trt.runtime import GraphedTrt, TrtModule


def main(engine_path: str) -> None:
    eng = TrtModule(engine_path)
    print(f"{'batch':>5s} {'eager ms':>9s} {'graph ms':>9s} "
          f"{'img/s':>8s} {'us/img':>8s}")
    for batch in (1, 2, 4, 8, 16, 32, 64):
        x = torch.rand(batch, 3, 224, 224, device="cuda",
                       dtype=eng.input_dtype)
        eager = bench_cuda(lambda: eng(x), warmup=20, repeats=100)
        try:
            graph = GraphedTrt(eng, x)
            g = bench_cuda(lambda: graph(x), warmup=20,
                           repeats=100)
            g_ms = f"{g.median_ms:9.3f}"
            best = g.median_ms
        except Exception as err:
            g_ms = f"{'n/a':>9s}"
            best = eager.median_ms
            print(f"  (graph capture failed: {err})")
        print(f"{batch:5d} {eager.median_ms:9.3f} {g_ms} "
              f"{batch / (best / 1e3):8.0f} "
              f"{best * 1e3 / batch:8.1f}")


if __name__ == "__main__":
    main(sys.argv[1])
