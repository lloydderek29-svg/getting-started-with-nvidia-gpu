import torch
import torch.nn as nn

from gpubook.bench import bench_cuda
from gpubook.training.model import resnet18_cifar


def make_graph_step(model, opt, crit, sx, sy):
    # Warm up on a side stream so lazy init happens outside
    # the captured region.
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for _ in range(3):
            opt.zero_grad(set_to_none=True)
            crit(model(sx), sy).backward()
            opt.step()
    torch.cuda.current_stream().wait_stream(side)

    graph = torch.cuda.CUDAGraph()
    opt.zero_grad(set_to_none=True)
    with torch.cuda.graph(graph):
        loss = crit(model(sx), sy)
        loss.backward()
        opt.step()
    return graph, loss


def main(batch: int = 64) -> None:
    dev = torch.device("cuda")
    torch.manual_seed(0)
    model = resnet18_cifar().to(dev)
    opt = torch.optim.SGD(
        model.parameters(), lr=0.01, momentum=0.9)
    crit = nn.CrossEntropyLoss()

    sx = torch.randn(batch, 3, 32, 32, device=dev)
    sy = torch.randint(0, 10, (batch,), device=dev)
    nx = torch.randn_like(sx)
    ny = torch.randint_like(sy, 0, 10)

    def eager():
        sx.copy_(nx)
        sy.copy_(ny)
        opt.zero_grad(set_to_none=True)
        crit(model(sx), sy).backward()
        opt.step()

    e = bench_cuda(eager, warmup=10, repeats=50)

    graph, loss = make_graph_step(model, opt, crit, sx, sy)

    def replay():
        sx.copy_(nx)
        sy.copy_(ny)
        graph.replay()

    g = bench_cuda(replay, warmup=10, repeats=50)
    print(f"eager  : {e.median_ms:7.3f} ms per step")
    print(f"graph  : {g.median_ms:7.3f} ms per step")
    print(f"speedup: {e.median_ms / g.median_ms:7.2f}x")
    print(f"loss after replay: {loss.item():.4f}")


if __name__ == "__main__":
    main()
