from __future__ import annotations

from pathlib import Path

import tensorrt as trt
import torch

_DT = {
    trt.DataType.FLOAT: torch.float32,
    trt.DataType.HALF: torch.float16,
    trt.DataType.INT32: torch.int32,
    trt.DataType.INT8: torch.int8,
    trt.DataType.BOOL: torch.bool,
}
for _name, _t in (("BF16", torch.bfloat16),
                  ("INT64", torch.int64)):
    if hasattr(trt.DataType, _name):
        _DT[getattr(trt.DataType, _name)] = _t


class TrtModule:
    """Run a serialized engine on CUDA tensors."""

    def __init__(self, engine_path) -> None:
        self.logger = trt.Logger(trt.Logger.WARNING)
        self.runtime = trt.Runtime(self.logger)
        data = Path(engine_path).read_bytes()
        self.engine = self.runtime.deserialize_cuda_engine(data)
        if self.engine is None:
            raise RuntimeError(
                "could not deserialize the engine; the GPU or "
                "TensorRT version probably differs from the build")
        self.context = self.engine.create_execution_context()
        self.inputs, self.outputs = [], []
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            mode = self.engine.get_tensor_mode(name)
            if mode == trt.TensorIOMode.INPUT:
                self.inputs.append(name)
            else:
                self.outputs.append(name)

    @property
    def input_dtype(self) -> torch.dtype:
        return _DT[self.engine.get_tensor_dtype(self.inputs[0])]

    def __call__(self, *tensors: torch.Tensor):
        for name, t in zip(self.inputs, tensors):
            if not t.is_contiguous():
                raise ValueError("input must be contiguous")
            want = _DT[self.engine.get_tensor_dtype(name)]
            if t.dtype != want:
                raise TypeError(f"{name}: expected {want}, got "
                                f"{t.dtype}")
            if not self.context.set_input_shape(
                    name, tuple(t.shape)):
                raise ValueError(
                    f"shape {tuple(t.shape)} is outside the "
                    f"engine's optimization profile")
            self.context.set_tensor_address(name, t.data_ptr())

        outs = []
        for name in self.outputs:
            shape = tuple(self.context.get_tensor_shape(name))
            dtype = _DT[self.engine.get_tensor_dtype(name)]
            out = torch.empty(
                shape, dtype=dtype, device=tensors[0].device)
            self.context.set_tensor_address(name, out.data_ptr())
            outs.append(out)

        stream = torch.cuda.current_stream().cuda_stream
        if not self.context.execute_async_v3(stream):
            raise RuntimeError("TensorRT execution failed")
        return outs[0] if len(outs) == 1 else tuple(outs)


class GraphedTrt:
    """Replay an engine at one fixed input shape as a CUDA graph."""

    def __init__(self, module: TrtModule,
                 example: torch.Tensor) -> None:
        self.module = module
        self.x = example.clone()
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for _ in range(3):
                module(self.x)
        torch.cuda.current_stream().wait_stream(side)
        torch.cuda.synchronize()

        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            self.y = module(self.x)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        self.x.copy_(x)
        self.graph.replay()
        return self.y
