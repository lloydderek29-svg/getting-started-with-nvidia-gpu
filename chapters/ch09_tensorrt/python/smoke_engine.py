import sys

import torch

from gpubook.trt.runtime import TrtModule

engine = TrtModule(sys.argv[1])
x = torch.rand(5, 3, 224, 224, device="cuda",
               dtype=engine.input_dtype)
y = engine(x)
print("output", tuple(y.shape), y.dtype)
print("finite", bool(torch.isfinite(y).all()))
