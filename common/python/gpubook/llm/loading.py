from __future__ import annotations

import importlib.util
import json
import struct
from pathlib import Path

import torch
import transformers
from packaging.version import Version
from transformers import AutoModelForCausalLM, AutoTokenizer

GIB = 2**30

BYTES_PER_DTYPE = {
    "F64": 8, "F32": 4, "F16": 2, "BF16": 2,
    "I64": 8, "I32": 4, "I16": 2, "I8": 1,
    "U8": 1, "BOOL": 1, "F8_E4M3": 1, "F8_E5M2": 1,
}


def read_safetensors_header(path) -> dict:
    with open(path, "rb") as f:
        (n,) = struct.unpack("<Q", f.read(8))
        header = json.loads(f.read(n))
    header.pop("__metadata__", None)
    return header


def inventory(model_dir) -> dict:
    """Count parameters and bytes across all shards."""
    params, nbytes, dtypes = 0, 0, {}
    for shard in sorted(Path(model_dir).glob("*.safetensors")):
        for info in read_safetensors_header(shard).values():
            n = 1
            for dim in info["shape"]:
                n *= dim
            params += n
            nbytes += n * BYTES_PER_DTYPE[info["dtype"]]
            key = info["dtype"]
            dtypes[key] = dtypes.get(key, 0) + n
    return {"params": params, "bytes": nbytes, "dtypes": dtypes}

def fits(weights_gib, free_gib, headroom=0.9, reserve_gib=1.5):
    return weights_gib + reserve_gib <= free_gib * headroom


def has_module(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def compat_report() -> dict:
    props = torch.cuda.get_device_properties(0)
    free, total = torch.cuda.mem_get_info()
    cc = (props.major, props.minor)
    return {
        "gpu": props.name,
        "compute_capability": f"{cc[0]}.{cc[1]}",
        "bf16": torch.cuda.is_bf16_supported(),
        "flash_attn_2": cc >= (8, 0) and has_module("flash_attn"),
        "bitsandbytes": has_module("bitsandbytes"),
        "free_gib": round(free / GIB, 2),
        "total_gib": round(total / GIB, 2),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
    }


def fetch_model(name: str) -> str:
    """Return a local directory, downloading if needed."""
    if Path(name).is_dir():
        return name
    from huggingface_hub import snapshot_download

    patterns = ["*.json", "*.safetensors", "*.model",
                "*.txt", "tokenizer*"]
    return snapshot_download(name, allow_patterns=patterns)

def _dtype_kwarg(dtype) -> dict:
    new_api = Version(transformers.__version__) >= Version("4.56.0")
    return {("dtype" if new_api else "torch_dtype"): dtype}


def load_model(name, dtype=torch.bfloat16, quant=None,
               attn="sdpa", device_map="cuda"):
    path = fetch_model(name)
    kwargs = _dtype_kwarg(dtype)
    if quant is not None:
        kwargs["quantization_config"] = quant
    model = AutoModelForCausalLM.from_pretrained(
        path, attn_implementation=attn,
        device_map=device_map, **kwargs)
    tok = AutoTokenizer.from_pretrained(path)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    model.eval()

    if quant is None:
        got = next(model.parameters()).dtype
        if got != dtype:
            raise RuntimeError(
                f"requested {dtype} but loaded {got}")
    return model, tok
