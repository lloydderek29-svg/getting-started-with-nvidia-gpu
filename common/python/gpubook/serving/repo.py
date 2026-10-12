from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from string import Template

PLATFORM_FILE = {
    "tensorrt_plan": "model.plan",
    "onnxruntime_onnx": "model.onnx",
    "pytorch_libtorch": "model.pt",
}

@dataclass(frozen=True)
class BatchSettings:
    max_batch: int = 64
    dynamic: bool = True
    preferred: tuple[int, ...] = (8, 32)
    max_delay_us: int = 2000
    instances: int = 1
    cuda_graphs: bool = False
    max_queue_size: int = 0
    timeout_us: int = 0
    warmup_batches: tuple[int, ...] = (1, 8)


PETS_TEMPLATE = Template("""\
name: "$name"
platform: "$platform"
max_batch_size: $max_batch
input [
  { name: "input" data_type: TYPE_FP32 dims: [ 3, 224, 224 ] }
]
output [
  { name: "logits" data_type: TYPE_FP32 dims: [ $classes ] }
]
instance_group [
  { count: $instances kind: KIND_GPU }
]
$batching$graphs$warmup""")

def _batching_block(s: BatchSettings) -> str:
    if not s.dynamic:
        return ""
    lines = ["dynamic_batching {"]
    if s.preferred:
        sizes = ", ".join(str(b) for b in s.preferred)
        lines.append(f"  preferred_batch_size: [ {sizes} ]")
    lines.append(
        f"  max_queue_delay_microseconds: {s.max_delay_us}")
    if s.max_queue_size or s.timeout_us:
        lines.append("  default_queue_policy {")
        if s.timeout_us:
            lines.append("    timeout_action: REJECT")
            lines.append("    default_timeout_microseconds: "
                         f"{s.timeout_us}")
        if s.max_queue_size:
            lines.append(
                f"    max_queue_size: {s.max_queue_size}")
        lines.append("  }")
    lines.append("}")
    return "\n".join(lines) + "\n"


def _warmup_block(batches) -> str:
    if not batches:
        return ""
    items = []
    for b in batches:
        items.append(
            "  {\n"
            f'    name: "warmup_b{b}"\n'
            f"    batch_size: {b}\n"
            "    inputs {\n"
            '      key: "input"\n'
            "      value {\n"
            "        data_type: TYPE_FP32\n"
            "        dims: [ 3, 224, 224 ]\n"
            "        random_data: true\n"
            "      }\n"
            "    }\n"
            "  }")
    return "model_warmup [\n" + ",\n".join(items) + "\n]\n"


def render_pets_config(name: str, platform: str,
                       s: BatchSettings, classes: int) -> str:
    if any(b > s.max_batch for b in s.preferred):
        raise ValueError("preferred batch exceeds max_batch")
    graphs = ""
    if s.cuda_graphs and platform == "tensorrt_plan":
        graphs = "optimization {\n  cuda { graphs: true }\n}\n"
    return PETS_TEMPLATE.substitute(
        name=name, platform=platform, max_batch=s.max_batch,
        classes=classes, instances=s.instances,
        batching=_batching_block(s), graphs=graphs,
        warmup=_warmup_block(s.warmup_batches))


def render_template(path: Path, **values) -> str:
    return Template(Path(path).read_text()).substitute(**values)

def _kind(text: str):
    m = re.search(r'^\s*platform:\s*"([^"]+)"', text, re.M)
    if m:
        return m.group(1)
    m = re.search(r'^\s*backend:\s*"([^"]+)"', text, re.M)
    return f"backend:{m.group(1)}" if m else None


def check_repository(root) -> list[str]:
    root = Path(root)
    models = sorted(p for p in root.iterdir() if p.is_dir())
    if not models:
        return [f"no models in {root}"]
    names = {p.name for p in models}
    problems = []
    for m in models:
        cfg = m / "config.pbtxt"
        if not cfg.exists():
            problems.append(f"{m.name}: missing config.pbtxt")
            continue
        text = cfg.read_text()
        declared = re.search(
            r'^\s*name:\s*"([^"]+)"', text, re.M)
        if declared and declared.group(1) != m.name:
            problems.append(
                f"{m.name}: config names {declared.group(1)!r}")
        versions = [p for p in m.iterdir()
                    if p.is_dir() and p.name.isdigit()]
        if not versions:
            problems.append(f"{m.name}: no version directory")
            continue
        kind = _kind(text)
        if kind == "ensemble":
            for ref in re.findall(
                    r'model_name:\s*"([^"]+)"', text):
                if ref not in names:
                    problems.append(
                        f"{m.name}: unknown step model {ref!r}")
        elif kind in PLATFORM_FILE:
            for v in versions:
                if not (v / PLATFORM_FILE[kind]).exists():
                    problems.append(f"{m.name}/{v.name}: "
                                    f"missing {PLATFORM_FILE[kind]}")
        elif kind == "backend:python":
            for v in versions:
                if not (v / "model.py").exists():
                    problems.append(
                        f"{m.name}/{v.name}: missing model.py")
        else:
            problems.append(
                f"{m.name}: unrecognized platform or backend")
    return problems
