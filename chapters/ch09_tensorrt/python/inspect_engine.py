import json 
import sys

import onnx
import tensorrt as trt
from gpubook.trt.runtime import TrtModule


def layer_names(engine) -> list[str]:
    inspector = engine.create_engine_inspector()
    info = inspector.get_engine_information(
        trt.LayerInformationFormat.JSON)
    layers = json.loads(info)["Layers"]
    return [l["Name"] if isinstance(l, dict) else str(l)
            for l in layers]


def main(engine_path: str, onnx_path: str) -> None:
    names = layer_names(TrtModule(engine_path).engine)
    nodes = len(onnx.load(onnx_path).graph.node)
    fused = [n for n in names if "+" in n]
    print(f"ONNX nodes      : {nodes}")
    print(f"engine layers   : {len(names)}")
    print(f"fused layers    : {len(fused)}")
    print("\nexamples of fused layers:")
    for n in sorted(fused, key=len, reverse=True)[:5]:
        print("  " + (n if len(n) < 70 else n[:67] + "..."))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
    