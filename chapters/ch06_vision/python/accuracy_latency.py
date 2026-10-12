from __future__ import annotations

import argparse

import cv2
import torch
from ultralytics import YOLO

from gpubook.bench import bench_cpu
from gpubook.vision.letterbox import letterbox_gpu


def pareto_front(points):
    """points: (name, latency_ms, map). Lower latency and
    higher mAP are better. Returns names not dominated."""
    keep = []
    for name, lat, acc in points:
        dominated = any(
            (l2 <= lat and a2 >= acc) and (l2 < lat or a2 > acc)
            for n2, l2, a2 in points if n2 != name)
        if not dominated:
            keep.append(name)
    return keep


def frame_latency(model, frame, size, half) -> float:
    dev = torch.device("cuda")

    def run():
        rgb = torch.from_numpy(frame).to(dev).flip(-1)
        x, _, _ = letterbox_gpu(rgb, size)
        model.predict(x, half=half, verbose=False)

    return bench_cpu(run, warmup=10, repeats=50,
                     sync_cuda=True).median_ms


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--video", required=True)
    a = ap.parse_args()

    cap = cv2.VideoCapture(a.video)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise SystemExit("could not read a frame from video")

    model = YOLO(a.model)
    points = []
    for size in (320, 480, 640):
        for half in (False, True):
            m = model.val(data=a.data, imgsz=size, half=half,
                          verbose=False)
            lat = frame_latency(model, frame, size, half)
            name = f"{size}px {'fp16' if half else 'fp32'}"
            points.append((name, lat, float(m.box.map)))

    best = set(pareto_front(points))
    print(f"{'config':12s} {'ms/frame':>9s} "
          f"{'mAP50-95':>9s}  pareto")
    for name, lat, acc in points:
        mark = "yes" if name in best else ""
        print(f"{name:12s} {lat:9.2f} {acc:9.3f}  {mark}")


if __name__ == "__main__":
    main()
