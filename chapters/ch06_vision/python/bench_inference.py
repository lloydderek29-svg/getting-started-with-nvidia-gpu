from __future__ import annotations

import argparse

import cv2
import numpy as np
import torch
from ultralytics import YOLO

from gpubook.bench import bench_cpu
from gpubook.vision.letterbox import letterbox_gpu


def load_frames(video, n):
    if video:
        cap = cv2.VideoCapture(video)
        frames = []
        while len(frames) < n:
            ok, f = cap.read()
            if not ok:
                break
            frames.append(f)
        cap.release()
        if frames:
            return frames
    rng = np.random.default_rng(0)
    return [rng.integers(0, 255, (720, 1280, 3), np.uint8)
            for _ in range(n)]


def make_runner(model, frames, mode, half, size):
    dev = torch.device("cuda")
    if mode == "cpu":
        return lambda: model.predict(
            frames, imgsz=size, half=half, verbose=False)

    def run():
        tensors = []
        for f in frames:
            rgb = torch.from_numpy(f).to(dev).flip(-1)
            tensors.append(letterbox_gpu(rgb, size)[0])
        batch = torch.cat(tensors)
        return model.predict(batch, half=half, verbose=False)

    return run


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="yolo11s.pt")
    ap.add_argument("--video", default=None)
    ap.add_argument("--imgsz", type=int, default=640)
    a = ap.parse_args()

    model = YOLO(a.model)
    pool = load_frames(a.video, 32)
    print(f"{'pre':4s} {'half':5s} {'batch':>5s} "
          f"{'ms/call':>8s} {'ms/img':>7s} {'img/s':>7s}")
    for mode in ("cpu", "gpu"):
        for half in (False, True):
            for batch in (1, 4, 8, 16, 32):
                frames = (pool * 32)[:batch]
                run = make_runner(
                    model, frames, mode, half, a.imgsz)
                s = bench_cpu(run, warmup=5, repeats=20,
                              sync_cuda=True)
                per_img = s.median_ms / batch
                print(f"{mode:4s} {str(half):5s} {batch:5d} "
                      f"{s.median_ms:8.1f} {per_img:7.2f} "
                      f"{1000 / per_img:7.0f}")


if __name__ == "__main__":
    main()
