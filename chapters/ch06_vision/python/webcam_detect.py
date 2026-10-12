from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2
import torch
from ultralytics import YOLO

from gpubook.env import snapshot
from gpubook.vision.latency import LatencyTracker
from gpubook.vision.letterbox import (
    boxes_to_frame, letterbox_gpu)
from gpubook.vision.video import FrameSource


def detect(model, frame, args, dev):
    """Return an Nx6 CPU array: x1, y1, x2, y2, conf, class."""
    if not args.gpu_preprocess:
        res = model.predict(
            frame, imgsz=args.imgsz, half=args.half,
            conf=args.conf, verbose=False)[0]
        return res.boxes.data.cpu().numpy()

    h, w = frame.shape[:2]
    rgb = torch.from_numpy(frame).to(dev).flip(-1)
    x, r, pad = letterbox_gpu(rgb, args.imgsz)
    res = model.predict(
        x, half=args.half, conf=args.conf, verbose=False)[0]
    data = res.boxes.data.clone()
    data[:, :4] = boxes_to_frame(data[:, :4], r, pad, w, h)
    return data.cpu().numpy()


def draw(frame, dets, names, fps_text):
    for x1, y1, x2, y2, conf, cls in dets:
        p1, p2 = (int(x1), int(y1)), (int(x2), int(y2))
        cv2.rectangle(frame, p1, p2, (0, 220, 0), 2)
        label = f"{names[int(cls)]} {conf:.2f}"
        cv2.putText(frame, label, (p1[0], max(p1[1] - 6, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 220, 0), 1, cv2.LINE_AA)
    cv2.putText(frame, fps_text, (10, 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                (255, 255, 255), 2, cv2.LINE_AA)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="yolo11s.pt")
    ap.add_argument("--source", default="0")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--half", action="store_true")
    ap.add_argument("--gpu-preprocess", action="store_true")
    ap.add_argument("--no-display", action="store_true")
    ap.add_argument("--max-frames", type=int, default=0)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--save-json", default=None)
    args = ap.parse_args()

    source = int(args.source) if args.source.isdigit() \
        else args.source
    is_camera = isinstance(source, int)
    dev = torch.device("cuda")
    model = YOLO(args.model)
    src = FrameSource(source, drop_stale=is_camera)
    tracker = LatencyTracker()

    started, done = time.perf_counter(), 0
    try:
        while True:
            item = src.read()
            if item is None:
                break
            frame, stamp, _ = item
            t0 = time.perf_counter()
            tracker.add("queue_wait", (t0 - stamp) * 1e3)

            dets = detect(model, frame, args, dev)
            t1 = time.perf_counter()
            tracker.add("detect", (t1 - t0) * 1e3)

            elapsed = max(t1 - started, 1e-6)
            fps = done / elapsed
            draw(frame, dets, model.names, f"{fps:5.1f} FPS")
            t2 = time.perf_counter()
            tracker.add("draw", (t2 - t1) * 1e3)
            tracker.add("total", (t2 - stamp) * 1e3)

            if not args.no_display:
                cv2.imshow("detections", frame)
                if cv2.waitKey(1) & 0xFF in (27, ord("q")):
                    break
            done += 1
            if args.max_frames and done >= args.max_frames:
                break
    finally:
        src.close()
        cv2.destroyAllWindows()

    wall = time.perf_counter() - started
    report = {
        "frames": done,
        "dropped_by_reader": src.dropped,
        "fps": done / wall,
        "latency_ms": tracker.summary(skip=args.warmup),
        "settings": vars(args),
        "env": snapshot(),
    }
    print(f"\n{done} frames, {report['fps']:.1f} FPS, "
          f"{src.dropped} stale frames dropped")
    print(f"{'stage':11s} {'p50':>7s} {'p95':>7s} "
          f"{'p99':>7s} {'max':>7s}")
    for stage, s in report["latency_ms"].items():
        print(f"{stage:11s} {s['p50']:7.1f} {s['p95']:7.1f} "
              f"{s['p99']:7.1f} {s['max']:7.1f}")
    if args.save_json:
        Path(args.save_json).parent.mkdir(
            parents=True, exist_ok=True)
        Path(args.save_json).write_text(
            json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
