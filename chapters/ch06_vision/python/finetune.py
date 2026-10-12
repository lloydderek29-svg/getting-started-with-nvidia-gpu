from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import torch
import torch.nn as nn
from torchvision.models import ResNet50_Weights, resnet50

from gpubook.training.config import TrainConfig
from gpubook.training.engine import evaluate, train_one_epoch
from gpubook.training.repro import seed_everything
from gpubook.vision.imagefolder import build_image_loaders


def build_model(num_classes: int) -> nn.Module:
    weights = ResNet50_Weights.IMAGENET1K_V2
    model = resnet50(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def set_backbone_trainable(model, flag: bool) -> None:
    for name, p in model.named_parameters():
        if not name.startswith("fc."):
            p.requires_grad = flag


def run_phase(name, model, loaders, epochs, groups, cfg, dev):
    train, val = loaders
    opt = torch.optim.AdamW(groups, weight_decay=1e-2)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=[g["lr"] for g in groups],
        epochs=epochs, steps_per_epoch=len(train))
    crit = nn.CrossEntropyLoss(label_smoothing=0.1)
    scaler = torch.amp.GradScaler("cuda", enabled=False)
    for ep in range(epochs):
        tr = train_one_epoch(model, train, crit, opt, sched,
                             dev, cfg, scaler)
        ev = evaluate(model, val, crit, dev, cfg)
        print(f"{name} epoch {ep}: loss {tr['loss']:.3f} "
              f"val top1 {ev['top1']:.4f} "
              f"({tr['images_per_s']:.0f} img/s)")
    return ev


@torch.no_grad()
def predict_all(model, loader, dev, cfg):
    model.eval()
    preds, labels = [], []
    for x, y in loader:
        x = x.to(dev).contiguous(
            memory_format=torch.channels_last)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            preds.append(model(x).argmax(1))
        labels.append(y)
    return torch.cat(preds).cpu(), torch.cat(labels).cpu()


def top_confusions(preds, labels, classes, k=5):
    pairs = Counter(
        (int(t), int(p)) for p, t in zip(preds, labels)
        if p != t)
    return [(classes[t], classes[p], n)
            for (t, p), n in pairs.most_common(k)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/pets")
    ap.add_argument("--epochs-head", type=int, default=3)
    ap.add_argument("--epochs-ft", type=int, default=7)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr-head", type=float, default=1e-3)
    ap.add_argument("--lr-backbone", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results/ch06/pets.pt")
    a = ap.parse_args()

    seed_everything(a.seed)
    dev = torch.device("cuda")
    train, val, classes = build_image_loaders(
        a.data, batch_size=a.batch, device="cuda", seed=a.seed)
    cfg = TrainConfig(amp="bf16", channels_last=True)

    model = build_model(len(classes)).to(dev)
    model = model.to(memory_format=torch.channels_last)

    set_backbone_trainable(model, False)
    head = [p for p in model.parameters() if p.requires_grad]
    run_phase("head", model, (train, val), a.epochs_head,
              [{"params": head, "lr": a.lr_head}], cfg, dev)

    set_backbone_trainable(model, True)
    back = [p for n, p in model.named_parameters()
            if not n.startswith("fc.")]
    ev = run_phase(
        "full", model, (train, val), a.epochs_ft,
        [{"params": back, "lr": a.lr_backbone},
         {"params": model.fc.parameters(), "lr": a.lr_head}],
        cfg, dev)

    preds, labels = predict_all(model, val, dev, cfg)
    print("\nMost confused pairs (true, predicted, count):")
    for t, p, n in top_confusions(preds, labels, classes):
        print(f"  {t:24s} {p:24s} {n}")

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(),
                "classes": classes,
                "val_top1": ev["top1"]}, a.out)
    print(f"saved {a.out}  (val top1 {ev['top1']:.4f})")


if __name__ == "__main__":
    main()
