from pathlib import Path

from torchvision.datasets import OxfordIIITPet

ROOT = Path("data")


def export(split: str, out_dir: Path) -> int:
    ds = OxfordIIITPet(
        str(ROOT / "raw"), split=split,
        target_types="category", download=True)
    for i, (img, label) in enumerate(ds):
        cls = ds.classes[label].replace(" ", "_")
        folder = out_dir / cls
        folder.mkdir(parents=True, exist_ok=True)
        img.convert("RGB").save(
            folder / f"{i:05d}.jpg", quality=95)
    return len(ds)


def main() -> None:
    n_train = export("trainval", ROOT / "pets" / "train")
    n_val = export("test", ROOT / "pets" / "val")
    print(f"train: {n_train} images, val: {n_val} images")


if __name__ == "__main__":
    main()
