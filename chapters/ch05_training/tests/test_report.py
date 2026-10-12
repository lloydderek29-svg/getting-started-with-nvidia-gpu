from gpubook.training.report import (
    mean_std, steady_throughput, time_to_target)

HIST = [
    {"train_seconds": 10.0, "train_images_per_s": 3000.0,
     "val_top1": 0.50},
    {"train_seconds": 5.0, "train_images_per_s": 9000.0,
     "val_top1": 0.80},
    {"train_seconds": 5.0, "train_images_per_s": 10000.0,
     "val_top1": 0.91},
]


def test_first_epoch_is_excluded_from_throughput():
    assert steady_throughput(HIST) == 9500.0


def test_time_to_target_accumulates_epochs():
    assert time_to_target(HIST, 0.9) == 20.0
    assert time_to_target(HIST, 0.99) is None


def test_mean_std_handles_single_sample():
    assert mean_std([4.0]) == (4.0, 0.0)
