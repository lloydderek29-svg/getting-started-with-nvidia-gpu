import pytest

from gpubook.serving.repo import (
    BatchSettings, check_repository, render_pets_config)


def make(root, name, kind, files):
    d = root / name / "1"
    d.mkdir(parents=True)
    (root / name / "config.pbtxt").write_text(
        f'name: "{name}"\n{kind}\n')
    for f in files:
        (d / f).write_text("x")


def test_good_repository_has_no_problems(tmp_path):
    make(tmp_path, "pets", 'platform: "tensorrt_plan"',
         ["model.plan"])
    make(tmp_path, "pre", 'backend: "python"', ["model.py"])
    make(tmp_path, "pipe",
         'platform: "ensemble"\n  model_name: "pets"', [])
    assert check_repository(tmp_path) == []


def test_missing_model_file_is_reported(tmp_path):
    make(tmp_path, "pets", 'platform: "tensorrt_plan"', [])
    assert any("model.plan" in p
               for p in check_repository(tmp_path))


def test_unknown_ensemble_step_is_reported(tmp_path):
    make(tmp_path, "pipe",
         'platform: "ensemble"\n  model_name: "ghost"', [])
    assert any("ghost" in p for p in check_repository(tmp_path))


def test_missing_config_is_reported(tmp_path):
    (tmp_path / "pets" / "1").mkdir(parents=True)
    assert any("config.pbtxt" in p
               for p in check_repository(tmp_path))


def test_render_contains_requested_settings():
    s = BatchSettings(preferred=(8, 32), max_delay_us=2000,
                      instances=2, max_queue_size=64,
                      timeout_us=50000)
    text = render_pets_config("pets", "tensorrt_plan", s, 37)
    for want in ('platform: "tensorrt_plan"',
                 "preferred_batch_size: [ 8, 32 ]",
                 "max_queue_delay_microseconds: 2000",
                 "count: 2", "max_queue_size: 64",
                 "dims: [ 37 ]", "model_warmup"):
        assert want in text


def test_dynamic_batching_can_be_disabled():
    text = render_pets_config(
        "pets", "tensorrt_plan",
        BatchSettings(dynamic=False), 37)
    assert "dynamic_batching" not in text


def test_preferred_batch_above_maximum_is_rejected():
    with pytest.raises(ValueError):
        render_pets_config(
            "p", "tensorrt_plan",
            BatchSettings(max_batch=16, preferred=(32,)), 37)
