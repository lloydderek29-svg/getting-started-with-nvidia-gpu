# Chapter 2: Production-Ready CUDA and Python Environments

This directory verifies that the driver, toolkit, and frameworks work together, and it holds the checks that make an environment reproducible. The container and automation files it relies on live at the repository root.

## What Is Here

| Path | Purpose |
|---|---|
| `src/check_runtime.cu` | Launches a real kernel and verifies the result |
| `python/verify_torch.py` | Checks PyTorch build, architectures, and a matrix multiply |
| `python/verify_tf.py` | Checks TensorFlow GPU visibility |
| `tests/test_validate.py` | Tests the CUDA version parser |

Related files elsewhere in the repository:

| Path | Purpose |
|---|---|
| `scripts/doctor.sh` | Shell checks for driver, compiler, CMake, Python, and Docker |
| `common/python/gpubook/validate.py` | Python validator comparing versions across layers |
| `common/python/gpubook/env.py` | Environment snapshot for saving beside results |
| `docker/Dockerfile` and `docker/docker-compose.yml` | Development container |
| `requirements/*.in` | Direct dependencies, compiled into `*.txt` lock files |

## Setup

Create the environment and lock the dependencies.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install pip-tools
pip-compile requirements/base.in -o requirements/base.txt
pip-compile requirements/dev.in -o requirements/dev.txt
pip-sync requirements/base.txt requirements/dev.txt
pip install -e .
```

Install PyTorch from `requirements/torch.txt`, which names the CUDA build explicitly. Keep TensorFlow in a separate virtual environment, because the two frameworks pin different CUDA libraries.

```bash
pip install -r requirements/torch.txt
python3 -m venv .venv-tf
.venv-tf/bin/pip install "tensorflow[and-cuda]"
```

## Build and Run

```bash
cmake --preset release
cmake --build --preset release -j
./build/release/bin/check_runtime
```

```text
Driver supports : 12.8
Runtime         : 12.8
Kernel launch and copy back: OK
```

Run the framework checks.

```bash
python chapters/ch02_environments/python/verify_torch.py
.venv-tf/bin/python chapters/ch02_environments/python/verify_tf.py
```

The PyTorch check prints the CUDA version the wheel was built for, the architecture list, and a measured matrix multiply rate. If your card's compute capability is missing from the architecture list, PyTorch fails with a "no kernel image" error, so install a newer build.

## One Command for Everything

```bash
make doctor
```

This runs `scripts/doctor.sh` and `python -m gpubook.validate`. Each check prints OK or FAIL, and the script exits nonzero if anything fails. Run it first whenever something behaves strangely, and save its output when asking for help.

## Container

```bash
docker compose -f docker/docker-compose.yml build
docker compose -f docker/docker-compose.yml run --rm dev bash
```

The container prints the visible GPUs and the `nvcc` version at startup. Inside it, the repository is mounted at `/workspace`. If no GPU appears, see the troubleshooting table.

## Save an Environment Snapshot

```bash
python -c "from gpubook.env import save_snapshot; save_snapshot('results/env.json')"
```

Store this file next to any benchmark result.

## Tests

```bash
pytest chapters/ch02_environments -v
ctest --preset release
```

The CTest run executes `check_runtime`, which doubles as an installation check.

## Troubleshooting

| Message | Fix |
|---|---|
| Driver version is insufficient for runtime | Upgrade the host driver |
| no kernel image is available | Rebuild for your architecture or use a matching wheel |
| libcudart.so.12 not found | Add the toolkit `lib64` directory to `LD_LIBRARY_PATH` |
| Driver/library version mismatch | Reboot after a driver upgrade |
| could not select device driver | Run `nvidia-ctk runtime configure`, then restart Docker |
| Python reports no GPU but `nvidia-smi` works | Check that `CUDA_VISIBLE_DEVICES` is not set empty |