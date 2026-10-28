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
| `requirements/*.txt` | Direct dependencies, compiled into `*.txt` lock files |

## Setup

Create the environment and install the dependencies.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements/base.txt
pip install -r requirements/dev.txt
pip install -e .
```

Install PyTorch from `requirements/torch.txt`, which names the CUDA build explicitly. Keep TensorFlow in a separate virtual environment, because the two frameworks pin different CUDA libraries.

```bash
pip install -r requirements/torch.txt
python3 -m venv .venv-tf
.venv-tf/bin/pip install "tensorflow[and-cuda]"
```

Once everything works, record the exact versions so the environment can be rebuilt later.

```bash
pip freeze > requirements/lock-YYYY-MM.txt
```