#!/usr/bin/env bash
# scripts/doctor.sh
set -u
PASS=0
FAIL=0

check() {
  local name="$1"
  shift
  if "$@" >/dev/null 2>&1; then
    printf '[ OK ] %s\n' "$name"
    PASS=$((PASS + 1))
  else
    printf '[FAIL] %s\n' "$name"
    FAIL=$((FAIL + 1))
  fi
}

cmake_ok() {
  local v
  v=$(cmake --version | head -n1 | awk '{print $3}')
  [ "$(printf '%s\n3.24\n' "$v" | sort -V | head -n1)" \
    = "3.24" ]
}

check "NVIDIA driver (nvidia-smi)" nvidia-smi
check "CUDA compiler (nvcc)" nvcc --version
check "C++ compiler (g++)" g++ --version
check "CMake >= 3.24" cmake_ok
check "Python 3" python3 --version
check "PyTorch sees the GPU" python3 -c \
  "import torch; assert torch.cuda.is_available()"

if command -v docker >/dev/null 2>&1; then
  check "Docker can use the GPU" \
    docker run --rm --gpus all \
    nvidia/cuda:12.8.1-base-ubuntu24.04 nvidia-smi
else
  echo "[SKIP] Docker not installed"
fi

printf '\n%d passed, %d failed\n' "$PASS" "$FAIL"
[ "$FAIL" -eq 0 ]
